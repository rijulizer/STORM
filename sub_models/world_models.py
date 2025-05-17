import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal

# from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast

from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import (
    get_subsequent_mask_with_batch_length,
    get_subsequent_mask,
)
from sub_models.transformer_model import (
    StochasticTransformerKVCache,
    MLATransformerKVCache,
    TEMTransformerKVCache,
)
from sub_models.constants import DEVICE, DTYPE_16

# import sub_models.agents as agents
from sub_models.director_agents import DirectorAgent


class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()
        # Create a sequence of convolutional layers
        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        feature_width = 64 // 2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))

        # layers
        while True:
            # Keep adding layers until we reach the final feature width
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        # Store the final number of channels
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(-1, *x.shape[2:])  # [B, L, C, H, W] -> [B*L, C, H, W]
        x = self.backbone(x)
        # [B*L, C', H', W'] -> [B, L, C'*H'* W'];     H' = W' = 4
        x = x.reshape(batch_size, -1, x.shape[1] * x.shape[2] * x.shape[3])
        return x


class DecoderBN(nn.Module):
    def __init__(
        self,
        stoch_dim,
        last_channels,
        original_in_channels,
        stem_channels,
        final_feature_width,
    ) -> None:
        super().__init__()

        backbone = []
        # Create a sequence of linear layer followed by de-convolutional layers
        # stem
        backbone.append(
            nn.Linear(
                stoch_dim,
                last_channels * final_feature_width * final_feature_width,
                bias=False,
            )
        )
        backbone.append(
            Rearrange(
                "B L (C H W) -> (B L) C H W", C=last_channels, H=final_feature_width
            )
        )
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        # [B*L, C, H, W] -> [B, L, C, H, W]
        obs_hat = obs_hat.reshape(batch_size, -1, *obs_hat.shape[1:])
        return obs_hat


class DistHead(nn.Module):
    """
    DistHead is a collection of different heads (mappings) of different latent spaces.
    Like: Encoder to Posterior, Transformer to Prior, etc.
    1. post_head: mapping from the encoder output to posterior distribution
    2. prior_head: mapping from transformer hidden states to prior distribution
    """

    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim * stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim * stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = (
            mixing_ratio * torch.ones_like(probs) / self.stoch_dim
            + (1 - mixing_ratio) * probs
        )
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)  # [B, L, C'*H'* W'] -> [B, L, K*K] ; K = stoch_dim
        # Reshape: [B, L, K*K] -> [B, L, K, K]
        logits = logits.reshape(logits.shape[0], logits.shape[1], self.stoch_dim, -1)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        # Reshape: [B, L, K*K] -> [B, L, K, K]
        logits = logits.reshape(logits.shape[0], logits.shape[1], self.stoch_dim, -1)
        logits = self.unimix(logits)
        return logits


class RewardDecoder(nn.Module):
    """
    Decode the reward from the transformer latent embeddings.
    Final mapping: transformer_hidden_dim -> num_classes
    """

    def __init__(self, num_classes, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    """
    Decode the termination from the transformer latent embeddings.
    Final mapping: transformer_hidden_dim -> 1
    """

    def __init__(self, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs) ** 2
        # [B, L, C, H, W] -> [B, L]
        loss = loss.sum(dim=(2, 3, 4))  # Sum over C, H, W dimensions
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        # [B, L, D] -> [B, L]
        kl_div = kl_div.sum(dim=2)
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div) * self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(
        self,
        in_channels,
        action_dim,
        transformer_max_length,
        transformer_hidden_dim,
        transformer_num_layers,
        transformer_num_heads,
    ):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.final_feature_width = 4
        self.stoch_dim = 32
        self.stoch_flattened_dim = self.stoch_dim * self.stoch_dim
        self.use_amp = False  # FIXME: True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1

        # Instantiate the different modules
        self.encoder = EncoderBN(
            in_channels=in_channels,
            stem_channels=32,
            final_feature_width=self.final_feature_width,
        )
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1,
        )
        self.dist_head = DistHead(
            image_feat_dim=self.encoder.last_channels
            * self.final_feature_width
            * self.final_feature_width,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=self.stoch_dim,
        )
        self.image_decoder = DecoderBN(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=in_channels,
            stem_channels=32,
            final_feature_width=self.final_feature_width,
        )
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            # embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim,
        )
        self.termination_decoder = TerminationDecoder(
            # embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim,
        )

        # All types of loss functions
        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(
            num_classes=255, lower_bound=-20, upper_bound=20
        )
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        # Enable scaler based on device type
        self.scaler = (
            torch.amp.GradScaler(enabled=self.use_amp)
            if DEVICE.type == "cuda"
            else None
        )

    def encode_obs(self, obs):
        """
        Encode the observation using the encoder and sample from posterior.
        """

        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(
                post_logits, sample_mode="random_sample"
            )
            # Reshape: [B, L, K, K] -> [B, L, K*K]; K = stoch_dim
            flattened_sample = sample.reshape(sample.shape[0], sample.shape[1], -1)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            prior_logits = self.dist_head.forward_prior(last_dist_feat)
            prior_sample = self.stright_throught_gradient(
                prior_logits, sample_mode="random_sample"
            )
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, last_flattened_sample, action, log_video=True):
        """
        A single step of world model prediction in the imagination phase.
        For the WM, only sample and action are required to predict the next token.
        """
        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):
            # posterior_logits sample + action -> trasnformer hidden states
            trans_feat = self.storm_transformer.forward_with_kv_cache(
                last_flattened_sample, action
            )
            # trasnformer hidden states -> prior_logits
            prior_logits = self.dist_head.forward_prior(trans_feat)

            # decoding: prior_logits -> prior_sample -> prior_flattened_sample
            prior_sample = self.stright_throught_gradient(
                prior_logits, sample_mode="random_sample"
            )
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            # decode reward and termination
            reward_hat = self.reward_decoder(trans_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(trans_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, trans_feat

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        # Reshape: [B, L, K, K] -> [B, L, K*K]; K = stoch_dim
        return sample.reshape(sample.shape[0], sample.shape[1], -1)

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        """
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        """
        # if (
        #     self.imagine_batch_size != imagine_batch_size
        #     or self.imagine_batch_length != imagine_batch_length
        # ): # probably not needed
        print(
            f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}"
        )
        self.B = imagine_batch_size
        self.L = imagine_batch_length

        # define the size of the buffers
        sample_size = (self.B, self.L + 1, self.stoch_flattened_dim)
        hidden_size = (self.B, self.L + 1, self.transformer_hidden_dim)
        scalar_size = (self.B, self.L)
        goal_size = (self.B, self.L, self.stoch_flattened_dim)
        skill_size = (self.B, self.L, 8, 8)  # FIXME: makeit a variable

        # Initiate buffers with zeros
        self.sample_buffer = torch.zeros(sample_size, dtype=dtype, device=DEVICE)
        self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device=DEVICE)
        self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device=DEVICE)
        self.reward_hat_buffer = torch.zeros(
            scalar_size, dtype=dtype, device=DEVICE.type
        )
        self.termination_hat_buffer = torch.zeros(
            scalar_size, dtype=dtype, device=DEVICE.type
        )
        self.goal_buffer = torch.zeros(goal_size, dtype=dtype, device=DEVICE.type)
        self.skill_buffer = torch.zeros(skill_size, dtype=dtype, device=DEVICE.type)

    def imagine_data(
        self,
        agent: DirectorAgent,
        buffer_sample,
        imagine_batch_size,
        imagine_batch_length,
        log_video,
        logger=None,
    ):
        B = imagine_batch_size
        L = imagine_batch_length
        imagine_rollout = {}

        self.init_imagine_buffer(B, L, dtype=self.tensor_dtype)
        obs_hat_list = []

        self.storm_transformer.reset_kv_cache_list(B, dtype=self.tensor_dtype)
        # context
        context_latent = self.encode_obs(buffer_sample["obs"])
        # initiate the buffer with the first sample
        for i in range(buffer_sample["obs"].shape[1]):
            (
                last_obs_hat,
                last_reward_hat,
                last_termination_hat,
                last_flattened_sample,
                last_dist_feat,
            ) = self.predict_next(
                context_latent[:, i : i + 1],
                buffer_sample["action"][:, i : i + 1],
                log_video=log_video,
            )
        self.sample_buffer[:, 0:1] = last_flattened_sample
        self.hidden_buffer[:, 0:1] = last_dist_feat
        if agent.__class__.__name__ == "DirectorAgent":
            # Agent needs to refresh carry in the loop L
            # Goal and skill from a differenet batch wont make sense
            agent.initiate_carry()
        # Imagine, incrementaly get the next tokens
        for i in range(L):  # len (imagination)/ context_length 16
            latent = torch.cat(
                [
                    self.sample_buffer[:, i : i + 1],
                    self.hidden_buffer[:, i : i + 1],
                ],
                dim=-1,  # [B, 1, Z+Z]
            )
            # # load goal from the r
            # exist_goal = buffer_sample["goal"][:, i : i + 1]
            # exist_skill = buffer_sample["skill"][:, i : i + 1]

            action = agent.sample(latent)  # , exist_goal, exist_skill)
            # Add action, goal skill to the buffer
            self.action_buffer[:, i : i + 1] = action
            if agent.__class__.__name__ == "DirectorAgent":
                self.goal_buffer[:, i : i + 1] = agent.carry["goal"]
                self.skill_buffer[:, i : i + 1] = agent.carry["skill"]
            else:
                self.goal_buffer = None
                self.skill_buffer = None

            (
                last_obs_hat,
                last_reward_hat,
                last_termination_hat,
                last_flattened_sample,  # S: prior_flattened_sample, Sample
                last_dist_feat,  # H: transformer hidden states
            ) = self.predict_next(
                self.sample_buffer[:, i : i + 1],
                self.action_buffer[:, i : i + 1],
                log_video=log_video,
            )
            # store variables in the next index
            self.sample_buffer[:, i + 1 : i + 2] = last_flattened_sample
            self.hidden_buffer[:, i + 1 : i + 2] = last_dist_feat
            self.reward_hat_buffer[:, i : i + 1] = last_reward_hat
            self.termination_hat_buffer[:, i : i + 1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[:: B // 16])  # uniform sample vec_env

        if log_video:
            logger.log(
                "Imagine/predict_video",
                torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1)
                .cpu()
                .float()
                .detach()
                .numpy(),
            )
        # ensure the last token is removed to make the length of the buffer same
        # [B, L+1, C] -> [B, L, C]
        self.sample_buffer = self.sample_buffer[:, 0:L]  # remove the last token
        self.hidden_buffer = self.hidden_buffer[:, 0:L]
        # return imagine_rollout
        imagine_rollout = {
            "sample": self.sample_buffer,
            "hidden": self.hidden_buffer,
            "action": self.action_buffer,
            "reward": self.reward_hat_buffer,
            "termination": self.termination_hat_buffer,
            "goal": self.goal_buffer,
            "skill": self.skill_buffer,
        }
        return imagine_rollout

    def update(self, obs, action, reward, termination, logger=None):
        """
        A Single training step of the world model using the observation, action, reward, and termination.
        """
        metrics = {}
        self.train()
        L = obs.shape[1]  # B, L
        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):
            # Encoding and sampling from posterior
            # [B, L, C, H, W] -> [B, L, K*K];   K = stoch_dim
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(
                post_logits, sample_mode="random_sample"
            )
            flattened_sample = self.flatten_sample(sample)

            # Decoding image from the sample
            # [B, L, K*K] -> [B, L, C, H, W]
            obs_hat = self.image_decoder(flattened_sample)

            # Transformer
            temporal_mask = get_subsequent_mask_with_batch_length(
                L, device=flattened_sample.contiguous().device
            )
            trans_hidden = self.storm_transformer(
                flattened_sample, action, temporal_mask
            )
            # Prior logits
            prior_logits = self.dist_head.forward_prior(trans_hidden)
            # Decoding reward and termination with trans_hidden
            reward_hat = self.reward_decoder(trans_hidden)
            termination_hat = self.termination_decoder(trans_hidden)

            # Env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(
                termination_hat, termination
            )
            # Dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(
                post_logits[:, 1:].detach(), prior_logits[:, :-1]
            )
            representation_loss, representation_real_kl_div = (
                self.categorical_kl_div_loss(
                    post_logits[:, 1:], prior_logits[:, :-1].detach()
                )
            )
            total_loss = (
                reconstruction_loss
                + reward_loss
                + termination_loss
                + 0.5 * dynamics_loss
                + 0.1 * representation_loss
            )

        # Gradient descent
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)  # for clip grad
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item())
            logger.log("WorldModel/reward_loss", reward_loss.item())
            logger.log("WorldModel/termination_loss", termination_loss.item())
            logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
            logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
            logger.log("WorldModel/representation_loss", representation_loss.item())
            logger.log(
                "WorldModel/representation_real_kl_div",
                representation_real_kl_div.item(),
            )
            logger.log("WorldModel/total_loss", total_loss.item())
        metrics = {
            "WM/reconstruction_loss": reconstruction_loss.item(),
            "WM/reward_loss": reward_loss.item(),
            "WM/termination_loss": termination_loss.item(),
            "WM/dynamics_loss": dynamics_loss.item(),
            "WM/dynamics_real_kl_div": dynamics_real_kl_div.item(),
            "WM/representation_loss": representation_loss.item(),
            "WM/representation_real_kl_div": representation_real_kl_div.item(),
            "WM/total_loss": total_loss.item(),
        }
        return metrics
