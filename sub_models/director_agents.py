from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchrl.data import AdaptiveKLController

from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.torch_utils import MSEDist, OneHotDist
from utils import EMAScalar

from sub_models.constants import DEVICE, DTYPE_16


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage * len(flat_x))
    if DEVICE.type == "mps":
        # MPS does not support kthvalue, use sorting instead
        sorted_x, _ = torch.sort(flat_x)
        per = sorted_x[kth]
    else:
        per = torch.kthvalue(flat_x, kth + 1).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device=device)
    gamma_return = torch.zeros(
        (batch_size, batch_length + 1), dtype=dtype, device=DEVICE
    )
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = (
            rewards[:, t]
            + gamma * inv_termination[:, t] * (1 - lam) * values[:, t]
            + gamma * inv_termination[:, t] * lam * gamma_return[:, t + 1]
        )
    return gamma_return[:, :-1]


class BaseAgent(nn.Module):
    def __init__(
        self,
        critics: list[dict],
        input_dim: int,
        action_dim: int | tuple[int, int],
        actor_dist: str,
    ) -> None:
        """
        Args:
            crtitcs: A list of dict containing different critic info.
                    Ex: {"critic": "name", "scale": 0.5, "reward": reward key}
            input_dim: The input dimension of the actor model. This class expects action_dim to be int.
                So if actual input_dim is a tuple, then it should be flattened to int.
            action_dim: The output dimension of the actor model. This class expects action_dim to be int.
            actor_dist: The distribution type of the actor model. like OneHotDist, MSEDist, etc.
        """
        super().__init__()
        self.critics = critics
        self.input_dim = input_dim
        self.action_dim = action_dim
        if isinstance(action_dim, tuple):
            self.action_dim_flatten = action_dim[0] * action_dim[1]
        else:
            self.action_dim_flatten = action_dim
        self.actor_dist = actor_dist

        self.critic_op_dim = 255  # TODO: Check this # 255 in STORM
        self.hidden_dim = 512  # config
        self.num_layers = 4  # config
        self.gamma = 0.985
        self.clip_value = 100.0  # Gradient clipping value
        self.discount = 0.99  # config
        self.lambd = 0.95  # config
        self.entropy_coef = 0.95  # config TODO: Check this
        self.use_amp = True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32
        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        # Sequential actor model to map from feat_dim to action_dim
        actor_model = [
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        ]
        for i in range(self.num_layers - 2):
            actor_model.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                ]
            )
        self.actor = nn.Sequential(
            *actor_model, nn.Linear(self.hidden_dim, self.action_dim_flatten)
        )

        # Sequential critic model to map from feat_dim to 255 dim
        critic_model_backbone = [
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        ]
        for i in range(self.num_layers - 2):
            critic_model_backbone.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                ]
            )

        # Create three models with shared backbone but different additional layers
        for idx in range(len(self.critics)):
            critic_head = nn.Sequential(
                *critic_model_backbone,
                nn.Linear(self.hidden_dim, self.critic_op_dim, bias=True),
                nn.LayerNorm(self.critic_op_dim),
                nn.ReLU(),
            )
            # Make a copy of critic for slow critic
            self.critics[idx]["model"] = critic_head
            self.critics[idx]["slow_model"] = deepcopy(critic_head)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        # Enable scaler based on DEVICE type
        self.scaler = (
            torch.cuda.amp.GradScaler(enabled=self.use_amp)
            if DEVICE.type == "cuda"
            else None
        )

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        """
        Update slow critic models parameters with decay.
        """
        for critic in self.critics:
            slow_model = critic["slow_model"]
            model = critic["model"]
            for slow_param, param in zip(slow_model.parameters(), model.parameters()):
                slow_param.data.copy_(
                    slow_param.data * decay + param.data * (1 - decay)
                )

    def policy(self, x):
        """
        Returns the action distribution based on the actor model.
        Expects L=1
        """
        logits = self.actor(x)
        B, L = logits.shape[0], logits.shape[1]
        if isinstance(self.action_dim, tuple):
            # e.g. self.action_dim = (K, K) # manager
            logits = logits.reshape(B, L, *self.action_dim)
        else:
            # self.action_dim is an int # worker
            logits = logits.reshape(B, L, self.action_dim)
        if self.actor_dist == "OneHotDist":
            dist = OneHotDist(logits=logits)
        elif self.actor_dist == "MSEDist":
            dist = MSEDist(logits=logits, dims=1)
        elif self.actor_dist == "Categorical":
            dist = torch.distributions.Categorical(logits=logits)
        else:
            raise ValueError(f"Unknown actor distribution: {self.actor_dist}")
        return dist

    def get_critic_value(self, critic_model, x):
        """
        Use the critic model to get the value of the state.
        """
        value = critic_model(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def get_slow_value(self, slwo_critic_model, x):
        """
        Use the slow critic model to get the slow-value of the state.
        """
        value = slwo_critic_model(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def update(self, traj, logger=None):
        """
        Update policy and value models using imagine rollout.
        Args:
            traj: A dict that holds the state representation from WM
                and other important entities.
        """
        metrics = {}
        # All have the shape [B, L, *]
        hidden = traj["hidden"]  # The hidden state from WM
        sample = traj["sample"]  # The sample from WM
        # reward = imagine_rollout["reward"]
        action = traj["action"]  # [B, L]
        # cont = imagine_rollout["cont"]
        termination = traj["termination"]
        goal = traj.get("goal", None)
        if goal is not None:
            # for the case of worker the goal is also part of latent
            latent = torch.cat((hidden, sample, goal), dim=-1)  # [B, L, 3*]
        else:
            latent = torch.cat((hidden, sample), dim=-1)  # [B, L, 2*]
        self.train()
        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):
            # Get action logits using actor model
            # action_logits = self.actor(latent)
            # # [B, L, action_dim]
            # action_dist = distributions.Categorical(logits=action_logits)
            action_dist = self.policy(latent)  # [B, L, action_dim]
            # get the log prob of the actual action
            # Expects action to have values between 0 and action_dim-1
            log_prob = action_dist.log_prob(action)  # [B, L]

            total_critic_loss = 0.0
            total_value_loss = 0.0
            total_slow_value_loss = 0.0
            norm_aqdvantages = []  # TODO: check this logic

            # Iterate over all critics and calculate values
            for critic in self.critics:
                # get value for each critic model
                raw_value = critic["model"](latent)
                value = self.symlog_twohot_loss.decode(raw_value)

                # Generate critic reward function specific reward
                # reward functions operate on Deter in Director ~ Sample in STORM
                reward = traj[critic["reward"]]  # TODO: Check input sample
                lambda_return = calc_lambda_return(
                    reward, value, termination, self.gamma, self.lambd
                )
                # get slow-value for each slow-critic-model
                slow_value = self.get_slow_value(critic["slow_model"], latent)
                slow_lambda_return = calc_lambda_return(
                    reward, slow_value, termination, self.gamma, self.lambd
                )

                # update value function with slow critic regularization
                value_loss = self.symlog_twohot_loss(raw_value, lambda_return.detach())
                slow_value_regularization_loss = self.symlog_twohot_loss(
                    raw_value, slow_lambda_return.detach()
                )  # [:, :-1]
                # Apply the critic scale as a multiplicative factor
                # #TODO: for now the scales are used to scale lossess
                scaled_value_loss = critic["scale"] * value_loss
                scaled_slow_value_regularization_loss = (
                    critic["scale"] * slow_value_regularization_loss
                )
                # update the critic losses
                total_value_loss += scaled_value_loss
                total_slow_value_loss += scaled_slow_value_regularization_loss
                total_critic_loss += (
                    scaled_value_loss + scaled_slow_value_regularization_loss
                )

                lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
                upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
                S = upper_bound - lower_bound
                norm_ratio = torch.max(
                    torch.ones(1).to(DEVICE), S
                )  # max(1, S) in the paper
                norm_aqdvantages.append(
                    (lambda_return - value) / norm_ratio
                )  # [:, :-1]

            # Calcuate the average normed advantage
            avg_norm_advantage = torch.mean(
                torch.stack(norm_aqdvantages), dim=0
            )  # TODO: Check this logic #Dennis
            # Calculate Actor related losses
            if len(log_prob.shape) == 3:
                # for manager the log_prob is [B, L, K]
                avg_norm_advantage = avg_norm_advantage.unsqueeze(-1)  # [B,L,1]
            policy_loss = -(log_prob * avg_norm_advantage.detach()).mean()
            entropy_loss = action_dist.entropy().mean()

            # Calculate total loss
            loss = policy_loss + total_critic_loss - self.entropy_coef * entropy_loss

        # gradient descent
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)  # for clip grad
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()
        # Update metrics
        metrics["AC/policy_loss"] = policy_loss.item()
        metrics["AC/critic_loss"] = total_critic_loss.item()
        metrics["AC/entropy_loss"] = entropy_loss.item()
        metrics["AC/S"] = S.item()
        metrics["AC/norm_ratio"] = norm_ratio.item()
        metrics["AC/total_loss"] = loss.item()
        # Log metrics
        if logger is not None:
            logger.log("AC/policy_loss", policy_loss.item())
            logger.log("AC/critic_loss", total_critic_loss.item())
            logger.log("AC/entropy_loss", entropy_loss.item())
            logger.log("AC/S", S.item())
            logger.log("AC/norm_ratio", norm_ratio.item())
            logger.log("AC/total_loss", loss.item())

        return metrics


class GoalEncoder(nn.Module):
    def __init__(self, input_dim, op_dim):

        super().__init__()
        self._op_dim = op_dim
        if isinstance(op_dim, tuple):
            self._op_dim_flatten = op_dim[0] * op_dim[1]
        else:
            self._op_dim_flatten = op_dim
        self._layers = 4  # config
        self._units = 512  # config
        # self._inputs = inputs
        # self._dims = dims
        self._unimix = 0.0  # config
        self._outscale = 0.1  # config

        # Build dense layers
        self.dense_layers = nn.ModuleList()
        # add the first layer with input_dim
        self.dense_layers.append(
            nn.Sequential(
                nn.Linear(input_dim, self._units, bias=True),
                nn.LayerNorm(self._units),
                nn.ELU(),
            )
        )
        for _ in range(self._layers - 2):
            self.dense_layers.append(
                nn.Sequential(
                    nn.Linear(self._units, self._units, bias=True),
                    nn.LayerNorm(self._units),
                    nn.ELU(),
                )
            )
        self.dense_layers.append(
            nn.Sequential(
                nn.Linear(self._units, self._op_dim_flatten, bias=True),
                nn.LayerNorm(self._op_dim_flatten),
                nn.ELU(),
            )
        )

    def forward(self, x):
        B, L, Z = x.shape[0], x.shape[1], x.shape[2]
        # # Flatten the input for dense layers: [B, L, Z] -> [B*L, Z]
        x = x.reshape(-1, Z)
        # Pass through dense layers
        for layer in self.dense_layers:
            x = layer(x)
        # Reshape back to match input batch dimensions
        x = x.reshape(B, L, *self._op_dim)  # [B, L, K, K]
        # Apply the distribution layer
        probs = F.softmax(x, dim=-1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        dist = torch.distributions.OneHotCategorical(probs=probs)
        return dist


class GoalDecoder(nn.Module):
    def __init__(self, input_dim, op_dim):

        super().__init__()
        if isinstance(input_dim, tuple):
            self._input_dim = input_dim[0] * input_dim[1]
        else:
            self._input_dim = input_dim
        self._op_dim = op_dim
        self._layers = 4  # config
        self._units = 512  # config
        # self._inputs = inputs
        # self._dims = dims
        self._unimix = 0.0  # config
        self._outscale = 0.1  # config

        # Build dense layers
        self.dense_layers = nn.ModuleList()
        # add the first layer with input_dim
        self.dense_layers.append(
            nn.Sequential(
                nn.Linear(self._input_dim, self._units, bias=True),
                nn.LayerNorm(self._units),
                nn.ELU(),
            )
        )
        for _ in range(self._layers - 2):
            self.dense_layers.append(
                nn.Sequential(
                    nn.Linear(self._units, self._units, bias=True),
                    nn.LayerNorm(self._units),
                    nn.ELU(),
                )
            )
        self.dense_layers.append(
            nn.Sequential(
                nn.Linear(self._units, self._op_dim, bias=True),
                nn.LayerNorm(self._op_dim),
                nn.ELU(),
            )
        )

    def forward(self, x):
        B, L, Z = x.shape[0], x.shape[1], x.shape[2]
        # # Flatten the input for dense layers: [B, L, Z, Z] -> [B*L, Z*Z]
        x = x.reshape(B * L, -1)
        # Pass through dense layers
        for layer in self.dense_layers:
            x = layer(x)
        x = x.reshape(B, L, -1)  # Reshape back to match input batch dimensions
        # Apply the distribution layer
        dist = MSEDist(x, dims=1)
        return dist


class DirectorAgent(nn.Module):
    def __init__(self, wm_hidden_dim: int, wm_sample_dim: int, wm_action_dim: int):
        super().__init__()

        self.wm_sample_dim = wm_sample_dim
        self.wm_feat_dim = wm_sample_dim + wm_hidden_dim  # WM latent dim
        self.skill_duration = 8  # config
        self.skill_shape = (8, 8)  # config
        # self.skill_shape_flatten = self.skill_shape[0] * self.skill_shape[1]
        self.goal_encoder = GoalEncoder(self.wm_sample_dim, self.skill_shape)
        self.goal_decoder = GoalDecoder(self.skill_shape, self.wm_sample_dim)
        self.manager = BaseAgent(
            [  # Manager gets only external WM reward and exploration reward
                {"critic": "extr", "scale": 1.0, "reward": "reward_extr"},
                {"critic": "expl", "scale": 0.1, "reward": "reward_expl"},
            ],
            input_dim=self.wm_feat_dim,
            action_dim=self.skill_shape,
            actor_dist="OneHotDist",
        )
        self.worker = BaseAgent(
            critics=[
                {"critic": "goal", "scale": 1.0, "reward": "reward_goal"},
            ],
            input_dim=self.wm_feat_dim + self.wm_sample_dim,  # goal_dim = wm_sample_dim
            action_dim=wm_action_dim,
            actor_dist="Categorical",  # TODO: check teh distribution worker
        )
        self.skill_prior = self.get_skill_prior()
        self.discount = 0.99  # config
        self.kl_controller = AdaptiveKLController(
            init_kl_coef=0.0, target=10.0, horizon=100
        )  # config
        self.initiate_carry()
        # director module only trains the goal encoder and decoder
        # manager and worker are trained using the base agent update function
        vae_params = list(self.goal_encoder.parameters()) + list(
            self.goal_decoder.parameters()
        )
        self.optimizer = torch.optim.Adam(
            vae_params, lr=3e-5, eps=1e-5
        )  # FIXME: check which params are optimized!
        # Enable scaler based on DEVICE type
        self.use_amp = True
        self.scaler = (
            torch.cuda.amp.GradScaler(enabled=self.use_amp)
            if DEVICE.type == "cuda"
            else None
        )

    def get_skill_prior(self):
        """
        Returns a prior distribution over the skill space.
        """

        # Create logits = 0 → uniform categorical
        logits = torch.zeros(self.skill_shape)
        dist = torch.distributions.OneHotCategorical(logits=logits)
        # Wrap in Independent if shape > 1
        # TODO: FInd alternative for Independent in torch distributions
        # as this dirstribution is not supported for KL divergence
        # if len(self.skill_shape) > 1:
        #     # this will be the case if the shape is like [8, 8]
        #     dist = torch.distributions.Independent(
        #         dist, reinterpreted_batch_ndims=len(self.skill_shape) - 1
        #     )
        return dist

    def initiate_carry(
        self,
    ) -> dict:
        """
        Initialize the internal carry states of director agent.
        Holds entities like- goal, skill, step etc.
        """
        self.carry = defaultdict(str)
        self.carry["step"] = 0
        self.carry["goal"] = torch.zeros(1, 1, self.wm_sample_dim)  # [1, 1, Z] B=1, L=1
        self.carry["skill"] = torch.zeros(1, 1, *self.skill_shape)
        # carry["action"] = torch.zeros(self.wm_action_dim)

    @torch.no_grad()
    def extr_reward(self, imagine_rollout):
        """
        Returns the actual reward based on the imagined rollout.
        """
        return imagine_rollout["reward"]

    @torch.no_grad()
    def explr_reward(self, imagine_rollout):
        """
        Computes the ELBO reward based on the imagined rollout.
        """
        wm_sample = imagine_rollout["sample"]  # [B, L, Z]
        # Get encoded distribution
        encoded_dist = self.goal_encoder(wm_sample)
        # Get decoded distribution
        decoded_dist = self.goal_decoder(encoded_dist.sample())
        # Compute ELBO reward: MSE bteween decoded and actual
        # the OP shape: [B, L]
        reward = ((decoded_dist.mode() - wm_sample) ** 2).mean(-1)
        # return second element onwards [B, L]
        return reward  # [:, 1:][B, L-1]

    @torch.no_grad()
    def goal_reward(self, imagine_rollout):
        """
        Cosine Max similarity
        Calculate reward based on the goal and the transition state.
        """
        wm_sample = imagine_rollout["sample"]
        goal = self.carry["goal"]
        # calculate normilization factor
        norm = torch.maximum(
            goal.norm(dim=-1, keepdim=True), wm_sample.norm(dim=-1, keepdim=True)
        ).clamp_min(1e-12)
        # [B, L, Z] -> [B, L]
        reward = (goal / norm * wm_sample / norm).sum(dim=-1)
        # return the second element onward [B, L]
        return reward  # [:, 1:]

    @torch.no_grad()
    def policy_step(self, latent):  # , goal, skill):
        """
        Hierarchical policy step function. First decides whether to update the goal from the manager.
        Then based on the goal, get the workers action logits. Policy step expects one slice of time dim L=1.
        Args: Latent: The latent state from the world model. cat([sample, hidden]) [B, 1, 2Z]
        Returns:
            action_dist: A torch distribution object for actions.
            goal: Existing goal.
            skill: Existing skill.
        """
        # if goal is None:  # FIXME: Think in this logic, what happens first few rounds
        #     goal = torch.zeros(latent.shape[0], latent.shape[1], self.wm_sample_dim)
        # if skill is None:
        #     skill = torch.zeros(
        #         latent.shape[0],
        #         latent.shape[1],
        #         self.skill_shape[0],
        #         self.skill_shape[1],
        #     )
        goal = self.carry["goal"]  # [B, 1, Z]
        skill = self.carry["skill"]  # [B, 1, K, K]
        step = self.carry["step"]
        if step % self.skill_duration == 0:
            # Get new skill and goal from the manager
            # Get skill: manager actor logits from latent
            skill = self.manager.policy(latent).sample()
            # Decode new goal from skill #TODO: Director uses latent as a context
            goal = self.goal_decoder(skill).mode()  # shape: [B, 1, goal_dim]
            self.carry["goal"] = goal
            self.carry["skill"] = skill
        else:
            # Ensure goal's batch size matches latent's batch size
            # can heppens when fist time latent has batch but if criterion is not met
            try:
                if goal.shape[0] != latent.shape[0] or goal.shape[1] != latent.shape[1]:
                    goal = goal.expand(latent.shape[0], latent.shape[1], -1)
            except RuntimeError as e:
                print(f"Error in concatenating latent and goal: {e}.")
                print(
                    f"Latent shape: {latent.shape}, Goal shape: {goal.shape}, step: {step}"
                )
                raise e
        # Input to the worker actor is latent and goal concat # [B, 1, 3*Z]
        try:
            worker_input = torch.cat([latent, goal], dim=-1)  # [B, 1, *]
        except RuntimeError as e:
            print(f"Error in concatenating latent and goal: {e}.")
            print(
                f"Latent shape: {latent.shape}, Goal shape: {goal.shape}, step: {step}"
            )
            raise e

        # Finally generate primitive action distribution
        action_dist = self.worker.policy(worker_input)
        # TODO: Have mechnanism to save the goal for visualization
        # Update the carry state
        self.carry["step"] += 1  # everytime the policy step is called

        return action_dist  # , goal, skill

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        """
        Get the action from the Agent's policy distribution using the latent state.
        Based on greedy or sampling.
        Args:
            latent: The latent state from the world model. cat([sample, hidden]) [B, 1, *]
            goal: The Existing goal state from the world model. [B, 1, Z]
            skill: The Existing skill state from the world model. [B, 1, K, K]
            greedy: Whether to use greedy sampling or not.
        """
        self.eval()
        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):

            action_dist = self.policy_step(latent)
            if greedy:
                action = action_dist.probs.argmax(dim=-1)
            else:
                action = action_dist.sample()
        return action

    def sample_as_env_action(self, latent, greedy=False):
        # This is required to integrate with the train loop.
        action = self.sample(latent, greedy)
        return action.detach().cpu().squeeze(-1).numpy()

    def train_goal_vae_step(self, imagine_rollout: dict):
        """
        Train the skill encoder (GoalEncoder) and decoder (GoalDecoder)
        together using VAE loss (ELBO = Recon + KL), with optimizer step.
        """
        metrics = {}
        self.goal_encoder.train()
        self.goal_decoder.train()

        wm_sample = imagine_rollout["sample"]  # [B, L, Z]

        # --- Forward pass ---
        # Get encoded distribution
        encoded_dist = self.goal_encoder(wm_sample)  # q(z|x)
        skill_sample = encoded_dist.sample()
        # Get decoded distribution
        decoded_dist = self.goal_decoder(skill_sample)  # p(x|z)
        # Reconstruction loss (negative log-likelihood)
        # [B, L] -> [B]
        recon_loss = -decoded_dist.log_prob(wm_sample.detach()).mean(-1)
        # KL divergence
        # [B, L] -> [B]
        kl_loss = torch.distributions.kl_divergence(
            encoded_dist, self.skill_prior
        ).mean((-2, -1))
        kl_coef = self.kl_controller.update(kl_loss.detach())

        vae_loss = (recon_loss + kl_coef * kl_loss).mean()  # [B] -> scalar

        # --- Backward pass for VAE only ---
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(vae_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.optimizer.step()
        # --- Metrics ---
        metrics["Director/goal_recon_loss"] = recon_loss.mean().item()
        metrics["Director/goal_kl_loss"] = kl_loss.mean().item()
        metrics["Director/goal_VAE_loss"] = vae_loss.item()

        return metrics

    def train_manager_worker(self, imagine_rollout: dict):
        """
        Trains the manager and worker using the imagined rollout.
        Args:
            imagine_rollout: A dictionary containing imagined rollout data from the  WM.
        """
        metrics = {}
        imagine_rollout["reward_extr"] = self.extr_reward(imagine_rollout)
        imagine_rollout["reward_expl"] = self.explr_reward(imagine_rollout)
        imagine_rollout["reward_goal"] = self.goal_reward(imagine_rollout)
        # imagine_rollout["delta"] = imagine_rollout["goal"] - imagine_rollout["sample"]

        # generate the manager and worker trajectories
        manager_traj = self.manager_traj(imagine_rollout)
        worker_traj = self.worker_traj(imagine_rollout)
        # Train the manager and worker
        mets = self.worker.update(worker_traj)
        metrics.update({f"Worker_{k}": v for k, v in mets.items()})
        mets = self.manager.update(manager_traj)
        metrics.update({f"Manager_{k}": v for k, v in mets.items()})
        return imagine_rollout, metrics

    def manager_traj(self, imagine_rollout: dict) -> dict:
        """
        Modify trajectory to be used for training the manager.
        The time dimension is split into N sub-trajectories of length K; L = N*K
        Then averge/sum along the K dim to get output in shape [B, N, Z]
        Args:
            imagine_rollout: A dictionary containing WM rollout trajectory data.
        """
        traj = deepcopy(imagine_rollout)
        # for manager the action is the skill
        traj["action"] = traj.pop("skill")  # Replace "skill" with "action"
        # also pop the world model reward as its present as extr_reward
        traj.pop("reward")
        # remove the goal from the manager's trajectory; its not used in actor critics
        traj.pop("goal")
        traj["cont"] = 1 - traj["termination"]  # [1,1,1,0] -> [0,0,0,1] # [B, L]
        k = self.skill_duration  # Skill duration\
        reshape = lambda x: x.reshape(x.shape[0], x.shape[1] // k, k, *x.shape[2:])
        for key, value in traj.items():
            # For the manager the reward is the mean of the rewards in the skill duration
            if "reward" in key:
                # Compute weights for continuity along skill duration dimension
                # all the elements after zero would be 0; else 1
                weights = torch.cumprod((traj["cont"]), dim=1)  # B, L, 1
                # Average rewards weighted by continuity along N dimension
                # [B, L, *] -> [B, N, L, *] -> [B, N, *]
                traj[key] = reshape(value * weights).mean(dim=2)
            elif key in ["cont", "termination"]:
                # prod along the skill duration dimension. If one element is 0 then the product is 0
                traj[key] = reshape(value).prod(dim=2)  # [B, L]- > [B, N]
            else:  # [hidden, sample, action]
                # take the first one from every K
                traj[key] = reshape(value)[:, :, 0, :]  # [B, N, Z]

        # Compute trajectory weights
        traj["weight"] = (
            torch.cumprod(self.discount * traj["cont"], dim=1) / self.discount
        )  # [B, N] # example: [0.9, 0.81, 0.729, 0.6561, 0] # discount=0.9

        return traj

    def worker_traj(self, imagine_rollout: dict) -> dict:
        """
        Modifies the trajectory for worker training.
        There must be one dimesion for the skill duration (K) representing the trajectory.
        Output shape [B*N, K, F*] beacuse for the worker each sub trajectory is
        of len K, so the Batch and N dimensions are flattened into a single dimension.
        """
        traj = deepcopy(imagine_rollout)
        # also pop the world model reward as its present as extr_reward
        traj.pop("reward")
        traj["cont"] = 1 - traj["termination"]
        # TODO: Worker trajectory should have goal
        k = self.skill_duration  # Skill duration
        # Helper function to reshape tensors
        reshape = lambda x: x.reshape(x.shape[0], x.shape[1] // k, k, *x.shape[2:])
        for key, val in traj.items():
            val = reshape(val)  # [B, N, K, *]
            # Flatten batch dimensions (N and B) into a single dimension
            val = val.reshape(
                val.shape[0] * val.shape[1], -1, *val.shape[3:]
            )  # [B*N, K, F]
            # update the trajectory with the reshaped values
            traj[key] = val
        # Compute trajectory weights
        traj["weight"] = (
            torch.cumprod(self.discount * traj["cont"], dim=1) / self.discount
        )  # [B*N, K] # example: [0.9, 0.81, 0.729, 0.6561, 0] # discount=0.9
        return traj

    def update(self, imagine_rollout: dict):
        """
        Performs full update step:
        - VAE is trained here in DirectorAgent
        - Manager and Worker are trained separately via their own update() methods
        """
        metrics = {}
        success = lambda rew: (rew[:, -1] > 0.7).float().mean()

        self.train()

        # --- Train VAE only (GoalEncoder + GoalDecoder) ---
        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):
            vae_metrics = self.train_goal_vae_step(imagine_rollout)

        metrics.update(vae_metrics)

        # --- Train manager and worker independently ---
        traj, mw_metrics = self.train_manager_worker(imagine_rollout)
        metrics.update(mw_metrics)

        # --- Success tracking ---
        metrics["Director/success_manager"] = success(traj["reward_goal"]).item()

        return metrics
