import torch
from sub_models.constants import DEVICE, DTYPE_16
from sub_models.director_agents import GoalEncoder, GoalDecoder, DirectorAgent

# Write some sample example cases for GoalEncoder and GoalDecoder
# Write some sample example cases for GoalEncoder and GoalDecoder
wm_hidden_dim = 32
wm_sample_dim = 32
wm_action_dim = 4
skill_dim = 8
B = 3
L = 16
wm_latent = torch.randn(B, L, wm_hidden_dim)
wm_sample = torch.randn(B, L, wm_hidden_dim)
skill = torch.randn(B, L, skill_dim, skill_dim)
wm_termination = torch.zeros(B, L)
wm_termination[:, -1] = 1
wm_action = torch.randint(low=0, high=wm_action_dim, size=(B, L))  # Shape: [B, L]
wm_reward = torch.randn(B, L)  # Shape: [B, L]

imagine_rollout = {
    "hidden": wm_latent,
    "sample": wm_sample,
    "action": wm_action,
    "termination": wm_termination,
    "reward": wm_reward,
    "goal": wm_sample,
    "skill": skill,
}
print("------- Test GoalEncoder-------")
## Test Encoder model
goal_encoder = GoalEncoder(wm_hidden_dim, (skill_dim, skill_dim))
endoer_op = goal_encoder(wm_sample)

print(f"\n\nEncoder smaple shape: {endoer_op.sample().shape}")
print(f"Encoder smaple example: {endoer_op.sample()[0][0]}")

print("\n------- Test GoalDecoder-------")
## Test Decoder model
goal_decoder = GoalDecoder((skill_dim, skill_dim), wm_hidden_dim)
decoder_op = goal_decoder(endoer_op.sample())
print(f"\n\nDecoder dist mode shape: {decoder_op.mode().shape}")
print(f"Decoder dist mode sample: {decoder_op.mode()[0][0]}")
print(f"Decoder dist log_prob shape: {decoder_op.log_prob(wm_sample).shape}")


print("\n------- Test Goal VAE Training-------")
## Define the DirectorAgent
agent = DirectorAgent(wm_hidden_dim, wm_sample_dim, wm_action_dim).to(DEVICE)

# ## Test tarin-goal-vae step
# Move all tensors in imagine_rollout to DEVICE
imagine_rollout_on_device = {
    k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in imagine_rollout.items()
}
metrics = agent.train_goal_vae_step(imagine_rollout_on_device)
print(f"\nMetrics after training: {metrics}")


## Test GOAL-VAE
# print("\n\nTest GOAL-VAE-->")
# metrics = agent.train_goal_vae_step(imagine_rollout)
# print(f"Metrics after training: {metrics}")
## Breakdown of the goal-vae
# metrics = {}
# agent.goal_encoder.train()
# agent.goal_decoder.train()

# wm_sample = imagine_rollout["sample"]  # [B, L, Z]

# # --- Forward pass ---
# # Get encoded distribution
# encoded_dist = agent.goal_encoder(wm_sample)  # q(z|x)
# skill_sample = encoded_dist.sample()
# # Get decoded distribution
# decoded_dist = agent.goal_decoder(skill_sample)  # p(x|z)
# # Reconstruction loss (negative log-likelihood)
# # [B, L] -> [B]
# recon_loss = -decoded_dist.log_prob(wm_sample.detach()).mean(-1)
# # KL divergence
# # [B, L] -> [B]
# kl_loss = torch.distributions.kl_divergence(
#     encoded_dist, agent.skill_prior
# ).mean((-2, -1))
# kl_coef = agent.kl_controller.update(kl_loss.detach().cpu())

# vae_loss = (recon_loss + kl_coef * kl_loss).mean()  # [B] -> scalar

# # --- Backward pass for VAE only ---
# self.optimizer.zero_grad(set_to_none=True)
# if self.scaler is not None:
#     self.scaler.scale(vae_loss).backward()
#     self.scaler.unscale_(self.optimizer)
#     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
#     self.scaler.step(self.optimizer)
#     self.scaler.update()
# else:
#     vae_loss.backward()
#     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
#     self.optimizer.step()
# # --- Metrics ---
# metrics["Director/goal_recon_loss"] = recon_loss.mean().item()
# metrics["Director/goal_kl_loss"] = kl_loss.mean().item()
# metrics["Director/goal_VAE_loss"] = vae_loss.item()
# print(f"\n\nskill_sample device: {skill_sample.device}")
# print(f"recon_loss device: {recon_loss.device}")
# print(f"kl_loss device: {kl_loss.device, kl_loss.detach().device}")
# print(f"VAE_loss device: {vae_loss.device}")
