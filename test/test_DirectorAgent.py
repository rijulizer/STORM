import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from sub_models.director_agents import DirectorAgent

# Write some sample example cases for GoalEncoder and GoalDecoder
wm_hidden_dim = 32
wm_sample_dim = 32
wm_action_dim = 4
B = 3
L = 16
wm_latent = torch.randn(B, L, wm_hidden_dim)
wm_sample = torch.randn(B, L, wm_hidden_dim)
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
}


## Define the DirectorAgent
agent = DirectorAgent(wm_hidden_dim, wm_sample_dim, wm_action_dim)

## Test external reward function
extr_reward = agent.extr_reward(imagine_rollout)
print(f"\n\nExternal reward shape: {extr_reward.shape}")

## Test explr_reward function
explr_reward = agent.explr_reward(imagine_rollout)
print(f"Exploration reward shape: {explr_reward.shape}")

## Test Goal reward function
goal_reward = agent.goal_reward(imagine_rollout)
print(f"Goal reward shape: {goal_reward.shape}")

# Test Policy step function
latent = torch.cat((imagine_rollout["sample"], imagine_rollout["hidden"]), dim=-1)
action_dist = agent.policy_step(latent)
print(f"\n\nAction distribution shape: {action_dist.sample().shape}")
print(f"steps after call: {agent.carry["step"]}")
print(f"skill shape: {agent.carry['skill'].shape}")
print(f"goal shape: {agent.carry['goal'].shape}")

## Test sample function
sampled_action = agent.sample(latent)
print(f"Sampled action shape: {sampled_action.shape}")


# ## Policy-step breakdown
# hidden = imagine_rollout["hidden"]  # [B, L, Z]
# sample = imagine_rollout["sample"]  # [B, L, Z]
# goal = imagine_rollout["goal"]  # [B, L, Z]
# B, L, Z = goal.shape
# latent = torch.cat((hidden, sample), dim=-1)  # [B, L, 2*]
# update_goal = True
# with torch.no_grad():
#     if update_goal:
#         # Get new skill and goal from the manager
#         # Get skill: manager actor logits from latent
#         # TODO: Director has a .sample()
#         skill = agent.manager.policy(latent)
#         # Decode new goal from skill #TODO: Director uses latent as a context
#         goal = agent.goal_decoder(skill).mode()
#         # imagine rollout
#         imagine_rollout["skill"] = skill  # [B, L, 64]
#         imagine_rollout["goal"] = goal  # [B, L, Z]
#     # FIXME: Deviating from director implementation
#     # Get worker action logits from latent and goal and delta
#     # Input to the worker actor is laent and goal concat # [B, L, 3*Z]
#     action_logits = agent.worker.actor(torch.cat((latent, goal), dim=-1))
#     # because finally action is discrete, we need to convert the logits to action distribution
#     action_dist = torch.distributions.Categorical(logits=action_logits)

# print(f"skill shape: {skill.shape}")
# print(f"goal shape: {goal.shape}")
# print(
#     f"action logits shape: {action_logits.shape}, action dist shape: {action_dist.probs.shape}"
# )

## Test GOAL-VAE
# metrics = agent.train_goal_vae_step(imagine_rollout)
# print(f"\n\nMetrics after training: {metrics}")


## Breakdown of train_goal_vae_step
# metrics = {}
# agent.goal_encoder.train()
# agent.goal_decoder.train()

# wm_sample = imagine_rollout["sample"]  # [B, L, Z]
# B, L = wm_sample.shape[:2]
# # Forward pass of encoder and decoder
# # Get encoded distribution
# encoded_dist = agent.goal_encoder(wm_sample)
# skill_sample = encoded_dist.sample()
# # Get decoded distribution
# decoded_dist = agent.goal_decoder(skill_sample)
# # Reconstruction loss (negative log-likelihood)
# recon_loss = -decoded_dist.log_prob(wm_sample.detach())
# recon_loss = recon_loss.mean(-1)  # [B, L] -> [B]

# # KL divergence
# kl_loss = torch.distributions.kl_divergence(encoded_dist, agent.skill_prior).mean(
#     (-2, -1)  # [B, L, K, K] -> [B, L, K] -> [B]
# )
# # # during training
# kl_coef = agent.kl_controller.update(kl_loss.detach())
# total_loss = (recon_loss + kl_coef * kl_loss).mean()

# # # Backward
# # # TODO: move the optimizer steps togather in the update function
# agent.optimizer.zero_grad()
# total_loss.backward()
# agent.optimizer.step()
# print(
#     f"""\n\nGaol-VAE Shapes-> \nskill_sample: {skill_sample.shape},
#     \nEncoded dist: {encoded_dist.sample().shape},
#     \nRecon_loss: {recon_loss.shape},
#     \nskill_prior: {agent.skill_prior.sample().shape},
#     \nkl_loss: {kl_loss.shape},
#     \nkl_coef: {kl_coef},
#     \ntotal_loss: {total_loss}"""
# )

# print("\n\nWM Imagine Rollout Shapes->")
# for k, v in imagine_rollout.items():
#     print(f"{k}: {v.shape}")


# ## Train Manger Worker
# imagine_rollout["reward_extr"] = agent.extr_reward(imagine_rollout)
# imagine_rollout["reward_expl"] = agent.explr_reward(imagine_rollout)
# imagine_rollout["reward_goal"] = agent.goal_reward(imagine_rollout)
# imagine_rollout["delta"] = imagine_rollout["goal"] - imagine_rollout["sample"]
# # print("\nTrajectory Shapes->")
# # for k, v in imagine_rollout.items():
# #     print(f"{k}: {v.shape}")

# ## Test manager trajectory
# manager_trajectory = agent.manager_traj(imagine_rollout)
# print("\n\nManager Trajectory Shapes->")
# for k, v in manager_trajectory.items():
#     print(f"{k}: {v.shape}")
"""
Manager Trajectory Shapes->
hidden: torch.Size([3, 3, 32])
sample: torch.Size([3, 3, 32])
action: torch.Size([3, 3, 64])
termination: torch.Size([3, 3])
goal: torch.Size([3, 3, 32])
reward_extr: torch.Size([3, 2])
reward_expl: torch.Size([3, 2])
reward_goal: torch.Size([3, 2])
delta: torch.Size([3, 3, 32])
cont: torch.Size([3, 3])
weight: torch.Size([3, 3])
"""
# ## Breakdown of manager_traj
# traj = imagine_rollout.copy()
# # for manager the action is the skill
# traj["action"] = traj.pop("skill")  # Replace "skill" with "action"
# # also pop the world model reward as its present as extr_reward
# traj.pop("reward")
# traj["cont"] = 1 - traj["termination"]  # [1,1,1,0] -> [0,0,0,1] # [B, L]

# k = agent.skill_duration  # Skill duration\
# reshape = lambda x: x.reshape(x.shape[0], x.shape[1] // k, k, *x.shape[2:])
# for key, value in traj.items():
#     # For the manager the reward is the mean of the rewards in the skill duration
#     if "reward" in key:
#         # Case example: [reward_extr, reward_expl, reward_goal]
#         # Compute weights for continuity along skill duration dimension
#         # all the elements after zero would be 0; else 1; cumprod on kth dim
#         weights = torch.cumprod((traj["cont"]), dim=1)  # [B, L-1, 1]
#         # Average rewards weighted by continuity along N dimension
#         # [B, L, *] -> [B, N, L, *] -> [B, N, *]
#         traj[key] = reshape(value * weights).mean(dim=2)
#     elif key in ["cont", "termination"]:
#         # cont has the shape [B, L]
#         # [B,1] + [B, N-1]
#         # prod along the skill duration dimension
#         # concat along the N dimension, If one element is 0 then the product is 0
#         traj[key] = torch.cat(
#             [value[:, 0].unsqueeze(1), reshape(value).prod(dim=2)], dim=1
#         )  # ->[B, N+1]
#     else:
#         #     # Last value for the last sub-trajectory
#         #     last_value = value[:, -1, :]  # [B, 1, Z]
#         #     first_values = value[:, :-1, :]  # First value for the first sub-trajectory
#         #     first_values = reshape(first_values)[:, :, 0, :]  # [B, N, Z]
#         #     traj[key] = torch.cat([first_values, last_value], dim=1)  # [B, N+1, Z]
#         traj[key] = reshape(value)[:, :, 0, :]  # take the first value
# traj["weight"] = torch.cumprod(agent.discount * traj["cont"], dim=1) / agent.discount

# print("\n\nManager Trajectory Shapes->")
# for key, value in traj.items():
#     print(f"{key}: {value.shape}")

## Test Worker Trajectory
# worker_trajerctory = agent.worker_traj(imagine_rollout)
# print("\n\nWorker Trajectory Shapes->")
# for k, v in worker_trajerctory.items():
#     print(f"{k}: {v.shape}")
"""
Worker Trajectory Shapes->
hidden: torch.Size([6, 9, 32])
sample: torch.Size([6, 9, 32])
action: torch.Size([6, 9])
termination: torch.Size([6, 9])
goal: torch.Size([6, 9, 32])
skill: torch.Size([6, 9, 64])
reward_extr: torch.Size([6, 8])
reward_expl: torch.Size([6, 8])
reward_goal: torch.Size([6, 8])
delta: torch.Size([6, 9, 32])
cont: torch.Size([6, 9])
weight: torch.Size([6, 9])
"""

## Breakdown


# traj = imagine_rollout.copy()
# # also pop the world model reward as its present as extr_reward
# traj.pop("reward")
# traj["cont"] = 1 - traj["termination"]
# k = agent.skill_duration  # Skill duration
# # assert (
# #     len(traj["action"]) % k == 1
# # ), "Trajectory length must be divisible by skill duration + 1."

# # Helper function to reshape tensors
# # [16,64] -> [2, 8, 64]; k=8
# reshape = lambda x: x.reshape(x.shape[0], x.shape[1] // k, k, *x.shape[2:])
# print("\n\nTrajectory Shapes->")
# for key, value in traj.items():
#     print(f"{key}: {value.shape}")
# print("\n\n")

# for key, val in traj.items():
#     if "reward" in key:
#         # Prepend a zero to align rewards with sub-trajectories
#         # val = torch.cat(
#         #     [torch.zeros_like(val[:, 0]), val], dim=1
#         # )  # Concat L dimension [B, L+1]
#         val[:, 0] = 0  # Set the first value to zero for the worker reward
#     # Split into overlapping sub-trajectories
#     # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
#     # Exclude the last element and reshape
#     # reshaped_val = reshape(val[:, :-1, :])  # [B, N, K, *]
#     reshaped_val = reshape(val)  # [B, N, K, *]
#     # Take every k-th element starting from k
#     # overlap = val[:, k::k].unsqueeze(2)  # [B, N, 1, F]
#     overlap = val[:, k - 1 :: k].unsqueeze(2)
#     val = torch.cat([reshaped_val, overlap], dim=2)  # (B, N, k+1, F)
#     # Flatten batch dimensions (N and B) into a single dimension
#     val = val.reshape(val.shape[0] * val.shape[1], -1, *val.shape[3:])  # [B*N, K+1, F]
#     # Remove the first sub-trajectory for rewards
#     if "reward" in key:
#         val = val[:, 1:]  # [B*N, K]
#     # update the trajectory with the reshaped values
#     traj[key] = val

# # Bootstrap sub-trajectory against the current goal, not the next
# traj["goal"] = torch.cat([traj["goal"][:, :-1, :], traj["goal"][:, :1, :]], dim=1)
# # Compute trajectory weights
# traj["weight"] = (
#     torch.cumprod(agent.discount * traj["cont"], dim=1) / agent.discount
# )  # [B*N, K+1] # example: [0.9, 0.81, 0.729, 0.6561, 0] # discount=0.9
# print("\n\nWorker Trajectory Shapes->")
# for key, value in traj.items():
#     print(f"{key}: {value.shape}")

# metrics = {}
# # generate the manager and worker trajectories
# manager_traj = agent.manager_traj(imagine_rollout)
# worker_traj = agent.worker_traj(imagine_rollout)
# # Train the manager and worker
# mets = agent.worker.update(worker_traj)
# metrics.update({f"worker_{k}": v for k, v in mets.items()})
# mets = agent.manager.update(manager_traj)
# metrics.update({f"manager_{k}": v for k, v in mets.items()})
# print(f"\n\nMetrics after training: {metrics}")
#
