import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from sub_models.director_agents import DirectorAgent, percentile, calc_lambda_return
from sub_models.constants import DEVICE, DTYPE_16

from pprint import pprint

# Write some sample example cases for GoalEncoder and GoalDecoder
wm_hidden_dim = 32
wm_sample_dim = 32
wm_action_dim = 4
skill_dim = 8
B = 3
L = 16
wm_latent = torch.randn(B, L+1, wm_hidden_dim)
wm_sample = torch.randn(B, L+1, wm_hidden_dim)
goal = torch.randn(B, L, wm_sample_dim)
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
# Send each tensor to the device
for k, v in imagine_rollout.items():
    if isinstance(v, torch.Tensor):
        # Check if the tensor is already on the device
        if v.device != DEVICE:
            # Move the tensor to the specified device
            imagine_rollout[k] = v.to(DEVICE)


## Define the DirectorAgent
agent = DirectorAgent(wm_hidden_dim, wm_sample_dim, wm_action_dim).to(DEVICE)

print("\n----------- Test Reward Functions -----------")
## Test external reward function
extr_reward = agent.extr_reward(imagine_rollout)
print(f"\n\nExternal reward shape: {extr_reward.shape, extr_reward.device}")
## Test explr_reward function
explr_reward = agent.explr_reward(imagine_rollout)
print(f"Exploration reward shape: {explr_reward.shape, explr_reward.device}")
## Test Goal reward function
goal_reward = agent.goal_reward(imagine_rollout)
print(f"Goal reward shape: {goal_reward.shape, goal_reward.device}")
"""
External reward shape: torch.Size([3, 16])
Exploration reward shape: torch.Size([3, 16])
Goal reward shape: torch.Size([3, 16])
"""

##Test Policy step function
print("\n\n----------- Test Policy Step -----------")
latent = torch.cat((imagine_rollout["sample"], imagine_rollout["hidden"]), dim=-1)[:, 0:1]
action_dist = agent.policy_step(latent)
print(f"Latent input shape: {latent.shape}")
print(f"Action distribution shape: {action_dist.sample().shape} in {action_dist.sample().device}")
print(f"steps after call: {agent.carry["step"]}")
print(f"skill shape: {agent.carry["skill"].shape} in {agent.carry["skill"].device}")
print(f"goal shape: {agent.carry["goal"].shape} in {agent.carry["goal"].device}")

action_dist = agent.policy_step(latent)
print(f"\n\nLatent input shape: {latent.shape}")
print(f"Action distribution shape: {action_dist.sample().shape} in {action_dist.sample().device}")
print(f"steps after call: {agent.carry["step"]}")
print(f"skill shape: {agent.carry["skill"].shape} in {agent.carry["skill"].device}")
print(f"goal shape: {agent.carry["goal"].shape} in {agent.carry["goal"].device}")

"""
Action distribution shape: torch.Size([3, 1])
steps after call: 1
skill shape: torch.Size([3, 1, 8, 8])
goal shape: torch.Size([3, 1, 32])
"""
## Test sample function
print("\n\n----------- Test Sample Function -----------")
sampled_action = agent.sample(latent)
print(f"Sampled action shape: {sampled_action.shape}")
print(sampled_action)
"Sampled action shape: torch.Size([3, 1])"

## Policy-step breakdown
# step = agent.carry["step"]
# if step % agent.skill_duration == 0:
#     # Get new skill and goal from the manager
#     # Get skill: manager actor logits from latent
#     skill = agent.manager.policy(latent).sample()
#     # Decode new goal from skill #TODO: Director uses latent as a context
#     goal = agent.goal_decoder(skill).mode()  # shape: [B, 1, goal_dim]
# # Input to the worker actor is latent and goal concat # [B, 1, 3*Z]
# worker_input = torch.cat([latent, goal], dim=-1)  # [B, 1, 3*Z]
# # Finally generate primitive action distribution
# action_dist = agent.worker.policy(worker_input)
# # TODO: Have mechnanism to save the goal for visualization
# agent.carry["step"] += 1  # everytime the policy step is called

# print(f"skill shape: {skill.shape}")
# print(f"goal shape: {goal.shape}")
# print(f"action dist shape: {action_dist.probs.shape}")
