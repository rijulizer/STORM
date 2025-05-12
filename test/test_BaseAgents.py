import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from sub_models.director_agents import BaseAgent

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

imagine_rollout = {
    "hidden": wm_latent,
    "sample": wm_sample,
    "action": wm_action,
    "termination": wm_termination,
    "goal": wm_sample,
}


def goal_reward(imagine_rollout):
    """
    Cosine Max similarity
    Calculate reward based on the goal and the transition state.
    """
    with torch.no_grad():  # Stop gradient
        wm_sample = imagine_rollout["sample"]
        goal = imagine_rollout["goal"]
        print("Shapes: ", wm_sample.shape, goal.shape)
        # calculate normalization factor
        norm = torch.maximum(
            goal.norm(dim=-1, keepdim=True), wm_sample.norm(dim=-1, keepdim=True)
        ).clamp_min(1e-12)
        # [B, L, Z] -> [B, L]
        reward = (goal / norm * wm_sample / norm).sum(dim=-1)
        # return the second element onward [B, L-1]
        return reward[:, 1:]


## Define the BaseAgent
worker = BaseAgent(
    critics=[
        {"critic": "goal", "scale": 1.0, "reward_fn": goal_reward},
    ],
    input_dim=wm_hidden_dim + wm_sample_dim + wm_sample_dim,
    action_dim=wm_action_dim,
)
# ## Actor model
# print("Actor model: ", worker.actor)

# ## Critic model
# print("Critic model: ", worker.critics)

## Test sample
x = torch.randn(B, L, wm_hidden_dim + wm_sample_dim + wm_sample_dim)
# print("Sample shape: ", worker.sample(x).shape)

## Test Update function
metrics = worker.update(imagine_rollout)
print("Metrics: ", metrics)

## Update Breakdown
# latent = torch.cat((wm_latent, wm_sample, wm_sample), dim=-1)  # [B, L, 3*]
# action_logits = worker.actor(latent)  # [B, L, action_dim]
# action_dist = distributions.Categorical(logits=action_logits)
# print(action_logits.shape, action_dist.probs.shape, wm_action.argmax(dim=-1).shape)
# log_prob = action_dist.log_prob(wm_action)  # [B, L]
# print(action_logits.shape, action_dist.probs.shape, log_prob.shape)

## test goal reward function
# reward = goal_reward(imagine_rollout)
# print("Reward shape: ", reward.shape)
