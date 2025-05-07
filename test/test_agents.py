import torch
from sub_models.director_agents import BaseAgent

# Write some sample example cases for GoalEncoder and GoalDecoder
wm_hidden_dim = 16
action_dim = 4
B = 3
L = 16
wm_latent = torch.randn(B, L, wm_hidden_dim)
wm_sample = torch.randn(B, L, wm_hidden_dim)
wm_termination = torch.zeros(B, L)
wm_termination[:, -1] = 1
wm_action = torch.randn(B, L, action_dim)
imagine_rollout = {
    "latent": wm_latent,
    "sample": wm_sample,
    "action": wm_action,
    "wm_termination": wm_termination,
}
# feat = torch.rand_like(wm_state)
critics = [{"critic": "test_name", "scale": 0.5, "reward_fn": "test_callable"}]
input_dim = wm_hidden_dim * 2

## Test Base agent initiation
base_agent = BaseAgent(critics, input_dim, action_dim)
