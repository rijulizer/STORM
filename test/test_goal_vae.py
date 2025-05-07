import torch
from sub_models.director_agents import GoalEncoder, GoalDecoder, DirectorAgent

# Write some sample example cases for GoalEncoder and GoalDecoder
wm_hidden_dim = 16
skill_dim = (8, 8)
B = 3
L = 16
wm_state = torch.randn(B, L, wm_hidden_dim)
skill = torch.randn(B, L, skill_dim[0], skill_dim[1])
feat = torch.rand_like(wm_state)

print("Input shape: ", wm_state.shape)

## Test Encoder model
goal_encoder = GoalEncoder(wm_hidden_dim, skill_dim)
endoer_op = goal_encoder(wm_state)

print(f"Encoder smaple shape: {endoer_op.sample().shape}")
print(f"Encoder smaple example: {endoer_op.sample()[0][0]}")

## Test Decoder model
goal_decoder = GoalDecoder(skill_dim, wm_hidden_dim)
decoder_op = goal_decoder(skill)
print(f"DEcoder smaple shape: {decoder_op.log_prob(feat).shape}")
print(f"Decoder smaple example: {decoder_op.log_prob(feat)[0][0]}")

## Test Hierarchichal Agent Initiation
imagine_rollout = {"sample": wm_state}
director_agent = DirectorAgent(wm_hidden_dim)

## Test skill prior
print(f"\n\nSKill Prior: {director_agent.skill_prior.sample().shape}")

## Test ELBO Reward
elbo_op = director_agent.elbo_reward(imagine_rollout)
print(f"\n\nelbo rewrad shape: {elbo_op.shape}")
print(f"elbo rewrad sample: {elbo_op[0]}")

## Test tarin-goal-vae step
director_agent.train_goal_vae_step(imagine_rollout)
print(f"\nMetrics after training: {director_agent.metrics}")
