import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from sub_models.director_agents import (
    BaseAgent,
    DirectorAgent,
    percentile,
    calc_lambda_return,
)
from sub_models.constants import DEVICE, DTYPE_16

from pprint import pprint

wm_hidden_dim = 32
wm_sample_dim = 32
wm_action_dim = 4
skill_dim = 8
B = 13
L = 16
wm_latent = torch.randn(B, L + 1, wm_hidden_dim)
wm_sample = torch.randn(B, L + 1, wm_hidden_dim)
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

imagine_rollout["reward_extr"] = agent.extr_reward(imagine_rollout)
imagine_rollout["reward_expl"] = agent.explr_reward(imagine_rollout)
imagine_rollout["reward_goal"] = agent.goal_reward(imagine_rollout)

manager_traj = agent.manager_traj(imagine_rollout)
worker_traj = agent.worker_traj(imagine_rollout)

worker = BaseAgent(
    critics={
        "critic_goal": {"reward": "reward_goal", "scale": 1.0},
    },
    input_dim=agent.wm_feat_dim + agent.wm_sample_dim,  # goal_dim = wm_sample_dim
    action_dim=wm_action_dim,
    actor_dist="Categorical",
).to(DEVICE)

manager = BaseAgent(
    critics={
        "critic_extr": {"reward": "reward_extr", "scale": 1.0},
        "critic_expl": {"reward": "reward_expl", "scale": 0.3},
    },
    input_dim=agent.wm_feat_dim,
    action_dim=agent.skill_shape,
    actor_dist="OneHotDist",  # "MultiOneHotCategorical",
).to(DEVICE)


def test_manager_base_update():
    """
    Test the Manager base agent.
    """
    print("\n----------------Test Manager Base Agent-----------------")
    metrics = {}
    manager_traj = agent.manager_traj(imagine_rollout)
    mets = agent.manager.update(manager_traj)
    metrics.update({f"Manager_{k}": v for k, v in mets.items()})
    pprint(metrics)


def test_update_broken(agent, traj):
    """
    Test the update_broken method.
    """

    metrics = {}
    agent.train()
    with torch.autocast(device_type=DEVICE.type, dtype=DTYPE_16, enabled=agent.use_amp):
        # All have the shape [B, L, *]
        hidden = traj["hidden"]  # The hidden state from WM
        sample = traj["sample"]  # The sample from WM
        # reward = imagine_rollout["reward"]
        action = traj["action"]
        # cont = imagine_rollout["cont"]
        termination = traj["termination"]
        weights = traj["weight"].detach()  # [B, L] # weights for the trajectory
        goal = traj.get("goal", None)
        if goal is not None:
            # for the case of worker the goal is also part of latent
            latent = torch.cat((sample, hidden, goal), dim=-1)  # [B, L, 3*]
        else:
            latent = torch.cat((sample, hidden), dim=-1)  # [B, L, 2*]
        # Get action logits using actor model
        # action_logits = agent.actor(latent)
        # # [B, L, action_dim]
        # action_dist = distributions.Categorical(logits=action_logits)
        action_dist = agent.policy(latent)  # [B, L, action_dim]
        # get the log prob of the actual action
        # Expects action to have values between 0 and action_dim-1
        log_prob = action_dist.log_prob(action)  # [B, L]
        print(f"log_prob shape {log_prob.shape}")
        print(f"action[0] shpe: {action[0].shape}")
        total_critic_loss = 0.0
        total_value_loss = 0.0
        total_slow_value_loss = 0.0
        norm_aqdvantages = []  # TODO: check this logic

        # Iterate over all critics and calculate values
        for name, critic in agent.critics.items():
            # get value for each critic model
            raw_value = critic["model"](latent)
            value = agent.symlog_twohot_loss.decode(raw_value)
            print("DEBUG: value shape", value.shape)
            # Generate critic reward function specific reward
            # reward functions operate on Deter in Director ~ Sample in STORM
            reward = traj[critic["reward"]]
            lambda_return = calc_lambda_return(
                reward, value, termination, agent.gamma, agent.lambd
            )
            # get slow-value for each slow-critic-model
            slow_value = agent.get_slow_value(critic["slow_model"], latent)
            slow_lambda_return = calc_lambda_return(
                reward, slow_value, termination, agent.gamma, agent.lambd
            )

            # update value function with slow critic regularization
            value_loss = agent.symlog_twohot_loss(raw_value, lambda_return.detach())
            slow_value_regularization_loss = agent.symlog_twohot_loss(
                raw_value, slow_lambda_return.detach()
            )  # [:, :-1]

            # update the critic losses
            total_value_loss += value_loss * critic["scale"]
            total_slow_value_loss += slow_value_regularization_loss * critic["scale"]
            total_critic_loss += (value_loss + slow_value_regularization_loss) * critic[
                "scale"
            ]
            print(f"\nCritic: {name}, Scale: {critic["scale"]}")
            print(
                f"\nscaled value_loss: {value_loss.item()* critic["scale"]} \n scaled slow_value_regularization_loss: {slow_value_regularization_loss.item()* critic["scale"]}"
            )
            print(
                f"\ntotal_value_loss: {total_value_loss.item()} \ntotal_slow_value_loss: {total_slow_value_loss.item()}"
            )
            print(f"\ntotal_critic_loss: ", total_critic_loss.item())
            lower_bound = agent.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = agent.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound - lower_bound
            norm_ratio = torch.max(torch.ones(1).to(DEVICE), S)
            norm_aqdvantages.append((lambda_return - value) / norm_ratio)  # [:, :-1]

        # Calcuate the average normed advantage
        avg_norm_advantage = torch.mean(
            torch.stack(norm_aqdvantages), dim=0
        )  # TODO: Check this logic #Dennis
        # Calculate Actor related losses
        if len(log_prob.shape) == 3:
            # for manager the log_prob is [B, L, K]
            avg_norm_advantage = avg_norm_advantage.unsqueeze(-1)  # [B,L,1]
            weights = weights.unsqueeze(-1)  # [B, L, 1]
        print(f"policy_loss shape {(log_prob * avg_norm_advantage.detach()).shape}")
        print(f"traj weights shape {weights.shape}")

        policy_loss = -(
            log_prob * avg_norm_advantage.detach() * weights
        ).sum()  # [B, L]->scalar
        entropy_loss = action_dist.entropy().mean()

        # Calculate total loss
        loss = policy_loss + total_critic_loss - (agent.entropy_coef * entropy_loss)

    # gradient descent
    if agent.scaler is not None:
        agent.scaler.scale(loss).backward()
        agent.scaler.unscale_(agent.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1000.0)
        agent.scaler.step(agent.optimizer)
        agent.scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1000.0)
        agent.optimizer.step()
    agent.optimizer.zero_grad(set_to_none=True)

    agent.update_slow_critic()
    # Update metrics
    metrics["AC/policy_loss"] = policy_loss.item()
    metrics["AC/critic_loss"] = total_critic_loss.item()
    metrics["AC/entropy_loss"] = entropy_loss.item()
    metrics["AC/S"] = S.item()
    metrics["AC/norm_ratio"] = norm_ratio.item()
    metrics["AC/total_loss"] = loss.item()
    pprint(metrics)


if __name__ == "__main__":
    # Test the manager base agent
    # test_manager_base_agent()

    # Test the update_broken method
    print("\n----------------Test Worker Base agent Update Broken-----------------")
    test_update_broken(worker, worker_traj)

    print("\n----------------Test Manager Base agent Update Broken-----------------")
    test_update_broken(manager, manager_traj)

    ## modify manager/wroker reward scales
    print(
        "\n----------------Test Manager Base agent Update Broken with modified scales-----------------"
    )
    worker.critics["critic_goal"]["scale"] = 100
    manager.critics["critic_extr"]["scale"] = 100
    manager.critics["critic_expl"]["scale"] = 0.001
    print(f"\t-----worker--------")
    test_update_broken(worker, worker_traj)
    print(f"\t-----manager--------")
    test_update_broken(manager, manager_traj)
