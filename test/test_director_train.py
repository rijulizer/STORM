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


def test_manager_traj():
    ## Test manager trajectory
    print("\n\n------------ Manager Trajectory -----------------")
    manager_trajectory = agent.manager_traj(imagine_rollout)
    print("Manager Trajectory Shapes->")
    for k, v in manager_trajectory.items():
        print(f"{k}: {v.shape, v.device}")
    """
    Manager Trajectory Shapes->
    hidden: torch.Size([3, 2, 32])
    sample: torch.Size([3, 2, 32])
    action: torch.Size([3, 2, 8, 8])
    termination: torch.Size([3, 2])

    reward_extr: torch.Size([3, 2])
    reward_expl: torch.Size([3, 2])
    reward_goal: torch.Size([3, 2])

    cont: torch.Size([3, 2])
    weight: torch.Size([3, 2])
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


def test_worker_traj():

    ## Test worker trajectory
    print("\n\n------------ Worker Trajectory -----------------")
    worker_trajerctory = agent.worker_traj(imagine_rollout)
    print("Worker Trajectory Shapes->")
    for k, v in worker_trajerctory.items():
        print(f"{k}: {v.shape}, {v.device}")
    """
    Worker Trajectory Shapes->
    hidden: torch.Size([6, 8, 32])
    sample: torch.Size([6, 8, 32])
    action: torch.Size([6, 8])
    termination: torch.Size([6, 8])
    cont: torch.Size([6, 8])

    goal: torch.Size([6, 8, 32])

    reward_extr: torch.Size([6, 8])
    reward_expl: torch.Size([6, 8])
    reward_goal: torch.Size([6, 8])


    weight: torch.Size([6, 8])
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


def test_worker():
    print("\n\n------------ Worker Initiation -----------------")
    worker_traj = agent.worker_traj(imagine_rollout)

    # ## Train the worker
    mets = agent.worker.update(worker_traj)
    print("\n\nWorker Metrics after training:")
    pprint(mets)


def test_worker_update():
    print("\n\n------------ Worker Update Broken Down-----------------")
    traj = agent.worker_traj(imagine_rollout)
    metrics = {}
    # All have the shape [B, L, *]
    hidden = traj["hidden"]  # The hidden state from WM
    sample = traj["sample"]  # The sample from WM
    # reward = imagine_rollout["reward"]
    action = traj["action"]
    # cont = imagine_rollout["cont"]
    termination = traj["termination"]
    goal = traj.get("goal", None)
    if goal is not None:
        # for the case of worker the goal is also part of latent
        latent = torch.cat((sample, hidden, goal), dim=-1)  # [B, L, 3*]
    else:
        latent = torch.cat((sample, hidden), dim=-1)  # [B, L, 2*]
    # Get action logits using actor model
    # action_logits = self.actor(latent)
    # # [B, L, action_dim]
    # action_dist = distributions.Categorical(logits=action_logits)
    action_dist = agent.worker.policy(latent)  # [B, L, action_dim]
    # get the log prob of the actual action
    # Expects action to have values between 0 and action_dim-1
    log_prob = action_dist.log_prob(action)  # [B, L]
    print(f"latent shape: {latent.shape}, \nlog_prob shape: {log_prob.shape}")
    total_critic_loss = 0.0
    total_value_loss = 0.0
    total_slow_value_loss = 0.0
    norm_aqdvantages = []  # TODO: check this logic

    # Iterate over all critics and calculate values
    for critic in agent.worker.critics:
        # get value for each critic model
        raw_value = critic["model"](latent)
        value = agent.worker.symlog_twohot_loss.decode(raw_value)

        # Generate critic reward function specific reward
        # reward functions operate on Deter in Director ~ Sample in STORM
        reward = traj[critic["reward"]]  # TODO: Check input sample
        lambda_return = calc_lambda_return(
            reward, value, termination, agent.worker.gamma, agent.worker.lambd
        )
        # get slow-value for each slow-critic-model
        slow_value = agent.worker.get_slow_value(critic["slow_model"], latent)
        slow_lambda_return = calc_lambda_return(
            reward, slow_value, termination, agent.worker.gamma, agent.worker.lambd
        )
        print(
            f"value shape: {value.shape}, \nlambda_return shape: {lambda_return.shape}, \nslow_lambda_return shape: {slow_lambda_return.shape}"
        )
        # update value function with slow critic regularization
        value_loss = agent.worker.symlog_twohot_loss(raw_value, lambda_return.detach())
        slow_value_regularization_loss = agent.worker.symlog_twohot_loss(
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
        total_critic_loss += scaled_value_loss + scaled_slow_value_regularization_loss

        lower_bound = agent.worker.lowerbound_ema(percentile(lambda_return, 0.05))
        upper_bound = agent.worker.upperbound_ema(percentile(lambda_return, 0.95))
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
    policy_loss = -(log_prob * avg_norm_advantage.detach()).mean()
    entropy_loss = action_dist.entropy().mean()

    # Calculate total loss
    loss = policy_loss + total_critic_loss - (agent.worker.entropy_coef * entropy_loss)

    # gradient descent
    if agent.worker.scaler is not None:
        agent.worker.scaler.scale(loss).backward()
        agent.worker.scaler.unscale_(agent.worker.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(agent.worker.parameters(), max_norm=1000.0)
        agent.worker.scaler.step(agent.worker.optimizer)
        agent.worker.scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.worker.parameters(), max_norm=1000.0)
        agent.worker.optimizer.step()
    agent.worker.optimizer.zero_grad(set_to_none=True)

    agent.worker.update_slow_critic()
    # Update metrics
    metrics["AC/policy_loss"] = policy_loss.item()
    metrics["AC/critic_loss"] = total_critic_loss.item()
    metrics["AC/entropy_loss"] = entropy_loss.item()
    metrics["AC/S"] = S.item()
    metrics["AC/norm_ratio"] = norm_ratio.item()
    metrics["AC/total_loss"] = loss.item()


def test_manager():
    print("\n\n------------ Manager Initiation -----------------")


def test_manager_update():
    print("\n\n------------ Manager Update Broken Down-----------------")
    traj = agent.manager_traj(imagine_rollout)
    metrics = {}
    # All have the shape [B, L, *]
    hidden = traj["hidden"]  # The hidden state from WM
    sample = traj["sample"]  # The sample from WM
    # reward = imagine_rollout["reward"]
    action = traj["action"]
    # cont = imagine_rollout["cont"]
    termination = traj["termination"]
    goal = traj.get("goal", None)
    if goal is not None:
        # for the case of worker the goal is also part of latent
        latent = torch.cat((sample, hidden, goal), dim=-1)  # [B, L, 3*]
    else:
        latent = torch.cat((sample, hidden), dim=-1)  # [B, L, 2*]
    # Get action logits using actor model
    # action_logits = self.actor(latent)
    # # [B, L, action_dim]
    # action_dist = distributions.Categorical(logits=action_logits)
    action_dist = agent.manager.policy(latent)  # [B, L, action_dim]
    # get the log prob of the actual action
    # Expects action to have values between 0 and action_dim-1
    log_prob = action_dist.log_prob(action)  # [B, L]
    print(f"latent shape: {latent.shape}, \nlog_prob shape: {log_prob.shape}")
    total_critic_loss = 0.0
    total_value_loss = 0.0
    total_slow_value_loss = 0.0
    norm_aqdvantages = []  # TODO: check this logic

    # Iterate over all critics and calculate values
    for critic in agent.manager.critics:
        # get value for each critic model
        raw_value = critic["model"](latent)
        value = agent.manager.symlog_twohot_loss.decode(raw_value)

        # Generate critic reward function specific reward
        # reward functions operate on Deter in Director ~ Sample in STORM
        reward = traj[critic["reward"]]  # TODO: Check input sample
        lambda_return = calc_lambda_return(
            reward, value, termination, agent.manager.gamma, agent.manager.lambd
        )
        # get slow-value for each slow-critic-model
        slow_value = agent.manager.get_slow_value(critic["slow_model"], latent)
        slow_lambda_return = calc_lambda_return(
            reward, slow_value, termination, agent.manager.gamma, agent.manager.lambd
        )
        print(
            f"value shape: {value.shape}, \nlambda_return shape: {lambda_return.shape}, \nslow_lambda_return shape: {slow_lambda_return.shape}"
        )
        # update value function with slow critic regularization
        value_loss = agent.manager.symlog_twohot_loss(raw_value, lambda_return.detach())
        slow_value_regularization_loss = agent.manager.symlog_twohot_loss(
            raw_value, slow_lambda_return.detach()
        )
        # Apply the critic scale as a multiplicative factor
        # #TODO: for now the scales are used to scale lossess
        scaled_value_loss = critic["scale"] * value_loss
        scaled_slow_value_regularization_loss = (
            critic["scale"] * slow_value_regularization_loss
        )
        # update the critic losses
        total_value_loss += scaled_value_loss
        total_slow_value_loss += scaled_slow_value_regularization_loss
        total_critic_loss += scaled_value_loss + scaled_slow_value_regularization_loss
        print(f"total_critic_loss: {total_critic_loss}")
        lower_bound = agent.manager.lowerbound_ema(percentile(lambda_return, 0.05))
        upper_bound = agent.manager.upperbound_ema(percentile(lambda_return, 0.95))
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
    policy_loss = -(log_prob * avg_norm_advantage.detach()).mean()
    entropy_loss = action_dist.entropy().mean()

    # Calculate total loss
    loss = policy_loss + total_critic_loss - (agent.manager.entropy_coef * entropy_loss)

    # gradient descent
    if agent.manager.scaler is not None:
        agent.manager.scaler.scale(loss).backward()
        agent.manager.scaler.unscale_(agent.manager.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(agent.manager.parameters(), max_norm=1000.0)
        agent.manager.scaler.step(agent.manager.optimizer)
        agent.manager.scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.manager.parameters(), max_norm=1000.0)
        agent.manager.optimizer.step()
    agent.manager.optimizer.zero_grad(set_to_none=True)

    agent.manager.update_slow_critic()
    # Update metrics
    metrics["AC/policy_loss"] = policy_loss.item()
    metrics["AC/critic_loss"] = total_critic_loss.item()
    metrics["AC/entropy_loss"] = entropy_loss.item()
    metrics["AC/S"] = S.item()
    metrics["AC/norm_ratio"] = norm_ratio.item()
    metrics["AC/total_loss"] = loss.item()


def test_manager_worker_training():
    print("\n\n------------ Manager-Worker Training -----------------")
    ## Test: Train Manger Worker
    print("\n\nTest: Train Manger Worker-->")
    metrics = agent.train_manager_worker(imagine_rollout)
    pprint(metrics)


def test_manager_worker_training_breakdown():
    print("\n\n------------ Manager-Worker Training Break Down-----------------")
    # Breakdown of the manager-worker training
    # Check device of manager and worker
    print(f"Manager device: {next(agent.manager.actor.parameters()).device}")
    print(
        f"Manager critics device: {next(agent.manager.critics[0]['model'].parameters()).device}"
    )
    print(f"Worker device: {next(agent.worker.actor.parameters()).device}")
    print(
        f"Worker critics device: {next(agent.worker.critics[0]['model'].parameters()).device}"
    )
    metrics = {}
    ## generate the manager and worker trajectories
    manager_traj = agent.manager_traj(imagine_rollout)
    worker_traj = agent.worker_traj(imagine_rollout)

    ## Train the worker
    mets = agent.worker.update(worker_traj)
    metrics.update({f"worker_{k}": v for k, v in mets.items()})
    # ## Train the manager
    mets = agent.manager.update(manager_traj)
    metrics.update({f"manager_{k}": v for k, v in mets.items()})
    print(f"\n\nMetrics after training:")
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}, {v.device}")
        else:
            print(f"{k}: {v}")

    # test main update()
    print("\n\nTest main update()-->")
    final_metrics = agent.update(imagine_rollout)
    for k, v in final_metrics.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}, {v.device}")
        else:
            print(f"{k}: {v}")


## Run test functions
test_manager_traj()
test_worker_traj()

test_worker_update()
test_worker()

test_manager_update()
test_manager()

test_manager_worker_training()
test_manager_worker_training_breakdown()
