import gymnasium
import minigrid
import argparse
from functools import partial
from tensorboardX import SummaryWriter

import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os

from utils import seed_np_torch, Logger, load_config
from sub_models.replay_buffer import ReplayBuffer
import env_wrapper

from sub_models.agents import ActorCriticAgent

# from sub_models.director_agents import DirectorAgent

from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss

from sub_models.constants import DEVICE


def build_single_env(env_name: str, image_size: int, env_observablity: str = "Full"):
    """
    Build a single env with wrappers and preprocesses env.
    """
    env = gymnasium.make(env_name, render_mode="rgb_array")
    # Convert int to tuple as gymnasium.wrappers.ResizeObservation requires tuple
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if env_observablity == "Full":
        env = minigrid.wrappers.RGBImgObsWrapper(env)
    elif env_observablity == "Partial":
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
    else:
        raise ValueError(f"Unknown env observability {env_observablity}")
    env = minigrid.wrappers.RGBImgPartialObsWrapper(env)  # Adds an "rgb" key to the obs
    env = minigrid.wrappers.ImgObsWrapper(env)  # Sets obs = obs["rgb"], discards others
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    # env = env_wrapper.LifeLossInfo(env)

    return env


def build_vec_env(env_names: list[str], image_size: int, env_observablity):
    """
    Build a vectorized env with n=num_envs parallel envs.
    """
    env_fns = [
        partial(build_single_env, env_name, image_size, env_observablity)
        for env_name in env_names
    ]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def build_world_model(conf, action_dim):
    """
    Return a world model with the specified configuration
    """
    return WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
    ).to(DEVICE)


def build_agent(conf, action_dim: int):
    """
    Return an agent with the specified configuration
    """
    # ActorCriticAgent
    return ActorCriticAgent(
        feat_dim=32 * 32 + conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
    ).to(DEVICE)

    # DirectorAgent
    # return DirectorAgent(
    #     conf.Models.WorldModel.TransformerHiddenDim,
    #     32 * 32,  # faltten sample dim
    #     action_dim,
    # ).to(DEVICE)


def train_world_model(
    replay_buffer: ReplayBuffer,
    world_model: WorldModel,
    batch_size: int,
    demonstration_batch_size,
    batch_length: int,
    logger=None,
):
    """
    Train single step of the world model with the sampled data from replay buffer
    """
    # Sample from replay buffer
    buffer_sample = replay_buffer.sample(
        batch_size, demonstration_batch_size, batch_length
    )
    # obs: [B, L, 3, 64, 64], action: [B, L], reward: [B, L], termination: [B, L]
    # Train world model with the sampled data
    metrics = world_model.update(
        buffer_sample["obs"],
        buffer_sample["action"],
        buffer_sample["reward"],
        buffer_sample["termination"],
        logger=None,
    )
    return metrics


@torch.no_grad()
def world_model_imagine_data(
    replay_buffer: ReplayBuffer,
    world_model: WorldModel,
    agent: ActorCriticAgent,
    imagine_batch_size,
    imagine_demonstration_batch_size,
    imagine_context_length,
    imagine_batch_length,
    log_video,
    logger=None,
):
    """
    Sample context from replay buffer, then imagine data with world model and agent
    """
    world_model.eval()
    agent.eval()

    # a dictionary of sampled data from the replay buffer where each key is a tensor
    buffer_sample = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length
    )
    # Buffer sample items:
    # obs: ([B, L, 3, 64, 64]); action: ([B, L]); reward: ([B, L]); termination: ([B, L])

    imagined_rollout = world_model.imagine_data(
        agent,
        buffer_sample,
        imagine_batch_size=imagine_batch_size + imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger,
    )
    # Imagine rollout items:
    # sample: ([B, L+1, 1024]); hidden: ([B, L+1, 512])
    # action: ([B, L]); reward: ([B, L]); termination: ([B, L])
    # goal: ([B, L, 1024]); skill: ([B, L, 8, 8])
    return imagined_rollout


def joint_train_world_model_agent(
    env_names: list[str],
    max_steps: int,
    num_envs: int,
    env_observablity: str,
    image_size: int,
    replay_buffer: ReplayBuffer,
    world_model: WorldModel,
    agent: ActorCriticAgent,
    train_dynamics_every_steps,
    train_agent_every_steps,
    batch_size,
    demonstration_batch_size,
    batch_length,
    imagine_batch_size,
    imagine_demonstration_batch_size,
    imagine_context_length,
    imagine_batch_length,
    save_every_steps,
    seed,
    logger,
    args,
):
    metrics = {}
    os.makedirs(f"ckpt/{args.exp_name}", exist_ok=True)
    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_names, image_size, env_observablity)
    print(
        "Current env: "
        + colorama.Fore.YELLOW
        + f"{len(env_names)} parallel envs"
        + colorama.Style.RESET_ALL
    )

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    step_counters = np.zeros(num_envs, dtype=int)
    current_obs, current_info = vec_env.reset()  # [E, 64, 64, 3] #E=num_envs
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample and train
    for total_steps in tqdm(range(max_steps)):
        # sample part >>>
        if replay_buffer.ready:  # ready only after warmpup
            # WM and Agent are in eval mode
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    # this is the case in the first step
                    action = vec_env.action_space.sample()  # [E]
                else:
                    context_latent = world_model.encode_obs(
                        torch.cat(list(context_obs), dim=1)
                    )
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).to(DEVICE)
                    prior_flattened_sample, last_dist_feat = (
                        world_model.calc_last_dist_feat(
                            context_latent, model_context_action
                        )
                    )  # [E,n,1024], [E,n,512]
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False,
                    )  # [E]
            # [E, H, W, C] -> [E, 1, C, H, W]
            context_obs.append(
                torch.permute(
                    torch.tensor(current_obs, device=DEVICE), (0, 3, 1, 2)
                ).unsqueeze(1)
                / 255
            )
            context_action.append(action)
        else:
            # sample single random action
            action = vec_env.action_space.sample()  # [E]

        # Perform action in the env and observe the next state, reward, done, truncated
        # Single Unbatched instances: # ((4, 64, 64, 3), (4,), (4,), (4,), (4,)); E=4
        obs, reward, done, truncated, info = vec_env.step(action)

        # Append the transition to the replay buffer
        replay_buffer.append(current_obs, action, reward, done)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():  # end of episode
            for i in range(num_envs):
                if done_flag[i]:
                    env_id = env_names[i][:16]
                    # Log reward for this environment
                    metrics[f"sample/{env_id}_reward"] = sum_reward[i]
                    metrics[f"sample/{env_id}_episode_steps"] = step_counters[i]
                    metrics["replay_buffer/length"] = len(replay_buffer)
                    # Reset reward tracker and step counter
                    sum_reward[i] = 0
                    step_counters[i] = 0

        # Update current_obs, current_info and sum_reward
        sum_reward += reward  # [E]
        current_obs = obs
        current_info = info
        step_counters += 1
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sample part

        # Train world model part >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if replay_buffer.ready and total_steps % train_dynamics_every_steps == 0:
            # print("Training World Model...")
            wm_train_metrics = train_world_model(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                # logger=logger,
            )
            # update metrics
            metrics.update(wm_train_metrics)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train world model part

        # Train agent on WM imagined data >>>>>>>>>>>>>>>>>>>>>>>>>>>
        if (
            replay_buffer.ready
            and total_steps % train_agent_every_steps == 0
            and total_steps * num_envs >= 0
        ):
            # print("Training Agent...")
            # if total_steps % (save_every_steps // num_envs) == 0:
            #     log_video = True
            # else:
            log_video = False
            # Generate imagined rollout data
            imagine_rollout = world_model_imagine_data(
                replay_buffer,
                world_model,
                agent,
                imagine_batch_size,
                imagine_demonstration_batch_size,
                imagine_context_length,
                imagine_batch_length,
                log_video,
                # logger,
            )
            # Update agent with imagined data
            agent_metrics = agent.update(imagine_rollout)
            # update metrics
            metrics.update(agent_metrics)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train agent part
        # Update logs
        if logger is not None:
            for key, value in metrics.items():
                logger.log(key, value, step=total_steps)
        # save model per episode
        if total_steps % (save_every_steps // num_envs) == 0:
            print(
                colorama.Fore.GREEN
                + f"Saving model at total steps {total_steps}"
                + colorama.Style.RESET_ALL
            )
            torch.save(
                world_model.state_dict(),
                f"ckpt/{args.exp_name}/world_model_{total_steps}.pth",
            )
            torch.save(
                agent.state_dict(), f"ckpt/{args.exp_name}/agent_{total_steps}.pth"
            )
    return metrics


if __name__ == "__main__":
    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-trajectory_path", type=str, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=args.seed)
    # tensorboard writer
    logger = Logger(path=f"runs/{args.n}")
    # copy config file
    shutil.copy(args.config_path, f"runs/{args.n}/config.yaml")

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
        action_dim = dummy_env.action_space.n

        # build world model and agent
        world_model = build_world_model(conf, action_dim)
        agent = build_agent(conf, action_dim)

        # build replay buffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
        )

        # judge whether to load demonstration trajectory
        if conf.JointTrainAgent.UseDemonstration:
            print(
                colorama.Fore.MAGENTA
                + f"loading demonstration trajectory from {args.trajectory_path}"
                + colorama.Style.RESET_ALL
            )
            replay_buffer.load_trajectory(path=args.trajectory_path)

        # train
        joint_train_world_model_agent(
            env_name=args.env_name,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=(
                conf.JointTrainAgent.DemonstrationBatchSize
                if conf.JointTrainAgent.UseDemonstration
                else 0
            ),
            batch_length=conf.JointTrainAgent.BatchLength,
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=(
                conf.JointTrainAgent.ImagineDemonstrationBatchSize
                if conf.JointTrainAgent.UseDemonstration
                else 0
            ),
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            seed=args.seed,
            logger=logger,
            args=args,
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")
