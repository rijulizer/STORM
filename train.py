import gymnasium
import ale_py
import argparse
from tensorboardX import SummaryWriter
import cv2
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

# import sub_models.agents as agents
from sub_models.director_agents import DirectorAgent
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss

from sub_models.constants import DEVICE


def build_single_env(env_name: str, image_size: int, seed: int):
    """
    Build a single env with wrappers and preprocesses env.
    """
    env = gymnasium.make(
        env_name, full_action_space=False, render_mode="rgb_array", frameskip=1
    )
    # Convert int to tuple as gymnasium.wrappers.ResizeObservation requires tuple
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)

    return env


def build_vec_env(env_name: str, image_size: int, num_envs: int, seed: int):
    """
    Build a vectorized env with n=num_envs parallel envs.
    """

    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)

    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
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
    return DirectorAgent(
        conf.Models.WorldModel.TransformerHiddenDim,
        32 * 32,  # faltten sample dim
        action_dim,
    ).to(DEVICE)


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
    agent: DirectorAgent,
    imagine_batch_size,
    imagine_demonstration_batch_size,
    imagine_context_length,
    imagine_batch_length,
    log_video,
    logger,
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
    env_name: str,
    max_steps: int,
    num_envs: int,
    image_size: int,
    replay_buffer: ReplayBuffer,
    world_model: WorldModel,
    agent: DirectorAgent,
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
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    print(
        "Current env: "
        + colorama.Fore.YELLOW
        + f"{env_name}"
        + colorama.Style.RESET_ALL
    )

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample and train
    for total_steps in tqdm(range(max_steps // num_envs)):
        # sample part >>>
        if replay_buffer.ready:  # ready only after warmpup
            # WM and Agent are in eval mode
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    # this is the case in the first step
                    action = vec_env.action_space.sample()
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
                    )  # [1,1,1024], [1,1,512]
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False,
                    )
            # [B, H, W, C] -> [B, 1, C, H, W] # B=1
            context_obs.append(
                torch.permute(
                    torch.tensor(current_obs, device=DEVICE), (0, 3, 1, 2)
                ).unsqueeze(1)
                / 255
            )
            context_action.append(action)
        else:
            # sample single random action
            action = vec_env.action_space.sample()

        # Perform action in the env and observe the next state, reward, done, truncated
        # Single Unbatched instances: ((1, 64, 64, 3), (1,), (1,), (1,), (1,))
        obs, reward, done, truncated, info = vec_env.step(action)

        # Append the transition to the replay buffer
        replay_buffer.append(
            current_obs, action, reward, np.logical_or(done, info["life_loss"])
        )

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():  # end of episode
            for i in range(num_envs):
                if done_flag[i]:
                    # logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    # logger.log(
                    #     f"sample/{env_name}_episode_steps",
                    #     current_info["episode_frame_number"][i] // 4,
                    # )  # framskip=4
                    # logger.log("replay_buffer/length", len(replay_buffer))
                    # sum_reward[i] = 0
                    metrics[f"sample/{env_name}_reward"] = sum_reward[i]
                    metrics[f"sample/{env_name}_episode_steps"] = (
                        current_info["episode_frame_number"][i] // 4
                    )  # framskip=4
                    metrics["replay_buffer/length"] = len(replay_buffer)
                    sum_reward[i] = 0

        # Update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sample part

        # Train world model part >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if (
            replay_buffer.ready
            and total_steps % (train_dynamics_every_steps // num_envs) == 0
        ):
            print("Training World Model...")
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
            and total_steps % (train_agent_every_steps // num_envs) == 0
            and total_steps * num_envs >= 0
        ):
            print("Training Agent...")
            if total_steps % (save_every_steps // num_envs) == 0:
                log_video = True
            else:
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
                logger,
            )
            # Update agent with imagined data
            agent_metrics = agent.update(imagine_rollout)
            # update metrics
            metrics.update(agent_metrics)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train agent part

        # save model per episode
        # if total_steps % (save_every_steps // num_envs) == 0:
        #     print(
        #         colorama.Fore.GREEN
        #         + f"Saving model at total steps {total_steps}"
        #         + colorama.Style.RESET_ALL
        #     )
        #     torch.save(
        #         world_model.state_dict(),
        #         f"ckpt/{args.exp_name}/world_model_{total_steps}.pth",
        #     )
        #     torch.save(
        #         agent.state_dict(), f"ckpt/{args.exp_name}/agent_{total_steps}.pth"
        #     )
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
        dummy_env = build_single_env(
            args.env_name, conf.BasicSettings.ImageSize, seed=0
        )
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
