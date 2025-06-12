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
import wandb
import importlib

from utils import seed_np_torch, Logger, load_config
from sub_models.replay_buffer import ReplayBuffer
from train import (
    build_single_env,
    build_vec_env,
    build_world_model,
    build_agent,
    joint_train_world_model_agent,
)
from sub_models.constants import DEVICE

# ignore warnings
import warnings

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class WandbLogger:
    def __init__(self, run):
        self.run = run

    def log(self, key, value, step=None):
        """Log a key-value pair to wandb with optional step."""
        log_dict = {key: value}
        if step is not None:
            self.run.log(log_dict, step=step)
        else:
            self.run.log(log_dict)


class RunParams:
    def __init__(self, env_names, exp_name: str):
        self.exp_name = exp_name
        self.seed = 1
        self.config_path = "config_files/STORM.yaml"
        # self.trajectory_path = f"D_TRAJ/{self._env_name}.pkl"
        self.env_names = env_names

        self.conf = load_config(self.config_path)
        self.print_args()

    def print_args(self):
        print(colorama.Fore.GREEN + "Arguments:" + colorama.Style.RESET_ALL)
        print(colorama.Fore.GREEN + "-----------------" + colorama.Style.RESET_ALL)
        print(
            colorama.Fore.GREEN
            + "exp_name: "
            + colorama.Style.RESET_ALL
            + self.exp_name
        )
        print(
            colorama.Fore.GREEN + "seed: " + colorama.Style.RESET_ALL + str(self.seed)
        )
        # print(colorama.Fore.GREEN + "config_path: " + colorama.Style.RESET_ALL + self.config_path)
        print(colorama.Fore.GREEN + "env_name: " + colorama.Style.RESET_ALL)
        print(self.env_names)
        print(colorama.Fore.GREEN + "-----------------" + colorama.Style.RESET_ALL)


def main():
    env_names = [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-SimpleCrossingS9N3-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-FourRooms-v0",
    ]
    run_params = RunParams(env_names, exp_name="MultiEnv-Baseline_v4")
    # set seed
    seed_np_torch(seed=run_params.seed)
    # copy config file
    # os.makedirs(f"runs/{run_params.exp_name}", exist_ok=True)
    # shutil.copy(run_params.config_path, f"runs/{run_params.exp_name}/config.yaml")

    print(f"Train Steps: {run_params.conf.JointTrainAgent.SampleMaxSteps}")
    print(f"Train Batch Size: {run_params.conf.JointTrainAgent.BatchSize}")
    print(f"Train Buffer Max Length: {run_params.conf.JointTrainAgent.BufferMaxLength}")
    # Setuop env, models, replay buffer
    # getting action_dim with dummy env
    dummy_env = build_single_env(
        run_params.env_names[0], run_params.conf.BasicSettings.ImageSize
    )
    action_dim = dummy_env.action_space.n

    # build world model and agent
    world_model = build_world_model(run_params.conf, action_dim)
    agent = build_agent(run_params.conf, action_dim)
    print(
        f"World model transformer: {world_model.storm_transformer.__class__.__name__}"
    )
    # Log the number of parameters for both models
    world_model_params = sum(
        p.numel() for p in world_model.parameters() if p.requires_grad
    )
    agent_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    # build replay buffer
    replay_buffer = ReplayBuffer(
        obs_shape=(
            run_params.conf.BasicSettings.ImageSize,
            run_params.conf.BasicSettings.ImageSize,
            3,
        ),
        num_envs=len(run_params.env_names),
        max_length=run_params.conf.JointTrainAgent.BufferMaxLength,
        warmup_length=run_params.conf.JointTrainAgent.BufferWarmUp,
        store_on_gpu=run_params.conf.BasicSettings.ReplayBufferOnGPU,
    )
    # judge whether to load demonstration trajectory
    if run_params.conf.JointTrainAgent.UseDemonstration:
        print(
            colorama.Fore.MAGENTA
            + f"loading demonstration trajectory from {run_params.trajectory_path}"
            + colorama.Style.RESET_ALL
        )
        replay_buffer.load_trajectory(path=run_params.trajectory_path)

    # Initialize wandb
    with wandb.init(
        project="Thesis",  # Replace with your project name
        name=run_params.exp_name,  # Use the experiment name from RunParam
        config={
            "env_name": str(run_params.env_names),
            "num_envs": len(run_params.env_names),
            "seed": run_params.seed,
        },
    ) as run:
        # Log the configuration to wandb
        run.config.update(run_params.conf)
        run.log(
            {
                "WM_params": f"{world_model_params:.2e}",
                "Agent_params": f"{agent_params:.2e}",
            }
        )
        logger = WandbLogger(run)
        # logger = None
        ## train
        joint_train_world_model_agent(
            env_names=run_params.env_names,
            num_envs=len(run_params.env_names),
            max_steps=run_params.conf.JointTrainAgent.SampleMaxSteps,
            env_observablity=run_params.conf.BasicSettings.EnvObservability,
            image_size=run_params.conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=run_params.conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=run_params.conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=run_params.conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=(
                run_params.conf.JointTrainAgent.DemonstrationBatchSize
                if run_params.conf.JointTrainAgent.UseDemonstration
                else 0
            ),
            batch_length=run_params.conf.JointTrainAgent.BatchLength,
            imagine_batch_size=run_params.conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=(
                run_params.conf.JointTrainAgent.ImagineDemonstrationBatchSize
                if run_params.conf.JointTrainAgent.UseDemonstration
                else 0
            ),
            imagine_context_length=run_params.conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=run_params.conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=run_params.conf.JointTrainAgent.SaveEverySteps,
            seed=run_params.seed,
            logger=logger,
            args=run_params,
        )


if __name__ == "__main__":
    main()
