import numpy as np
import torch
import pickle
from collections import defaultdict

from sub_models.constants import DEVICE, DTYPE_16


class ReplayBuffer:
    def __init__(
        self,
        num_envs,
        obs_shape,
        # agent_goal_shape: int,
        # agent_skill_shape: tuple,
        max_length=int(1e6),
        warmup_length=1024,
        store_on_gpu=False,
    ):

        self.store_on_gpu = store_on_gpu
        self.flag_goal_skill = False

        self.entities = ["obs", "action", "reward", "termination"]
        if self.flag_goal_skill:
            self.entities += ["goal", "skill"]
        # buffer that holds the all the data
        self.buffer = defaultdict(list)
        # initiate the buffer as empty list
        for entity in self.entities:
            if self.store_on_gpu:
                if entity == "obs":
                    self.buffer[entity] = torch.empty(
                        (max_length // num_envs, num_envs, *obs_shape),
                        dtype=torch.float32 if DEVICE.type == "mps" else torch.uint8,
                        device=DEVICE,
                        requires_grad=False,
                    )
                # elif entity == "goal":
                #     self.buffer[entity] = torch.empty(
                #         (max_length // num_envs, num_envs, agent_goal_shape),
                #         dtype=torch.float32,
                #         device=DEVICE,
                #         requires_grad=False,
                #     )
                # elif entity == "skill":
                #     self.buffer[entity] = torch.empty(
                #         (max_length // num_envs, num_envs, *agent_skill_shape),
                #         dtype=torch.float32,
                #         device=DEVICE,
                #         requires_grad=False,
                #     )
                else:
                    self.buffer[entity] = torch.empty(
                        (max_length // num_envs, num_envs),
                        dtype=torch.float32,
                        device=DEVICE,
                        requires_grad=False,
                    )
            else:
                if entity == "obs":
                    self.buffer[entity] = np.empty(
                        (max_length // num_envs, num_envs, *obs_shape),
                        dtype=np.uint8,
                    )
                # elif entity == "goal":
                #     self.buffer[entity] = np.empty(
                #         (max_length // num_envs, num_envs, agent_goal_shape),
                #         dtype=np.float32,
                #     )
                # elif entity == "skill":
                #     self.buffer[entity] = np.empty(
                #         (max_length // num_envs, num_envs, *agent_skill_shape),
                #         dtype=np.float32,
                #     )
                else:
                    self.buffer[entity] = np.empty(
                        (max_length // num_envs, num_envs), dtype=np.float32
                    )

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    @property
    def ready(self):
        return bool(self.length * self.num_envs > self.warmup_length)

    def append(self, obs, action, reward, termination, goal=None, skill=None):
        """
        Append raw data to the replay buffer and increase the length by 1.
        The last pointer also increases by 1.
        """
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % (self.max_length // self.num_envs)
        if self.store_on_gpu:
            self.buffer["obs"][self.last_pointer] = torch.from_numpy(obs)
            self.buffer["action"][self.last_pointer] = torch.from_numpy(action)
            self.buffer["reward"][self.last_pointer] = torch.from_numpy(reward)
            self.buffer["termination"][self.last_pointer] = torch.from_numpy(
                termination
            )
            # if goal is not None:
            #     self.buffer["goal"][self.last_pointer] = goal  # already a tensor
            # if skill is not None:
            #     self.buffer["skill"][self.last_pointer] = skill  # already a tensor
        else:
            self.buffer["obs"][self.last_pointer] = obs
            self.buffer["action"][self.last_pointer] = action
            self.buffer["reward"][self.last_pointer] = reward
            self.buffer["termination"][self.last_pointer] = termination
            # if goal is not None:
            #     self.buffer["goal"][self.last_pointer] = goal
            # if skill is not None:
            #     self.buffer["skill"][self.last_pointer] = skill

        if len(self) < self.max_length:
            self.length += 1

    def stack(self, input) -> callable:
        """
        Return the stack function based on the storage device.
        """
        if self.store_on_gpu:
            return torch.stack(input)
        else:
            return np.stack(input)

    def sample_external(self, batch_size, batch_length) -> dict:
        """
        Sample a batch of data from the external buffer.
        """
        indexes = np.random.randint(
            0, self.external_buffer_length + 1 - batch_length, size=batch_size
        )
        data = defaultdict(list)
        for entity in self.entities:
            data[entity] = self.stack(
                [
                    self.external_buffer[entity][idx : idx + batch_length]
                    for idx in indexes
                ]
            )

        return data

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length):
        """
        Sample a batch of data from the replay buffer and put it on the DEVICE.
        """

        external_buffer = None
        samples = defaultdict(list)
        # If external buffer is available, load it
        if self.external_buffer_length is not None and external_batch_size > 0:
            # If external buffer is available, sample from it
            external_buffer = self.sample_external(external_batch_size, batch_length)
        # iterate over the entities like obs, action, reward, done, goal, skill
        for entity in self.entities:
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(
                        0,
                        self.length + 1 - batch_length,
                        size=batch_size // self.num_envs,
                    )
                    samples[entity].append(
                        self.stack(
                            [
                                self.buffer[entity][idx : idx + batch_length, i]
                                for idx in indexes
                            ]
                        )
                    )
            if external_buffer is not None:
                samples[entity].append(external_buffer[entity])

            # Concat the array/stack of samples along the batch dimension
            if entity == "obs":
                if self.store_on_gpu:
                    samples[entity] = torch.cat(samples[entity], dim=0).float() / 255
                else:
                    samples[entity] = (
                        torch.from_numpy(np.concatenate(samples[entity], axis=0))
                        .to(
                            DEVICE,
                            dtype=(
                                torch.float32 if DEVICE.type == "mps" else torch.uint8
                            ),
                        )
                        .div_(255)
                    )

                # [B, T, H, W, C] -> [B, T, C, H, W]
                samples[entity] = samples[entity].permute(0, 1, 4, 2, 3).contiguous()

            else:  # action, reward, termination, goal, skill
                if self.store_on_gpu:
                    samples[entity] = torch.cat(samples[entity], dim=0)
                else:
                    samples[entity] = torch.from_numpy(
                        np.concatenate(samples[entity], axis=0)
                    ).to(DEVICE)

        return samples

    def __len__(self):
        return self.length * self.num_envs

    def load_trajectory(self, path):
        buffer = pickle.load(open(path, "rb"))
        if self.store_on_gpu:
            self.external_buffer = {
                name: torch.from_numpy(buffer[name]).to(DEVICE) for name in buffer
            }
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.external_buffer["obs"].shape[0]
