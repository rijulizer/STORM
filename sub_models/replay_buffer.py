import numpy as np
import torch
import pickle
from collections import defaultdict

from sub_models.constants import DEVICE, DTYPE_16


class ReplayBuffer:
    def __init__(
        self,
        obs_shape,
        num_envs,
        max_length=int(1e6),
        warmup_length=1024,
        store_on_gpu=False,
    ):

        self.store_on_gpu = store_on_gpu
        self.entities = ["obs", "action", "reward", "termination"]
        # buffer that holds the all the data
        self.buffer = {}
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
                else:
                    self.buffer[entity] = np.empty(
                        (max_length // num_envs, num_envs), dtype=np.float32
                    )

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer = {}
        self.external_buffer_length = None

    @property
    def ready(self):
        return bool(self.length * self.num_envs > self.warmup_length)

    def append(self, obs, action, reward, termination):
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
        else:
            self.buffer["obs"][self.last_pointer] = obs
            self.buffer["action"][self.last_pointer] = action
            self.buffer["reward"][self.last_pointer] = reward
            self.buffer["termination"][self.last_pointer] = termination

        if len(self) < self.max_length:
            self.length += 1

    def stack(self, input_list):
        """
        Return the stack function based on the storage device.
        """
        if self.store_on_gpu:
            return torch.stack(input_list)
        else:
            return np.stack(input_list)

    def sample_external(self, batch_size, batch_length) -> dict:
        """
        Sample a batch of data from the external buffer.
        """
        indexes = np.random.randint(
            0, self.external_buffer_length + 1 - batch_length, size=batch_size
        )
        data = {}
        for entity in self.entities:
            # Ensure consistency with key names, if external buffer uses 'done'
            # it should be handled in load_trajectory or here.
            # Assuming external buffer's termination key is consistent with self.entities
            # or is renamed in load_trajectory.
            key = (
                "done"
                if entity == "termination"
                and "done" in self.external_buffer
                and "termination" not in self.external_buffer
                else entity
            )
            data[entity] = self.stack(
                [self.external_buffer[key][idx : idx + batch_length] for idx in indexes]
            )

        return data

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length):
        """
        Sample a batch of data from the replay buffer and put it on the DEVICE.
        """

        # --- Sampling from internal buffer ---
        samples = defaultdict(list)
        if batch_size > 0:
            for i in range(self.num_envs):
                # for each environment, the indexes are randomly sampled and same for all entities
                indexes_for_env = np.random.randint(
                    0,
                    self.length + 1 - batch_length,
                    size=batch_size // self.num_envs,
                )
                # iterate over the entities: obs, action, reward, done, goal, skill
                for entity in self.entities:
                    samples[entity].append(
                        self.stack(
                            [
                                self.buffer[entity][idx : idx + batch_length, i]
                                for idx in indexes_for_env
                            ]
                        )
                    )
        # --- Sampling from external buffer ---
        if self.external_buffer_length is not None and external_batch_size > 0:
            external_data = self.sample_external(external_batch_size, batch_length)
            if external_data is not None:
                for entity in self.entities:
                    samples[entity].append(external_data[entity])

        # --- Concatenation and Post-processing ---
        for entity in self.entities:
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
        if "done" in buffer and "termination" not in buffer:
            buffer["termination"] = buffer.pop("done")
        if self.store_on_gpu:
            self.external_buffer = {
                name: torch.from_numpy(buffer[name]).to(DEVICE) for name in buffer
            }
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.external_buffer["obs"].shape[0]
