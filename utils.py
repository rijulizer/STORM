import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
import time
import yacs
from yacs.config import CfgNode as CN
import cv2


def seed_np_torch(seed=20010105):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar:
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False
    conf.BasicSettings.EnvObservability = "Full"  # Full/Partial

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.LogVideoSteps = 0
    conf.JointTrainAgent.UseDemonstration = False

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf


ACTION_MAP = {
    0: "Turn Left",
    1: "Turn Right",
    2: "Move Forward",
    3: "Pickup",
    4: "Drop",
    5: "Toggle",
    6: "Done",
    # Add more actions if you have them
}


def create_obs_action_frame(obs_frame, action_text):
    # Transpose from (C, H, W) to (H, W, C) for OpenCV
    obs_frame = obs_frame.transpose(1, 2, 0)
    # Add the action text to the frame using OpenCV
    font = cv2.FONT_HERSHEY_PLAIN
    position = (2, 4)  # Top-left corner
    font_scale = 0.4
    font_color = (0, 100, 155)  # White
    line_type = 4
    thickness = 1

    # Make a writable copy of the frame before drawing text
    obs_frame = np.ascontiguousarray(obs_frame)
    cv2.putText(
        obs_frame,
        f"Action: {action_text}",
        position,
        font,
        font_scale,
        font_color,
        thickness,
        line_type,
    )
    # Convert back to (C, H, W) for torchvision
    processed_frame = torch.from_numpy(obs_frame).permute(2, 0, 1)
    return processed_frame


def log_processed_video(observations, actions):
    # Process each batch item separately
    batched_video = []
    B, T = observations.shape[:2]
    for i in range(B):
        video_frames = []
        for t in range(T):
            # Get the frame and corresponding action
            frame = observations[i, t]
            action_text = str(ACTION_MAP[actions[i, t].item()])

            # process the frame with action text
            processed_frame = create_obs_action_frame(frame, action_text)

            video_frames.append(processed_frame)
        batched_video.append(video_frames)
    # Stack the frames for each batch item
    video = torch.stack([torch.stack(frames) for frames in batched_video])
    return video


def linear_decay(initial, final, step, total_decay_steps):
    ratio = min(step / total_decay_steps, 1.0)
    return initial * (1 - ratio) + final * ratio
