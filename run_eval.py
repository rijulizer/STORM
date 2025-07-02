from tqdm import tqdm
import copy
import torch
import colorama
import glob
import os

from utils import seed_np_torch, Logger, load_config
from train import (
    build_single_env,
    build_vec_env,
    build_world_model,
    build_agent,
)
from eval import eval_episodes
from sub_models.constants import DEVICE

# ignore warnings
import warnings

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class RunParams:
    def __init__(self, env_names, run_name: str):
        self.run_name = run_name
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
            + "run_name: "
            + colorama.Style.RESET_ALL
            + self.run_name
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
        # "ALE/MsPacman-v5",
        "MiniGrid-Empty-8x8-v0",
        # "MiniGrid-SimpleCrossingS9N1-v0",
        # "MiniGrid-FourRooms-v0",
        # "MiniGrid-MemoryS11-v0",
        ## "MiniGrid-RedBlueDoors-6x6-v0",
        # "MiniGrid-Empty-Random-6x6-v0",
    ]
    run_params = RunParams(env_names, run_name="EmptyFullObs-Baseline_v1")
    # set seed
    seed_np_torch(seed=run_params.seed)

    # build and load model/agent
    dummy_env = build_single_env(
        run_params.env_names[0], run_params.conf.BasicSettings.ImageSize
    )
    action_dim = dummy_env.action_space.n
    world_model = build_world_model(run_params.conf, action_dim)
    agent = build_agent(run_params.conf, action_dim)
    root_path = f"ckpt/{run_params.run_name}"

    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[-1:]
    print(steps)
    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # # eval
        episode_avg_return = eval_episodes(
            num_episode=20,
            env_names=run_params.env_names,
            env_observablity=run_params.conf.BasicSettings.EnvObservability,
            max_steps=run_params.conf.JointTrainAgent.SampleMaxSteps,
            image_size=run_params.conf.BasicSettings.ImageSize,
            world_model=world_model,
            agent=agent,
            imagine_batch_length=run_params.conf.JointTrainAgent.ImagineBatchLength,
        )
        results.append([step, episode_avg_return])
    os.makedirs("./eval_result", exist_ok=True)
    with open(f"eval_result/{run_params.run_name}.csv", "w") as fout:
        fout.write("step, episode_avg_return\n")
        for step, episode_avg_return in results:
            fout.write(f"{step},{episode_avg_return}\n")


if __name__ == "__main__":
    main()
