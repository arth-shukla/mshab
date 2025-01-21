# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import itertools
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import h5py
from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm

import gymnasium as gym

import numpy as np
import torch
import torch.nn.functional as F

# ManiSkill specific imports
import mani_skill.envs
from mani_skill import ASSET_DIR
from mani_skill.utils import common

from mshab.agents.bc import AgentMultiTask
from mshab.envs.multi_task import (
    AllSubtaskEnvConfig,
    AllSubtaskTrain,
    PerSubtaskEnvConfig,
)
from mshab.envs.wrappers.vector import (
    VectorObservationWrapper,
    VectorRecordEpisodeStatistics,
)
from mshab.utils.array import recursive_slice, to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader, ClosableDataset
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


ENV_KEY_TO_ONE_HOT_IDX = dict(pick=0, place=1, open=2, close=3)
ENV_KEY_TO_ONE_HOT_TENSOR = dict(
    (
        k,
        torch.tensor([0] * v + [1] + [0] * (max(ENV_KEY_TO_ONE_HOT_IDX.values()) - v)),
    )
    for k, v in ENV_KEY_TO_ONE_HOT_IDX.items()
)


@dataclass
class BCConfig:
    name: str = "bc"

    # Training
    lr: float = 3e-4
    """learning rate"""
    batch_size: int = 2048
    """batch size"""

    # Running
    epochs: int = 100
    """num epochs to run"""
    eval_freq: int = 1
    """evaluation frequency in terms of epochs"""
    log_freq: int = 1
    """log frequency in terms of epochs"""
    save_freq: int = 1
    """save frequency in terms of epochs"""
    save_backup_ckpts: bool = False
    """whether to save separate ckpts eace save_freq which are not overwritten"""

    # Dataset
    data_dir_fp: str = None
    """path to data dir containing data .h5 files"""
    max_image_cache_size: Union[int, Literal["all"]] = 0
    """max num data points to cache in cpu memory"""
    trajs_per_obj: Union[str, int] = "all"
    """num trajectories to use per object"""

    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    def _additional_processing(self):
        assert self.name == "bc", "Wrong algo config"

        try:
            self.trajs_per_obj = int(self.trajs_per_obj)
        except:
            pass
        assert isinstance(self.trajs_per_obj, int) or self.trajs_per_obj == "all"


@dataclass
class TrainConfig:
    seed: int
    algo: BCConfig
    logger: LoggerConfig

    wandb_id: Optional[str] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    def __post_init__(self):
        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))


def recursive_h5py_to_numpy(h5py_obs, slice=None):
    if isinstance(h5py_obs, h5py.Group) or isinstance(h5py_obs, dict):
        return dict(
            (k, recursive_h5py_to_numpy(h5py_obs[k], slice)) for k in h5py_obs.keys()
        )
    if isinstance(h5py_obs, list):
        return [recursive_h5py_to_numpy(x, slice) for x in h5py_obs]
    if isinstance(h5py_obs, tuple):
        return tuple(recursive_h5py_to_numpy(x, slice) for x in h5py_obs)
    if slice is not None:
        return h5py_obs[slice]
    return h5py_obs[:]


class ConcatOneHotWrapper(VectorObservationWrapper):
    @cached_property
    def single_observation_space(self):
        assert isinstance(self.env.single_observation_space, gym.spaces.Dict)
        spaces = self.env.single_observation_space.spaces.copy()

        # Combine state and subtask_one_hot spaces
        assert "state" in spaces and "subtask_one_hot" in spaces
        state_space = spaces["state"]
        subtask_space = spaces.pop("subtask_one_hot")

        # Create new state space with combined shape
        spaces["state"] = gym.spaces.Box(
            low=np.concatenate([state_space.low, subtask_space.low]),
            high=np.concatenate([state_space.high, subtask_space.high]),
            dtype=state_space.dtype,
        )

        return gym.spaces.Dict(spaces)

    @cached_property
    def observation_space(self):
        assert isinstance(self.env.observation_space, gym.spaces.Dict)
        spaces = self.env.observation_space.spaces.copy()

        # Combine state and subtask_one_hot spaces
        assert "state" in spaces and "subtask_one_hot" in spaces
        state_space = spaces["state"]
        subtask_space = spaces.pop("subtask_one_hot")

        # Create new state space with combined shape
        spaces["state"] = gym.spaces.Box(
            low=np.concatenate([state_space.low, subtask_space.low], axis=-1),
            high=np.concatenate([state_space.high, subtask_space.high], axis=-1),
            dtype=state_space.dtype,
        )

        return gym.spaces.Dict(spaces)

    def observation(self, observation: Dict):
        assert "state" in observation
        assert "subtask_one_hot" in observation
        observation["state"] = torch.cat(
            [observation["state"], observation.pop("subtask_one_hot")], dim=1
        )
        return observation


def make_multi_task_env(
    tasks: List[str] = ["tidy_house"],
    subtasks: List[str] = ["pick", "place"],
    num_envs_per_subtask: int = 2,
    max_episode_steps: int = 200,
) -> Union[AllSubtaskTrain, ConcatOneHotWrapper, VectorRecordEpisodeStatistics]:
    all_subtask_cfg = AllSubtaskEnvConfig(
        obs_mode="rgbd",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="all",
        shader_dir="minimal",
        robot_uids="fetch",
    )

    per_subtask_cfgs = dict(
        (
            subtask,
            PerSubtaskEnvConfig(
                env_id=f"{subtask.capitalize()}SubtaskTrain-v0",
                num_envs=num_envs_per_subtask,  # Reduced for testing
                max_episode_steps=max_episode_steps,
                task_plan_fp=(
                    ASSET_DIR
                    / f"scene_datasets/replica_cad_dataset/rearrange/task_plans/{task}/{subtask}/train/all.json"
                ),
                spawn_data_fp=(
                    ASSET_DIR
                    / f"scene_datasets/replica_cad_dataset/rearrange/spawn_data/{task}/{subtask}/train/spawn_data.pt"
                ),
                continuous_task=True,
                env_kwargs=dict(
                    robot_force_mult=0.001,
                    robot_force_penalty_min=0.2,
                    target_randomization=False,
                ),
            ),
        )
        for task, subtask in itertools.product(tasks, subtasks)
    )

    # Create the combined environment
    envs = AllSubtaskTrain(
        all_subtask_env_kwargs=all_subtask_cfg,
        per_subtask_env_kwargs=per_subtask_cfgs,
        env_key_to_idx=ENV_KEY_TO_ONE_HOT_IDX,
        # keep subtask_env_wrappers as default
    )
    envs = VectorRecordEpisodeStatistics(envs, max_episode_steps=max_episode_steps)
    return envs


class DPDataset(ClosableDataset):  # Load everything into memory
    def __init__(
        self,
        data_path,
        frame_stack=3,
        trajs_per_obj="all",
        max_image_cache_size=0,
        truncate_trajectories_at_success=False,
    ):
        data_path = Path(data_path)
        if data_path.is_dir():
            h5_fps: List[Path] = []
            file_subtask_types = []
            for root, _, files in os.walk(data_path):
                root = Path(root)
                for fp in files:
                    if fp.endswith(".h5"):
                        h5_fps.append(root / fp)
                        subtask_type = root.name
                        assert subtask_type in [
                            "pick",
                            "place",
                            "open",
                            "close",
                        ], f"dataset assumes each file is in a dir named after the subtask type, but {Path(root) / fp} is not"
                        file_subtask_types.append(subtask_type)
        else:
            h5_fps = [data_path]
            subtask_type = data_path.parent.name
            assert subtask_type in [
                "pick",
                "place",
                "open",
                "close",
            ], f"dataset assumes each file is in a dir named after the subtask type, but {data_path} is not"
            file_subtask_types = [subtask_type]

        trajectories = dict(actions=[], observations=[], subtask_one_hot=[])
        num_cached = 0
        self.h5_files: List[h5py.File] = []
        for fp_num, (fp, subtask_type) in enumerate(zip(h5_fps, file_subtask_types)):
            f = h5py.File(fp, "r")
            num_uncached_this_file = 0

            if trajs_per_obj == "all":
                keys = list(f.keys())
            else:
                keys = random.sample(list(f.keys()), k=trajs_per_obj)

            for k in tqdm(keys, desc=f"hf file {fp_num}"):
                obs, act = f[k]["obs"], f[k]["actions"][:]

                # note: len(obs) == len(act) + 1 == len(success) + 1, where last obs is a
                #   "next_obs" with no associated action
                # hence, we will cut off the last obs, as it is not useful in BC
                if truncate_trajectories_at_success:
                    success: List[bool] = f[k]["success"][:].tolist()
                    success_cutoff = min(success.index(True) + 1, len(success))
                    del success
                else:
                    success_cutoff = len(act)

                # NOTE (arth): we always cache state obs and actions because they take up very little memory.
                #       mostly constraints are on images, since those take up much more memory
                state_obs_list = [
                    *recursive_h5py_to_numpy(
                        obs["agent"], slice=slice(success_cutoff + 1)
                    ).values(),
                    *recursive_h5py_to_numpy(
                        obs["extra"], slice=slice(success_cutoff + 1)
                    ).values(),
                ]
                state_obs_list = [
                    x[:, None] if len(x.shape) == 1 else x for x in state_obs_list
                ]
                state_obs = torch.from_numpy(np.concatenate(state_obs_list, axis=1))
                act = torch.from_numpy(act)[: success_cutoff + 1]

                pixel_obs = dict(
                    fetch_head_depth=obs["sensor_data"]["fetch_head"]["depth"],
                    fetch_hand_depth=obs["sensor_data"]["fetch_hand"]["depth"],
                )
                if (
                    max_image_cache_size == "all"
                    or len(act) <= max_image_cache_size - num_cached
                ):
                    pixel_obs = to_tensor(
                        recursive_h5py_to_numpy(
                            pixel_obs, slice=slice(success_cutoff + 1)
                        )
                    )
                    num_cached += len(act)
                else:
                    num_uncached_this_file += len(act)

                trajectories["actions"].append(act)
                trajectories["observations"].append(dict(state=state_obs, **pixel_obs))
                trajectories["subtask_one_hot"].append(
                    ENV_KEY_TO_ONE_HOT_TENSOR[subtask_type]
                )

            if num_uncached_this_file == 0:
                f.close()
            else:
                self.h5_files.append(f)

        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            # NOTE (arth): since we cut off data at first success, we might have extra actions available
            #   after the end of slice which we can use instead of hand-made padded zero actions
            L = trajectories["observations"][traj_idx]["state"].shape[0]
            total_transitions += L

            # note that the last obs is a "next_obs" cut off in our __init__
            # |PAD|PAD|o|o|o|...|o|    o    |     frame_stack: 3
            # |   |   |a|a|a|...|a|(lastact)|
            pad_before = frame_stack - 1
            self.slices += [
                (traj_idx, start, start + frame_stack, start + frame_stack - 1)
                for start in range(-pad_before, L - frame_stack)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end, timestep = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = dict(pixels=dict(), state=None)
        for k, v in obs_traj.items():
            if k == "state":
                obs_seq[k] = v[timestep]
            else:
                obs_seq["pixels"][k] = v[
                    max(0, start) : end
                ]  # start+self.frame_stack is at least 1
                if len(obs_seq["pixels"][k].shape) == 4:
                    obs_seq["pixels"][k] = to_tensor(obs_seq["pixels"][k]).permute(
                        0, 3, 2, 1
                    )  # FS, D, H, W
                if start < 0:  # pad before the trajectory
                    pad_obs_seq = torch.stack(
                        [obs_seq["pixels"][k][0]] * abs(start), dim=0
                    )
                    obs_seq["pixels"][k] = torch.cat(
                        (pad_obs_seq, obs_seq["pixels"][k]), dim=0
                    )
        assert "state" in obs_seq, "state must be in obs_seq"
        obs_seq["subtask_one_hot"] = self.trajectories["subtask_one_hot"][traj_idx]

        act_seq = self.trajectories["actions"][traj_idx][timestep]

        return dict(
            observations=obs_seq,
            actions=act_seq,
        )

    def __len__(self):
        return len(self.slices)

    def close(self):
        for h5_file in self.h5_files:
            h5_file.close()


def train(cfg: TrainConfig):
    # seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    # NOTE (arth): maybe require cuda since we only allow gpu sim anyways
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("making eval env")
    eval_envs = make_multi_task_env(
        tasks=["tidy_house"], subtasks=["pick", "place"], num_envs_per_subtask=63
    )
    assert isinstance(
        eval_envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    print("made")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    eval_envs.action_space.seed(cfg.seed + 1_000_000)

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------
    agent = AgentMultiTask(eval_obs, eval_envs.single_action_space.shape).to(device)
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=cfg.algo.lr,
    )

    def save(save_path):
        torch.save(
            dict(
                agent=agent.state_dict(),
                optimizer=optimizer.state_dict(),
            ),
            save_path,
        )

    def load(load_path):
        checkpoint = torch.load(str(load_path), map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=save,
    )

    if cfg.model_ckpt is not None:
        load(cfg.model_ckpt)

    # -------------------------------------------------------------------------------------------------
    # DATALOADER
    # -------------------------------------------------------------------------------------------------
    bc_dataset = DPDataset(
        cfg.algo.data_dir_fp,
        frame_stack=3,
        trajs_per_obj=cfg.algo.trajs_per_obj,
        max_image_cache_size=cfg.algo.max_image_cache_size,
        truncate_trajectories_at_success=True,
    )

    logger.print(
        f"Made BC Dataset with {len(bc_dataset)} samples at {cfg.algo.trajs_per_obj} trajectories per object",
        flush=True,
    )
    bc_dataloader = ClosableDataLoader(
        bc_dataset,
        batch_size=cfg.algo.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )

    epoch = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0

    def check_freq(freq):
        return epoch % freq == 0

    def store_env_stats(key):
        assert key == "eval", "Only eval env for BC"
        logger.store(
            key,
            return_per_step=common.to_tensor(eval_envs.return_queue, device=device)
            .float()
            .mean()
            / eval_envs.max_episode_steps,
            success_once=common.to_tensor(eval_envs.success_once_queue, device=device)
            .float()
            .mean(),
            success_at_end=common.to_tensor(
                eval_envs.success_at_end_queue, device=device
            )
            .float()
            .mean(),
            len=common.to_tensor(eval_envs.length_queue, device=device).float().mean(),
        )
        eval_envs.reset_queues()

    print("start")
    timer = NonOverlappingTimeProfiler()
    for epoch in range(cfg.algo.epochs):

        if epoch + logger_start_log_step > cfg.algo.epochs:
            break

        logger.print(
            f"Overall epoch: {epoch + logger_start_log_step}; Curr process epoch: {epoch}"
        )

        # let agent update
        logger.print("Training")
        tot_loss, n_samples = 0, 0
        pbar = tqdm(iter(bc_dataloader), total=len(bc_dataloader), desc="Training")
        for batch in pbar:
            obs, act = to_tensor(
                batch["observations"], device=device, dtype=torch.float
            ), to_tensor(batch["actions"], device=device, dtype=torch.float)

            def recursive_shape(x):
                if isinstance(x, dict):
                    return dict((k, recursive_shape(v)) for k, v in x.items())
                return x.shape

            n_samples += act.size(0)

            pi = agent(obs)

            loss = F.mse_loss(pi, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            pbar.update(act.size(0))
        loss_logs = dict(loss=tot_loss / n_samples)
        timer.end(key="train")

        # Log
        if check_freq(cfg.algo.log_freq):
            pbar.set_description("Logging")
            logger.store(tag="losses", **loss_logs)
            if epoch > 0:
                logger.store("time", **timer.get_time_logs(epoch))
            logger.log(logger_start_log_step + epoch)
            timer.end(key="log")

        # Evaluation
        if cfg.algo.eval_freq:
            pbar.set_description("Evaluating")
            if check_freq(cfg.algo.eval_freq):
                agent.eval()
                eval_obs, _ = eval_envs.reset()  # don't seed here

                for _ in range(eval_envs.max_episode_steps):
                    with torch.no_grad():
                        action = agent(eval_obs)
                    eval_obs, _, _, _, _ = eval_envs.step(action)

                if len(eval_envs.return_queue) > 0:
                    store_env_stats("eval")
                logger.log(logger_start_log_step + epoch)
                timer.end(key="eval")

        # Checkpoint
        if check_freq(cfg.algo.save_freq):
            pbar.set_description("Saving")
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{epoch}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end(key="checkpoint")

    save(logger.model_path / "final_ckpt.pt")
    save(logger.model_path / "latest.pt")

    bc_dataloader.close()
    eval_envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
