# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

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
from mani_skill.utils import common

from mshab.agents.bc import Agent
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataclasses import default_field
from mshab.utils.dataset import ClosableDataLoader, ClosableDataset
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


@dataclass
class BCConfig:
    name: str = "bc"

    # Training
    lr: float = 3e-4
    """learning rate"""
    batch_size: int = 512
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
    max_cache_size: int = 0
    """max num data points to cache in cpu memory"""
    trajs_per_obj: Union[str, int] = "all"
    """num trajectories to use per object"""

    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    # passed from env/eval_env cfg
    num_eval_envs: int = field(init=False)
    """the number of parallel environments"""

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
    eval_env: EnvConfig
    algo: BCConfig
    logger: LoggerConfig

    wandb_id: Optional[str] = None
    resume_logdir: Optional[Union[Path, str]] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    def __post_init__(self):
        assert (
            self.resume_logdir is None or not self.logger.clear_out
        ), "Can't resume to a cleared out logdir!"

        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "if setting resume_logdir, must set logger workspace and exp_name accordingly"
            else:
                assert (
                    old_config_path.exists()
                ), f"Couldn't find old config at path {old_config_path}"
                old_config = get_mshab_train_cfg(
                    parse_cfg(default_cfg_path=old_config_path)
                )
                self.logger.workspace = old_config.logger.workspace
                self.logger.exp_path = old_config.logger.exp_path
                self.logger.log_path = old_config.logger.log_path
                self.logger.model_path = old_config.logger.model_path
                self.logger.train_video_path = old_config.logger.train_video_path
                self.logger.eval_video_path = old_config.logger.eval_video_path

            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))


# NOTE (arth): we assume any (leaf) list entries or dict values are tensors
#   this implementation would be wrong if, for example, some values were ints
def recursive_tensor_size_bytes(obj):
    extra_obj_size = 0
    if isinstance(obj, dict):
        extra_obj_size = sum([recursive_tensor_size_bytes(v) for v in obj.values()])
    elif isinstance(obj, list) or isinstance(obj, tuple):
        extra_obj_size = sum([recursive_tensor_size_bytes(x) for x in obj])
    elif isinstance(obj, torch.Tensor):
        extra_obj_size = obj.nelement() * obj.element_size()
    return sys.getsizeof(obj) + extra_obj_size


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


class DPDataset(ClosableDataset):  # Load everything into memory
    def __init__(
        self,
        data_path,
        obs_horizon,
        pred_horizon,
        control_mode,
        trajs_per_obj="all",
        max_image_cache_size=0,
        truncate_trajectories_at_success=False,
    ):
        data_path = Path(data_path)
        if data_path.is_dir():
            h5_fps = [
                data_path / fp for fp in os.listdir(data_path) if fp.endswith(".h5")
            ]
        else:
            h5_fps = [data_path]

        trajectories = dict(actions=[], observations=[])
        num_cached = 0
        self.h5_files = []
        for fp_num, fp in enumerate(h5_fps):
            f = h5py.File(fp, "r")
            num_uncached_this_file = 0

            if trajs_per_obj == "all":
                keys = list(f.keys())
            else:
                keys = random.sample(list(f.keys()), k=trajs_per_obj)

            for k in tqdm(keys, desc=f"hf file {fp_num}"):
                obs, act = f[k]["obs"], f[k]["actions"][:]

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
                # don't cut off actions in case we are able to use in place of padding
                act = torch.from_numpy(act)

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

            if num_uncached_this_file == 0:
                f.close()
            else:
                self.h5_files.append(f)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in control_mode
            or control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,)
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            raise NotImplementedError(f"Control Mode {control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            obs_horizon,
            pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            # NOTE (arth): since we cut off data at first success, we might have extra actions available
            #   after the end of slice which we can use instead of hand-made padded zero actions
            L = trajectories["observations"][traj_idx]["state"].shape[0] - 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if len(obs_seq[k].shape) == 4:
                obs_seq[k] = to_tensor(obs_seq[k]).permute(0, 3, 2, 1)  # FS, D, H, W
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

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
    eval_envs = make_env(
        cfg.eval_env,
        video_path=cfg.logger.eval_video_path,
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
    agent = Agent(eval_obs, eval_envs.unwrapped.single_action_space.shape).to(device)
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
    bc_dataset = BCDataset(
        cfg.algo.data_dir_fp,
        cfg.algo.max_cache_size,
        cat_state=cfg.eval_env.cat_state,
        cat_pixels=cfg.eval_env.cat_pixels,
        trajs_per_obj=cfg.algo.trajs_per_obj,
    )
    logger.print(
        f"Made BC Dataset with {len(bc_dataset)} samples at {cfg.algo.trajs_per_obj} trajectories per object for {len(bc_dataset.obj_names_in_loaded_order)} objects",
        flush=True,
    )
    bc_dataloader = ClosableDataLoader(
        bc_dataset,
        batch_size=cfg.algo.batch_size,
        shuffle=True,
        num_workers=2,
    )

    epoch = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0

    def check_freq(freq):
        return epoch % freq == 0

    def store_env_stats(key):
        assert key == "eval", "Only eval env for BC"
        log_env = eval_envs
        logger.store(
            key,
            return_per_step=common.to_tensor(log_env.return_queue, device=device)
            .float()
            .mean()
            / log_env.max_episode_steps,
            success_once=common.to_tensor(log_env.success_once_queue, device=device)
            .float()
            .mean(),
            success_at_end=common.to_tensor(log_env.success_at_end_queue, device=device)
            .float()
            .mean(),
            len=common.to_tensor(log_env.length_queue, device=device).float().mean(),
        )
        log_env.reset_queues()

    print("start")
    timer = NonOverlappingTimeProfiler()
    for epoch in range(cfg.algo.epochs):

        if epoch + logger_start_log_step > cfg.algo.epochs:
            break

        logger.print(
            f"Overall epoch: {epoch + logger_start_log_step}; Curr process epoch: {epoch}"
        )

        # let agent update
        tot_loss, n_samples = 0, 0
        for obs, act in iter(bc_dataloader):
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(
                act, device=device, dtype="float"
            )

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
        loss_logs = dict(loss=tot_loss / n_samples)
        timer.end(key="train")

        # Log
        if check_freq(cfg.algo.log_freq):
            logger.store(tag="losses", **loss_logs)
            if epoch > 0:
                logger.store("time", **timer.get_time_logs(epoch))
            logger.log(logger_start_log_step + epoch)
            timer.end(key="log")

        # Evaluation
        if cfg.algo.eval_freq:
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
