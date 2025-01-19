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


class BCDataset(ClosableDataset):
    def __init__(
        self,
        data_dir_fp: str,
        max_cache_size: int,
        transform_fn=torch.from_numpy,
        trajs_per_obj: Union[str, int] = "all",
        cat_state=True,
        cat_pixels=False,
    ):
        data_dir_fp: Path = Path(data_dir_fp)
        self.data_files: List[h5py.File] = []
        self.json_files: List[Dict] = []
        self.obj_names_in_loaded_order: List[str] = []

        if data_dir_fp.is_file():
            data_file_names = [data_dir_fp.name]
            data_dir_fp = data_dir_fp.parent
        else:
            data_file_names = os.listdir(data_dir_fp)
        for data_fn in data_file_names:
            if data_fn.endswith(".h5"):
                json_fn = data_fn.replace(".h5", ".json")
                self.data_files.append(h5py.File(data_dir_fp / data_fn, "r"))
                with open(data_dir_fp / json_fn, "rb") as f:
                    self.json_files.append(json.load(f))
                self.obj_names_in_loaded_order.append(data_fn.replace(".h5", ""))

        self.dataset_idx_to_data_idx = dict()
        dataset_idx = 0
        # NOTE (arth): for the rearrange dataset, each json/h5 file contains trajectories for one object
        for file_idx, json_file in enumerate(self.json_files):

            # sample trajectories to use by trajs_per_obj
            if trajs_per_obj == "all":
                use_ep_jsons = json_file["episodes"]
            else:
                assert trajs_per_obj <= len(
                    json_file["episodes"]
                ), f"got {trajs_per_obj=} but only have {len(json_file['episodes'])} for data for obj={self.obj_names_in_loaded_order[file_idx]}"
                use_ep_jsons = random.sample(json_file["episodes"], k=trajs_per_obj)

            for ep_json in use_ep_jsons:
                ep_id = ep_json["episode_id"]
                for step in range(ep_json["elapsed_steps"]):
                    self.dataset_idx_to_data_idx[dataset_idx] = (file_idx, ep_id, step)
                    dataset_idx += 1
        self._data_len = dataset_idx

        self.max_cache_size = max_cache_size
        self.cache = dict()

        self.transform_fn = transform_fn
        self.cat_state = cat_state
        self.cat_pixels = cat_pixels

    def transform_idx(self, x, data_index):
        if isinstance(x, h5py.Group) or isinstance(x, dict):
            return dict((k, self.transform_idx(v, data_index)) for k, v in x.items())
        out = self.transform_fn(np.array(x[data_index]))
        if len(out.shape) == 0:
            out = out.unsqueeze(0)
        return out

    def get_single_item(self, index):
        if index in self.cache:
            return self.cache[index]

        file_num, ep_num, step_num = self.dataset_idx_to_data_idx[index]
        ep_data = self.data_files[file_num][f"traj_{ep_num}"]

        observation = ep_data["obs"]
        agent_obs = self.transform_idx(observation["agent"], step_num)
        extra_obs = self.transform_idx(observation["extra"], step_num)
        # unsqueeze to emulate a single frame stack
        fetch_head_depth = (
            self.transform_idx(
                observation["sensor_data"]["fetch_head"]["depth"], step_num
            )
            .squeeze(-1)
            .unsqueeze(0)
        )
        fetch_hand_depth = (
            self.transform_idx(
                observation["sensor_data"]["fetch_hand"]["depth"], step_num
            )
            .squeeze(-1)
            .unsqueeze(0)
        )

        # NOTE (arth): this works for seq task envs, but may not work for generic env obs
        state_obs = (
            dict(
                state=torch.cat(
                    [
                        *agent_obs.values(),
                        *extra_obs.values(),
                    ],
                    axis=0,
                )
            )
            if self.cat_state
            else dict(agent_obs=agent_obs, extra_obs=extra_obs)
        )
        pixel_obs = (
            dict(all_depth=torch.stack([fetch_head_depth, fetch_hand_depth], axis=-3))
            if self.cat_pixels
            else dict(
                fetch_head_depth=fetch_head_depth,
                fetch_hand_depth=fetch_hand_depth,
            )
        )

        obs = dict(**state_obs, pixels=pixel_obs)
        act = self.transform_idx(ep_data["actions"], step_num)

        res = (obs, act)
        if len(self.cache) < self.max_cache_size:
            self.cache[index] = res
        return res

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            return self.get_single_item(indexes)
        return [self.get_single_item(i) for i in indexes]

    def __len__(self):
        return self._data_len

    def close(self):
        for f in self.data_files:
            f.close()


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
