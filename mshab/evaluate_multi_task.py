# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import itertools
import random
import sys
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm

import gymnasium as gym

import numpy as np
import torch

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
    RecordVideo,
    VectorObservationWrapper,
    VectorRecordEpisodeStatistics,
)
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.profile import Profiler
from mshab.utils.time import NonOverlappingTimeProfiler


ENV_KEY_TO_ONE_HOT_IDX = dict(pick=0, place=1, open=2, close=3)
ENV_KEY_TO_ONE_HOT_TENSOR = dict(
    (
        k,
        torch.tensor([0] * v + [1] + [0] * (max(ENV_KEY_TO_ONE_HOT_IDX.values()) - v)),
    )
    for k, v in ENV_KEY_TO_ONE_HOT_IDX.items()
)
print(ENV_KEY_TO_ONE_HOT_TENSOR)


@dataclass
class EvalConfig:
    seed: int
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
                old_config = get_mshab_eval_cfg(
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

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_eval_cfg(cfg: EvalConfig) -> EvalConfig:
    return from_dict(data_class=EvalConfig, data=OmegaConf.to_container(cfg))


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
    num_envs_per_subtask: int = 63,
    max_episode_steps: int = 200,
    video_path: str = None,
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
    if video_path is not None:
        envs = RecordVideo(envs, output_dir=video_path)
    envs = VectorRecordEpisodeStatistics(envs, max_episode_steps=max_episode_steps)
    return envs


def eval(cfg: EvalConfig):
    # seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # NOTE (arth): maybe require cuda since we only allow gpu sim anyways
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("making eval env")
    eval_envs = make_multi_task_env(
        tasks=["tidy_house"],
        subtasks=["pick", "place"],
        num_envs_per_subtask=2,
        video_path=cfg.logger.eval_video_path,
    )
    assert isinstance(
        eval_envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    print("made")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    eval_envs.action_space.seed(cfg.seed + 1_000_000)

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

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------
    agent = AgentMultiTask(eval_obs, eval_envs.single_action_space.shape).to(device)
    agent.eval()
    agent.load_state_dict(torch.load(str(cfg.model_ckpt), map_location=device)["agent"])

    logger = Logger(
        logger_cfg=cfg.logger,
    )

    print("start")
    total_steps = eval_envs.num_envs * eval_envs.max_episode_steps
    profiler = Profiler()
    timer = NonOverlappingTimeProfiler()
    with profiler.profile(
        "eval_multi_task",
        total_steps=total_steps,
        num_envs=eval_envs.num_envs,
    ):
        with torch.inference_mode():
            for _ in tqdm(range(eval_envs.max_episode_steps)):
                timer.end(key="other")
                action = agent(eval_obs)
                timer.end(key="act")
                eval_obs, _, _, _, _ = eval_envs.step(action)
                timer.end(key="sim_sample")

    if len(eval_envs.return_queue) > 0:
        store_env_stats("eval")
    logger.store(
        tag="timer",
        **timer.get_time_logs(total_steps),
    )
    logger.store(
        tag="system",
        **profiler.stats["eval_multi_task"],
    )
    logger.log(0)
    logger.print_summary()

    eval_envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_eval_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    eval(cfg)
