import itertools
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym

import numpy as np
import torch

from mani_skill.utils import common
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from mshab.envs.planner import plan_data_from_file
from mshab.envs.wrappers.action import FetchActionWrapper
from mshab.envs.wrappers.observation import FetchDepthObservationWrapper, FrameStack
from mshab.utils.dataclasses import recursive_asdict


@dataclass
class AllSubtaskEnvConfig:
    # NOTE (arth): should be constant across subtasks for consistent obs and act space
    #   technically, we could support these by splitting obs/acts via a dict using keys
    #   from per_subtask_env_kwargs, but the goal rn is to make a "seamless" combined env
    obs_mode: str = "rgbd"
    control_mode: str = "pd_joint_delta_pos"

    # NOTE (arth): technically, these can be different across subtask envs (e.g. if you want to
    #   make the same env but with different render modes to the policy can learn with diff images).
    #   as there are reasonable use cases, and these don't "break" the env, we'll allow these to be
    #   changed in PerSubtaskEnvConfig
    continuous_task: bool = True
    reward_mode: str = "normalized_dense"
    render_mode: str = "all"
    shader_dir: str = "minimal"
    robot_uids: str = "fetch"


@dataclass
class PerSubtaskEnvConfig:
    # Required fields
    env_id: str
    num_envs: int
    max_episode_steps: int
    task_plan_fp: Union[str, Path]
    spawn_data_fp: Union[str, Path]

    # NOTE (arth): see above for why only these are allowed to change "by default"
    continuous_task: bool = True
    reward_mode: str = "normalized_dense"
    render_mode: str = "all"
    shader_dir: str = "minimal"
    robot_uids: str = "fetch"

    # NOTE (arth): some subtask envs have unique kwargs (e.g. randomly open articulations
    #   to make RL training possible). users should avoid passing the above kwargs in env_kwargs
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


ObsType = Union[Dict[str, torch.Tensor], torch.Tensor]
InfoType = Dict[str, torch.Tensor]


class AllSubtaskTrain(gym.vector.VectorEnv):
    def __init__(
        self,
        all_subtask_env_kwargs: AllSubtaskEnvConfig,
        per_subtask_env_kwargs: Dict[str, PerSubtaskEnvConfig],
        env_key_to_idx: Dict[str, int] = dict(pick=0, place=1, open=2, close=3),
        subtask_env_wrappers: List[Callable[[], gym.Env]] = [
            partial(FetchDepthObservationWrapper, cat_state=True, cat_pixels=False),
            partial(
                FrameStack,
                num_stack=3,
                stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
            ),
            partial(
                FetchActionWrapper,
                stationary_base=False,
                stationary_torso=False,
                stationary_head=True,
            ),
        ],
    ):
        # Create individual environments
        self.envs: List[ManiSkillVectorEnv] = []
        self.env_keys: List[str] = []
        self.env_slices: List[slice] = []

        start_idx = 0

        # Helper to create envs
        def create_env(cfg: PerSubtaskEnvConfig):
            # redo each create to avoid mutating in different creates
            all_kwargs_dict: Dict = recursive_asdict(all_subtask_env_kwargs)
            curr_kwargs_dict: Dict = recursive_asdict(cfg)

            # all kwargs dict overwrites per-subtask kwargs dict
            common.dict_merge(curr_kwargs_dict, all_kwargs_dict)

            plan_data = plan_data_from_file(curr_kwargs_dict.pop("task_plan_fp"))
            spawn_data_fp = curr_kwargs_dict.pop("spawn_data_fp")

            env_id = curr_kwargs_dict.pop("env_id")
            max_episode_steps = curr_kwargs_dict.pop("max_episode_steps")
            continuous_task = curr_kwargs_dict.pop("continuous_task")
            additional_kwargs = curr_kwargs_dict.pop("env_kwargs")

            env = gym.make(
                env_id,
                **curr_kwargs_dict,
                **additional_kwargs,
                task_plans=plan_data.plans,
                scene_builder_cls=plan_data.dataset,
                spawn_data_fp=spawn_data_fp,
            )
            for wrapper_cls in subtask_env_wrappers:
                env = wrapper_cls(env)
            # NOTE (arth): while it is a bit odd to have a VectorEnv managing other VectorEnvs, the
            #   ManiSkillVectorEnv wrapper provides necessary timeout and partial reset functionality
            env = ManiSkillVectorEnv(
                env,
                max_episode_steps=max_episode_steps,
                ignore_terminations=continuous_task,
            )
            return env, slice(start_idx, start_idx + env.num_envs)

        # Create each environment based on provided kwargs
        for env_key, per_subtask_env_cfg in per_subtask_env_kwargs.items():

            env, slice_idx = create_env(per_subtask_env_cfg)
            if env is not None:
                self.envs.append(env)
                self.env_keys.append(env_key)
                self.env_slices.append(slice_idx)
                start_idx += env.num_envs

        if not self.envs:
            raise ValueError("No environments were created")

        # Store env_key_to_idx and create one-hot encodings
        self.env_key_to_idx = env_key_to_idx
        self.one_hot_size = max(env_key_to_idx.values()) + 1
        self.env_one_hots = {}
        for env_key, idx in env_key_to_idx.items():
            one_hot = torch.zeros(
                self.one_hot_size, dtype=torch.float, device=self.device
            )
            one_hot[idx] = 1.0
            self.env_one_hots[env_key] = one_hot
        print(self.env_one_hots)

        # Get observation and action spaces from first env
        self.single_observation_space = self.envs[0].single_observation_space
        self.single_action_space = self.envs[0].single_action_space

        # Modify observation space to include one-hot encoding
        if isinstance(self.single_observation_space, gym.spaces.Dict):
            spaces = self.single_observation_space.spaces
            spaces["subtask_one_hot"] = gym.spaces.Box(
                low=0, high=1, shape=(self.one_hot_size,), dtype=np.float32
            )
            self.single_observation_space = gym.spaces.Dict(spaces)
        else:
            old_space: gym.spaces.Box = self.single_observation_space
            low = np.concatenate([old_space.low, np.zeros(self.one_hot_size)], axis=0)
            high = np.concatenate([old_space.high, np.ones(self.one_hot_size)], axis=0)

            new_shape = old_space.shape
            new_shape[0] += self.one_hot_size

            self.single_observation_space = gym.spaces.Box(
                low=low,
                high=high,
                shape=new_shape,
                dtype=old_space.dtype,
            )

        super().__init__(
            num_envs=sum(env.num_envs for env in self.envs),
            observation_space=self.single_observation_space,
            action_space=self.single_action_space,
        )

    def _concat_outputs(
        self, outputs: List[Union[Dict, torch.Tensor]], dim: int = 0
    ) -> Union[Dict, torch.Tensor]:
        """Recursively concatenate outputs from multiple environments.

        Args:
            outputs: List of outputs from each environment (can be tensors or nested dicts)
            dim: Dimension along which to concatenate tensors

        Returns:
            Combined output with the same structure as inputs but concatenated along dim
        """
        # Base case: tensor outputs
        if all(isinstance(x, torch.Tensor) for x in outputs):
            return torch.cat(outputs, dim=dim)

        # Base case: non-tensor, non-dict outputs (e.g., lists, primitives)
        if not any(isinstance(x, (torch.Tensor, dict)) for x in outputs):
            return list(outputs)

        # Recursive case: dictionary outputs
        if all(isinstance(x, dict) for x in outputs):
            combined = {}
            all_keys = set().union(*[x.keys() for x in outputs])

            for key in all_keys:
                values = [x[key] for x in outputs if key in x]
                combined[key] = self._concat_outputs(values, dim=dim)
            return combined

        raise ValueError(
            "Mixed types in outputs - all items must be either tensors or dicts or non-tensor+non-dicts"
        )

    def _add_one_hot(self, obs: ObsType, env_key: str, num_envs: int) -> ObsType:
        """Add one-hot encoding to observation."""
        one_hot = self.env_one_hots[env_key].repeat(num_envs, 1)

        if isinstance(obs, dict):
            obs["subtask_one_hot"] = one_hot
            return obs
        else:
            return torch.cat([obs, one_hot], dim=1)

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Dict[str, Any] = dict(),
    ) -> Tuple[ObsType, Dict[str, torch.Tensor]]:
        outputs = []

        for env, env_key, slice_idx in zip(self.envs, self.env_keys, self.env_slices):
            obs, info = env.reset(
                seed=seed if seed is None or isinstance(seed, int) else seed[slice_idx],
                options=options,
            )
            # Add one-hot encoding to observation
            obs = self._add_one_hot(obs, env_key, env.num_envs)
            outputs.append((obs, info))

        all_obs = self._concat_outputs([x[0] for x in outputs])
        all_info = self._concat_outputs([x[1] for x in outputs])

        return all_obs, all_info

    def step(self, actions: torch.Tensor) -> Tuple[
        ObsType,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        InfoType,
    ]:
        outputs = []

        for env, env_key, slice_idx in zip(self.envs, self.env_keys, self.env_slices):
            step_outputs = env.step(actions[slice_idx])
            # Add one-hot encoding to observation
            step_outputs = list(step_outputs)
            step_outputs[0] = self._add_one_hot(step_outputs[0], env_key, env.num_envs)
            outputs.append(tuple(step_outputs))

        # Unzip the outputs
        all_obs = self._concat_outputs([x[0] for x in outputs])
        all_rewards = self._concat_outputs([x[1] for x in outputs])
        all_terminateds = self._concat_outputs([x[2] for x in outputs])
        all_truncateds = self._concat_outputs([x[3] for x in outputs])
        all_infos = self._concat_outputs([x[4] for x in outputs])

        return all_obs, all_rewards, all_terminateds, all_truncateds, all_infos

    def render(self):
        if self.render_mode == "human":
            self.envs[0].render()
            return
        return torch.cat([env.render() for env in self.envs], dim=0)

    def close(self):
        for env in self.envs:
            env.close()

    @property
    def device(self):
        return self.envs[0].device

    @property
    def render_mode(self):
        return self.envs[0].render_mode
