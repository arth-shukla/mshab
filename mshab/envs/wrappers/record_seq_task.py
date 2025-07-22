import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

import h5py

import gymnasium as gym

import numpy as np
import torch

import sapien.physx as physx

from mani_skill import get_commit_info
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from mani_skill.utils.wrappers import CPUGymWrapper

from mshab.envs import SequentialTaskEnv
from mshab.utils.io import NoIndent, NoIndentSupportingJSONEncoder


# NOTE (stao): The code for record.py is quite messy and perhaps confusing as it is trying to support both recording on CPU and GPU seamlessly
# and handle partial resets. It works but can be claned up a lot.


def parse_env_info(env: gym.Env):
    # spec can be None if not initialized from gymnasium.make
    env = env.unwrapped
    if env.spec is None:
        return None
    if hasattr(env.spec, "_kwargs"):
        # gym<=0.21
        env_kwargs = env.spec._kwargs
    else:
        # gym>=0.22
        env_kwargs = env.spec.kwargs
    env_kwargs.pop("task_plans")
    return dict(
        env_id=env.spec.id,
        env_kwargs=env_kwargs,
    )


def temp_deep_print_shapes(x, prefix=""):
    if isinstance(x, dict):
        for k in x:
            temp_deep_print_shapes(x[k], prefix=prefix + "/" + k)
    else:
        print(prefix, x.shape)


def clean_trajectories(h5_file: h5py.File, json_dict: dict, prune_empty_action=True):
    """Clean trajectories by renaming and pruning trajectories in place.

    After cleanup, trajectory names are consecutive integers (traj_0, traj_1, ...),
    and trajectories with empty action are pruned.

    Args:
        h5_file: raw h5 file
        json_dict: raw JSON dict
        prune_empty_action: whether to prune trajectories with empty action
    """
    json_episodes = json_dict["episodes"]
    assert len(h5_file) == len(json_episodes)

    # Assumes each trajectory is named "traj_{i}"
    prefix_length = len("traj_")
    ep_ids = sorted([int(x[prefix_length:]) for x in h5_file.keys()])

    new_json_episodes = []
    new_ep_id = 0

    for i, ep_id in enumerate(ep_ids):
        traj_id = f"traj_{ep_id}"
        ep = json_episodes[i]
        assert ep["episode_id"] == ep_id
        new_traj_id = f"traj_{new_ep_id}"

        if prune_empty_action and ep["elapsed_steps"] == 0:
            del h5_file[traj_id]
            continue

        if new_traj_id != traj_id:
            ep["episode_id"] = new_ep_id
            h5_file[new_traj_id] = h5_file[traj_id]
            del h5_file[traj_id]

        new_json_episodes.append(ep)
        new_ep_id += 1

    json_dict["episodes"] = new_json_episodes


@dataclass
class Step:
    state: np.ndarray
    observation: np.ndarray
    info: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray
    env_episode_ptr: np.ndarray
    """points to index in above data arrays where current episode started (any data before should already be flushed)"""

    success: np.ndarray = None
    fail: np.ndarray = None


class RecordEpisodeSequentialTask(gym.Wrapper):

    def __init__(
        self,
        env: SequentialTaskEnv,
        output_dir: str,
        save_trajectory: bool = True,
        trajectory_name: Optional[str] = None,
        save_video: bool = True,
        info_on_video: bool = False,
        save_on_reset: bool = True,
        save_video_trigger: Optional[Callable[[int], bool]] = None,
        max_steps_per_video: Optional[int] = None,
        clean_on_close: bool = True,
        record_reward: bool = True,
        record_env_state: bool = True,
        max_trajectories: Optional[int] = None,
        demo_filter: Literal["success", "min_success_subtasks", "any"] = "any",
        min_success_subtasks: Optional[int] = None,
        video_fps: int = 20,
        render_substeps: bool = False,
        avoid_overwriting_video: bool = False,
        source_type: Optional[str] = None,
        source_desc: Optional[str] = None,
    ) -> None:
        super().__init__(env)

        self.output_dir = Path(output_dir)
        if save_trajectory or save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = video_fps
        self._elapsed_record_steps = 0
        self._episode_id = -1
        self._video_id = -1
        self._video_steps = 0
        self._closed = False

        self.save_video_trigger = save_video_trigger

        self._trajectory_buffer: Step = None

        self.max_steps_per_video = max_steps_per_video
        self.max_episode_steps = gym_utils.find_max_episode_steps_value(env)

        self.save_on_reset = save_on_reset
        self.save_trajectory = save_trajectory
        if self.base_env.num_envs > 1 and save_video:
            assert (
                max_steps_per_video is not None
            ), "On GPU parallelized environments, \
                there must be a given max steps per video value in order to flush videos in order \
                to avoid issues caused by partial resets. If your environment does not do partial \
                resets you may set max_steps_per_video equal to the max_episode_steps"
        self.clean_on_close = clean_on_close
        self.record_reward = record_reward
        self.record_env_state = record_env_state
        if self.save_trajectory:
            if not trajectory_name:
                trajectory_name = time.strftime("%Y%m%d_%H%M%S")

            self._h5_file = h5py.File(self.output_dir / f"{trajectory_name}.h5", "w")

            # Use a separate json to store non-array data
            self._json_path = self._h5_file.filename.replace(".h5", ".json")
            self._json_data = dict(
                env_info=parse_env_info(self.env),
                commit_info=get_commit_info(),
                episodes=[],
            )
            if self._json_data["env_info"] is not None:
                self._json_data["env_info"][
                    "max_episode_steps"
                ] = self.max_episode_steps
            if source_type is not None:
                self._json_data["source_type"] = source_type
            if source_desc is not None:
                self._json_data["source_desc"] = source_desc
        self._save_video = save_video
        self.info_on_video = info_on_video
        self.render_images = []
        self.video_nrows = int(np.sqrt(self.unwrapped.num_envs))
        self._avoid_overwriting_video = avoid_overwriting_video

        assert isinstance(
            self.base_env,
            SequentialTaskEnv,
        ), f"Episode labeling not available for {self.base_env.__class__.__name__}"
        self.max_trajectories = max_trajectories

        assert demo_filter in ["success", "min_success_subtasks", "any"]
        if demo_filter == "min_success_subtasks":
            assert min_success_subtasks is not None
        self.demo_filter = demo_filter
        self.min_success_subtasks = min_success_subtasks

        self._already_warned_about_state_dict_inconsistency = False

        # check if wrapped env is already wrapped by a CPU gym wrapper
        cur_env = self.env
        self.cpu_wrapped_env = False
        while cur_env is not None:
            if isinstance(cur_env, CPUGymWrapper):
                self.cpu_wrapped_env = True
                break
            if hasattr(cur_env, "env"):
                cur_env = cur_env.env
            else:
                break

        self.render_substeps = render_substeps
        if self.render_substeps:
            _original_after_simulation_step = self.base_env._after_simulation_step

            def wrapped_after_simulation_step():
                _original_after_simulation_step()
                if self.save_video:
                    if self.base_env.gpu_sim_enabled:
                        self.base_env.scene._gpu_fetch_all()
                    self.render_images.append(self.capture_image())

            self.base_env._after_simulation_step = wrapped_after_simulation_step

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def base_env(self) -> SequentialTaskEnv:
        return self.env.unwrapped

    @property
    def num_saved_trajectories(self) -> int:
        return len(self._json_data["episodes"])

    @property
    def reached_max_trajectories(self) -> bool:
        if self.max_trajectories is None:
            return False
        return self.num_saved_trajectories >= self.max_trajectories

    @property
    def save_video(self):
        if not self._save_video:
            return False
        if self.save_video_trigger is not None:
            return self.save_video_trigger(self._elapsed_record_steps)
        else:
            return self._save_video

    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]
        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=self.video_nrows)
        return img

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        **kwargs,
    ):

        if self.save_on_reset:
            if self.save_video and self.num_envs == 1:
                self.flush_video()
            # if doing a full reset then we flush all trajectories including incompleted ones
            if self._trajectory_buffer is not None:
                if "env_idx" not in options:
                    self.flush_trajectory(env_idxs_to_flush=np.arange(self.num_envs))
                else:
                    self.flush_trajectory(
                        env_idxs_to_flush=common.to_numpy(options["env_idx"])
                    )

        obs, info = super().reset(*args, seed=seed, options=options, **kwargs)
        self._first_step_info = info
        if info["reconfigure"]:
            # if we reconfigure, there is the possibility that state dictionary looks different now
            # so trajectory buffer must be wiped
            self._trajectory_buffer = None
        if self.save_trajectory:
            state_dict = self.base_env.get_state_dict()
            action = common.batch(
                self.env.get_wrapper_attr("single_action_space").sample()
            )
            first_step_info = info.copy()
            first_step_info.pop("reconfigure")
            # check if state_dict is consistent
            if not sapien_utils.is_state_dict_consistent(state_dict):
                self.record_env_state = False
                if not self._already_warned_about_state_dict_inconsistency:
                    logger.warn(
                        f"State dictionary is not consistent, disabling recording of environment states for {self.env}"
                    )
                    self._already_warned_about_state_dict_inconsistency = True
            first_step = Step(
                state=None,
                observation=common.to_numpy(common.batch(obs)),
                info=common.to_numpy(common.batch(first_step_info)),
                # note first reward/action etc. are ignored when saving trajectories to disk
                action=common.to_numpy(common.batch(action.repeat(self.num_envs, 0))),
                reward=np.zeros(
                    (
                        1,
                        self.num_envs,
                    ),
                    dtype=float,
                ),
                # terminated and truncated are fixed to be True at the start to indicate the start of an episode.
                # an episode is done when one of these is True otherwise the trajectory is incomplete / a partial episode
                terminated=np.ones((1, self.num_envs), dtype=bool),
                truncated=np.ones((1, self.num_envs), dtype=bool),
                done=np.ones((1, self.num_envs), dtype=bool),
                success=np.zeros((1, self.num_envs), dtype=bool),
                fail=np.zeros((1, self.num_envs), dtype=bool),
                env_episode_ptr=np.zeros((self.num_envs,), dtype=int),
            )
            if self.record_env_state:
                first_step.state = common.to_numpy(common.batch(state_dict))
            env_idx = np.arange(self.num_envs)
            if "env_idx" in options:
                env_idx = common.to_numpy(options["env_idx"])
            if self._trajectory_buffer is None:
                # Initialize trajectory buffer on the first episode based on given observation (which should be generated after all wrappers)
                self._trajectory_buffer = first_step
            else:

                def recursive_replace(x, y):
                    if isinstance(x, np.ndarray):
                        x[-1, env_idx] = y[-1, env_idx]
                    else:
                        for k in x.keys():
                            recursive_replace(x[k], y[k])

                # TODO (stao): how do we store states from GPU sim of tasks with objects not in every sub-scene?
                # Maybe we shouldn't?
                if self.record_env_state:
                    recursive_replace(self._trajectory_buffer.state, first_step.state)
                recursive_replace(
                    self._trajectory_buffer.observation, first_step.observation
                )
                recursive_replace(self._trajectory_buffer.info, first_step.info)
                recursive_replace(self._trajectory_buffer.action, first_step.action)
                if self.record_reward:
                    recursive_replace(self._trajectory_buffer.reward, first_step.reward)
                recursive_replace(
                    self._trajectory_buffer.terminated, first_step.terminated
                )
                recursive_replace(
                    self._trajectory_buffer.truncated, first_step.truncated
                )
                recursive_replace(self._trajectory_buffer.done, first_step.done)
                if self._trajectory_buffer.success is not None:
                    recursive_replace(
                        self._trajectory_buffer.success, first_step.success
                    )
                if self._trajectory_buffer.fail is not None:
                    recursive_replace(self._trajectory_buffer.fail, first_step.fail)
        if options is not None and "env_idx" in options:
            options["env_idx"] = common.to_numpy(options["env_idx"])
        self.last_reset_kwargs = copy.deepcopy(dict(options=options, **kwargs))
        if seed is not None:
            self.last_reset_kwargs.update(seed=seed)
        return obs, info

    def step(self, action):
        if self.save_video and self._video_steps == 0:
            # save the first frame of the video here (s_0) instead of inside reset as user
            # may call env.reset(...) multiple times but we want to ignore empty trajectories
            self.render_images.append(self.capture_image(self._first_step_info))
            self._first_step_info = None
        obs, rew, terminated, truncated, info = super().step(action)

        if self.save_trajectory:
            state_dict = self.base_env.get_state_dict()
            if self.record_env_state:
                self._trajectory_buffer.state = common.append_dict_array(
                    self._trajectory_buffer.state,
                    common.to_numpy(common.batch(state_dict)),
                )
            self._trajectory_buffer.observation = common.append_dict_array(
                self._trajectory_buffer.observation,
                common.to_numpy(common.batch(obs)),
            )
            self._trajectory_buffer.info = common.append_dict_array(
                self._trajectory_buffer.info,
                common.to_numpy(common.batch(info)),
            )

            self._trajectory_buffer.action = common.append_dict_array(
                self._trajectory_buffer.action,
                common.to_numpy(common.batch(action)),
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.append_dict_array(
                    self._trajectory_buffer.reward,
                    common.to_numpy(common.batch(rew)),
                )
            self._trajectory_buffer.terminated = common.append_dict_array(
                self._trajectory_buffer.terminated,
                common.to_numpy(common.batch(terminated)),
            )
            self._trajectory_buffer.truncated = common.append_dict_array(
                self._trajectory_buffer.truncated,
                common.to_numpy(common.batch(truncated)),
            )
            done = terminated | truncated
            self._trajectory_buffer.done = common.append_dict_array(
                self._trajectory_buffer.done,
                common.to_numpy(common.batch(done)),
            )
            if "success" in info:
                self._trajectory_buffer.success = common.append_dict_array(
                    self._trajectory_buffer.success,
                    common.to_numpy(common.batch(info["success"])),
                )
            else:
                self._trajectory_buffer.success = None
            if "fail" in info:
                self._trajectory_buffer.fail = common.append_dict_array(
                    self._trajectory_buffer.fail,
                    common.to_numpy(common.batch(info["fail"])),
                )
            else:
                self._trajectory_buffer.fail = None

        if self.save_video:
            self._video_steps += 1
            if self.info_on_video:
                scalar_info = gym_utils.extract_scalars_from_info(
                    common.to_numpy(info), batch_size=self.num_envs
                )
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
                image = self.capture_image(scalar_info)
            else:
                image = self.capture_image()

            self.render_images.append(image)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()
        self._elapsed_record_steps += 1
        return obs, rew, terminated, truncated, info

    def flush_trajectory(
        self,
        verbose=False,
        ignore_empty_transition=True,
        env_idxs_to_flush=None,
        save: bool = True,
    ):
        """
        Flushes a trajectory and by default saves it to disk

        Arguments:
            verbose (bool): whether to print out information about the flushed trajectory
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            env_idxs_to_flush: which environments by id to flush. If None, all environments are flushed.
            save (bool): whether to save the trajectory to disk
        """
        if self.reached_max_trajectories:
            return

        flush_count = 0
        if env_idxs_to_flush is None:
            env_idxs_to_flush = np.arange(0, self.num_envs)
        for env_idx in env_idxs_to_flush:
            if self.reached_max_trajectories:
                return

            start_ptr = self._trajectory_buffer.env_episode_ptr[env_idx]
            end_ptr = len(self._trajectory_buffer.done)
            if ignore_empty_transition and end_ptr - start_ptr <= 1:
                continue
            flush_count += 1
            if save:
                ep_success = common.index_dict_array(
                    self._trajectory_buffer.success,
                    (
                        slice(start_ptr + 1, end_ptr),
                        env_idx,
                    ),
                    inplace=False,
                )
                subtask_nums = common.index_dict_array(
                    self._trajectory_buffer.info,
                    (
                        slice(start_ptr + 1, end_ptr),
                        env_idx,
                    ),
                    inplace=False,
                )["subtask"]

                has_success = np.any(ep_success)
                if self.demo_filter == "success" and not has_success:
                    continue

                if not has_success:
                    num_success_subtasks = np.max(subtask_nums)
                    if (
                        self.demo_filter == "min_success_subtasks"
                        and num_success_subtasks < self.min_success_subtasks
                    ):
                        continue

                    end_ptr -= len(subtask_nums) - np.argmax(subtask_nums)

                if ignore_empty_transition and end_ptr - start_ptr <= 1:
                    continue

                self._episode_id += 1
                traj_id = "traj_{}".format(self._episode_id)
                group = self._h5_file.create_group(traj_id, track_order=True)

                def recursive_add_to_h5py(
                    group: h5py.Group, data: Union[dict, Array], key
                ):
                    """simple recursive data insertion for nested data structures into h5py, optimizing for visual data as well"""
                    if isinstance(data, dict):
                        subgrp = group.create_group(key, track_order=True)
                        for k in data.keys():
                            recursive_add_to_h5py(subgrp, data[k], k)
                    else:
                        if key == "rgb":
                            # NOTE(jigu): It is more efficient to use gzip than png for a sequence of images.
                            group.create_dataset(
                                "rgb",
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        elif key == "depth":
                            # NOTE (stao): By default now cameras in ManiSkill return depth values of type uint16 for numpy
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        elif key == "seg":
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        else:
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                            )

                # Observations need special processing
                if isinstance(self._trajectory_buffer.observation, dict):
                    recursive_add_to_h5py(
                        group, self._trajectory_buffer.observation, "obs"
                    )
                elif isinstance(self._trajectory_buffer.observation, np.ndarray):
                    if self.cpu_wrapped_env:
                        group.create_dataset(
                            "obs",
                            data=self._trajectory_buffer.observation[start_ptr:end_ptr],
                            dtype=self._trajectory_buffer.observation.dtype,
                        )
                    else:
                        group.create_dataset(
                            "obs",
                            data=self._trajectory_buffer.observation[
                                start_ptr:end_ptr, env_idx
                            ],
                            dtype=self._trajectory_buffer.observation.dtype,
                        )
                else:
                    raise NotImplementedError(
                        f"RecordEpisode wrapper does not know how to handle observation data of type {type(self._trajectory_buffer.observation)}"
                    )
                episode_info = dict(
                    episode_id=self._episode_id,
                    episode_seed=NoIndent(self.base_env._episode_seed.tolist()),
                    build_config_idx=self.base_env.build_config_idxs[env_idx],
                    task_plan_idx=self.base_env.task_plan_idxs[env_idx].item(),
                    init_config_idx=self.base_env.init_config_idxs[env_idx],
                    control_mode=self.base_env.control_mode,
                    elapsed_steps=end_ptr - start_ptr - 1,
                )
                if self.num_envs == 1:
                    episode_info.update(reset_kwargs=self.last_reset_kwargs)
                else:
                    # NOTE (stao): With multiple envs in GPU simulation, reset_kwargs do not make much sense
                    episode_info.update(reset_kwargs=dict())

                # NOTE (arth): infos also need special processing
                if isinstance(self._trajectory_buffer.info, dict):
                    recursive_add_to_h5py(group, self._trajectory_buffer.info, "info")
                elif isinstance(self._trajectory_buffer.info, np.ndarray):
                    group.create_dataset(
                        "info",
                        data=self._trajectory_buffer.info[start_ptr:end_ptr, env_idx],
                        dtype=self._trajectory_buffer.info.dtype,
                    )
                else:
                    raise NotImplementedError(
                        f"RecordEpisode wrapper does not know how to handle info data of type {type(self._trajectory_buffer.info)}"
                    )

                # slice some data to remove the first dummy frame.
                actions = common.index_dict_array(
                    self._trajectory_buffer.action,
                    (slice(start_ptr + 1, end_ptr), env_idx),
                )
                terminated = self._trajectory_buffer.terminated[
                    start_ptr + 1 : end_ptr, env_idx
                ]
                truncated = self._trajectory_buffer.truncated[
                    start_ptr + 1 : end_ptr, env_idx
                ]
                if isinstance(self._trajectory_buffer.action, dict):
                    recursive_add_to_h5py(group, actions, "actions")
                else:
                    group.create_dataset("actions", data=actions, dtype=np.float32)
                group.create_dataset("terminated", data=terminated, dtype=bool)
                group.create_dataset("truncated", data=truncated, dtype=bool)

                if self._trajectory_buffer.success is not None:
                    group.create_dataset(
                        "success",
                        data=self._trajectory_buffer.success[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=bool,
                    )
                    episode_info.update(
                        success=self._trajectory_buffer.success[end_ptr - 1, env_idx]
                    )
                if self._trajectory_buffer.fail is not None:
                    group.create_dataset(
                        "fail",
                        data=self._trajectory_buffer.fail[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=bool,
                    )
                    episode_info.update(
                        fail=self._trajectory_buffer.fail[end_ptr - 1, env_idx]
                    )
                if self.record_env_state:
                    recursive_add_to_h5py(
                        group, self._trajectory_buffer.state, "env_states"
                    )
                if self.record_reward:
                    group.create_dataset(
                        "rewards",
                        data=self._trajectory_buffer.reward[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=np.float32,
                    )

                self._json_data["episodes"].append(common.to_numpy(episode_info))
                dump_json(
                    self._json_path,
                    self._json_data,
                    encoder_cls=NoIndentSupportingJSONEncoder,
                    indent=2,
                )
                if verbose:
                    if flush_count == 1:
                        print(f"Recorded episode {self._episode_id}")
                    else:
                        print(
                            f"Recorded episodes {self._episode_id - flush_count} to {self._episode_id}"
                        )

        # truncate self._trajectory_buffer down to save memory
        if flush_count > 0:
            self._trajectory_buffer.env_episode_ptr[env_idxs_to_flush] = (
                len(self._trajectory_buffer.done) - 1
            )
            min_env_ptr = self._trajectory_buffer.env_episode_ptr.min()
            N = len(self._trajectory_buffer.done)

            if self.record_env_state:
                self._trajectory_buffer.state = common.index_dict_array(
                    self._trajectory_buffer.state, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.observation = common.index_dict_array(
                self._trajectory_buffer.observation, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.info = common.index_dict_array(
                self._trajectory_buffer.info, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.action = common.index_dict_array(
                self._trajectory_buffer.action, slice(min_env_ptr, N)
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.index_dict_array(
                    self._trajectory_buffer.reward, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.terminated = common.index_dict_array(
                self._trajectory_buffer.terminated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.truncated = common.index_dict_array(
                self._trajectory_buffer.truncated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.done = common.index_dict_array(
                self._trajectory_buffer.done, slice(min_env_ptr, N)
            )
            if self._trajectory_buffer.success is not None:
                self._trajectory_buffer.success = common.index_dict_array(
                    self._trajectory_buffer.success, slice(min_env_ptr, N)
                )
            if self._trajectory_buffer.fail is not None:
                self._trajectory_buffer.fail = common.index_dict_array(
                    self._trajectory_buffer.fail, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.env_episode_ptr -= min_env_ptr

    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
                if self._avoid_overwriting_video:
                    while (
                        Path(self.output_dir)
                        / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
                    ).exists():
                        self._video_id += 1
                        video_name = "{}".format(self._video_id)
                        if suffix:
                            video_name += "_" + suffix
            else:
                video_name = name
            images_to_video(
                self.render_images,
                str(self.output_dir),
                video_name=video_name,
                fps=self.video_fps,
                verbose=verbose,
            )
        self._video_steps = 0
        self.render_images = []

    def close(self) -> None:
        if self._closed:
            # There is some strange bug when vector envs using record wrapper are closed/deleted, this code runs twice
            return
        self._closed = True
        if self.save_trajectory:
            # Handle the last episode only when `save_on_reset=True`
            if self.save_on_reset and self._trajectory_buffer is not None:
                self.flush_trajectory(
                    ignore_empty_transition=True,
                    env_idxs_to_flush=np.arange(self.num_envs),
                )
            if self.clean_on_close:
                clean_trajectories(self._h5_file, self._json_data)
                dump_json(
                    self._json_path,
                    self._json_data,
                    encoder_cls=NoIndentSupportingJSONEncoder,
                    indent=2,
                )
            self._h5_file.close()
        if self.save_video:
            if self.save_on_reset:
                self.flush_video()
        return super().close()
