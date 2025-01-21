from pathlib import Path
from typing import Callable, List, Optional, Union

import gymnasium as gym

import numpy as np

from mani_skill.utils import gym_utils, visualization

from mshab.utils.array import to_numpy
from mshab.utils.video import images_to_video, put_info_on_image


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def chunked_string_list(arr, name, chunk_size=10):
    if isinstance(arr, np.number) and len(arr.shape) < 1:
        arr = [arr]
    strs = [",".join(c) for c in chunks([f"{x:.2f}" for x in arr], chunk_size)]
    for i in range(len(strs)):
        if i == 0:
            strs[i] = f"{name}: " + strs[i]
        else:
            strs[i] = "    " + strs[i]
    return strs


class RecordVideo(gym.vector.VectorEnvWrapper):

    def __init__(
        self,
        env,
        output_dir: str,
        save_video: bool = True,
        info_on_video: bool = False,
        chunk_infos: Optional[List[str]] = None,
        save_on_reset: bool = True,
        save_video_trigger: Optional[Callable[[int], bool]] = None,
        max_steps_per_video: Optional[int] = None,
        video_fps: int = 20,
        avoid_overwriting_video: bool = False,
    ) -> None:
        super().__init__(env)

        self.output_dir = Path(output_dir)
        self.video_fps = video_fps
        self._elapsed_record_steps = 0
        self._episode_id = -1
        self._video_id = -1
        self._video_steps = 0
        self._closed = False

        self.save_video_trigger = save_video_trigger

        self.max_steps_per_video = max_steps_per_video

        self.save_on_reset = save_on_reset
        self._save_video = save_video
        self.info_on_video = info_on_video
        self.chunk_infos = chunk_infos
        self.render_images = []
        self.video_nrows = int(np.sqrt(self.unwrapped.num_envs))
        self._avoid_overwriting_video = avoid_overwriting_video

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def base_env(self):
        return self.env.unwrapped

    @property
    def save_video(self):
        if not self._save_video:
            return False
        if self.save_video_trigger is not None:
            return self.save_video_trigger(self._elapsed_record_steps)
        else:
            return self._save_video

    def capture_image(self, info=dict()):
        images = to_numpy(self.base_env.render())

        if self.info_on_video:
            # add infos to images
            current_info = to_numpy(info)

            infos_per_env = [dict() for _ in range(self.base_env.num_envs)]
            for k, v in current_info.items():
                if isinstance(v, dict):
                    continue
                if not np.iterable(v):
                    v = [v] * self.base_env.num_envs
                for i in range(self.base_env.num_envs):
                    infos_per_env[i][k] = v[i]

            for i, (image, env_info) in enumerate(zip(images, infos_per_env)):
                base_lin_vel = env_info.pop("base_lin_vel", [])
                base_ang_vel = env_info.pop("base_ang_vel", [])
                projected_gravity = env_info.pop("projected_gravity", [])
                command = env_info.pop("command", [])
                qpos_relative_to_default = env_info.pop("qpos_relative_to_default", [])
                action = env_info.pop("action", [])
                reward = env_info.pop("reward", -np.inf)

                qpos = env_info.pop("qpos", [])
                qvel = env_info.pop("qvel", [])

                image_before_extras = []
                if self.chunk_infos is not None:
                    for ck in self.chunk_infos:
                        image_before_extras += chunked_string_list(
                            env_info.pop(ck, []), ck, chunk_size=10
                        )
                image_info = gym_utils.extract_scalars_from_info(env_info)
                image_extras = [
                    f"reward: {reward:.3f}",
                    *chunked_string_list(command, "command", chunk_size=3),
                    *chunked_string_list(base_lin_vel, "base_lin_vel", chunk_size=10),
                    *chunked_string_list(base_ang_vel, "base_ang_vel", chunk_size=10),
                    *chunked_string_list(
                        projected_gravity, "projected_gravity", chunk_size=10
                    ),
                    *chunked_string_list(
                        qpos_relative_to_default,
                        "qpos_relative_to_default",
                        chunk_size=10,
                    ),
                    *chunked_string_list(action, "action", chunk_size=10),
                    *chunked_string_list(qpos, "qpos", chunk_size=10),
                    *chunked_string_list(qvel, "qvel", chunk_size=10),
                ]

                image = put_info_on_image(
                    image,
                    image_info,
                    before_extras=image_before_extras,
                    extras=image_extras,
                    rgb=(0, 0, 0),
                    font_thickness=2,
                )
                image = put_info_on_image(
                    image,
                    image_info,
                    before_extras=image_before_extras,
                    extras=image_extras,
                    rgb=(0, 255, 0),
                    font_thickness=1,
                )
                images[i] = image

            if len(images.shape) > 3:
                images = visualization.tile_images(images, nrows=self.video_nrows)
            return images

        img = self.base_env.render()
        img = to_numpy(img)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = visualization.tile_images(img, nrows=self.video_nrows)
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
        obs, info = self.env.reset(*args, seed=seed, options=options, **kwargs)
        self._first_step_info = info
        return obs, info

    def step(self, action):
        if self.save_video and self._video_steps == 0:
            # save the first frame of the video here (s_0) instead of inside reset as user
            # may call env.reset(...) multiple times but we want to ignore empty trajectories
            self.render_images.append(self.capture_image(self._first_step_info))
            self._first_step_info = None
        obs, rew, terminated, truncated, info = self.env.step(action)
        if self.save_video:
            self._video_steps += 1
            image = self.capture_image(
                dict(
                    **info,
                    action=to_numpy(action),
                    reward=to_numpy(rew),
                )
            )
            self.render_images.append(image)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()
        self._elapsed_record_steps += 1
        return obs, rew, terminated, truncated, info

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
        self._closed = True
        if self.save_video:
            if self.save_on_reset:
                self.flush_video()
        return super().close()
