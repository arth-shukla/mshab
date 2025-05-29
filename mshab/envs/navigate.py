from collections import defaultdict
from typing import Any, Dict, List

import torch

from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose

from mshab.envs.planner import NavigateSubtask, NavigateSubtaskConfig, TaskPlan
from mshab.envs.sequential_task import GOAL_POSE_Q
from mshab.envs.subtask import SubtaskTrainEnv
from mshab.utils.array import tensor_intersection, tensor_intersection_idx


@register_env("NavigateSubtaskTrain-v0", max_episode_steps=200)
class NavigateSubtaskTrainEnv(SubtaskTrainEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    navigate_cfg = NavigateSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        robot_cumulative_force_limit=torch.inf,
    )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        **kwargs,
    ):

        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], NavigateSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {NavigateSubtask.__name__} long"

        self.subtask_cfg = self.navigate_cfg

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT ROBOT SPAWN RANDOMIZATION
    # -------------------------------------------------------------------------------------------------

    def _load_scene(self, options):
        super()._load_scene(options)
        self.premade_goal_list: List[Actor] = [
            self._make_goal(
                radius=0.15,
                name="goal_0",
                goal_type="sphere",
            )
        ]
        self.prev_goal_pos_goal = self._make_goal(
            radius=0.15, name="prev_goal_0", goal_type="sphere", color=[1, 0, 0, 1]
        )

    def _after_reconfigure(self, options):
        with torch.device(self.device):
            super()._after_reconfigure(options)
            self.starting_qpos = torch.zeros_like(self.agent.robot.qpos)
            self.last_distance_from_goal = torch.full(
                (self.num_envs,), -1, dtype=torch.float
            )

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            super()._initialize_episode(env_idx, options)
            self.starting_qpos[env_idx] = self.agent.robot.qpos[env_idx]
            self.last_distance_from_goal[env_idx] = torch.norm(
                self.agent.robot.pose.p[env_idx, :2]
                - self.subtask_goals[0].pose.p[env_idx, :2],
                dim=1,
            )

    def _apply_premade_spawns(self, env_idx, options: Dict):
        with torch.device(self.device):
            current_subtask = self.task_plan[0]
            batched_spawn_data = defaultdict(list)
            spawn_selection_idxs = options.get(
                "spawn_selection_idxs", [None] * env_idx.numel()
            )
            for env_num, subtask_uid, spawn_selection_idx in zip(
                env_idx,
                [
                    current_subtask.composite_subtask_uids[env_num]
                    for env_num in env_idx
                ],
                spawn_selection_idxs,
            ):
                spawn_data: Dict[str, torch.Tensor] = self.spawn_data[subtask_uid]
                for k, v in spawn_data.items():
                    if spawn_selection_idx is None:
                        spawn_selection_idx = torch.randint(
                            low=0, high=len(v), size=(1,)
                        )
                        self.spawn_selection_idxs[env_num] = spawn_selection_idx.item()
                    elif isinstance(spawn_selection_idx, int):
                        self.spawn_selection_idxs[env_num] = spawn_selection_idx
                        spawn_selection_idx = [spawn_selection_idx]
                    batched_spawn_data[k].append(v[spawn_selection_idx])
            for k, v in batched_spawn_data.items():
                if k == "articulation_qpos":
                    articulation_qpos = torch.zeros(
                        (env_idx.numel(), self.subtask_articulations[0].max_dof),
                        device=self.device,
                        dtype=torch.float,
                    )
                    for i in range(env_idx.numel()):
                        articulation_qpos[i, : v[i].size(1)] = v[i].squeeze(0)
                    batched_spawn_data[k] = articulation_qpos
                else:
                    batched_spawn_data[k] = torch.cat(v, dim=0)
            if "robot_pos" in batched_spawn_data:
                self.agent.robot.set_pose(
                    Pose.create_from_pq(p=batched_spawn_data["robot_pos"])
                )
            if "robot_qpos" in batched_spawn_data:
                self.agent.robot.set_qpos(batched_spawn_data["robot_qpos"])
            subtask_obj = self.subtask_objs[0]
            if subtask_obj is not None:
                obj_reset_idxs = tensor_intersection_idx(
                    env_idx, subtask_obj._scene_idxs
                )
                if "obj_raw_pose" in batched_spawn_data:
                    subtask_obj.set_pose(
                        Pose.create(batched_spawn_data["obj_raw_pose"][obj_reset_idxs])
                    )
                if "obj_raw_pose_wrt_tcp" in batched_spawn_data:
                    if self.gpu_sim_enabled:
                        self.scene._gpu_apply_all()
                        self.scene.px.gpu_update_articulation_kinematics()
                        self.scene._gpu_fetch_all()
                    subtask_obj.set_pose(
                        Pose.create(
                            self.agent.tcp.pose.raw_pose[
                                tensor_intersection(env_idx, subtask_obj._scene_idxs)
                            ]
                        )  # NOTE (arth): use tcp.pose for spawning for slightly better accuracy
                        * Pose.create(
                            batched_spawn_data["obj_raw_pose_wrt_tcp"][obj_reset_idxs]
                        )
                    )
            if "articulation_qpos" in batched_spawn_data:
                self.subtask_articulations[0].set_qpos(
                    batched_spawn_data["articulation_qpos"]
                )
                self.subtask_articulations[0].set_qvel(
                    self.subtask_articulations[0].qvel[env_idx] * 0
                )
                if self.gpu_sim_enabled and len(env_idx) == self.num_envs:
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene.px.step()
                    self.scene._gpu_fetch_all()

    # NOTE (arth): sometimes will need to nav w/ object, sometimes not
    #       override _merge_navigate_subtasks to allow obj in only some envs
    def _merge_navigate_subtasks(
        self,
        env_idx: torch.Tensor,
        last_subtask0,
        subtask_num: int,
        parallel_subtasks: List[NavigateSubtask],
    ):
        obj_ids = []
        scene_idxs = []
        for i, subtask in enumerate(parallel_subtasks):
            if subtask.obj_id is not None:
                scene_idxs.append(i)
                obj_ids.append(subtask.obj_id)

        if len(obj_ids) > 0:
            merged_obj = Actor.create_from_entities(
                [
                    self._get_actor_entity(actor_id=f"env-{i}_{oid}", env_num=i)
                    for i, oid in zip(scene_idxs, obj_ids)
                ],
                scene=self.scene,
                scene_idxs=torch.tensor(scene_idxs, dtype=torch.int),
            )
            merged_obj.name = merged_obj_name = f"obj_{subtask_num}"
        else:
            merged_obj = None
            merged_obj_name = None
        self.subtask_objs.append(merged_obj)

        self.subtask_goals.append(self.premade_goal_list[subtask_num])
        self.subtask_goals[-1].set_pose(
            Pose.create_from_pq(
                q=GOAL_POSE_Q,
                p=[subtask.goal_pos for subtask in parallel_subtasks],
            )
        )
        self.prev_goal_pos_goal.set_pose(
            Pose.create_from_pq(
                q=GOAL_POSE_Q,
                p=[
                    (
                        subtask.prev_goal_pos
                        if subtask.prev_goal_pos is not None
                        else [-1, 0, 0.02]
                    )
                    for subtask in parallel_subtasks
                ],
            )
        )

        self.subtask_articulations.append(None)

        self.task_plan.append(
            NavigateSubtask(
                obj_id=merged_obj_name,
                goal_pos=self.subtask_goals[-1].pose.p,
            )
        )

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        info = super().evaluate()
        info["distance_from_goal"] = torch.norm(
            self.agent.robot.pose.p[..., :2] - self.subtask_goals[0].pose.p[..., :2],
            dim=1,
        )
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj = self.subtask_objs[0]

            begin_navigating = torch.ones(self.num_envs, dtype=torch.bool)
            if obj is not None:
                if len(obj._scene_idxs) != self.num_envs:
                    should_grasp = torch.zeros(self.num_envs, dtype=torch.bool)
                    should_grasp[obj._scene_idxs] = True
                    begin_navigating[should_grasp & ~info["is_grasped"]] = False
                else:
                    begin_navigating[~info["is_grasped"]] = False
            reward += 2 * begin_navigating

            if torch.any(begin_navigating):
                done_moving = info["oriented_correctly"] & info["navigated_close"]
                done_navigating = info["navigated_close"]
                still_navigating = ~done_navigating

                done_moving &= begin_navigating
                done_navigating &= begin_navigating
                still_navigating &= begin_navigating

                reward[done_navigating] += 12
                reward[still_navigating] += 10 * torch.tanh(
                    torch.abs(
                        self.last_distance_from_goal[still_navigating]
                        - info["distance_from_goal"][still_navigating]
                    )
                )

                bqvel_rew = torch.tanh(
                    torch.norm(self.agent.robot.qvel[done_moving, :3], dim=1) / 3
                )
                reward[done_moving] += 2 * (1 - bqvel_rew)

            # collisions
            step_no_col_rew = 5 * (
                1
                - torch.tanh(
                    3
                    * (
                        torch.clamp(
                            self.robot_force_mult * info["robot_force"],
                            min=self.robot_force_penalty_min,
                        )
                        - self.robot_force_penalty_min
                    )
                )
            )
            reward += step_no_col_rew

            # enforce arm in similar position as at start of episode
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:] - self.starting_qpos[..., 3:],
                dim=1,
            )
            arm_resting_orientation_rew = 3 * (1 - torch.tanh(arm_to_resting_diff / 5))
            reward += arm_resting_orientation_rew

            # set for next step
            self.last_distance_from_goal = info["distance_from_goal"]

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 24.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
