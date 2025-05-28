from typing import Any, Dict, List

import torch

from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose

from mshab.envs.planner import NavigateSubtask, NavigateSubtaskConfig, TaskPlan
from mshab.envs.sequential_task import GOAL_POSE_Q
from mshab.envs.subtask import SubtaskTrainEnv


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
                    for i, oid in enumerate(obj_ids)
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
                p=[subtask.prev_goal_pos for subtask in parallel_subtasks],
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
                    torch.norm(self.agent.robot.qvel[..., :3], dim=1) / 3
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
