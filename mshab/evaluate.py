import json
import random
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm

from gymnasium import spaces

import numpy as np
import torch

# ManiSkill specific imports
import mani_skill.envs
from mani_skill import ASSET_DIR
from mani_skill.utils import common

from mshab.agents.bc import Agent as BCAgent
from mshab.agents.dp import Agent as DPAgent
from mshab.agents.ppo import Agent as PPOAgent
from mshab.agents.sac import Agent as SACAgent
from mshab.envs.make import EnvConfig, make_env
from mshab.envs.planner import CloseSubtask, OpenSubtask, PickSubtask, PlaceSubtask
from mshab.utils.array import recursive_slice, to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


if TYPE_CHECKING:
    from mshab.envs import SequentialTaskEnv

POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS = dict(
    bc_placed_500=dict(
        prepare_groceries=dict(
            place=["all"],
        ),
    ),
    bc_dropped_500=dict(
        prepare_groceries=dict(
            place=["all"],
        ),
    ),
    bc_placed_dropped_500=dict(
        prepare_groceries=dict(
            place=["all"],
        ),
    ),
    bc=dict(
        tidy_house=dict(
            pick=["all"],
            place=["all"],
        ),
        prepare_groceries=dict(
            pick=["all"],
            place=["all"],
        ),
        set_table=dict(
            pick=["all"],
            place=["all"],
            open=["fridge", "kitchen_counter"],
            close=["fridge", "kitchen_counter"],
        ),
    ),
    dp=dict(
        tidy_house=dict(
            pick=["all"],
            place=["all"],
        ),
        prepare_groceries=dict(
            pick=["all"],
            place=["all"],
        ),
        set_table=dict(
            pick=["all"],
            place=["all"],
            open=["fridge", "kitchen_counter"],
            close=["fridge", "kitchen_counter"],
        ),
    ),
    rl=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                "all",
            ],
            place=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                "all",
            ],
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                "all",
            ],
            place=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                "all",
            ],
        ),
        set_table=dict(
            pick=["013_apple", "024_bowl", "all"],
            place=["013_apple", "024_bowl", "all"],
            open=["fridge", "kitchen_counter"],
            close=["fridge", "kitchen_counter"],
        ),
    ),
)

SPAWN_LOC_RADIUS = 1.8
SPAWN_XY_NOISE_STD = 0.1
SPAWN_XY_NOISE_MAX = 0.2
SPAWN_ROT_NOISE_STD = 0.25
SPAWN_ROT_NOISE_MAX = 0.5
MAX_SPAWN_ATTEMPTS = 40


@dataclass
class EvalConfig:
    seed: int
    task: str
    eval_env: EnvConfig
    logger: LoggerConfig

    policy_type: str = "rl_per_obj"
    max_trajectories: int = 1000
    save_trajectory: bool = False

    policy_key: str = field(init=False)

    def __post_init__(self):
        assert self.task in ["tidy_house", "prepare_groceries", "set_table"]
        assert self.task in self.eval_env.task_plan_fp

        assert self.policy_type in ["rl_all_obj", "rl_per_obj"] + list(
            POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS.keys()
        )
        self.policy_key = (
            self.policy_type.split("_")[0]
            if "rl" in self.policy_type
            else self.policy_type
        )

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]


def get_mshab_train_cfg(cfg: EvalConfig) -> EvalConfig:
    return from_dict(data_class=EvalConfig, data=OmegaConf.to_container(cfg))


def eval(cfg: EvalConfig):
    # timer
    timer = NonOverlappingTimeProfiler()

    # seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # NOTE (arth): maybe require cuda since we only allow gpu sim anyways
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------------------------------
    # ENVS
    # -------------------------------------------------------------------------------------------------

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=None,
    )
    eval_envs = make_env(
        cfg.eval_env,
        video_path=logger.eval_video_path,
    )
    uenv: SequentialTaskEnv = eval_envs.unwrapped
    eval_obs, _ = eval_envs.reset()

    # -------------------------------------------------------------------------------------------------
    # SPACES
    # -------------------------------------------------------------------------------------------------

    obs_space = uenv.single_observation_space
    act_space = uenv.single_action_space

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------

    # TODO (arth): make this oop, originally this was easier but with 4 algos it's getting messy
    dp_action_history = deque([])

    def get_policy_act_fn(algo_cfg_path, algo_ckpt_path):
        algo_cfg = parse_cfg(default_cfg_path=algo_cfg_path).algo
        if algo_cfg.name == "ppo":
            policy = PPOAgent(eval_obs, act_space.shape)
            policy.eval()
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            policy.to(device)
            policy_act_fn = lambda obs: policy.get_action(obs, deterministic=True)
        elif algo_cfg.name == "sac":
            pixels_obs_space: spaces.Dict = obs_space["pixels"]
            state_obs_space: spaces.Box = obs_space["state"]
            model_pixel_obs_space = dict()
            for k, space in pixels_obs_space.items():
                shape, low, high, dtype = (
                    space.shape,
                    space.low,
                    space.high,
                    space.dtype,
                )
                if len(shape) == 4:
                    shape = (shape[0] * shape[1], shape[-2], shape[-1])
                    low = low.reshape((-1, *low.shape[-2:]))
                    high = high.reshape((-1, *high.shape[-2:]))
                model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
            model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)
            policy = SACAgent(
                model_pixel_obs_space,
                state_obs_space.shape,
                act_space.shape,
                actor_hidden_dims=list(algo_cfg.actor_hidden_dims),
                critic_hidden_dims=list(algo_cfg.critic_hidden_dims),
                critic_layer_norm=algo_cfg.critic_layer_norm,
                critic_dropout=algo_cfg.critic_dropout,
                encoder_pixels_feature_dim=algo_cfg.encoder_pixels_feature_dim,
                encoder_state_feature_dim=algo_cfg.encoder_state_feature_dim,
                cnn_features=list(algo_cfg.cnn_features),
                cnn_filters=list(algo_cfg.cnn_filters),
                cnn_strides=list(algo_cfg.cnn_strides),
                cnn_padding=algo_cfg.cnn_padding,
                log_std_min=algo_cfg.actor_log_std_min,
                log_std_max=algo_cfg.actor_log_std_max,
                device=device,
            )
            policy.eval()
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            policy.to(device)
            policy_act_fn = lambda obs: policy.actor(
                obs["pixels"],
                obs["state"],
                compute_pi=False,
                compute_log_pi=False,
            )[0]
        elif algo_cfg.name == "bc":
            policy = BCAgent(eval_obs, act_space.shape)
            policy.eval()
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            policy.to(device)
            policy_act_fn = lambda obs: policy(obs)
        elif algo_cfg.name == "diffusion_policy":
            assert cfg.eval_env.continuous_task
            assert cfg.eval_env.stack is not None and cfg.eval_env.frame_stack is None
            policy = DPAgent(
                single_observation_space=obs_space,
                single_action_space=act_space,
                obs_horizon=algo_cfg.obs_horizon,
                act_horizon=algo_cfg.act_horizon,
                pred_horizon=algo_cfg.pred_horizon,
                diffusion_step_embed_dim=algo_cfg.diffusion_step_embed_dim,
                unet_dims=algo_cfg.unet_dims,
                n_groups=algo_cfg.n_groups,
                device=device,
            )
            policy.eval()
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            policy.to(device)

            def get_dp_act(obs):
                if len(dp_action_history) == 0:
                    dp_action_history.extend(policy.get_action(obs).transpose(0, 1))

                return dp_action_history.popleft()

            policy_act_fn = get_dp_act
        else:
            raise NotImplementedError(f"algo {algo_cfg.name} not supported")
        policy_act_fn(to_tensor(eval_obs, device=device, dtype="float"))
        return policy_act_fn

    mshab_ckpt_dir = ASSET_DIR / "mshab_checkpoints"
    if not mshab_ckpt_dir.exists():
        mshab_ckpt_dir = Path("mshab_checkpoints")

    policies = dict()
    for subtask_name, subtask_targs in POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS[
        cfg.policy_key
    ][cfg.task].items():
        policies[subtask_name] = dict()
        for targ_name in subtask_targs:
            cfg_path = (
                mshab_ckpt_dir
                / cfg.policy_key
                / cfg.task
                / subtask_name
                / targ_name
                / "config.yml"
            )
            ckpt_path = (
                mshab_ckpt_dir
                / cfg.policy_key
                / cfg.task
                / subtask_name
                / targ_name
                / "policy.pt"
            )
            policies[subtask_name][targ_name] = get_policy_act_fn(cfg_path, ckpt_path)

    def act(obs):
        with torch.no_grad():
            with torch.device(device):
                action = torch.zeros(eval_envs.num_envs, *act_space.shape)

                # get subtask_type for subtask policy querying
                subtask_pointer = uenv.subtask_pointer.clone()
                get_subtask_type = lambda: uenv.task_ids[
                    torch.clip(
                        subtask_pointer,
                        max=len(uenv.task_plan) - 1,
                    )
                ]
                subtask_type = get_subtask_type()

                # find correct envs for each subtask policy
                pick_env_idx = subtask_type == 0
                place_env_idx = subtask_type == 1
                navigate_env_idx = subtask_type == 2
                open_env_idx = subtask_type == 3
                close_env_idx = subtask_type == 4

                # get targ names to query per-obj policies
                sapien_obj_names = [None] * uenv.num_envs
                for env_num, subtask_num in enumerate(
                    torch.clip(subtask_pointer, max=len(uenv.task_plan) - 1)
                ):
                    subtask = uenv.task_plan[subtask_num]
                    if isinstance(subtask, PickSubtask) or isinstance(
                        subtask, PlaceSubtask
                    ):
                        sapien_obj_names[env_num] = (
                            uenv.subtask_objs[subtask_num]._objs[env_num].name
                        )
                    elif isinstance(subtask, OpenSubtask) or isinstance(
                        subtask, CloseSubtask
                    ):
                        sapien_obj_names[env_num] = (
                            uenv.subtask_articulations[subtask_num]._objs[env_num].name
                        )
                targ_names = []
                for sapien_on in sapien_obj_names:
                    if sapien_on is None:
                        targ_names.append(None)
                    else:
                        for tn in task_targ_names:
                            if tn in sapien_on:
                                targ_names.append(tn)
                                break
                assert len(targ_names) == uenv.num_envs

                # if policy_type == "rl_per_obj" or doing open/close env, need to query per-obj policy
                if (
                    cfg.policy_type == "rl_per_obj"
                    or torch.any(open_env_idx)
                    or torch.any(close_env_idx)
                ):
                    tn_env_idxs = dict()
                    for env_num, tn in enumerate(targ_names):
                        if tn not in tn_env_idxs:
                            tn_env_idxs[tn] = []
                        tn_env_idxs[tn].append(env_num)
                    for k, v in tn_env_idxs.items():
                        bool_env_idx = torch.zeros(uenv.num_envs, dtype=torch.bool)
                        bool_env_idx[v] = True
                        tn_env_idxs[k] = bool_env_idx

                # query appropriate policy and place in action
                def set_subtask_targ_policy_act(subtask_name, subtask_env_idx):
                    if (
                        cfg.policy_type == "rl_per_obj"
                        or subtask_name
                        in [
                            "open",
                            "close",
                        ]
                    ) and subtask_name != "navigate":
                        for tn, targ_env_idx in tn_env_idxs.items():
                            subtask_targ_env_idx = subtask_env_idx & targ_env_idx
                            if torch.any(subtask_targ_env_idx):
                                action[subtask_targ_env_idx] = policies[subtask_name][
                                    tn
                                ](recursive_slice(obs, subtask_targ_env_idx))
                    else:
                        action[subtask_env_idx] = policies[subtask_name]["all"](
                            recursive_slice(obs, subtask_env_idx)
                        )

                if torch.any(pick_env_idx):
                    set_subtask_targ_policy_act("pick", pick_env_idx)
                if torch.any(place_env_idx):
                    set_subtask_targ_policy_act("place", place_env_idx)
                if torch.any(navigate_env_idx):
                    set_subtask_targ_policy_act("navigate", navigate_env_idx)
                if torch.any(open_env_idx):
                    set_subtask_targ_policy_act("open", open_env_idx)
                if torch.any(close_env_idx):
                    set_subtask_targ_policy_act("close", close_env_idx)

                return action

    # -------------------------------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------------------------------

    task_targ_names = set()
    for subtask_name in POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][cfg.task]:
        task_targ_names.update(
            POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][cfg.task][subtask_name]
        )

    eval_obs = to_tensor(
        eval_envs.reset(seed=cfg.seed)[0], device=device, dtype="float"
    )
    subtask_fail_counts = defaultdict(int)
    last_subtask_pointer = uenv.subtask_pointer.clone()
    pbar = tqdm(range(cfg.max_trajectories), total=cfg.max_trajectories)
    step_num = 0

    def check_done():
        if cfg.save_trajectory:
            # NOTE (arth): eval_envs.env._env is bad, fix in wrappers instead (prob with get_attr func)
            return eval_envs.env._env.reached_max_trajectories
        return len(eval_envs.return_queue) >= cfg.max_trajectories

    def update_pbar(step_num):
        if cfg.save_trajectory:
            diff = eval_envs.env._env.num_saved_trajectories - pbar.last_print_n
        else:
            diff = len(eval_envs.return_queue) - pbar.last_print_n

        if diff > 0:
            pbar.update(diff)

        pbar.set_description(f"{step_num=}")

    def update_fail_subtask_counts(done):
        if torch.any(done):
            subtask_nums = last_subtask_pointer[done]
            for fail_subtask, num_envs in zip(
                *np.unique(subtask_nums.cpu().numpy(), return_counts=True)
            ):
                subtask_fail_counts[fail_subtask] += num_envs
            with open(logger.exp_path / "subtask_fail_counts.json", "w+") as f:
                json.dump(
                    dict(
                        (str(k), int(subtask_fail_counts[k]))
                        for k in sorted(subtask_fail_counts.keys())
                    ),
                    f,
                )

    while not check_done():
        timer.end("other")
        last_subtask_pointer = uenv.subtask_pointer.clone()
        action = act(eval_obs)
        timer.end("sample")
        eval_obs, _, term, trunc, _ = eval_envs.step(action)
        timer.end("sim_sample")
        eval_obs = to_tensor(
            eval_obs,
            device=device,
            dtype="float",
        )
        update_pbar(step_num)
        update_fail_subtask_counts(term | trunc)
        if cfg.policy_key == "dp":
            if torch.any(term | trunc):
                dp_action_history.clear()
        step_num += 1

    # -------------------------------------------------------------------------------------------------
    # PRINT/SAVE RESULTS
    # -------------------------------------------------------------------------------------------------

    if len(cfg.eval_env.extra_stat_keys):
        torch.save(
            eval_envs.extra_stats,
            logger.exp_path / "eval_extra_stat_keys.pt",
        )

    print(
        "subtask_fail_counts",
        dict((k, subtask_fail_counts[k]) for k in sorted(subtask_fail_counts.keys())),
    )

    results_logs = dict(
        num_trajs=len(eval_envs.return_queue),
        return_per_step=common.to_tensor(eval_envs.return_queue, device=device)
        .float()
        .mean()
        / eval_envs.max_episode_steps,
        success_once=common.to_tensor(eval_envs.success_once_queue, device=device)
        .float()
        .mean(),
        success_at_end=common.to_tensor(eval_envs.success_at_end_queue, device=device)
        .float()
        .mean(),
        len=common.to_tensor(eval_envs.length_queue, device=device).float().mean(),
    )
    time_logs = timer.get_time_logs(pbar.last_print_n * cfg.eval_env.max_episode_steps)
    print(
        "results",
        results_logs,
    )
    print("time", time_logs)
    print("total_time", timer.total_time_elapsed)

    with open(logger.exp_path / "output.txt", "w") as f:
        f.write("results\n" + str(results_logs) + "\n")
        f.write("time\n" + str(time_logs) + "\n")

    # -------------------------------------------------------------------------------------------------
    # CLOSE
    # -------------------------------------------------------------------------------------------------

    eval_envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    eval(cfg)
