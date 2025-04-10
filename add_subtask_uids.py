import json
import os
from collections import defaultdict
from pathlib import Path

import h5py
from tqdm import tqdm

import gymnasium as gym

import torch

from mani_skill import ASSET_DIR
from mani_skill.utils.io_utils import dump_json

from mshab.envs.planner import plan_data_from_file
from mshab.envs.subtask import SubtaskTrainEnv
from mshab.envs.wrappers import RecordEpisode
from mshab.utils.io import NoIndent, NoIndentSupportingJSONEncoder


REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
REARRANGE_DATASET_DIR = (
    ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange-dataset"
)

# Walk through all directories and files in REARRANGE_DATASET_DIR
for root, dirs, files in os.walk(REARRANGE_DATASET_DIR):
    for file in files:
        if file.endswith(".json"):
            fp = Path(root) / file
            print(fp)

            if "with_subtask_uid" in str(fp):
                continue

            if "save" in str(fp):
                continue

            if not "all.json" in str(fp):
                continue

            new_fp = fp.parent / f"{fp.stem}_with_subtask_uid{fp.suffix}"
            print("old:", fp, "new:", new_fp)
            # if new_fp.exists():
            #     print(f"Skipping {fp} because {new_fp} already exists")
            #     continue

            task = fp.parent.parent.name
            subtask = fp.parent.name
            obj = fp.stem

            with open(fp, "r") as f:
                data = json.load(f)
            # traj_data = h5py.File(fp.with_suffix(".h5"), "r")

            plan_data_fp = (
                REARRANGE_DIR / "task_plans" / task / subtask / "train" / f"{obj}.json"
            )
            plan_data = plan_data_from_file(plan_data_fp)
            spawn_data_fp = (
                REARRANGE_DIR / "spawn_data" / task / subtask / "train/spawn_data.pt"
            )

            env_kwargs = {
                "id": data["env_info"]["env_id"],
                "task_plans": plan_data.plans,
                "scene_builder_cls": plan_data.dataset,
                "spawn_data_fp": spawn_data_fp,
                **data["env_info"]["env_kwargs"],
                "num_envs": 1,
                "reconfiguration_freq": 0,
                "render_mode": "rgb_array",
            }

            uenv: SubtaskTrainEnv = gym.make(**env_kwargs).unwrapped
            # uenv = RecordEpisode(uenv, output_dir=Path("videos"), save_trajectory=False)

            bci_to_episodes = defaultdict(list)

            for episode in data["episodes"]:
                bci_to_episodes[episode["build_config_idx"]].append(episode)

            new_episodes_list = [None] * len(data["episodes"])
            pbar = tqdm(len(data["episodes"]), total=len(data["episodes"]))
            for bci, episodes in bci_to_episodes.items():
                pbar.set_description(f"Eps with {bci=}", refresh=False)
                uenv.reset(
                    options=dict(
                        reconfigure=True,
                        build_config_idxs=[bci],
                    )
                )
                for episode in episodes:
                    assert bci == episode["build_config_idx"]

                    uenv.reset(
                        options=dict(
                            reconfigure=False,
                            task_plan_idxs=torch.tensor([episode["task_plan_idx"]]),
                            spawn_selection_idxs=[episode["spawn_selection_idx"]],
                        )
                    )

                    assert (
                        uenv.build_config_idxs[0] == episode["build_config_idx"]
                    ), f"{uenv.build_config_idxs[0]} != {episode['build_config_idx']}"
                    assert (
                        uenv.task_plan_idxs.tolist()[0] == episode["task_plan_idx"]
                    ), f"{uenv.task_plan_idxs.tolist()[0]} != {episode['task_plan_idx']}"
                    # assert uenv.init_config_idxs[0] == episode["init_config_idx"]
                    assert (
                        uenv.spawn_selection_idxs[0] == episode["spawn_selection_idx"]
                    ), f"{uenv.spawn_selection_idxs[0]} != {episode['spawn_selection_idx']}"

                    sampled_task_plans = [
                        uenv.build_config_idx_to_task_plans[bci][tpi]
                        for bci, tpi in zip(uenv.build_config_idxs, uenv.task_plan_idxs)
                    ]
                    episode_id = episode["episode_id"]
                    subtask_uid = sampled_task_plans[0].subtasks[0].uid
                    new_episode = {
                        "episode_id": episode_id,
                        "subtask_uid": subtask_uid,
                        **episode,
                    }
                    new_episode["episode_seed"] = NoIndent(episode["episode_seed"])
                    new_episode["events"] = NoIndent(episode["events"])
                    new_episode["events_verbose"] = NoIndent(episode["events_verbose"])

                    # # Get trajectory data and replay actions
                    # traj_group = traj_data[f"traj_{episode_id}"]
                    # actions = traj_group["actions"][:]
                    # for action in actions:
                    #     obs, reward, term, trunc, info = env.step(action)

                    new_episodes_list[episode_id] = new_episode
                    data["episodes"] = new_episodes_list
                    dump_json(
                        new_fp,
                        data,
                        encoder_cls=NoIndentSupportingJSONEncoder,
                        indent=2,
                    )
                    pbar.update(1)

            assert None not in new_episodes_list
            assert all(["subtask_uid" in ep for ep in new_episodes_list])

            uenv.close()
            # traj_data.close()

            data["episodes"] = new_episodes_list
            dump_json(new_fp, data, encoder_cls=NoIndentSupportingJSONEncoder, indent=2)