"""Prepare or execute a compiled VLM route with the existing MS-HAB evaluator."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from mshab.experiments.granularity.vlm.run import (
    DEFAULT_ASSET_ROOT,
    TASK_FAMILIES,
    _read_json,
    official_task_plan_path,
)
from mshab.experiments.granularity.vlm.simulator import (
    FlowProgram,
    MSHabPlanGrounder,
    route_filename,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXPERIMENT_ROOT = REPOSITORY_ROOT.parents[1] / "mshab_exps"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_manifest.json"
    return _read_json(path) if path.exists() else {}


def evaluation_command(
    *,
    plan_path: Path,
    plan_data: Mapping[str, Any],
    task_family: str,
    run_name: str,
    seed: int,
    max_trajectories: int,
    num_envs: int,
    experiment_root: Path,
    record_video: bool,
) -> List[str]:
    selection = plan_data["selection"]
    return [
        sys.executable,
        "-m",
        "mshab.evaluate",
        str(REPOSITORY_ROOT / "configs" / "evaluate.yml"),
        "seed={}".format(seed),
        "task={}".format(task_family),
        "policy_type={}".format(selection["recommended_policy_type"]),
        "save_trajectory=False",
        "max_trajectories={}".format(max_trajectories),
        "eval_env.env_id=SequentialTask-v0",
        "eval_env.task_plan_fp={}".format(plan_path.resolve()),
        "eval_env.spawn_data_fp=null",
        "eval_env.num_envs={}".format(num_envs),
        "eval_env.max_episode_steps={}".format(
            selection["estimated_max_episode_steps"]
        ),
        "eval_env.continuous_task=False",
        "eval_env.frame_stack=3",
        "eval_env.stack=null",
        "eval_env.record_video={}".format(str(record_video)),
        "eval_env.info_on_video=True",
        "eval_env.save_video_freq=1",
        (
            "eval_env.extra_stat_keys=<list>success,fail,subtask,"
            "subtask_type,subtasks_steps_left,robot_force,"
            "robot_cumulative_force</list>"
        ),
        "eval_env.env_kwargs.invisible_goals_in_human_render=False",
        "eval_env.env_kwargs.task_cfgs.navigate.ignore_arm_checkers=True",
        "logger.workspace={}".format(experiment_root.resolve()),
        "logger.exp_name=vlm-plans/{}".format(run_name),
        "logger.clear_out=False",
        "logger.tensorboard=True",
        "logger.wandb=False",
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="A VLM output directory containing predicted_flow.json.",
    )
    parser.add_argument("--flow", type=Path)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--task-family", choices=TASK_FAMILIES)
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--plan-index", type=int)
    parser.add_argument(
        "--route",
        default="nominal",
        help="nominal or fallback.<occurrence_id>",
    )
    parser.add_argument("--list-routes", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trajectories", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--no-video", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = _parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    manifest = _load_run_manifest(run_dir)
    task_family = args.task_family or manifest.get("task_family")
    if task_family not in TASK_FAMILIES:
        raise ValueError(
            "task family is unavailable; pass --task-family explicitly"
        )
    split = args.split or manifest.get("split") or "val"
    plan_index = (
        args.plan_index
        if args.plan_index is not None
        else manifest.get("gt_plan_index", 0)
    )
    if plan_index is None:
        plan_index = 0

    flow_path = (args.flow or run_dir / "predicted_flow.json").resolve()
    source_path = args.source_plan
    if source_path is None and manifest.get("gt"):
        source_path = Path(manifest["gt"])
    if source_path is None:
        source_path = official_task_plan_path(
            task_family,
            split,
            Path(os.environ.get("MS_ASSET_DIR", str(DEFAULT_ASSET_ROOT))),
        )
    source_path = source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            "official source plan not found: {}".format(source_path)
        )

    program = FlowProgram(_read_json(flow_path))
    if args.list_routes:
        for route in program.routes():
            print(
                "{:<48} {:<9} {} steps".format(
                    route.id,
                    route.kind,
                    len(route.node_order),
                )
            )
        return

    route = program.route(args.route)
    grounder = MSHabPlanGrounder(
        _read_json(source_path),
        task_family=task_family,
        plan_index=int(plan_index),
    )
    plan_data = grounder.compile(program, route)
    output_dir = (args.output_dir or run_dir / "simulator").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / route_filename(route.id)
    _write_json(plan_path, plan_data)

    run_name = "{}/{}".format(run_dir.name, route_filename(route.id)[:-5])
    command = evaluation_command(
        plan_path=plan_path,
        plan_data=plan_data,
        task_family=task_family,
        run_name=run_name,
        seed=args.seed,
        max_trajectories=args.max_trajectories,
        num_envs=args.num_envs,
        experiment_root=args.experiment_root,
        record_video=not args.no_video,
    )
    execution = {
        "flow": str(flow_path),
        "source_plan": str(source_path),
        "source_plan_index": int(plan_index),
        "route_id": route.id,
        "task_plan": str(plan_path),
        "command": command,
        "shell_preview": "SAPIEN_NO_DISPLAY=1 {}".format(shlex.join(command)),
    }
    _write_json(output_dir / "last_execution.json", execution)
    print("wrote simulator-ready PlanData to {}".format(plan_path))
    print(execution["shell_preview"])

    if args.run:
        environment = os.environ.copy()
        environment["SAPIEN_NO_DISPLAY"] = "1"
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
