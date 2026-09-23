"""Run the full SetTable graph-constrained planning and execution experiment.

The experiment is resumable and produces two primary tables:

1. Planning: GT versus DeepSeek + SkillGraph.
2. Execution: GT, VLM nominal, and six canonical fallback routes.

Planning makes exactly one model call per instruction. Scene grounding and
simulator execution never call the model again.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from mshab.experiments.granularity.vlm.execute import evaluation_command
from mshab.experiments.granularity.vlm.prompt import build_planner_messages
from mshab.experiments.granularity.vlm.run import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_GRAPH,
    DeepSeekClient,
    DeepSeekConfig,
    FourLayerGraphIndex,
    _read_json,
    _selected_plan,
    _validate_gt_task_family,
    compare_flows,
    normalize_ground_truth,
    official_task_plan_path,
    render_flowchart,
)
from mshab.experiments.granularity.vlm.simulator import (
    FlowProgram,
    MSHabPlanGrounder,
    SUBTASK_HORIZONS,
    route_filename,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = PACKAGE_DIR / "set_table_experiment.yaml"
DEFAULT_OUTPUT = PACKAGE_DIR / "outputs" / "set_table_full_experiment"
EXPERIMENT_SCHEMA_VERSION = "mshab.set-table-graph-experiment.v1"


@dataclass(frozen=True)
class InstructionCase:
    id: str
    text: str


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    task_family: str
    split: str
    plan_indices: tuple[int, ...]
    canonical_instruction_id: str
    seed: int
    instructions: tuple[InstructionCase, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentSpec":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("unsupported experiment schema_version")
        instructions = tuple(
            InstructionCase(id=str(item["id"]), text=str(item["text"]).strip())
            for item in payload["instructions"]
        )
        ids = [item.id for item in instructions]
        if len(instructions) != 10 or len(ids) != len(set(ids)):
            raise ValueError("SetTable experiment requires 10 unique instructions")
        if any(not item.text for item in instructions):
            raise ValueError("experiment instructions must be non-empty")
        plan_indices = tuple(int(value) for value in payload["plan_indices"])
        if plan_indices != tuple(range(10)):
            raise ValueError("SetTable experiment requires plan-index 0 through 9")
        canonical = str(payload["canonical_instruction_id"])
        if canonical not in ids:
            raise ValueError("canonical instruction is not defined")
        task_family = str(payload["task_family"])
        if task_family != "set_table":
            raise ValueError("this experiment is specific to set_table")
        return cls(
            name=str(payload["name"]),
            task_family=task_family,
            split=str(payload["split"]),
            plan_indices=plan_indices,
            canonical_instruction_id=canonical,
            seed=int(payload.get("seed", 0)),
            instructions=instructions,
        )


@dataclass(frozen=True)
class ExecutionJob:
    id: str
    condition: str
    label: str
    instruction_id: Optional[str]
    plan_index: int
    route_id: str
    groundable: bool
    task_plan: Optional[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "condition": self.condition,
            "label": self.label,
            "instruction_id": self.instruction_id,
            "plan_index": self.plan_index,
            "route_id": self.route_id,
            "groundable": self.groundable,
            "task_plan": self.task_plan,
            "error": self.error,
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(
    values: Iterable[Optional[float]],
    *,
    missing_as_zero: bool = False,
) -> Optional[float]:
    materialized = list(values)
    if missing_as_zero:
        return sum(float(value or 0.0) for value in materialized) / max(
            len(materialized),
            1,
        )
    numeric = [float(value) for value in materialized if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _format_metric(value: Optional[float]) -> str:
    return "Pending" if value is None else "{:.3f}".format(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")


def _parse_evaluator_log(text: str) -> Dict[str, Optional[float]]:
    def number(name: str) -> Optional[float]:
        match = re.search(
            r"['\"]{}['\"]:\s*tensor\(([-+0-9.eE]+)".format(name),
            text,
        )
        return float(match.group(1)) if match else None

    return {
        "simulator_success": number("success_once"),
        "success_at_end": number("success_at_end"),
        "episode_steps": number("len"),
    }


class SetTableExperiment:
    """Resumable orchestrator for the complete 10-instruction experiment."""

    def __init__(
        self,
        *,
        spec: ExperimentSpec,
        output_dir: Path,
        graph_path: Path,
        config_path: Path,
        asset_root: Path,
    ) -> None:
        self.spec = spec
        self.output_dir = output_dir.resolve()
        self.graph_path = graph_path.resolve()
        self.config_path = config_path.resolve()
        self.asset_root = asset_root.resolve()
        self.gt_path = official_task_plan_path(
            spec.task_family,
            spec.split,
            self.asset_root,
        ).resolve()
        self.planning_dir = self.output_dir / "planning"
        self.plans_dir = self.output_dir / "execution_plans"
        self.status_dir = self.output_dir / "execution_status"
        self.logs_dir = self.output_dir / "execution_logs"
        self.simulator_workspace = self.output_dir / "simulator_runs"
        self.tables_dir = self.output_dir / "tables"

    def initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.graph_path.is_file():
            raise FileNotFoundError("graph not found: {}".format(self.graph_path))
        if not self.gt_path.is_file():
            raise FileNotFoundError("GT not found: {}".format(self.gt_path))
        _write_json(
            self.output_dir / "experiment_manifest.json",
            {
                "schema_version": EXPERIMENT_SCHEMA_VERSION,
                "name": self.spec.name,
                "task_family": self.spec.task_family,
                "split": self.spec.split,
                "plan_indices": list(self.spec.plan_indices),
                "canonical_instruction_id": self.spec.canonical_instruction_id,
                "seed": self.spec.seed,
                "instructions": [
                    {"id": item.id, "text": item.text}
                    for item in self.spec.instructions
                ],
                "graph": str(self.graph_path),
                "ground_truth": str(self.gt_path),
                "design": {
                    "deepseek_calls": 10,
                    "gt_execution_cases": 10,
                    "vlm_nominal_execution_cases": 100,
                    "fallback_execution_cases": 60,
                    "total_execution_cases": 170,
                },
            },
        )

    def _case_dir(self, case: InstructionCase) -> Path:
        return self.planning_dir / case.id

    def planning(self, *, force: bool = False, fail_fast: bool = False) -> None:
        self.initialize()
        graph = _read_json(self.graph_path)
        index = FourLayerGraphIndex(graph)
        config = DeepSeekConfig.from_yaml(self.config_path)
        config.resolved_api_key()
        client = DeepSeekClient(config)
        scene_context = {
            "task_family": self.spec.task_family,
            "objects": [],
            "articulations": [],
            "source": "none",
            "note": "No GT-derived scene context is supplied to the planner.",
        }

        for position, case in enumerate(self.spec.instructions, start=1):
            case_dir = self._case_dir(case)
            status_path = case_dir / "status.json"
            if not force and status_path.exists():
                prior = _read_json(status_path)
                if prior.get("status") == "complete":
                    print(
                        "planning {}/10 {} already complete".format(
                            position,
                            case.id,
                        )
                    )
                    continue
            case_dir.mkdir(parents=True, exist_ok=True)
            print("planning {}/10 {}".format(position, case.id))
            messages = build_planner_messages(
                instruction=case.text,
                task_family=self.spec.task_family,
                scene_context=scene_context,
                graph=graph,
            )
            _write_json(
                case_dir / "planner_request.json",
                config.request_payload(messages),
            )
            _write_json(case_dir / "scene_context.json", scene_context)
            started = time.time()
            api_metadata: Optional[Dict[str, Any]] = None
            try:
                decision, api_metadata = client.complete(messages)
                # Persist the model output before graph compilation.  Invalid
                # decisions are experiment evidence and must not disappear
                # merely because the strict compiler rejects them.
                _write_json(case_dir / "strategy_decision.json", decision)
                _write_json(case_dir / "planner_api_metadata.json", api_metadata)
                flow = index.compile_decision(
                    decision,
                    case.text,
                    self.spec.task_family,
                    scene_context,
                )

                # GT is loaded only after the model response has been compiled.
                gt_source = _read_json(self.gt_path)
                _validate_gt_task_family(gt_source, self.spec.task_family)
                gt_flow = normalize_ground_truth(
                    gt_source,
                    plan_index=0,
                    instruction=case.text,
                    task_family=self.spec.task_family,
                )
                metrics = compare_flows(flow, gt_flow)
                _write_json(case_dir / "predicted_flow.json", flow)
                _write_json(case_dir / "ground_truth_flow.json", gt_flow)
                _write_json(case_dir / "deterministic_metrics.json", metrics)
                render_flowchart(flow, case_dir)
                _write_json(
                    status_path,
                    {
                        "status": "complete",
                        "instruction_id": case.id,
                        "graph_valid": True,
                        "elapsed_seconds": time.time() - started,
                        "api": api_metadata,
                    },
                )
            except Exception as exc:
                _write_json(
                    status_path,
                    {
                        "status": "failed",
                        "instruction_id": case.id,
                        "graph_valid": False,
                        "elapsed_seconds": time.time() - started,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "api": api_metadata,
                    },
                )
                if fail_fast:
                    raise
                print("planning {} failed: {}".format(case.id, exc))
        self.summarize()

    @staticmethod
    def _gt_plan_data(
        source: Mapping[str, Any],
        *,
        plan_index: int,
    ) -> Dict[str, Any]:
        plan = copy.deepcopy(_selected_plan(source, plan_index))
        subtasks = plan["subtasks"]
        return {
            "dataset": source.get("dataset"),
            "plans": [plan],
            "selection": {
                "task_family": "set_table",
                "source_plan_index": plan_index,
                "route_id": "gt.nominal",
                "route_kind": "ground_truth",
                "recommended_policy_type": "rl_per_obj",
                "estimated_max_episode_steps": sum(
                    SUBTASK_HORIZONS[item["type"]] for item in subtasks
                ),
            },
        }

    @staticmethod
    def _fallback_label(
        program: FlowProgram,
        route_id: str,
    ) -> str:
        route = program.route(route_id)
        if route.failed_node_id is None:
            return "Nominal"
        node = program.nodes[route.failed_node_id]
        contract_type = str(node["contract_type"])
        arguments = node.get("arguments", {})
        if contract_type in ("pick", "place"):
            target = str(arguments.get("object", "object"))
            object_name = "Bowl" if "bowl" in target else "Apple"
            return "{} {} Fallback".format(object_name, contract_type.title())
        articulation = str(arguments.get("articulation", "container"))
        container = "Drawer" if articulation == "kitchen_counter" else "Fridge"
        return "{} Close Fallback".format(container)

    def _write_plan(self, relative_path: Path, payload: Mapping[str, Any]) -> str:
        path = self.output_dir / relative_path
        _write_json(path, payload)
        return str(relative_path)

    def prepare_execution(self) -> List[ExecutionJob]:
        self.initialize()
        source = _read_json(self.gt_path)
        _validate_gt_task_family(source, self.spec.task_family)
        jobs: List[ExecutionJob] = []

        for plan_index in self.spec.plan_indices:
            relative = Path("execution_plans") / "gt" / "plan_{:03d}.json".format(
                plan_index
            )
            task_plan = self._write_plan(
                relative,
                self._gt_plan_data(source, plan_index=plan_index),
            )
            jobs.append(
                ExecutionJob(
                    id="gt.plan_{:03d}".format(plan_index),
                    condition="gt_nominal",
                    label="GT Nominal",
                    instruction_id=None,
                    plan_index=plan_index,
                    route_id="gt.nominal",
                    groundable=True,
                    task_plan=task_plan,
                )
            )

        canonical_program: Optional[FlowProgram] = None
        for case in self.spec.instructions:
            flow_path = self._case_dir(case) / "predicted_flow.json"
            if not flow_path.exists():
                for plan_index in self.spec.plan_indices:
                    jobs.append(
                        ExecutionJob(
                            id="vlm.{}.plan_{:03d}".format(case.id, plan_index),
                            condition="vlm_nominal",
                            label="VLM Nominal",
                            instruction_id=case.id,
                            plan_index=plan_index,
                            route_id="nominal",
                            groundable=False,
                            task_plan=None,
                            error="planning output is unavailable",
                        )
                    )
                continue
            program = FlowProgram(_read_json(flow_path))
            if case.id == self.spec.canonical_instruction_id:
                canonical_program = program
            for plan_index in self.spec.plan_indices:
                job_id = "vlm.{}.plan_{:03d}".format(case.id, plan_index)
                try:
                    plan = MSHabPlanGrounder(
                        source,
                        task_family=self.spec.task_family,
                        plan_index=plan_index,
                    ).compile(program, program.route("nominal"))
                    relative = (
                        Path("execution_plans")
                        / "vlm_nominal"
                        / case.id
                        / "plan_{:03d}.json".format(plan_index)
                    )
                    task_plan = self._write_plan(relative, plan)
                    jobs.append(
                        ExecutionJob(
                            id=job_id,
                            condition="vlm_nominal",
                            label="VLM Nominal",
                            instruction_id=case.id,
                            plan_index=plan_index,
                            route_id="nominal",
                            groundable=True,
                            task_plan=task_plan,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    jobs.append(
                        ExecutionJob(
                            id=job_id,
                            condition="vlm_nominal",
                            label="VLM Nominal",
                            instruction_id=case.id,
                            plan_index=plan_index,
                            route_id="nominal",
                            groundable=False,
                            task_plan=None,
                            error=str(exc),
                        )
                    )

        if canonical_program is None:
            raise RuntimeError(
                "canonical planning output is required to prepare fallback routes"
            )
        fallback_routes = [
            route
            for route in canonical_program.routes()
            if route.kind == "fallback"
        ]
        if len(fallback_routes) != 6:
            raise ValueError(
                "canonical flow must expose exactly six fallback routes"
            )
        for route in fallback_routes:
            label = self._fallback_label(canonical_program, route.id)
            condition = "fallback.{}".format(_safe_id(label.lower()))
            for plan_index in self.spec.plan_indices:
                job_id = "{}.plan_{:03d}".format(condition, plan_index)
                try:
                    plan = MSHabPlanGrounder(
                        source,
                        task_family=self.spec.task_family,
                        plan_index=plan_index,
                    ).compile(canonical_program, route)
                    relative = (
                        Path("execution_plans")
                        / "fallback"
                        / route_filename(route.id).removesuffix(".json")
                        / "plan_{:03d}.json".format(plan_index)
                    )
                    task_plan = self._write_plan(relative, plan)
                    jobs.append(
                        ExecutionJob(
                            id=job_id,
                            condition=condition,
                            label=label,
                            instruction_id=self.spec.canonical_instruction_id,
                            plan_index=plan_index,
                            route_id=route.id,
                            groundable=True,
                            task_plan=task_plan,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    jobs.append(
                        ExecutionJob(
                            id=job_id,
                            condition=condition,
                            label=label,
                            instruction_id=self.spec.canonical_instruction_id,
                            plan_index=plan_index,
                            route_id=route.id,
                            groundable=False,
                            task_plan=None,
                            error=str(exc),
                        )
                    )

        if len(jobs) != 170:
            raise AssertionError(
                "expected 170 execution jobs, got {}".format(len(jobs))
            )
        payload = [job.to_dict() for job in jobs]
        _write_json(self.output_dir / "execution_jobs.json", payload)
        _write_csv(self.output_dir / "execution_jobs.csv", payload)
        self.summarize()
        print("prepared {} execution jobs".format(len(jobs)))
        return jobs

    def _load_jobs(self) -> List[ExecutionJob]:
        path = self.output_dir / "execution_jobs.json"
        if not path.exists():
            raise FileNotFoundError("run --phase prepare before execution")
        return [ExecutionJob(**item) for item in json.loads(path.read_text())]

    def execute(
        self,
        *,
        max_jobs: Optional[int],
        condition: Optional[str],
        instruction_id: Optional[str],
        record_video: bool,
        retry_failed: bool,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not visible to PyTorch; MS-HAB execution cannot start"
            )
        jobs = self._load_jobs()
        selected = [
            job
            for job in jobs
            if job.groundable
            and (condition is None or job.condition == condition)
            and (instruction_id is None or job.instruction_id == instruction_id)
        ]
        completed_this_run = 0
        for position, job in enumerate(selected, start=1):
            status_path = self.status_dir / "{}.json".format(job.id)
            if status_path.exists():
                prior = _read_json(status_path)
                if prior.get("status") == "complete" or (
                    prior.get("status") == "failed" and not retry_failed
                ):
                    continue
            if max_jobs is not None and completed_this_run >= max_jobs:
                break
            plan_path = self.output_dir / str(job.task_plan)
            plan_data = _read_json(plan_path)
            command = evaluation_command(
                plan_path=plan_path,
                plan_data=plan_data,
                task_family=self.spec.task_family,
                run_name="{}/{}".format(self.spec.name, job.id),
                seed=self.spec.seed,
                max_trajectories=1,
                num_envs=1,
                experiment_root=self.simulator_workspace,
                record_video=record_video,
            )
            log_path = self.logs_dir / "{}.log".format(job.id)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                "execution {}/{} {}".format(
                    position,
                    len(selected),
                    job.id,
                )
            )
            started = time.time()
            _write_json(
                status_path,
                {
                    "status": "running",
                    "job": job.to_dict(),
                    "command": command,
                },
            )
            environment = os.environ.copy()
            environment["SAPIEN_NO_DISPLAY"] = "1"
            environment.setdefault("MPLCONFIGDIR", "/tmp/mshab-matplotlib")
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            parsed = _parse_evaluator_log(log_path.read_text(encoding="utf-8"))
            _write_json(
                status_path,
                {
                    "status": "complete" if process.returncode == 0 else "failed",
                    "job": job.to_dict(),
                    "returncode": process.returncode,
                    "elapsed_seconds": time.time() - started,
                    "log": str(log_path.relative_to(self.output_dir)),
                    **parsed,
                },
            )
            completed_this_run += 1
            self.summarize()

    def _planning_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for case in self.spec.instructions:
            case_dir = self._case_dir(case)
            status = (
                _read_json(case_dir / "status.json")
                if (case_dir / "status.json").exists()
                else {"status": "pending", "graph_valid": False}
            )
            metrics = (
                _read_json(case_dir / "deterministic_metrics.json")
                if (case_dir / "deterministic_metrics.json").exists()
                else {}
            )
            rows.append(
                {
                    "instruction_id": case.id,
                    "instruction": case.text,
                    "status": status.get("status"),
                    "graph_valid": bool(status.get("graph_valid", False)),
                    "lcs_recall": metrics.get("lcs_recall"),
                    "grounding_accuracy": metrics.get("grounded_target_accuracy"),
                    "edit_distance": metrics.get("edit_distance"),
                    "predicted_length": metrics.get("predicted_length"),
                    "gt_length": metrics.get("gt_length"),
                }
            )
        return rows

    def _execution_rows(self) -> List[Dict[str, Any]]:
        jobs_path = self.output_dir / "execution_jobs.json"
        if not jobs_path.exists():
            return []
        rows = []
        for item in json.loads(jobs_path.read_text(encoding="utf-8")):
            status_path = self.status_dir / "{}.json".format(item["id"])
            status = _read_json(status_path) if status_path.exists() else {}
            execution_status = status.get("status", "pending")
            if not item["groundable"]:
                execution_status = "not_groundable"
            rows.append(
                {
                    **item,
                    "execution_status": execution_status,
                    "simulator_success": status.get("simulator_success"),
                    "success_at_end": status.get("success_at_end"),
                    "episode_steps": status.get("episode_steps"),
                    "elapsed_seconds": status.get("elapsed_seconds"),
                }
            )
        return rows

    def summarize(self) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        planning_rows = self._planning_rows()
        execution_rows = self._execution_rows()
        _write_csv(self.tables_dir / "planning_results.csv", planning_rows)
        if execution_rows:
            _write_csv(self.tables_dir / "execution_results.csv", execution_rows)

        gt_execution = [
            row for row in execution_rows if row["condition"] == "gt_nominal"
        ]
        vlm_execution = [
            row for row in execution_rows if row["condition"] == "vlm_nominal"
        ]
        gt_success = self._end_to_end_success(gt_execution)
        vlm_success = self._end_to_end_success(vlm_execution)
        planning_terminal = all(
            row["status"] in ("complete", "failed") for row in planning_rows
        )
        planning_table = [
            {
                "method": "GT Plan",
                "graph_valid_rate": None,
                "lcs_recall": 1.0,
                "grounding_accuracy": 1.0,
                "simulator_success": gt_success,
            },
            {
                "method": "DeepSeek + Skill Graph",
                "graph_valid_rate": (
                    _mean(
                        [
                            1.0 if row["graph_valid"] else 0.0
                            for row in planning_rows
                        ]
                    )
                    if planning_terminal
                    else None
                ),
                "lcs_recall": (
                    _mean(
                        [row["lcs_recall"] for row in planning_rows],
                        missing_as_zero=True,
                    )
                    if planning_terminal
                    else None
                ),
                "grounding_accuracy": (
                    _mean(
                        [row["grounding_accuracy"] for row in planning_rows],
                        missing_as_zero=True,
                    )
                    if planning_terminal
                    else None
                ),
                "simulator_success": vlm_success,
            },
        ]
        _write_csv(self.tables_dir / "planning_table.csv", planning_table)

        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for row in execution_rows:
            grouped.setdefault(row["condition"], []).append(row)
        execution_table = []
        order = ["gt_nominal", "vlm_nominal"] + sorted(
            key for key in grouped if key.startswith("fallback.")
        )
        for condition_name in order:
            rows = grouped.get(condition_name, [])
            if not rows:
                continue
            execution_table.append(
                {
                    "route": rows[0]["label"],
                    "cases": len(rows),
                    "groundable_rate": sum(bool(row["groundable"]) for row in rows)
                    / len(rows),
                    "simulator_success": self._end_to_end_success(rows),
                    "completed_cases": sum(
                        row["execution_status"] == "complete" for row in rows
                    ),
                }
            )
        if execution_table:
            _write_csv(self.tables_dir / "execution_table.csv", execution_table)

        planning_md = _markdown_table(
            [
                "Method",
                "Graph-valid Rate ↑",
                "LCS Recall ↑",
                "Grounding Accuracy ↑",
                "Simulator Success ↑",
            ],
            [
                [
                    row["method"],
                    (
                        "—"
                        if row["method"] == "GT Plan"
                        else _format_metric(row["graph_valid_rate"])
                    ),
                    _format_metric(row["lcs_recall"]),
                    _format_metric(row["grounding_accuracy"]),
                    _format_metric(row["simulator_success"]),
                ]
                for row in planning_table
            ],
        )
        execution_md = _markdown_table(
            [
                "Route",
                "Cases",
                "Groundable Rate ↑",
                "Simulator Success ↑",
                "Completed",
            ],
            [
                [
                    row["route"],
                    row["cases"],
                    _format_metric(row["groundable_rate"]),
                    _format_metric(row["simulator_success"]),
                    "{}/{}".format(row["completed_cases"], row["cases"]),
                ]
                for row in execution_table
            ],
        )
        (self.tables_dir / "planning_table.md").write_text(
            planning_md + "\n",
            encoding="utf-8",
        )
        (self.tables_dir / "execution_table.md").write_text(
            execution_md + "\n",
            encoding="utf-8",
        )
        (self.tables_dir / "report.md").write_text(
            "# SetTable Graph-Constrained Planning and Execution\n\n"
            "## Table 1 — Planning and end-to-end nominal execution\n\n"
            + planning_md
            + "\n\n## Table 2 — Scene execution and fallback routes\n\n"
            + execution_md
            + "\n\nSimulator Success is end-to-end: ungroundable or failed cases "
            "count as failures. Pending means the GPU job has not completed.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _end_to_end_success(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not rows:
            return None
        if any(
            row["execution_status"]
            not in ("complete", "failed", "not_groundable")
            for row in rows
        ):
            return None
        return sum(float(row.get("simulator_success") or 0.0) for row in rows) / len(
            rows
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("planning", "prepare", "execute", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("MS_ASSET_DIR", str(DEFAULT_ASSET_ROOT))),
    )
    parser.add_argument("--force-planning", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--condition")
    parser.add_argument("--instruction-id")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    spec = ExperimentSpec.from_yaml(args.manifest)
    experiment = SetTableExperiment(
        spec=spec,
        output_dir=args.output_dir,
        graph_path=args.graph,
        config_path=args.config,
        asset_root=args.asset_root,
    )
    experiment.initialize()
    if args.phase in ("planning", "all"):
        experiment.planning(
            force=args.force_planning,
            fail_fast=args.fail_fast,
        )
    if args.phase in ("prepare", "all"):
        experiment.prepare_execution()
    if args.phase in ("execute", "all"):
        experiment.execute(
            max_jobs=args.max_jobs,
            condition=args.condition,
            instruction_id=args.instruction_id,
            record_video=args.record_video,
            retry_failed=args.retry_failed,
        )
    experiment.summarize()
    print("report: {}".format(experiment.tables_dir / "report.md"))


if __name__ == "__main__":
    main()
