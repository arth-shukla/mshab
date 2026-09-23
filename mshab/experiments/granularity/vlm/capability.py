"""Capability study: can DeepSeek plan over the coarse skill graph, and does
the resulting plan execute in MS-HAB?

This replaces the earlier flat-vs-graph ablation, which could not answer its
own question for two reasons found during review:

1. the "flat" view still exposed ``subgoal_id``, ``achieves`` and the full
   Layer-3 contracts, so the withheld Layer-2 edges were recoverable by
   precondition chaining -- the conditions scored identically;
2. plan correctness was decided by one hard-coded segment order, which marks
   valid alternative routes wrong, and by an edge-following rule whose
   information the flat condition never received.

Here a single condition is measured against two independent criteria:

``graph_valid``  the path follows Layer-2 edges and segment ownership
``goal_reached`` contract simulation from the initial state satisfies the goal

Stage 1 (planning) needs only the API.  Stage 2 (execution) runs the grounded
plan in MS-HAB, paired against the GT oracle on the same episode and seed so
that controller noise cancels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from mshab.experiments.granularity.vlm.prompt import build_path_planner_messages
from mshab.experiments.granularity.vlm.execute import evaluation_command
from mshab.experiments.granularity.vlm.run import (
    DeepSeekClient,
    DeepSeekConfig,
    official_task_plan_path,
)
from mshab.experiments.granularity.vlm.set_table_graph_ablation import (
    AblationCase,
    ExplicitPathIndex,
    PATH_DECISION_SCHEMA_VERSION,
)
from mshab.experiments.granularity.vlm.simulator import (
    FlowProgram,
    MSHabPlanGrounder,
)
from mshab.experiments.granularity.vlm.symbolic import (
    ContractSimulator,
    OUTCOME_ALTERNATIVE,
    OUTCOME_API,
    OUTCOME_EXACT,
    OUTCOME_ORDER,
    OUTCOME_SCHEMA,
    OUTCOME_STRUCTURE,
    TaskSemantics,
    VALID_OUTCOMES,
    lcs_f1,
    plan_from_decision,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
GRAPH_DOCUMENT = (
    REPO_ROOT
    / "mshab"
    / "experiments"
    / "granularity"
    / "artifacts"
    / "coarse_four_layers.json"
)
CONDITION = "full_graph"

SET_TABLE_SEMANTICS = TaskSemantics(
    initial_facts={
        "present(kitchen_counter)",
        "present(fridge)",
        "present(dining_table)",
        "present(countertop)",
        "present(024_bowl)",
        "present(013_apple)",
        "closed(kitchen_counter)",
        "closed(fridge)",
        "gripper_empty()",
    },
    goal_predicates={
        "at(024_bowl,dining_table)",
        "closed(kitchen_counter)",
        "at(013_apple,dining_table)",
        "closed(fridge)",
    },
    vocabulary=(
        "024_bowl",
        "013_apple",
        "kitchen_counter",
        "fridge",
        "dining_table",
        "countertop",
    ),
)

GT_SEGMENTS: Tuple[Tuple[str, Dict[str, str], Tuple[str, ...]], ...] = (
    (
        "retrieve_bowl",
        {"object": "024_bowl"},
        (
            "navigate_drawer_source",
            "open_drawer_source",
            "navigate_drawer_object",
            "pick_drawer_object",
        ),
    ),
    ("deliver_bowl", {"object": "024_bowl"}, ("navigate_table", "place_on_table")),
    ("restore_drawer", {}, ("navigate_drawer_for_restore", "close_drawer")),
    (
        "retrieve_apple",
        {"object": "013_apple"},
        (
            "navigate_fridge_source",
            "open_fridge_source",
            "navigate_fridge_object",
            "pick_fridge_object",
        ),
    ),
    ("deliver_apple", {"object": "013_apple"}, ("navigate_table", "place_on_table")),
    ("restore_fridge", {}, ("navigate_fridge_for_restore", "close_fridge")),
)

#: Recovery route per segment, taken from the Layer-2 FALLBACK_TO chains.
#: Each replaces the failed primary achiever and re-enters from the branch
#: point its subgraph declares (``open_*_source`` / ``reapproach_*`` /
#: ``reposition_*``), so the failed SkillNode never appears in the path.
FALLBACK_PATHS: Dict[str, Tuple[str, ...]] = {
    "retrieve_bowl": (
        "navigate_drawer_source",
        "open_drawer_source",
        "relocalize_drawer_object",
        "regrasp_drawer_object",
    ),
    "retrieve_apple": (
        "navigate_fridge_source",
        "open_fridge_source",
        "relocalize_fridge_object",
        "regrasp_fridge_object",
    ),
    "deliver_bowl": ("reapproach_table", "recover_place_on_table"),
    "deliver_apple": ("reapproach_table", "recover_place_on_table"),
    "restore_drawer": ("reposition_drawer", "recover_close_drawer"),
    "restore_fridge": ("reposition_fridge", "recover_close_fridge"),
}

SUBGOAL_OF = {
    "retrieve_bowl": "retrieve",
    "deliver_bowl": "deliver",
    "restore_drawer": "restore",
    "retrieve_apple": "retrieve",
    "deliver_apple": "deliver",
    "restore_fridge": "restore",
}


def gt_decision(case: AblationCase) -> Dict[str, Any]:
    """The reference plan for one case, in the schema the model must emit.

    For a fallback case the single segment named by ``oracle_route_id`` is
    swapped for its recovery route; every other segment is unchanged, so the
    reference differs from the nominal plan by exactly the injected failure.
    """

    replaced = ""
    if case.failed_skill_node_id is not None:
        replaced = case.oracle_route_id.split(".", 1)[-1]
        if replaced not in FALLBACK_PATHS:
            raise KeyError(
                "no recovery route declared for segment {!r}".format(replaced)
            )
    segments = []
    for segment_id, bindings, path in GT_SEGMENTS:
        nodes = FALLBACK_PATHS[segment_id] if segment_id == replaced else path
        segments.append(
            {
                "id": segment_id,
                "subgoal_id": SUBGOAL_OF[segment_id],
                "bindings": dict(bindings),
                "skill_node_path": list(nodes),
            }
        )
    return {
        "schema_version": PATH_DECISION_SCHEMA_VERSION,
        "instruction": case.instruction,
        "task_family": "set_table",
        "condition": CONDITION,
        "failed_skill_node_id": case.failed_skill_node_id,
        "segments": segments,
        "segment_order": [item[0] for item in GT_SEGMENTS],
        "decision_summary": "Reference SetTable route.",
    }


class CapabilityIndex(ExplicitPathIndex):
    """Structural validation without the hard-coded canonical segment order.

    Segment ordering is a semantic question, and it is answered by contract
    simulation instead.  Every other structural rule -- node existence,
    segment ownership, binding completeness, edge following, ending at an
    achiever -- is inherited unchanged.
    """

    def _validate_set_table_order(self, order, segments) -> None:  # noqa: D102
        return

    def library_view(self, condition: str) -> Dict[str, Any]:
        """Flat view withholds achiever labels as well as Layer-2 edges.

        The earlier ablation removed only the edges, and still published
        ``achieves`` plus the full contracts, so a planner could rebuild the
        ordering by chaining preconditions and read the achiever straight off
        the node.  The two conditions scored identically as a result.  Here the
        flat condition must infer *both* the order and which node satisfies a
        SubGoal from contract effects alone.  ``subgoal_id`` has to stay: the
        output schema requires it.  Node names remain a residual leak.
        """

        view = super().library_view(condition)
        if condition == "flat_library":
            view["skill_nodes"] = [
                {key: value for key, value in node.items() if key != "achieves"}
                for node in view["skill_nodes"]
            ]
            view["achiever_labels"] = (
                "Withheld for the flat-library ablation; infer which node "
                "satisfies a SubGoal from its Contract effects."
            )
        return view


@dataclass
class PlanRecord:
    case_id: str
    condition: str
    ability: str
    sample: int
    outcome: str
    graph_valid: bool
    goal_reached: bool
    lcs: Optional[float]
    plan_length: Optional[int]
    error: str = ""
    decision: Optional[Dict[str, Any]] = field(default=None, repr=False)
    symbolic: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def correct(self) -> bool:
        return self.outcome in VALID_OUTCOMES

    def row(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "condition": self.condition,
            "ability": self.ability,
            "sample": self.sample,
            "outcome": self.outcome,
            "graph_valid": self.graph_valid,
            "goal_reached": self.goal_reached,
            "correct": self.correct,
            "lcs_f1": "" if self.lcs is None else round(self.lcs, 4),
            "plan_length": "" if self.plan_length is None else self.plan_length,
            "error": self.error[:300],
        }


class CapabilityExperiment:
    def __init__(self, spec_path: Path, output_root: Path, config_path: Path) -> None:
        spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
        self.spec_name = str(spec.get("name", "capability"))
        self.cases = [
            AblationCase(
                id=str(item["id"]),
                case_type=str(item["case_type"]),
                plan_index=int(item["plan_index"]),
                failed_skill_node_id=item.get("failed_skill_node_id"),
                oracle_route_id=str(item.get("oracle_route_id", "nominal")),
                instruction=" ".join(str(item["instruction"]).split()),
            )
            for item in spec["cases"]
        ]
        self.split = str(spec.get("split", "train"))
        self.root = Path(output_root)
        self.config_path = Path(config_path)
        self.scene_context = spec.get("scene_context")
        self.abilities = {
            str(item["id"]): str(item.get("ability", ""))
            for item in spec["cases"]
        }
        document = json.loads(GRAPH_DOCUMENT.read_text(encoding="utf-8"))
        self.index = CapabilityIndex(document)
        self.library_views = {}
        for name in ("flat_library", "full_graph"):
            view = self.index.library_view(name)
            if self.scene_context is not None:
                view["scene_context"] = copy.deepcopy(self.scene_context)
            self.library_views[name] = view
        self.library_view = self.library_views[CONDITION]
        self.simulator = ContractSimulator(
            {item["contract_type"]: item for item in view["contracts"]},
            {item["id"]: item for item in view["skill_nodes"]},
        )
        self.gt_paths = {
            case.id: [
                node
                for segment in gt_decision(case)["segments"]
                for node in segment["skill_node_path"]
            ]
            for case in self.cases
        }
        asset_root = Path(
            os.environ.get("MS_ASSET_DIR", str(REPO_ROOT.parent / "mshab-assets"))
        ).expanduser()
        gt_path = official_task_plan_path("set_table", self.split, asset_root)
        if not gt_path.is_file():
            raise FileNotFoundError(
                "official SetTable task plan not found: {}".format(gt_path)
            )
        self.official_source = json.loads(gt_path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- stage 1

    def plan(
        self,
        samples: int,
        retries: int,
        conditions: Sequence[str] = (CONDITION,),
        workers: int = 4,
    ) -> None:
        """Sample plans for every (case, condition, sample) cell.

        Calls are independent, so they run on a small thread pool; the model's
        latency, not the GPU, is the bottleneck here.
        """

        config = DeepSeekConfig.from_yaml(self.config_path)
        plan_dir = self.root / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        cells = [
            (case, condition, sample)
            for case in self.cases
            for condition in conditions
            for sample in range(samples)
        ]
        total = len(cells)
        done = 0
        records: List[PlanRecord] = []

        def work(cell):
            case, condition, sample = cell
            messages = build_path_planner_messages(
                instruction=case.instruction,
                task_family="set_table",
                condition=condition,
                failed_skill_node_id=case.failed_skill_node_id,
                library_view=self.library_views[condition],
            )
            client = DeepSeekClient(config)
            return cell, self._plan_one(
                client, case, messages, sample, retries, condition
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for cell, record in pool.map(work, cells):
                case, condition, sample = cell
                done += 1
                records.append(record)
                if record.decision is not None:
                    name = "{}.{}.s{}.json".format(case.id, condition, sample)
                    (plan_dir / name).write_text(
                        json.dumps(record.decision, indent=2) + "\n",
                        encoding="utf-8",
                    )
                print(
                    "  [{:>3}/{}] {} {:13s} s{} -> {}{}".format(
                        done, total, case.id, condition, sample, record.outcome,
                        "" if not record.error else "  ({})".format(record.error[:60]),
                    ),
                    flush=True,
                )
        records.sort(key=lambda item: (item.case_id, item.condition, item.sample))
        self._write_rows(
            self.root / "planning_results.csv", [item.row() for item in records]
        )
        print("wrote", self.root / "planning_results.csv")

    def _plan_one(
        self,
        client: DeepSeekClient,
        case: AblationCase,
        messages: Sequence[Mapping[str, str]],
        sample: int,
        retries: int,
        condition: str = CONDITION,
    ) -> PlanRecord:
        last_error = ""
        for attempt in range(retries + 1):
            try:
                # complete() already returns the parsed JSON object the model
                # emitted, not the OpenAI envelope.
                decision, _ = client.complete(messages)
            except Exception as exc:  # noqa: BLE001 - infrastructure, not model
                last_error = "{}: {}".format(type(exc).__name__, exc)
                time.sleep(2 * (attempt + 1))
                continue
            return self._score(case, decision, sample, condition)
        return PlanRecord(
            case_id=case.id,
            condition=condition,
            ability=self.abilities.get(case.id, ""),
            sample=sample,
            outcome=OUTCOME_API,
            graph_valid=False,
            goal_reached=False,
            lcs=None,
            plan_length=None,
            error=last_error,
        )

    def _score(
        self,
        case: AblationCase,
        decision: Mapping[str, Any],
        sample: int,
        condition: str = CONDITION,
    ) -> PlanRecord:
        graph_valid = True
        structure_error = ""
        try:
            self.index.validate_and_compile(
                decision, case=case, condition=condition
            )
        except Exception as exc:  # noqa: BLE001 - model output, expected to fail
            graph_valid = False
            structure_error = "{}: {}".format(type(exc).__name__, exc)
        try:
            flat = plan_from_decision(decision)
            symbolic = self.simulator.simulate(flat, SET_TABLE_SEMANTICS)
        except Exception as exc:  # noqa: BLE001 - malformed ids or bindings
            return PlanRecord(
                case_id=case.id,
                condition=condition,
                ability=self.abilities.get(case.id, ""),
                sample=sample,
                outcome=OUTCOME_SCHEMA,
                graph_valid=False,
                goal_reached=False,
                lcs=None,
                plan_length=None,
                error="{}: {}".format(type(exc).__name__, exc),
                decision=dict(decision),
            )
        path = [node for node, _ in flat]
        similarity = lcs_f1(path, self.gt_paths[case.id])
        if symbolic.valid and graph_valid:
            outcome = (
                OUTCOME_EXACT
                if path == self.gt_paths[case.id]
                else OUTCOME_ALTERNATIVE
            )
        elif symbolic.valid:
            outcome = OUTCOME_STRUCTURE
        else:
            outcome = symbolic.outcome
        error = structure_error if not symbolic.valid or not graph_valid else ""
        if not symbolic.valid:
            error = "; ".join(
                item
                for item in (
                    structure_error,
                    "unmet {} @{}".format(
                        list(symbolic.missing), symbolic.failed_node
                    )
                    if not symbolic.precondition_ok
                    else "goal not reached: {}".format(list(symbolic.missing_goal)),
                )
                if item
            )
        return PlanRecord(
            case_id=case.id,
            condition=condition,
            ability=self.abilities.get(case.id, ""),
            sample=sample,
            outcome=outcome,
            graph_valid=graph_valid,
            goal_reached=symbolic.reachable_goal,
            lcs=similarity,
            plan_length=len(path),
            error=error,
            decision=dict(decision),
            symbolic=symbolic.as_dict(),
        )

    # ---------------------------------------------------------------- stage 2

    def ground(self) -> List[Dict[str, Any]]:
        """Compile the GT plan and every correct model plan to MS-HAB PlanData."""

        jobs: List[Dict[str, Any]] = []
        # mshab.evaluate asserts the task name appears in the plan path.
        plan_root = self.root / "plandata" / "set_table"
        plan_root.mkdir(parents=True, exist_ok=True)
        rows = self._read_rows(self.root / "planning_results.csv")
        by_case: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row)
        for case in self.cases:
            jobs.extend(
                self._ground_one(
                    case, "oracle", 0, gt_decision(case), plan_root
                )
            )
            for row in by_case.get(case.id, []):
                if row["correct"] != "True":
                    continue
                sample = int(row["sample"])
                source = self.root / "plans" / "{}.s{}.json".format(case.id, sample)
                if not source.exists():
                    continue
                decision = json.loads(source.read_text(encoding="utf-8"))
                jobs.extend(
                    self._ground_one(case, "deepseek", sample, decision, plan_root)
                )
                break  # one grounded plan per case keeps the pairing 1:1
        (self.root / "grounding_jobs.json").write_text(
            json.dumps(jobs, indent=2) + "\n", encoding="utf-8"
        )
        print("grounded {} plans -> {}".format(len(jobs), self.root / "grounding_jobs.json"))
        return jobs

    def _ground_one(
        self,
        case: AblationCase,
        method: str,
        sample: int,
        decision: Mapping[str, Any],
        plan_root: Path,
    ) -> List[Dict[str, Any]]:
        job_id = "{}.{}".format(method, case.id)
        try:
            flow = self.index.validate_and_compile(
                decision, case=case, condition=CONDITION
            )
            program = FlowProgram(flow)
            plan = MSHabPlanGrounder(
                self.official_source,
                task_family="set_table",
                plan_index=case.plan_index,
            ).compile(program, program.route("nominal"))
        except Exception as exc:  # noqa: BLE001 - grounding is allowed to fail
            return [
                {
                    "job_id": job_id,
                    "case_id": case.id,
                    "case_type": case.case_type,
                    "method": method,
                    "sample": sample,
                    "plan_index": case.plan_index,
                    "groundable": False,
                    "error": "{}: {}".format(type(exc).__name__, exc)[:300],
                }
            ]
        path = plan_root / "{}.json".format(job_id)
        path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        return [
            {
                "job_id": job_id,
                "case_id": case.id,
                "case_type": case.case_type,
                "method": method,
                "sample": sample,
                "plan_index": case.plan_index,
                "groundable": True,
                "plan_path": str(path),
                "steps": len(plan["plans"][0]["subtasks"]),
                "policy_type": plan["selection"]["recommended_policy_type"],
            }
        ]

    def execute(
        self,
        seeds: Sequence[int],
        record_video: bool,
        methods: Optional[Sequence[str]] = None,
    ) -> None:
        """Run every grounded plan, resuming from a previous partial run.

        Rollouts already recorded in ``execution_results.csv`` are kept, so a
        long batch can be extended (another method, more seeds) without
        spending GPU time re-running what is already measured.
        """

        jobs = json.loads(
            (self.root / "grounding_jobs.json").read_text(encoding="utf-8")
        )
        runs_root = self.root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        previous = self._read_rows(self.root / "execution_results.csv")
        rows: List[Dict[str, Any]] = [
            dict(row) for row in previous if row.get("status") == "complete"
        ]
        already = {(row["job_id"], str(row["seed"])) for row in rows}
        runnable = [
            job
            for job in jobs
            if job.get("groundable")
            and (methods is None or job["method"] in methods)
        ]
        total = len(runnable) * len(seeds)
        done = 0
        for job in runnable:
            plan_path = Path(job["plan_path"])
            plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
            for seed in seeds:
                done += 1
                if (job["job_id"], str(seed)) in already:
                    continue
                run_name = "{}.seed{}".format(job["job_id"], seed)
                command = evaluation_command(
                    plan_path=plan_path,
                    plan_data=plan_data,
                    task_family="set_table",
                    run_name=run_name,
                    seed=seed,
                    max_trajectories=1,
                    num_envs=1,
                    experiment_root=runs_root,
                    record_video=record_video,
                )
                log_path = runs_root / "{}.log".format(run_name)
                env = dict(os.environ)
                env.setdefault("SAPIEN_NO_DISPLAY", "1")
                env.setdefault("MPLCONFIGDIR", "/tmp/mshab-matplotlib")
                started = time.time()
                with log_path.open("w", encoding="utf-8") as handle:
                    process = subprocess.run(
                        command,
                        cwd=str(REPO_ROOT),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        check=False,
                    )
                row = self._collect(
                    runs_root / "vlm-plans" / run_name, job, seed, process.returncode
                )
                row["wall_seconds"] = round(time.time() - started, 1)
                rows.append(row)
                print(
                    "  [{:>3}/{}] {} -> success={} stop@{}".format(
                        done, total, run_name, row["success"], row["stopped_at"]
                    ),
                    flush=True,
                )
        for job in jobs:
            if job.get("groundable"):
                continue
            if any(row["job_id"] == job["job_id"] for row in rows):
                continue
            rows.append(
                {
                    "job_id": job["job_id"],
                    "case_id": job["case_id"],
                    "case_type": job["case_type"],
                    "method": job["method"],
                    "seed": "",
                    "status": "not_groundable",
                    "success": "",
                    "stopped_at": "",
                    "episode_steps": "",
                    "wall_seconds": "",
                }
            )
        self._write_rows(self.root / "execution_results.csv", rows)
        print("wrote", self.root / "execution_results.csv")

    @staticmethod
    def _collect(
        run_dir: Path, job: Mapping[str, Any], seed: int, returncode: int
    ) -> Dict[str, Any]:
        success: Any = ""
        steps: Any = ""
        stopped: Any = ""
        status = "complete" if returncode == 0 else "crashed"
        output = run_dir / "output.txt"
        if output.exists():
            import re

            text = output.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"'success_at_end': tensor\(([\d.]+)", text)
            if match:
                success = int(float(match.group(1)) == 1.0)
            match = re.search(r"'len': tensor\(([\d.]+)", text)
            if match:
                steps = int(float(match.group(1)))
        counts = run_dir / "subtask_fail_counts.json"
        if counts.exists():
            payload = json.loads(counts.read_text(encoding="utf-8"))
            if payload:
                stopped = int(sorted(payload, key=lambda item: int(item))[0])
        if success == "":
            status = "no_result"
        return {
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "case_type": job["case_type"],
            "method": job["method"],
            "seed": seed,
            "status": status,
            "success": success,
            "stopped_at": stopped,
            "episode_steps": steps,
        }

    # ----------------------------------------------------------------- tables

    def report(self) -> str:
        planning = self._read_rows(self.root / "planning_results.csv")
        execution = self._read_rows(self.root / "execution_results.csv")
        lines = ["# SetTable capability study", ""]
        lines += self._table_one(planning)
        lines += self._table_two(execution)
        text = "\n".join(lines) + "\n"
        (self.root / "report.md").write_text(text, encoding="utf-8")
        return text

    def _table_one(self, rows: Sequence[Mapping[str, Any]]) -> List[str]:
        scored = [row for row in rows if row["outcome"] != OUTCOME_API]
        n = len(scored)
        api = len(rows) - n
        gt_len = statistics.fmean(
            len(path) for path in self.gt_paths.values()
        )

        def rate(predicate) -> str:
            if not n:
                return "n/a"
            return "{:.2f}".format(sum(1 for row in scored if predicate(row)) / n)

        lengths = [int(row["plan_length"]) for row in scored if row["plan_length"]]
        sims = [float(row["lcs_f1"]) for row in scored if row["lcs_f1"]]
        out = [
            "## Table 1 - Planning: DeepSeek vs GT reference",
            "",
            "| Metric | GT reference | DeepSeek |",
            "| --- | --- | --- |",
            "| Plans scored | {} | {} |".format(n, n),
            "| Graph-valid (follows Layer-2 edges) | 1.00 | {} |".format(
                rate(lambda row: row["graph_valid"] == "True")
            ),
            "| Goal reached (contract simulation) | 1.00 | {} |".format(
                rate(lambda row: row["goal_reached"] == "True")
            ),
            "| **Correct (both)** | **1.00** | **{}** |".format(
                rate(lambda row: row["correct"] == "True")
            ),
            "| Exact match to GT | 1.00 | {} |".format(
                rate(lambda row: row["outcome"] == OUTCOME_EXACT)
            ),
            "| Valid alternative route | 0.00 | {} |".format(
                rate(lambda row: row["outcome"] == OUTCOME_ALTERNATIVE)
            ),
            "| Path LCS-F1 vs GT | 1.00 | {} |".format(
                "{:.2f}".format(statistics.fmean(sims)) if sims else "n/a"
            ),
            "| Plan length (mean) | {:.1f} | {} |".format(
                gt_len,
                "{:.1f}".format(statistics.fmean(lengths)) if lengths else "n/a",
            ),
            "",
            "Excluded from the denominator: {} API failure(s).".format(api),
            "",
            "### Outcome breakdown",
            "",
            "| Outcome | Count | Share |",
            "| --- | --- | --- |",
        ]
        for label in OUTCOME_ORDER:
            if label == OUTCOME_API:
                continue
            count = sum(1 for row in scored if row["outcome"] == label)
            out.append(
                "| {} | {} | {} |".format(
                    label, count, "{:.2f}".format(count / n) if n else "n/a"
                )
            )
        out.append("")
        return out

    def _table_two(self, rows: Sequence[Mapping[str, Any]]) -> List[str]:
        out = [
            "## Table 2 - MS-HAB execution",
            "",
            "| Metric | GT oracle | DeepSeek |",
            "| --- | --- | --- |",
        ]
        by_method = {
            method: [row for row in rows if row["method"] == method]
            for method in ("oracle", "deepseek")
        }
        stats: Dict[str, Dict[str, Any]] = {}
        for method, subset in by_method.items():
            done = [row for row in subset if row["status"] == "complete"]
            successes = [row for row in done if str(row["success"]) == "1"]
            steps = [int(row["episode_steps"]) for row in done if row["episode_steps"]]
            stats[method] = {
                "plans": len({row["job_id"] for row in subset}),
                "rollouts": len(done),
                "success": len(successes) / len(done) if done else None,
                "steps": statistics.fmean(steps) if steps else None,
                "ungroundable": sum(
                    1 for row in subset if row["status"] == "not_groundable"
                ),
            }

        def cell(method: str, key: str, fmt: str = "{:.2f}") -> str:
            value = stats.get(method, {}).get(key)
            return "n/a" if value is None else fmt.format(value)

        out += [
            "| Plans executed | {} | {} |".format(
                stats["oracle"]["plans"], stats["deepseek"]["plans"]
            ),
            "| Rollouts completed | {} | {} |".format(
                stats["oracle"]["rollouts"], stats["deepseek"]["rollouts"]
            ),
            "| Not groundable | {} | {} |".format(
                stats["oracle"]["ungroundable"], stats["deepseek"]["ungroundable"]
            ),
            "| Simulator success | {} | {} |".format(
                cell("oracle", "success"), cell("deepseek", "success")
            ),
        ]
        oracle = stats["oracle"]["success"]
        model = stats["deepseek"]["success"]
        normalized = (
            "n/a" if not oracle or model is None else "{:.2f}".format(model / oracle)
        )
        out += [
            "| **Success / oracle ceiling** | **1.00** | **{}** |".format(normalized),
            "| Episode steps (mean) | {} | {} |".format(
                cell("oracle", "steps", "{:.0f}"), cell("deepseek", "steps", "{:.0f}")
            ),
            "",
            "The oracle row is the controller ceiling: it is the same graph "
            "executed from the reference plan, on the same episodes and seeds. "
            "Normalised success separates planning quality from controller "
            "reliability.",
            "",
        ]
        stops: Dict[str, int] = {}
        for row in rows:
            if row["method"] != "oracle" or not str(row["stopped_at"]):
                continue
            if str(row["success"]) == "1":
                continue
            stops[str(row["stopped_at"])] = stops.get(str(row["stopped_at"]), 0) + 1
        if stops:
            out += [
                "### Where oracle rollouts stop (failure attribution)",
                "",
                "| Subtask index | Failures |",
                "| --- | --- |",
            ]
            for key in sorted(stops, key=int):
                out.append("| {} | {} |".format(key, stops[key]))
            out.append("")
        return out

    # ------------------------------------------------------------------ io

    @staticmethod
    def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _read_rows(path: Path) -> List[Dict[str, str]]:
        if not Path(path).exists():
            return []
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "ground", "execute", "report"])
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).with_name("set_table_graph_ablation.yaml"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("outputs") / "capability"
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--conditions", nargs="+", default=[CONDITION],
        choices=["flat_library", "full_graph"],
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    experiment = CapabilityExperiment(args.spec, args.out, args.config)
    if args.stage == "plan":
        experiment.plan(
            samples=args.samples,
            retries=args.retries,
            conditions=args.conditions,
            workers=args.workers,
        )
    elif args.stage == "ground":
        experiment.ground()
    elif args.stage == "execute":
        experiment.execute(
            seeds=args.seeds,
            record_video=args.record_video,
            methods=args.methods,
        )
    else:
        print(experiment.report())


if __name__ == "__main__":
    main()
