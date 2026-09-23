"""Flat-library versus full-graph SetTable planning ablation.

Both VLM conditions receive identical SkillNodes, Contracts, instructions,
model settings, and output schema.  The flat condition omits Layer-2 edges;
the full condition includes them.  The model must emit explicit SkillNode
paths.  Deterministic code validates and grounds those paths but never fills
in missing steps.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from mshab.experiments.granularity.vlm.execute import evaluation_command
from mshab.experiments.granularity.vlm.prompt import (
    FLOW_SCHEMA_VERSION,
    PATH_DECISION_SCHEMA_VERSION,
    build_path_planner_messages,
)
from mshab.experiments.granularity.vlm.run import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_GRAPH,
    DeepSeekClient,
    DeepSeekConfig,
    FourLayerGraphIndex,
    _entities_match,
    _grounding_fields,
    _read_json,
    _validate_gt_task_family,
    official_task_plan_path,
    render_flowchart,
)
from mshab.experiments.granularity.vlm.set_table_experiment import (
    ExecutionJob,
    REPOSITORY_ROOT,
    _format_metric,
    _markdown_table,
    _mean,
    _parse_evaluator_log,
    _write_csv,
    _write_json,
)
from mshab.experiments.granularity.vlm.simulator import (
    FlowProgram,
    MSHabPlanGrounder,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PACKAGE_DIR / "set_table_graph_ablation.yaml"
DEFAULT_OUTPUT = PACKAGE_DIR / "outputs" / "set_table_graph_ablation_main"
ABLATION_SCHEMA_VERSION = "mshab.set-table-graph-ablation.v1"
CONDITIONS = ("flat_library", "full_graph")


@dataclass(frozen=True)
class AblationCase:
    id: str
    case_type: str
    plan_index: int
    failed_skill_node_id: Optional[str]
    oracle_route_id: str
    instruction: str


@dataclass(frozen=True)
class AblationSpec:
    name: str
    task_family: str
    split: str
    seed: int
    cases: Tuple[AblationCase, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "AblationSpec":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != ABLATION_SCHEMA_VERSION:
            raise ValueError("unsupported ablation schema_version")
        cases = tuple(
            AblationCase(
                id=str(item["id"]),
                case_type=str(item["case_type"]),
                plan_index=int(item["plan_index"]),
                failed_skill_node_id=(
                    None
                    if item.get("failed_skill_node_id") is None
                    else str(item["failed_skill_node_id"])
                ),
                oracle_route_id=str(item["oracle_route_id"]),
                instruction=str(item["instruction"]).strip(),
            )
            for item in payload["cases"]
        )
        if len(cases) != 10:
            raise ValueError("the SetTable ablation requires exactly 10 cases")
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("ablation case ids must be unique")
        if tuple(case.plan_index for case in cases) != tuple(range(10)):
            raise ValueError("cases must pair one-to-one with plan-index 0 through 9")
        if {case.case_type for case in cases} != {"nominal", "fallback"}:
            raise ValueError("cases must include nominal and fallback queries")
        if any(not case.instruction for case in cases):
            raise ValueError("case instructions must be non-empty")
        return cls(
            name=str(payload["name"]),
            task_family=str(payload["task_family"]),
            split=str(payload["split"]),
            seed=int(payload.get("seed", 0)),
            cases=cases,
        )


def _canonical_strategy_decision(instruction: str) -> Dict[str, Any]:
    """The graph oracle input; this object is never sent to the VLM."""

    occurrences = [
        ("retrieve_bowl", "retrieve_from_drawer", {"object": "024_bowl"}),
        ("deliver_bowl", "deliver_to_table", {"object": "024_bowl"}),
        ("restore_drawer", "restore_drawer", {}),
        ("retrieve_apple", "retrieve_from_fridge", {"object": "013_apple"}),
        ("deliver_apple", "deliver_to_table", {"object": "013_apple"}),
        ("restore_fridge", "restore_fridge", {}),
    ]
    return {
        "schema_version": "mshab.vlm-strategy-decisions.v1",
        "instruction": instruction,
        "task_family": "set_table",
        "strategy_occurrences": [
            {
                "id": occurrence_id,
                "strategy_id": strategy_id,
                "bindings": bindings,
            }
            for occurrence_id, strategy_id, bindings in occurrences
        ],
        "occurrence_order": [item[0] for item in occurrences],
        "decision_summary": "Deterministic SetTable graph oracle.",
    }


def _edge_key(edge: Mapping[str, Any]) -> Tuple[str, str, str]:
    return str(edge["source"]), str(edge["target"]), str(edge["relation"])


class ExplicitPathIndex:
    """Prompt projection, strict path validator, and flow compiler."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self.document = document
        self.base = FourLayerGraphIndex(document)
        self.causal_edges: set[Tuple[str, str]] = set()
        self.fallback_edges: set[Tuple[str, str]] = set()
        self.recovery_targets: set[str] = set()
        for subgraph in self.base.subgraphs.values():
            for edge in subgraph["edges"]:
                relation = edge["relation"]
                source = str(edge["source"])
                target = str(edge["target"])
                if relation == "enables":
                    self.causal_edges.add((source, target))
                elif relation == "requires":
                    self.causal_edges.add((target, source))
                elif relation == "fallback_to":
                    self.fallback_edges.add((source, target))
                    self.recovery_targets.add(target)

    def library_view(self, condition: str) -> Dict[str, Any]:
        if condition not in CONDITIONS:
            raise ValueError("unknown ablation condition")
        nodes = []
        edges = []
        for subgoal_id, subgraph in sorted(self.base.subgraphs.items()):
            for raw_node in subgraph["nodes"]:
                contract = self.base.contracts[raw_node["contract_id"]]
                nodes.append(
                    {
                        "id": raw_node["id"],
                        "subgoal_id": subgoal_id,
                        "contract_type": contract["contract_type"],
                        "arguments": raw_node["arguments"],
                        "achieves": raw_node["achieves"],
                    }
                )
            if condition == "full_graph":
                edges.extend(copy.deepcopy(subgraph["edges"]))
        return {
            "condition": condition,
            "task_family": "set_table",
            "subgoals": copy.deepcopy(self.document["layer1"]["subgoals"]),
            "skill_nodes": sorted(nodes, key=lambda item: item["id"]),
            "contracts": copy.deepcopy(self.document["layer3"]["contracts"]),
            "layer2_edges": sorted(edges, key=_edge_key),
            "edge_semantics": (
                copy.deepcopy(self.document["semantics"])
                if condition == "full_graph"
                else {
                    "note": (
                        "Edges are intentionally hidden for the flat-library "
                        "ablation."
                    )
                }
            ),
        }

    def required_bindings_for_nodes(self, node_ids: Sequence[str]) -> set[str]:
        required: set[str] = set()
        for node_id in node_ids:
            node = self.base.nodes[node_id]
            for value in node["arguments"].values():
                if not isinstance(value, str):
                    continue
                for _, field_name, _, _ in string.Formatter().parse(value):
                    if field_name:
                        required.add(field_name)
        return required

    @staticmethod
    def _object_role(value: str) -> str:
        if _entities_match(value, "024_bowl"):
            return "bowl"
        if _entities_match(value, "013_apple"):
            return "apple"
        raise ValueError(
            "SetTable object must identify the bowl or apple; got {!r}".format(
                value
            )
        )

    def _segment_role(self, segment: Mapping[str, Any]) -> str:
        subgoal_id = str(segment["subgoal_id"])
        if subgoal_id in ("retrieve", "deliver"):
            bindings = segment["bindings"]
            value = bindings.get("object")
            if not isinstance(value, str):
                raise ValueError(
                    "{} segment requires an object binding".format(subgoal_id)
                )
            return "{}_{}".format(subgoal_id, self._object_role(value))
        if subgoal_id == "restore":
            final_node = self.base.nodes[segment["skill_node_path"][-1]]
            articulation = final_node.get("arguments", {}).get("articulation")
            if articulation == "kitchen_counter":
                return "restore_drawer"
            if articulation == "fridge":
                return "restore_fridge"
            raise ValueError(
                "restore segment must close kitchen_counter or fridge"
            )
        raise ValueError("unsupported SetTable SubGoal {!r}".format(subgoal_id))

    def _validate_set_table_order(
        self,
        order: Sequence[str],
        segments: Mapping[str, Mapping[str, Any]],
    ) -> None:
        expected = (
            "retrieve_bowl",
            "deliver_bowl",
            "restore_drawer",
            "retrieve_apple",
            "deliver_apple",
            "restore_fridge",
        )
        actual = tuple(self._segment_role(segments[item]) for item in order)
        if actual != expected:
            raise ValueError(
                "SetTable task order must be {}; got {}".format(
                    list(expected),
                    list(actual),
                )
            )

    @staticmethod
    def _strict_keys(
        payload: Mapping[str, Any],
        expected: set[str],
        label: str,
    ) -> None:
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "{} fields must be {}; got {}".format(
                    label,
                    sorted(expected),
                    sorted(actual),
                )
            )

    def validate_and_compile(
        self,
        decision: Mapping[str, Any],
        *,
        case: AblationCase,
        condition: str,
    ) -> Dict[str, Any]:
        self._strict_keys(
            decision,
            {
                "schema_version",
                "instruction",
                "task_family",
                "condition",
                "failed_skill_node_id",
                "segments",
                "segment_order",
                "decision_summary",
            },
            "path decision",
        )
        if decision["schema_version"] != PATH_DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported path-decision schema_version")
        if decision["instruction"] != case.instruction:
            raise ValueError("decision must echo the instruction exactly")
        if decision["task_family"] != "set_table":
            raise ValueError("decision task_family must be set_table")
        if decision["condition"] != condition:
            raise ValueError("decision must echo its ablation condition")
        if decision["failed_skill_node_id"] != case.failed_skill_node_id:
            raise ValueError("decision must echo failed_skill_node_id exactly")
        if not isinstance(decision["decision_summary"], str):
            raise TypeError("decision_summary must be a string")
        raw_segments = decision["segments"]
        order = decision["segment_order"]
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("segments must be a non-empty list")
        if not isinstance(order, list) or not all(
            isinstance(item, str) for item in order
        ):
            raise TypeError("segment_order must be a list of strings")

        segments: Dict[str, Mapping[str, Any]] = {}
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, Mapping):
                raise TypeError("each segment must be an object")
            self._strict_keys(
                raw_segment,
                {"id", "subgoal_id", "bindings", "skill_node_path"},
                "segment",
            )
            segment_id = raw_segment["id"]
            subgoal_id = raw_segment["subgoal_id"]
            bindings = raw_segment["bindings"]
            path = raw_segment["skill_node_path"]
            if not isinstance(segment_id, str) or not segment_id:
                raise ValueError("segment id must be a non-empty string")
            if segment_id in segments:
                raise ValueError("segment ids must be unique")
            if subgoal_id not in self.base.subgraphs:
                raise ValueError("segment references an unknown SubGoal")
            if not isinstance(bindings, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in bindings.items()
            ):
                raise TypeError("segment bindings must map strings to strings")
            if not isinstance(path, list) or not path or not all(
                isinstance(item, str) for item in path
            ):
                raise ValueError("skill_node_path must be a non-empty string list")
            if len(path) != len(set(path)):
                raise ValueError("one segment cannot repeat a SkillNode")
            unknown = [item for item in path if item not in self.base.nodes]
            if unknown:
                raise ValueError(
                    "path references unknown SkillNodes {}".format(unknown)
                )
            foreign = [
                item
                for item in path
                if self.base.node_subgoal[item] != subgoal_id
            ]
            if foreign:
                raise ValueError("path contains SkillNodes from another SubGoal")
            required = self.required_bindings_for_nodes(path)
            if set(bindings) != required:
                raise ValueError(
                    "segment {} requires bindings {}; got {}".format(
                        segment_id,
                        sorted(required),
                        sorted(bindings),
                    )
                )
            for source, target in zip(path, path[1:]):
                if (source, target) not in self.causal_edges:
                    raise ValueError(
                        "no causal Layer-2 edge {} -> {}".format(source, target)
                    )
            if subgoal_id not in self.base.nodes[path[-1]]["achieves"]:
                raise ValueError("segment path does not end at a SubGoal achiever")
            segments[segment_id] = raw_segment

        if len(order) != len(set(order)) or set(order) != set(segments):
            raise ValueError("segment_order must contain every segment exactly once")
        self._validate_set_table_order(order, segments)

        fallback_matches = []
        for segment_id in order:
            path = list(segments[segment_id]["skill_node_path"])
            end = path[-1]
            if case.failed_skill_node_id is None:
                if end in self.recovery_targets:
                    raise ValueError("nominal query cannot select a fallback achiever")
            elif (case.failed_skill_node_id, end) in self.fallback_edges:
                if case.failed_skill_node_id in path:
                    raise ValueError("fallback path must replace the failed SkillNode")
                fallback_matches.append(segment_id)
            elif end in self.recovery_targets:
                raise ValueError("selected fallback does not match the stated failure")
        if case.failed_skill_node_id is not None and len(fallback_matches) != 1:
            raise ValueError("exactly one segment must recover the stated failure")

        nodes: List[Dict[str, Any]] = []
        nominal_order: List[str] = []
        selected_segments = []
        for segment_id in order:
            segment = segments[segment_id]
            subgoal_id = str(segment["subgoal_id"])
            bindings = dict(segment["bindings"])
            pseudo_strategy = {"id": "explicit_path", "subgoal_id": subgoal_id}
            for position, skill_node_id in enumerate(segment["skill_node_path"]):
                flow_id = "{}:{:02d}:{}".format(
                    segment_id,
                    position,
                    skill_node_id,
                )
                nominal_order.append(flow_id)
                nodes.append(
                    self.base._flow_node(
                        flow_id=flow_id,
                        occurrence_id=segment_id,
                        strategy=pseudo_strategy,
                        skill_node_id=skill_node_id,
                        bindings=bindings,
                        task_family="set_table",
                        kind="nominal",
                    )
                )
            selected_segments.append(
                {
                    "occurrence_id": segment_id,
                    "strategy_id": "explicit_skill_node_path",
                    "bindings": bindings,
                }
            )
        return {
            "schema_version": FLOW_SCHEMA_VERSION,
            "source": condition,
            "instruction": case.instruction,
            "task_family": "set_table",
            "metadata": {
                "decision_schema_version": PATH_DECISION_SCHEMA_VERSION,
                "failed_skill_node_id": case.failed_skill_node_id,
                "graph_validation": "passed",
            },
            "nodes": nodes,
            "edges": [
                {"source": source, "target": target, "relation": "next"}
                for source, target in zip(nominal_order, nominal_order[1:])
            ],
            "nominal_order": nominal_order,
            "fallback_branches": [],
            "alternative_choices": [],
            "selected_strategies": selected_segments,
        }


def _decision_from_oracle_route(
    *,
    flow: Mapping[str, Any],
    route_id: str,
    case: AblationCase,
) -> Dict[str, Any]:
    program = FlowProgram(flow)
    route = program.route(route_id)
    bindings = {
        item["occurrence_id"]: dict(item["bindings"])
        for item in flow["selected_strategies"]
    }
    occurrence_order = list(program.occurrence_order)
    segments = []
    for occurrence_id in occurrence_order:
        local = [
            program.nodes[node_id]
            for node_id in route.node_order
            if program.nodes[node_id]["occurrence_id"] == occurrence_id
        ]
        if not local:
            raise ValueError("oracle route omitted occurrence {}".format(occurrence_id))
        segments.append(
            {
                "id": occurrence_id,
                "subgoal_id": local[0]["subgoal_id"],
                "bindings": bindings[occurrence_id],
                "skill_node_path": [item["skill_node_id"] for item in local],
            }
        )
    return {
        "schema_version": PATH_DECISION_SCHEMA_VERSION,
        "instruction": case.instruction,
        "task_family": "set_table",
        "condition": "graph_oracle",
        "failed_skill_node_id": case.failed_skill_node_id,
        "segments": segments,
        "segment_order": occurrence_order,
        "decision_summary": "Deterministic graph-oracle path.",
    }


def _path_from_decision(decision: Mapping[str, Any]) -> List[str]:
    try:
        raw_segments = decision.get("segments", [])
        segments = {
            item.get("id"): item
            for item in raw_segments
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        order = decision.get("segment_order", [])
        if not isinstance(order, list):
            return []
        result = []
        for segment_id in order:
            segment = segments.get(segment_id, {})
            path = segment.get("skill_node_path", [])
            if isinstance(path, list):
                result.extend(item for item in path if isinstance(item, str))
        return result
    except (AttributeError, TypeError):
        return []


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _grounded_actions(
    decision: Mapping[str, Any],
    index: ExplicitPathIndex,
) -> List[Dict[str, Any]]:
    raw_segments = decision.get("segments", [])
    if not isinstance(raw_segments, list):
        return []
    segments = {
        item.get("id"): item
        for item in raw_segments
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    order = decision.get("segment_order", [])
    if not isinstance(order, list):
        return []
    actions = []
    for segment_id in order:
        segment = segments.get(segment_id)
        if not isinstance(segment, Mapping):
            continue
        bindings = segment.get("bindings", {})
        path = segment.get("skill_node_path", [])
        if not isinstance(bindings, Mapping) or not isinstance(path, list):
            continue
        for node_id in path:
            if node_id not in index.base.nodes:
                actions.append({"skill_node_id": node_id, "contract_type": None})
                continue
            node = index.base.nodes[node_id]
            contract = index.base.contracts[node["contract_id"]]
            try:
                arguments = index.base._ground_arguments(node_id, bindings)
            except (KeyError, ValueError):
                arguments = {}
            actions.append(
                {
                    "skill_node_id": node_id,
                    "contract_type": contract["contract_type"],
                    "arguments": arguments,
                }
            )
    return actions


def compare_path_decisions(
    prediction: Mapping[str, Any],
    oracle: Mapping[str, Any],
    index: ExplicitPathIndex,
) -> Dict[str, Any]:
    predicted_path = _path_from_decision(prediction)
    oracle_path = _path_from_decision(oracle)
    lcs = _lcs_length(predicted_path, oracle_path)
    lcs_f1 = (
        2 * lcs / (len(predicted_path) + len(oracle_path))
        if predicted_path or oracle_path
        else 1.0
    )
    predicted_actions = _grounded_actions(prediction, index)
    oracle_actions = _grounded_actions(oracle, index)
    compared = 0
    matches = 0
    mismatches = []
    for position, expected in enumerate(oracle_actions):
        contract_type = expected.get("contract_type")
        if not isinstance(contract_type, str):
            continue
        fields = _grounding_fields(contract_type)
        actual = (
            predicted_actions[position]
            if position < len(predicted_actions)
            else {}
        )
        for field in fields:
            expected_value = expected.get("arguments", {}).get(field)
            if expected_value is None:
                continue
            compared += 1
            actual_value = (
                actual.get("arguments", {}).get(field)
                if actual.get("contract_type") == contract_type
                else None
            )
            if _entities_match(actual_value, expected_value):
                matches += 1
            else:
                mismatches.append(
                    {
                        "position": position,
                        "field": field,
                        "predicted": actual_value,
                        "expected": expected_value,
                    }
                )
    return {
        "predicted_path": predicted_path,
        "oracle_path": oracle_path,
        "predicted_length": len(predicted_path),
        "oracle_length": len(oracle_path),
        "lcs_length": lcs,
        "path_lcs_f1": lcs_f1,
        "grounding_accuracy": matches / compared if compared else 0.0,
        "grounding_fields_compared": compared,
        "grounding_mismatches": mismatches,
    }


class SetTableGraphAblation:
    """Resumable 20-call planning ablation and 30-job simulator study."""

    def __init__(
        self,
        *,
        spec: AblationSpec,
        output_dir: Path,
        graph_path: Path,
        config_path: Path,
        asset_root: Path,
        execution_split: Optional[str] = None,
        planning_source_dir: Optional[Path] = None,
        execution_seeds: Optional[Sequence[int]] = None,
    ) -> None:
        self.spec = spec
        self.output_dir = output_dir.resolve()
        self.graph_path = graph_path.resolve()
        self.config_path = config_path.resolve()
        self.asset_root = asset_root.resolve()
        self.execution_split = execution_split or spec.split
        if self.execution_split not in ("train", "val"):
            raise ValueError("execution_split must be 'train' or 'val'")
        self.execution_seeds = tuple(
            int(seed)
            for seed in (
                execution_seeds if execution_seeds is not None else (spec.seed,)
            )
        )
        if not self.execution_seeds:
            raise ValueError("execution_seeds cannot be empty")
        if len(self.execution_seeds) != len(set(self.execution_seeds)):
            raise ValueError("execution_seeds must be unique")
        if any(seed < 0 for seed in self.execution_seeds):
            raise ValueError("execution_seeds must be non-negative")
        self.reuses_planning = planning_source_dir is not None
        self.planning_source_dir = (
            planning_source_dir.resolve()
            if planning_source_dir is not None
            else self.output_dir
        )
        if self.reuses_planning and self.planning_source_dir == self.output_dir:
            raise ValueError(
                "planning_source_dir must differ from output_dir when reusing "
                "planning artifacts"
            )
        self.gt_path = official_task_plan_path(
            spec.task_family,
            self.execution_split,
            self.asset_root,
        ).resolve()
        self.planning_dir = self.output_dir / "planning"
        self.planning_input_dir = self.planning_source_dir / "planning"
        self.oracle_dir = self.output_dir / "oracle"
        self.plans_dir = self.output_dir / "execution_plans"
        self.status_dir = self.output_dir / "execution_status"
        self.logs_dir = self.output_dir / "execution_logs"
        self.simulator_workspace = self.output_dir / "simulator_runs"
        self.tables_dir = self.output_dir / "tables"

    def _validate_planning_source(self) -> None:
        if not self.reuses_planning:
            return
        manifest_path = self.planning_source_dir / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "planning source manifest not found: {}".format(manifest_path)
            )
        if not self.planning_input_dir.is_dir():
            raise FileNotFoundError(
                "planning artifacts not found: {}".format(
                    self.planning_input_dir
                )
            )
        source = _read_json(manifest_path)
        expected_cases = [case.__dict__ for case in self.spec.cases]
        source_split = source.get("planning_split", source.get("split"))
        if source.get("schema_version") != ABLATION_SCHEMA_VERSION:
            raise ValueError("planning source uses an incompatible schema")
        if source.get("task_family") != self.spec.task_family:
            raise ValueError("planning source task_family does not match")
        if source_split != self.spec.split:
            raise ValueError("planning source split does not match")
        if source.get("conditions") != list(CONDITIONS):
            raise ValueError("planning source conditions do not match")
        if source.get("cases") != expected_cases:
            raise ValueError("planning source cases do not match")
        source_graph = source.get("graph")
        if not isinstance(source_graph, str):
            raise ValueError("planning source graph provenance is missing")
        if Path(source_graph).resolve() != self.graph_path:
            raise ValueError("planning source graph does not match")

    def _validate_existing_destination(self, manifest_path: Path) -> None:
        if not manifest_path.is_file():
            return
        prior = _read_json(manifest_path)
        prior_execution_split = prior.get(
            "execution_split",
            prior.get("split"),
        )
        prior_planning_source = Path(
            prior.get(
                "planning_source",
                str(self.output_dir / "planning"),
            )
        ).resolve()
        if prior_execution_split != self.execution_split:
            raise ValueError(
                "existing output uses execution_split={!r}; pass the same "
                "--execution-split or choose a new --output-dir".format(
                    prior_execution_split
                )
            )
        if prior_planning_source != self.planning_input_dir:
            raise ValueError(
                "existing output uses a different planning source; pass the "
                "same --planning-source-dir or choose a new --output-dir"
            )
        prior_seeds = tuple(prior.get("execution_seeds", (self.spec.seed,)))
        if prior_seeds != self.execution_seeds:
            raise ValueError(
                "existing output uses execution_seeds={!r}; pass the same "
                "--execution-seeds or choose a new --output-dir".format(
                    prior_seeds
                )
            )

    def initialize(self) -> None:
        self._validate_planning_source()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.graph_path.is_file():
            raise FileNotFoundError("graph not found: {}".format(self.graph_path))
        if not self.gt_path.is_file():
            raise FileNotFoundError("GT not found: {}".format(self.gt_path))
        manifest_path = self.output_dir / "experiment_manifest.json"
        self._validate_existing_destination(manifest_path)
        _write_json(
            manifest_path,
            {
                "schema_version": ABLATION_SCHEMA_VERSION,
                "name": self.spec.name,
                "task_family": self.spec.task_family,
                "split": self.spec.split,
                "planning_split": self.spec.split,
                "execution_split": self.execution_split,
                "planning_source": str(self.planning_input_dir),
                "seed": self.spec.seed,
                "execution_seeds": list(self.execution_seeds),
                "conditions": list(CONDITIONS),
                "cases": [case.__dict__ for case in self.spec.cases],
                "graph": str(self.graph_path),
                "ground_truth": str(self.gt_path),
                "design": {
                    "deepseek_calls_in_this_output": (
                        0 if self.reuses_planning else 20
                    ),
                    "planning_reused": self.reuses_planning,
                    "planning_cases_per_condition": 10,
                    "execution_plan_jobs": 30,
                    "execution_rollouts": 30 * len(self.execution_seeds),
                    "only_independent_variable": "Layer-2 graph edges",
                },
            },
        )

    def _oracle_pair(
        self,
        case: AblationCase,
        index: ExplicitPathIndex,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        compiled = index.base.compile_decision(
            _canonical_strategy_decision(case.instruction),
            case.instruction,
            "set_table",
            {},
        )
        decision = _decision_from_oracle_route(
            flow=compiled,
            route_id=case.oracle_route_id,
            case=case,
        )
        validation_copy = copy.deepcopy(decision)
        validation_copy["condition"] = "full_graph"
        flow = index.validate_and_compile(
            validation_copy,
            case=case,
            condition="full_graph",
        )
        flow["source"] = "graph_oracle"
        return decision, flow

    def write_oracles(self, index: ExplicitPathIndex) -> None:
        for case in self.spec.cases:
            decision, flow = self._oracle_pair(case, index)
            case_dir = self.oracle_dir / case.id
            _write_json(case_dir / "path_decision.json", decision)
            _write_json(case_dir / "execution_flow.json", flow)
            render_flowchart(flow, case_dir)

    def planning(
        self,
        *,
        condition_filter: Optional[str],
        case_filter: Optional[str],
        force: bool,
        fail_fast: bool,
    ) -> None:
        self.initialize()
        document = _read_json(self.graph_path)
        index = ExplicitPathIndex(document)
        self.write_oracles(index)
        config = DeepSeekConfig.from_yaml(self.config_path)
        config.resolved_api_key()
        if config.max_tokens < 16384:
            raise ValueError(
                "graph ablation requires max_tokens >= 16384; fallback "
                "reasoning was observed to exceed 8192 tokens"
            )
        if config.retries != 0:
            raise ValueError(
                "graph ablation requires retries: 0 so one case equals one "
                "model sample"
            )
        client = DeepSeekClient(config)
        conditions = (
            (condition_filter,)
            if condition_filter is not None
            else CONDITIONS
        )
        cases = [
            case
            for case in self.spec.cases
            if case_filter is None or case.id == case_filter
        ]
        if not cases:
            raise ValueError("case filter did not match a case")

        total = len(conditions) * len(cases)
        position = 0
        for condition in conditions:
            if condition not in CONDITIONS:
                raise ValueError("unknown condition {}".format(condition))
            library_view = index.library_view(condition)
            for case in cases:
                position += 1
                case_dir = self.planning_dir / condition / case.id
                status_path = case_dir / "status.json"
                if not force and status_path.exists():
                    prior = _read_json(status_path)
                    if prior.get("status") in ("complete", "api_failed"):
                        print(
                            "planning {}/{} {} {} already sampled".format(
                                position,
                                total,
                                condition,
                                case.id,
                            )
                        )
                        continue
                case_dir.mkdir(parents=True, exist_ok=True)
                print(
                    "planning {}/{} {} {}".format(
                        position,
                        total,
                        condition,
                        case.id,
                    )
                )
                messages = build_path_planner_messages(
                    instruction=case.instruction,
                    task_family="set_table",
                    condition=condition,
                    failed_skill_node_id=case.failed_skill_node_id,
                    library_view=library_view,
                )
                _write_json(
                    case_dir / "planner_request.json",
                    config.request_payload(messages),
                )
                oracle = _read_json(
                    self.oracle_dir / case.id / "path_decision.json"
                )
                started = time.time()
                api_metadata: Optional[Dict[str, Any]] = None
                try:
                    decision, api_metadata = client.complete(messages)
                    _write_json(case_dir / "path_decision.json", decision)
                    _write_json(
                        case_dir / "planner_api_metadata.json",
                        api_metadata,
                    )
                except Exception as exc:
                    _write_json(
                        status_path,
                        {
                            "status": "api_failed",
                            "condition": condition,
                            "case_id": case.id,
                            "elapsed_seconds": time.time() - started,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    if fail_fast:
                        raise
                    print(
                        "API call failed for {} {}: {}".format(
                            condition,
                            case.id,
                            exc,
                        )
                    )
                    continue

                metrics = compare_path_decisions(decision, oracle, index)
                _write_json(case_dir / "path_metrics.json", metrics)
                graph_valid = False
                validation_error = None
                try:
                    flow = index.validate_and_compile(
                        decision,
                        case=case,
                        condition=condition,
                    )
                    graph_valid = True
                    _write_json(case_dir / "execution_flow.json", flow)
                    render_flowchart(flow, case_dir)
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    validation_error = str(exc)
                    if fail_fast:
                        raise
                _write_json(
                    status_path,
                    {
                        "status": "complete",
                        "condition": condition,
                        "case_id": case.id,
                        "graph_valid": graph_valid,
                        "validation_error": validation_error,
                        "elapsed_seconds": time.time() - started,
                        "api": api_metadata,
                    },
                )
                self.summarize()
        self.summarize()

    def _planning_rows(self) -> List[Dict[str, Any]]:
        rows = []
        index = ExplicitPathIndex(_read_json(self.graph_path))
        for condition in CONDITIONS:
            for case in self.spec.cases:
                case_dir = self.planning_input_dir / condition / case.id
                status = (
                    _read_json(case_dir / "status.json")
                    if (case_dir / "status.json").exists()
                    else {"status": "pending"}
                )
                metrics = (
                    _read_json(case_dir / "path_metrics.json")
                    if (case_dir / "path_metrics.json").exists()
                    else {}
                )
                decision_path = case_dir / "path_decision.json"
                oracle_path = (
                    self.planning_source_dir
                    / "oracle"
                    / case.id
                    / "path_decision.json"
                )
                graph_valid = bool(status.get("graph_valid", False))
                validation_error = status.get("validation_error")
                if decision_path.exists():
                    decision = _read_json(decision_path)
                    if oracle_path.exists():
                        try:
                            metrics = compare_path_decisions(
                                decision,
                                _read_json(oracle_path),
                                index,
                            )
                        except (KeyError, TypeError, ValueError):
                            # Preserve sampling-time comparison metrics when a
                            # malformed response cannot be normalized again.
                            pass
                    try:
                        index.validate_and_compile(
                            decision,
                            case=case,
                            condition=condition,
                        )
                        graph_valid = True
                        validation_error = None
                    except (KeyError, TypeError, ValueError) as exc:
                        graph_valid = False
                        validation_error = str(exc)
                rows.append(
                    {
                        "condition": condition,
                        "case_id": case.id,
                        "case_type": case.case_type,
                        "plan_index": case.plan_index,
                        "status": status.get("status"),
                        "graph_valid": graph_valid,
                        "path_lcs_f1": metrics.get("path_lcs_f1"),
                        "grounding_accuracy": metrics.get("grounding_accuracy"),
                        "predicted_length": metrics.get("predicted_length"),
                        "oracle_length": metrics.get("oracle_length"),
                        "validation_error": validation_error,
                    }
                )
        return rows

    def prepare_execution(self) -> List[ExecutionJob]:
        self.initialize()
        if not self.planning_input_dir.is_dir():
            raise FileNotFoundError(
                "planning artifacts not found: {}".format(
                    self.planning_input_dir
                )
            )
        source = _read_json(self.gt_path)
        _validate_gt_task_family(source, "set_table")
        document = _read_json(self.graph_path)
        index = ExplicitPathIndex(document)
        self.write_oracles(index)
        jobs: List[ExecutionJob] = []

        for case in self.spec.cases:
            for condition, label, artifact_path in (
                (
                    "graph_oracle",
                    "Graph Oracle",
                    self.oracle_dir / case.id / "execution_flow.json",
                ),
                (
                    "flat_library",
                    "Flat Skill Library",
                    self.planning_input_dir
                    / "flat_library"
                    / case.id
                    / "path_decision.json",
                ),
                (
                    "full_graph",
                    "Full Skill Graph",
                    self.planning_input_dir
                    / "full_graph"
                    / case.id
                    / "path_decision.json",
                ),
            ):
                job_id = "{}.{}".format(condition, case.id)
                if not artifact_path.exists():
                    jobs.append(
                        ExecutionJob(
                            id=job_id,
                            condition=condition,
                            label=label,
                            instruction_id=case.id,
                            plan_index=case.plan_index,
                            route_id=case.oracle_route_id,
                            groundable=False,
                            task_plan=None,
                            error="graph-valid execution flow is unavailable",
                        )
                    )
                    continue
                try:
                    if condition == "graph_oracle":
                        flow = _read_json(artifact_path)
                    else:
                        flow = index.validate_and_compile(
                            _read_json(artifact_path),
                            case=case,
                            condition=condition,
                        )
                    program = FlowProgram(flow)
                    plan = MSHabPlanGrounder(
                        source,
                        task_family="set_table",
                        plan_index=case.plan_index,
                    ).compile(program, program.route("nominal"))
                    relative = (
                        Path("execution_plans")
                        / condition
                        / "{}.json".format(case.id)
                    )
                    _write_json(self.output_dir / relative, plan)
                    jobs.append(
                        ExecutionJob(
                            id=job_id,
                            condition=condition,
                            label=label,
                            instruction_id=case.id,
                            plan_index=case.plan_index,
                            route_id=case.oracle_route_id,
                            groundable=True,
                            task_plan=str(relative),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    jobs.append(
                        ExecutionJob(
                            id=job_id,
                            condition=condition,
                            label=label,
                            instruction_id=case.id,
                            plan_index=case.plan_index,
                            route_id=case.oracle_route_id,
                            groundable=False,
                            task_plan=None,
                            error=str(exc),
                        )
                    )
        if len(jobs) != 30:
            raise AssertionError("expected 30 execution jobs")
        payload = [job.to_dict() for job in jobs]
        _write_json(self.output_dir / "execution_jobs.json", payload)
        _write_csv(self.output_dir / "execution_jobs.csv", payload)
        self.summarize()
        print("prepared {} paired execution jobs".format(len(jobs)))
        return jobs

    def _load_jobs(self) -> List[ExecutionJob]:
        path = self.output_dir / "execution_jobs.json"
        if not path.exists():
            raise FileNotFoundError("run --phase prepare before execution")
        return [ExecutionJob(**item) for item in json.loads(path.read_text())]

    def _rollout_suffix(self, seed: int) -> str:
        if self.execution_seeds == (self.spec.seed,):
            return ""
        return ".seed{}".format(seed)

    def _status_path(self, job_id: str, seed: int) -> Path:
        return self.status_dir / "{}{}.json".format(
            job_id,
            self._rollout_suffix(seed),
        )

    def _log_path(self, job_id: str, seed: int) -> Path:
        return self.logs_dir / "{}{}.log".format(
            job_id,
            self._rollout_suffix(seed),
        )

    @staticmethod
    def _plan_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def execute(
        self,
        *,
        max_jobs: Optional[int],
        condition: Optional[str],
        case_id: Optional[str],
        case_type: Optional[str],
        record_video: bool,
        retry_failed: bool,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible; simulator execution cannot start")
        jobs = self._load_jobs()
        selected = [
            job
            for job in jobs
            if job.groundable
            and (condition is None or job.condition == condition)
            and (case_id is None or job.instruction_id == case_id)
            and (
                case_type is None
                or next(
                    case.case_type
                    for case in self.spec.cases
                    if case.id == job.instruction_id
                )
                == case_type
            )
        ]
        rollouts = [
            (job, seed) for job in selected for seed in self.execution_seeds
        ]
        reusable: Dict[Tuple[str, int], Tuple[Path, Dict[str, Any]]] = {}
        for existing in self.status_dir.glob("*.json"):
            payload = _read_json(existing)
            signature = payload.get("plan_sha256")
            seed = payload.get("seed")
            if (
                payload.get("status") == "complete"
                and payload.get("simulator_success") is not None
                and isinstance(signature, str)
                and isinstance(seed, int)
            ):
                reusable[(signature, seed)] = (existing, payload)
        completed_this_run = 0
        for position, (job, seed) in enumerate(rollouts, start=1):
            status_path = self._status_path(job.id, seed)
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
            plan_sha256 = self._plan_sha256(plan_path)
            reusable_key = (plan_sha256, seed)
            if reusable_key in reusable:
                source_path, source = reusable[reusable_key]
                reused = {
                    key: value
                    for key, value in source.items()
                    if key not in ("job", "command")
                }
                reused.update(
                    {
                        "job": job.to_dict(),
                        "seed": seed,
                        "plan_sha256": plan_sha256,
                        "reused_from": str(source_path.relative_to(self.output_dir)),
                    }
                )
                _write_json(status_path, reused)
                print(
                    "execution {}/{} {} seed={} reused {}".format(
                        position,
                        len(rollouts),
                        job.id,
                        seed,
                        source_path.name,
                    )
                )
                completed_this_run += 1
                self.summarize()
                continue
            command = evaluation_command(
                plan_path=plan_path,
                plan_data=plan_data,
                task_family="set_table",
                run_name="{}/{}{}".format(
                    self.spec.name,
                    job.id,
                    self._rollout_suffix(seed),
                ),
                seed=seed,
                max_trajectories=1,
                num_envs=1,
                experiment_root=self.simulator_workspace,
                record_video=record_video,
            )
            log_path = self._log_path(job.id, seed)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                "execution {}/{} {} seed={}".format(
                    position,
                    len(rollouts),
                    job.id,
                    seed,
                )
            )
            started = time.time()
            _write_json(
                status_path,
                {
                    "status": "running",
                    "job": job.to_dict(),
                    "seed": seed,
                    "plan_sha256": plan_sha256,
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
                    "seed": seed,
                    "plan_sha256": plan_sha256,
                    "returncode": process.returncode,
                    "elapsed_seconds": time.time() - started,
                    "log": str(log_path.relative_to(self.output_dir)),
                    **parsed,
                },
            )
            completed_status = _read_json(status_path)
            if (
                completed_status.get("status") == "complete"
                and completed_status.get("simulator_success") is not None
            ):
                reusable[reusable_key] = (status_path, completed_status)
            completed_this_run += 1
            self.summarize()

    def _execution_rows(self) -> List[Dict[str, Any]]:
        jobs_path = self.output_dir / "execution_jobs.json"
        if not jobs_path.exists():
            return []
        case_types = {case.id: case.case_type for case in self.spec.cases}
        rows = []
        for item in json.loads(jobs_path.read_text(encoding="utf-8")):
            for seed in self.execution_seeds:
                status_path = self._status_path(str(item["id"]), seed)
                status = _read_json(status_path) if status_path.exists() else {}
                execution_status = status.get("status", "pending")
                if not item["groundable"]:
                    execution_status = "not_groundable"
                rows.append(
                    {
                        **item,
                        "case_type": case_types[str(item["instruction_id"])],
                        "seed": seed,
                        "execution_status": execution_status,
                        "simulator_success": status.get("simulator_success"),
                        "success_at_end": status.get("success_at_end"),
                        "episode_steps": status.get("episode_steps"),
                        "reused_from": status.get("reused_from"),
                    }
                )
        return rows

    @staticmethod
    def _conditional_simulator_success(
        rows: Sequence[Mapping[str, Any]],
    ) -> Optional[float]:
        groundable = [row for row in rows if row["groundable"]]
        if not groundable or any(
            row["execution_status"] not in ("complete", "failed")
            for row in groundable
        ):
            return None
        return sum(
            float(row.get("simulator_success") or 0.0) for row in groundable
        ) / len(groundable)

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

    @staticmethod
    def _method_label(condition: str) -> str:
        return {
            "flat_library": "DeepSeek + Flat Skill Library",
            "full_graph": "DeepSeek + Full Skill Graph",
            "graph_oracle": "GT Graph Oracle",
        }[condition]

    def summarize(self) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        planning_rows = self._planning_rows()
        execution_rows = self._execution_rows()
        _write_csv(self.tables_dir / "planning_results.csv", planning_rows)
        if execution_rows:
            _write_csv(self.tables_dir / "execution_results.csv", execution_rows)

        overall = []
        for condition in CONDITIONS:
            rows = [row for row in planning_rows if row["condition"] == condition]
            terminal = all(
                row["status"] in ("complete", "api_failed") for row in rows
            )
            overall.append(
                {
                    "method": self._method_label(condition),
                    "cases": len(rows),
                    "valid_path_rate": (
                        _mean([1.0 if row["graph_valid"] else 0.0 for row in rows])
                        if terminal
                        else None
                    ),
                    "path_lcs_f1": (
                        _mean(
                            [row["path_lcs_f1"] for row in rows],
                            missing_as_zero=True,
                        )
                        if terminal
                        else None
                    ),
                    "grounding_accuracy": (
                        _mean(
                            [row["grounding_accuracy"] for row in rows],
                            missing_as_zero=True,
                        )
                        if terminal
                        else None
                    ),
                }
            )
        _write_csv(self.tables_dir / "planning_overall.csv", overall)

        breakdown = []
        for condition in CONDITIONS:
            for case_type in ("nominal", "fallback"):
                rows = [
                    row
                    for row in planning_rows
                    if row["condition"] == condition
                    and row["case_type"] == case_type
                ]
                terminal = all(
                    row["status"] in ("complete", "api_failed")
                    for row in rows
                )
                breakdown.append(
                    {
                        "method": self._method_label(condition),
                        "case_type": case_type.title(),
                        "cases": len(rows),
                        "valid_path_rate": (
                            _mean(
                                [1.0 if row["graph_valid"] else 0.0 for row in rows]
                            )
                            if terminal
                            else None
                        ),
                        "path_lcs_f1": (
                            _mean(
                                [row["path_lcs_f1"] for row in rows],
                                missing_as_zero=True,
                            )
                            if terminal
                            else None
                        ),
                    }
                )
        _write_csv(self.tables_dir / "planning_breakdown.csv", breakdown)

        execution_table = []
        calibration_table = []
        if execution_rows:
            calibration_rows = [
                row
                for row in execution_rows
                if row["condition"] == "graph_oracle"
                and row["case_type"] == "nominal"
            ]
            calibration_plans = {
                str(row["id"]): row for row in calibration_rows
            }
            calibration_table.append(
                {
                    "method": "GT Graph Oracle (Nominal)",
                    "cases": len(calibration_plans),
                    "rollouts": len(calibration_rows),
                    "simulator_success": self._conditional_simulator_success(
                        calibration_rows
                    ),
                    "completed": sum(
                        row["execution_status"] in ("complete", "failed")
                        for row in calibration_rows
                    ),
                }
            )
            _write_csv(
                self.tables_dir / "controller_calibration.csv",
                calibration_table,
            )
            for condition in ("graph_oracle",) + CONDITIONS:
                rows = [
                    row for row in execution_rows if row["condition"] == condition
                ]
                plan_rows = {str(row["id"]): row for row in rows}
                groundable_rollouts = [
                    row for row in rows if bool(row["groundable"])
                ]
                execution_table.append(
                    {
                        "method": self._method_label(condition),
                        "cases": len(plan_rows),
                        "rollouts": len(rows),
                        "groundable_rate": sum(
                            bool(row["groundable"])
                            for row in plan_rows.values()
                        )
                        / len(plan_rows),
                        "conditional_simulator_success": (
                            self._conditional_simulator_success(rows)
                        ),
                        "end_to_end_success": self._end_to_end_success(rows),
                        "completed": sum(
                            row["execution_status"] in ("complete", "failed")
                            for row in groundable_rollouts
                        ),
                        "groundable_rollouts": len(groundable_rollouts),
                    }
                )
            _write_csv(self.tables_dir / "execution_overall.csv", execution_table)

        planning_md = _markdown_table(
            [
                "Method",
                "Cases",
                "Valid Path ↑",
                "Path LCS-F1 ↑",
                "Grounding Accuracy ↑",
            ],
            [
                [
                    row["method"],
                    row["cases"],
                    _format_metric(row["valid_path_rate"]),
                    _format_metric(row["path_lcs_f1"]),
                    _format_metric(row["grounding_accuracy"]),
                ]
                for row in overall
            ],
        )
        breakdown_md = _markdown_table(
            ["Method", "Case Type", "Cases", "Valid Path ↑", "Path LCS-F1 ↑"],
            [
                [
                    row["method"],
                    row["case_type"],
                    row["cases"],
                    _format_metric(row["valid_path_rate"]),
                    _format_metric(row["path_lcs_f1"]),
                ]
                for row in breakdown
            ],
        )
        calibration_md = _markdown_table(
            ["Method", "Cases", "Rollouts", "Simulator Success ↑", "Completed"],
            [
                [
                    row["method"],
                    row["cases"],
                    row["rollouts"],
                    _format_metric(row["simulator_success"]),
                    "{}/{}".format(row["completed"], row["rollouts"]),
                ]
                for row in calibration_table
            ],
        )
        execution_md = _markdown_table(
            [
                "Method",
                "Cases",
                "Rollouts",
                "Groundable Rate ↑",
                "Conditional Simulator Success ↑",
                "End-to-End Success ↑",
                "Completed",
            ],
            [
                [
                    row["method"],
                    row["cases"],
                    row["rollouts"],
                    _format_metric(row["groundable_rate"]),
                    _format_metric(row["conditional_simulator_success"]),
                    _format_metric(row["end_to_end_success"]),
                    "{}/{}".format(
                        row["completed"],
                        row["groundable_rollouts"],
                    ),
                ]
                for row in execution_table
            ],
        )
        (self.tables_dir / "planning_overall.md").write_text(
            planning_md + "\n", encoding="utf-8"
        )
        (self.tables_dir / "planning_breakdown.md").write_text(
            breakdown_md + "\n", encoding="utf-8"
        )
        (self.tables_dir / "execution_overall.md").write_text(
            execution_md + "\n", encoding="utf-8"
        )
        (self.tables_dir / "controller_calibration.md").write_text(
            calibration_md + "\n", encoding="utf-8"
        )
        (self.tables_dir / "report.md").write_text(
            "# SetTable Flat Library vs Full Skill Graph\n\n"
            + "Planning samples: `{}` split; simulator execution: `{}` "
            "split.\n\n".format(
                self.spec.split,
                self.execution_split,
            )
            + "## Table 1 — Planning ablation\n\n"
            + planning_md
            + "\n\n## Table 2 — Nominal vs fallback planning\n\n"
            + breakdown_md
            + "\n\n## Table 3 — Controller calibration\n\n"
            + calibration_md
            + "\n\n## Table 4 — Paired MS-HAB execution\n\n"
            + execution_md
            + "\n\nThe two VLM conditions differ only in whether Layer-2 edges "
            "are visible. Invalid or ungroundable outputs count as end-to-end "
            "execution failures. Conditional success is computed only over "
            "grounded rollouts. Identical grounded plans reuse the same "
            "rollout for each seed.\n",
            encoding="utf-8",
        )


def _execution_seeds(value: str) -> Tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "execution seeds must be comma-separated integers"
        ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one execution seed is required")
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "execution seeds must be unique non-negative integers"
        )
    return seeds


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
        "--execution-split",
        choices=("train", "val"),
        help="MS-HAB split used only for grounding and simulator execution",
    )
    parser.add_argument(
        "--planning-source-dir",
        type=Path,
        help=(
            "existing experiment output whose planning/ artifacts should be "
            "reused without new API calls"
        ),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("MS_ASSET_DIR", str(DEFAULT_ASSET_ROOT))),
    )
    parser.add_argument("--condition", choices=CONDITIONS + ("graph_oracle",))
    parser.add_argument("--case-id")
    parser.add_argument("--case-type", choices=("nominal", "fallback"))
    parser.add_argument(
        "--execution-seeds",
        type=_execution_seeds,
        help="comma-separated paired simulator seeds, for example 0,1,2,3,4",
    )
    parser.add_argument("--force-planning", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.phase == "planning" and args.condition == "graph_oracle":
        raise ValueError("graph_oracle is deterministic and makes no API call")
    if args.planning_source_dir is not None and args.phase in ("planning", "all"):
        raise ValueError(
            "--planning-source-dir is for prepare/execute/summarize; "
            "planning always writes into --output-dir"
        )
    spec = AblationSpec.from_yaml(args.manifest)
    experiment = SetTableGraphAblation(
        spec=spec,
        output_dir=args.output_dir,
        graph_path=args.graph,
        config_path=args.config,
        asset_root=args.asset_root,
        execution_split=args.execution_split,
        planning_source_dir=args.planning_source_dir,
        execution_seeds=args.execution_seeds,
    )
    experiment.initialize()
    if args.phase in ("planning", "all"):
        experiment.planning(
            condition_filter=args.condition,
            case_filter=args.case_id,
            force=args.force_planning,
            fail_fast=args.fail_fast,
        )
    if args.phase in ("prepare", "all"):
        experiment.prepare_execution()
    if args.phase in ("execute", "all"):
        experiment.execute(
            max_jobs=args.max_jobs,
            condition=args.condition,
            case_id=args.case_id,
            case_type=args.case_type,
            record_video=args.record_video,
            retry_failed=args.retry_failed,
        )
    experiment.summarize()
    print("report: {}".format(experiment.tables_dir / "report.md"))


if __name__ == "__main__":
    main()
