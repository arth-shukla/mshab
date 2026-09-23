"""Compile a graph-constrained flow into native MS-HAB ``PlanData`` routes.

MS-HAB's ``SequentialTask-v0`` consumes a linear subtask list.  The semantic
flow remains the authoritative branching representation; this module lowers
its nominal route and each Layer-2 fallback route into separate, executable
linear plans without redefining Contracts or Policies.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mshab.experiments.granularity.vlm.prompt import FLOW_SCHEMA_VERSION
from mshab.experiments.granularity.vlm.run import (
    _comparison_target,
    _entities_match,
    _gt_arguments,
    _object_category,
    _require_mapping,
    _require_string,
    _selected_plan,
)


SIMULATOR_BUNDLE_SCHEMA_VERSION = "mshab.simulator-route-bundle.v1"
SUPPORTED_FLOW_SCHEMA_VERSIONS = frozenset(
    ("mshab.execution-flow.v1", FLOW_SCHEMA_VERSION)
)
SUBTASK_HORIZONS = {
    "navigate": 500,
    "pick": 200,
    "place": 200,
    "open": 200,
    "close": 200,
}


def route_filename(route_id: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", route_id).strip("._")
    if not value:
        raise ValueError("route id cannot produce an empty filename")
    return value + ".json"


@dataclass(frozen=True)
class ExecutionRoute:
    """One linear route selected from the branching execution flow."""

    id: str
    kind: str
    node_order: Tuple[str, ...]
    fallback_occurrence_id: Optional[str] = None
    failed_node_id: Optional[str] = None


class FlowProgram:
    """Validated read-only view over nominal and fallback execution routes."""

    def __init__(self, flow: Mapping[str, Any]) -> None:
        if flow.get("schema_version") not in SUPPORTED_FLOW_SCHEMA_VERSIONS:
            raise ValueError("unsupported execution-flow schema_version")
        self.flow = flow
        self.nodes: Dict[str, Mapping[str, Any]] = {}
        for raw_node in flow.get("nodes", []):
            node = _require_mapping(raw_node, "flow node")
            node_id = _require_string(node.get("id"), "flow node id")
            if node_id in self.nodes:
                raise ValueError("duplicate flow node {}".format(node_id))
            self.nodes[node_id] = node
        if not self.nodes:
            raise ValueError("execution flow has no nodes")

        nominal = tuple(flow.get("nominal_order", ()))
        self._validate_order(nominal, "nominal_order")
        self._nominal = nominal

        selected = flow.get("selected_strategies", [])
        self.occurrence_order = tuple(
            _require_string(item.get("occurrence_id"), "occurrence_id")
            for item in selected
        )
        if len(self.occurrence_order) != len(set(self.occurrence_order)):
            raise ValueError("selected strategy occurrences must be unique")
        nominal_occurrences = tuple(
            dict.fromkeys(self.nodes[node_id]["occurrence_id"] for node_id in nominal)
        )
        if self.occurrence_order and nominal_occurrences != self.occurrence_order:
            raise ValueError(
                "selected_strategies order disagrees with nominal_order"
            )
        if not self.occurrence_order:
            self.occurrence_order = nominal_occurrences

        self._branches: Dict[str, Mapping[str, Any]] = {}
        for raw_branch in flow.get("fallback_branches", []):
            branch = _require_mapping(raw_branch, "fallback branch")
            failed = _require_string(
                branch.get("failed_node_id"),
                "failed_node_id",
            )
            if failed not in self.nodes:
                raise ValueError("fallback branch has an unknown failed node")
            occurrence_id = str(self.nodes[failed]["occurrence_id"])
            if occurrence_id in self._branches:
                raise ValueError(
                    "occurrence {} has more than one fallback branch".format(
                        occurrence_id
                    )
                )
            recovery = tuple(branch.get("recovery_order", ()))
            self._validate_order(recovery, "recovery_order")
            if any(self.nodes[item]["kind"] != "recovery" for item in recovery):
                raise ValueError("recovery_order must contain recovery nodes")
            self._branches[occurrence_id] = branch

    def _validate_order(self, order: Sequence[str], label: str) -> None:
        if not order:
            raise ValueError("{} must be non-empty".format(label))
        if len(order) != len(set(order)):
            raise ValueError("{} cannot contain duplicate nodes".format(label))
        missing = [node_id for node_id in order if node_id not in self.nodes]
        if missing:
            raise ValueError("{} references unknown nodes {}".format(label, missing))

    def _nominal_for_occurrence(self, occurrence_id: str) -> List[str]:
        return [
            node_id
            for node_id in self._nominal
            if self.nodes[node_id]["occurrence_id"] == occurrence_id
        ]

    def _legacy_fallback_route(
        self,
        occurrence_id: str,
        branch: Mapping[str, Any],
    ) -> List[str]:
        """Recover v1 flow routes that predate ``fallback_route_order``.

        In the current graph, a recovery starts with a new Navigate.  Shared
        container-opening prerequisites are retained, while the primary's
        final Navigate + achiever pair is replaced by the recovery pair.
        """

        nominal = self._nominal_for_occurrence(occurrence_id)
        failed = str(branch["failed_node_id"])
        failed_index = nominal.index(failed)
        prefix = nominal[:failed_index]
        if prefix and self.nodes[prefix[-1]]["contract_type"] == "navigate":
            prefix = prefix[:-1]
        return prefix + list(branch["recovery_order"])

    def routes(self) -> List[ExecutionRoute]:
        routes = [
            ExecutionRoute(
                id="nominal",
                kind="nominal",
                node_order=self._nominal,
            )
        ]
        for occurrence_id in self.occurrence_order:
            branch = self._branches.get(occurrence_id)
            if branch is None:
                continue
            fallback_local = list(branch.get("fallback_route_order", ()))
            if fallback_local:
                self._validate_order(
                    fallback_local,
                    "fallback_route_order",
                )
            else:
                fallback_local = self._legacy_fallback_route(
                    occurrence_id,
                    branch,
                )
            order: List[str] = []
            for current in self.occurrence_order:
                if current == occurrence_id:
                    order.extend(fallback_local)
                else:
                    order.extend(self._nominal_for_occurrence(current))
            self._validate_order(order, "compiled fallback route")
            routes.append(
                ExecutionRoute(
                    id="fallback.{}".format(occurrence_id),
                    kind="fallback",
                    node_order=tuple(order),
                    fallback_occurrence_id=occurrence_id,
                    failed_node_id=str(branch["failed_node_id"]),
                )
            )
        return routes

    def route(self, route_id: str) -> ExecutionRoute:
        routes = {route.id: route for route in self.routes()}
        try:
            return routes[route_id]
        except KeyError as exc:
            raise KeyError(
                "unknown route {!r}; available={}".format(
                    route_id,
                    sorted(routes),
                )
            ) from exc


class MSHabPlanGrounder:
    """Ground FlowProgram nodes with one official MS-HAB scene plan."""

    def __init__(
        self,
        source: Mapping[str, Any],
        *,
        task_family: str,
        plan_index: int,
    ) -> None:
        self.source = source
        self.task_family = task_family
        self.plan_index = plan_index
        self.plan = _selected_plan(source, plan_index)
        raw_subtasks = self.plan.get("subtasks")
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            raise ValueError("source task plan has no subtasks")
        self.source_steps: List[Dict[str, Any]] = []
        for index, raw_subtask in enumerate(raw_subtasks):
            subtask = _require_mapping(raw_subtask, "source subtask")
            arguments, target = _gt_arguments(
                raw_subtasks,
                index,
                task_family,
            )
            target_entity_id: Optional[str] = None
            contract_type = str(subtask["type"])
            if contract_type in ("pick", "place"):
                target_entity_id = subtask.get("obj_id")
            elif contract_type in ("open", "close"):
                target_entity_id = (
                    subtask.get("articulation_id")
                    or subtask.get("articulation_type")
                )
            elif contract_type == "navigate" and index + 1 < len(raw_subtasks):
                following = _require_mapping(
                    raw_subtasks[index + 1],
                    "subtask after Navigate",
                )
                following_type = following.get("type")
                if following_type == "pick":
                    target_entity_id = following.get("obj_id")
                elif following_type in ("open", "close"):
                    target_entity_id = (
                        following.get("articulation_id")
                        or following.get("articulation_type")
                    )
                else:
                    target_entity_id = target
            self.source_steps.append(
                {
                    "index": index,
                    "subtask": subtask,
                    "contract_type": contract_type,
                    "arguments": arguments,
                    "grounded_target": target,
                    "target_entity_id": target_entity_id,
                }
            )

    @staticmethod
    def _place_destination_matches(
        node: Mapping[str, Any],
        source_step: Mapping[str, Any],
    ) -> bool:
        expected = node.get("arguments", {}).get("destination")
        actual = source_step.get("arguments", {}).get("destination")
        return expected is None or actual is None or expected == actual

    def _source_step_for(self, node: Mapping[str, Any]) -> Mapping[str, Any]:
        contract_type = _require_string(
            node.get("contract_type"),
            "flow contract_type",
        )
        target_field = {
            "navigate": "goal",
            "pick": "object",
            "place": "object",
            "open": "articulation",
            "close": "articulation",
        }[contract_type]
        target = node.get("arguments", {}).get(target_field)
        if target is None:
            target = node.get("grounded_target")
        if target is None:
            target = _comparison_target(contract_type, node.get("arguments", {}))
        target_text = str(target)
        target_is_instance = bool(re.search(r"-\d+$", target_text))
        candidates = [
            item
            for item in self.source_steps
            if item["contract_type"] == contract_type
            and (
                item.get("target_entity_id") == target_text
                if target_is_instance
                else _entities_match(item["grounded_target"], target)
            )
            and (
                contract_type != "place"
                or self._place_destination_matches(node, item)
            )
        ]
        if not candidates:
            raise ValueError(
                "cannot ground flow node {} ({}, target={!r}) in source plan {}".format(
                    node["id"],
                    contract_type,
                    target,
                    self.plan_index,
                )
            )
        if not target_is_instance:
            identities = {
                str(item.get("target_entity_id") or item["grounded_target"])
                for item in candidates
            }
            if len(identities) > 1:
                raise ValueError(
                    "ambiguous semantic target {!r} resolves to {} in source "
                    "plan {}".format(
                        _object_category(target_text),
                        sorted(identities),
                        self.plan_index,
                    )
                )
        return candidates[0]

    @staticmethod
    def _validate_linear_route(
        node_order: Sequence[str],
        nodes: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for index, node_id in enumerate(node_order):
            if nodes[node_id]["contract_type"] != "navigate":
                continue
            if index + 1 >= len(node_order):
                raise ValueError("MS-HAB route cannot end with Navigate")
            following = nodes[node_order[index + 1]]["contract_type"]
            if following == "navigate":
                raise ValueError(
                    "MS-HAB derives a Navigate goal from the next subtask; "
                    "consecutive Navigate nodes are not executable"
                )

    @staticmethod
    def _grounded_policy_id(
        node: Mapping[str, Any],
        source_step: Mapping[str, Any],
    ) -> Optional[str]:
        """Resolve object-specific policies after scene grounding.

        A semantic VLM binding such as ``bowl`` cannot select
        ``pick.024_bowl`` before the scene is known.  The source MS-HAB step
        supplies that missing identity, so prefer the exact candidate here
        and keep the generic policy only when no exact candidate exists.
        """

        current = node.get("policy_id")
        contract_type = str(node.get("contract_type", ""))
        if contract_type not in ("pick", "place"):
            return None if current is None else str(current)
        target = source_step.get("target_entity_id")
        if target is None:
            target = source_step.get("grounded_target")
        category = _object_category(None if target is None else str(target))
        candidates = [
            str(item) for item in node.get("candidate_policy_ids", ())
        ]
        if category:
            suffix = ".{}.{}".format(contract_type, category)
            exact = [item for item in candidates if item.endswith(suffix)]
            if exact:
                return sorted(exact)[0]
        return None if current is None else str(current)

    @staticmethod
    def _recommended_policy_type(policy_ids: Sequence[Optional[str]]) -> str:
        uses_generic_object_policy = any(
            str(policy_id or "").endswith(".all") for policy_id in policy_ids
        )
        return "rl_all_obj" if uses_generic_object_policy else "rl_per_obj"

    def compile(
        self,
        program: FlowProgram,
        route: ExecutionRoute,
    ) -> Dict[str, Any]:
        self._validate_linear_route(route.node_order, program.nodes)
        subtasks: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        grounded_object_policy_ids: List[Optional[str]] = []
        route_slug = route_filename(route.id).removesuffix(".json")
        for step_index, node_id in enumerate(route.node_order):
            node = program.nodes[node_id]
            source_step = self._source_step_for(node)
            grounded_policy_id = self._grounded_policy_id(node, source_step)
            if node["contract_type"] in ("pick", "place"):
                grounded_object_policy_ids.append(grounded_policy_id)
            subtask = copy.deepcopy(source_step["subtask"])
            uid = "vlm-{}-{:03d}".format(route_slug, step_index)
            subtask["uid"] = uid
            subtask["composite_subtask_uids"] = [uid]
            subtasks.append(subtask)
            decisions.append(
                {
                    "step": step_index,
                    "flow_node_id": node_id,
                    "skill_node_id": node.get("skill_node_id"),
                    "contract_id": node.get("contract_id"),
                    "contract_type": node["contract_type"],
                    "policy_id": grounded_policy_id,
                    "semantic_policy_id": node.get("policy_id"),
                    "arguments": copy.deepcopy(node.get("arguments", {})),
                    "source_subtask_index": source_step["index"],
                    "source_subtask_uid": source_step["subtask"].get("uid"),
                }
            )
        return {
            "dataset": self.source.get("dataset"),
            "plans": [
                {
                    "subtasks": subtasks,
                    "build_config_name": self.plan.get("build_config_name"),
                    "init_config_name": self.plan.get("init_config_name"),
                }
            ],
            "selection": {
                "schema_version": SIMULATOR_BUNDLE_SCHEMA_VERSION,
                "task_family": self.task_family,
                "source_plan_index": self.plan_index,
                "route_id": route.id,
                "route_kind": route.kind,
                "fallback_occurrence_id": route.fallback_occurrence_id,
                "failed_node_id": route.failed_node_id,
                "flow_node_order": list(route.node_order),
                "skill_decisions": decisions,
                "recommended_policy_type": self._recommended_policy_type(
                    grounded_object_policy_ids
                ),
                "policy_resolution": "post_scene_grounding",
                "estimated_max_episode_steps": sum(
                    SUBTASK_HORIZONS[subtask["type"]] for subtask in subtasks
                ),
                "execution_semantics": (
                    "nominal_path"
                    if route.kind == "nominal"
                    else "precompiled_fallback_path_for_failure_injection"
                ),
            },
        }

    def compile_all(self, program: FlowProgram) -> Dict[str, Dict[str, Any]]:
        return {
            route.id: self.compile(program, route) for route in program.routes()
        }


def write_simulator_bundle(
    flow: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    task_family: str,
    plan_index: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Write all simulator-ready routes and return a serializable manifest."""

    program = FlowProgram(flow)
    grounder = MSHabPlanGrounder(
        source,
        task_family=task_family,
        plan_index=plan_index,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = grounder.compile_all(program)
    manifest_routes = []
    for route in program.routes():
        filename = route_filename(route.id)
        path = output_dir / filename
        path.write_text(
            json.dumps(routes[route.id], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_routes.append(
            {
                "route_id": route.id,
                "kind": route.kind,
                "task_plan": filename,
                "fallback_occurrence_id": route.fallback_occurrence_id,
                "failed_node_id": route.failed_node_id,
                "num_steps": len(route.node_order),
            }
        )
    manifest = {
        "schema_version": SIMULATOR_BUNDLE_SCHEMA_VERSION,
        "task_family": task_family,
        "source_plan_index": plan_index,
        "routes": manifest_routes,
        "note": (
            "MS-HAB PlanData is linear. Each fallback is therefore emitted as "
            "a separate executable route for controlled failure-injection runs; "
            "the branching semantics remain in predicted_flow.json."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
