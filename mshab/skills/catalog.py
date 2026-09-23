"""Canonical, machine-independent skill-graph catalog documents.

The checked-in catalog is a portable description of the four-layer library,
not an inventory of one workstation.  Checkpoint readiness is deliberately
excluded: it belongs to :class:`ContractLibrary` runtime discovery.  Loading a
catalog reconstructs and validates Layer 1/2 rather than trusting duplicated
derived views such as flattened nodes, edges, execution order, or the
per-policy list of contracts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from mshab.skills import schema
from mshab.skills.graph import SubGoalGraph, SkillGraph
from mshab.skills.plan import SkillPlan


CATALOG_SCHEMA_VERSION = "mshab.library-catalog.v1"
_LAYER_KEYS = (
    "1_subgoal_graph",
    "2_skill_graph",
    "3_grounded_skills",
    "4_contracts_and_policies",
)


class LibraryCatalog:
    """Validated four-layer artifact suitable for source control and VLM input."""

    def __init__(
        self,
        *,
        task: str,
        construction: Mapping[str, Any],
        execution_plan_note: str,
        execution_plans: Mapping[str, SkillPlan],
        subgoal_graph: SubGoalGraph,
        skill_graph: SkillGraph,
        grounded_skills: Mapping[str, Mapping[str, Any]],
        contracts: Sequence[Mapping[str, Any]],
        policies: Sequence[Mapping[str, Any]],
    ) -> None:
        schema.require_identifier({"task": task}, "task", where="library_catalog")
        if skill_graph.task != task:
            raise ValueError("catalog task must match its skill graph")
        if skill_graph.subgoal_graph is not subgoal_graph:
            raise ValueError("catalog graphs must share the same sub-goal graph object")
        skill_graph.validate()
        if not isinstance(construction, Mapping):
            raise TypeError("catalog construction metadata must be a mapping")
        if not isinstance(execution_plan_note, str) or not execution_plan_note:
            raise ValueError("catalog execution-plan note must be a non-empty string")
        if not isinstance(execution_plans, Mapping):
            raise TypeError("catalog execution_plans must be a mapping")
        if not isinstance(grounded_skills, Mapping):
            raise TypeError("catalog grounded_skills must be a mapping")

        self.task = task
        self.construction = _json_copy(construction, "construction")
        self.execution_plan_note = execution_plan_note
        self.execution_plans = dict(execution_plans)
        self.subgoal_graph = subgoal_graph
        self.skill_graph = skill_graph
        self.grounded_skills = _json_copy(grounded_skills, "grounded_skills")
        self.contracts = tuple(_json_copy(contracts, "contracts"))
        self.policies = tuple(_json_copy(policies, "policies"))
        self._validate_plans()
        self._validate_grounded_skills()
        self._validate_contracts_and_policies()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LibraryCatalog":
        where = "library_catalog"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=(
                "schema_version",
                "task",
                "construction",
                "execution_plans",
                "summary",
                "layers",
            ),
        )
        version = schema.require_str(payload, "schema_version", where=where)
        if version != CATALOG_SCHEMA_VERSION:
            raise schema.SchemaError(
                "{} declares unsupported schema_version {!r}; expected {!r}".format(
                    where, version, CATALOG_SCHEMA_VERSION
                )
            )
        task = schema.require_identifier(payload, "task", where=where)
        layers = schema.require_mapping(payload["layers"], where="skill_catalog.layers")
        schema.require_keys(layers, where="skill_catalog.layers", required=_LAYER_KEYS)

        raw_layer_1 = schema.require_mapping(
            layers[_LAYER_KEYS[0]], where="skill_catalog.layer_1"
        )
        schema.require_keys(
            raw_layer_1,
            where="skill_catalog.layer_1",
            required=("goal", "subgoals", "dependencies", "execution_order"),
        )
        subgoal_graph = SubGoalGraph.from_dict(raw_layer_1)
        declared_subgoal_order = schema.require_str_tuple(
            raw_layer_1, "execution_order", where="skill_catalog.layer_1"
        )
        if declared_subgoal_order != subgoal_graph.execution_order():
            raise schema.SchemaError("catalog Layer-1 execution_order is stale")

        raw_layer_2 = schema.require_mapping(
            layers[_LAYER_KEYS[1]], where="skill_catalog.layer_2"
        )
        schema.require_keys(
            raw_layer_2,
            where="skill_catalog.layer_2",
            required=(
                "task",
                "subgraphs",
                "cross_edges",
                "_derived",
                "nodes",
                "edges",
                "candidate_partial_order",
            ),
        )
        skill_graph = SkillGraph.from_dict(
            raw_layer_2, subgoal_graph=subgoal_graph
        )
        declared_candidate_order = schema.require_str_tuple(
            raw_layer_2,
            "candidate_partial_order",
            where="skill_catalog.layer_2",
        )
        if declared_candidate_order != skill_graph.execution_order():
            raise schema.SchemaError("catalog candidate_partial_order is stale")
        regenerated_layer_2 = skill_graph.as_dict()
        regenerated_layer_2.pop("subgoal_graph", None)
        for key in ("nodes", "edges"):
            if raw_layer_2[key] != regenerated_layer_2[key]:
                raise schema.SchemaError(
                    "catalog Layer-2 derived {} view is stale".format(key)
                )
        skill_graph.validate()

        plan_section = schema.require_mapping(
            payload["execution_plans"], where="skill_catalog.execution_plans"
        )
        schema.require_keys(
            plan_section,
            where="skill_catalog.execution_plans",
            required=("note",),
            optional=tuple(key for key in plan_section if key != "note"),
        )
        note = schema.require_str(plan_section, "note", where="skill_catalog.execution_plans")
        plans = {
            name: SkillPlan.from_dict(value)
            for name, value in plan_section.items()
            if name != "note"
        }
        if not plans:
            raise schema.SchemaError("catalog must contain at least one execution plan")

        raw_layer_4 = schema.require_mapping(
            layers[_LAYER_KEYS[3]], where="skill_catalog.layer_4"
        )
        schema.require_keys(
            raw_layer_4,
            where="skill_catalog.layer_4",
            required=("contracts", "policies"),
        )

        catalog = cls(
            task=task,
            construction=schema.require_mapping(
                payload["construction"], where="skill_catalog.construction"
            ),
            execution_plan_note=note,
            execution_plans=plans,
            subgoal_graph=subgoal_graph,
            skill_graph=skill_graph,
            grounded_skills=schema.require_mapping(
                layers[_LAYER_KEYS[2]], where="skill_catalog.layer_3"
            ),
            contracts=schema.require_sequence(
                raw_layer_4, "contracts", where="skill_catalog.layer_4"
            ),
            policies=schema.require_sequence(
                raw_layer_4, "policies", where="skill_catalog.layer_4"
            ),
        )
        if _json_copy(payload["summary"], "summary") != catalog.summary:
            raise schema.SchemaError("catalog summary is stale")
        return catalog

    @property
    def summary(self) -> Dict[str, int]:
        nominal = self.execution_plans.get("nominal")
        if nominal is None:
            nominal = next(iter(self.execution_plans.values()))
        return {
            "subgoals": len(self.subgoal_graph.subgoals),
            "skill_subgraphs": len(self.skill_graph.subgraphs),
            "cross_edges": len(self.skill_graph.cross_edges),
            "skill_nodes": len(self.skill_graph.nodes),
            "skill_edges": len(self.skill_graph.edges),
            "planned_steps": len(nominal.order),
            "grounded_skills": len(self.grounded_skills),
            "registered_contracts": len(self.contracts),
            "registered_policies": len(self.policies),
            "policy_bindings": sum(
                len(record["policies"]) for record in self.contracts
            ),
        }

    def as_dict(self) -> Dict[str, Any]:
        layer_1 = self.subgoal_graph.as_dict()
        layer_1["execution_order"] = list(self.subgoal_graph.execution_order())
        layer_2 = self.skill_graph.as_dict()
        layer_2.pop("subgoal_graph", None)
        layer_2["candidate_partial_order"] = list(
            self.skill_graph.execution_order()
        )
        plans = {"note": self.execution_plan_note}
        plans.update(
            (name, plan.as_dict()) for name, plan in self.execution_plans.items()
        )
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "task": self.task,
            "construction": _json_copy(self.construction, "construction"),
            "execution_plans": plans,
            "summary": self.summary,
            "layers": {
                _LAYER_KEYS[0]: layer_1,
                _LAYER_KEYS[1]: layer_2,
                _LAYER_KEYS[2]: _json_copy(
                    self.grounded_skills, "grounded_skills"
                ),
                _LAYER_KEYS[3]: {
                    "contracts": list(_json_copy(self.contracts, "contracts")),
                    "policies": list(_json_copy(self.policies, "policies")),
                },
            },
        }

    def _validate_plans(self) -> None:
        subgoals = self.subgoal_graph.subgoals
        nodes = self.skill_graph.nodes
        for name, plan in self.execution_plans.items():
            if not isinstance(name, str) or not name:
                raise TypeError("execution plan names must be non-empty strings")
            if not isinstance(plan, SkillPlan):
                raise TypeError("execution_plans must contain SkillPlan values")
            if set(plan.selections) != set(subgoals):
                raise ValueError(
                    "execution plan {!r} must select one achiever for every sub-goal".format(
                        name
                    )
                )
            unknown = sorted(set(plan.order) - set(nodes))
            if unknown:
                raise ValueError(
                    "execution plan {!r} contains unknown nodes {}".format(name, unknown)
                )
            for subgoal_id, node_id in plan.selections.items():
                if subgoal_id not in nodes[node_id].achieves:
                    raise ValueError(
                        "execution plan {!r} selects node {!r} for unrelated sub-goal {!r}".format(
                            name, node_id, subgoal_id
                        )
                    )
                executed_achievers = [
                    candidate
                    for candidate in plan.order
                    if subgoal_id in nodes[candidate].achieves
                ]
                if executed_achievers != [node_id]:
                    raise ValueError(
                        "execution plan {!r} must execute exactly its selected "
                        "achiever for sub-goal {!r}".format(name, subgoal_id)
                    )
            completed = set()
            for node_id in plan.order:
                unsatisfied = [
                    sorted(group)
                    for group in self.skill_graph.prerequisite_groups(node_id)
                    if not group & completed
                ]
                if unsatisfied:
                    raise ValueError(
                        "execution plan {!r} schedules {!r} before prerequisite "
                        "groups {}".format(name, node_id, unsatisfied)
                    )
                completed.add(node_id)

    def _validate_grounded_skills(self) -> None:
        if set(self.grounded_skills) != set(self.skill_graph.nodes):
            raise ValueError("catalog must contain one grounded skill per skill node")
        for node_id, raw in self.grounded_skills.items():
            where = "grounded_skills.{}".format(node_id)
            record = schema.require_mapping(raw, where=where)
            schema.require_keys(
                record,
                where=where,
                required=(
                    "contract_id",
                    "arguments",
                    "preconditions",
                    "effects",
                    "invariants",
                    "verification",
                    "failure_modes",
                    "deletes",
                ),
            )
            node = self.skill_graph.nodes[node_id]
            if schema.require_contract_id(record, "contract_id", where=where) != node.contract_id:
                raise ValueError("catalog grounded skill contract_id does not match its node")
            if schema.require_arguments(record, "arguments", where=where) != dict(
                node.arguments
            ):
                raise ValueError("catalog grounded skill arguments do not match its node")
            for key in (
                "preconditions",
                "effects",
                "invariants",
                "verification",
                "failure_modes",
                "deletes",
            ):
                schema.require_str_tuple(record, key, where=where)

    def _validate_contracts_and_policies(self) -> None:
        """Layer 4: contract records, policy records, and their bindings.

        Bindings are stored on both sides -- ``contract.policies`` in
        preference order and ``policy.contracts`` sorted -- so the two views
        are cross-checked the same way the derived Layer-2 views are.
        """

        bound_by_contract: Dict[str, Tuple[str, ...]] = {}
        for index, raw in enumerate(self.contracts):
            where = "contracts[{}]".format(index)
            record = schema.require_mapping(raw, where=where)
            schema.require_keys(
                record,
                where=where,
                required=(
                    "id",
                    "contract_type",
                    "target",
                    "env_id",
                    "max_episode_steps",
                    "policies",
                ),
            )
            contract_id = schema.require_contract_id(record, "id", where=where)
            if contract_id in bound_by_contract:
                raise ValueError("duplicate contract id {!r}".format(contract_id))
            schema.require_str(record, "contract_type", where=where)
            schema.require_str(record, "target", where=where)
            schema.require_str(record, "env_id", where=where)
            steps = record["max_episode_steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
                raise schema.SchemaError("{}.max_episode_steps must be positive".format(where))
            policy_ids = schema.require_str_tuple(record, "policies", where=where)
            if not policy_ids:
                raise schema.SchemaError("{}.policies cannot be empty".format(where))
            if len(set(policy_ids)) != len(policy_ids):
                raise schema.SchemaError("{}.policies has duplicates".format(where))
            for policy_id in policy_ids:
                schema.require_identifier(
                    {"policy": policy_id}, "policy", where="{}.policies".format(where)
                )
            bound_by_contract[contract_id] = policy_ids

        contracts_by_policy: Dict[str, Tuple[str, ...]] = {}
        for index, raw in enumerate(self.policies):
            where = "policies[{}]".format(index)
            record = schema.require_mapping(raw, where=where)
            schema.require_keys(
                record,
                where=where,
                required=("id", "kind", "contracts"),
                optional=(
                    "family",
                    "policy_type",
                    "checkpoint",
                    "config",
                    "checkpoint_sha256",
                ),
            )
            policy_id = schema.require_identifier(record, "id", where=where)
            if policy_id in contracts_by_policy:
                raise ValueError("duplicate policy id {!r}".format(policy_id))
            schema.require_str(record, "kind", where=where)
            for key in (
                "family",
                "policy_type",
                "checkpoint",
                "config",
                "checkpoint_sha256",
            ):
                if key in record and record[key] is not None:
                    schema.require_str(record, key, where=where)
            contracts_by_policy[policy_id] = schema.require_str_tuple(
                record, "contracts", where=where
            )

        for contract_id, policy_ids in bound_by_contract.items():
            unknown = sorted(set(policy_ids) - set(contracts_by_policy))
            if unknown:
                raise ValueError(
                    "contract {!r} is bound to unknown policies {}".format(
                        contract_id, unknown
                    )
                )
        for policy_id, declared in contracts_by_policy.items():
            derived = tuple(
                sorted(
                    contract_id
                    for contract_id, policy_ids in bound_by_contract.items()
                    if policy_id in policy_ids
                )
            )
            if not derived:
                raise ValueError(
                    "policy {!r} is not bound to any contract".format(policy_id)
                )
            if declared != derived:
                raise schema.SchemaError(
                    "catalog policy {!r} contracts view is stale: declared={}, "
                    "derived={}".format(policy_id, list(declared), list(derived))
                )

        referenced = {node.contract_id for node in self.skill_graph.nodes.values()}
        missing = sorted(referenced - set(bound_by_contract))
        if missing:
            raise ValueError("catalog has no contract records for {}".format(missing))


def _json_copy(value: Any, where: str) -> Any:
    """Return a detached JSON value and reject non-serialisable metadata."""

    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TypeError("{} must be JSON-serialisable".format(where)) from exc
