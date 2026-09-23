"""Contract-level symbolic validation of a planned SkillNode path.

The graph-structural checks in :mod:`set_table_graph_ablation` answer "does
this path follow the Layer-2 edges?".  They cannot answer "does this path
actually achieve the task?", and the previous experiment substituted a
hard-coded canonical segment order for that question.  That rejected plans
which are correct: ``close`` has no ``gripper_empty()`` precondition, so
closing the drawer while already holding the bowl is legal and saves a trip.

This module answers the semantic question directly by simulating the plan
against the Layer-3 contracts that the project already declares.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Outcome labels, ordered from best to worst.
OUTCOME_EXACT = "exact_match"
OUTCOME_ALTERNATIVE = "valid_alternative"
OUTCOME_PRECONDITION = "invalid_precondition"
OUTCOME_GOAL = "invalid_goal"
OUTCOME_STRUCTURE = "invalid_structure"
OUTCOME_SCHEMA = "invalid_schema"
OUTCOME_API = "api_failure"

OUTCOME_ORDER = (
    OUTCOME_EXACT,
    OUTCOME_ALTERNATIVE,
    OUTCOME_PRECONDITION,
    OUTCOME_GOAL,
    OUTCOME_STRUCTURE,
    OUTCOME_SCHEMA,
    OUTCOME_API,
)

#: Outcomes that count as a correct plan.
VALID_OUTCOMES = frozenset({OUTCOME_EXACT, OUTCOME_ALTERNATIVE})


@dataclass(frozen=True)
class SymbolicResult:
    """Outcome of simulating one plan against the Layer-3 contracts."""

    reachable_goal: bool
    precondition_ok: bool
    failed_step: Optional[int] = None
    failed_node: Optional[str] = None
    missing: Tuple[str, ...] = ()
    missing_goal: Tuple[str, ...] = ()
    final_facts: FrozenSet[str] = frozenset()
    steps: int = 0

    @property
    def valid(self) -> bool:
        return self.precondition_ok and self.reachable_goal

    @property
    def outcome(self) -> str:
        if not self.precondition_ok:
            return OUTCOME_PRECONDITION
        if not self.reachable_goal:
            return OUTCOME_GOAL
        return OUTCOME_ALTERNATIVE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "precondition_ok": self.precondition_ok,
            "reachable_goal": self.reachable_goal,
            "failed_step": self.failed_step,
            "failed_node": self.failed_node,
            "missing_preconditions": list(self.missing),
            "missing_goal_predicates": list(self.missing_goal),
            "steps": self.steps,
        }


@dataclass(frozen=True)
class TaskSemantics:
    """Initial symbolic state and goal predicates for one task instance.

    ``vocabulary`` lists the canonical entity ids this task is written in.  A
    planner that binds ``bowl`` rather than ``024_bowl`` has made a naming
    choice, not a reasoning error, so argument values are canonicalised
    against this list before predicates are grounded -- using the same alias
    rule the grounder applies when compiling to MS-HAB PlanData.
    """

    initial_facts: FrozenSet[str]
    goal_predicates: FrozenSet[str]
    vocabulary: Tuple[str, ...] = ()
    exclusive_reachable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_facts", frozenset(self.initial_facts))
        object.__setattr__(self, "goal_predicates", frozenset(self.goal_predicates))
        object.__setattr__(self, "vocabulary", tuple(self.vocabulary))

    def canonical(self, value: str) -> str:
        """Map an entity name onto this task's vocabulary when it is an alias."""

        from mshab.experiments.granularity.vlm.run import _entities_match

        if value in self.vocabulary:
            return value
        for known in self.vocabulary:
            if _entities_match(value, known):
                return known
        return value


class ContractSimulator:
    """Apply Layer-3 contracts to a symbolic state, one SkillNode at a time.

    ``exclusive_reachable`` models the MS-HAB convention that the robot is
    positioned for exactly one target at a time: a ``navigate`` retracts every
    other ``reachable(...)`` fact.  Without it a plan that omits a navigation
    step would still validate, because ``reachable`` would never be retracted.
    """

    def __init__(
        self,
        contracts: Mapping[str, Mapping[str, Any]],
        nodes: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.contracts = {
            contract["contract_type"]: contract for contract in contracts.values()
        } if isinstance(contracts, Mapping) and not _looks_like_type_map(contracts) else dict(contracts)
        self.nodes = dict(nodes)

    def bind_node(
        self,
        node_id: str,
        bindings: Mapping[str, str],
        semantics: Optional[TaskSemantics] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """Resolve one node's arguments against a segment's bindings."""

        node = self.nodes[node_id]
        arguments: Dict[str, str] = {}
        for name, value in node["arguments"].items():
            if isinstance(value, str):
                missing = [
                    field_name
                    for field_name in _PLACEHOLDER.findall(value)
                    if field_name not in bindings
                ]
                if missing:
                    raise KeyError(
                        "node {} argument {} needs bindings {}".format(
                            node_id, name, sorted(missing)
                        )
                    )
                value = _PLACEHOLDER.sub(
                    lambda match: bindings[match.group(1)], value
                )
            if semantics is not None and isinstance(value, str):
                value = semantics.canonical(value)
            arguments[name] = value
        return node["contract_type"], arguments

    def step(
        self, facts: Set[str], contract_type: str, arguments: Mapping[str, str]
    ) -> Tuple[Set[str], List[str]]:
        """Return the next state, or the unmet preconditions."""

        contract = self.contracts[contract_type]
        preconditions = _ground_all(contract.get("preconditions", ()), arguments)
        unmet = [item for item in preconditions if item not in facts]
        if unmet:
            return facts, unmet
        effects = _ground_all(contract.get("effects", ()), arguments)
        deletes = set(_ground_all(contract.get("deletes", ()), arguments))
        if contract_type == "navigate":
            target = arguments.get("goal")
            deletes |= {
                fact
                for fact in facts
                if fact.startswith("reachable(") and fact != "reachable({})".format(target)
            }
        return (facts - deletes) | set(effects), []

    def simulate(
        self,
        plan: Sequence[Tuple[str, Mapping[str, str]]],
        semantics: TaskSemantics,
    ) -> SymbolicResult:
        """Run a ``(node_id, bindings)`` sequence from the initial state."""

        facts: Set[str] = set(semantics.initial_facts)
        for index, (node_id, bindings) in enumerate(plan):
            contract_type, arguments = self.bind_node(
                node_id, bindings, semantics
            )
            facts, unmet = self.step(facts, contract_type, arguments)
            if unmet:
                return SymbolicResult(
                    reachable_goal=False,
                    precondition_ok=False,
                    failed_step=index,
                    failed_node=node_id,
                    missing=tuple(unmet),
                    final_facts=frozenset(facts),
                    steps=index,
                )
        missing_goal = tuple(
            sorted(item for item in semantics.goal_predicates if item not in facts)
        )
        return SymbolicResult(
            reachable_goal=not missing_goal,
            precondition_ok=True,
            missing_goal=missing_goal,
            final_facts=frozenset(facts),
            steps=len(plan),
        )


def _looks_like_type_map(contracts: Mapping[str, Any]) -> bool:
    return all(
        isinstance(value, Mapping) and value.get("contract_type") == key
        for key, value in contracts.items()
    )


def _ground_all(
    predicates: Sequence[str], arguments: Mapping[str, str]
) -> List[str]:
    grounded = []
    for predicate in predicates:
        missing = [
            name for name in _PLACEHOLDER.findall(predicate) if name not in arguments
        ]
        if missing:
            raise KeyError(
                "predicate {!r} needs arguments {}".format(predicate, sorted(missing))
            )
        grounded.append(
            _PLACEHOLDER.sub(lambda match: arguments[match.group(1)], predicate)
        )
    return grounded


def plan_from_decision(
    decision: Mapping[str, Any]
) -> List[Tuple[str, Dict[str, str]]]:
    """Flatten a path decision into ``(node_id, bindings)`` execution order."""

    segments = {segment["id"]: segment for segment in decision["segments"]}
    plan: List[Tuple[str, Dict[str, str]]] = []
    for segment_id in decision["segment_order"]:
        segment = segments[segment_id]
        bindings = dict(segment.get("bindings", {}))
        for node_id in segment["skill_node_path"]:
            plan.append((node_id, bindings))
    return plan


def lcs_f1(left: Sequence[str], right: Sequence[str]) -> float:
    """Similarity of two node paths; 1.0 only for an identical sequence."""

    if not left and not right:
        return 1.0
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, start=1):
        for j, b in enumerate(right, start=1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if a == b
                else max(table[i - 1][j], table[i][j - 1])
            )
    lcs = table[len(left)][len(right)]
    return 2 * lcs / (len(left) + len(right))
