"""Selecting one execution path out of the candidate graph.

The Layer-2 graph is a *candidate* structure: a sub-goal subgraph may hold several
achievers, and ``SkillGraph.execution_order()`` is therefore only a
partial order over candidates -- never a plan.  Something has to choose.  That
chooser is a deterministic policy here and a trained decision model later; both
answer the same question, one node at a time:

    graph + completed + failed  ->  the next SkillNode

Keeping the interface incremental (rather than "emit the whole sequence") is
what lets a future model re-decide after every execution, and what lets a
failed primary candidate hand off to its ``FALLBACK_TO`` successor.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from mshab.skills import schema
from mshab.skills.graph import (
    SubGoalGraph,
    SkillSubgraph,
    SkillGraph,
    SkillNode,
    SkillRelation,
)


class NoViableCandidate(RuntimeError):
    """Every achiever for a sub-goal has been exhausted."""


@dataclass(frozen=True)
class SkillPlan:
    """One resolved execution path: exactly one achiever per sub-goal."""

    selections: Mapping[str, str]
    order: Tuple[str, ...]

    def __post_init__(self) -> None:
        selections = dict(self.selections)
        order = tuple(self.order)
        if len(order) != len(set(order)):
            raise ValueError("a skill plan cannot execute the same node twice")
        missing = sorted(set(selections.values()) - set(order))
        if missing:
            raise ValueError(
                "selected achievers missing from skill-plan order {}".format(missing)
            )
        object.__setattr__(self, "selections", MappingProxyType(selections))
        object.__setattr__(self, "order", order)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillPlan":
        where = "skill_plan"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload, where=where, required=("selections", "order")
        )
        raw_selections = schema.require_mapping(
            payload["selections"], where="{}.selections".format(where)
        )
        selections = {}
        for subgoal_id, node_id in raw_selections.items():
            schema.require_identifier({"goal": subgoal_id}, "goal", where=where)
            selections[subgoal_id] = schema.require_identifier(
                {"node": node_id}, "node", where=where
            )
        order = schema.require_str_tuple(payload, "order", where=where)
        for node_id in order:
            schema.require_identifier({"node": node_id}, "node", where=where)
        return cls(selections=selections, order=order)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selections": dict(sorted(self.selections.items())),
            "order": list(self.order),
        }


class SkillPlanner:
    """Deterministic reference decision model over a candidate skill graph.

    Candidate choice is read from the graph rather than hard-coded: within a
    sub-goal subgraph the achiever that is the *source* of a ``FALLBACK_TO`` chain
    is the primary, and each ``FALLBACK_TO`` target is the next candidate to try
    once its predecessor is reported failed.
    """

    def __init__(
        self,
        subgoals: SubGoalGraph,
        graph: SkillGraph,
    ) -> None:
        if graph.subgoal_graph is not None and graph.subgoal_graph is not subgoals:
            raise ValueError("graph is bound to a different sub-goal graph")
        graph.validate()
        self.subgoals = subgoals
        self.graph = graph

    def decide(
        self,
        completed: Iterable[str] = (),
        failed: Iterable[str] = (),
    ) -> Optional[SkillNode]:
        """Return exactly one next skill node, or ``None`` when nothing remains."""

        completed_ids = self._known(completed, "completed")
        failed_ids = self._known(failed, "failed")
        # ``execution_order()`` on the sub-goal graph is only a deterministic
        # iteration order.  Layer 2 may impose additional cross-subgraph
        # requirements, so a node is selectable only when the composition
        # graph says that all of its (possibly disjunctive) prerequisite
        # groups are satisfied.  Without this gate a valid relation such as
        # ``subgoal_b ENABLES subgoal_a`` was ignored whenever Layer 1 left the two
        # subgoals unordered.
        ready_ids = {
            node.id for node in self.graph.ready_nodes(completed=completed_ids)
        }
        for subgoal_id in self.subgoals.execution_order():
            if subgoal_id not in self.graph.subgraphs:
                continue
            subgraph = self.graph.subgraph_for_subgoal(subgoal_id)
            selected = self.select_achiever(subgraph, failed_ids)
            wanted = self._causal_closure(subgraph, selected.id)
            for node_id in subgraph.execution_order():
                if (
                    node_id not in wanted
                    or node_id in completed_ids
                    or node_id not in ready_ids
                ):
                    continue
                if node_id in failed_ids:
                    raise NoViableCandidate(
                        "instrumental node {!r} failed and has no alternative".format(
                            node_id
                        )
                    )
                return subgraph.nodes[node_id]
        return None

    def plan(self, failed: Iterable[str] = ()) -> SkillPlan:
        """Roll the incremental decisions forward into a complete path."""

        failed_ids = tuple(failed)
        completed: List[str] = []
        while True:
            node = self.decide(completed, failed_ids)
            if node is None:
                break
            completed.append(node.id)
        selections = {}
        for subgoal_id in self.graph.subgraphs:
            chosen = [
                node_id
                for node_id in completed
                if subgoal_id in self.graph.nodes[node_id].achieves
            ]
            if len(chosen) != 1:
                raise NoViableCandidate(
                    "sub-goal {!r} selected {} achievers".format(subgoal_id, len(chosen))
                )
            selections[subgoal_id] = chosen[0]
        return SkillPlan(selections=selections, order=tuple(completed))

    def select_achiever(
        self,
        subgraph: SkillSubgraph,
        failed: Iterable[str] = (),
    ) -> SkillNode:
        """The first achiever in this sub-goal's fallback chain that has not failed."""

        failed_ids = set(failed)
        chain = self.fallback_chain(subgraph)
        for node in chain:
            if node.id not in failed_ids:
                return node
        raise NoViableCandidate(
            "every achiever for sub-goal {!r} failed: {}".format(
                subgraph.subgoal_id, [node.id for node in chain]
            )
        )

    @staticmethod
    def fallback_chain(subgraph: SkillSubgraph) -> Tuple[SkillNode, ...]:
        """Achievers ordered primary-first along ``FALLBACK_TO`` edges."""

        achievers = {node.id: node for node in subgraph.achievers}
        if not achievers:
            raise ValueError(
                "sub-goal skill subgraph {!r} has no achiever".format(subgraph.subgoal_id)
            )
        successor: Dict[str, str] = {}
        for edge in subgraph.edges:
            if edge.relation != SkillRelation.FALLBACK_TO:
                continue
            if edge.source not in achievers or edge.target not in achievers:
                continue
            if edge.source in successor:
                raise ValueError(
                    "achiever {!r} declares more than one fallback".format(edge.source)
                )
            successor[edge.source] = edge.target
        if not successor:
            if len(achievers) > 1:
                raise ValueError(
                    "sub-goal {!r} has {} achievers but no FALLBACK_TO order".format(
                        subgraph.subgoal_id, len(achievers)
                    )
                )
            return tuple(achievers.values())

        heads = sorted(set(successor) - set(successor.values()))
        if len(heads) != 1:
            raise ValueError(
                "sub-goal {!r} does not have one unambiguous primary achiever".format(
                    subgraph.subgoal_id
                )
            )
        chain = []
        seen: Set[str] = set()
        current: Optional[str] = heads[0]
        while current is not None:
            if current in seen:
                raise ValueError(
                    "FALLBACK_TO cycle in sub-goal {!r}".format(subgraph.subgoal_id)
                )
            seen.add(current)
            chain.append(achievers[current])
            current = successor.get(current)
        unreachable = sorted(set(achievers) - seen)
        if unreachable:
            raise ValueError(
                "achievers {} in sub-goal {!r} are not on the fallback chain".format(
                    unreachable, subgraph.subgoal_id
                )
            )
        return tuple(chain)

    @staticmethod
    def _causal_closure(subgraph: SkillSubgraph, node_id: str) -> Set[str]:
        """``node_id`` plus the instrumental nodes it transitively depends on."""

        prerequisites: Dict[str, Set[str]] = {item: set() for item in subgraph.nodes}
        for edge in subgraph.edges:
            if edge.relation == SkillRelation.ENABLES:
                prerequisites[edge.target].add(edge.source)
            elif edge.relation == SkillRelation.REQUIRES:
                prerequisites[edge.source].add(edge.target)
        closure = {node_id}
        frontier = list(prerequisites[node_id])
        while frontier:
            current = frontier.pop()
            if current in closure:
                continue
            closure.add(current)
            frontier.extend(prerequisites[current] - closure)
        return closure

    def _known(self, node_ids: Iterable[str], label: str) -> Set[str]:
        values = set(node_ids)
        unknown = sorted(values - set(self.graph.nodes))
        if unknown:
            raise KeyError("unknown {} nodes {}".format(label, unknown))
        return values
