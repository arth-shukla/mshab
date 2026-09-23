"""Validated extension interface for manual and future VLM graph builders.

Three kinds of change reach Layer 1 and Layer 2, and all of them travel as one
declarative :class:`SkillGraphPatch`:

1. a brand-new sub-goal together with its implementation subgraph;
2. a relation connecting that subgraph to subgraphs that already exist;
3. a new candidate node inside a subgraph that is already registered.

The third case is what :class:`SkillSubgraphExtension` exists for.  A
registered subgraph is sealed, so it is never edited in place: the extension
builds a validated successor and the aggregate swaps it in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from mshab.skills import schema
from mshab.skills.graph import (
    SubGoal,
    SubGoalGraph,
    SkillSubgraph,
    SubGoalDependency,
    SkillGraph,
    SkillEdge,
    SkillNode,
    CrossSubgraphEdge,
)
from mshab.skills.library import ContractLibrary


@dataclass(frozen=True)
class SkillSubgraphExtension:
    """Add candidates or internal relations to an already-registered sub-goal.

    The extension never mutates the sealed subgraph.  :meth:`rebuild` returns an
    unsealed successor which the composition graph revalidates and installs, so
    a rejected extension leaves the live graph untouched.
    """

    subgoal_id: str
    nodes: Tuple[SkillNode, ...] = ()
    edges: Tuple[SkillEdge, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        if not self.subgoal_id:
            raise ValueError("subgraph extension subgoal_id must be non-empty")
        if not self.nodes and not self.edges:
            raise ValueError("subgraph extension must add a node or a relation")
        for node in self.nodes:
            if not isinstance(node, SkillNode):
                raise TypeError(
                    "subgraph extension nodes must be SkillNode, got {}".format(
                        type(node).__name__
                    )
                )
        for edge in self.edges:
            if not isinstance(edge, SkillEdge):
                raise TypeError(
                    "subgraph extension edges must be SkillEdge, got {}".format(
                        type(edge).__name__
                    )
                )

    def rebuild(self, current: SkillSubgraph) -> SkillSubgraph:
        if current.subgoal_id != self.subgoal_id:
            raise ValueError(
                "extension targets sub-goal {!r}, not {!r}".format(
                    self.subgoal_id, current.subgoal_id
                )
            )
        successor = current.unsealed_copy()
        for node in self.nodes:
            successor.add_node(node)
        for edge in self.edges:
            successor.relate(edge.source, edge.target, edge.relation)
        successor.validate()
        return successor

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillSubgraphExtension":
        where = "subgraph_extension"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload, where=where, required=("subgoal_id",), optional=("nodes", "edges")
        )
        return cls(
            subgoal_id=schema.require_identifier(payload, "subgoal_id", where=where),
            nodes=tuple(
                SkillNode.from_dict(item)
                for item in schema.require_sequence(payload, "nodes", where=where)
            ),
            edges=tuple(
                SkillEdge.from_dict(item)
                for item in schema.require_sequence(payload, "edges", where=where)
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class SkillGraphPatch:
    """A declarative, reviewable addition to Layer 1 and Layer 2.

    Manual code and a future VLM emit this same object.  Applying the patch goes
    through the existing duplicate, namespace, unknown-sub-goal, achiever, and cycle
    checks instead of letting a proposer mutate graph internals.
    """

    subgoals: Tuple[SubGoal, ...] = ()
    subgoal_dependencies: Tuple[SubGoalDependency, ...] = ()
    skill_subgraphs: Tuple[SkillSubgraph, ...] = ()
    cross_edges: Tuple[CrossSubgraphEdge, ...] = ()
    subgraph_extensions: Tuple[SkillSubgraphExtension, ...] = ()

    _FIELD_TYPES = {
        "subgoals": SubGoal,
        "subgoal_dependencies": SubGoalDependency,
        "skill_subgraphs": SkillSubgraph,
        "cross_edges": CrossSubgraphEdge,
        "subgraph_extensions": SkillSubgraphExtension,
    }

    def __post_init__(self) -> None:
        for name, expected in self._FIELD_TYPES.items():
            items = tuple(getattr(self, name))
            for item in items:
                # Without this, a duck-typed object walks straight into the
                # graph and can behave differently on the staging and commit
                # passes -- exactly the trust boundary a VLM sits behind.
                if not isinstance(item, expected):
                    raise TypeError(
                        "patch.{} must contain {} instances, got {}".format(
                            name, expected.__name__, type(item).__name__
                        )
                    )
            object.__setattr__(self, name, items)

    def with_extension(
        self,
        subgoal_id: str,
        nodes: Iterable[SkillNode] = (),
        edges: Iterable[SkillEdge] = (),
    ) -> "SkillGraphPatch":
        """Return a new patch that also extends an existing sub-goal subgraph."""

        extension = SkillSubgraphExtension(
            subgoal_id=subgoal_id, nodes=tuple(nodes), edges=tuple(edges)
        )
        return SkillGraphPatch(
            subgoals=self.subgoals,
            subgoal_dependencies=self.subgoal_dependencies,
            skill_subgraphs=self.skill_subgraphs,
            cross_edges=self.cross_edges,
            subgraph_extensions=self.subgraph_extensions + (extension,),
        )

    def apply(
        self,
        subgoal_graph: SubGoalGraph,
        skill_graph: SkillGraph,
        library: Optional[ContractLibrary] = None,
    ) -> None:
        """Atomically apply this patch, or leave both graphs untouched.

        Everything is built and validated on a staging copy; the live objects
        are only touched by the final two adopt calls, which cannot fail.  The
        earlier design replayed the mutations on the live graph after a dry
        run, so any divergence between the two passes left a rejected patch
        half-applied.
        """

        if skill_graph.subgoal_graph is not subgoal_graph:
            raise ValueError("skill_graph must reference the supplied subgoal_graph")
        staging_subgoals, staging_skills = _staging_copy(subgoal_graph, skill_graph)
        self._apply_to(staging_subgoals, staging_skills, library)
        staging_skills.validate()
        subgoal_graph._adopt(staging_subgoals)
        skill_graph._adopt(staging_skills)

    def _apply_to(
        self,
        subgoal_graph: SubGoalGraph,
        skill_graph: SkillGraph,
        library: Optional[ContractLibrary],
    ) -> None:
        if library is not None:
            self.validate_contract_ids(library)
        for subgoal in self.subgoals:
            subgoal_graph.add_subgoal(subgoal)
        for dependency in self.subgoal_dependencies:
            subgoal_graph.add_dependency(dependency.source, dependency.target)
        for subgraph in self.skill_subgraphs:
            skill_graph.add_subgraph(subgraph.unsealed_copy())
        for extension in self.subgraph_extensions:
            current = skill_graph.subgraph_for_subgoal(extension.subgoal_id)
            skill_graph.replace_subgraph(extension.rebuild(current))
        for relation in self.cross_edges:
            skill_graph.relate_subgraphs(
                relation.source_subgoal,
                relation.target_subgoal,
                relation.source_node,
                relation.target_node,
                relation.relation,
            )

    def validate_contract_ids(self, library: ContractLibrary) -> None:
        """Reject a patch that references a contract the library does not have."""

        referenced = {
            node.contract_id
            for subgraph in self.skill_subgraphs
            for node in subgraph.nodes.values()
        }
        referenced |= {
            node.contract_id
            for extension in self.subgraph_extensions
            for node in extension.nodes
        }
        missing = sorted(referenced - {contract.id for contract in library.find()})
        if missing:
            raise KeyError(
                "graph patch references unregistered contracts {}".format(missing)
            )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, task: str
    ) -> "SkillGraphPatch":
        """Parse an untrusted patch document under a strict schema.

        ``task`` is supplied by the caller rather than read from the payload:
        a proposer must not be able to redirect a patch into another task's
        namespace.
        """

        where = "skill_graph_patch"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            optional=(
                "subgoals",
                "subgoal_dependencies",
                "skill_subgraphs",
                "cross_edges",
                "subgraph_extensions",
                "schema_version",
            ),
        )
        return cls(
            subgoals=tuple(
                SubGoal.from_dict(item)
                for item in schema.require_sequence(payload, "subgoals", where=where)
            ),
            subgoal_dependencies=tuple(
                SubGoalDependency.from_dict(item)
                for item in schema.require_sequence(
                    payload, "subgoal_dependencies", where=where
                )
            ),
            skill_subgraphs=tuple(
                SkillSubgraph.from_dict(item, task=task)
                for item in schema.require_sequence(
                    payload, "skill_subgraphs", where=where
                )
            ),
            cross_edges=tuple(
                CrossSubgraphEdge.from_dict(item)
                for item in schema.require_sequence(
                    payload, "cross_edges", where=where
                )
            ),
            subgraph_extensions=tuple(
                SkillSubgraphExtension.from_dict(item)
                for item in schema.require_sequence(
                    payload, "subgraph_extensions", where=where
                )
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": schema.SCHEMA_VERSION,
            "subgoals": [subgoal.as_dict() for subgoal in self.subgoals],
            "subgoal_dependencies": [
                item.as_dict() for item in self.subgoal_dependencies
            ],
            "skill_subgraphs": [
                subgraph.as_dict() for subgraph in self.skill_subgraphs
            ],
            "cross_edges": [
                relation.as_dict() for relation in self.cross_edges
            ],
            "subgraph_extensions": [
                extension.as_dict() for extension in self.subgraph_extensions
            ],
        }


class SkillGraphBuilder(ABC):
    """Extension point shared by hand-written and future VLM builders."""

    @abstractmethod
    def propose(
        self,
        goal: str,
        task: str,
        context: Mapping[str, Any],
    ) -> SkillGraphPatch:
        """Return a patch; do not mutate graph internals directly."""

    def build(
        self,
        goal: str,
        task: str,
        context: Optional[Mapping[str, Any]] = None,
        library: Optional[ContractLibrary] = None,
    ) -> Tuple[SubGoalGraph, SkillGraph]:
        subgoal_graph = SubGoalGraph(goal)
        skill_graph = SkillGraph(task, subgoal_graph=subgoal_graph)
        patch = self.propose(goal, task, context or {})
        patch.apply(subgoal_graph, skill_graph, library=library)
        return subgoal_graph, skill_graph


def _staging_copy(
    subgoal_graph: SubGoalGraph,
    skill_graph: SkillGraph,
) -> Tuple[SubGoalGraph, SkillGraph]:
    staging_subgoals = SubGoalGraph(subgoal_graph.goal)
    staging_skills = SkillGraph(skill_graph.task, subgoal_graph=staging_subgoals)
    for subgoal in subgoal_graph.subgoals.values():
        staging_subgoals.add_subgoal(subgoal)
    for dependency in subgoal_graph.dependencies:
        staging_subgoals.add_dependency(dependency.source, dependency.target)
    for subgraph in skill_graph.subgraphs.values():
        staging_skills.add_subgraph(subgraph.unsealed_copy())
    for relation in skill_graph.cross_edges:
        staging_skills.relate_subgraphs(
            relation.source_subgoal,
            relation.target_subgoal,
            relation.source_node,
            relation.target_node,
            relation.relation,
        )
    return staging_subgoals, staging_skills
