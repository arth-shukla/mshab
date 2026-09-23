"""Task-conditioned sub-goal and skill graphs.

Vocabulary: the *goal* is the task text.  Layer 1 decomposes it into
*sub-goals*.  Layer 2 gives every sub-goal a subgraph of *skill nodes*; a
skill node references one contract by id.  "Skill" in this package always
means a Layer-2 node, never a contract or a policy.

The graph is the only owner of skill-to-skill composition semantics.  There is
no ``AlternativeSkill`` or ``SequentialSkill`` class: alternatives, fallback,
requirements, and enabling order are relations between skill nodes.

Relations carry the *semantic/logical* ordering of a task (heating food comes
before serving it).  Whether a node can physically start in the current
world state is decided later by its contract, not by the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from mshab.skills import schema


class SkillRelation(str, Enum):
    """Closed vocabulary for edges in a candidate skill graph.

    Edges express semantic or logical ordering between skill nodes.  They do
    not restate physical preconditions; those live on the contract.
    """

    IS_A = "is_a"
    ENABLES = "enables"
    REQUIRES = "requires"
    ALTERNATIVE_TO = "alternative_to"
    FALLBACK_TO = "fallback_to"


@dataclass(frozen=True)
class SubGoal:
    """One sub-goal: a desired world-state predicate decomposed from the goal."""

    id: str
    predicate: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.predicate:
            raise ValueError("sub-goal id and predicate must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubGoal":
        where = "subgoal"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("id", "predicate"),
            optional=("description",),
        )
        return cls(
            id=schema.require_identifier(payload, "id", where=where),
            predicate=schema.require_str(payload, "predicate", where=where),
            description=schema.optional_str(payload, "description", where=where),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "predicate": self.predicate,
            "description": self.description,
        }


@dataclass(frozen=True)
class SubGoalDependency:
    """The source sub-goal must be achieved before the target sub-goal."""

    source: str
    target: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("sub-goal dependency needs two distinct non-empty ids")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubGoalDependency":
        where = "subgoal_dependency"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(payload, where=where, required=("source", "target"))
        return cls(
            source=schema.require_identifier(payload, "source", where=where),
            target=schema.require_identifier(payload, "target", where=where),
        )

    def as_dict(self) -> Dict[str, str]:
        return {"source": self.source, "target": self.target}


class SubGoalGraph:
    """Layer 1: the task goal decomposed into ordered sub-goals."""

    def __init__(self, goal: str) -> None:
        if not goal:
            raise ValueError("task goal must be non-empty")
        self.goal = goal
        self._subgoals: Dict[str, SubGoal] = {}
        self._dependencies: List[SubGoalDependency] = []

    @property
    def subgoals(self) -> Mapping[str, SubGoal]:
        return dict(self._subgoals)

    @property
    def dependencies(self) -> Tuple[SubGoalDependency, ...]:
        return tuple(self._dependencies)

    def add_subgoal(self, subgoal: SubGoal) -> None:
        if subgoal.id in self._subgoals:
            raise ValueError("duplicate sub-goal {!r}".format(subgoal.id))
        self._subgoals[subgoal.id] = subgoal

    def add_dependency(self, source: str, target: str) -> None:
        self._require_subgoal(source)
        self._require_subgoal(target)
        dependency = SubGoalDependency(source, target)
        if dependency in self._dependencies:
            raise ValueError("duplicate sub-goal dependency {} -> {}".format(source, target))
        self._dependencies.append(dependency)
        try:
            self.execution_order()  # reject a cycle at insertion time
        except ValueError:
            self._dependencies.pop()
            raise

    def achieved(self, subgoal_id: str, facts: Iterable[str]) -> bool:
        return self._require_subgoal(subgoal_id).predicate in set(facts)

    def ready_subgoals(
        self,
        facts: Iterable[str],
        completed: Iterable[str] = (),
    ) -> Tuple[SubGoal, ...]:
        fact_set = set(facts)
        completed_set = set(completed)
        unknown_completed = sorted(completed_set - set(self._subgoals))
        if unknown_completed:
            raise KeyError("unknown completed subgoals {}".format(unknown_completed))
        ready = []
        for subgoal in self._subgoals.values():
            if subgoal.id in completed_set or subgoal.predicate in fact_set:
                continue
            prerequisites = {
                dependency.source
                for dependency in self._dependencies
                if dependency.target == subgoal.id
            }
            if all(
                item in completed_set or self._subgoals[item].predicate in fact_set
                for item in prerequisites
            ):
                ready.append(subgoal)
        return tuple(sorted(ready, key=lambda item: item.id))

    def execution_order(self) -> Tuple[str, ...]:
        edges = [(item.source, item.target) for item in self._dependencies]
        return _topological_order(self._subgoals, edges, "sub-goal")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "subgoals": [
                subgoal.as_dict()
                for subgoal in sorted(self._subgoals.values(), key=lambda item: item.id)
            ],
            "dependencies": [item.as_dict() for item in self._dependencies],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubGoalGraph":
        where = "subgoal_graph"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("goal",),
            optional=("subgoals", "dependencies", "schema_version", "execution_order"),
        )
        graph = cls(schema.require_str(payload, "goal", where=where))
        for item in schema.require_sequence(payload, "subgoals", where=where):
            graph.add_subgoal(SubGoal.from_dict(item))
        for item in schema.require_sequence(payload, "dependencies", where=where):
            dependency = SubGoalDependency.from_dict(item)
            graph.add_dependency(dependency.source, dependency.target)
        return graph

    def _adopt(self, other: "SubGoalGraph") -> None:
        """Replace this graph's contents with a validated staging copy.

        Used to commit a :class:`~mshab.skills.extension.SkillGraphPatch` in one
        step so a rejected patch can never leave the live graph half-modified.
        """

        self.goal = other.goal
        self._subgoals = dict(other._subgoals)
        self._dependencies = list(other._dependencies)

    def _require_subgoal(self, subgoal_id: str) -> SubGoal:
        try:
            return self._subgoals[subgoal_id]
        except KeyError as exc:
            raise KeyError("unknown sub-goal {!r}".format(subgoal_id)) from exc


@dataclass(frozen=True)
class SkillNode:
    """One skill node: a scene-independent request to run one contract.

    Layer 2 stores only a stable contract id plus symbolic arguments.  It does
    not own a bound contract, checkpoint, simulator object, or policy choice;
    those are resolved by Layer 3/4 after an environment is selected.
    """

    id: str
    contract_id: str
    arguments: Mapping[str, Any]
    achieves: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.contract_id:
            raise ValueError("skill node id and contract_id must be non-empty")
        arguments = dict(self.arguments)
        for name, value in arguments.items():
            if not isinstance(value, schema.SCALAR_TYPES):
                raise TypeError(
                    "skill node {} argument {!r} must be a scalar, got {}".format(
                        self.id, name, type(value).__name__
                    )
                )
        object.__setattr__(self, "arguments", MappingProxyType(arguments))
        object.__setattr__(self, "achieves", tuple(self.achieves))

    def __hash__(self) -> int:
        # dataclass(frozen=True) derives __hash__ from the field tuple, which
        # contains an unhashable mappingproxy.  Hash the scalar contents instead
        # so nodes may be used in sets and as dict keys.
        return hash(
            (self.id, self.contract_id, tuple(sorted(self.arguments.items())), self.achieves)
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillNode":
        where = "skill_node"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("id", "contract_id"),
            optional=("arguments", "achieves"),
        )
        achieves = schema.require_str_tuple(payload, "achieves", where=where)
        for item in achieves:
            schema.require_identifier({"achieves": item}, "achieves", where=where)
        return cls(
            id=schema.require_identifier(payload, "id", where=where),
            contract_id=schema.require_contract_id(payload, "contract_id", where=where),
            arguments=schema.require_arguments(payload, "arguments", where=where),
            achieves=achieves,
        )

    def as_dict(self) -> Dict[str, Any]:
        return _node_dict(self)


@dataclass(frozen=True)
class SkillEdge:
    """One typed relation between two skill nodes.

    Edges carry semantic/logical ordering only.  Physical readiness of a node
    in the current world state is the job of its contract.

    Direction semantics:

    - ``IS_A``: source is a specialization of target;
    - ``ENABLES``: source completion enables target;
    - ``REQUIRES``: source requires target to complete first;
    - ``ALTERNATIVE_TO``: symmetric candidate equivalence;
    - ``FALLBACK_TO``: try target after source fails.
    """

    source: str
    target: str
    relation: SkillRelation

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", _require_relation(self.relation))
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("skill edge needs two distinct non-empty node ids")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillEdge":
        where = "skill_edge"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload, where=where, required=("source", "target", "relation")
        )
        return cls(
            source=schema.require_identifier(payload, "source", where=where),
            target=schema.require_identifier(payload, "target", where=where),
            relation=_require_relation(
                schema.require_str(payload, "relation", where=where)
            ),
        )

    def as_dict(self) -> Dict[str, str]:
        return _edge_dict(self)


class SkillSubgraph:
    """Layer-2 implementation subgraph owned by one sub-goal.

    A subgraph contains both instrumental nodes (for example, navigation) and
    one or more achiever nodes.  An achiever may only claim the sub-goal
    owned by this subgraph.  This makes the Layer-1-to-Layer-2 boundary explicit
    instead of reconstructing it later from a flat collection of nodes.
    """

    def __init__(self, subgoal_id: str, task: str) -> None:
        if not subgoal_id or not task:
            raise ValueError("subgoal_id and task must be non-empty")
        self._subgoal_id = subgoal_id
        self._task = task
        self._nodes: Dict[str, SkillNode] = {}
        self._edges: List[SkillEdge] = []
        self._sealed = False

    def __setattr__(self, name: str, value: Any) -> None:
        """Freeze the whole object once a skill graph has registered it.

        ``_seal`` used to guard only ``add_node``/``relate``, which left the
        aggregate handing out a live object whose ``subgoal_id`` could be
        reassigned -- silently breaking single-sub-goal ownership, the achiever
        invariant, and global node-id uniqueness.
        """

        if getattr(self, "_sealed", False):
            raise RuntimeError(
                "sub-goal skill subgraph {!r} is registered and sealed; build a "
                "replacement with SkillSubgraphExtension instead".format(
                    self._subgoal_id
                )
            )
        object.__setattr__(self, name, value)

    @property
    def subgoal_id(self) -> str:
        return self._subgoal_id

    @property
    def task(self) -> str:
        return self._task

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def nodes(self) -> Mapping[str, SkillNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> Tuple[SkillEdge, ...]:
        return tuple(self._edges)

    @property
    def achievers(self) -> Tuple[SkillNode, ...]:
        return tuple(
            node
            for node in sorted(self._nodes.values(), key=lambda item: item.id)
            if self.subgoal_id in node.achieves
        )

    def add_node(self, node: SkillNode) -> None:
        self._ensure_mutable()
        if node.id in self._nodes:
            raise ValueError("duplicate skill node {!r}".format(node.id))
        expected_prefix = "mshab.{}.".format(self.task)
        if not node.contract_id.startswith(expected_prefix):
            raise ValueError(
                "node {} contract {!r} is outside task namespace {!r}".format(
                    node.id, node.contract_id, self.task
                )
            )
        foreign_subgoals = sorted(set(node.achieves) - {self.subgoal_id})
        if foreign_subgoals:
            raise ValueError(
                "node {} is inside sub-goal subgraph {!r} but achieves {}".format(
                    node.id, self.subgoal_id, foreign_subgoals
                )
            )
        self._nodes[node.id] = node

    def relate(self, source: str, target: str, relation: SkillRelation) -> None:
        self._ensure_mutable()
        self._require_node(source)
        self._require_node(target)
        edge = SkillEdge(source, target, _require_relation(relation))
        _append_checked_edge(self._edges, edge)
        try:
            self.execution_order()
        except ValueError:
            self._edges.pop()
            raise

    def execution_order(self) -> Tuple[str, ...]:
        return _topological_order(
            self._nodes, _causal_pairs(self._edges), "sub-goal skill subgraph"
        )

    def validate(self) -> None:
        if not self._nodes:
            raise ValueError(
                "sub-goal skill subgraph {!r} must contain at least one node".format(
                    self.subgoal_id
                )
            )
        if not self.achievers:
            raise ValueError(
                "sub-goal skill subgraph {!r} has no achiever node".format(self.subgoal_id)
            )
        self.execution_order()

    def _seal(self) -> None:
        """Transfer structural ownership to a SkillGraph."""

        self.validate()
        object.__setattr__(self, "_sealed", True)

    def unsealed_copy(self) -> "SkillSubgraph":
        """A mutable clone: the only supported way to revise a sealed subgraph."""

        clone = SkillSubgraph(self._subgoal_id, self._task)
        for node in self._nodes.values():
            clone.add_node(node)
        for edge in self._edges:
            clone.relate(edge.source, edge.target, edge.relation)
        return clone

    def _ensure_mutable(self) -> None:
        if self._sealed:
            raise RuntimeError(
                "sub-goal skill subgraph {!r} is registered and sealed".format(
                    self.subgoal_id
                )
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "nodes": [
                _node_dict(node)
                for node in sorted(
                    self._nodes.values(), key=lambda item: item.id
                )
            ],
            "edges": [_edge_dict(edge) for edge in self._edges],
            "achiever_nodes": [node.id for node in self.achievers],
            "execution_order": list(self.execution_order()),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, task: str
    ) -> "SkillSubgraph":
        where = "skill_subgraph"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("subgoal_id",),
            # achiever_nodes/execution_order are derived views in as_dict();
            # they are accepted on input but recomputed, never trusted.
            optional=("nodes", "edges", "achiever_nodes", "execution_order"),
        )
        subgraph = cls(
            schema.require_identifier(payload, "subgoal_id", where=where), task
        )
        for item in schema.require_sequence(payload, "nodes", where=where):
            subgraph.add_node(SkillNode.from_dict(item))
        for item in schema.require_sequence(payload, "edges", where=where):
            edge = SkillEdge.from_dict(item)
            subgraph.relate(edge.source, edge.target, edge.relation)
        return subgraph

    def _require_node(self, node_id: str) -> SkillNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(
                "unknown skill node {!r} in sub-goal subgraph {!r}".format(
                    node_id, self.subgoal_id
                )
            ) from exc


@dataclass(frozen=True)
class CrossSubgraphEdge:
    """A typed edge crossing two sub-goal subgraphs.

    An endpoint is either a specific node id or ``None``, which means *any
    achiever of that sub-goal*.  Sub-goal-level endpoints are what make alternative
    candidates usable: pinning a downstream ``ENABLES`` to one specific
    achiever silently makes that candidate mandatory, so a task that recovers
    through a fallback can never reach the next sub-goal.
    """

    source_subgoal: str
    target_subgoal: str
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    relation: SkillRelation = SkillRelation.ENABLES

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", _require_relation(self.relation))
        if not self.source_subgoal or not self.target_subgoal:
            raise ValueError("cross-subgraph edge subgoals must be non-empty")
        if self.source_subgoal == self.target_subgoal:
            raise ValueError("cross-subgraph edge must connect two distinct subgoals")
        for name in ("source_node", "target_node"):
            value = getattr(self, name)
            if value is not None and not value:
                raise ValueError("cross-subgraph edge {} must be non-empty".format(name))
        if (
            self.source_node is not None
            and self.target_node is not None
            and self.source_node == self.target_node
        ):
            raise ValueError("cross-subgraph edge needs two distinct node ids")
        if self.is_subgoal_level and self.relation not in _CAUSAL_RELATIONS:
            raise ValueError(
                "a sub-goal-level endpoint is only meaningful for {}; got {}".format(
                    sorted(item.value for item in _CAUSAL_RELATIONS),
                    self.relation.value,
                )
            )

    @property
    def is_subgoal_level(self) -> bool:
        return self.source_node is None or self.target_node is None

    def endpoints(
        self, subgraphs: Mapping[str, "SkillSubgraph"]
    ) -> Tuple[Tuple[str, str], ...]:
        """Resolve to concrete ``(source_node, target_node)`` pairs."""

        sources = self._resolve(self.source_subgoal, self.source_node, subgraphs)
        targets = self._resolve(self.target_subgoal, self.target_node, subgraphs)
        return tuple(
            (source, target)
            for source in sources
            for target in targets
            if source != target
        )

    def edges(
        self, subgraphs: Mapping[str, "SkillSubgraph"]
    ) -> Tuple[SkillEdge, ...]:
        return tuple(
            SkillEdge(source, target, self.relation)
            for source, target in self.endpoints(subgraphs)
        )

    @staticmethod
    def _resolve(
        subgoal_id: str,
        node_id: Optional[str],
        subgraphs: Mapping[str, "SkillSubgraph"],
    ) -> Tuple[str, ...]:
        try:
            subgraph = subgraphs[subgoal_id]
        except KeyError as exc:
            raise KeyError(
                "unknown sub-goal skill subgraph {!r}".format(subgoal_id)
            ) from exc
        if node_id is not None:
            if node_id not in subgraph.nodes:
                raise ValueError(
                    "node {!r} is not owned by subgraph {!r}".format(node_id, subgoal_id)
                )
            return (node_id,)
        achievers = tuple(node.id for node in subgraph.achievers)
        if not achievers:
            raise ValueError(
                "sub-goal-level endpoint needs an achiever in subgraph {!r}".format(subgoal_id)
            )
        return achievers

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossSubgraphEdge":
        where = "cross_subgraph_edge"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("source_subgoal", "target_subgoal"),
            optional=("source_node", "target_node", "relation"),
        )
        return cls(
            source_subgoal=schema.require_identifier(payload, "source_subgoal", where=where),
            target_subgoal=schema.require_identifier(payload, "target_subgoal", where=where),
            source_node=schema.optional_identifier(payload, "source_node", where=where),
            target_node=schema.optional_identifier(payload, "target_node", where=where),
            relation=_require_relation(
                payload.get("relation", SkillRelation.ENABLES.value)
            ),
        )

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "source_subgoal": self.source_subgoal,
            "target_subgoal": self.target_subgoal,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "relation": self.relation.value,
        }


class SkillGraph:
    """Layer 2: one sub-goal-owned subgraph per Layer-1 sub-goal.

    The skill graph is an aggregate root.  Each
    :class:`SkillSubgraph` owns its nodes and internal relations, while this
    object owns the edges crossing sub-goal boundaries.  ``nodes`` and ``edges``
    expose read-only flattened views for planners and legacy query code.
    """

    def __init__(
        self,
        task: str,
        subgoal_graph: Optional[SubGoalGraph] = None,
    ) -> None:
        if not task:
            raise ValueError("skill graph task must be non-empty")
        self.task = task
        self.subgoal_graph = subgoal_graph
        self._subgraphs: Dict[str, SkillSubgraph] = {}
        self._cross_edges: List[CrossSubgraphEdge] = []

    @property
    def subgraphs(self) -> Mapping[str, SkillSubgraph]:
        return dict(self._subgraphs)

    @property
    def cross_edges(self) -> Tuple[CrossSubgraphEdge, ...]:
        return tuple(self._cross_edges)

    @property
    def nodes(self) -> Mapping[str, SkillNode]:
        return {
            node_id: node
            for subgraph in self._subgraphs.values()
            for node_id, node in subgraph.nodes.items()
        }

    @property
    def edges(self) -> Tuple[SkillEdge, ...]:
        """Flattened read-only view; sub-goal-level relations are expanded here.

        A sub-goal-level relation yields one edge per achiever of its open
        endpoint.  That is the correct *partial order over candidates*: every
        achiever precedes the downstream node, and exactly one of them will be
        chosen by :class:`~mshab.skills.plan.SkillPlanner`.
        """

        internal = tuple(
            edge for subgraph in self._subgraphs.values() for edge in subgraph.edges
        )
        cross = tuple(
            edge
            for item in self._cross_edges
            for edge in item.edges(self._subgraphs)
        )
        return internal + cross

    def add_subgraph(self, subgraph: SkillSubgraph) -> None:
        if subgraph.task != self.task:
            raise ValueError("subgraph task must match skill graph task")
        if subgraph.subgoal_id in self._subgraphs:
            raise ValueError(
                "duplicate sub-goal skill subgraph {!r}".format(subgraph.subgoal_id)
            )
        if (
            self.subgoal_graph is not None
            and subgraph.subgoal_id not in self.subgoal_graph.subgoals
        ):
            raise ValueError(
                "subgraph references unknown sub-goal {!r}".format(
                    subgraph.subgoal_id
                )
            )
        duplicate_nodes = sorted(set(subgraph.nodes) & set(self.nodes))
        if duplicate_nodes:
            raise ValueError("duplicate skill nodes {}".format(duplicate_nodes))
        subgraph._seal()
        self._subgraphs[subgraph.subgoal_id] = subgraph

    def replace_subgraph(self, subgraph: SkillSubgraph) -> None:
        """Swap in a revised implementation for an already-registered sub-goal.

        This is the commit half of
        :class:`~mshab.skills.extension.SkillSubgraphExtension`: a sealed
        subgraph is never edited in place, it is replaced by a validated
        successor.  The replacement must keep every node that a cross-sub-goal
        relation still points at, and must not collide with another sub-goal's ids.
        """

        subgoal_id = subgraph.subgoal_id
        current = self._require_subgraph(subgoal_id)
        if subgraph is current:
            raise ValueError("replacement must be a distinct subgraph object")
        if subgraph.task != self.task:
            raise ValueError("subgraph task must match skill graph task")
        foreign = {
            node_id
            for other_id, other in self._subgraphs.items()
            if other_id != subgoal_id
            for node_id in other.nodes
        }
        duplicate_nodes = sorted(set(subgraph.nodes) & foreign)
        if duplicate_nodes:
            raise ValueError("duplicate skill nodes {}".format(duplicate_nodes))
        subgraph._seal()
        self._subgraphs[subgoal_id] = subgraph
        try:
            self._check_relation_integrity()
            self.execution_order()
        except (KeyError, ValueError):
            self._subgraphs[subgoal_id] = current
            raise

    def add_node(self, node: SkillNode, subgoal_id: Optional[str] = None) -> None:
        """Compatibility helper; explicit ``add_subgraph`` is preferred.

        A node can only be inserted this way when its owning sub-goal is explicit
        (or uniquely declared by ``node.achieves``).  Instrumental nodes should
        be added directly to a :class:`SkillSubgraph`.
        """

        owner = subgoal_id
        if owner is None and len(node.achieves) == 1:
            owner = node.achieves[0]
        if owner is None:
            raise ValueError("subgoal_id is required for an instrumental skill node")
        subgraph = self._subgraphs.get(owner)
        if subgraph is None:
            subgraph = SkillSubgraph(owner, self.task)
            subgraph.add_node(node)
            self.add_subgraph(subgraph)
        else:
            raise RuntimeError(
                "subgraph {!r} is already registered; assemble it before "
                "calling add_subgraph()".format(owner)
            )

    def relate(self, source: str, target: str, relation: SkillRelation) -> None:
        source_subgoal = self.owner_of(source)
        target_subgoal = self.owner_of(target)
        if source_subgoal == target_subgoal:
            raise RuntimeError(
                "internal relation belongs to sealed subgraph {!r}; add it "
                "before calling add_subgraph()".format(source_subgoal)
            )
        self.relate_subgraphs(source_subgoal, target_subgoal, source, target, relation)

    def relate_subgraphs(
        self,
        source_subgoal: str,
        target_subgoal: str,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        relation: SkillRelation = SkillRelation.ENABLES,
    ) -> None:
        """Relate two sub-goal subgraphs; ``None`` endpoints mean *any achiever*."""

        self._require_subgraph(source_subgoal)
        self._require_subgraph(target_subgoal)
        item = CrossSubgraphEdge(
            source_subgoal, target_subgoal, source_node, target_node, relation
        )
        # Resolving here verifies both endpoint owners before anything is stored.
        new_edges = item.edges(self._subgraphs)
        existing = [
            edge
            for previous in self._cross_edges
            for edge in previous.edges(self._subgraphs)
        ]
        for edge in new_edges:
            _check_duplicate_edge(existing, edge)
            existing.append(edge)
        self._cross_edges.append(item)
        try:
            self.execution_order()  # reject causal cycles at insertion time
        except ValueError:
            self._cross_edges.pop()
            raise

    def owner_of(self, node_id: str) -> str:
        owners = [
            subgoal_id
            for subgoal_id, subgraph in self._subgraphs.items()
            if node_id in subgraph.nodes
        ]
        if not owners:
            raise KeyError("unknown skill node {!r}".format(node_id))
        return owners[0]

    def subgraph_for_subgoal(self, subgoal_id: str) -> SkillSubgraph:
        """Return the Layer-2 implementation owned by a Layer-1 sub-goal."""

        if self.subgoal_graph is not None and subgoal_id not in self.subgoal_graph.subgoals:
            raise KeyError("unknown sub-goal {!r}".format(subgoal_id))
        return self._require_subgraph(subgoal_id)

    def alternatives(self, node_id: str) -> Tuple[SkillNode, ...]:
        """The complete ALTERNATIVE_TO connected component, excluding self."""

        self._require_node(node_id)
        nodes = self.nodes
        adjacency: Dict[str, Set[str]] = {item: set() for item in nodes}
        for edge in self.edges:
            if edge.relation == SkillRelation.ALTERNATIVE_TO:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)
        seen = {node_id}
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            for candidate in adjacency[current] - seen:
                seen.add(candidate)
                frontier.append(candidate)
        seen.remove(node_id)
        return tuple(nodes[item] for item in sorted(seen))

    def fallbacks(self, node_id: str) -> Tuple[SkillNode, ...]:
        self._require_node(node_id)
        nodes = self.nodes
        return tuple(
            nodes[edge.target]
            for edge in self.edges
            if edge.source == node_id and edge.relation == SkillRelation.FALLBACK_TO
        )

    def candidates_for_subgoal(self, subgoal_id: str) -> Tuple[SkillNode, ...]:
        """Candidate skill nodes that achieve one sub-goal."""

        if self.subgoal_graph is None:
            raise RuntimeError("skill graph has no sub-goal graph")
        if subgoal_id not in self.subgoal_graph.subgoals:
            raise KeyError("unknown sub-goal {!r}".format(subgoal_id))
        subgraph = self._subgraphs.get(subgoal_id)
        return subgraph.achievers if subgraph is not None else ()

    def uncovered_subgoals(self) -> Tuple[SubGoal, ...]:
        """Functional subgoals for which the graph currently has no candidate."""

        if self.subgoal_graph is None:
            return ()
        return tuple(
            subgoal
            for subgoal in sorted(self.subgoal_graph.subgoals.values(), key=lambda item: item.id)
            if subgoal.id not in self._subgraphs or not self._subgraphs[subgoal.id].achievers
        )

    def prerequisite_groups(self, node_id: str) -> Tuple[FrozenSet[str], ...]:
        """Causal prerequisites as disjunctive groups.

        Each group must be satisfied by *at least one* completed member.  A
        node-level relation yields a singleton group; a sub-goal-level relation
        yields one group holding every achiever of that sub-goal, so recovering
        through a fallback still satisfies the downstream dependency.
        """

        self._require_node(node_id)
        groups: List[FrozenSet[str]] = []
        for subgraph in self._subgraphs.values():
            for edge in subgraph.edges:
                group = _causal_requirement(edge.source, edge.target, edge.relation, node_id)
                if group is not None:
                    groups.append(frozenset(group))
        for item in self._cross_edges:
            if item.relation not in _CAUSAL_RELATIONS:
                continue
            sources, targets = self._relation_sides(item)
            if item.relation == SkillRelation.ENABLES and node_id in targets:
                groups.append(frozenset(sources))
            elif item.relation == SkillRelation.REQUIRES and node_id in sources:
                groups.append(frozenset(targets))
        return tuple(sorted(groups, key=lambda group: sorted(group)))

    def prerequisites(self, node_id: str) -> Tuple[SkillNode, ...]:
        """Every node appearing in any prerequisite group.

        This is the conjunctive flattening, kept for display and for the common
        case where each group is a singleton.  Scheduling must use
        :meth:`prerequisite_groups`, which preserves the disjunction.
        """

        nodes = self.nodes
        required = {item for group in self.prerequisite_groups(node_id) for item in group}
        return tuple(nodes[item] for item in sorted(required))

    def ready_nodes(self, completed: Iterable[str] = ()) -> Tuple[SkillNode, ...]:
        """Dependency-ready nodes, without consulting an environment.

        Contract admission belongs to :class:`SkillRuntime`, because facts and
        policy readiness are Layer-3/4 environment-specific concerns.
        """

        completed_set = set(completed)
        nodes = self.nodes
        unknown_completed = sorted(completed_set - set(nodes))
        if unknown_completed:
            raise KeyError("unknown completed nodes {}".format(unknown_completed))
        ready = []
        for node in nodes.values():
            if node.id in completed_set:
                continue
            if all(
                group & completed_set
                for group in self.prerequisite_groups(node.id)
            ):
                ready.append(node)
        return tuple(sorted(ready, key=lambda item: item.id))

    def _relation_sides(
        self, item: CrossSubgraphEdge
    ) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        pairs = item.endpoints(self._subgraphs)
        return (
            frozenset(source for source, _ in pairs),
            frozenset(target for _, target in pairs),
        )

    def execution_order(self) -> Tuple[str, ...]:
        return _topological_order(
            self.nodes, _causal_pairs(self.edges), "skill graph"
        )

    def validate(self) -> None:
        """Whole-aggregate invariants that no single subgraph can check alone."""

        for subgraph in self._subgraphs.values():
            subgraph.validate()
        self._check_relation_integrity()
        self.execution_order()
        if self.subgoal_graph is None:
            return
        uncovered = [subgoal.id for subgoal in self.uncovered_subgoals()]
        if uncovered:
            raise ValueError("sub-goals without an achiever {}".format(uncovered))
        connected = {
            (item.source_subgoal, item.target_subgoal)
            for item in self._cross_edges
            if item.relation == SkillRelation.ENABLES
        }
        missing = sorted(
            "{} -> {}".format(dependency.source, dependency.target)
            for dependency in self.subgoal_graph.dependencies
            if (dependency.source, dependency.target) not in connected
        )
        if missing:
            raise ValueError(
                "sub-goal dependencies without a Layer-2 ENABLES relation {}".format(missing)
            )

    def _check_relation_integrity(self) -> None:
        """Every cross-sub-goal relation still resolves to owned, existing nodes."""

        seen: List[SkillEdge] = []
        for item in self._cross_edges:
            for edge in item.edges(self._subgraphs):
                _check_duplicate_edge(seen, edge)
                seen.append(edge)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "subgoal_graph": self.subgoal_graph.as_dict() if self.subgoal_graph else None,
            "subgraphs": [
                self._subgraphs[subgoal_id].as_dict()
                for subgoal_id in self._ordered_subgoal_ids()
            ],
            "cross_edges": [
                item.as_dict() for item in self._cross_edges
            ],
            # Derived flattened views, retained for generic planners.  The
            # subgraphs above remain the single source of truth: these are
            # recomputed on load, never trusted as input.
            "_derived": True,
            "nodes": [
                _node_dict(node)
                for node in sorted(self.nodes.values(), key=lambda item: item.id)
            ],
            "edges": [_edge_dict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        subgoal_graph: Optional[SubGoalGraph] = None,
    ) -> "SkillGraph":
        where = "skill_graph"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("task",),
            optional=(
                "subgoal_graph",
                "subgraphs",
                "cross_edges",
                "nodes",
                "edges",
                "execution_order",
                "candidate_partial_order",
                "schema_version",
                "_derived",
            ),
        )
        task = schema.require_identifier(payload, "task", where=where)
        if subgoal_graph is None and isinstance(payload.get("subgoal_graph"), Mapping):
            subgoal_graph = SubGoalGraph.from_dict(payload["subgoal_graph"])
        graph = cls(task, subgoal_graph=subgoal_graph)
        for item in schema.require_sequence(payload, "subgraphs", where=where):
            graph.add_subgraph(SkillSubgraph.from_dict(item, task=task))
        for item in schema.require_sequence(payload, "cross_edges", where=where):
            relation = CrossSubgraphEdge.from_dict(item)
            graph.relate_subgraphs(
                relation.source_subgoal,
                relation.target_subgoal,
                relation.source_node,
                relation.target_node,
                relation.relation,
            )
        return graph

    def _adopt(self, other: "SkillGraph") -> None:
        """Replace this aggregate's contents with a validated staging copy."""

        self._subgraphs = dict(other._subgraphs)
        self._cross_edges = list(other._cross_edges)

    def _require_node(self, node_id: str) -> SkillNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError("unknown skill node {!r}".format(node_id)) from exc

    def _require_subgraph(self, subgoal_id: str) -> SkillSubgraph:
        try:
            return self._subgraphs[subgoal_id]
        except KeyError as exc:
            raise KeyError(
                "unknown sub-goal skill subgraph {!r}".format(subgoal_id)
            ) from exc

    def _ordered_subgoal_ids(self) -> Tuple[str, ...]:
        if self.subgoal_graph is None:
            return tuple(sorted(self._subgraphs))
        ordered = [
            subgoal_id
            for subgoal_id in self.subgoal_graph.execution_order()
            if subgoal_id in self._subgraphs
        ]
        ordered.extend(sorted(set(self._subgraphs) - set(ordered)))
        return tuple(ordered)


def _node_dict(node: SkillNode) -> Dict[str, Any]:
    return {
        "id": node.id,
        "contract_id": node.contract_id,
        "arguments": dict(node.arguments),
        "achieves": list(node.achieves),
    }


def _edge_dict(edge: SkillEdge) -> Dict[str, str]:
    return {
        "source": edge.source,
        "target": edge.target,
        "relation": edge.relation.value,
    }


#: Relations that impose an execution ordering.
_CAUSAL_RELATIONS = frozenset({SkillRelation.ENABLES, SkillRelation.REQUIRES})


def _require_relation(value: Any) -> SkillRelation:
    try:
        return SkillRelation(value)
    except ValueError as exc:
        raise ValueError(
            "unknown skill relation {!r}; expected one of {}".format(
                value, sorted(item.value for item in SkillRelation)
            )
        ) from exc


def _check_duplicate_edge(edges: Sequence[SkillEdge], edge: SkillEdge) -> None:
    if edge in edges:
        raise ValueError(
            "duplicate skill relation {} -{}-> {}".format(
                edge.source, edge.relation.value, edge.target
            )
        )
    if edge.relation == SkillRelation.ALTERNATIVE_TO:
        reverse = SkillEdge(edge.target, edge.source, edge.relation)
        if reverse in edges:
            raise ValueError(
                "ALTERNATIVE_TO is symmetric; reverse edge already exists"
            )


def _append_checked_edge(edges: List[SkillEdge], edge: SkillEdge) -> None:
    _check_duplicate_edge(edges, edge)
    edges.append(edge)


def _causal_requirement(
    source: str, target: str, relation: SkillRelation, node_id: str
) -> Optional[Tuple[str, ...]]:
    """The prerequisite one node-level causal edge imposes on ``node_id``."""

    if relation == SkillRelation.ENABLES and target == node_id:
        return (source,)
    if relation == SkillRelation.REQUIRES and source == node_id:
        return (target,)
    return None


def _causal_pairs(edges: Iterable[SkillEdge]) -> List[Tuple[str, str]]:
    pairs = []
    for edge in edges:
        if edge.relation == SkillRelation.ENABLES:
            pairs.append((edge.source, edge.target))
        elif edge.relation == SkillRelation.REQUIRES:
            pairs.append((edge.target, edge.source))
    return pairs


def _topological_order(
    nodes: Mapping[str, Any],
    edges: Sequence[Tuple[str, str]],
    label: str,
) -> Tuple[str, ...]:
    """Stable Kahn order; raise when directed causal edges contain a cycle."""

    indegree = {node_id: 0 for node_id in nodes}
    outgoing: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}
    for source, target in edges:
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1
    frontier = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order = []
    while frontier:
        node_id = frontier.pop(0)
        order.append(node_id)
        for target in sorted(outgoing[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
                frontier.sort()
    if len(order) != len(nodes):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree)
        raise ValueError("{} graph has a causal cycle: {}".format(label, cyclic))
    return tuple(order)
