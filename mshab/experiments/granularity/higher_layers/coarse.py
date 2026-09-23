"""Task-agnostic coarse Layer-1/2 semantic graph.

Layer 1 contains exactly three independent outcome templates: Retrieve,
Deliver, and Restore.  Layer 2 gives each outcome one semantic SkillSubgraph.
Alternative edges distinguish meaningful strategies (for example retrieving
from a fridge versus a drawer); fallback edges connect a failed primary skill
to another recovery skill within the same strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple

from mshab.experiments.granularity.lower_layers.layer3 import (
    Layer3Contracts,
    build_layer3,
)
from mshab.skills.graph import (
    SkillGraph,
    SkillNode,
    SkillRelation,
    SkillSubgraph,
    SubGoal,
    SubGoalGraph,
)
from mshab.skills.model import Contract, ContractType


GRAPH_TASK = "granularity"
COARSE_SCHEMA_VERSION = "mshab.granularity-coarse-higher-layers.v3"

RETRIEVE = "retrieve"
DELIVER = "deliver"
RESTORE = "restore"
SUBGOAL_IDS = (RETRIEVE, DELIVER, RESTORE)


@dataclass(frozen=True)
class SemanticStrategy:
    """Human/VLM-facing meaning for one alternative route in a subgraph."""

    id: str
    subgoal_id: str
    label: str
    description: str
    applicable_when: str
    node_ids: Tuple[str, ...]
    primary_achiever: str
    fallback_achiever: str

    def __post_init__(self) -> None:
        if not self.id or not self.label or not self.description:
            raise ValueError("semantic strategy identity must be non-empty")
        if self.subgoal_id not in SUBGOAL_IDS:
            raise ValueError("strategy references an unknown coarse subgoal")
        if not self.applicable_when:
            raise ValueError("strategy must state when it applies")
        if not self.node_ids:
            raise ValueError("strategy must contain at least one SkillNode")
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ValueError("strategy cannot repeat a SkillNode")
        if self.primary_achiever not in self.node_ids:
            raise ValueError("primary achiever must belong to the strategy")
        if self.fallback_achiever not in self.node_ids:
            raise ValueError("fallback achiever must belong to the strategy")
        if self.primary_achiever == self.fallback_achiever:
            raise ValueError("primary and fallback achievers must differ")

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "subgoal_id": self.subgoal_id,
            "label": self.label,
            "description": self.description,
            "applicable_when": self.applicable_when,
            "node_ids": list(self.node_ids),
            "primary_achiever": self.primary_achiever,
            "fallback_achiever": self.fallback_achiever,
        }


@dataclass(frozen=True)
class StrategyExecutionView:
    """One selected semantic strategy projected into an executable graph.

    The catalog SkillSubgraph intentionally contains several alternatives.
    The reference planner, however, expects one unambiguous primary/fallback
    chain per active SubGoal.  This view is the boundary between those two
    responsibilities: a VLM (or a deterministic oracle) selects a strategy,
    then the planner receives only that strategy's SkillNodes and edges.
    """

    strategy: SemanticStrategy
    subgoals: SubGoalGraph
    skills: SkillGraph

    def __post_init__(self) -> None:
        subgoal_id = self.strategy.subgoal_id
        if tuple(self.subgoals.subgoals) != (subgoal_id,):
            raise ValueError("execution view must contain its selected SubGoal only")
        if tuple(self.skills.subgraphs) != (subgoal_id,):
            raise ValueError("execution view must contain one selected SkillSubgraph")
        if set(self.skills.nodes) != set(self.strategy.node_ids):
            raise ValueError("execution view must contain exactly the strategy nodes")
        self.skills.validate()


@dataclass(frozen=True)
class CoarseHigherLayers:
    """The three independent SubGoals and their three semantic subgraphs."""

    layer3: Layer3Contracts
    subgoals: SubGoalGraph
    skills: SkillGraph
    strategies: Mapping[str, Tuple[SemanticStrategy, ...]]

    def __post_init__(self) -> None:
        strategies = {
            subgoal_id: tuple(items)
            for subgoal_id, items in self.strategies.items()
        }
        if tuple(self.subgoals.subgoals) != SUBGOAL_IDS:
            raise ValueError("Layer 1 must contain Retrieve, Deliver, Restore only")
        if self.subgoals.dependencies:
            raise ValueError("coarse Layer-1 SubGoals must have no directed edges")
        if set(self.skills.subgraphs) != set(SUBGOAL_IDS):
            raise ValueError("every SubGoal must own exactly one SkillSubgraph")
        if self.skills.cross_edges:
            raise ValueError("independent coarse SubGoals need no cross edges")
        if set(strategies) != set(SUBGOAL_IDS):
            raise ValueError("every SkillSubgraph needs semantic strategies")

        all_nodes = self.skills.nodes
        unknown_contracts = sorted(
            {
                node.contract_id
                for node in all_nodes.values()
                if node.contract_id not in self.layer3.ids
            }
        )
        if unknown_contracts:
            raise ValueError(
                "SkillNodes reference unknown Layer-3 Contracts {}".format(
                    unknown_contracts
                )
            )
        seen_strategy_ids = set()
        semantic_owner: Dict[str, str] = {}
        for subgoal_id, items in strategies.items():
            if not items:
                raise ValueError("each SkillSubgraph needs at least one strategy")
            subgraph = self.skills.subgraph_for_subgoal(subgoal_id)
            achievers = {node.id for node in subgraph.achievers}
            fallback_edges = {
                (edge.source, edge.target)
                for edge in subgraph.edges
                if edge.relation == SkillRelation.FALLBACK_TO
            }
            for strategy in items:
                if strategy.id in seen_strategy_ids:
                    raise ValueError("semantic strategy ids must be globally unique")
                seen_strategy_ids.add(strategy.id)
                if strategy.subgoal_id != subgoal_id:
                    raise ValueError("strategy is stored under the wrong subgoal")
                if not set(strategy.node_ids) <= set(subgraph.nodes):
                    raise ValueError("strategy references a foreign SkillNode")
                if strategy.primary_achiever not in achievers:
                    raise ValueError("strategy primary must achieve its SubGoal")
                if strategy.fallback_achiever not in achievers:
                    raise ValueError("strategy fallback must achieve its SubGoal")
                if (
                    strategy.primary_achiever,
                    strategy.fallback_achiever,
                ) not in fallback_edges:
                    raise ValueError("strategy must declare its FALLBACK_TO edge")
                for node_id in strategy.node_ids:
                    if node_id not in all_nodes:
                        raise ValueError("strategy references an unknown SkillNode")
                    if node_id in semantic_owner:
                        raise ValueError(
                            "SkillNode {!r} belongs to two semantic strategies".format(
                                node_id
                            )
                        )
                    semantic_owner[node_id] = strategy.id

        missing_semantics = sorted(set(all_nodes) - set(semantic_owner))
        if missing_semantics:
            raise ValueError(
                "SkillNodes without semantic strategy metadata {}".format(
                    missing_semantics
                )
            )

        self.skills.validate()
        object.__setattr__(
            self,
            "strategies",
            MappingProxyType(strategies),
        )

    @property
    def skill_node_count(self) -> int:
        return len(self.skills.nodes)

    @property
    def strategy_count(self) -> int:
        return sum(len(items) for items in self.strategies.values())

    def strategies_for(self, subgoal_id: str) -> Tuple[SemanticStrategy, ...]:
        return self.strategies[subgoal_id]

    def contract_for(self, skill_node_id: str) -> Contract:
        """Resolve one SkillNode's one-and-only Layer-3 Contract."""

        try:
            node = self.skills.nodes[skill_node_id]
        except KeyError as exc:
            raise KeyError(
                "unknown SkillNode {!r}".format(skill_node_id)
            ) from exc
        return self.layer3.by_id(node.contract_id)

    def strategy(self, strategy_id: str) -> SemanticStrategy:
        """Look up one semantic alternative by its globally unique id."""

        for items in self.strategies.values():
            for strategy in items:
                if strategy.id == strategy_id:
                    return strategy
        raise KeyError("unknown semantic strategy {!r}".format(strategy_id))

    def execution_view(self, strategy_id: str) -> StrategyExecutionView:
        """Project one selected alternative into a planner-compatible graph.

        This does not mutate or replace the canonical three-subgraph catalog.
        It creates a small execution-time view containing one SubGoal and the
        nodes owned by one :class:`SemanticStrategy`.  Cross-strategy
        ``ALTERNATIVE_TO`` edges are deliberately excluded; the strategy's
        internal ``ENABLES`` and ``FALLBACK_TO`` edges are retained.
        """

        strategy = self.strategy(strategy_id)
        subgoal = self.subgoals.subgoals[strategy.subgoal_id]
        selected_subgoals = SubGoalGraph(
            "Execute semantic strategy {}".format(strategy.id)
        )
        selected_subgoals.add_subgoal(subgoal)

        source = self.skills.subgraph_for_subgoal(strategy.subgoal_id)
        selected_ids = set(strategy.node_ids)
        selected_subgraph = SkillSubgraph(strategy.subgoal_id, self.skills.task)
        for node_id in strategy.node_ids:
            selected_subgraph.add_node(source.nodes[node_id])
        for edge in source.edges:
            if edge.source in selected_ids and edge.target in selected_ids:
                selected_subgraph.relate(
                    edge.source,
                    edge.target,
                    edge.relation,
                )

        selected_skills = SkillGraph(
            self.skills.task,
            subgoal_graph=selected_subgoals,
        )
        selected_skills.add_subgraph(selected_subgraph)
        return StrategyExecutionView(
            strategy=strategy,
            subgoals=selected_subgoals,
            skills=selected_skills,
        )


class CoarseGraphBuilder:
    """Build the task-agnostic Layer-1/2 semantic candidate graph."""

    def __init__(self, layer3: Layer3Contracts) -> None:
        self.layer3 = layer3

    def build(self) -> CoarseHigherLayers:
        subgoals = SubGoalGraph(
            "Retrieve, deliver, or restore entities in a household environment"
        )
        subgoals.add_subgoal(
            SubGoal(
                RETRIEVE,
                "holding({object})",
                "Obtain and hold the requested object.",
            )
        )
        subgoals.add_subgoal(
            SubGoal(
                DELIVER,
                "at({object},{destination})",
                "Place the requested object at its destination.",
            )
        )
        subgoals.add_subgoal(
            SubGoal(
                RESTORE,
                "closed({source})",
                "Return an articulated source to its closed state.",
            )
        )

        graph = SkillGraph(GRAPH_TASK, subgoal_graph=subgoals)
        retrieve, retrieve_strategies = self._build_retrieve()
        deliver, deliver_strategies = self._build_deliver()
        restore, restore_strategies = self._build_restore()
        for subgraph in (retrieve, deliver, restore):
            graph.add_subgraph(subgraph)
        graph.validate()

        return CoarseHigherLayers(
            layer3=self.layer3,
            subgoals=subgoals,
            skills=graph,
            strategies={
                RETRIEVE: retrieve_strategies,
                DELIVER: deliver_strategies,
                RESTORE: restore_strategies,
            },
        )

    def _build_retrieve(
        self,
    ) -> Tuple[SkillSubgraph, Tuple[SemanticStrategy, ...]]:
        subgraph = SkillSubgraph(RETRIEVE, GRAPH_TASK)
        strategies = (
            self._retrieve_from_surface(subgraph),
            self._retrieve_from_container(
                subgraph,
                strategy_id="retrieve_from_fridge",
                label="Retrieve from fridge",
                container="fridge",
                semantic_source="fridge",
                source_description="the object is stored in a fridge",
            ),
            self._retrieve_from_container(
                subgraph,
                strategy_id="retrieve_from_drawer",
                label="Retrieve from drawer",
                container="kitchen_counter",
                semantic_source="drawer",
                source_description="the object is stored in a counter drawer",
            ),
        )
        self._connect_alternatives(
            subgraph,
            tuple(item.primary_achiever for item in strategies),
        )
        return subgraph, strategies

    def _retrieve_from_surface(
        self,
        subgraph: SkillSubgraph,
    ) -> SemanticStrategy:
        strategy_id = "retrieve_surface"
        navigate = "navigate_surface_object"
        primary = "pick_surface_object"
        relocalize = "relocalize_surface_object"
        fallback = "regrasp_surface_object"
        self._add_nodes(
            subgraph,
            (
                self._node(navigate, ContractType.NAVIGATE, {"goal": "{object}"}),
                self._node(
                    primary,
                    ContractType.PICK,
                    {"object": "{object}"},
                    achieves=(RETRIEVE,),
                ),
                self._node(
                    relocalize,
                    ContractType.NAVIGATE,
                    {"goal": "{object}"},
                ),
                self._node(
                    fallback,
                    ContractType.PICK,
                    {"object": "{object}"},
                    achieves=(RETRIEVE,),
                ),
            ),
        )
        self._enable_chain(subgraph, navigate, primary)
        self._enable_chain(subgraph, relocalize, fallback)
        subgraph.relate(primary, fallback, SkillRelation.FALLBACK_TO)
        return SemanticStrategy(
            id=strategy_id,
            subgoal_id=RETRIEVE,
            label="Retrieve from open surface",
            description=(
                "Navigate directly to an exposed object and grasp it; "
                "relocalize and regrasp if the first grasp fails."
            ),
            applicable_when="object is exposed on a reachable surface",
            node_ids=(navigate, primary, relocalize, fallback),
            primary_achiever=primary,
            fallback_achiever=fallback,
        )

    def _retrieve_from_container(
        self,
        subgraph: SkillSubgraph,
        *,
        strategy_id: str,
        label: str,
        container: str,
        semantic_source: str,
        source_description: str,
    ) -> SemanticStrategy:
        navigate_source = "navigate_{}_source".format(semantic_source)
        open_source = "open_{}_source".format(semantic_source)
        navigate_object = "navigate_{}_object".format(semantic_source)
        primary = "pick_{}_object".format(semantic_source)
        relocalize = "relocalize_{}_object".format(semantic_source)
        fallback = "regrasp_{}_object".format(semantic_source)
        self._add_nodes(
            subgraph,
            (
                self._node(
                    navigate_source,
                    ContractType.NAVIGATE,
                    {"goal": container},
                ),
                self._node(
                    open_source,
                    ContractType.OPEN,
                    {"articulation": container},
                ),
                self._node(
                    navigate_object,
                    ContractType.NAVIGATE,
                    {"goal": "{object}"},
                ),
                self._node(
                    primary,
                    ContractType.PICK,
                    {"object": "{object}"},
                    achieves=(RETRIEVE,),
                ),
                self._node(
                    relocalize,
                    ContractType.NAVIGATE,
                    {"goal": "{object}"},
                ),
                self._node(
                    fallback,
                    ContractType.PICK,
                    {"object": "{object}"},
                    achieves=(RETRIEVE,),
                ),
            ),
        )
        self._enable_chain(
            subgraph,
            navigate_source,
            open_source,
            navigate_object,
            primary,
        )
        self._enable_chain(subgraph, open_source, relocalize, fallback)
        subgraph.relate(primary, fallback, SkillRelation.FALLBACK_TO)
        return SemanticStrategy(
            id=strategy_id,
            subgoal_id=RETRIEVE,
            label=label,
            description=(
                "Navigate to and open the source, approach the object, and "
                "grasp it; relocalize and regrasp after a failed grasp."
            ),
            applicable_when=source_description,
            node_ids=(
                navigate_source,
                open_source,
                navigate_object,
                primary,
                relocalize,
                fallback,
            ),
            primary_achiever=primary,
            fallback_achiever=fallback,
        )

    def _build_deliver(
        self,
    ) -> Tuple[SkillSubgraph, Tuple[SemanticStrategy, ...]]:
        subgraph = SkillSubgraph(DELIVER, GRAPH_TASK)
        strategies = (
            self._deliver_to_surface(
                subgraph,
                strategy_id="deliver_to_table",
                label="Deliver to table",
                destination="dining_table",
                semantic_destination="table",
                context="destination is a dining table",
            ),
            self._deliver_to_surface(
                subgraph,
                strategy_id="deliver_to_counter",
                label="Deliver to counter",
                destination="countertop",
                semantic_destination="counter",
                context="destination is an open countertop",
            ),
            self._deliver_to_container(
                subgraph,
                strategy_id="deliver_to_fridge",
                label="Deliver into fridge",
                container="fridge",
                semantic_destination="fridge",
                context="destination is inside a fridge",
            ),
            self._deliver_to_container(
                subgraph,
                strategy_id="deliver_to_drawer",
                label="Deliver into drawer",
                container="kitchen_counter",
                semantic_destination="drawer",
                context="destination is inside a counter drawer",
            ),
        )
        self._connect_alternatives(
            subgraph,
            tuple(item.primary_achiever for item in strategies),
        )
        return subgraph, strategies

    def _deliver_to_surface(
        self,
        subgraph: SkillSubgraph,
        *,
        strategy_id: str,
        label: str,
        destination: str,
        semantic_destination: str,
        context: str,
    ) -> SemanticStrategy:
        navigate = "navigate_{}".format(semantic_destination)
        primary = "place_on_{}".format(semantic_destination)
        reapproach = "reapproach_{}".format(semantic_destination)
        fallback = "recover_place_on_{}".format(semantic_destination)
        self._add_nodes(
            subgraph,
            (
                self._node(
                    navigate,
                    ContractType.NAVIGATE,
                    {"goal": destination},
                ),
                self._node(
                    primary,
                    ContractType.PLACE,
                    {"object": "{object}", "destination": destination},
                    achieves=(DELIVER,),
                ),
                self._node(
                    reapproach,
                    ContractType.NAVIGATE,
                    {"goal": destination},
                ),
                self._node(
                    fallback,
                    ContractType.PLACE,
                    {"object": "{object}", "destination": destination},
                    achieves=(DELIVER,),
                ),
            ),
        )
        self._enable_chain(subgraph, navigate, primary)
        self._enable_chain(subgraph, reapproach, fallback)
        subgraph.relate(primary, fallback, SkillRelation.FALLBACK_TO)
        return SemanticStrategy(
            id=strategy_id,
            subgoal_id=DELIVER,
            label=label,
            description=(
                "Navigate to the destination and place the object; "
                "re-approach and retry with another Place skill on failure."
            ),
            applicable_when=context,
            node_ids=(navigate, primary, reapproach, fallback),
            primary_achiever=primary,
            fallback_achiever=fallback,
        )

    def _deliver_to_container(
        self,
        subgraph: SkillSubgraph,
        *,
        strategy_id: str,
        label: str,
        container: str,
        semantic_destination: str,
        context: str,
    ) -> SemanticStrategy:
        navigate_source = "navigate_{}_for_delivery".format(
            semantic_destination
        )
        open_source = "open_{}_for_delivery".format(semantic_destination)
        navigate_destination = "navigate_{}_interior".format(
            semantic_destination
        )
        primary = "place_in_{}".format(semantic_destination)
        reapproach = "reapproach_{}_interior".format(semantic_destination)
        fallback = "recover_place_in_{}".format(semantic_destination)
        self._add_nodes(
            subgraph,
            (
                self._node(
                    navigate_source,
                    ContractType.NAVIGATE,
                    {"goal": container},
                ),
                self._node(
                    open_source,
                    ContractType.OPEN,
                    {"articulation": container},
                ),
                self._node(
                    navigate_destination,
                    ContractType.NAVIGATE,
                    {"goal": container},
                ),
                self._node(
                    primary,
                    ContractType.PLACE,
                    {"object": "{object}", "destination": container},
                    achieves=(DELIVER,),
                ),
                self._node(
                    reapproach,
                    ContractType.NAVIGATE,
                    {"goal": container},
                ),
                self._node(
                    fallback,
                    ContractType.PLACE,
                    {"object": "{object}", "destination": container},
                    achieves=(DELIVER,),
                ),
            ),
        )
        self._enable_chain(
            subgraph,
            navigate_source,
            open_source,
            navigate_destination,
            primary,
        )
        self._enable_chain(subgraph, open_source, reapproach, fallback)
        subgraph.relate(primary, fallback, SkillRelation.FALLBACK_TO)
        return SemanticStrategy(
            id=strategy_id,
            subgoal_id=DELIVER,
            label=label,
            description=(
                "Navigate to and open the destination container, place the "
                "object inside, then re-approach and retry on failure."
            ),
            applicable_when=context,
            node_ids=(
                navigate_source,
                open_source,
                navigate_destination,
                primary,
                reapproach,
                fallback,
            ),
            primary_achiever=primary,
            fallback_achiever=fallback,
        )

    def _build_restore(
        self,
    ) -> Tuple[SkillSubgraph, Tuple[SemanticStrategy, ...]]:
        subgraph = SkillSubgraph(RESTORE, GRAPH_TASK)
        strategies = (
            self._restore_container(
                subgraph,
                strategy_id="restore_fridge",
                label="Restore fridge",
                container="fridge",
                semantic_source="fridge",
                context="the used source or destination is a fridge",
            ),
            self._restore_container(
                subgraph,
                strategy_id="restore_drawer",
                label="Restore drawer",
                container="kitchen_counter",
                semantic_source="drawer",
                context="the used source or destination is a counter drawer",
            ),
        )
        self._connect_alternatives(
            subgraph,
            tuple(item.primary_achiever for item in strategies),
        )
        return subgraph, strategies

    def _restore_container(
        self,
        subgraph: SkillSubgraph,
        *,
        strategy_id: str,
        label: str,
        container: str,
        semantic_source: str,
        context: str,
    ) -> SemanticStrategy:
        navigate = "navigate_{}_for_restore".format(semantic_source)
        primary = "close_{}".format(semantic_source)
        reposition = "reposition_{}".format(semantic_source)
        fallback = "recover_close_{}".format(semantic_source)
        self._add_nodes(
            subgraph,
            (
                self._node(
                    navigate,
                    ContractType.NAVIGATE,
                    {"goal": container},
                ),
                self._node(
                    primary,
                    ContractType.CLOSE,
                    {"articulation": container},
                    achieves=(RESTORE,),
                ),
                self._node(
                    reposition,
                    ContractType.NAVIGATE,
                    {"goal": container},
                ),
                self._node(
                    fallback,
                    ContractType.CLOSE,
                    {"articulation": container},
                    achieves=(RESTORE,),
                ),
            ),
        )
        self._enable_chain(subgraph, navigate, primary)
        self._enable_chain(subgraph, reposition, fallback)
        subgraph.relate(primary, fallback, SkillRelation.FALLBACK_TO)
        return SemanticStrategy(
            id=strategy_id,
            subgoal_id=RESTORE,
            label=label,
            description=(
                "Navigate to and close the articulation; reposition and use "
                "a second Close skill if the first close fails."
            ),
            applicable_when=context,
            node_ids=(navigate, primary, reposition, fallback),
            primary_achiever=primary,
            fallback_achiever=fallback,
        )

    def _node(
        self,
        node_id: str,
        contract_type: ContractType,
        arguments: Mapping[str, str],
        *,
        achieves: Tuple[str, ...] = (),
    ) -> SkillNode:
        return SkillNode(
            id=node_id,
            contract_id=self.layer3.contract(contract_type).id,
            arguments=arguments,
            achieves=achieves,
        )

    @staticmethod
    def _add_nodes(
        subgraph: SkillSubgraph,
        nodes: Iterable[SkillNode],
    ) -> None:
        for node in nodes:
            subgraph.add_node(node)

    @staticmethod
    def _enable_chain(subgraph: SkillSubgraph, *node_ids: str) -> None:
        for source, target in zip(node_ids, node_ids[1:]):
            subgraph.relate(source, target, SkillRelation.ENABLES)

    @staticmethod
    def _connect_alternatives(
        subgraph: SkillSubgraph,
        primary_achievers: Tuple[str, ...],
    ) -> None:
        for source, target in zip(primary_achievers, primary_achievers[1:]):
            subgraph.relate(source, target, SkillRelation.ALTERNATIVE_TO)


def build_coarse_higher_layers(
    layer3: Optional[Layer3Contracts] = None,
) -> CoarseHigherLayers:
    """Build the three task-agnostic coarse SubGoals and SkillSubgraphs."""

    return CoarseGraphBuilder(layer3 or build_layer3()).build()


def coarse_higher_layers_document(
    higher: CoarseHigherLayers,
) -> Dict[str, object]:
    """Serialize the semantic graph for VLM input and human inspection."""

    references = [
        {
            "source": node.id,
            "target": node.contract_id,
            "relation": "references",
        }
        for node in sorted(
            higher.skills.nodes.values(),
            key=lambda item: item.id,
        )
    ]
    contract_usage: Dict[str, int] = {
        contract_id: sum(
            reference["target"] == contract_id for reference in references
        )
        for contract_id in higher.layer3.ids
    }

    layer2 = higher.skills.as_dict()
    layer2["semantic_strategies"] = {
        subgoal_id: [strategy.as_dict() for strategy in strategies]
        for subgoal_id, strategies in higher.strategies.items()
    }
    return {
        "schema_version": COARSE_SCHEMA_VERSION,
        "granularity": "coarse",
        "semantics": {
            "task_classification": "none",
            "layer1_relations": "none",
            "layer1_to_layer2": "one_subgoal_to_one_skill_subgraph",
            "alternative_to": (
                "different semantic contexts that achieve the same SubGoal"
            ),
            "fallback_to": (
                "a failed primary SkillNode transitions to another recovery SkillNode"
            ),
            "skill_node_to_contract": "many_to_one",
            "policy_ids_in_higher_layers": False,
            "execution_protocol": (
                "select one SemanticStrategy for an active SubGoal, project "
                "its execution view, then resolve its directed fallback chain"
            ),
        },
        "summary": {
            "subgoals": len(higher.subgoals.subgoals),
            "subgoal_dependencies": len(higher.subgoals.dependencies),
            "skill_subgraphs": len(higher.skills.subgraphs),
            "semantic_strategies": higher.strategy_count,
            "skill_nodes": higher.skill_node_count,
            "cross_subgraph_edges": len(higher.skills.cross_edges),
            "contract_usage": dict(sorted(contract_usage.items())),
        },
        "layer1": higher.subgoals.as_dict(),
        "layer2": layer2,
        "layer3": {
            "contracts": [
                {
                    "id": contract.id,
                    "contract_type": contract.contract_type_name,
                }
                for contract in sorted(
                    higher.layer3.contracts.values(),
                    key=lambda item: item.id,
                )
            ],
        },
        "layer2_to_layer3": {
            "cardinality": "many_to_one",
            "references": references,
        },
    }
