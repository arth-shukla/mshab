"""First hand-authored, scene-independent MS-HAB skill graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mshab.skills.extension import SkillGraphBuilder, SkillGraphPatch
from mshab.skills.graph import (
    SubGoal,
    SubGoalGraph,
    SkillSubgraph,
    SubGoalDependency,
    SkillGraph,
    SkillNode,
    SkillRelation,
    CrossSubgraphEdge,
)
from mshab.skills.library import ContractLibrary
from mshab.skills.model import (
    CheckpointPolicy,
    CloseContract,
    NavigateContract,
    OpenContract,
    PickContract,
    PlaceContract,
)
from mshab.skills.runtime import SkillGrounder


class SetTableAppleGraphBuilder(SkillGraphBuilder):
    """Manual v1 graph using the downloaded SetTable apple checkpoints.

    The graph uses symbolic category names only.  It can be built before a
    scene is loaded; an environment adapter later maps these names to the
    concrete actor, articulation, and target objects for one episode.
    """

    def propose(
        self,
        goal: str,
        task: str,
        context: Mapping[str, Any],
    ) -> SkillGraphPatch:
        if task != "set_table":
            raise ValueError("SetTableAppleGraphBuilder requires task='set_table'")
        object_name = str(context.get("object", "013_apple"))
        source = str(context.get("source", "fridge"))
        destination = str(context.get("destination", "dining_table"))
        segment = _build_segment_subgraphs(
            task=task,
            label="object",
            object_name=object_name,
            source=source,
            destination=destination,
            subgoal_ids=(
                "source_open",
                "object_retrieved",
                "object_placed",
                "source_closed",
            ),
        )
        return SkillGraphPatch(
            subgoals=segment.subgoals,
            subgoal_dependencies=segment.dependencies,
            skill_subgraphs=segment.subgraphs,
            cross_edges=segment.relations,
        )


class SetTableGraphBuilder(SkillGraphBuilder):
    """Manual v1 of the complete official SetTable semantic graph.

    The official task plan handles the bowl from the kitchen counter first,
    followed by the apple from the fridge.  Each object uses the eight-step
    Navigate/Open/Navigate/Pick/Navigate/Place/Navigate/Close pattern.
    """

    _SEGMENTS = (
        ("bowl", "024_bowl", "kitchen_counter"),
        ("apple", "013_apple", "fridge"),
    )

    def propose(
        self,
        goal: str,
        task: str,
        context: Mapping[str, Any],
    ) -> SkillGraphPatch:
        if task != "set_table":
            raise ValueError("SetTableGraphBuilder requires task='set_table'")
        destination = str(context.get("destination", "dining_table"))
        subgoals = []
        dependencies = []
        subgraphs = []
        relations = []
        previous_close_subgoal = None

        for label, object_name, source in self._SEGMENTS:
            open_subgoal = "{}_source_open".format(label)
            retrieved_subgoal = "{}_retrieved".format(label)
            placed_subgoal = "{}_placed".format(label)
            close_subgoal = "{}_source_closed".format(label)
            segment = _build_segment_subgraphs(
                task=task,
                label=label,
                object_name=object_name,
                source=source,
                destination=destination,
                subgoal_ids=(open_subgoal, retrieved_subgoal, placed_subgoal, close_subgoal),
            )
            subgoals.extend(segment.subgoals)
            dependencies.extend(segment.dependencies)
            subgraphs.extend(segment.subgraphs)
            relations.extend(segment.relations)
            if previous_close_subgoal is not None:
                dependencies.append(SubGoalDependency(previous_close_subgoal, open_subgoal))
            if previous_close_subgoal is not None:
                relations.append(
                    CrossSubgraphEdge(
                        previous_close_subgoal,
                        open_subgoal,
                        None,
                        segment.entry_node,
                        SkillRelation.ENABLES,
                    )
                )
            previous_close_subgoal = close_subgoal

        return SkillGraphPatch(
            subgoals=tuple(subgoals),
            subgoal_dependencies=tuple(dependencies),
            skill_subgraphs=tuple(subgraphs),
            cross_edges=tuple(relations),
        )


@dataclass(frozen=True)
class _SegmentSubgraphs:
    subgoals: tuple
    dependencies: tuple
    subgraphs: tuple
    relations: tuple
    entry_node: str
    exit_node: str


def _build_segment_subgraphs(
    task: str,
    label: str,
    object_name: str,
    source: str,
    destination: str,
    subgoal_ids: tuple,
) -> _SegmentSubgraphs:
    """Create four sub-goal-owned subgraphs for one SetTable object segment."""

    open_subgoal, retrieved_subgoal, placed_subgoal, close_subgoal = subgoal_ids
    prefix = "mshab.{}.".format(task)
    navigate_source = "navigate_to_{}_source".format(label)
    open_source = "open_{}_source".format(label)
    navigate_object = "navigate_to_{}".format(label)
    pick_specialized = "pick_{}_specialized".format(label)
    pick_generic = "pick_{}_generic".format(label)
    navigate_destination = "navigate_{}_to_destination".format(label)
    place_specialized = "place_{}_specialized".format(label)
    place_generic = "place_{}_generic".format(label)
    navigate_back = "navigate_back_to_{}_source".format(label)
    close_source = "close_{}_source".format(label)

    open_subgraph = SkillSubgraph(open_subgoal, task)
    open_subgraph.add_node(
        SkillNode(navigate_source, prefix + "navigate.all", {"goal": source})
    )
    open_subgraph.add_node(
        SkillNode(open_source, prefix + "open." + source, {}, (open_subgoal,))
    )
    open_subgraph.relate(navigate_source, open_source, SkillRelation.ENABLES)

    retrieve_subgraph = SkillSubgraph(retrieved_subgoal, task)
    retrieve_subgraph.add_node(
        SkillNode(navigate_object, prefix + "navigate.all", {"goal": object_name})
    )
    retrieve_subgraph.add_node(
        SkillNode(
            pick_specialized,
            prefix + "pick." + object_name,
            {},
            (retrieved_subgoal,),
        )
    )
    retrieve_subgraph.add_node(
        SkillNode(
            pick_generic,
            prefix + "pick.all",
            {"object": object_name},
            (retrieved_subgoal,),
        )
    )
    retrieve_subgraph.relate(
        navigate_object, pick_specialized, SkillRelation.ENABLES
    )
    retrieve_subgraph.relate(pick_generic, navigate_object, SkillRelation.REQUIRES)
    retrieve_subgraph.relate(pick_specialized, pick_generic, SkillRelation.IS_A)
    retrieve_subgraph.relate(
        pick_specialized, pick_generic, SkillRelation.ALTERNATIVE_TO
    )
    retrieve_subgraph.relate(
        pick_specialized, pick_generic, SkillRelation.FALLBACK_TO
    )

    place_subgraph = SkillSubgraph(placed_subgoal, task)
    place_subgraph.add_node(
        SkillNode(
            navigate_destination,
            prefix + "navigate.all",
            {"goal": destination},
        )
    )
    place_subgraph.add_node(
        SkillNode(
            place_specialized,
            prefix + "place." + object_name,
            {"destination": destination},
            (placed_subgoal,),
        )
    )
    place_subgraph.add_node(
        SkillNode(
            place_generic,
            prefix + "place.all",
            {"object": object_name, "destination": destination},
            (placed_subgoal,),
        )
    )
    place_subgraph.relate(
        navigate_destination, place_specialized, SkillRelation.ENABLES
    )
    place_subgraph.relate(
        place_generic, navigate_destination, SkillRelation.REQUIRES
    )
    place_subgraph.relate(place_specialized, place_generic, SkillRelation.IS_A)
    place_subgraph.relate(
        place_specialized, place_generic, SkillRelation.ALTERNATIVE_TO
    )
    place_subgraph.relate(
        place_specialized, place_generic, SkillRelation.FALLBACK_TO
    )

    close_subgraph = SkillSubgraph(close_subgoal, task)
    close_subgraph.add_node(
        SkillNode(navigate_back, prefix + "navigate.all", {"goal": source})
    )
    close_subgraph.add_node(
        SkillNode(close_source, prefix + "close." + source, {}, (close_subgoal,))
    )
    close_subgraph.relate(navigate_back, close_source, SkillRelation.ENABLES)

    subgoals = (
        SubGoal(open_subgoal, "open({})".format(source)),
        SubGoal(retrieved_subgoal, "holding({})".format(object_name)),
        SubGoal(placed_subgoal, "at({},{})".format(object_name, destination)),
        SubGoal(close_subgoal, "closed({})".format(source)),
    )
    dependencies = (
        SubGoalDependency(open_subgoal, retrieved_subgoal),
        SubGoalDependency(retrieved_subgoal, placed_subgoal),
        SubGoalDependency(placed_subgoal, close_subgoal),
    )
    # Cross-sub-goal edges leave the source endpoint open (``None`` = any achiever
    # of that sub-goal).  Pinning them to the specialized candidate would make the
    # generic fallback a dead end: recovering through it would never enable the
    # next sub-goal.
    relations = (
        CrossSubgraphEdge(
            open_subgoal,
            retrieved_subgoal,
            None,
            navigate_object,
            SkillRelation.ENABLES,
        ),
        CrossSubgraphEdge(
            retrieved_subgoal,
            placed_subgoal,
            None,
            navigate_destination,
            SkillRelation.ENABLES,
        ),
        CrossSubgraphEdge(
            placed_subgoal,
            close_subgoal,
            None,
            navigate_back,
            SkillRelation.ENABLES,
        ),
    )
    return _SegmentSubgraphs(
        subgoals,
        dependencies,
        (open_subgraph, retrieve_subgraph, place_subgraph, close_subgraph),
        relations,
        navigate_source,
        close_source,
    )


_APPLE_GOAL = (
    "Retrieve the apple, close the fridge, and set the apple on the table."
)
_SET_TABLE_GOAL = (
    "Set the table with the bowl and apple, then close their storage."
)


# Canonical SetTable contract inventory.  This is a manifest, not filesystem
# discovery: the semantic library and its contracts must be constructible on a
# clean clone even before the external policy artifacts have been downloaded.
_SET_TABLE_CONTRACTS = (
    (NavigateContract, "navigate", "all"),
    (OpenContract, "open", "fridge"),
    (OpenContract, "open", "kitchen_counter"),
    (PickContract, "pick", "013_apple"),
    (PickContract, "pick", "024_bowl"),
    (PickContract, "pick", "all"),
    (PlaceContract, "place", "013_apple"),
    (PlaceContract, "place", "024_bowl"),
    (PlaceContract, "place", "all"),
    (CloseContract, "close", "fridge"),
    (CloseContract, "close", "kitchen_counter"),
)


def build_set_table_library(checkpoint_root: Path) -> ContractLibrary:
    """Build the canonical SetTable inventory around the official RL layout.

    The 11 contracts and their 11 RL checkpoint policies always exist; each
    policy computes ``ready/missing/partial`` at runtime from
    ``checkpoint_root``, so artifact availability never changes the canonical
    Layer-1/2 graph or prevents documentation from being generated.

    Bindings are many-to-many.  Every contract is first bound to its own
    checkpoint, then :meth:`ContractLibrary.bind_generic_policies` also binds
    the ``pick.all`` and ``place.all`` checkpoints to the specialized
    ``pick.*``/``place.*`` contracts they can execute: 15 bindings in total.
    """

    root = Path(checkpoint_root)
    library = ContractLibrary()
    for contract_class, raw_type, target in _SET_TABLE_CONTRACTS:
        contract = contract_class(task="set_table", target=target)
        library.register(contract)
        policy = CheckpointPolicy.from_leaf(root, "rl", "set_table", raw_type, target)
        library.register_policy(policy)
        library.bind(contract.id, policy.id)
    library.bind_generic_policies()
    return library


@dataclass(frozen=True)
class StarterSkillStack:
    """Concrete handles for all four layers of the starter example."""

    subgoal_graph: SubGoalGraph
    skill_graph: SkillGraph
    library: ContractLibrary
    grounder: SkillGrounder

    @property
    def grounded_skills(self):
        """Layer-3 grounded skills keyed by Layer-2 node id."""

        return self.grounder.grounded_skills(self.skill_graph)


def build_set_table_apple_graph(
    goal: str = _APPLE_GOAL,
    **context: Any
):
    """Build scene-independent Layer 1 and Layer 2 only."""

    return SetTableAppleGraphBuilder().build(
        goal=goal,
        task="set_table",
        context=context,
    )


def build_set_table_starter(checkpoint_root: Path, **context: Any) -> StarterSkillStack:
    """Build the four-layer starter stack from the canonical contract manifest."""

    library = build_set_table_library(Path(checkpoint_root))
    subgoals, graph = SetTableAppleGraphBuilder().build(
        goal=_APPLE_GOAL,
        task="set_table",
        context=context,
        library=library,
    )
    grounder = SkillGrounder(library)
    # Eagerly bind every Layer-3 contract so an invalid manual graph fails now.
    grounder.grounded_skills(graph)
    return StarterSkillStack(subgoals, graph, library, grounder)


def build_set_table_graph(
    goal: str = _SET_TABLE_GOAL,
    **context: Any
):
    """Build complete scene-independent SetTable Layer 1 and Layer 2 graphs."""

    return SetTableGraphBuilder().build(
        goal=goal,
        task="set_table",
        context=context,
    )


def build_set_table_stack(
    checkpoint_root: Path, **context: Any
) -> StarterSkillStack:
    """Build all four layers of the complete manual SetTable graph."""

    library = build_set_table_library(Path(checkpoint_root))
    # One patch, validated against the library and then applied: validating a
    # second, independently proposed patch only worked because propose() is
    # deterministic.
    subgoals, graph = SetTableGraphBuilder().build(
        goal=_SET_TABLE_GOAL,
        task="set_table",
        context=context,
        library=library,
    )
    grounder = SkillGrounder(library)
    grounder.grounded_skills(graph)
    return StarterSkillStack(subgoals, graph, library, grounder)
