"""Regression tests for the four Layer-1/2 semantics fixed in graph v1.

Each class pins one behaviour that the structural/count assertions elsewhere
could not have caught: the previous suite was fully green while the fallback
path was a dead end, a sealed subgraph was reopenable, and a rejected patch
left the live graph half-modified.
"""

from __future__ import annotations

import json
from unittest import TestCase

from mshab.skills import (
    SubGoal,
    SubGoalGraph,
    SubGoalDependency,
    SkillSubgraph,
    SkillSubgraphExtension,
    NoViableCandidate,
    SchemaError,
    SkillGraph,
    SkillEdge,
    SkillGraphPatch,
    ContractLibrary,
    SkillNode,
    SkillPlanner,
    SkillRelation,
    CrossSubgraphEdge,
    build_set_table_graph,
)


def _pick(node_id, target, goal, **arguments):
    return SkillNode(
        node_id, "mshab.set_table.pick." + target, arguments, achieves=(goal,)
    )


def _tiny_graph():
    """One goal, one achiever, plus a second goal downstream of it."""

    subgoals = SubGoalGraph("tiny")
    subgoals.add_subgoal(SubGoal("retrieved", "holding(024_bowl)"))
    subgoals.add_subgoal(SubGoal("placed", "at(024_bowl,table)"))
    subgoals.add_dependency("retrieved", "placed")
    graph = SkillGraph("set_table", subgoal_graph=subgoals)

    retrieved = SkillSubgraph("retrieved", "set_table")
    retrieved.add_node(_pick("pick_primary", "024_bowl", "retrieved"))
    graph.add_subgraph(retrieved)

    placed = SkillSubgraph("placed", "set_table")
    placed.add_node(
        SkillNode(
            "place_it",
            "mshab.set_table.place.024_bowl",
            {"destination": "table"},
            achieves=("placed",),
        )
    )
    graph.add_subgraph(placed)
    graph.relate_subgraphs("retrieved", "placed", None, "place_it")
    return subgoals, graph


class FallbackReachabilityTests(TestCase):
    """A goal-level ENABLES must be satisfiable by *any* achiever."""

    def test_recovering_through_the_fallback_unblocks_the_next_goal(self):
        _, graph = build_set_table_graph()
        completed = [
            "navigate_to_bowl_source",
            "open_bowl_source",
            "navigate_to_bowl",
            "pick_bowl_generic",
        ]

        ready = [node.id for node in graph.ready_nodes(completed)]

        # Before the fix this returned only pick_bowl_specialized -- the node
        # that had just failed -- so the rollout could never continue.
        self.assertIn("navigate_bowl_to_destination", ready)

    def test_downstream_prerequisite_is_a_disjunction_over_achievers(self):
        _, graph = build_set_table_graph()

        groups = graph.prerequisite_groups("navigate_bowl_to_destination")

        self.assertEqual(
            [sorted(group) for group in groups],
            [["pick_bowl_generic", "pick_bowl_specialized"]],
        )

    def test_every_cross_goal_relation_survives_a_full_generic_rollout(self):
        subgoals, graph = build_set_table_graph()
        planner = SkillPlanner(subgoals, graph)
        primaries = [
            planner.select_achiever(subgraph).id
            for subgraph in graph.subgraphs.values()
            if len(planner.fallback_chain(subgraph)) > 1
        ]

        recovery = planner.plan(failed=primaries)

        self.assertEqual(len(recovery.order), 16)
        self.assertEqual(recovery.order[-1], "close_apple_source")
        self.assertEqual(recovery.selections["bowl_retrieved"], "pick_bowl_generic")
        self.assertEqual(recovery.selections["apple_placed"], "place_apple_generic")


class ExecutionPlanTests(TestCase):
    """``execution_order()`` ranks candidates; ``SkillPlanner`` picks a path."""

    EXPECTED = (
        "navigate_to_bowl_source",
        "open_bowl_source",
        "navigate_to_bowl",
        "pick_bowl_specialized",
        "navigate_bowl_to_destination",
        "place_bowl_specialized",
        "navigate_back_to_bowl_source",
        "close_bowl_source",
    )

    def test_plan_selects_exactly_one_achiever_per_goal(self):
        subgoals, graph = build_set_table_graph()

        plan = SkillPlanner(subgoals, graph).plan()

        self.assertEqual(len(plan.order), 16)
        self.assertEqual(plan.order[:8], self.EXPECTED)
        self.assertEqual(len(plan.selections), 8)
        self.assertFalse([item for item in plan.order if item.endswith("_generic")])

    def test_candidate_partial_order_is_not_the_plan(self):
        subgoals, graph = build_set_table_graph()

        order = graph.execution_order()
        plan = SkillPlanner(subgoals, graph).plan()

        self.assertEqual(len(order), 20)
        self.assertNotEqual(tuple(order), plan.order)
        # Both picks appear, and the alphabetical tie-break even puts the
        # fallback first: following this order would execute the bowl twice.
        self.assertLess(
            order.index("pick_bowl_generic"), order.index("pick_bowl_specialized")
        )

    def test_exhausting_a_fallback_chain_raises_rather_than_looping(self):
        subgoals, graph = build_set_table_graph()
        planner = SkillPlanner(subgoals, graph)

        with self.assertRaisesRegex(NoViableCandidate, "every achiever"):
            planner.plan(failed=("pick_bowl_specialized", "pick_bowl_generic"))

    def test_goal_dependency_without_a_layer_two_relation_is_rejected(self):
        subgoals, graph = _tiny_graph()
        subgoals.add_subgoal(SubGoal("closed", "closed(fridge)"))
        subgoals.add_dependency("placed", "closed")
        orphan = SkillSubgraph("closed", "set_table")
        orphan.add_node(
            SkillNode(
                "close_it", "mshab.set_table.close.fridge", {}, achieves=("closed",)
            )
        )
        graph.add_subgraph(orphan)

        with self.assertRaisesRegex(ValueError, "without a Layer-2 ENABLES"):
            graph.validate()

    def test_planner_obeys_an_additional_layer_two_dependency(self):
        """Layer 2, not alphabetical Layer-1 order, controls readiness."""

        subgoals = SubGoalGraph("two otherwise independent outcomes")
        subgoals.add_subgoal(SubGoal("a", "a_done()"))
        subgoals.add_subgoal(SubGoal("b", "b_done()"))
        graph = SkillGraph("set_table", subgoal_graph=subgoals)

        for subgoal_id in ("a", "b"):
            subgraph = SkillSubgraph(subgoal_id, "set_table")
            subgraph.add_node(
                SkillNode(
                    subgoal_id,
                    "mshab.set_table.pick.all",
                    {"object": subgoal_id},
                    achieves=(subgoal_id,),
                )
            )
            graph.add_subgraph(subgraph)

        # Layer 1 deliberately leaves A/B unordered.  This Layer-2 relation is
        # therefore the only source of the required B -> A execution order.
        graph.relate("b", "a", SkillRelation.ENABLES)
        planner = SkillPlanner(subgoals, graph)

        self.assertEqual(planner.decide().id, "b")
        self.assertEqual(planner.decide(completed=("b",)).id, "a")
        self.assertEqual(planner.plan().order, ("b", "a"))


class SealingTests(TestCase):
    """A registered subgraph is frozen, not merely guarded on two methods."""

    def test_sealed_subgraph_rejects_every_attribute_assignment(self):
        _, graph = build_set_table_graph()
        subgraph = graph.subgraphs["bowl_retrieved"]

        for name, value in (("subgoal_id", "apple_placed"), ("_sealed", False)):
            with self.assertRaisesRegex(RuntimeError, "registered and sealed"):
                setattr(subgraph, name, value)

        self.assertEqual(subgraph.subgoal_id, "bowl_retrieved")
        self.assertTrue(subgraph.sealed)

    def test_sealed_subgraph_rejects_structural_mutation(self):
        _, graph = build_set_table_graph()
        subgraph = graph.subgraphs["bowl_retrieved"]

        with self.assertRaisesRegex(RuntimeError, "registered and sealed"):
            subgraph.add_node(_pick("smuggled", "all", "bowl_retrieved", object="x"))
        self.assertEqual(len(graph.nodes), 20)

    def test_unsealed_copy_is_independent_of_the_registered_original(self):
        _, graph = build_set_table_graph()

        clone = graph.subgraphs["bowl_retrieved"].unsealed_copy()
        clone.add_node(_pick("extra", "all", "bowl_retrieved", object="024_bowl"))

        self.assertIn("extra", clone.nodes)
        self.assertNotIn("extra", graph.nodes)


class SubgraphExtensionTests(TestCase):
    """Adding a candidate to a goal that is already registered."""

    def test_extension_adds_a_candidate_and_keeps_cross_goal_relations(self):
        subgoals, graph = _tiny_graph()
        patch = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("pick_backup", "all", "retrieved", object="024_bowl"),),
            edges=(
                SkillEdge("pick_primary", "pick_backup", SkillRelation.FALLBACK_TO),
            ),
        )

        patch.apply(subgoals, graph)

        self.assertEqual(len(graph.subgraphs["retrieved"].achievers), 2)
        self.assertEqual(graph.owner_of("pick_backup"), "retrieved")
        # The pre-existing goal-level ENABLES now covers the new candidate too.
        self.assertEqual(
            [sorted(group) for group in graph.prerequisite_groups("place_it")],
            [["pick_backup", "pick_primary"]],
        )
        planner = SkillPlanner(subgoals, graph)
        self.assertEqual(
            planner.plan(failed=("pick_primary",)).selections["retrieved"],
            "pick_backup",
        )

    def test_extension_rejects_a_node_id_owned_by_another_goal(self):
        subgoals, graph = _tiny_graph()
        patch = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("place_it", "all", "retrieved", object="024_bowl"),),
        )

        with self.assertRaisesRegex(ValueError, "duplicate skill nodes"):
            patch.apply(subgoals, graph)
        self.assertEqual(graph.owner_of("place_it"), "placed")

    def test_extension_rejects_a_node_claiming_another_goal(self):
        subgoals, graph = _tiny_graph()
        patch = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("pick_backup", "all", "placed", object="024_bowl"),),
        )

        with self.assertRaisesRegex(ValueError, "achieves"):
            patch.apply(subgoals, graph)


class PatchAtomicityTests(TestCase):
    """A rejected patch must leave Layer 1 and Layer 2 byte-identical."""

    def _snapshot(self, subgoals, graph):
        return json.dumps([subgoals.as_dict(), graph.as_dict()], sort_keys=True)

    def test_rejected_relation_leaves_both_graphs_untouched(self):
        subgoals, graph = _tiny_graph()
        before = self._snapshot(subgoals, graph)
        stranger = SkillSubgraph("inspected", "set_table")
        stranger.add_node(
            SkillNode(
                "inspect", "mshab.set_table.pick.all", {"object": "x"},
                achieves=("inspected",),
            )
        )
        patch = SkillGraphPatch(
            subgoals=(SubGoal("inspected", "inspected(x)"),),
            subgoal_dependencies=(SubGoalDependency("placed", "inspected"),),
            skill_subgraphs=(stranger,),
            cross_edges=(
                CrossSubgraphEdge("placed", "inspected", None, "does_not_exist"),
            ),
        )

        with self.assertRaises(ValueError):
            patch.apply(subgoals, graph)

        self.assertEqual(self._snapshot(subgoals, graph), before)
        self.assertNotIn("inspected", subgoals.subgoals)

    def test_rejected_extension_leaves_the_original_subgraph_installed(self):
        subgoals, graph = _tiny_graph()
        before = self._snapshot(subgoals, graph)
        patch = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("place_it", "all", "retrieved", object="x"),),
        )

        with self.assertRaises(ValueError):
            patch.apply(subgoals, graph)

        self.assertEqual(self._snapshot(subgoals, graph), before)

    def test_patch_rejects_foreign_objects_before_touching_the_graph(self):
        class NotASubgraph:
            subgoal_id = "spoofed"
            task = "set_table"
            nodes = {}
            edges = ()

        with self.assertRaisesRegex(TypeError, "SkillSubgraph"):
            SkillGraphPatch(skill_subgraphs=(NotASubgraph(),))

    def test_apply_rejects_a_skill_the_library_does_not_have(self):
        subgoals, graph = _tiny_graph()
        before = self._snapshot(subgoals, graph)
        invented = SkillSubgraph("inspected", "set_table")
        invented.add_node(
            SkillNode(
                "inspect", "mshab.set_table.teleport.moon", {},
                achieves=("inspected",),
            )
        )
        patch = SkillGraphPatch(
            subgoals=(SubGoal("inspected", "inspected(x)"),),
            skill_subgraphs=(invented,),
        )

        with self.assertRaisesRegex(KeyError, "unregistered contracts"):
            patch.apply(subgoals, graph, library=ContractLibrary())

        self.assertEqual(self._snapshot(subgoals, graph), before)


class PatchSchemaTests(TestCase):
    """Untrusted patch documents are parsed strictly or not at all."""

    VALID = {
        "subgoals": [{"id": "inspected", "predicate": "inspected(013_apple)"}],
        "skill_subgraphs": [
            {
                "subgoal_id": "inspected",
                "nodes": [
                    {
                        "id": "inspect_apple",
                        "contract_id": "mshab.set_table.pick.all",
                        "arguments": {"object": "013_apple"},
                        "achieves": ["inspected"],
                    }
                ],
                "edges": [],
            }
        ],
    }

    def test_patch_round_trips_through_dict(self):
        patch = SkillGraphPatch.from_dict(self.VALID, task="set_table")

        restored = SkillGraphPatch.from_dict(
            json.loads(json.dumps(patch.as_dict())), task="set_table"
        )

        self.assertEqual(restored.as_dict(), patch.as_dict())
        self.assertEqual(restored.subgoals[0].id, "inspected")
        self.assertEqual(
            dict(restored.skill_subgraphs[0].nodes["inspect_apple"].arguments),
            {"object": "013_apple"},
        )

    def test_set_table_catalog_subgraphs_round_trip(self):
        _, graph = build_set_table_graph()

        restored = SkillGraph.from_dict(
            json.loads(json.dumps(graph.as_dict()))
        )

        self.assertEqual(restored.as_dict(), graph.as_dict())

    def test_malformed_patches_are_rejected(self):
        cases = {
            "unknown key": {"subgoals": [], "wat": 1},
            "unknown goal key": {"subgoals": [{"id": "g", "predicate": "p()", "x": 1}]},
            "bad identifier": {"subgoals": [{"id": "../etc", "predicate": "p()"}]},
            "unknown relation": {
                "cross_edges": [
                    {"source_subgoal": "a", "target_subgoal": "b", "relation": "teleports"}
                ]
            },
            "nested argument": {
                "skill_subgraphs": [
                    {
                        "subgoal_id": "g",
                        "nodes": [
                            {
                                "id": "n",
                                "contract_id": "mshab.set_table.pick.all",
                                "arguments": {"waypoints": [1, 2, 3]},
                            }
                        ],
                    }
                ]
            },
            "subgoals not a list": {"subgoals": {"id": "g"}},
            "bad schema version": {"schema_version": "something.else"},
        }
        for label, payload in cases.items():
            with self.subTest(label):
                with self.assertRaises((SchemaError, ValueError)):
                    SkillGraphPatch.from_dict(payload, task="set_table")

    def test_nested_arguments_cannot_reach_a_skill_node(self):
        with self.assertRaisesRegex(TypeError, "must be a scalar"):
            SkillNode("n", "mshab.set_table.pick.all", {"waypoints": [1, 2, 3]})

    def test_skill_nodes_are_hashable(self):
        node = _pick("n", "all", "g", object="013_apple")

        self.assertEqual(len({node, _pick("n", "all", "g", object="013_apple")}), 1)
