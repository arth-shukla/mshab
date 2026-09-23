import json
import xml.etree.ElementTree as ET
from collections import Counter
from unittest import TestCase

from mshab.experiments.granularity.higher_layers import (
    DELIVER,
    RESTORE,
    RETRIEVE,
    SUBGOAL_IDS,
    build_coarse_higher_layers,
    coarse_higher_layers_document,
)
from mshab.experiments.granularity.higher_layers.render import (
    DEFAULT_ARTIFACT_DIR,
    coarse_higher_layers_svg,
)
from mshab.experiments.granularity.lower_layers.layer3 import build_layer3
from mshab.skills import SkillRelation
from mshab.skills.plan import SkillPlanner


class CoarseHigherLayerTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.higher = build_coarse_higher_layers()

    def test_layer_one_has_exactly_three_independent_subgoals(self):
        self.assertEqual(tuple(self.higher.subgoals.subgoals), SUBGOAL_IDS)
        self.assertEqual(self.higher.subgoals.dependencies, ())
        self.assertEqual(
            {
                subgoal_id: self.higher.subgoals.subgoals[subgoal_id].predicate
                for subgoal_id in SUBGOAL_IDS
            },
            {
                RETRIEVE: "holding({object})",
                DELIVER: "at({object},{destination})",
                RESTORE: "closed({source})",
            },
        )

    def test_each_subgoal_owns_exactly_one_subgraph(self):
        self.assertEqual(set(self.higher.skills.subgraphs), set(SUBGOAL_IDS))
        self.assertEqual(len(self.higher.skills.subgraphs), 3)
        self.assertEqual(self.higher.skills.cross_edges, ())
        for subgoal_id in SUBGOAL_IDS:
            subgraph = self.higher.skills.subgraph_for_subgoal(subgoal_id)
            self.assertEqual(subgraph.subgoal_id, subgoal_id)
            self.assertTrue(subgraph.achievers)

    def test_semantic_alternatives_are_contextually_distinct(self):
        self.assertEqual(
            [item.id for item in self.higher.strategies_for(RETRIEVE)],
            [
                "retrieve_surface",
                "retrieve_from_fridge",
                "retrieve_from_drawer",
            ],
        )
        self.assertEqual(
            [item.id for item in self.higher.strategies_for(DELIVER)],
            [
                "deliver_to_table",
                "deliver_to_counter",
                "deliver_to_fridge",
                "deliver_to_drawer",
            ],
        )
        self.assertEqual(
            [item.id for item in self.higher.strategies_for(RESTORE)],
            ["restore_fridge", "restore_drawer"],
        )
        for subgoal_id in SUBGOAL_IDS:
            primary_ids = {
                item.primary_achiever
                for item in self.higher.strategies_for(subgoal_id)
            }
            first = next(iter(primary_ids))
            alternative_ids = {
                node.id for node in self.higher.skills.alternatives(first)
            }
            self.assertEqual(alternative_ids, primary_ids - {first})

    def test_every_fallback_moves_to_another_recovery_skill(self):
        for subgoal_id in SUBGOAL_IDS:
            subgraph = self.higher.skills.subgraph_for_subgoal(subgoal_id)
            fallback_edges = {
                (edge.source, edge.target)
                for edge in subgraph.edges
                if edge.relation == SkillRelation.FALLBACK_TO
            }
            for strategy in self.higher.strategies_for(subgoal_id):
                self.assertNotEqual(
                    strategy.primary_achiever,
                    strategy.fallback_achiever,
                )
                self.assertIn(
                    (
                        strategy.primary_achiever,
                        strategy.fallback_achiever,
                    ),
                    fallback_edges,
                )

    def test_selected_strategy_projects_to_an_unambiguous_execution_view(self):
        for subgoal_id in SUBGOAL_IDS:
            for strategy in self.higher.strategies_for(subgoal_id):
                view = self.higher.execution_view(strategy.id)
                planner = SkillPlanner(view.subgoals, view.skills)

                self.assertEqual(
                    set(view.skills.nodes),
                    set(strategy.node_ids),
                )
                self.assertFalse(
                    any(
                        edge.relation == SkillRelation.ALTERNATIVE_TO
                        for edge in view.skills.edges
                    )
                )

                nominal = planner.plan()
                self.assertIn(strategy.primary_achiever, nominal.order)
                self.assertNotIn(strategy.fallback_achiever, nominal.order)

                recovery = planner.plan(failed=(strategy.primary_achiever,))
                self.assertNotIn(strategy.primary_achiever, recovery.order)
                self.assertIn(strategy.fallback_achiever, recovery.order)

    def test_every_node_has_exactly_one_semantic_strategy_owner(self):
        strategy_nodes = [
            node_id
            for subgoal_id in SUBGOAL_IDS
            for strategy in self.higher.strategies_for(subgoal_id)
            for node_id in strategy.node_ids
        ]
        self.assertEqual(len(strategy_nodes), len(set(strategy_nodes)))
        self.assertEqual(set(strategy_nodes), set(self.higher.skills.nodes))

    def test_skill_node_to_contract_is_many_to_one(self):
        layer3 = build_layer3()
        contract_ids = {contract.id for contract in layer3.contracts.values()}
        references = [
            node.contract_id for node in self.higher.skills.nodes.values()
        ]
        self.assertEqual(set(references), contract_ids)
        self.assertEqual(len(references), 44)
        self.assertEqual(
            {
                contract_id: references.count(contract_id)
                for contract_id in sorted(contract_ids)
            },
            {
                "mshab.granularity.close.all": 4,
                "mshab.granularity.navigate.all": 22,
                "mshab.granularity.open.all": 4,
                "mshab.granularity.pick.all": 6,
                "mshab.granularity.place.all": 8,
            },
        )
        self.assertTrue(
            all(
                node.contract_id
                for node in self.higher.skills.nodes.values()
            )
        )
        for node_id, node in self.higher.skills.nodes.items():
            self.assertEqual(
                self.higher.contract_for(node_id).id,
                node.contract_id,
            )

        document = coarse_higher_layers_document(self.higher)
        references = document["layer2_to_layer3"]["references"]
        self.assertEqual(len(references), len(self.higher.skills.nodes))
        self.assertEqual(
            Counter(item["source"] for item in references),
            Counter({node_id: 1 for node_id in self.higher.skills.nodes}),
        )
        self.assertEqual(
            {
                item["source"]: item["target"]
                for item in references
            },
            {
                node_id: node.contract_id
                for node_id, node in self.higher.skills.nodes.items()
            },
        )

    def test_higher_layers_never_reference_policies(self):
        document = coarse_higher_layers_document(self.higher)
        serialized = json.dumps(document, sort_keys=True)
        self.assertNotIn('"policy_id"', serialized)
        self.assertNotIn('"rl.', serialized)
        for task_family in ("prepare_groceries", "set_table", "tidy_house"):
            self.assertNotIn(task_family, serialized)
        self.assertEqual(
            document["semantics"]["skill_node_to_contract"],
            "many_to_one",
        )
        self.assertIn(
            "select one SemanticStrategy",
            document["semantics"]["execution_protocol"],
        )

    def test_portable_artifacts_match_the_oop_graph(self):
        document = coarse_higher_layers_document(self.higher)
        self.assertEqual(document["summary"]["subgoals"], 3)
        self.assertEqual(document["summary"]["subgoal_dependencies"], 0)
        self.assertEqual(document["summary"]["skill_subgraphs"], 3)
        self.assertEqual(document["summary"]["semantic_strategies"], 9)
        self.assertEqual(document["summary"]["skill_nodes"], 44)
        self.assertEqual(document["summary"]["cross_subgraph_edges"], 0)

        svg = coarse_higher_layers_svg(document)
        root = ET.fromstring(svg)
        rendered_skill_nodes = [
            element
            for element in root.iter()
            if element.attrib.get("id", "").startswith("skill-node-")
        ]
        semantic_names = [
            element.attrib["data-semantic-name"]
            for element in rendered_skill_nodes
        ]
        self.assertEqual(len(semantic_names), 44)
        self.assertEqual(len(set(semantic_names)), 44)
        self.assertIn("NavigateTable", semantic_names)
        self.assertIn("PlaceOnTable", semantic_names)
        self.assertIn("RecoverPlaceOnTable", semantic_names)
        for node_id in self.higher.skills.nodes:
            self.assertEqual(svg.count('id="skill-node-{}"'.format(node_id)), 1)
            self.assertEqual(
                svg.count('id="contract-reference-{}"'.format(node_id)),
                1,
            )
        for edge in self.higher.skills.edges:
            edge_id = 'id="skill-edge-{}-{}-{}"'.format(
                edge.relation.value,
                edge.source,
                edge.target,
            )
            self.assertEqual(svg.count(edge_id), 1)
        self.assertEqual(svg.count('id="contract-node-'), 5)
        self.assertNotIn("primary:", svg)
        self.assertNotIn("fallback:", svg)
        for removed_lane_label in (
            "Deliver to table",
            "Deliver into fridge",
            "Retrieve from fridge",
            "Restore fridge",
        ):
            self.assertNotIn(removed_lane_label, svg)
        expected_json = json.loads(
            (DEFAULT_ARTIFACT_DIR / "coarse_graph.json").read_text()
        )
        expected_svg = (
            DEFAULT_ARTIFACT_DIR / "coarse_higher_layers.svg"
        ).read_text()
        self.assertEqual(document, expected_json)
        self.assertEqual(svg, expected_svg)
