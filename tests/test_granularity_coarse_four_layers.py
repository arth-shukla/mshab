import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity.coarse_four_layers import (
    build_coarse_four_layer_document,
    compose_documents,
)
from mshab.experiments.granularity.higher_layers import (
    build_coarse_higher_layers,
    coarse_higher_layers_document,
)
from mshab.experiments.granularity.higher_layers.render import (
    coarse_higher_layers_svg,
)
from mshab.experiments.granularity.lower_layers.connections import (
    build_connected_layers,
    connected_layers_document,
)
from mshab.experiments.granularity.render import DEFAULT_ARTIFACT_DIR


class CoarseFourLayerArtifactTests(TestCase):
    def test_composer_reuses_every_existing_layer_document(self):
        with TemporaryDirectory() as directory:
            stack = build_connected_layers(Path(directory))
            higher = build_coarse_higher_layers(stack.layer3)
            self.assertIs(higher.layer3, stack.layer3)

            higher_document = coarse_higher_layers_document(higher)
            lower_document = connected_layers_document(stack)
            combined = compose_documents(higher_document, lower_document)

        self.assertEqual(combined["layer1"], higher_document["layer1"])
        self.assertEqual(combined["layer2"], higher_document["layer2"])
        self.assertEqual(
            combined["layer2_to_layer3"],
            higher_document["layer2_to_layer3"],
        )
        self.assertEqual(
            combined["layer3"]["contracts"],
            lower_document["layer3_contracts"],
        )
        self.assertEqual(
            combined["layer4"]["policies"],
            lower_document["layer4_policies"],
        )
        self.assertEqual(
            combined["layer4_to_layer3"]["connections"],
            lower_document["connections"],
        )

    def test_four_layer_cardinalities_and_endpoints(self):
        with TemporaryDirectory() as directory:
            document = build_coarse_four_layer_document(Path(directory))

        self.assertEqual(
            document["summary"],
            {
                "subgoals": 3,
                "subgoal_dependencies": 0,
                "skill_subgraphs": 3,
                "semantic_strategies": 9,
                "skill_nodes": 44,
                "cross_subgraph_edges": 0,
                "contract_usage": {
                    "mshab.granularity.close.all": 4,
                    "mshab.granularity.navigate.all": 22,
                    "mshab.granularity.open.all": 4,
                    "mshab.granularity.pick.all": 6,
                    "mshab.granularity.place.all": 8,
                },
                "skill_edges": 45,
                "skill_node_contract_references": 44,
                "contracts": 5,
                "policies": 53,
                "executes_connections": 53,
                "policies_by_contract": {
                    "navigate": 3,
                    "pick": 23,
                    "place": 23,
                    "open": 2,
                    "close": 2,
                },
            },
        )

        skill_ids = {item["id"] for item in document["layer2"]["nodes"]}
        policy_ids = {item["id"] for item in document["layer4"]["policies"]}
        contract_ids = {
            item["id"] for item in document["layer3"]["contracts"]
        }
        references = document["layer2_to_layer3"]["references"]
        connections = document["layer4_to_layer3"]["connections"]

        self.assertEqual(
            Counter(item["source"] for item in references),
            Counter({item: 1 for item in skill_ids}),
        )
        self.assertEqual(
            Counter(item["source_policy_id"] for item in connections),
            Counter({item: 1 for item in policy_ids}),
        )
        self.assertTrue(
            {item["target"] for item in references}.issubset(contract_ids)
        )
        self.assertTrue(
            {
                item["target_contract_id"] for item in connections
            }.issubset(contract_ids)
        )
        self.assertEqual(
            {item["relation"] for item in connections},
            {"EXECUTES"},
        )

    def test_output_is_portable_and_checkpoint_root_independent(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            first_document = build_coarse_four_layer_document(Path(first))
            second_document = build_coarse_four_layer_document(Path(second))

        first_json = json.dumps(first_document, indent=2, sort_keys=True) + "\n"
        second_json = json.dumps(second_document, indent=2, sort_keys=True) + "\n"
        self.assertEqual(first_json, second_json)
        self.assertEqual(
            coarse_higher_layers_svg(first_document),
            coarse_higher_layers_svg(second_document),
        )
        for forbidden in (
            first,
            second,
            '"status"',
            '"checkpoint_path"',
            '"config_path"',
        ):
            self.assertNotIn(forbidden, first_json)

    def test_svg_contains_every_real_object_and_boundary_relation(self):
        with TemporaryDirectory() as directory:
            document = build_coarse_four_layer_document(Path(directory))
        svg = coarse_higher_layers_svg(document)
        root = ET.fromstring(svg)
        identifiers = [element.attrib.get("id", "") for element in root.iter()]

        self.assertEqual(
            sum(item.startswith("skill-node-") for item in identifiers),
            44,
        )
        self.assertEqual(
            sum(item.startswith("contract-node-") for item in identifiers),
            5,
        )
        self.assertEqual(
            sum(item.startswith("policy-node-") for item in identifiers),
            53,
        )
        self.assertEqual(
            sum(item.startswith("contract-reference-") for item in identifiers),
            44,
        )
        self.assertEqual(
            sum(item.startswith("policy-executes-") for item in identifiers),
            53,
        )

    def test_checked_in_artifacts_match_the_existing_oop_objects(self):
        with TemporaryDirectory() as directory:
            document = build_coarse_four_layer_document(Path(directory))
        expected_json = json.loads(
            (DEFAULT_ARTIFACT_DIR / "coarse_four_layers.json").read_text()
        )
        expected_svg = (
            DEFAULT_ARTIFACT_DIR / "coarse_four_layers.svg"
        ).read_text()
        self.assertEqual(document, expected_json)
        self.assertEqual(coarse_higher_layers_svg(document), expected_svg)
