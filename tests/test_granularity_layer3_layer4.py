import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import (
    build_connected_layers,
    build_layer3,
    build_layer4,
    connect_layers,
    connected_layers_document,
)
from mshab.experiments.granularity.render import (
    DEFAULT_ARTIFACT_DIR,
    lower_layer_svg,
)
from mshab.skills.model import ArtifactStatus, ContractType


class GranularityLayer3Layer4Tests(TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.checkpoint_root = Path(self.temporary_directory.name)

    def test_layers_are_built_independently(self):
        layer3 = build_layer3()
        layer4 = build_layer4(self.checkpoint_root)

        self.assertEqual(len(layer3.contracts), 5)
        self.assertEqual(len(layer4.policies), 53)
        self.assertFalse(hasattr(layer4, "router"))
        self.assertFalse(hasattr(layer4, "connections"))

    def test_connections_define_only_policy_executes_contract(self):
        stack = connect_layers(
            build_layer3(),
            build_layer4(self.checkpoint_root),
        )
        self.assertEqual(len(stack.connections), 53)
        self.assertEqual(
            {connection.as_dict()["relation"] for connection in stack.connections},
            {"EXECUTES"},
        )
        for connection in stack.connections:
            spec = stack.layer4.spec(connection.policy_id)
            self.assertEqual(
                stack.layer3.contract(spec.contract_type).id,
                connection.contract_id,
            )

    def test_contract_and_policy_counts(self):
        stack = build_connected_layers(self.checkpoint_root)
        counts = {
            contract_type.value: sum(
                connection.contract_id
                == stack.layer3.contract(contract_type).id
                for connection in stack.connections
            )
            for contract_type in ContractType
        }
        self.assertEqual(
            counts,
            {"navigate": 3, "pick": 23, "place": 23, "open": 2, "close": 2},
        )

    def test_generic_contracts_ground_the_symbolic_interface(self):
        layer3 = build_layer3()
        pick = layer3.contract("pick").bind({"object": "013_apple"})
        self.assertEqual(
            pick.preconditions,
            ("reachable(013_apple)", "gripper_empty()"),
        )
        self.assertEqual(pick.effects, ("holding(013_apple)",))
        self.assertEqual(pick.deletes, ("gripper_empty()",))

        place = layer3.contract("place").bind(
            {"object": "013_apple", "destination": "dining_table"}
        )
        self.assertEqual(
            place.preconditions,
            ("holding(013_apple)", "reachable(dining_table)"),
        )
        self.assertEqual(
            place.effects,
            ("at(013_apple,dining_table)", "gripper_empty()"),
        )
        self.assertEqual(place.deletes, ("holding(013_apple)",))

    def test_layer4_keeps_real_artifact_state_without_routing(self):
        layer4 = build_layer4(self.checkpoint_root)
        policy = layer4.policy("rl.set_table.navigate.all")
        self.assertEqual(policy.status, ArtifactStatus.MISSING)

    def test_document_has_no_fallback_or_policy_relations(self):
        document = connected_layers_document(
            build_connected_layers(self.checkpoint_root)
        )
        self.assertEqual(document["summary"]["contracts"], 5)
        self.assertEqual(document["summary"]["policies"], 53)
        self.assertEqual(document["summary"]["executes_connections"], 53)
        self.assertEqual(len(document["connections"]), 53)
        serialized = json.dumps(document, sort_keys=True)
        self.assertNotIn(self.temporary_directory.name, serialized)
        self.assertNotIn('"status"', serialized)
        self.assertNotIn("fallback", serialized.lower())
        self.assertNotIn("alternative", serialized.lower())

        svg = lower_layer_svg(document)
        ET.fromstring(svg)
        expected_json = json.loads(
            (DEFAULT_ARTIFACT_DIR / "layer3_layer4.json").read_text()
        )
        expected_svg = (
            DEFAULT_ARTIFACT_DIR / "contract_policy_layers.svg"
        ).read_text()
        self.assertEqual(document, expected_json)
        self.assertEqual(svg, expected_svg)
