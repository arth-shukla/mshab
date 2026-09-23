"""CPU-only integration tests for the downloaded SetTable checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import SkipTest, TestCase

from mshab.skills import (
    ArtifactStatus,
    SkillGrounder,
    ContractLibrary,
    ContractType,
    build_set_table_stack,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path(
    os.environ.get("MS_ASSET_DIR", str(REPOSITORY_ROOT.parent / "mshab-assets"))
).expanduser()
CHECKPOINT_ROOT = DEFAULT_ASSET_ROOT / "data" / "mshab_checkpoints"


class DownloadedCheckpointTests(TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT_ROOT.exists():
            raise SkipTest(
                "downloaded checkpoints are unavailable at {}".format(
                    CHECKPOINT_ROOT
                )
            )
        cls.library = ContractLibrary.from_checkpoint_root(CHECKPOINT_ROOT)

    def test_all_eleven_set_table_contracts_are_discovered_and_ready(self):
        contracts = self.library.find(task="set_table")

        self.assertEqual(len(contracts), 11)
        for contract in contracts:
            self.assertTrue(self.library.ready(contract.id), contract.id)
            for policy in self.library.policies_for(contract.id):
                self.assertEqual(policy.status, ArtifactStatus.READY, policy.id)
        self.assertTrue(json.dumps(self.library.to_dict()))

    def test_generic_checkpoints_are_bound_to_specialized_contracts(self):
        self.assertEqual(
            [
                contract.id
                for contract in self.library.contracts_for("rl.set_table.pick.all")
            ],
            [
                "mshab.set_table.pick.013_apple",
                "mshab.set_table.pick.024_bowl",
                "mshab.set_table.pick.all",
            ],
        )
        # The specialized checkpoint stays the default; the generic one is
        # bound after it.
        bound = [
            policy.id
            for policy in self.library.policies_for("mshab.set_table.pick.013_apple")
        ]
        self.assertEqual(bound[0], "rl.set_table.pick.013_apple")
        self.assertIn("rl.set_table.pick.all", bound[1:])

    def test_downloaded_apple_pick_grounds_to_expected_contract(self):
        specialized = self.library.find(
            task="set_table",
            contract_type=ContractType.PICK,
            target="013_apple",
        )[0]
        grounded = specialized.bind({})
        self.assertEqual(
            self.library.select_policy(specialized.id).id,
            "rl.set_table.pick.013_apple",
        )

        self.assertEqual(grounded.arguments, {"object": "013_apple"})
        self.assertEqual(
            grounded.preconditions,
            ("reachable(013_apple)", "gripper_empty()"),
        )
        self.assertEqual(
            grounded.effects,
            ("holding(013_apple)",),
        )

    def test_complete_manual_graph_grounds_all_twenty_nodes(self):
        stack = build_set_table_stack(CHECKPOINT_ROOT)
        grounded_skills = SkillGrounder(self.library).grounded_skills(stack.skill_graph)

        self.assertEqual(len(stack.subgoal_graph.subgoals), 8)
        self.assertEqual(len(stack.skill_graph.subgraphs), 8)
        self.assertEqual(len(stack.skill_graph.nodes), 20)
        self.assertEqual(len(grounded_skills), 20)
        self.assertEqual(
            grounded_skills["place_apple_specialized"].effects,
            ("at(013_apple,dining_table)", "gripper_empty()"),
        )
