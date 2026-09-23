"""End-to-end Layer-1/2 test: manual graph -> one skill per decision.

:class:`~mshab.skills.plan.SkillPlanner` is a deterministic reference decision model,
not a VLM.  It specifies the interface of the future trained decision model:
consume the candidate graph plus completion/failure history, emit one next
SkillNode.  Repeated decisions form the 16-step SetTable rollout, and the same
interface is what a failed candidate hands off through.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

from mshab.skills import LibraryCatalog, SkillPlanner, build_set_table_graph
from scripts.build_set_table_graph_plan import build_plan_data


OFFICIAL_TASK_PLAN = (
    Path(__file__).resolve().parents[2]
    / "mshab-assets"
    / "data"
    / "scene_datasets"
    / "replica_cad_dataset"
    / "rearrange"
    / "task_plans"
    / "set_table"
    / "sequential"
    / "train"
    / "all.json"
)
CATALOG = (
    Path(__file__).resolve().parents[1]
    / "mshab"
    / "skills"
    / "catalogs"
    / "set_table.json"
)


def _rollout(planner, failed=()):
    """Drive ``decide`` one node at a time, exactly as an executor would."""

    completed = []
    while True:
        node = planner.decide(completed, failed)
        if node is None:
            return completed
        completed.append(node.id)


class SetTableGraphDecisionTests(TestCase):
    EXPECTED_NODE_SEQUENCE = (
        "navigate_to_bowl_source",
        "open_bowl_source",
        "navigate_to_bowl",
        "pick_bowl_specialized",
        "navigate_bowl_to_destination",
        "place_bowl_specialized",
        "navigate_back_to_bowl_source",
        "close_bowl_source",
        "navigate_to_apple_source",
        "open_apple_source",
        "navigate_to_apple",
        "pick_apple_specialized",
        "navigate_apple_to_destination",
        "place_apple_specialized",
        "navigate_back_to_apple_source",
        "close_apple_source",
    )
    EXPECTED_SKILL_TYPES = (
        "navigate",
        "open",
        "navigate",
        "pick",
        "navigate",
        "place",
        "navigate",
        "close",
    ) * 2

    def setUp(self):
        self.subgoals, self.graph = build_set_table_graph()
        self.planner = SkillPlanner(self.subgoals, self.graph)

    def test_manual_graph_then_planner_decides_one_skill_at_a_time(self):
        self.assertEqual(len(self.subgoals.subgoals), 8)
        self.assertEqual(len(self.graph.subgraphs), 8)
        self.assertEqual(len(self.graph.nodes), 20)

        decisions = _rollout(self.planner)

        self.assertEqual(len(decisions), 16)
        self.assertEqual(tuple(decisions), self.EXPECTED_NODE_SEQUENCE)
        self.assertEqual(
            tuple(
                self.graph.nodes[node_id].contract_id.split(".")[2]
                for node_id in decisions
            ),
            self.EXPECTED_SKILL_TYPES,
        )
        for subgoal_id in self.subgoals.subgoals:
            chosen = [
                node_id
                for node_id in decisions
                if subgoal_id in self.graph.nodes[node_id].achieves
            ]
            self.assertEqual(
                len(chosen), 1, "{} must select exactly one achiever".format(subgoal_id)
            )
        self.assertNotIn("pick_bowl_generic", decisions)
        self.assertNotIn("place_apple_generic", decisions)

        # The raw graph order is a partial order over all candidates, not a plan.
        self.assertEqual(len(self.graph.execution_order()), 20)
        self.assertNotEqual(tuple(self.graph.execution_order()), tuple(decisions))

    def test_a_failed_candidate_is_replaced_mid_rollout(self):
        completed, failed = [], []
        while True:
            node = self.planner.decide(completed, failed)
            if node is None:
                break
            if node.id == "pick_bowl_specialized" and not failed:
                failed.append(node.id)  # the policy reports the failure and retries
                continue
            completed.append(node.id)

        self.assertEqual(len(completed), 16)
        self.assertIn("pick_bowl_generic", completed)
        self.assertNotIn("pick_bowl_specialized", completed)
        self.assertEqual(completed[-1], "close_apple_source")

    def test_decisions_are_json_serialisable_for_a_model_interface(self):
        payload = [
            {
                "node_id": node_id,
                "contract_id": self.graph.nodes[node_id].contract_id,
                "arguments": dict(self.graph.nodes[node_id].arguments),
            }
            for node_id in _rollout(self.planner)
        ]

        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_repeated_skill_decisions_match_downloaded_official_task_plan(self):
        if not OFFICIAL_TASK_PLAN.exists():
            self.skipTest("official downloaded SetTable task plan is unavailable")
        decisions = _rollout(self.planner)
        official = json.loads(OFFICIAL_TASK_PLAN.read_text())["plans"][0]["subtasks"]

        self.assertEqual(
            [self.graph.nodes[node_id].contract_id.split(".")[2] for node_id in decisions],
            [subtask["type"] for subtask in official],
        )
        self.assertEqual(official[1]["articulation_type"], "kitchen_counter")
        self.assertTrue(official[3]["obj_id"].startswith("024_bowl-"))
        self.assertEqual(official[9]["articulation_type"], "fridge")
        self.assertTrue(official[11]["obj_id"].startswith("013_apple-"))

    def test_recovery_rollout_still_matches_the_official_skill_type_order(self):
        if not OFFICIAL_TASK_PLAN.exists():
            self.skipTest("official downloaded SetTable task plan is unavailable")
        decisions = _rollout(
            self.planner,
            failed=("pick_bowl_specialized", "place_apple_specialized"),
        )
        official = json.loads(OFFICIAL_TASK_PLAN.read_text())["plans"][0]["subtasks"]

        # Recovering through a generic candidate must not change the task shape.
        self.assertEqual(
            [self.graph.nodes[node_id].contract_id.split(".")[2] for node_id in decisions],
            [subtask["type"] for subtask in official],
        )

    def test_catalog_plan_is_grounded_into_executable_mshab_plan_data(self):
        catalog = LibraryCatalog.from_dict(json.loads(CATALOG.read_text()))
        subtasks = []
        for index, contract_type in enumerate(self.EXPECTED_SKILL_TYPES):
            item = {"type": contract_type}
            bowl = index < 8
            if contract_type in ("pick", "place"):
                item["obj_id"] = "024_bowl-3" if bowl else "013_apple-0"
            if contract_type in ("open", "close"):
                item["articulation_type"] = (
                    "kitchen_counter" if bowl else "fridge"
                )
            subtasks.append(item)
        source = {
            "dataset": "replica_cad",
            "plans": [
                {
                    "subtasks": subtasks,
                    "build_config_name": "build.json",
                    "init_config_name": "init.json",
                }
            ],
        }

        nominal = build_plan_data(
            catalog, source, execution_plan="nominal", source_plan_index=0
        )
        recovery = build_plan_data(
            catalog,
            source,
            execution_plan="recovery_all_primaries_failed",
            source_plan_index=0,
        )

        self.assertEqual(
            nominal["selection"]["recommended_policy_type"], "rl_per_obj"
        )
        self.assertEqual(
            recovery["selection"]["recommended_policy_type"], "rl_all_obj"
        )
        self.assertEqual(
            nominal["selection"]["skill_decisions"][3]["node_id"],
            "pick_bowl_specialized",
        )
        self.assertEqual(
            recovery["selection"]["skill_decisions"][3]["node_id"],
            "pick_bowl_generic",
        )
