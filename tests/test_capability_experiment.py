"""Regression tests for the capability study's scoring.

The experiment this replaces decided plan correctness with one hard-coded
segment order, which marked valid alternative routes wrong.  These tests pin
the behaviour that must not regress: contract simulation accepts any order
whose preconditions chain and whose final state satisfies the goal, and
rejects plans that skip a navigation or leave a goal predicate unmet.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import TestCase, skipUnless

from mshab.experiments.granularity.vlm.symbolic import (
    ContractSimulator,
    OUTCOME_ALTERNATIVE,
    OUTCOME_EXACT,
    TaskSemantics,
    lcs_f1,
    plan_from_decision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH = (
    REPO_ROOT
    / "mshab"
    / "experiments"
    / "granularity"
    / "artifacts"
    / "coarse_four_layers.json"
)
SPEC = (
    REPO_ROOT
    / "mshab"
    / "experiments"
    / "granularity"
    / "vlm"
    / "set_table_graph_ablation.yaml"
)


def _experiment():
    from mshab.experiments.granularity.vlm.capability import CapabilityExperiment

    return CapabilityExperiment(
        SPEC,
        REPO_ROOT / "mshab" / "experiments" / "granularity" / "vlm" / "outputs" / "_test",
        REPO_ROOT / "mshab" / "experiments" / "granularity" / "vlm" / "config.yaml",
    )


class SymbolicSimulationTests(TestCase):
    def setUp(self):
        document = json.loads(GRAPH.read_text(encoding="utf-8"))
        from mshab.experiments.granularity.vlm.capability import (
            CapabilityIndex,
            SET_TABLE_SEMANTICS,
        )

        view = CapabilityIndex(document).library_view("full_graph")
        self.simulator = ContractSimulator(
            {item["contract_type"]: item for item in view["contracts"]},
            {item["id"]: item for item in view["skill_nodes"]},
        )
        self.semantics = SET_TABLE_SEMANTICS

    def test_reference_plan_satisfies_the_goal(self):
        from mshab.experiments.granularity.vlm.capability import GT_SEGMENTS

        plan = [
            (node, bindings)
            for _, bindings, path in GT_SEGMENTS
            for node in path
        ]

        result = self.simulator.simulate(plan, self.semantics)

        self.assertTrue(result.valid)
        self.assertEqual(result.missing_goal, ())

    def test_restore_before_deliver_is_a_valid_alternative(self):
        # close() has no gripper_empty() precondition, so closing the drawer
        # while still holding the bowl is legal and saves a trip back.
        from mshab.experiments.granularity.vlm.capability import GT_SEGMENTS

        order = [GT_SEGMENTS[0], GT_SEGMENTS[2], GT_SEGMENTS[1], GT_SEGMENTS[3], GT_SEGMENTS[5], GT_SEGMENTS[4]]
        plan = [(node, bindings) for _, bindings, path in order for node in path]

        result = self.simulator.simulate(plan, self.semantics)

        self.assertTrue(result.valid, result.as_dict())

    def test_skipping_a_navigation_violates_a_precondition(self):
        plan = [
            ("navigate_drawer_source", {}),
            ("open_drawer_source", {}),
            ("pick_drawer_object", {"object": "024_bowl"}),
        ]

        result = self.simulator.simulate(plan, self.semantics)

        self.assertFalse(result.precondition_ok)
        self.assertEqual(result.failed_node, "pick_drawer_object")
        self.assertIn("reachable(024_bowl)", result.missing)

    def test_navigating_elsewhere_retracts_the_previous_reachable(self):
        plan = [
            ("navigate_drawer_source", {}),
            ("open_drawer_source", {}),
            ("navigate_drawer_object", {"object": "024_bowl"}),
            ("pick_drawer_object", {"object": "024_bowl"}),
            ("navigate_table", {}),
            ("close_drawer", {}),
        ]

        result = self.simulator.simulate(plan, self.semantics)

        self.assertFalse(result.precondition_ok)
        self.assertEqual(result.failed_node, "close_drawer")

    def test_unmet_goal_is_distinguished_from_an_illegal_step(self):
        from mshab.experiments.granularity.vlm.capability import GT_SEGMENTS

        plan = [
            (node, bindings)
            for _, bindings, path in GT_SEGMENTS[:5]
            for node in path
        ]

        result = self.simulator.simulate(plan, self.semantics)

        self.assertTrue(result.precondition_ok)
        self.assertFalse(result.reachable_goal)
        self.assertIn("closed(fridge)", result.missing_goal)

    def test_lcs_f1_is_one_only_for_an_identical_path(self):
        self.assertEqual(lcs_f1(["a", "b"], ["a", "b"]), 1.0)
        self.assertLess(lcs_f1(["a", "b"], ["b", "a"]), 1.0)


@skipUnless(GRAPH.is_file(), "coarse graph artifact is unavailable")
class ReferencePlanTests(TestCase):
    def test_every_case_reference_scores_as_an_exact_match(self):
        from mshab.experiments.granularity.vlm.capability import gt_decision

        experiment = _experiment()

        for case in experiment.cases:
            with self.subTest(case.id):
                record = experiment._score(case, gt_decision(case), 0)
                self.assertEqual(record.outcome, OUTCOME_EXACT)
                self.assertTrue(record.graph_valid)
                self.assertTrue(record.goal_reached)

    def test_fallback_reference_replaces_only_the_failing_segment(self):
        # place_on_table is shared by the bowl and apple deliveries, so the
        # failed node may legitimately survive in the other segment; only the
        # segment named by oracle_route_id must route around it.
        from mshab.experiments.granularity.vlm.capability import (
            GT_SEGMENTS,
            gt_decision,
        )

        experiment = _experiment()
        nominal = {item[0]: list(item[2]) for item in GT_SEGMENTS}

        for case in experiment.cases:
            if case.failed_skill_node_id is None:
                continue
            with self.subTest(case.id):
                replaced = case.oracle_route_id.split(".", 1)[-1]
                segments = {
                    segment["id"]: segment["skill_node_path"]
                    for segment in gt_decision(case)["segments"]
                }
                self.assertNotIn(
                    case.failed_skill_node_id, segments[replaced]
                )
                for segment_id, path in segments.items():
                    if segment_id != replaced:
                        self.assertEqual(path, nominal[segment_id])

    def test_reordered_plan_is_scored_as_a_valid_alternative(self):
        from mshab.experiments.granularity.vlm.capability import gt_decision

        experiment = _experiment()
        case = next(item for item in experiment.cases if item.case_type == "nominal")
        decision = copy.deepcopy(gt_decision(case))
        decision["segment_order"] = [
            "retrieve_bowl",
            "restore_drawer",
            "deliver_bowl",
            "retrieve_apple",
            "restore_fridge",
            "deliver_apple",
        ]

        record = experiment._score(case, decision, 0)

        self.assertEqual(record.outcome, OUTCOME_ALTERNATIVE)
        self.assertTrue(record.correct)

    def test_unsubstituted_placeholder_is_rejected(self):
        from mshab.experiments.granularity.vlm.capability import gt_decision

        experiment = _experiment()
        case = experiment.cases[0]
        decision = copy.deepcopy(gt_decision(case))
        decision["segments"][0]["bindings"] = {"object": "{object}"}

        record = experiment._score(case, decision, 0)

        self.assertFalse(record.correct)
