import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity.vlm.prompt import DECISION_SCHEMA_VERSION
from mshab.experiments.granularity.vlm.run import FourLayerGraphIndex
from mshab.experiments.granularity.vlm.set_table_experiment import (
    DEFAULT_MANIFEST,
    ExperimentSpec,
    SetTableExperiment,
    _parse_evaluator_log,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = (
    REPOSITORY_ROOT
    / "mshab"
    / "experiments"
    / "granularity"
    / "artifacts"
    / "coarse_four_layers.json"
)


def _source_subtasks():
    types = [
        "navigate",
        "open",
        "navigate",
        "pick",
        "navigate",
        "place",
        "navigate",
        "close",
    ] * 2
    subtasks = []
    for index, contract_type in enumerate(types):
        bowl = index < 8
        item = {"type": contract_type, "uid": "source-{:02d}".format(index)}
        if contract_type in ("pick", "place"):
            item["obj_id"] = "024_bowl-4" if bowl else "013_apple-0"
        if contract_type == "pick":
            item["articulation_config"] = {
                "articulation_type": "kitchen_counter" if bowl else "fridge"
            }
        if contract_type == "place":
            item["goal_pos"] = [float(index), 0.0, 1.0]
        if contract_type in ("open", "close"):
            item["articulation_type"] = (
                "kitchen_counter" if bowl else "fridge"
            )
            item["articulation_id"] = (
                "kitchen_counter-0" if bowl else "fridge-0"
            )
        subtasks.append(item)
    return subtasks


def _source():
    plans = []
    for index in range(10):
        plans.append(
            {
                "subtasks": copy.deepcopy(_source_subtasks()),
                "build_config_name": "build-{}.json".format(index),
                "init_config_name": "init-{}.json".format(index),
            }
        )
    return {"dataset": "ReplicaCADSetTableVal", "plans": plans}


def _decision(instruction):
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "instruction": instruction,
        "task_family": "set_table",
        "strategy_occurrences": [
            {
                "id": "bowl_retrieve",
                "strategy_id": "retrieve_from_drawer",
                "bindings": {"object": "024_bowl"},
            },
            {
                "id": "bowl_deliver",
                "strategy_id": "deliver_to_table",
                "bindings": {"object": "024_bowl"},
            },
            {"id": "bowl_restore", "strategy_id": "restore_drawer", "bindings": {}},
            {
                "id": "apple_retrieve",
                "strategy_id": "retrieve_from_fridge",
                "bindings": {"object": "013_apple"},
            },
            {
                "id": "apple_deliver",
                "strategy_id": "deliver_to_table",
                "bindings": {"object": "013_apple"},
            },
            {"id": "apple_restore", "strategy_id": "restore_fridge", "bindings": {}},
        ],
        "occurrence_order": [
            "bowl_retrieve",
            "bowl_deliver",
            "bowl_restore",
            "apple_retrieve",
            "apple_deliver",
            "apple_restore",
        ],
        "decision_summary": (
            "Complete and restore the bowl stage, then the apple stage."
        ),
    }


class SetTableExperimentTests(TestCase):
    def test_manifest_defines_full_ten_by_ten_experiment(self):
        spec = ExperimentSpec.from_yaml(DEFAULT_MANIFEST)
        self.assertEqual(len(spec.instructions), 10)
        self.assertEqual(spec.plan_indices, tuple(range(10)))
        self.assertEqual(spec.canonical_instruction_id, "I01")
        for case in spec.instructions:
            lower = case.text.lower()
            self.assertIn("bowl", lower)
            self.assertIn("apple", lower)
            self.assertIn("drawer", lower)
            self.assertIn("fridge", lower)

    def test_prepare_emits_170_resumable_execution_jobs_and_two_tables(self):
        spec = ExperimentSpec.from_yaml(DEFAULT_MANIFEST)
        graph = json.loads(GRAPH_PATH.read_text())
        index = FourLayerGraphIndex(graph)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "assets"
            gt_path = (
                asset_root
                / "data"
                / "scene_datasets"
                / "replica_cad_dataset"
                / "rearrange"
                / "task_plans"
                / "set_table"
                / "sequential"
                / "val"
                / "all.json"
            )
            gt_path.parent.mkdir(parents=True)
            gt_path.write_text(json.dumps(_source()))
            experiment = SetTableExperiment(
                spec=spec,
                output_dir=root / "output",
                graph_path=GRAPH_PATH,
                config_path=root / "unused-config.yaml",
                asset_root=asset_root,
            )
            experiment.initialize()
            for case in spec.instructions:
                case_dir = experiment.planning_dir / case.id
                case_dir.mkdir(parents=True)
                flow = index.compile_decision(
                    _decision(case.text),
                    case.text,
                    "set_table",
                )
                (case_dir / "predicted_flow.json").write_text(json.dumps(flow))
                (case_dir / "status.json").write_text(
                    json.dumps({"status": "complete", "graph_valid": True})
                )
            jobs = experiment.prepare_execution()
            self.assertEqual(len(jobs), 170)
            self.assertEqual(sum(job.condition == "gt_nominal" for job in jobs), 10)
            self.assertEqual(sum(job.condition == "vlm_nominal" for job in jobs), 100)
            self.assertEqual(
                sum(job.condition.startswith("fallback.") for job in jobs),
                60,
            )
            self.assertTrue(all(job.groundable for job in jobs))
            self.assertTrue(
                (experiment.tables_dir / "planning_table.csv").is_file()
            )
            self.assertTrue(
                (experiment.tables_dir / "execution_table.csv").is_file()
            )

    def test_evaluator_log_parser_extracts_success_and_length(self):
        parsed = _parse_evaluator_log(
            "results {'success_once': tensor(1., device='cuda:0'), "
            "'success_at_end': tensor(1., device='cuda:0'), "
            "'len': tensor(1262., device='cuda:0')}"
        )
        self.assertEqual(parsed["simulator_success"], 1.0)
        self.assertEqual(parsed["success_at_end"], 1.0)
        self.assertEqual(parsed["episode_steps"], 1262.0)
