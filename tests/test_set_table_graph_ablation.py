from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mshab.experiments.granularity.vlm.run import DEFAULT_GRAPH, _read_json
from mshab.experiments.granularity.vlm.set_table_graph_ablation import (
    CONDITIONS,
    DEFAULT_MANIFEST,
    AblationSpec,
    ExplicitPathIndex,
    SetTableGraphAblation,
    compare_path_decisions,
)


ASSET_ROOT = Path(__file__).resolve().parents[1].parent / "mshab-assets"


class TestSetTableGraphAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = AblationSpec.from_yaml(DEFAULT_MANIFEST)
        cls.document = _read_json(DEFAULT_GRAPH)
        cls.index = ExplicitPathIndex(cls.document)

    def experiment(
        self,
        output_dir: Path,
        *,
        execution_split: str | None = None,
        planning_source_dir: Path | None = None,
        execution_seeds: tuple[int, ...] | None = None,
    ) -> SetTableGraphAblation:
        return SetTableGraphAblation(
            spec=self.spec,
            output_dir=output_dir,
            graph_path=DEFAULT_GRAPH,
            config_path=Path("unused-config.yaml"),
            asset_root=ASSET_ROOT,
            execution_split=execution_split,
            planning_source_dir=planning_source_dir,
            execution_seeds=execution_seeds,
        )

    def write_oracle_planning_flows(
        self,
        experiment: SetTableGraphAblation,
    ) -> None:
        experiment.initialize()
        experiment.write_oracles(self.index)
        for condition in CONDITIONS:
            for case in self.spec.cases:
                decision, _ = experiment._oracle_pair(case, self.index)
                decision["condition"] = condition
                flow = self.index.validate_and_compile(
                    decision,
                    case=case,
                    condition=condition,
                )
                path = (
                    experiment.planning_dir
                    / condition
                    / case.id
                    / "path_decision.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(decision), encoding="utf-8")

    def test_manifest_is_ten_paired_cases(self) -> None:
        self.assertEqual(len(self.spec.cases), 10)
        self.assertEqual(
            [case.plan_index for case in self.spec.cases],
            list(range(10)),
        )
        self.assertEqual(
            sum(case.case_type == "nominal" for case in self.spec.cases),
            4,
        )
        self.assertEqual(
            sum(case.case_type == "fallback" for case in self.spec.cases),
            6,
        )

    def test_default_planning_input_is_local_output(self) -> None:
        experiment = self.experiment(Path("example-output"))
        self.assertEqual(experiment.execution_split, "val")
        self.assertEqual(experiment.planning_input_dir, experiment.planning_dir)

    def test_reused_planning_requires_a_separate_output(self) -> None:
        output = Path("same-output")
        with self.assertRaisesRegex(ValueError, "must differ"):
            self.experiment(output, planning_source_dir=output)

    def test_only_edges_differ_between_prompt_conditions(self) -> None:
        flat = self.index.library_view("flat_library")
        full = self.index.library_view("full_graph")
        self.assertEqual(flat["skill_nodes"], full["skill_nodes"])
        self.assertEqual(flat["contracts"], full["contracts"])
        self.assertEqual(flat["subgoals"], full["subgoals"])
        self.assertEqual(flat["layer2_edges"], [])
        self.assertGreater(len(full["layer2_edges"]), 0)
        serialized = json.dumps(full)
        self.assertNotIn("execution_order", serialized)
        self.assertNotIn("semantic_strategies", serialized)
        self.assertNotIn("primary_achiever", serialized)

    def test_all_oracle_routes_are_valid_and_score_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self.experiment(Path(temporary))
            for case in self.spec.cases:
                decision, flow = experiment._oracle_pair(case, self.index)
                self.assertEqual(len(flow["nominal_order"]), 16)
                prediction = copy.deepcopy(decision)
                prediction["condition"] = "full_graph"
                compiled = self.index.validate_and_compile(
                    prediction,
                    case=case,
                    condition="full_graph",
                )
                self.assertEqual(len(compiled["nominal_order"]), 16)
                metrics = compare_path_decisions(prediction, decision, self.index)
                self.assertEqual(metrics["path_lcs_f1"], 1.0)
                self.assertEqual(metrics["grounding_accuracy"], 1.0)

    def test_semantic_object_aliases_score_as_grounded(self) -> None:
        case = self.spec.cases[0]
        oracle, _ = self.experiment(Path("unused"))._oracle_pair(
            case,
            self.index,
        )
        prediction = copy.deepcopy(oracle)
        for segment in prediction["segments"]:
            obj = segment["bindings"].get("object")
            if obj == "024_bowl":
                segment["bindings"]["object"] = "bowl"
            elif obj == "013_apple":
                segment["bindings"]["object"] = "apple"
        metrics = compare_path_decisions(prediction, oracle, self.index)
        self.assertEqual(metrics["grounding_accuracy"], 1.0)

    def test_validator_rejects_cross_subgoal_reordering(self) -> None:
        case = self.spec.cases[1]
        decision, _ = self.experiment(Path("unused"))._oracle_pair(
            case,
            self.index,
        )
        decision["condition"] = "full_graph"
        order = list(decision["segment_order"])
        order[1], order[2] = order[2], order[1]
        decision["segment_order"] = order
        with self.assertRaisesRegex(ValueError, "SetTable task order"):
            self.index.validate_and_compile(
                decision,
                case=case,
                condition="full_graph",
            )

    def test_flat_and_full_oracle_outputs_prepare_thirty_jobs(self) -> None:
        if not ASSET_ROOT.exists():
            self.skipTest("MS-HAB assets are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self.experiment(Path(temporary))
            self.write_oracle_planning_flows(experiment)
            jobs = experiment.prepare_execution()
            self.assertEqual(len(jobs), 30)
            self.assertTrue(all(job.groundable for job in jobs))
            self.assertEqual(
                {job.condition for job in jobs},
                {"graph_oracle", "flat_library", "full_graph"},
            )
            report = (experiment.tables_dir / "report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Table 1 — Planning ablation", report)
            self.assertIn("Table 2 — Nominal vs fallback planning", report)
            self.assertIn("Table 3 — Controller calibration", report)
            self.assertIn("Table 4 — Paired MS-HAB execution", report)

    def test_train_execution_can_reuse_val_planning_artifacts(self) -> None:
        if not ASSET_ROOT.exists():
            self.skipTest("MS-HAB assets are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            val_output = root / "val-planning"
            train_output = root / "train-execution"
            val_experiment = self.experiment(val_output)
            self.write_oracle_planning_flows(val_experiment)

            train_experiment = self.experiment(
                train_output,
                execution_split="train",
                planning_source_dir=val_output,
            )
            jobs = train_experiment.prepare_execution()
            self.assertEqual(len(jobs), 30)
            self.assertTrue(all(job.groundable for job in jobs))

            manifest = json.loads(
                (train_output / "experiment_manifest.json").read_text()
            )
            self.assertEqual(manifest["planning_split"], "val")
            self.assertEqual(manifest["execution_split"], "train")
            self.assertEqual(
                Path(manifest["planning_source"]),
                val_output.resolve() / "planning",
            )
            train_plan = json.loads(
                (
                    train_output
                    / "execution_plans"
                    / "graph_oracle"
                    / "N01.json"
                ).read_text()
            )
            self.assertEqual(train_plan["dataset"], "ReplicaCADSetTableTrain")
            self.assertEqual(
                train_plan["selection"]["recommended_policy_type"],
                "rl_per_obj",
            )

    def test_multi_seed_manifest_and_execution_rows(self) -> None:
        if not ASSET_ROOT.exists():
            self.skipTest("MS-HAB assets are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self.experiment(
                Path(temporary),
                execution_seeds=(0, 1, 2, 3, 4),
            )
            self.write_oracle_planning_flows(experiment)
            experiment.prepare_execution()
            manifest = json.loads(
                (experiment.output_dir / "experiment_manifest.json").read_text()
            )
            self.assertEqual(manifest["execution_seeds"], [0, 1, 2, 3, 4])
            self.assertEqual(len(experiment._execution_rows()), 150)
            report = (experiment.tables_dir / "report.md").read_text()
            self.assertIn("Rollouts", report)
            self.assertIn("Conditional Simulator Success", report)

    def test_identical_grounded_plans_share_the_seeded_rollout(self) -> None:
        if not ASSET_ROOT.exists():
            self.skipTest("MS-HAB assets are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self.experiment(Path(temporary))
            self.write_oracle_planning_flows(experiment)
            experiment.prepare_execution()
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True)
            )

            def successful_run(*args, **kwargs):
                kwargs["stdout"].write(
                    "results {'success_once': tensor(1.), "
                    "'success_at_end': tensor(1.), 'len': tensor(10.)}\n"
                )
                return SimpleNamespace(returncode=0)

            with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch(
                "mshab.experiments.granularity.vlm."
                "set_table_graph_ablation.subprocess.run",
                side_effect=successful_run,
            ) as run:
                experiment.execute(
                    max_jobs=None,
                    condition=None,
                    case_id="N01",
                    case_type=None,
                    record_video=False,
                    retry_failed=False,
                )
            self.assertEqual(run.call_count, 1)
            statuses = [
                json.loads((experiment.status_dir / "{}.json".format(job)).read_text())
                for job in (
                    "graph_oracle.N01",
                    "flat_library.N01",
                    "full_graph.N01",
                )
            ]
            self.assertNotIn("reused_from", statuses[0])
            self.assertIn("reused_from", statuses[1])
            self.assertIn("reused_from", statuses[2])

    def test_existing_output_rejects_an_execution_split_change(self) -> None:
        if not ASSET_ROOT.exists():
            self.skipTest("MS-HAB assets are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.experiment(output).initialize()
            changed = self.experiment(output, execution_split="train")
            with self.assertRaisesRegex(ValueError, "existing output uses"):
                changed.initialize()

    def test_api_failure_counts_as_a_terminal_zero_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self.experiment(Path(temporary))
            experiment.initialize()
            for condition in CONDITIONS:
                for case in self.spec.cases:
                    case_dir = experiment.planning_dir / condition / case.id
                    case_dir.mkdir(parents=True, exist_ok=True)
                    status = {
                        "status": "complete",
                        "graph_valid": True,
                    }
                    metrics = {
                        "path_lcs_f1": 1.0,
                        "grounding_accuracy": 1.0,
                        "predicted_length": 16,
                        "oracle_length": 16,
                    }
                    (case_dir / "status.json").write_text(
                        json.dumps(status), encoding="utf-8"
                    )
                    (case_dir / "path_metrics.json").write_text(
                        json.dumps(metrics), encoding="utf-8"
                    )
            failed = experiment.planning_dir / "full_graph" / "F04"
            (failed / "status.json").write_text(
                json.dumps({"status": "api_failed"}), encoding="utf-8"
            )
            (failed / "path_metrics.json").unlink()

            experiment.summarize()
            with (experiment.tables_dir / "planning_overall.csv").open(
                newline="", encoding="utf-8"
            ) as table:
                rows = list(csv.DictReader(table))
            full_graph = next(
                row
                for row in rows
                if row["method"] == "DeepSeek + Full Skill Graph"
            )
            self.assertEqual(float(full_graph["valid_path_rate"]), 0.9)
            self.assertEqual(float(full_graph["path_lcs_f1"]), 0.9)
            self.assertEqual(float(full_graph["grounding_accuracy"]), 0.9)


if __name__ == "__main__":
    unittest.main()
