import copy
import json
import os
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from mshab.experiments.granularity.vlm.prompt import (
    DECISION_SCHEMA_VERSION,
    build_planner_messages,
    compact_graph_for_prompt,
    message_text,
)
from mshab.experiments.granularity.vlm.run import (
    DeepSeekConfig,
    FourLayerGraphIndex,
    _NoRedirectHandler,
    _entities_match,
    _parse_chat_completion,
    _validate_gt_task_family,
    compare_flows,
    normalize_ground_truth,
    render_flowchart,
    main,
)
from mshab.experiments.granularity.vlm.simulator import (
    FlowProgram,
    MSHabPlanGrounder,
    write_simulator_bundle,
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
INSTRUCTION = (
    "Set the table by moving the bowl from the kitchen counter drawer and "
    "the apple from the fridge to the dining table, then close both containers."
)


def _decision():
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "instruction": INSTRUCTION,
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
            {
                "id": "bowl_restore",
                "strategy_id": "restore_drawer",
                "bindings": {},
            },
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
            {
                "id": "apple_restore",
                "strategy_id": "restore_fridge",
                "bindings": {},
            },
        ],
        "occurrence_order": [
            "bowl_retrieve",
            "bowl_deliver",
            "bowl_restore",
            "apple_retrieve",
            "apple_deliver",
            "apple_restore",
        ],
        "decision_summary": "Move the bowl, restore its drawer, then move the apple and restore the fridge.",
    }


def _official_source():
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
        item = {
            "type": contract_type,
            "uid": "SECRET-GT-{:02d}".format(index),
        }
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
        if contract_type == "open":
            item["obj_id"] = "024_bowl-4" if bowl else "013_apple-0"
        subtasks.append(item)
    return {
        "dataset": "replica_cad",
        "plans": [
            {
                "subtasks": subtasks,
                "build_config_name": "build.json",
                "init_config_name": "init.json",
            }
        ],
    }


class GranularityVLMTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = json.loads(GRAPH_PATH.read_text())
        cls.index = FourLayerGraphIndex(cls.graph)

    def test_semantic_object_names_match_dataset_categories(self):
        self.assertTrue(_entities_match("bowl", "024_bowl"))
        self.assertTrue(_entities_match("024_bowl", "024_bowl-4"))
        self.assertTrue(_entities_match("apple", "013_apple-0"))
        self.assertFalse(_entities_match("bowl", "013_apple"))

    def test_prompt_keeps_four_layers_and_uses_no_gt_derived_context(self):
        context = {
            "task_family": "set_table",
            "objects": ["024_bowl", "013_apple"],
            "articulations": ["kitchen_counter", "fridge"],
            "source": "user_supplied",
        }
        compact = compact_graph_for_prompt(self.graph)
        self.assertEqual(
            set(compact),
            {
                "schema_version",
                "granularity",
                "semantics",
                "layer1",
                "layer2",
                "layer2_to_layer3",
                "layer3",
                "layer4",
                "layer4_to_layer3",
            },
        )
        self.assertNotIn("edges", compact["layer2"])
        self.assertNotIn("_derived", compact["layer2"])

        messages = build_planner_messages(
            instruction=INSTRUCTION,
            task_family="set_table",
            scene_context=context,
            graph=self.graph,
        )
        serialized = message_text(messages)
        self.assertNotIn("SECRET-GT", serialized)
        self.assertNotIn('"subtasks"', serialized)
        self.assertNotIn("goal_position", serialized)
        self.assertNotIn("build_config_name", serialized)
        self.assertNotIn("init_config_name", serialized)
        self.assertIn("retrieve_from_drawer", serialized)
        self.assertIn("rl.set_table.pick.024_bowl", serialized)

    def test_valid_strategy_decision_compiles_to_official_set_table_shape(self):
        flow = self.index.compile_decision(
            _decision(),
            INSTRUCTION,
            "set_table",
        )
        by_id = {item["id"]: item for item in flow["nodes"]}
        types = [by_id[item]["contract_type"] for item in flow["nominal_order"]]
        self.assertEqual(
            types,
            [
                "navigate",
                "open",
                "navigate",
                "pick",
                "navigate",
                "place",
                "navigate",
                "close",
            ]
            * 2,
        )
        self.assertEqual(len(flow["nominal_order"]), 16)
        self.assertEqual(len(flow["fallback_branches"]), 6)
        self.assertEqual(len(flow["alternative_choices"]), 6)
        self.assertTrue(flow["metadata"]["semantic_validation"]["valid"])
        self.assertEqual(
            {edge["relation"] for edge in flow["edges"]} - {"next", "rejoins"},
            {"on_failure"},
        )
        self.assertTrue(
            all(
                branch["source_layer2_relation"]["relation"] == "fallback_to"
                for branch in flow["fallback_branches"]
            )
        )
        self.assertEqual(
            {item["kind"] for item in flow["nodes"]},
            {"nominal", "recovery"},
        )
        self.assertTrue(
            all(
                item["policy_id"].startswith("rl.set_table.")
                for item in flow["nodes"]
            )
        )
        self.assertTrue(
            all(
                branch["fallback_route_order"]
                for branch in flow["fallback_branches"]
            )
        )
        self.assertEqual(
            {
                candidate["strategy_id"]
                for choice in flow["alternative_choices"]
                if choice["occurrence_id"] == "bowl_retrieve"
                for candidate in choice["candidates"]
            },
            {"retrieve_surface", "retrieve_from_fridge"},
        )
        self.assertTrue(
            all(
                step["relation"] == "alternative_to"
                for choice in flow["alternative_choices"]
                for candidate in choice["candidates"]
                for step in candidate["source_layer2_relation_path"]
            )
        )

    def test_flow_program_and_grounder_emit_native_nominal_and_fallback_routes(self):
        flow = self.index.compile_decision(
            _decision(),
            INSTRUCTION,
            "set_table",
        )
        program = FlowProgram(flow)
        routes = {route.id: route for route in program.routes()}
        self.assertEqual(len(routes), 7)
        self.assertIn("fallback.bowl_retrieve", routes)
        fallback = routes["fallback.bowl_retrieve"]
        self.assertNotIn(
            "bowl_retrieve:pick_drawer_object",
            fallback.node_order,
        )
        self.assertIn(
            "bowl_retrieve:regrasp_drawer_object",
            fallback.node_order,
        )

        grounder = MSHabPlanGrounder(
            _official_source(),
            task_family="set_table",
            plan_index=0,
        )
        nominal_plan = grounder.compile(program, routes["nominal"])
        fallback_plan = grounder.compile(program, fallback)
        self.assertEqual(len(nominal_plan["plans"][0]["subtasks"]), 16)
        self.assertEqual(len(fallback_plan["plans"][0]["subtasks"]), 16)
        self.assertEqual(
            nominal_plan["selection"]["recommended_policy_type"],
            "rl_per_obj",
        )
        self.assertEqual(
            fallback_plan["selection"]["execution_semantics"],
            "precompiled_fallback_path_for_failure_injection",
        )
        self.assertEqual(
            fallback_plan["selection"]["flow_node_order"],
            list(fallback.node_order),
        )
        fallback_types = [
            item["type"] for item in fallback_plan["plans"][0]["subtasks"]
        ]
        for index, contract_type in enumerate(fallback_types[:-1]):
            if contract_type == "navigate":
                self.assertNotEqual(fallback_types[index + 1], "navigate")

    def test_grounder_resolves_semantic_object_bindings(self):
        decision = copy.deepcopy(_decision())
        for occurrence in decision["strategy_occurrences"]:
            obj = occurrence["bindings"].get("object")
            if obj == "024_bowl":
                occurrence["bindings"]["object"] = "bowl"
            elif obj == "013_apple":
                occurrence["bindings"]["object"] = "apple"
        flow = self.index.compile_decision(decision, INSTRUCTION, "set_table")
        program = FlowProgram(flow)
        plan = MSHabPlanGrounder(
            _official_source(),
            task_family="set_table",
            plan_index=0,
        ).compile(program, program.route("nominal"))
        self.assertEqual(len(plan["plans"][0]["subtasks"]), 16)
        self.assertEqual(
            plan["selection"]["recommended_policy_type"],
            "rl_per_obj",
        )
        object_policies = [
            item["policy_id"]
            for item in plan["selection"]["skill_decisions"]
            if item["contract_type"] in ("pick", "place")
        ]
        self.assertTrue(object_policies)
        self.assertTrue(all(not item.endswith(".all") for item in object_policies))

    def test_grounder_rejects_unknown_explicit_instance(self):
        decision = copy.deepcopy(_decision())
        for occurrence in decision["strategy_occurrences"]:
            if occurrence["bindings"].get("object") == "024_bowl":
                occurrence["bindings"]["object"] = "024_bowl-999"
        flow = self.index.compile_decision(decision, INSTRUCTION, "set_table")
        program = FlowProgram(flow)
        grounder = MSHabPlanGrounder(
            _official_source(),
            task_family="set_table",
            plan_index=0,
        )
        with self.assertRaisesRegex(ValueError, "cannot ground flow node"):
            grounder.compile(program, program.route("nominal"))

    def test_grounder_rejects_ambiguous_semantic_category(self):
        source = _official_source()
        for subtask in source["plans"][0]["subtasks"][8:]:
            if subtask.get("obj_id") == "013_apple-0":
                subtask["obj_id"] = "024_bowl-5"
        decision = copy.deepcopy(_decision())
        for occurrence in decision["strategy_occurrences"]:
            if occurrence["bindings"].get("object") == "024_bowl":
                occurrence["bindings"]["object"] = "bowl"
        flow = self.index.compile_decision(decision, INSTRUCTION, "set_table")
        program = FlowProgram(flow)
        grounder = MSHabPlanGrounder(
            source,
            task_family="set_table",
            plan_index=0,
        )
        with self.assertRaisesRegex(ValueError, "ambiguous semantic target"):
            grounder.compile(program, program.route("nominal"))

    def test_simulator_bundle_writes_every_route(self):
        flow = self.index.compile_decision(
            _decision(),
            INSTRUCTION,
            "set_table",
        )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = write_simulator_bundle(
                flow,
                _official_source(),
                task_family="set_table",
                plan_index=0,
                output_dir=output_dir,
            )
            self.assertEqual(len(manifest["routes"]), 7)
            self.assertTrue((output_dir / "nominal.json").is_file())
            self.assertTrue(
                (output_dir / "fallback.bowl_retrieve.json").is_file()
            )
            self.assertTrue((output_dir / "manifest.json").is_file())

    def test_normalized_gt_uses_same_flow_schema_and_matches_prediction(self):
        source = _official_source()
        prediction = self.index.compile_decision(
            _decision(),
            INSTRUCTION,
            "set_table",
        )
        ground_truth = normalize_ground_truth(
            source,
            plan_index=0,
            instruction=INSTRUCTION,
            task_family="set_table",
        )
        self.assertEqual(
            prediction["schema_version"],
            ground_truth["schema_version"],
        )
        metrics = compare_flows(prediction, ground_truth)
        self.assertTrue(metrics["contract_type_exact_match"])
        self.assertEqual(metrics["edit_distance"], 0)
        self.assertEqual(metrics["lcs_recall"], 1.0)
        self.assertEqual(metrics["grounded_target_accuracy"], 1.0)
        self.assertTrue(metrics["semantic_grounding_exact_match"])
        self.assertFalse(metrics["fallback_gt_available"])

    def test_wrong_delivery_destination_is_not_scored_as_a_match(self):
        wrong = _decision()
        wrong["strategy_occurrences"][1]["strategy_id"] = "deliver_to_counter"
        wrong["strategy_occurrences"][4]["strategy_id"] = "deliver_to_counter"
        prediction = self.index.compile_decision(wrong, INSTRUCTION, "set_table")
        ground_truth = normalize_ground_truth(
            _official_source(),
            plan_index=0,
            instruction=INSTRUCTION,
            task_family="set_table",
        )
        metrics = compare_flows(prediction, ground_truth)
        self.assertTrue(metrics["contract_type_exact_match"])
        self.assertFalse(metrics["semantic_grounding_exact_match"])
        self.assertLess(metrics["grounded_target_accuracy"], 1.0)
        self.assertTrue(
            any(item["field"] == "destination" for item in metrics["grounding_mismatches"])
        )

    def test_unobservable_tidy_house_destination_cannot_be_an_exact_match(self):
        instruction = "Move the cracker box to its requested destination."
        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "instruction": instruction,
            "task_family": "tidy_house",
            "strategy_occurrences": [
                {
                    "id": "retrieve",
                    "strategy_id": "retrieve_surface",
                    "bindings": {"object": "003_cracker_box"},
                },
                {
                    "id": "deliver",
                    "strategy_id": "deliver_to_table",
                    "bindings": {"object": "003_cracker_box"},
                },
            ],
            "occurrence_order": ["retrieve", "deliver"],
            "decision_summary": "Retrieve and deliver the cracker box.",
        }
        source = {
            "dataset": "ReplicaCADTidyHouseVal",
            "plans": [
                {
                    "subtasks": [
                        {"type": "navigate"},
                        {"type": "pick", "obj_id": "003_cracker_box-0"},
                        {"type": "navigate"},
                        {
                            "type": "place",
                            "obj_id": "003_cracker_box-0",
                            "goal_pos": [1.0, 2.0, 0.8],
                        },
                    ]
                }
            ],
        }
        prediction = self.index.compile_decision(
            decision,
            instruction,
            "tidy_house",
        )
        ground_truth = normalize_ground_truth(
            source,
            plan_index=0,
            instruction=instruction,
            task_family="tidy_house",
        )
        metrics = compare_flows(prediction, ground_truth)
        self.assertTrue(metrics["contract_type_exact_match"])
        self.assertFalse(metrics["gt_grounding_fully_observable"])
        self.assertFalse(metrics["semantic_grounding_exact_match"])

    def test_invented_strategy_and_wrong_bindings_are_rejected(self):
        invented = _decision()
        invented["strategy_occurrences"][0]["strategy_id"] = "invented_route"
        with self.assertRaisesRegex(ValueError, "unknown strategy_id"):
            self.index.compile_decision(invented, INSTRUCTION, "set_table")

        wrong_bindings = _decision()
        wrong_bindings["strategy_occurrences"][0]["bindings"] = {}
        with self.assertRaisesRegex(ValueError, "requires bindings"):
            self.index.compile_decision(wrong_bindings, INSTRUCTION, "set_table")

        extra_field = _decision()
        extra_field["rationale"] = "not part of the closed schema"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.index.compile_decision(extra_field, INSTRUCTION, "set_table")

    def test_graph_index_rejects_broken_cross_layer_and_fallback_mappings(self):
        missing_fallback = copy.deepcopy(self.graph)
        retrieve = next(
            item
            for item in missing_fallback["layer2"]["subgraphs"]
            if item["subgoal_id"] == "retrieve"
        )
        retrieve["edges"] = [
            edge
            for edge in retrieve["edges"]
            if not (
                edge["relation"] == "fallback_to"
                and edge["source"] == "pick_drawer_object"
            )
        ]
        with self.assertRaisesRegex(ValueError, "no matching fallback_to edge"):
            FourLayerGraphIndex(missing_fallback)

        bad_reference = copy.deepcopy(self.graph)
        bad_reference["layer2_to_layer3"]["references"][0]["target"] = (
            "missing.contract"
        )
        with self.assertRaisesRegex(ValueError, "invalid SkillNode-to-Contract"):
            FourLayerGraphIndex(bad_reference)

        bad_executes = copy.deepcopy(self.graph)
        bad_executes["layer4_to_layer3"]["connections"][0]["relation"] = (
            "NOT_EXECUTES"
        )
        with self.assertRaisesRegex(ValueError, "must be EXECUTES"):
            FourLayerGraphIndex(bad_executes)

    def test_semantic_order_and_explicit_scene_membership_are_reported(self):
        wrong = _decision()
        wrong["occurrence_order"] = [
            "bowl_deliver",
            "bowl_retrieve",
            "bowl_restore",
            "apple_retrieve",
            "apple_deliver",
            "apple_restore",
        ]
        context = {
            "objects": ["013_apple"],
            "articulations": ["fridge", "kitchen_counter"],
        }
        flow = self.index.compile_decision(
            wrong,
            INSTRUCTION,
            "set_table",
            context,
        )
        validation = flow["metadata"]["semantic_validation"]
        self.assertFalse(validation["valid"])
        self.assertEqual(
            {item["code"] for item in validation["issues"]},
            {"deliver_before_retrieve", "object_not_in_scene_context"},
        )

    def test_api_key_is_not_part_of_request_payload(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                """deepseek:
  base_url: https://api.deepseek.com
  model: deepseek-flash
  api_key_env: TEST_DEEPSEEK_KEY
  api_key: file-secret
  temperature: 0
  max_tokens: 100
  timeout_seconds: 10
  retries: 0
"""
            )
            config_path.chmod(0o600)
            config = DeepSeekConfig.from_yaml(config_path)
        with mock.patch.dict(os.environ, {"TEST_DEEPSEEK_KEY": "env-secret"}):
            self.assertEqual(config.resolved_api_key(), "env-secret")
        payload = config.request_payload([{"role": "user", "content": "json"}])
        serialized = json.dumps(payload)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("file-secret", serialized)
        self.assertNotIn("env-secret", serialized)
        self.assertNotIn("Authorization", serialized)
        self.assertNotIn("file-secret", repr(config))

    def test_malformed_api_envelope_and_wrong_gt_family_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty choices"):
            _parse_chat_completion({"choices": []})
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_gt_task_family(
                {"dataset": "ReplicaCADTidyHouseVal"},
                "set_table",
            )

    def test_http_redirects_are_rejected_before_reusing_authorization(self):
        handler = _NoRedirectHandler()
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": "Bearer secret"},
        )
        self.assertIsNone(
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://untrusted.example/collect",
            )
        )

    def test_graphviz_flowchart_is_valid_svg_when_dot_is_available(self):
        if shutil.which("dot") is None:
            self.skipTest("Graphviz dot is unavailable")
        flow = self.index.compile_decision(
            _decision(),
            INSTRUCTION,
            "set_table",
        )
        with TemporaryDirectory() as directory:
            dot_path, svg_path = render_flowchart(flow, Path(directory))
            self.assertTrue(dot_path.is_file())
            self.assertIsNotNone(svg_path)
            ET.fromstring(svg_path.read_text())

    def test_offline_cli_compiles_response_and_compares_ground_truth(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            response_path = root / "decision.json"
            ground_truth_path = root / "ground_truth.json"
            output_dir = root / "outputs"
            config_path.write_text(
                """deepseek:
  base_url: https://api.deepseek.com
  model: deepseek-flash
  api_key_env: UNUSED_TEST_KEY
  api_key: ""
  temperature: 0
  max_tokens: 100
  timeout_seconds: 10
  retries: 0
"""
            )
            response_path.write_text(json.dumps(_decision()))
            ground_truth_path.write_text(json.dumps(_official_source()))
            argv = [
                "mshab-vlm",
                "--instruction",
                INSTRUCTION,
                "--task-family",
                "set_table",
                "--config",
                str(config_path),
                "--gt",
                str(ground_truth_path),
                "--response-file",
                str(response_path),
                "--skip-judge",
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                main()

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(FileExistsError, "not empty"):
                    main()

            result = json.loads((output_dir / "result.json").read_text())
            request_text = (output_dir / "planner_request.json").read_text()
            self.assertNotIn("SECRET-GT", request_text)
            self.assertNotIn("build.json", request_text)
            self.assertNotIn("init.json", request_text)
            self.assertTrue(
                result["deterministic_metrics"]["contract_type_exact_match"]
            )
            self.assertEqual(
                result["deterministic_metrics"]["edit_distance"],
                0,
            )
            self.assertTrue((output_dir / "predicted_flow.json").is_file())
            self.assertTrue((output_dir / "predicted_flow.dot").is_file())
            self.assertTrue((output_dir / "simulator" / "nominal.json").is_file())
            self.assertTrue(
                (output_dir / "simulator" / "manifest.json").is_file()
            )
            self.assertEqual(
                result["simulator_compilation"]["status"],
                "ready",
            )
            if shutil.which("dot") is not None:
                self.assertTrue((output_dir / "predicted_flow.svg").is_file())
