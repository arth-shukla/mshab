import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.skills import (
    ArtifactStatus,
    PolicyExecution,
    PolicyExecutor,
    CheckpointPolicy,
    EnvironmentDescription,
    EnvironmentEntity,
    SubGoal,
    SubGoalGraph,
    SubGoalDependency,
    SkillSubgraph,
    MSHabEnvironmentAdapter,
    NavigateContract,
    PickContract,
    SkillGraphPatch,
    SkillGraph,
    LibraryCatalog,
    SkillGrounder,
    ContractLibrary,
    SkillNode,
    SkillRelation,
    SkillRuntime,
    CrossSubgraphEdge,
    ContractType,
    YourContract,
    build_set_table_apple_graph,
    build_set_table_graph,
    build_set_table_stack,
    build_set_table_starter,
)
from scripts.generate_set_table_skill_graph import graph_document, set_table_svg


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "mshab"
    / "skills"
    / "catalogs"
    / "set_table.json"
)


class SkillModelTests(TestCase):
    def test_specialized_atomic_skill_binds_target_and_contract(self):
        contract = PickContract(task="set_table", target="013_apple")
        grounded = contract.bind({})

        self.assertEqual(grounded.arguments, {"object": "013_apple"})
        self.assertEqual(
            grounded.preconditions,
            ("reachable(013_apple)", "gripper_empty()"),
        )
        self.assertEqual(grounded.effects, ("holding(013_apple)",))

        with self.assertRaisesRegex(ValueError, "specialized"):
            contract.bind({"object": "024_bowl"})

    def test_checkpoint_policy_reports_missing_partial_and_ready(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = CheckpointPolicy(
                id="rl.set_table.pick.013_apple",
                family="rl",
                checkpoint_path=root / "policy.pt",
                config_path=root / "config.yml",
            )
            self.assertEqual(policy.status, ArtifactStatus.MISSING)
            (root / "config.yml").write_text("name: ppo\n")
            self.assertEqual(policy.status, ArtifactStatus.PARTIAL)
            (root / "policy.pt").write_bytes(b"weights")
            self.assertEqual(policy.status, ArtifactStatus.READY)

    def test_discovery_binds_families_and_generic_checkpoints_many_to_many(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family, target in (
                ("rl", "013_apple"),
                ("bc", "013_apple"),
                ("rl", "all"),
            ):
                leaf = root / family / "set_table" / "pick" / target
                leaf.mkdir(parents=True)
                (leaf / "config.yml").write_text("name: {}\n".format(family))
                (leaf / "policy.pt").write_bytes(b"weights")

            library = ContractLibrary.from_checkpoint_root(root)
            apple = library.get("mshab.set_table.pick.013_apple")
            generic = library.get("mshab.set_table.pick.all")

            self.assertEqual(
                len(library.find(task="set_table", contract_type=ContractType.PICK)), 2
            )
            # One contract <- several policies: its own two families first (in
            # discovery order), then the all-object checkpoint as the
            # Layer-4 fallback.
            self.assertEqual(
                [policy.id for policy in library.policies_for(apple.id)],
                [
                    "bc.set_table.pick.013_apple",
                    "rl.set_table.pick.013_apple",
                    "rl.set_table.pick.all",
                ],
            )
            # One policy -> several contracts.
            self.assertEqual(
                [contract.id for contract in library.contracts_for("rl.set_table.pick.all")],
                [apple.id, generic.id],
            )
            self.assertEqual(
                [policy.id for policy in library.policies_for(generic.id)],
                ["rl.set_table.pick.all"],
            )
            self.assertTrue(library.ready(apple.id))
            self.assertEqual(
                library.select_policy(apple.id).id, "bc.set_table.pick.013_apple"
            )

    def test_composition_is_expressed_only_with_graph_relations(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for contract_type, target in (
                ("navigate", "all"),
                ("pick", "013_apple"),
                ("pick", "all"),
            ):
                leaf = root / "rl" / "set_table" / contract_type / target
                leaf.mkdir(parents=True)
                (leaf / "config.yml").write_text("name: ppo\n")
                (leaf / "policy.pt").write_bytes(b"weights")

            library = ContractLibrary.from_checkpoint_root(root)
            navigate = library.get("mshab.set_table.navigate.all")
            pick = library.get("mshab.set_table.pick.013_apple")
            generic_pick = library.get("mshab.set_table.pick.all")

            subgoals = SubGoalGraph("Retrieve the apple")
            subgoals.add_subgoal(SubGoal("reachable", "reachable(013_apple)"))
            subgoals.add_subgoal(SubGoal("retrieved", "holding(013_apple)"))
            subgoals.add_dependency("reachable", "retrieved")

            graph = SkillGraph("set_table", subgoal_graph=subgoals)
            reachable_subgraph = SkillSubgraph("reachable", "set_table")
            reachable_subgraph.add_node(
                SkillNode(
                    "navigate",
                    navigate.id,
                    {"goal": "013_apple"},
                    achieves=("reachable",),
                )
            )
            retrieved_subgraph = SkillSubgraph("retrieved", "set_table")
            retrieved_subgraph.add_node(
                SkillNode(
                    "pick",
                    pick.id,
                    {},
                    achieves=("retrieved",),
                )
            )
            retrieved_subgraph.add_node(
                SkillNode(
                    "generic_pick",
                    generic_pick.id,
                    {"object": "013_apple"},
                    achieves=("retrieved",),
                )
            )
            retrieved_subgraph.relate(
                "pick", "generic_pick", SkillRelation.IS_A
            )
            retrieved_subgraph.relate(
                "pick", "generic_pick", SkillRelation.ALTERNATIVE_TO
            )
            retrieved_subgraph.relate(
                "pick", "generic_pick", SkillRelation.FALLBACK_TO
            )
            graph.add_subgraph(reachable_subgraph)
            graph.add_subgraph(retrieved_subgraph)
            with self.assertRaisesRegex(RuntimeError, "registered and sealed"):
                retrieved_subgraph.add_node(
                    SkillNode(
                        "late_mutation",
                        generic_pick.id,
                        {"object": "013_apple"},
                        achieves=("retrieved",),
                    )
                )
            graph.relate("navigate", "pick", SkillRelation.ENABLES)
            graph.relate("generic_pick", "navigate", SkillRelation.REQUIRES)

            order = graph.execution_order()
            self.assertLess(order.index("navigate"), order.index("pick"))
            self.assertLess(order.index("navigate"), order.index("generic_pick"))
            self.assertEqual(
                tuple(item.id for item in graph.prerequisites("pick")),
                ("navigate",),
            )
            self.assertEqual(
                tuple(item.id for item in graph.alternatives("pick")),
                ("generic_pick",),
            )
            self.assertEqual(
                tuple(item.id for item in graph.fallbacks("pick")),
                ("generic_pick",),
            )
            self.assertEqual(
                tuple(item.id for item in graph.candidates_for_subgoal("retrieved")),
                ("generic_pick", "pick"),
            )
            self.assertEqual(graph.uncovered_subgoals(), ())
            self.assertEqual(
                {edge.relation for edge in graph.edges}, set(SkillRelation)
            )
            self.assertEqual(
                tuple(item.id for item in graph.ready_nodes()),
                ("navigate",),
            )

            grounder = SkillGrounder(library)
            self.assertEqual(
                grounder.ground(graph.nodes["pick"]).effects,
                ("holding(013_apple)",),
            )

            with self.assertRaisesRegex(ValueError, "causal cycle"):
                graph.relate("pick", "navigate", SkillRelation.ENABLES)
            self.assertEqual(len(graph.edges), 5)  # failed relation was rolled back

    def test_invocation_arguments_are_read_only(self):
        call = PickContract(task="set_table", target="013_apple").bind({})
        with self.assertRaises(TypeError):
            call.arguments["object"] = "024_bowl"

    def test_starter_graph_builds_layers_one_and_two_without_environment(self):
        subgoals, graph = build_set_table_apple_graph()

        self.assertEqual(len(subgoals.subgoals), 4)
        self.assertEqual(len(graph.subgraphs), 4)
        self.assertEqual(len(graph.nodes), 10)
        self.assertEqual(graph.uncovered_subgoals(), ())
        self.assertEqual(
            dict(graph.nodes["navigate_to_object"].arguments),
            {"goal": "013_apple"},
        )
        self.assertNotIn("policy_id", graph.as_dict()["nodes"][0])
        self.assertEqual(
            set(graph.subgraphs),
            {"source_open", "object_retrieved", "object_placed", "source_closed"},
        )
        self.assertEqual(
            graph.owner_of("navigate_to_object"), "object_retrieved"
        )
        self.assertEqual(
            tuple(
                goal.id
                for goal in subgoals.ready_subgoals(
                    {"closed(fridge)"},
                    completed=("source_open", "object_retrieved"),
                )
            ),
            ("object_placed",),
        )

    def test_complete_set_table_graph_matches_official_two_object_order(self):
        subgoals, graph = build_set_table_graph()

        self.assertEqual(len(subgoals.subgoals), 8)
        self.assertEqual(len(graph.subgraphs), 8)
        self.assertEqual(len(graph.nodes), 20)
        # 24 internal edges + 7 cross-goal relations that expand to 11
        # concrete edges (a goal-level endpoint yields one per achiever).
        self.assertEqual(len(graph.edges), 35)
        self.assertEqual(len(graph.cross_edges), 7)
        self.assertEqual(
            sum(len(item.edges(graph.subgraphs)) for item in graph.cross_edges),
            11,
        )
        self.assertEqual(graph.uncovered_subgoals(), ())
        subgoal_order = subgoals.execution_order()
        self.assertLess(
            subgoal_order.index("bowl_source_closed"),
            subgoal_order.index("apple_source_open"),
        )
        node_order = graph.execution_order()
        self.assertLess(
            node_order.index("close_bowl_source"),
            node_order.index("navigate_to_apple_source"),
        )
        self.assertEqual(
            graph.nodes["pick_bowl_specialized"].contract_id,
            "mshab.set_table.pick.024_bowl",
        )
        self.assertEqual(
            set(graph.subgraphs["bowl_retrieved"].nodes),
            {"navigate_to_bowl", "pick_bowl_specialized", "pick_bowl_generic"},
        )
        self.assertEqual(
            graph.subgraphs["bowl_retrieved"].achievers,
            (
                graph.nodes["pick_bowl_generic"],
                graph.nodes["pick_bowl_specialized"],
            ),
        )

    def test_checked_in_set_table_json_and_svg_match_graph(self):
        repository_root = Path(__file__).resolve().parents[1]
        document = json.loads(
            (
                repository_root
                / "mshab"
                / "skills"
                / "catalogs"
                / "set_table.json"
            ).read_text()
        )
        subgoals, graph = build_set_table_graph()
        layer_1 = document["layers"]["1_subgoal_graph"]
        layer_2 = document["layers"]["2_skill_graph"]

        self.assertEqual(
            layer_1["execution_order"], list(subgoals.execution_order())
        )
        self.assertEqual(layer_2["subgraphs"], graph.as_dict()["subgraphs"])
        self.assertEqual(
            layer_2["cross_edges"], graph.as_dict()["cross_edges"]
        )
        self.assertNotIn("/home/", json.dumps(document))
        restored = LibraryCatalog.from_dict(document)
        self.assertEqual(restored.as_dict(), document)
        stale = json.loads(json.dumps(document))
        stale["layers"]["2_skill_graph"]["nodes"][0][
            "contract_id"
        ] = "mshab.set_table.pick.all"
        with self.assertRaisesRegex(ValueError, "derived nodes view is stale"):
            LibraryCatalog.from_dict(stale)

        svg_path = (
            repository_root
            / "docs"
            / "static"
            / "images"
            / "set_table_skill_graph.svg"
        )
        ET.parse(svg_path)

        # Canonical artifacts must regenerate on a clean clone: no checkpoint
        # file is created in this temporary root, and local ready/missing state
        # is not part of the checked-in catalog.
        with TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp) / "mshab_checkpoints"
            stack = build_set_table_stack(checkpoint_root)
            generated = graph_document(stack, checkpoint_root)

        self.assertEqual(generated, document)
        self.assertEqual(set_table_svg(generated), svg_path.read_text())
        self.assertNotIn('"ready"', json.dumps(document))
        self.assertNotIn('"status"', json.dumps(document))

    def test_starter_stack_binds_layer_three_from_downloaded_layout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for contract_type, target in (
                ("navigate", "all"),
                ("open", "fridge"),
                ("pick", "013_apple"),
                ("pick", "all"),
                ("close", "fridge"),
                ("place", "013_apple"),
                ("place", "all"),
            ):
                leaf = root / "rl" / "set_table" / contract_type / target
                leaf.mkdir(parents=True)
                (leaf / "config.yml").write_text("name: ppo\n")
                (leaf / "policy.pt").write_bytes(b"weights")

            stack = build_set_table_starter(root)

            self.assertEqual(len(stack.grounded_skills), 10)
            self.assertEqual(
                stack.grounded_skills["place_object_specialized"].effects,
                ("at(013_apple,dining_table)", "gripper_empty()"),
            )
            # Seven contracts have their own checkpoint.  pick.024_bowl and
            # place.024_bowl have none, but the all-object checkpoints are
            # bound to them too, so nine of the eleven contracts are executable.
            self.assertEqual(len(stack.library.find(ready=True)), 9)
            self.assertEqual(
                stack.library.select_policy("mshab.set_table.pick.024_bowl").id,
                "rl.set_table.pick.all",
            )
            self.assertEqual(
                stack.library.select_policy("mshab.set_table.pick.013_apple").id,
                "rl.set_table.pick.013_apple",
            )

    def test_graph_patch_can_place_a_new_skill_subgraph(self):
        subgoals = SubGoalGraph("Retrieve and inspect the apple")
        subgoals.add_subgoal(SubGoal("retrieved", "holding(013_apple)"))
        graph = SkillGraph("set_table", subgoal_graph=subgoals)
        retrieved = SkillSubgraph("retrieved", "set_table")
        retrieved.add_node(
            SkillNode(
                "retrieve_apple",
                "mshab.set_table.pick.013_apple",
                {},
                achieves=("retrieved",),
            )
        )
        graph.add_subgraph(retrieved)
        inspected = SkillSubgraph("inspected", "set_table")
        inspected.add_node(
            SkillNode(
                "inspect_apple",
                "mshab.set_table.your_contract.013_apple",
                {},
                achieves=("inspected",),
            )
        )
        patch = SkillGraphPatch(
            subgoals=(SubGoal("inspected", "inspected(013_apple)"),),
            subgoal_dependencies=(SubGoalDependency("retrieved", "inspected"),),
            skill_subgraphs=(inspected,),
            cross_edges=(
                CrossSubgraphEdge(
                    "retrieved",
                    "inspected",
                    "retrieve_apple",
                    "inspect_apple",
                    SkillRelation.ENABLES,
                ),
            ),
        )

        patch.apply(subgoals, graph)

        self.assertIn("inspected", subgoals.subgoals)
        self.assertIn("inspect_apple", graph.nodes)
        self.assertEqual(graph.owner_of("inspect_apple"), "inspected")

        broken = SkillSubgraph("broken", "set_table")
        with self.assertRaisesRegex(ValueError, "outside task namespace"):
            broken.add_node(
                SkillNode(
                    "broken_node",
                    "mshab.wrong_task.your_contract.target",
                    {},
                    achieves=("broken",),
                )
            )

    def test_custom_atomic_skill_type_is_extendable(self):
        contract = YourContract("set_table", "013_apple")
        call = contract.bind({})

        self.assertEqual(contract.contract_type, "your_contract")
        self.assertEqual(call.arguments, {"target": "013_apple"})
        self.assertEqual(
            call.effects, ("your_contract_done(013_apple)",)
        )

    def test_runtime_uses_environment_facts_and_verifies_contract(self):
        class FakeEnv:
            def __init__(self):
                self.facts = {
                    "reachable(013_apple)",
                    "gripper_empty()",
                    "collision_safe()",
                }

            def reset(self, **kwargs):
                return {}, {"facts": set(self.facts)}

            def step(self, action):
                self.facts -= set(action["delete_facts"])
                self.facts |= set(action["add_facts"])
                return {}, 1.0, False, False, {"facts": set(self.facts)}

        class FakeExecutor(PolicyExecutor):
            def execute(self, grounded, policy, environment, monitor):
                snapshot = environment.step(
                    {
                        "add_facts": set(grounded.effects),
                        "delete_facts": set(grounded.deletes),
                    }
                )
                monitor(snapshot)
                return PolicyExecution(success=True, steps=1)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.yml").write_text("name: ppo\n")
            (root / "policy.pt").write_bytes(b"weights")
            contract = PickContract("set_table", "013_apple")
            policy = CheckpointPolicy(
                "rl.set_table.pick.013_apple",
                "rl",
                root / "policy.pt",
                root / "config.yml",
            )
            library = ContractLibrary(
                (contract,), (policy,), ((contract.id, policy.id),)
            )
            subgoals = SubGoalGraph("Retrieve the apple")
            subgoals.add_subgoal(SubGoal("retrieved", "holding(013_apple)"))
            graph = SkillGraph("set_table", subgoals)
            subgraph = SkillSubgraph("retrieved", "set_table")
            subgraph.add_node(
                SkillNode(
                    "pick",
                    contract.id,
                    {},
                    achieves=("retrieved",),
                )
            )
            graph.add_subgraph(subgraph)
            description = EnvironmentDescription(
                "SequentialTask-v0",
                entities={
                    "013_apple": EnvironmentEntity(
                        "013_apple", "object", "obj_0"
                    )
                },
                compatible_contract_env_ids=("PickSubtaskTrain-v0",),
            )
            env = FakeEnv()
            adapter = MSHabEnvironmentAdapter(
                env,
                description.environment_id,
                lambda observation, info, desc: info["facts"],
                entities=description.entities.values(),
                compatible_contract_env_ids=description.compatible_contract_env_ids,
            )
            adapter.reset()
            runtime = SkillRuntime(library, adapter)

            self.assertEqual(
                tuple(node.id for node in runtime.ready_nodes(graph)), ("pick",)
            )
            # Pinning a policy that is not bound to the contract makes the
            # node unexecutable rather than raising.
            self.assertEqual(
                runtime.ready_nodes(graph, policy_ids={"pick": "vla.manipulation"}),
                (),
            )
            result = runtime.execute_node(graph, "pick", FakeExecutor(), policy.id)

            self.assertTrue(result.success)
            self.assertEqual(result.policy_id, policy.id)
            self.assertEqual(result.unretracted_deletes, ())
            self.assertEqual(result.violated_invariants, ())
            self.assertNotIn("gripper_empty()", result.after.facts)


class LibraryBindingTests(TestCase):
    """Contract <-> policy is a many-to-many relation owned by the library."""

    @staticmethod
    def _checkpoint(root, policy_id, ready=True):
        leaf = root / policy_id
        leaf.mkdir(parents=True)
        if ready:
            (leaf / "config.yml").write_text("name: ppo\n")
            (leaf / "policy.pt").write_bytes(b"weights")
        return CheckpointPolicy(
            policy_id, "rl", leaf / "policy.pt", leaf / "config.yml"
        )

    def test_one_policy_executes_several_contracts_and_vice_versa(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            apple = PickContract("set_table", "013_apple")
            bowl = PickContract("set_table", "024_bowl")
            generic = PickContract("set_table", "all")
            per_apple = self._checkpoint(root, "rl.set_table.pick.013_apple")
            all_objects = self._checkpoint(root, "rl.set_table.pick.all")
            vla = self._checkpoint(root, "vla.manipulation")
            library = ContractLibrary(
                contracts=(apple, bowl, generic),
                policies=(per_apple, all_objects, vla),
                bindings=(
                    (apple.id, per_apple.id),
                    (apple.id, all_objects.id),
                    (bowl.id, all_objects.id),
                    (generic.id, all_objects.id),
                    (apple.id, vla.id),
                    (bowl.id, vla.id),
                ),
            )

            self.assertEqual(
                [policy.id for policy in library.policies_for(apple.id)],
                [per_apple.id, all_objects.id, vla.id],
            )
            self.assertEqual(
                [contract.id for contract in library.contracts_for(all_objects.id)],
                [apple.id, bowl.id, generic.id],
            )
            self.assertEqual(
                [contract.id for contract in library.contracts_for(vla.id)],
                [apple.id, bowl.id],
            )
            self.assertEqual(
                [policy.id for policy in library.policies],
                [per_apple.id, all_objects.id, vla.id],
            )
            # Neither side owns the other.
            self.assertNotIn("policies", apple.as_dict())
            self.assertNotIn("policies", apple.as_dict()["execution"])
            self.assertNotIn("contracts", vla.as_dict())

            exported = library.to_dict()
            self.assertEqual(
                exported["contracts"][0]["policies"],
                [per_apple.id, all_objects.id, vla.id],
            )
            self.assertEqual(
                exported["policies"][1]["contracts"], [apple.id, bowl.id, generic.id]
            )
            self.assertTrue(json.dumps(exported))

    def test_default_policy_is_the_first_ready_one_in_binding_order(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            apple = PickContract("set_table", "013_apple")
            own = self._checkpoint(root, "rl.set_table.pick.013_apple", ready=False)
            fallback = self._checkpoint(root, "rl.set_table.pick.all")
            library = ContractLibrary(
                (apple,), (own, fallback), ((apple.id, own.id), (apple.id, fallback.id))
            )

            self.assertTrue(library.ready(apple.id))
            self.assertIs(library.select_policy(apple.id), fallback)
            # An explicit request only has to be bound; readiness is the
            # caller's concern (SkillRuntime.ready_nodes checks it).
            self.assertIs(library.select_policy(apple.id, own.id), own)
            with self.assertRaisesRegex(KeyError, "not bound"):
                library.select_policy(apple.id, "vla.manipulation")

            (root / own.id / "config.yml").write_text("name: ppo\n")
            (root / own.id / "policy.pt").write_bytes(b"weights")
            self.assertIs(library.select_policy(apple.id), own)

    def test_bindings_need_registered_ids_and_reject_duplicates(self):
        apple = PickContract("set_table", "013_apple")
        policy = CheckpointPolicy(
            "rl.set_table.pick.013_apple",
            "rl",
            Path("policy.pt"),
            Path("config.yml"),
        )
        library = ContractLibrary((apple,), (policy,))

        with self.assertRaisesRegex(KeyError, "unknown policy"):
            library.bind(apple.id, "vla.manipulation")
        with self.assertRaisesRegex(KeyError, "unknown contract"):
            library.bind("mshab.set_table.pick.all", policy.id)
        self.assertEqual(library.policies_for(apple.id), ())
        self.assertEqual(library.contracts_for(policy.id), ())

        library.bind(apple.id, policy.id)
        with self.assertRaisesRegex(ValueError, "already bound"):
            library.bind(apple.id, policy.id)
        with self.assertRaisesRegex(ValueError, "duplicate policy id"):
            library.register_policy(policy)
        self.assertFalse(library.ready(apple.id))
        self.assertEqual(library.find(ready=True), [])
        with self.assertRaisesRegex(RuntimeError, "no ready policy"):
            library.select_policy(apple.id)

    def test_generic_policies_bind_only_to_same_task_and_type(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            apple = PickContract("set_table", "013_apple")
            generic = PickContract("set_table", "all")
            navigate = NavigateContract("set_table")
            other_task = PickContract("tidy_house", "all")
            all_pick = self._checkpoint(root, "rl.set_table.pick.all")
            all_navigate = self._checkpoint(root, "rl.set_table.navigate.all")
            tidy_pick = self._checkpoint(root, "rl.tidy_house.pick.all")
            library = ContractLibrary(
                (apple, generic, navigate, other_task),
                (all_pick, all_navigate, tidy_pick),
                (
                    (generic.id, all_pick.id),
                    (navigate.id, all_navigate.id),
                    (other_task.id, tidy_pick.id),
                ),
            )
            self.assertEqual(library.policies_for(apple.id), ())

            library.bind_generic_policies()
            library.bind_generic_policies()  # idempotent

            self.assertEqual(
                [policy.id for policy in library.policies_for(apple.id)],
                [all_pick.id],
            )
            self.assertEqual(
                [contract.id for contract in library.contracts_for(all_navigate.id)],
                [navigate.id],
            )
            self.assertEqual(
                [contract.id for contract in library.contracts_for(tidy_pick.id)],
                [other_task.id],
            )

    def test_catalog_layer_four_lists_bindings_from_both_sides(self):
        document = json.loads(CATALOG_PATH.read_text())
        layer_4 = document["layers"]["4_contracts_and_policies"]
        by_contract = {record["id"]: record for record in layer_4["contracts"]}
        by_policy = {record["id"]: record for record in layer_4["policies"]}

        self.assertEqual(
            by_contract["mshab.set_table.pick.013_apple"]["policies"],
            ["rl.set_table.pick.013_apple", "rl.set_table.pick.all"],
        )
        self.assertEqual(
            by_contract["mshab.set_table.open.fridge"]["policies"],
            ["rl.set_table.open.fridge"],
        )
        self.assertEqual(
            by_policy["rl.set_table.pick.all"]["contracts"],
            [
                "mshab.set_table.pick.013_apple",
                "mshab.set_table.pick.024_bowl",
                "mshab.set_table.pick.all",
            ],
        )
        self.assertEqual(document["summary"]["registered_contracts"], 11)
        self.assertEqual(document["summary"]["registered_policies"], 11)
        self.assertEqual(document["summary"]["policy_bindings"], 15)

        stale = json.loads(json.dumps(document))
        for record in stale["layers"]["4_contracts_and_policies"]["policies"]:
            if record["id"] == "rl.set_table.pick.all":
                record["contracts"].pop()
        with self.assertRaisesRegex(ValueError, "contracts view is stale"):
            LibraryCatalog.from_dict(stale)

        dangling = json.loads(json.dumps(document))
        dangling["layers"]["4_contracts_and_policies"]["contracts"][0][
            "policies"
        ].append("vla.manipulation")
        with self.assertRaisesRegex(ValueError, "unknown policies"):
            LibraryCatalog.from_dict(dangling)
