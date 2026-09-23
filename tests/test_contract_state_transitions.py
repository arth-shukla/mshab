"""Contracts as state transitions: what a contract asserts *and* what it retracts.

Without delete effects the symbolic state is monotone, so a SetTable rollout
ends with ``open(kitchen_counter)`` and ``closed(kitchen_counter)`` both true
and the bowl still ``holding`` after it was placed.  These tests pin the
transition semantics that ``GroundedSkill.deletes``/``apply_to`` restore.
"""

from __future__ import annotations

from unittest import TestCase

from mshab.skills import (
    CloseContract,
    OpenContract,
    PickContract,
    PlaceContract,
    Contract,
    ContractParameter,
    ParameterType,
    SkillPlanner,
    build_set_table_graph,
)
from mshab.skills.library import ContractLibrary
from mshab.skills.model import NavigateContract


def _set_table_library():
    library = ContractLibrary()
    library.register(NavigateContract("set_table"))
    for target in ("024_bowl", "013_apple"):
        library.register(PickContract("set_table", target))
        library.register(PlaceContract("set_table", target))
    library.register(PickContract("set_table", "all"))
    library.register(PlaceContract("set_table", "all"))
    for target in ("kitchen_counter", "fridge"):
        library.register(OpenContract("set_table", target))
        library.register(CloseContract("set_table", target))
    return library


def _probe(**predicates):
    """A minimal generic contract for exercising predicate validation."""

    return Contract(
        "probe",
        "set_table",
        "all",
        target_parameter="object",
        parameters=(ContractParameter("object", ParameterType.ENTITY),),
        preconditions=(),
        env_id="Probe-v0",
        max_episode_steps=1,
        **predicates
    )


class DeleteEffectTests(TestCase):
    def test_open_and_close_retract_each_other(self):
        opened = OpenContract("set_table", "fridge").bind({})
        closed = CloseContract("set_table", "fridge").bind({})

        after_open = opened.apply_to({"closed(fridge)", "reachable(fridge)"})
        after_close = closed.apply_to(after_open)

        self.assertIn("open(fridge)", after_open)
        self.assertNotIn("closed(fridge)", after_open)
        self.assertIn("closed(fridge)", after_close)
        self.assertNotIn("open(fridge)", after_close)

    def test_place_releases_the_held_object(self):
        contract = (
            PlaceContract("set_table", "024_bowl")
            .bind({"destination": "dining_table"})
        )

        after = contract.apply_to({"holding(024_bowl)", "reachable(dining_table)"})

        self.assertIn("at(024_bowl,dining_table)", after)
        self.assertIn("gripper_empty()", after)
        self.assertNotIn("holding(024_bowl)", after)

    def test_pick_consumes_the_empty_gripper(self):
        contract = PickContract("set_table", "024_bowl").bind({})

        after = contract.apply_to({"gripper_empty()", "reachable(024_bowl)"})

        self.assertIn("holding(024_bowl)", after)
        self.assertNotIn("gripper_empty()", after)

    def test_a_predicate_cannot_be_both_asserted_and_retracted(self):
        with self.assertRaisesRegex(ValueError, "asserted and retracted"):
            _probe(effects=("holding({object})",), deletes=("holding({object})",))

    def test_a_skill_cannot_retract_its_own_invariant(self):
        with self.assertRaisesRegex(ValueError, "retract its own invariants"):
            _probe(
                effects=("holding({object})",),
                invariants=("collision_safe()",),
                deletes=("collision_safe()",),
            )

    def test_grounding_cannot_create_an_assert_delete_conflict(self):
        contract = _probe(
            effects=("holding({object})",), deletes=("holding(013_apple)",)
        )

        with self.assertRaisesRegex(ValueError, "grounded predicates"):
            contract.bind({"object": "013_apple"})


class SetTableStateTransitionTests(TestCase):
    """The full 16-step plan must stay symbolically consistent throughout."""

    INITIAL = frozenset(
        {
            "present(kitchen_counter)",
            "present(024_bowl)",
            "present(dining_table)",
            "present(fridge)",
            "present(013_apple)",
            "closed(kitchen_counter)",
            "closed(fridge)",
            "gripper_empty()",
            "collision_safe()",
        }
    )

    def _rollout(self, failed=()):
        """Replay a plan symbolically, returning the final and per-goal states."""

        subgoals, graph = build_set_table_graph()
        library = _set_table_library()
        plan = SkillPlanner(subgoals, graph).plan(failed=failed)
        facts = set(self.INITIAL)
        at_goal_completion = {}
        for node_id in plan.order:
            node = graph.nodes[node_id]
            grounded = library.get(node.contract_id).bind(node.arguments)
            missing = [item for item in grounded.preconditions if item not in facts]
            self.assertEqual(
                missing, [], "{} cannot start: missing {}".format(node_id, missing)
            )
            self.assertTrue(
                set(grounded.invariants).issubset(facts),
                "{} invariants do not hold".format(node_id),
            )
            facts = set(grounded.apply_to(facts))
            for subgoal_id in node.achieves:
                at_goal_completion[subgoal_id] = frozenset(facts)
        return subgoals, facts, at_goal_completion

    def test_every_goal_predicate_holds_when_its_achiever_completes(self):
        # open(kitchen_counter) and holding(bowl) are transient: they are true
        # when their subgraph finishes, and correctly retracted later on.
        subgoals, _, at_goal_completion = self._rollout()

        self.assertEqual(set(at_goal_completion), set(subgoals.subgoals))
        for goal in subgoals.subgoals.values():
            self.assertIn(goal.predicate, at_goal_completion[goal.id], goal.id)
            self.assertTrue(
                subgoals.achieved(goal.id, at_goal_completion[goal.id]), goal.id
            )

    def test_nominal_plan_leaves_no_contradictory_state(self):
        _, facts, _ = self._rollout()

        for container in ("kitchen_counter", "fridge"):
            self.assertIn("closed({})".format(container), facts)
            self.assertNotIn("open({})".format(container), facts)
        for obj in ("024_bowl", "013_apple"):
            self.assertNotIn("holding({})".format(obj), facts)
        self.assertIn("gripper_empty()", facts)

    def test_fallback_rollout_is_symbolically_identical_to_the_nominal_one(self):
        _, nominal, _ = self._rollout()
        _, recovered, _ = self._rollout(
            failed=(
                "pick_bowl_specialized",
                "place_bowl_specialized",
                "pick_apple_specialized",
                "place_apple_specialized",
            )
        )

        self.assertEqual(nominal, recovered)
