"""Prompts and compact graph views for the DeepSeek planning experiment.

The planner never receives ground truth.  It selects existing semantic
strategies and supplies bindings; the runner compiles those decisions through
the authoritative graph instead of trusting the model to invent SkillNodes,
Contracts, Policies, or edges.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence


DECISION_SCHEMA_VERSION = "mshab.vlm-strategy-decisions.v1"
FLOW_SCHEMA_VERSION = "mshab.execution-flow.v2"
JUDGE_SCHEMA_VERSION = "mshab.vlm-plan-judge.v1"
PATH_DECISION_SCHEMA_VERSION = "mshab.vlm-skill-node-path.v1"


DECISION_EXAMPLE: Dict[str, Any] = {
    "schema_version": DECISION_SCHEMA_VERSION,
    "instruction": "Pick up the visible cracker box.",
    "task_family": "tidy_house",
    "strategy_occurrences": [
        {
            "id": "retrieve_cracker_box",
            "strategy_id": "retrieve_surface",
            "bindings": {"object": "003_cracker_box"},
        },
    ],
    "occurrence_order": ["retrieve_cracker_box"],
    "decision_summary": "Retrieve the visible cracker box from the surface.",
}


JUDGE_EXAMPLE: Dict[str, Any] = {
    "schema_version": JUDGE_SCHEMA_VERSION,
    "verdict": "partial_match",
    "score": 0.5,
    "instruction_assessment": (
        "The prediction and supplied GT only partially satisfy the instruction."
    ),
    "sequence_assessment": "Some nominal steps align and some differ.",
    "grounding_assessment": "Only the available GT fields were compared.",
    "graph_assessment": "All selected graph identifiers are valid.",
    "discrepancies": [],
    "limitations": ["The official GT contains no fallback branches."],
}


PLANNER_SYSTEM_PROMPT = """You are a graph-constrained household robot planner.

Select routes from the supplied four-layer MS-HAB graph for the user's task.
The graph identifier vocabulary is closed: never invent a strategy id. A
strategy occurrence may be reused with different object bindings. Return JSON
only, matching the supplied example exactly at the field level.

Rules:
1. Select the minimum ordered strategy occurrences that fully satisfy the
   instruction.
2. Choose alternatives from their semantic applicability and scene context.
3. Preserve task logic: retrieve before deliver; restore a used container only
   after manipulation is finished.
4. Bind every placeholder required by a selected strategy. Use semantic object
   categories such as 024_bowl, not simulator instance suffixes unless the
   context requires an instance.
5. Do not list SkillNodes, Contracts, Policies, or fallback edges yourself.
   The deterministic compiler derives them from the selected strategies.
6. Do not assume access to ground truth; it is intentionally hidden.
7. Echo the given instruction and task_family exactly in their output fields.
8. The response must be a single valid JSON object with no markdown fences.
"""


JUDGE_SYSTEM_PROMPT = """You are an independent evaluator of an MS-HAB plan.

Compare a compiled prediction with one normalized official MS-HAB TaskPlan.
Use the deterministic metrics as evidence, inspect semantic grounding, and
independently assess whether both the prediction and supplied GT actually
match the natural-language instruction. A prediction-vs-GT exact match does
not prove the instruction was paired with the right GT plan. Return JSON only
in the requested schema. Do not claim that fallback branches have official GT
coverage: official TaskPlan files contain nominal linear subtasks only. The
score must be a number from 0 to 1.
"""


PATH_PLANNER_SYSTEM_PROMPT = """You are a graph-constrained household robot planner.

Return an explicit ordered SkillNode path. Do not return a high-level strategy
id and do not invent SkillNodes. Each segment achieves exactly one supplied
SubGoal. The same SkillNode may appear in separate segments with different
object bindings.

Rules:
1. Use only SkillNode ids and SubGoal ids in the supplied library view.
2. Supply exactly the placeholders required by the selected nodes. Fixed
   arguments such as fridge and kitchen_counter are already stored in nodes.
3. When graph edges are supplied, follow ENABLES edges inside each segment.
4. If failed_skill_node_id is non-null, replace that primary route with its
   FALLBACK_TO recovery route and include the recovery prerequisite chain.
5. Order segments to satisfy the instruction. Layer-1 SubGoals intentionally
   have no ordering edges, so their task-level order comes from the instruction.
6. Return JSON only. Echo instruction, task_family, condition, and
   failed_skill_node_id exactly.
"""


PATH_DECISION_EXAMPLE: Dict[str, Any] = {
    "schema_version": PATH_DECISION_SCHEMA_VERSION,
    "instruction": "Retrieve the bowl from the drawer.",
    "task_family": "set_table",
    "condition": "full_graph",
    "failed_skill_node_id": None,
    "segments": [
        {
            "id": "retrieve_bowl",
            "subgoal_id": "retrieve",
            "bindings": {"object": "024_bowl"},
            "skill_node_path": [
                "navigate_drawer_source",
                "open_drawer_source",
                "navigate_drawer_object",
                "pick_drawer_object",
            ],
        }
    ],
    "segment_order": ["retrieve_bowl"],
    "decision_summary": "Use the drawer retrieval route.",
}


def build_path_planner_messages(
    *,
    instruction: str,
    task_family: str,
    condition: str,
    failed_skill_node_id: Any,
    library_view: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Build an ablation prompt whose output is an explicit SkillNode path."""

    if condition not in ("flat_library", "full_graph"):
        raise ValueError("unknown path-planning condition")
    user_content = """Construct the complete SkillNode execution path.

INSTRUCTION:
{instruction}

TASK FAMILY:
{task_family}

CONDITION:
{condition}

FAILED SKILL NODE (null means nominal execution):
{failed}

LIBRARY VIEW:
{library}

OUTPUT EXAMPLE (shape only; solve the supplied instruction):
{example}
""".format(
        instruction=instruction,
        task_family=task_family,
        condition=condition,
        failed=json.dumps(failed_skill_node_id),
        library=_json(library_view),
        example=_json(PATH_DECISION_EXAMPLE),
    )
    return [
        {"role": "system", "content": PATH_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def compact_graph_for_prompt(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove only duplicated derived views while retaining all four layers."""

    if document.get("scope") != "layers_1_through_4":
        raise ValueError("the VLM experiment requires a four-layer graph")
    layer2 = document["layer2"]
    return {
        "schema_version": document["schema_version"],
        "granularity": document["granularity"],
        "semantics": document["semantics"],
        "layer1": document["layer1"],
        "layer2": {
            "subgraphs": layer2["subgraphs"],
            "semantic_strategies": layer2["semantic_strategies"],
        },
        "layer2_to_layer3": document["layer2_to_layer3"],
        "layer3": document["layer3"],
        "layer4": document["layer4"],
        "layer4_to_layer3": document["layer4_to_layer3"],
    }


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def build_planner_messages(
    *,
    instruction: str,
    task_family: str,
    scene_context: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Build the first-call messages; no GT sequence is accepted here."""

    if not instruction.strip():
        raise ValueError("instruction must be non-empty")
    user_content = """Plan this instruction using the supplied graph.

INSTRUCTION:
{instruction}

TASK FAMILY:
{task_family}

USER-SUPPLIED SCENE CONTEXT (may be empty; never derived from GT):
{scene_context}

FOUR-LAYER GRAPH JSON:
{graph}

EXAMPLE JSON OUTPUT SHAPE:
{example}
""".format(
        instruction=instruction,
        task_family=task_family,
        scene_context=_json(scene_context),
        graph=_json(compact_graph_for_prompt(graph)),
        example=_json(DECISION_EXAMPLE),
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_judge_messages(
    *,
    prediction: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    deterministic_metrics: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Build the second-call messages after planning has completed."""

    user_content = """Compare the prediction with the official normalized GT.

COMPILED PREDICTION JSON:
{prediction}

NORMALIZED OFFICIAL GT JSON:
{ground_truth}

DETERMINISTIC METRICS JSON:
{metrics}

EXAMPLE JSON OUTPUT SHAPE:
{example}
""".format(
        prediction=_json(prediction),
        ground_truth=_json(ground_truth),
        metrics=_json(deterministic_metrics),
        example=_json(JUDGE_EXAMPLE),
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def message_text(messages: Sequence[Mapping[str, str]]) -> str:
    """Convenience helper used by tests and dry-run inspection."""

    return "\n".join(message["content"] for message in messages)
