#!/usr/bin/env python3
"""Ground one catalog execution plan with an official MS-HAB SetTable plan.

Layer 1/2 decide the semantic node sequence.  The official MS-HAB PlanData
supplies scene-specific object instances, articulation ids, and goal poses that
do not belong in the scene-independent graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mshab.skills import LibraryCatalog


SUBTASK_HORIZONS = {
    "navigate": 500,
    "pick": 200,
    "place": 200,
    "open": 200,
    "close": 200,
}
DEFAULT_CATALOG = REPO_ROOT / "mshab" / "skills" / "catalogs" / "set_table.json"


def _semantic_target(
    node: Any, contract_record: Mapping[str, Any], contract_type: str
) -> str:
    argument_name = {
        "pick": "object",
        "place": "object",
        "open": "articulation",
        "close": "articulation",
        "navigate": "goal",
    }[contract_type]
    return str(node.arguments.get(argument_name, contract_record["target"]))


def _validate_grounding(
    index: int,
    node: Any,
    contract_record: Mapping[str, Any],
    subtask: Mapping[str, Any],
) -> Dict[str, str]:
    contract_type = str(contract_record["contract_type"])
    if subtask.get("type") != contract_type:
        raise ValueError(
            "graph node {} is {!r}, but source subtask {} is {!r}".format(
                node.id, contract_type, index, subtask.get("type")
            )
        )
    target = _semantic_target(node, contract_record, contract_type)
    if contract_type in ("pick", "place"):
        grounded = str(subtask.get("obj_id", ""))
        if not grounded.startswith(target + "-"):
            raise ValueError(
                "graph node {} targets {!r}, source subtask {} grounds {!r}".format(
                    node.id, target, index, grounded
                )
            )
    elif contract_type in ("open", "close"):
        grounded = str(subtask.get("articulation_type", ""))
        if grounded != target:
            raise ValueError(
                "graph node {} targets {!r}, source subtask {} grounds {!r}".format(
                    node.id, target, index, grounded
                )
            )
    return {"node_id": node.id, "contract_id": node.contract_id, "target": target}


def build_plan_data(
    catalog: LibraryCatalog,
    source: Mapping[str, Any],
    *,
    execution_plan: str,
    source_plan_index: int,
) -> Dict[str, Any]:
    try:
        plan = catalog.execution_plans[execution_plan]
    except KeyError as exc:
        raise KeyError(
            "unknown execution plan {!r}; available={}".format(
                execution_plan, sorted(catalog.execution_plans)
            )
        ) from exc
    source_plans = source.get("plans")
    if not isinstance(source_plans, list) or not source_plans:
        raise ValueError("source PlanData contains no plans")
    if source_plan_index < 0 or source_plan_index >= len(source_plans):
        raise IndexError("source plan index is out of range")
    source_plan = source_plans[source_plan_index]
    subtasks = source_plan["subtasks"]
    if len(subtasks) != len(plan.order):
        raise ValueError(
            "graph plan has {} nodes but source plan has {} subtasks".format(
                len(plan.order), len(subtasks)
            )
        )

    contracts_by_id = {
        record["id"]: record for record in catalog.contracts
    }
    decisions = []
    uses_all_object_policy = False
    for index, (node_id, subtask) in enumerate(zip(plan.order, subtasks)):
        node = catalog.skill_graph.nodes[node_id]
        try:
            contract_record = contracts_by_id[node.contract_id]
        except KeyError as exc:
            raise KeyError("catalog has no contract record for {}".format(node.contract_id)) from exc
        decisions.append(_validate_grounding(index, node, contract_record, subtask))
        uses_all_object_policy |= (
            contract_record["contract_type"] in ("pick", "place")
            and contract_record["target"] == "all"
        )

    recommended_policy = "rl_all_obj" if uses_all_object_policy else "rl_per_obj"
    return {
        "dataset": source["dataset"],
        "plans": [
            {
                "subtasks": subtasks,
                "build_config_name": source_plan["build_config_name"],
                "init_config_name": source_plan["init_config_name"],
            }
        ],
        "selection": {
            "source_plan_index": source_plan_index,
            "execution_plan": execution_plan,
            "skill_decisions": decisions,
            "recommended_policy_type": recommended_policy,
            "estimated_max_episode_steps": sum(
                SUBTASK_HORIZONS[subtask["type"]] for subtask in subtasks
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="official sequential all.json")
    parser.add_argument("output", type=Path, help="grounded MS-HAB PlanData JSON")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--execution-plan", default="nominal")
    parser.add_argument("--source-plan-index", type=int, default=0)
    args = parser.parse_args()

    catalog = LibraryCatalog.from_dict(json.loads(args.catalog.read_text()))
    source = json.loads(args.source.read_text())
    output = build_plan_data(
        catalog,
        source,
        execution_plan=args.execution_plan,
        source_plan_index=args.source_plan_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    print("wrote {}".format(args.output))
    print("graph decision / contract / grounded target")
    for index, decision in enumerate(output["selection"]["skill_decisions"]):
        print(
            "{:>2}  {:<38}  {:<38}  {}".format(
                index,
                decision["node_id"],
                decision["contract_id"],
                decision["target"],
            )
        )
    print(
        "recommended_policy_type={}".format(
            output["selection"]["recommended_policy_type"]
        )
    )


if __name__ == "__main__":
    main()
