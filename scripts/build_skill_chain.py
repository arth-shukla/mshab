#!/usr/bin/env python3
"""Build a reusable skill chain by selecting subtasks from an MS-HAB TaskPlan.

The output remains an ordinary MS-HAB PlanData JSON: scene, episode, grounded
objects, goal poses, and articulation parameters are copied from the official
source plan. This tool changes composition only; it does not invent grounding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUBTASK_HORIZONS = {"navigate": 500, "pick": 200, "place": 200, "open": 200, "close": 200}


def parse_selection(spec: str, count: int) -> list[int]:
    """Parse either a Python-like half-open range (``8:14``) or ``1,3,5``."""
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"invalid range {spec!r}; use start:end[:step]")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else count
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if step <= 0:
            raise ValueError("selection step must be positive")
        indices = list(range(start, end, step))
    else:
        indices = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not indices:
        raise ValueError("selection produced no subtasks")
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise IndexError(f"subtask indices out of range for {count} subtasks: {invalid}")
    if indices != sorted(set(indices)):
        raise ValueError("subtask indices must be unique and increasing")
    return indices


def describe(index: int, subtask: dict[str, Any]) -> str:
    target = subtask.get("obj_id") or subtask.get("articulation_type") or "-"
    return f"{index:>2}  {subtask['type']:<8}  {target}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="official sequential task-plan JSON")
    parser.add_argument("output", type=Path, help="derived PlanData JSON")
    parser.add_argument("--plan-index", type=int, default=0)
    parser.add_argument(
        "--select",
        required=True,
        help="half-open range such as 8:14, or increasing indices such as 0,1,4,5",
    )
    parser.add_argument(
        "--expect-types",
        help="optional comma-separated guard, e.g. navigate,open,navigate,pick",
    )
    args = parser.parse_args()

    data = json.loads(args.source.read_text())
    plans = data.get("plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError(f"{args.source} contains no plans")
    if args.plan_index < 0 or args.plan_index >= len(plans):
        raise IndexError(f"plan index {args.plan_index} outside [0, {len(plans)})")

    source_plan = plans[args.plan_index]
    source_subtasks = source_plan["subtasks"]
    indices = parse_selection(args.select, len(source_subtasks))
    subtasks = [source_subtasks[index] for index in indices]
    actual_types = [subtask["type"] for subtask in subtasks]

    if args.expect_types:
        expected = [item.strip() for item in args.expect_types.split(",") if item.strip()]
        if actual_types != expected:
            raise ValueError(f"selected types {actual_types}; expected {expected}")

    output = {
        "dataset": data["dataset"],
        "plans": [
            {
                "subtasks": subtasks,
                "build_config_name": source_plan["build_config_name"],
                "init_config_name": source_plan["init_config_name"],
            }
        ],
        "selection": {
            "source": str(args.source),
            "source_plan_index": args.plan_index,
            "source_subtask_indices": indices,
            "estimated_max_episode_steps": sum(SUBTASK_HORIZONS[t] for t in actual_types),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    print(f"wrote {args.output}")
    print("source index / skill / grounded target")
    for index, subtask in zip(indices, subtasks):
        print(describe(index, subtask))
    print(f"estimated_max_episode_steps={output['selection']['estimated_max_episode_steps']}")


if __name__ == "__main__":
    main()
