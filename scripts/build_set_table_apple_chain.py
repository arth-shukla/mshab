#!/usr/bin/env python3
"""Derive a minimal executable apple chain from an official SetTable plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_TYPES = ["navigate", "open", "navigate", "pick", "navigate", "place"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan-index", type=int, default=0)
    args = parser.parse_args()

    data = json.loads(args.source.read_text())
    source_plan = data["plans"][args.plan_index]

    # Official SetTable plans first handle the bowl (0:8), then the apple.
    # Keep the apple segment through Place, excluding the final Navigate/Close.
    subtasks = source_plan["subtasks"][8:14]
    actual_types = [subtask["type"] for subtask in subtasks]
    if actual_types != EXPECTED_TYPES:
        raise ValueError(
            f"unexpected official SetTable layout: {actual_types}; "
            f"expected {EXPECTED_TYPES}"
        )
    for subtask in subtasks:
        obj_id = subtask.get("obj_id")
        if obj_id is not None and not obj_id.startswith("013_apple-"):
            raise ValueError(f"non-apple object in extracted chain: {obj_id}")

    output = {
        "dataset": data["dataset"],
        "plans": [
            {
                "subtasks": subtasks,
                "build_config_name": source_plan["build_config_name"],
                "init_config_name": source_plan["init_config_name"],
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)
    print(" -> ".join(EXPECTED_TYPES))


if __name__ == "__main__":
    main()
