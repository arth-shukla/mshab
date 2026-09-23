#!/usr/bin/env python3
"""Generate the complete manual SetTable four-layer graph as JSON and SVG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from html import escape
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mshab.skills import (
    CheckpointPolicy,
    LibraryCatalog,
    SkillPlanner,
    build_set_table_stack,
)


DEFAULT_ASSET_ROOT = Path(
    os.environ.get("MS_ASSET_DIR", str(REPO_ROOT.parent / "mshab-assets"))
).expanduser()
DEFAULT_CHECKPOINT_ROOT = DEFAULT_ASSET_ROOT / "data" / "mshab_checkpoints"
DEFAULT_JSON = REPO_ROOT / "mshab" / "skills" / "catalogs" / "set_table.json"
DEFAULT_SVG = REPO_ROOT / "docs" / "static" / "images" / "set_table_skill_graph.svg"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    return parser.parse_args()


def predicates_dict(grounded):
    return {
        "preconditions": list(grounded.preconditions),
        "effects": list(grounded.effects),
        "invariants": list(grounded.invariants),
        "verification": list(grounded.verification),
        "failure_modes": list(grounded.failure_modes),
        "deletes": list(grounded.deletes),
    }


def relative_path(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def graph_document(stack, checkpoint_root):
    grounded_skills = stack.grounded_skills

    planner = SkillPlanner(stack.subgoal_graph, stack.skill_graph)
    nominal = planner.plan()
    # Fail only the primaries that actually declare a fallback; a sub-goal whose
    # single achiever fails has no recovery and correctly aborts the plan.
    recoverable = sorted(
        nominal.selections[subgoal_id]
        for subgoal_id, subgraph in stack.skill_graph.subgraphs.items()
        if len(planner.fallback_chain(subgraph)) > 1
    )
    recovery = planner.plan(failed=recoverable)
    layer_3 = {
        node_id: {
            "contract_id": node.contract_id,
            "arguments": dict(node.arguments),
            **predicates_dict(grounded_skills[node_id]),
        }
        for node_id, node in sorted(stack.skill_graph.nodes.items())
    }

    # Layer 4 is stored from both sides of the many-to-many binding: each
    # contract lists its policies in preference order, each policy lists the
    # contracts it executes.  Local ready/missing state is deliberately absent.
    library = stack.library
    contract_records = [
        {
            "id": contract.id,
            "contract_type": contract.contract_type_name,
            "target": contract.target,
            "env_id": contract.env_id,
            "max_episode_steps": contract.max_episode_steps,
            "policies": [
                policy.id for policy in library.policies_for(contract.id)
            ],
        }
        for contract in library.find(task="set_table")
    ]
    policy_records = []
    for policy in library.policies:
        record = {"id": policy.id, "kind": policy.kind.value}
        if isinstance(policy, CheckpointPolicy):
            record.update(
                {
                    "family": policy.family,
                    "policy_type": policy.policy_type,
                    "checkpoint": relative_path(
                        policy.checkpoint_path, checkpoint_root
                    ),
                    "config": relative_path(policy.config_path, checkpoint_root),
                }
            )
        record["contracts"] = [
            contract.id for contract in library.contracts_for(policy.id)
        ]
        policy_records.append(record)

    return LibraryCatalog(
        task="set_table",
        construction={
            "method": "manual",
            "builder": "SetTableGraphBuilder",
            "scene_independent_layers": [1, 2],
            "environment_specific_layers": [3, 4],
            "official_primary_sequence_per_object": [
                "navigate",
                "open",
                "navigate",
                "pick",
                "navigate",
                "place",
                "navigate",
                "close",
            ],
            "object_order": ["024_bowl", "013_apple"],
        },
        execution_plan_note=(
            "One achiever per sub-goal, chosen from the candidate "
            "graph. The recovery plan is the same graph re-decided after "
            "every primary achiever is reported failed."
        ),
        execution_plans={
            "nominal": nominal,
            "recovery_all_primaries_failed": recovery,
        },
        subgoal_graph=stack.subgoal_graph,
        skill_graph=stack.skill_graph,
        grounded_skills=layer_3,
        contracts=contract_records,
        policies=policy_records,
    ).as_dict()


def text(x, y, value, css_class="text", anchor="middle"):
    return '<text class="{}" x="{}" y="{}" text-anchor="{}">{}</text>'.format(
        css_class, x, y, anchor, escape(value)
    )


def node(x, y, title, subtitle, color, width=190, height=72):
    return [
        '<rect class="node" x="{}" y="{}" width="{}" height="{}" '
        'fill="{}"/>'.format(x, y, width, height, color),
        text(x + width // 2, y + 29, title, "node-title"),
        text(x + width // 2, y + 53, subtitle, "small"),
    ]


def arrow(x1, y1, x2, y2, css_class="edge"):
    return (
        '<path class="{}" d="M {} {} L {} {}" marker-end="url(#arrow)"/>'.format(
            css_class, x1, y1, x2, y2
        )
    )


def set_table_svg(document):
    width, height = 1900, 1510
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}" role="img">'.format(width, height, width, height),
        '<title>Complete MS-HAB SetTable four-layer skill graph</title>',
        '<desc>Bowl and apple SetTable skill composition with contracts and '
        'the downloaded policies that execute them.</desc>',
        """<defs>
          <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3"
                  orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#334155"/>
          </marker>
          <style>
            .layer { stroke-width: 2; rx: 16; }
            .title { font: 700 27px Arial, sans-serif; fill: #0f172a; }
            .subtitle { font: 700 17px Arial, sans-serif; fill: #475569; }
            .text { font: 15px Arial, sans-serif; fill: #1e293b; }
            .small { font: 13px Arial, sans-serif; fill: #475569; }
            .tiny { font: 12px Arial, sans-serif; fill: #475569; }
            .node { stroke: #64748b; stroke-width: 1.5; rx: 10; }
            .subgraph { stroke: #16a34a; stroke-width: 2; rx: 12; }
            .node-title { font: 700 15px Arial, sans-serif; fill: #0f172a; }
            .edge { fill: none; stroke: #334155; stroke-width: 2; }
            .subgoal-edge { fill: none; stroke: #2563eb; stroke-width: 2; }
            .relation { fill: none; stroke: #7c3aed; stroke-width: 2;
                        stroke-dasharray: 6 4; }
          </style>
        </defs>""",
        '<rect width="1900" height="1510" fill="#f8fafc"/>',
    ]

    # Layer 1: eight ordered sub-goals.
    parts.append(
        '<rect class="layer" x="24" y="20" width="1852" height="240" '
        'fill="#eaf4ff" stroke="#3b82f6"/>'
    )
    parts.append(text(950, 58, "1. Sub-goal Graph", "title"))
    parts.append(text(1815, 56, "SCENE-INDEPENDENT", "subtitle", "end"))
    subgoal_labels = [
        ("Open counter", "open(kitchen_counter)"),
        ("Retrieve bowl", "holding(024_bowl)"),
        ("Place bowl", "at(024_bowl,dining_table)"),
        ("Close counter", "closed(kitchen_counter)"),
        ("Open fridge", "open(fridge)"),
        ("Retrieve apple", "holding(013_apple)"),
        ("Place apple", "at(013_apple,dining_table)"),
        ("Close fridge", "closed(fridge)"),
    ]
    subgoal_x = [55 + index * 228 for index in range(8)]
    for index, ((title, subtitle), x) in enumerate(zip(subgoal_labels, subgoal_x)):
        parts.extend(node(x, 105, title, subtitle, "#ffffff", width=196, height=78))
        if index:
            parts.append(arrow(subgoal_x[index - 1] + 196, 144, x - 6, 144, "subgoal-edge"))

    # Layer 2: every sub-goal owns one explicit implementation subgraph.
    parts.append(
        '<rect class="layer" x="24" y="285" width="1852" height="670" '
        'fill="#ecfdf3" stroke="#22c55e"/>'
    )
    parts.append(
        text(950, 325, "2. Sub-goal-owned Candidate Skill Subgraphs", "title")
    )
    parts.append(
        text(1815, 323, "SCENE-INDEPENDENT · MANUAL V1", "subtitle", "end")
    )

    lanes = [
        ("BOWL", 365, "bowl", "kitchen_counter", "024_bowl", 1),
        ("APPLE", 670, "apple", "fridge", "013_apple", 9),
    ]
    box_x = [55, 505, 955, 1405]
    for lane_index, (lane, y, label, source, obj, start) in enumerate(lanes):
        parts.append(text(55, y - 12, "{} subgraphs".format(lane), "subtitle", "start"))
        specs = [
            (
                "{}_source_open".format(label),
                ("{} Navigate".format(start), source, "#dbeafe"),
                ("{} Open".format(start + 1), source, "#fef3c7"),
                None,
            ),
            (
                "{}_retrieved".format(label),
                ("{} Navigate".format(start + 2), obj, "#dbeafe"),
                ("{} Pick".format(start + 3), obj, "#dcfce7"),
                ("Pick all", "alternative / fallback", "#ede9fe"),
            ),
            (
                "{}_placed".format(label),
                ("{} Navigate".format(start + 4), "dining_table", "#dbeafe"),
                ("{} Place".format(start + 5), obj, "#fce7f3"),
                ("Place all", "alternative / fallback", "#ede9fe"),
            ),
            (
                "{}_source_closed".format(label),
                ("{} Navigate".format(start + 6), source, "#dbeafe"),
                ("{} Close".format(start + 7), source, "#ffedd5"),
                None,
            ),
        ]
        for index, (subgoal_id, first, second, generic) in enumerate(specs):
            x = box_x[index]
            parts.append(
                '<rect class="subgraph" x="{}" y="{}" width="420" height="235" '
                'fill="#f7fff9"/>'.format(x, y, 235)
            )
            parts.append(text(x + 210, y + 27, "Sub-goal: " + subgoal_id, "subtitle"))
            parts.extend(node(x + 18, y + 52, *first, width=175, height=68))
            parts.extend(node(x + 227, y + 52, *second, width=175, height=68))
            parts.append(arrow(x + 193, y + 86, x + 221, y + 86))
            parts.append(text(x + 207, y + 78, "ENABLES", "tiny"))
            if generic:
                parts.extend(node(x + 227, y + 150, *generic, width=175, height=62))
                parts.append(
                    arrow(x + 314, y + 120, x + 314, y + 145, "relation")
                )
                parts.append(
                    text(
                        x + 215,
                        y + 141,
                        "IS_A / ALT / FALLBACK",
                        "tiny",
                        "end",
                    )
                )
            if index:
                parts.append(
                    arrow(box_x[index - 1] + 420, y + 118, x - 6, y + 118)
                )
        if lane_index == 0:
            parts.append(
                '<path class="edge" d="M 1825 483 L 1850 483 L 1850 645 '
                'L 35 645 L 35 788 L 49 788" marker-end="url(#arrow)"/>'
            )
            parts.append(
                text(
                    950,
                    640,
                    "cross-subgraph ENABLES: bowl complete → apple starts",
                    "small",
                )
            )

    legend = (
        "Green boxes: one SkillSubgraph per sub-goal    "
        "Solid: ENABLES (including cross-subgraph)    Dashed: alternative skill nodes"
    )
    parts.append(text(950, 930, legend, "small"))
    parts.append(arrow(950, 955, 950, 978))
    parts.append(
        text(
            970,
            972,
            "future decision VLM: graph + state → one next skill",
            "tiny",
            "start",
        )
    )

    # Layer 3: the 20 graph nodes grounded to their contracts.
    parts.append(
        '<rect class="layer" x="24" y="980" width="1852" height="255" '
        'fill="#fffbeb" stroke="#eab308"/>'
    )
    parts.append(text(950, 1020, "3. Grounded Skills", "title"))
    parts.append(text(1815, 1018, "ENVIRONMENT-SPECIFIC", "subtitle", "end"))
    contract_specs = [
        ("Navigate", "pre: present(goal)", "effect: reachable(goal)"),
        ("Open", "pre: reachable + closed", "effect: open(articulation)"),
        ("Pick", "pre: reachable + empty", "effect: holding(object)"),
        ("Place", "pre: holding + reachable", "effect: at(object,destination)"),
        ("Close", "pre: reachable + open", "effect: closed(articulation)"),
    ]
    for index, (title, precondition, effect) in enumerate(contract_specs):
        x = 55 + index * 365
        parts.extend(
            node(
                x,
                1060,
                title,
                precondition,
                "#ffffff",
                width=325,
                height=120,
            )
        )
        parts.append(text(x + 162, 1144, effect, "small"))
        parts.append(text(x + 162, 1165, "physical feasibility only: verify effect + collision invariant", "tiny"))

    # Layer 4: all 11 contracts, each with the policies bound to execute it.
    parts.append(
        '<rect class="layer" x="24" y="1260" width="1852" height="225" '
        'fill="#f5f3ff" stroke="#8b5cf6"/>'
    )
    parts.append(
        text(950, 1300, "4. Contracts and the Policies Bound to Them", "title")
    )
    parts.append(text(1815, 1298, "ENVIRONMENT-SPECIFIC", "subtitle", "end"))
    layer_4 = document["layers"]["4_contracts_and_policies"]
    for index, contract in enumerate(layer_4["contracts"]):
        row, column = divmod(index, 6)
        x = 55 + column * 300
        y = 1330 + row * 70
        label = contract["id"].replace("mshab.set_table.", "")
        bound = " + ".join(
            policy_id.replace("set_table.", "") for policy_id in contract["policies"]
        )
        parts.extend(node(x, y, label, bound, "#ffffff", width=265, height=56))
    summary = document["summary"]
    parts.append(
        text(
            950,
            1476,
            "{} contracts · {} policies · {} bindings (many-to-many): an all-object "
            "checkpoint is also bound to every specialized contract of its type".format(
                summary["registered_contracts"],
                summary["registered_policies"],
                summary["policy_bindings"],
            ),
            "small",
        )
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    args = parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    stack = build_set_table_stack(checkpoint_root)
    document = graph_document(stack, checkpoint_root)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(document, indent=2) + "\n")
    print("wrote JSON:", args.json)

    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(set_table_svg(document))
    print("wrote SVG:", args.svg)


if __name__ == "__main__":
    main()
