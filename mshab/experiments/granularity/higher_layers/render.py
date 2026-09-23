"""Render the coarse graph with one box per real SkillNode.

The SVG is a direct view of the OOP graph. It never turns a whole semantic
strategy into a pseudo-node: strategy records only define visual lanes, while
SkillSubgraph edges define every sequential, alternative, and fallback arrow.
Each SkillNode also contributes exactly one reference edge to Layer 3.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
from xml.sax.saxutils import escape

from mshab.experiments.granularity.higher_layers.coarse import (
    build_coarse_higher_layers,
    coarse_higher_layers_document,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = PACKAGE_DIR / "artifacts"

SVG_WIDTH = 3000
SVG_HEIGHT = 1940
FOUR_LAYER_SVG_HEIGHT = 2520
PANEL_Y = 390
PANEL_WIDTH = 920
PANEL_HEIGHT = 980
PANEL_X = {
    "retrieve": 40,
    "deliver": 1040,
    "restore": 2040,
}
SUBGOAL_ORDER = ("retrieve", "deliver", "restore")
CONTRACT_ORDER = ("navigate", "pick", "place", "open", "close")

NODE_WIDTH = 140
NODE_HEIGHT = 54
NODE_SLOT_X = (75, 270, 465, 660)
STRATEGY_FIRST_Y = 450
STRATEGY_HEIGHT = 205
STRATEGY_GAP = 15


@dataclass(frozen=True)
class NodeBox:
    """Deterministic SVG position for one real Layer-2 SkillNode."""

    node_id: str
    contract_id: str
    strategy_id: str
    x: float
    y: float
    width: float = NODE_WIDTH
    height: float = NODE_HEIGHT

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class ContractBox:
    """Position of one shared Layer-3 Contract node."""

    contract_id: str
    contract_type: str
    x: float
    y: float
    width: float = 400
    height: float = 96

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class PolicyBox:
    """Deterministic SVG position for one existing Layer-4 Policy."""

    policy_id: str
    contract_id: str
    short_id: str
    x: float
    y: float
    width: float
    height: float = 21

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def top(self) -> float:
        return self.y


def _text(
    x: float,
    y: float,
    value: object,
    *,
    size: float = 16,
    weight: int = 400,
    fill: str = "#2f3a4d",
    anchor: str = "start",
) -> str:
    return (
        '<text x="{x}" y="{y}" font-size="{size}" '
        'font-weight="{weight}" fill="{fill}" '
        'text-anchor="{anchor}">{value}</text>'
    ).format(
        x=x,
        y=y,
        size=size,
        weight=weight,
        fill=fill,
        anchor=anchor,
        value=escape(str(value)),
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str,
    radius: float = 12,
    dashed: bool = False,
    extra: str = "",
) -> str:
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    return (
        '<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        'rx="{radius}" fill="{fill}" stroke="{stroke}" '
        'stroke-width="2"{dash}{extra}/>'
    ).format(
        x=x,
        y=y,
        width=width,
        height=height,
        radius=radius,
        fill=fill,
        stroke=stroke,
        dash=dash,
        extra=extra,
    )


def _path(
    path_data: str,
    *,
    color: str,
    marker: str = "",
    dashed: bool = False,
    opacity: float = 1.0,
    width: float = 2.2,
    extra: str = "",
) -> str:
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    marker_end = (
        ' marker-end="url(#{})"'.format(marker) if marker else ""
    )
    return (
        '<path d="{path_data}" fill="none" stroke="{color}" '
        'stroke-width="{width}" stroke-opacity="{opacity}"'
        '{dash}{marker_end}{extra}/>'
    ).format(
        path_data=path_data,
        color=color,
        width=width,
        opacity=opacity,
        dash=dash,
        marker_end=marker_end,
        extra=extra,
    )


def _marker(marker_id: str, color: str) -> str:
    return (
        '<marker id="{marker_id}" viewBox="0 0 10 10" refX="9" '
        'refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/>'
        "</marker>"
    ).format(marker_id=marker_id, color=color)


def _contract_type(contract_id: str) -> str:
    return contract_id.split(".")[-2]


def _semantic_node_name(node_id: str) -> str:
    """Turn an action-first semantic id into its compact SVG name."""

    return "".join(token.capitalize() for token in node_id.split("_"))


def _svg_token(value: str) -> str:
    """Return a stable XML id fragment without changing the domain id."""

    return "".join(
        character if character.isalnum() or character in "_-" else "-"
        for character in value
    )


def _subgraph_maps(
    document: Mapping[str, Any],
) -> Tuple[
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    layer2 = document["layer2"]
    subgraphs = {
        item["subgoal_id"]: item for item in layer2["subgraphs"]
    }
    nodes = {item["id"]: item for item in layer2["nodes"]}
    return subgraphs, nodes


def _layout_skill_nodes(
    document: Mapping[str, Any],
) -> Dict[str, NodeBox]:
    _, nodes = _subgraph_maps(document)
    strategies = document["layer2"]["semantic_strategies"]
    boxes: Dict[str, NodeBox] = {}

    for subgoal_id in SUBGOAL_ORDER:
        panel_x = PANEL_X[subgoal_id]
        for index, strategy in enumerate(strategies[subgoal_id]):
            row_y = STRATEGY_FIRST_Y + index * (
                STRATEGY_HEIGHT + STRATEGY_GAP
            )
            node_ids = list(strategy["node_ids"])
            primary_index = node_ids.index(strategy["primary_achiever"])
            main_path = node_ids[: primary_index + 1]
            recovery_path = node_ids[primary_index + 1 :]

            for path_nodes, node_y in (
                (main_path, row_y + 48),
                (recovery_path, row_y + 130),
            ):
                slot_offset = len(NODE_SLOT_X) - len(path_nodes)
                for slot_index, node_id in enumerate(path_nodes):
                    node = nodes[node_id]
                    boxes[node_id] = NodeBox(
                        node_id=node_id,
                        contract_id=node["contract_id"],
                        strategy_id=strategy["id"],
                        x=panel_x + NODE_SLOT_X[slot_offset + slot_index],
                        y=node_y,
                    )

    if set(boxes) != set(nodes):
        missing = sorted(set(nodes) - set(boxes))
        extra = sorted(set(boxes) - set(nodes))
        raise ValueError(
            "SVG layout does not cover SkillNodes: missing={}, extra={}".format(
                missing,
                extra,
            )
        )
    return boxes


def _layout_contracts(
    document: Mapping[str, Any],
) -> Dict[str, ContractBox]:
    contracts = {
        item["contract_type"]: item for item in document["layer3"]["contracts"]
    }
    boxes: Dict[str, ContractBox] = {}
    for index, contract_type in enumerate(CONTRACT_ORDER):
        contract = contracts[contract_type]
        box = ContractBox(
            contract_id=contract["id"],
            contract_type=contract_type,
            x=100 + index * 570,
            y=1660,
        )
        boxes[box.contract_id] = box
    return boxes


def _edge_path(
    source: NodeBox,
    target: NodeBox,
    relation: str,
    panel_right: float,
) -> str:
    if relation == "alternative_to":
        gutter_x = panel_right - 18
        return (
            "M {sx} {sy} H {gx} V {ty} H {tx}"
        ).format(
            sx=source.right,
            sy=source.center_y,
            gx=gutter_x,
            ty=target.center_y,
            tx=target.right,
        )
    if source.y == target.y:
        if source.x < target.x:
            return "M {} {} H {}".format(
                source.right,
                source.center_y,
                target.left,
            )
        return "M {} {} H {}".format(
            source.left,
            source.center_y,
            target.right,
        )
    return (
        "M {sx} {sy} C {sx} {mid}, {tx} {mid}, {tx} {ty}"
    ).format(
        sx=source.center_x,
        sy=source.bottom,
        tx=target.center_x,
        ty=target.top,
        mid=(source.bottom + target.top) / 2,
    )


def _render_skill_edge(
    edge: Mapping[str, Any],
    boxes: Mapping[str, NodeBox],
    panel_right: float,
) -> str:
    relation = edge["relation"]
    styles = {
        "enables": ("#2b965d", "arrow-green", False),
        "requires": ("#2b965d", "arrow-green", False),
        "fallback_to": ("#c43d3d", "arrow-red", True),
        "alternative_to": ("#7354c7", "arrow-purple", True),
        "is_a": ("#66758d", "arrow-gray", True),
    }
    color, marker, dashed = styles[relation]
    marker_start = (
        ' marker-start="url(#arrow-purple)"'
        if relation == "alternative_to"
        else ""
    )
    return _path(
        _edge_path(
            boxes[edge["source"]],
            boxes[edge["target"]],
            relation,
            panel_right,
        ),
        color=color,
        marker=marker,
        dashed=dashed,
        width=2.6,
        extra=(
            ' id="skill-edge-{relation}-{source}-{target}" '
            'data-relation="{relation}"{marker_start}'
        ).format(
            relation=relation,
            source=edge["source"],
            target=edge["target"],
            marker_start=marker_start,
        ),
    )


def _render_node(
    box: NodeBox,
    node: Mapping[str, Any],
) -> List[str]:
    semantic_name = _semantic_node_name(box.node_id)
    font_size = 12 if len(semantic_name) <= 21 else 10
    return [
        _rect(
            box.x,
            box.y,
            box.width,
            box.height,
            fill="#ffffff",
            stroke="#258457",
            radius=10,
            extra=(
                ' id="skill-node-{node_id}" '
                'data-contract-id="{contract_id}" '
                'data-semantic-name="{semantic_name}"'
            ).format(
                node_id=box.node_id,
                contract_id=box.contract_id,
                semantic_name=semantic_name,
            ),
        ),
        _text(
            box.center_x,
            box.y + 33,
            semantic_name,
            size=font_size,
            weight=700,
            fill="#23744e",
            anchor="middle",
        ),
        (
            '<circle cx="{x}" cy="{y}" r="4" fill="#ca5a24" '
            'data-node-contract-port="{node_id}"/>'
        ).format(
            x=box.center_x,
            y=box.bottom,
            node_id=node["id"],
        ),
    ]


def _render_contract_references(
    document: Mapping[str, Any],
    node_boxes: Mapping[str, NodeBox],
    contract_boxes: Mapping[str, ContractBox],
) -> List[str]:
    rail_y = {
        contract_type: 1505 + index * 27
        for index, contract_type in enumerate(CONTRACT_ORDER)
    }
    references = document["layer2_to_layer3"]["references"]
    grouped: Dict[str, List[NodeBox]] = {
        contract_id: [] for contract_id in contract_boxes
    }
    out: List[str] = []
    for reference in references:
        source = node_boxes[reference["source"]]
        target_id = reference["target"]
        grouped[target_id].append(source)
        y = rail_y[_contract_type(target_id)]
        out.append(
            _path(
                "M {x} {start} V {end}".format(
                    x=source.center_x,
                    start=source.bottom,
                    end=y,
                ),
                color="#ca5a24",
                opacity=0.20,
                width=1.4,
                extra=(
                    ' id="contract-reference-{source}" '
                    'data-relation="references" '
                    'data-contract-id="{target}"'
                ).format(
                    source=reference["source"],
                    target=target_id,
                ),
            )
        )

    for contract_id, sources in grouped.items():
        contract = contract_boxes[contract_id]
        y = rail_y[contract.contract_type]
        source_x = [source.center_x for source in sources]
        left = min(source_x + [contract.center_x])
        right = max(source_x + [contract.center_x])
        out.extend(
            [
                _path(
                    "M {} {} H {}".format(left, y, right),
                    color="#ca5a24",
                    opacity=0.48,
                    width=2.2,
                    extra=(
                        ' data-contract-reference-rail="{}"'
                    ).format(contract_id),
                ),
                _path(
                    "M {x} {start} V {end}".format(
                        x=contract.center_x,
                        start=y,
                        end=contract.y,
                    ),
                    color="#ca5a24",
                    marker="arrow-orange",
                    opacity=0.72,
                    width=2.4,
                    extra=(
                        ' data-contract-reference-bus="{}"'
                    ).format(contract_id),
                ),
            ]
        )
    return out


def _layout_policies(
    document: Mapping[str, Any],
    contract_boxes: Mapping[str, ContractBox],
) -> Dict[str, PolicyBox]:
    """Lay out the existing Layer-4 records under their target Contracts."""

    policies = {
        item["id"]: item for item in document["layer4"]["policies"]
    }
    connections = document["layer4_to_layer3"]["connections"]
    target_by_policy: Dict[str, str] = {}
    for connection in connections:
        if connection["relation"] != "EXECUTES":
            raise ValueError("Layer 4 may only contain EXECUTES connections")
        policy_id = connection["source_policy_id"]
        if policy_id in target_by_policy:
            raise ValueError("a Policy must execute exactly one Contract")
        target_by_policy[policy_id] = connection["target_contract_id"]

    if set(target_by_policy) != set(policies):
        raise ValueError("every rendered Policy needs one EXECUTES connection")
    if not set(target_by_policy.values()).issubset(contract_boxes):
        raise ValueError("a Policy references an unknown Contract")

    boxes: Dict[str, PolicyBox] = {}
    for contract_id, contract in contract_boxes.items():
        group = sorted(
            (
                policies[policy_id]
                for policy_id, target_id in target_by_policy.items()
                if target_id == contract_id
            ),
            key=lambda item: item["id"],
        )
        columns = 2 if len(group) > 12 else 1
        rows = (len(group) + columns - 1) // columns
        gap = 8
        box_width = (contract.width - 28 - gap * (columns - 1)) / columns
        for policy_index, policy in enumerate(group):
            column = policy_index // rows
            row = policy_index % rows
            boxes[policy["id"]] = PolicyBox(
                policy_id=policy["id"],
                contract_id=contract_id,
                short_id=policy["short_id"],
                x=contract.x + 14 + column * (box_width + gap),
                y=1980 + row * 28,
                width=box_width,
            )
    return boxes


def _render_policy_layer(
    document: Mapping[str, Any],
    contract_boxes: Mapping[str, ContractBox],
) -> List[str]:
    """Render existing Policy records and their sole EXECUTES relation."""

    policy_boxes = _layout_policies(document, contract_boxes)
    connections = document["layer4_to_layer3"]["connections"]
    grouped: Dict[str, List[PolicyBox]] = {
        contract_id: [] for contract_id in contract_boxes
    }
    for connection in connections:
        grouped[connection["target_contract_id"]].append(
            policy_boxes[connection["source_policy_id"]]
        )

    out: List[str] = [
        _rect(
            30,
            1835,
            2940,
            520,
            fill="#f6f3fc",
            stroke="#f6f3fc",
            radius=20,
        ),
        _text(
            55,
            1875,
            "Layer 4 · Policies",
            size=22,
            weight=700,
            fill="#6842cb",
        ),
        _text(
            2945,
            1875,
            "stored concrete executables · no Policy-to-Policy edges",
            size=14,
            fill="#71809b",
            anchor="end",
        ),
    ]

    for contract_id, contract in sorted(
        contract_boxes.items(),
        key=lambda item: CONTRACT_ORDER.index(item[1].contract_type),
    ):
        out.extend(
            [
                _rect(
                    contract.x,
                    1940,
                    contract.width,
                    390,
                    fill="#fbfaff",
                    stroke="#7552d6",
                    radius=15,
                    dashed=True,
                ),
            ]
        )

    rail_y = 1910
    for contract_id, contract in sorted(
        contract_boxes.items(),
        key=lambda item: CONTRACT_ORDER.index(item[1].contract_type),
    ):
        group = sorted(grouped[contract_id], key=lambda item: item.policy_id)
        if not group:
            raise ValueError("every Contract must be executed by a Policy")
        policy_xs = [box.center_x for box in group]
        out.extend(
            [
                _path(
                    "M {} {} H {}".format(
                        min(policy_xs + [contract.center_x]),
                        rail_y,
                        max(policy_xs + [contract.center_x]),
                    ),
                    color="#7552d6",
                    opacity=0.48,
                    width=2.2,
                    extra=(
                        ' data-policy-executes-rail="{}"'
                    ).format(contract_id),
                ),
                _path(
                    "M {x} {start} V {end}".format(
                        x=contract.center_x,
                        start=rail_y,
                        end=contract.bottom,
                    ),
                    color="#7552d6",
                    marker="arrow-purple",
                    opacity=0.78,
                    width=2.5,
                    extra=(
                        ' data-policy-executes-bus="{}"'
                    ).format(contract_id),
                ),
            ]
        )
        for box in group:
            out.append(
                _path(
                    "M {x} {start} V {end}".format(
                        x=box.center_x,
                        start=box.top,
                        end=rail_y,
                    ),
                    color="#7552d6",
                    opacity=0.22,
                    width=1.4,
                    extra=(
                        ' id="policy-executes-{token}" '
                        'data-policy-id="{policy_id}" '
                        'data-relation="EXECUTES" '
                        'data-contract-id="{contract_id}"'
                    ).format(
                        token=_svg_token(box.policy_id),
                        policy_id=box.policy_id,
                        contract_id=contract_id,
                    ),
                )
            )

    for contract_id, contract in sorted(
        contract_boxes.items(),
        key=lambda item: CONTRACT_ORDER.index(item[1].contract_type),
    ):
        group = sorted(grouped[contract_id], key=lambda item: item.policy_id)
        out.append(
            _text(
                contract.center_x,
                1968,
                "{} policies".format(len(group)),
                size=15,
                weight=700,
                fill="#6842cb",
                anchor="middle",
            )
        )
        for box in group:
            out.extend(
                [
                    _rect(
                        box.x,
                        box.y,
                        box.width,
                        box.height,
                        fill="#ffffff",
                        stroke="#8d6ee0",
                        radius=10,
                        extra=(
                            ' id="policy-node-{token}" '
                            'data-policy-id="{policy_id}" '
                            'data-contract-id="{contract_id}"'
                        ).format(
                            token=_svg_token(box.policy_id),
                            policy_id=box.policy_id,
                            contract_id=box.contract_id,
                        ),
                    ),
                    _text(
                        box.center_x,
                        box.y + 15,
                        box.short_id,
                        size=10.5,
                        fill="#6245b1",
                        anchor="middle",
                    ),
                ]
            )
    return out


def coarse_higher_layers_svg(document: Mapping[str, Any]) -> str:
    """Render Layer 1/2/3, plus Layer 4 when existing records are supplied."""

    summary = document["summary"]
    if (
        summary["subgoals"] != 3
        or summary["subgoal_dependencies"] != 0
        or summary["skill_subgraphs"] != 3
    ):
        raise ValueError("diagram requires three independent coarse SubGoals")

    has_layer4 = "layer4" in document or "layer4_to_layer3" in document
    if has_layer4 and not (
        "layer4" in document and "layer4_to_layer3" in document
    ):
        raise ValueError("Layer 4 records and connections must be supplied together")
    svg_height = FOUR_LAYER_SVG_HEIGHT if has_layer4 else SVG_HEIGHT

    subgraphs, nodes = _subgraph_maps(document)
    node_boxes = _layout_skill_nodes(document)
    contract_boxes = _layout_contracts(document)
    subgoals = {
        item["id"]: item for item in document["layer1"]["subgoals"]
    }

    out: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ).format(width=SVG_WIDTH, height=svg_height),
        (
            '<title id="title">Four-layer coarse semantic skill graph</title>'
            if has_layer4
            else '<title id="title">Atomic coarse semantic skill graph</title>'
        ),
        (
            '<desc id="description">Three independent SubGoals, forty-four '
            "atomic SkillNodes, five shared Contracts, and fifty-three "
            "stored Policies.</desc>"
            if has_layer4
            else '<desc id="description">Three independent SubGoals, forty-four '
            "atomic SkillNodes, typed graph edges, and five shared Contracts."
            "</desc>"
        ),
        "<defs>",
        _marker("arrow-navy", "#40536a"),
        _marker("arrow-green", "#2b965d"),
        _marker("arrow-red", "#c43d3d"),
        _marker("arrow-purple", "#7354c7"),
        _marker("arrow-orange", "#ca5a24"),
        _marker("arrow-gray", "#66758d"),
        "</defs>",
        '<rect width="3000" height="{}" fill="#ffffff"/>'.format(svg_height),
        _text(
            55,
            55,
            "Coarse semantic graph · atomic SkillNodes",
            size=30,
            weight=700,
            fill="#243247",
        ),
        _text(
            2945,
            55,
            (
                "3 SubGoals · 3 SkillSubgraphs · 44 SkillNodes · "
                "5 Contracts · 53 Policies"
                if has_layer4
                else "3 SubGoals · 3 SkillSubgraphs · 44 SkillNodes · "
                "5 Contracts"
            ),
            size=16,
            fill="#66758d",
            anchor="end",
        ),
        _rect(
            30,
            85,
            2940,
            230,
            fill="#eef5fd",
            stroke="#eef5fd",
            radius=20,
        ),
        _rect(
            30,
            335,
            2940,
            1060,
            fill="#eff8f2",
            stroke="#eff8f2",
            radius=20,
        ),
        _rect(
            30,
            1415,
            2940,
            400,
            fill="#fbf5f1",
            stroke="#fbf5f1",
            radius=20,
        ),
        _text(
            55,
            120,
            "Layer 1 · SubGoals",
            size=22,
            weight=700,
            fill="#276db2",
        ),
        _text(
            55,
            370,
            "Layer 2 · SkillSubgraphs",
            size=22,
            weight=700,
            fill="#257c50",
        ),
        _text(
            55,
            1455,
            "Layer 3 · Contracts",
            size=22,
            weight=700,
            fill="#b94a17",
        ),
    ]

    for subgoal_id in SUBGOAL_ORDER:
        panel_x = PANEL_X[subgoal_id]
        center_x = panel_x + PANEL_WIDTH / 2
        out.extend(
            [
                _rect(
                    panel_x + 110,
                    145,
                    700,
                    120,
                    fill="#ffffff",
                    stroke="#4b8bc6",
                ),
                _text(
                    center_x,
                    185,
                    subgoal_id.capitalize(),
                    size=21,
                    weight=700,
                    fill="#276db2",
                    anchor="middle",
                ),
                _text(
                    center_x,
                    225,
                    subgoals[subgoal_id]["predicate"],
                    size=14,
                    fill="#52657a",
                    anchor="middle",
                ),
                _path(
                    "M {x} 265 V {end}".format(
                        x=center_x,
                        end=PANEL_Y,
                    ),
                    color="#40536a",
                    marker="arrow-navy",
                    width=2.4,
                ),
                _rect(
                    panel_x,
                    PANEL_Y,
                    PANEL_WIDTH,
                    PANEL_HEIGHT,
                    fill="#f8fffa",
                    stroke="#298f5b",
                    radius=16,
                    dashed=True,
                ),
                _text(
                    center_x,
                    412,
                    "{} SkillSubgraph".format(subgoal_id.capitalize()),
                    size=19,
                    weight=700,
                    fill="#257c50",
                    anchor="middle",
                ),
            ]
        )
    out.extend(
        _render_contract_references(
            document,
            node_boxes,
            contract_boxes,
        )
    )

    for subgoal_id in SUBGOAL_ORDER:
        panel_right = PANEL_X[subgoal_id] + PANEL_WIDTH
        for edge in subgraphs[subgoal_id]["edges"]:
            out.append(_render_skill_edge(edge, node_boxes, panel_right))

    for node_id in sorted(node_boxes):
        out.extend(_render_node(node_boxes[node_id], nodes[node_id]))

    usage = summary["contract_usage"]
    for contract_id, box in sorted(
        contract_boxes.items(),
        key=lambda item: CONTRACT_ORDER.index(item[1].contract_type),
    ):
        out.extend(
            [
                _rect(
                    box.x,
                    box.y,
                    box.width,
                    box.height,
                    fill="#fffaf2",
                    stroke="#c6531a",
                    radius=12,
                    extra=(
                        ' id="contract-node-{}"'
                    ).format(box.contract_type),
                ),
                _text(
                    box.center_x,
                    box.y + 38,
                    "{}Contract".format(box.contract_type.capitalize()),
                    size=18,
                    weight=700,
                    fill="#b94a17",
                    anchor="middle",
                ),
                _text(
                    box.center_x,
                    box.y + 70,
                    "{} incoming SkillNodes".format(usage[contract_id]),
                    size=13,
                    fill="#6e5b52",
                    anchor="middle",
                ),
            ]
        )

    if has_layer4:
        out.extend(_render_policy_layer(document, contract_boxes))

    legend_y = 2455 if has_layer4 else 1870
    out.extend(
        [
            _path(
                "M 90 {y} H 180".format(y=legend_y),
                color="#2b965d",
                marker="arrow-green",
            ),
            _text(195, legend_y + 5, "Sequential", size=14),
            _path(
                "M 570 {y} H 660".format(y=legend_y),
                color="#7354c7",
                marker="arrow-purple",
                dashed=True,
                extra=' marker-start="url(#arrow-purple)"',
            ),
            _text(675, legend_y + 5, "Alternative", size=14),
            _path(
                "M 1050 {y} H 1140".format(y=legend_y),
                color="#c43d3d",
                marker="arrow-red",
                dashed=True,
            ),
            _text(1155, legend_y + 5, "Fallback", size=14),
            _path(
                "M 1530 {y} H 1620".format(y=legend_y),
                color="#ca5a24",
                marker="arrow-orange",
                opacity=0.72,
            ),
            _text(1635, legend_y + 5, "References Contract", size=14),
            *(
                [
                    _path(
                        "M 2050 {y} H 2140".format(y=legend_y),
                        color="#7552d6",
                        marker="arrow-purple",
                        opacity=0.78,
                    ),
                    _text(2155, legend_y + 5, "Executes Contract", size=14),
                ]
                if has_layer4
                else []
            ),
            _text(
                2910,
                legend_y + 5,
                (
                    "each SkillNode / Policy → exactly one Contract"
                    if has_layer4
                    else "one SkillNode → exactly one Contract"
                ),
                size=14,
                weight=700,
                fill="#b94a17",
                anchor="end",
            ),
            "</svg>",
        ]
    )
    return "\n".join(out) + "\n"


def write_artifacts(json_path: Path, svg_path: Path) -> Dict[str, object]:
    document = coarse_higher_layers_document(build_coarse_higher_layers())
    json_path = Path(json_path)
    svg_path = Path(svg_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    svg_path.write_text(coarse_higher_layers_svg(document))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the atomic coarse Layer-1/2/3 semantic graph."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "coarse_graph.json",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "coarse_higher_layers.svg",
    )
    args = parser.parse_args()
    document = write_artifacts(args.json, args.svg)
    print(
        "wrote {} SubGoals, {} SkillSubgraphs, {} semantic strategies, "
        "{} SkillNodes, and {} SkillNode-to-Contract references".format(
            document["summary"]["subgoals"],
            document["summary"]["skill_subgraphs"],
            document["summary"]["semantic_strategies"],
            document["summary"]["skill_nodes"],
            len(document["layer2_to_layer3"]["references"]),
        )
    )


if __name__ == "__main__":
    main()
