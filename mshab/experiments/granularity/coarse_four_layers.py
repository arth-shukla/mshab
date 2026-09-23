"""Compose the existing coarse higher and lower layers into one artifact.

This module is deliberately an orchestration boundary.  It does not define
SubGoals, SkillNodes, Contracts, Policies, or graph relations.  Instead it
builds the existing lower-layer stack once, shares its Layer-3 object with the
existing higher-layer builder, and serializes the two existing documents.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

from mshab.experiments.granularity.higher_layers import (
    build_coarse_higher_layers,
    coarse_higher_layers_document,
)
from mshab.experiments.granularity.higher_layers.render import (
    coarse_higher_layers_svg,
)
from mshab.experiments.granularity.lower_layers.connections import (
    build_connected_layers,
    connected_layers_document,
)
from mshab.experiments.granularity.render import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CHECKPOINT_ROOT,
)


SCHEMA_VERSION = "mshab.granularity-coarse-four-layers.v1"


def _validate_composition(
    higher: Mapping[str, Any],
    lower: Mapping[str, Any],
) -> None:
    """Reject a broken join instead of silently drawing disconnected layers."""

    higher_contract_ids = {
        contract["id"] for contract in higher["layer3"]["contracts"]
    }
    lower_contract_ids = {
        contract["id"] for contract in lower["layer3_contracts"]
    }
    if higher_contract_ids != lower_contract_ids:
        raise ValueError("higher and lower layers do not share the same Contracts")

    skill_node_ids = {
        node["id"] for node in higher["layer2"]["nodes"]
    }
    references = higher["layer2_to_layer3"]["references"]
    if Counter(item["source"] for item in references) != Counter(
        {node_id: 1 for node_id in skill_node_ids}
    ):
        raise ValueError("every SkillNode must reference exactly one Contract")
    if not {
        item["target"] for item in references
    }.issubset(lower_contract_ids):
        raise ValueError("a SkillNode references an unknown Contract")

    policy_ids = {
        policy["id"] for policy in lower["layer4_policies"]
    }
    connections = lower["connections"]
    if Counter(item["source_policy_id"] for item in connections) != Counter(
        {policy_id: 1 for policy_id in policy_ids}
    ):
        raise ValueError("every Policy must execute exactly one Contract")
    if {item["relation"] for item in connections} != {"EXECUTES"}:
        raise ValueError("Policy-to-Contract connections must be EXECUTES")
    if not {
        item["target_contract_id"] for item in connections
    }.issubset(lower_contract_ids):
        raise ValueError("a Policy executes an unknown Contract")


def compose_documents(
    higher: Mapping[str, Any],
    lower: Mapping[str, Any],
) -> Dict[str, Any]:
    """Join existing serialized layers without recreating domain objects."""

    _validate_composition(higher, lower)
    skill_edges = sum(
        len(subgraph["edges"])
        for subgraph in higher["layer2"]["subgraphs"]
    ) + len(higher["layer2"].get("cross_subgraph_edges", ()))

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "graph_granularity",
        "granularity": higher["granularity"],
        "scope": "layers_1_through_4",
        "source_schema_versions": {
            "higher_layers": higher["schema_version"],
            "lower_layers": lower["schema_version"],
        },
        "semantics": {
            **higher["semantics"],
            "layer4_to_layer3": "Policy --EXECUTES--> Contract",
            "layer4_role": lower["semantics"]["layer_4_role"],
            "policy_policy_relations": lower["semantics"][
                "policy_policy_relations"
            ],
        },
        "summary": {
            **higher["summary"],
            "skill_edges": skill_edges,
            "skill_node_contract_references": len(
                higher["layer2_to_layer3"]["references"]
            ),
            "contracts": lower["summary"]["contracts"],
            "policies": lower["summary"]["policies"],
            "executes_connections": lower["summary"][
                "executes_connections"
            ],
            "policies_by_contract": lower["summary"][
                "policies_by_contract"
            ],
        },
        "layer1": higher["layer1"],
        "layer2": higher["layer2"],
        "layer2_to_layer3": higher["layer2_to_layer3"],
        "layer3": {"contracts": lower["layer3_contracts"]},
        "layer4": {"policies": lower["layer4_policies"]},
        "layer4_to_layer3": {
            "cardinality": "many_to_one",
            "relation": "EXECUTES",
            "connections": lower["connections"],
        },
    }


def build_coarse_four_layer_document(checkpoint_root: Path) -> Dict[str, Any]:
    """Call the existing builders and return one portable four-layer graph."""

    lower_stack = build_connected_layers(Path(checkpoint_root))
    higher = build_coarse_higher_layers(lower_stack.layer3)
    return compose_documents(
        coarse_higher_layers_document(higher),
        connected_layers_document(lower_stack),
    )


def write_artifacts(
    checkpoint_root: Path,
    json_path: Path,
    svg_path: Path,
) -> Dict[str, Any]:
    """Generate deterministic JSON and SVG from the existing OOP layers."""

    document = build_coarse_four_layer_document(Path(checkpoint_root))
    json_path = Path(json_path)
    svg_path = Path(svg_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    svg_path.write_text(coarse_higher_layers_svg(document))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the existing SubGoal, SkillSubgraph, Contract, and "
            "Policy layers into one coarse graph."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="used to construct existing Policy objects; files are not loaded",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "coarse_four_layers.json",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "coarse_four_layers.svg",
    )
    args = parser.parse_args()
    document = write_artifacts(
        args.checkpoint_root,
        args.json,
        args.svg,
    )
    print(
        "wrote {} SubGoals, {} SkillNodes, {} Contracts, and {} Policies "
        "to {} and {}".format(
            document["summary"]["subgoals"],
            document["summary"]["skill_nodes"],
            document["summary"]["contracts"],
            document["summary"]["policies"],
            args.json,
            args.svg,
        )
    )


if __name__ == "__main__":
    main()
