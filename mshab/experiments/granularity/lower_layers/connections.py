"""The sole Layer-4-to-Layer-3 relation: Policy EXECUTES Contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from mshab.experiments.granularity.lower_layers.layer3 import (
    Layer3Contracts,
    build_layer3,
)
from mshab.experiments.granularity.lower_layers.layer4 import (
    Layer4PolicyStore,
    build_layer4,
)


SCHEMA_VERSION = "mshab.granularity-layer3-layer4.v1"


@dataclass(frozen=True)
class ExecutesConnection:
    """One directed Policy --EXECUTES--> Contract connection."""

    policy_id: str
    contract_id: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "source_policy_id": self.policy_id,
            "relation": "EXECUTES",
            "target_contract_id": self.contract_id,
        }


@dataclass(frozen=True)
class ConnectedLayers:
    """Layer stores plus their explicit EXECUTES connections."""

    layer3: Layer3Contracts
    layer4: Layer4PolicyStore
    connections: Tuple[ExecutesConnection, ...]

    def __post_init__(self) -> None:
        if len(self.connections) != 53:
            raise ValueError("there must be exactly 53 EXECUTES connections")
        policy_ids = {connection.policy_id for connection in self.connections}
        if policy_ids != set(self.layer4.policies):
            raise ValueError("every Layer-4 policy must have one connection")
        if len(policy_ids) != len(self.connections):
            raise ValueError("a Layer-4 policy may execute only one contract here")
        contract_ids = {contract.id for contract in self.layer3.contracts.values()}
        for connection in self.connections:
            if connection.contract_id not in contract_ids:
                raise ValueError("connection references an unknown contract")
            spec = self.layer4.spec(connection.policy_id)
            expected = self.layer3.contract(spec.contract_type).id
            if connection.contract_id != expected:
                raise ValueError("policy is connected to the wrong contract")


def connect_layers(
    layer3: Layer3Contracts,
    layer4: Layer4PolicyStore,
) -> ConnectedLayers:
    """Connect each stored policy to its matching generic contract."""

    connections = []
    for policy_id in sorted(layer4.policies):
        spec = layer4.spec(policy_id)
        contract = layer3.contract(spec.contract_type)
        connections.append(ExecutesConnection(policy_id, contract.id))
    return ConnectedLayers(layer3, layer4, tuple(connections))


def build_connected_layers(checkpoint_root: Path) -> ConnectedLayers:
    """Build both independent layers and then connect them."""

    return connect_layers(build_layer3(), build_layer4(checkpoint_root))


def connected_layers_document(stack: ConnectedLayers) -> Dict[str, Any]:
    """Return a portable document without machine-local readiness state."""

    contracts = []
    for contract_type, contract in stack.layer3.contracts.items():
        record = contract.as_dict()
        record["policy_count"] = sum(
            connection.contract_id == contract.id
            for connection in stack.connections
        )
        contracts.append(record)

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "graph_granularity",
        "scope": "layers_3_and_4_only",
        "summary": {
            "contracts": len(stack.layer3.contracts),
            "policies": len(stack.layer4.policies),
            "executes_connections": len(stack.connections),
            "policies_by_contract": {
                contract_type.value: sum(
                    connection.contract_id == contract.id
                    for connection in stack.connections
                )
                for contract_type, contract in stack.layer3.contracts.items()
            },
        },
        "semantics": {
            "only_relation": "Policy --EXECUTES--> Contract",
            "policy_policy_relations": "none",
            "layer_4_role": "policy_storage_only",
        },
        "layer3_contracts": contracts,
        "layer4_policies": [
            stack.layer4.spec(policy_id).as_dict()
            for policy_id in sorted(stack.layer4.policies)
        ],
        "connections": [
            connection.as_dict() for connection in stack.connections
        ],
    }
