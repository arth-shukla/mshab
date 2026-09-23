"""Layer 3: the five reusable symbolic MS-HAB contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet, Mapping, Union

from mshab.skills.model import (
    CloseContract,
    Contract,
    ContractType,
    NavigateContract,
    OpenContract,
    PickContract,
    PlaceContract,
)


EXPERIMENT_TASK = "granularity"

_CONTRACT_CLASSES = {
    ContractType.NAVIGATE: NavigateContract,
    ContractType.PICK: PickContract,
    ContractType.PLACE: PlaceContract,
    ContractType.OPEN: OpenContract,
    ContractType.CLOSE: CloseContract,
}


@dataclass(frozen=True)
class Layer3Contracts:
    """Read-only storage for the experiment's parameterized contracts."""

    contracts: Mapping[ContractType, Contract]

    def __post_init__(self) -> None:
        contracts = dict(self.contracts)
        expected = set(ContractType)
        if set(contracts) != expected:
            raise ValueError(
                "Layer 3 must contain exactly {}; got {}".format(
                    sorted(item.value for item in expected),
                    sorted(item.value for item in contracts),
                )
            )
        for contract_type, contract in contracts.items():
            if contract.contract_type != contract_type:
                raise ValueError("contract is stored under the wrong type")
            if contract.task != EXPERIMENT_TASK or contract.target != "all":
                raise ValueError(
                    "Layer-3 contracts must be generic experiment schemas"
                )
        object.__setattr__(self, "contracts", MappingProxyType(contracts))

    def contract(self, contract_type: Union[ContractType, str]) -> Contract:
        return self.contracts[ContractType(contract_type)]

    @property
    def ids(self) -> FrozenSet[str]:
        """Stable ids accepted by Layer-2 ``SkillNode.contract_id`` fields."""

        return frozenset(contract.id for contract in self.contracts.values())

    def by_id(self, contract_id: str) -> Contract:
        """Resolve the single Contract referenced by one SkillNode."""

        for contract in self.contracts.values():
            if contract.id == contract_id:
                return contract
        raise KeyError("unknown Layer-3 contract {!r}".format(contract_id))


def build_layer3() -> Layer3Contracts:
    """Construct Navigate, Pick, Place, Open, and Close contracts."""

    return Layer3Contracts(
        {
            contract_type: contract_class(
                task=EXPERIMENT_TASK,
                target="all",
            )
            for contract_type, contract_class in _CONTRACT_CLASSES.items()
        }
    )
