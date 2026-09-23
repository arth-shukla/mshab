"""Layer 4: read-only storage for the 53 concrete MS-HAB policies.

There are intentionally no policy-policy edges, fallback rules, alternative
rules, or selection logic in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

from mshab.skills.model import CheckpointPolicy, ContractType, Policy


TASK_ABBREVIATIONS = {
    "prepare_groceries": "PG",
    "set_table": "ST",
    "tidy_house": "TH",
}

HOUSEHOLD_OBJECTS = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "024_bowl",
)

TASK_OBJECT_CATEGORIES = {
    "prepare_groceries": HOUSEHOLD_OBJECTS,
    "set_table": ("013_apple", "024_bowl"),
    "tidy_house": HOUSEHOLD_OBJECTS,
}


@dataclass(frozen=True)
class PolicySpec:
    """Portable storage metadata for one checkpoint policy."""

    task_family: str
    contract_type: ContractType
    target: str
    family: str = "rl"

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_type", ContractType(self.contract_type))
        if self.task_family not in TASK_ABBREVIATIONS:
            raise ValueError("unsupported task family {!r}".format(self.task_family))
        if not self.target:
            raise ValueError("policy target must be non-empty")
        if self.contract_type == ContractType.NAVIGATE:
            if self.target != "all":
                raise ValueError("navigation only has an all-target checkpoint")
        elif self.contract_type in (ContractType.PICK, ContractType.PLACE):
            allowed = TASK_OBJECT_CATEGORIES[self.task_family] + ("all",)
            if self.target not in allowed:
                raise ValueError(
                    "unsupported {} target {!r} for {}".format(
                        self.contract_type.value,
                        self.target,
                        self.task_family,
                    )
                )
        elif self.contract_type in (ContractType.OPEN, ContractType.CLOSE):
            if self.task_family != "set_table":
                raise ValueError("open/close checkpoints only exist for SetTable")
            if self.target not in ("fridge", "kitchen_counter"):
                raise ValueError(
                    "unsupported articulation target {!r}".format(self.target)
                )

    @property
    def id(self) -> str:
        return ".".join(
            (self.family, self.task_family, self.contract_type.value, self.target)
        )

    @property
    def short_id(self) -> str:
        return "{}.{}.{}".format(
            TASK_ABBREVIATIONS[self.task_family],
            self.contract_type.value,
            self.target,
        )

    @property
    def relative_leaf(self) -> str:
        return "/".join(
            (self.family, self.task_family, self.contract_type.value, self.target)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "short_id": self.short_id,
            "family": self.family,
            "task_family": self.task_family,
            "contract_type": self.contract_type.value,
            "target": self.target,
            "checkpoint_leaf": self.relative_leaf,
        }


def _policy_specs() -> Tuple[PolicySpec, ...]:
    specs = []
    for task_family in ("prepare_groceries", "set_table", "tidy_house"):
        specs.append(PolicySpec(task_family, ContractType.NAVIGATE, "all"))
        targets = TASK_OBJECT_CATEGORIES[task_family] + ("all",)
        for contract_type in (ContractType.PICK, ContractType.PLACE):
            specs.extend(
                PolicySpec(task_family, contract_type, target)
                for target in targets
            )
        if task_family == "set_table":
            for contract_type in (ContractType.OPEN, ContractType.CLOSE):
                specs.extend(
                    PolicySpec(task_family, contract_type, target)
                    for target in ("fridge", "kitchen_counter")
                )
    result = tuple(sorted(specs, key=lambda item: item.id))
    if len(result) != 53 or len({item.id for item in result}) != 53:
        raise AssertionError("the canonical RL policy store must contain 53 items")
    return result


POLICY_SPECS = _policy_specs()


@dataclass(frozen=True)
class Layer4PolicyStore:
    """Read-only policy inventory with no relations or routing behavior."""

    policies: Mapping[str, Policy]
    specs: Mapping[str, PolicySpec]

    def __post_init__(self) -> None:
        policies = dict(self.policies)
        specs = dict(self.specs)
        if len(policies) != 53 or len(specs) != 53:
            raise ValueError("Layer 4 must contain exactly 53 policies")
        if set(policies) != set(specs):
            raise ValueError("Layer-4 policies and specifications must match")
        object.__setattr__(self, "policies", MappingProxyType(policies))
        object.__setattr__(self, "specs", MappingProxyType(specs))

    def policy(self, policy_id: str) -> Policy:
        return self.policies[policy_id]

    def spec(self, policy_id: str) -> PolicySpec:
        return self.specs[policy_id]

def build_layer4(checkpoint_root: Path) -> Layer4PolicyStore:
    """Construct the 53 stored CheckpointPolicy objects."""

    root = Path(checkpoint_root)
    policies = {
        spec.id: CheckpointPolicy.from_leaf(
            root,
            spec.family,
            spec.task_family,
            spec.contract_type,
            spec.target,
        )
        for spec in POLICY_SPECS
    }
    return Layer4PolicyStore(
        policies=policies,
        specs={spec.id: spec for spec in POLICY_SPECS},
    )
