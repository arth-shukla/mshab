"""Object model for Layer-3 contracts and the Layer-4 policies that execute them.

The classes in this module describe *what* a contract promises separately from
*how* it is executed.  ``pick(apple)`` is one contract even when an RL, BC,
DP, or remote VLA policy executes it.

A :class:`Contract` carries its preconditions, effects, invariants and
verification predicates directly as attributes.  Those predicates describe
only whether the contract can physically start and what it changes in the
current world state.  They never encode task order or semantic plausibility;
that belongs to the relations between Layer-2 skill nodes.

A :class:`Contract` knows nothing about policies and a :class:`Policy` knows
nothing about contracts.  Which policy executes which contract is a
many-to-many relation owned by :class:`~mshab.skills.library.ContractLibrary`:
one contract may be executed by several policies (an RL and a BC checkpoint
for the same pick), and one policy may execute several contracts (the
``pick.all`` checkpoint executes every ``pick.*`` contract of its task).

This module deliberately has no torch or ManiSkill imports.  Listing and
planning with the contract library must not allocate a GPU or load checkpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


class ContractType(str, Enum):
    """Built-in contract vocabulary currently implemented by MS-HAB."""

    NAVIGATE = "navigate"
    PICK = "pick"
    PLACE = "place"
    OPEN = "open"
    CLOSE = "close"


class ParameterType(str, Enum):
    """Planner-facing parameter types used by contract parameters."""

    ENTITY = "entity"
    LOCATION = "location"
    ARTICULATION = "articulation"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class PolicyKind(str, Enum):
    """Kinds of low-level policy represented in the architecture diagram."""

    CHECKPOINT = "checkpoint"
    VLA = "vla"
    NAVIGATION = "navigation"
    CONTROLLER = "controller"
    SCRIPT = "script"
    STATE_MACHINE = "state_machine"


class ArtifactStatus(str, Enum):
    """Local availability of a policy's artifacts."""

    MISSING = "missing"
    PARTIAL = "partial"
    READY = "ready"


@dataclass(frozen=True)
class ContractParameter:
    name: str
    type: ParameterType
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", ParameterType(self.type))
        if not self.name or not self.name.isidentifier():
            raise ValueError("contract parameter name must be a non-empty identifier")


_PREDICATE_GROUPS = (
    "preconditions",
    "effects",
    "invariants",
    "verification",
    "deletes",
)


def _check_predicate_consistency(
    effects: Sequence[str],
    invariants: Sequence[str],
    deletes: Sequence[str],
    *,
    grounded: bool,
) -> None:
    prefix = "grounded " if grounded else ""
    overlap = sorted(set(deletes) & set(effects))
    if overlap:
        raise ValueError(
            "{}predicates cannot be both asserted and retracted: {}".format(
                prefix, overlap
            )
        )
    invariant_overlap = sorted(set(deletes) & set(invariants))
    if invariant_overlap:
        raise ValueError(
            "a {}contract cannot retract its own invariants: {}".format(
                prefix, invariant_overlap
            )
        )


@dataclass(frozen=True)
class GroundedSkill:
    """One skill node bound to its contract with concrete arguments.

    The predicate tuples are the contract's templates with every parameter
    substituted, so ``can_start``/``verified`` can be evaluated directly
    against an environment snapshot's facts.
    """

    contract: "Contract" = field(repr=False)
    arguments: Mapping[str, Any]
    preconditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    invariants: Tuple[str, ...]
    verification: Tuple[str, ...]
    failure_modes: Tuple[str, ...]
    deletes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        for name in _PREDICATE_GROUPS + ("failure_modes",):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _check_predicate_consistency(
            self.effects, self.invariants, self.deletes, grounded=True
        )

    @property
    def id(self) -> str:
        args = ",".join(
            "{}={}".format(key, self.arguments[key]) for key in sorted(self.arguments)
        )
        return "{}({})".format(self.contract.id, args)

    def can_start(self, facts: Iterable[str]) -> bool:
        return set(self.preconditions).issubset(set(facts))

    def achieved(self, facts: Iterable[str]) -> bool:
        return set(self.effects).issubset(set(facts))

    def verified(self, facts: Iterable[str]) -> bool:
        """Whether the explicit success predicates hold after execution."""

        return set(self.verification).issubset(set(facts))

    def retracted(self, facts: Iterable[str]) -> bool:
        """Whether every predicate this contract retracts is actually gone."""

        return not (set(self.deletes) & set(facts))

    def apply_to(self, facts: Iterable[str]) -> FrozenSet[str]:
        """The symbolic state after a successful execution.

        Contracts reject an overlap between deletes and effects, so the order
        here is only an implementation detail.  Without delete effects, a
        state that records ``open(fridge)`` would keep ``closed(fridge)``
        alongside it, and ``holding(bowl)`` would survive placing the bowl.
        """

        return frozenset(set(facts) - set(self.deletes) | set(self.effects))


def _validate_parameter(parameter: ContractParameter, value: Any) -> None:
    expected = {
        ParameterType.ENTITY: str,
        ParameterType.LOCATION: str,
        ParameterType.ARTICULATION: str,
        ParameterType.STRING: str,
        ParameterType.INTEGER: int,
        ParameterType.FLOAT: (int, float),
        ParameterType.BOOLEAN: bool,
    }[parameter.type]
    if isinstance(value, bool) and parameter.type in (
        ParameterType.INTEGER,
        ParameterType.FLOAT,
    ):
        valid = False
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise TypeError(
            "argument {!r} must be {}, got {}".format(
                parameter.name, parameter.type.value, type(value).__name__
            )
        )


class Policy(ABC):
    """One low-level executable: a checkpoint, VLA, controller, or script.

    A policy is identified by a library-wide ``id`` and carries no reference
    to the contracts it executes; those bindings belong to
    :class:`~mshab.skills.library.ContractLibrary`.
    """

    def __init__(self, id: str, kind: PolicyKind) -> None:
        if not isinstance(id, str) or not id.strip():
            raise ValueError("policy id must be a non-empty string")
        self.id = id
        self.kind = PolicyKind(kind)

    @property
    @abstractmethod
    def status(self) -> ArtifactStatus:
        """Whether everything needed to execute this policy is available."""

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable library record."""


class CheckpointPolicy(Policy):
    """A local learned policy represented by ``config.yml`` + ``policy.pt``."""

    def __init__(
        self,
        id: str,
        family: str,
        checkpoint_path: Path,
        config_path: Path,
        policy_type: Optional[str] = None,
        checkpoint_sha256: Optional[str] = None,
    ) -> None:
        super().__init__(id=id, kind=PolicyKind.CHECKPOINT)
        self.family = family
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.policy_type = policy_type or family
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def from_leaf(
        cls,
        checkpoint_root: Path,
        family: str,
        task: str,
        contract_type: Union[ContractType, str],
        target: str,
    ) -> "CheckpointPolicy":
        """The policy stored at ``<root>/<family>/<task>/<type>/<target>/``.

        Its id is ``<family>.<task>.<type>.<target>``, so one checkpoint keeps
        one id however many contracts it is later bound to.
        """

        type_name = (
            contract_type.value
            if isinstance(contract_type, ContractType)
            else str(contract_type)
        )
        leaf = Path(checkpoint_root) / family / task / type_name / target
        return cls(
            id=".".join((family, task, type_name, target)),
            family=family,
            checkpoint_path=leaf / "policy.pt",
            config_path=leaf / "config.yml",
            policy_type=_checkpoint_policy_type(family, target),
        )

    @property
    def status(self) -> ArtifactStatus:
        present = (self.checkpoint_path.is_file(), self.config_path.is_file())
        if all(present):
            return ArtifactStatus.READY
        if any(present):
            return ArtifactStatus.PARTIAL
        return ArtifactStatus.MISSING

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "family": self.family,
            "policy_type": self.policy_type,
            "checkpoint": str(self.checkpoint_path),
            "config": str(self.config_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "status": self.status.value,
        }


def _checkpoint_policy_type(family: str, target: str) -> str:
    """The ``policy_type`` vocabulary ``mshab.evaluate`` uses for a checkpoint."""

    if family == "rl":
        return "rl_all_obj" if target == "all" else "rl_per_obj"
    return family


_TARGET_PARAMETER = {
    ContractType.NAVIGATE: "goal",
    ContractType.PICK: "object",
    ContractType.PLACE: "object",
    ContractType.OPEN: "articulation",
    ContractType.CLOSE: "articulation",
}


class Contract:
    """A Layer-3 contract: what a skill node asks for, and how it is verified.

    Predicates use ``str.format`` placeholders matching parameter names, for
    example ``holding({object})``.  They stay lightweight strings here; a
    simulator adapter is responsible for evaluating them against live state.

    A contract does not know which policies execute it.  That many-to-many
    relation, including the preference order among a contract's policies, is
    owned by :class:`~mshab.skills.library.ContractLibrary`.
    """

    def __init__(
        self,
        contract_type: Union[ContractType, str],
        task: str,
        target: str,
        *,
        parameters: Sequence[ContractParameter],
        preconditions: Sequence[str],
        effects: Sequence[str],
        env_id: str,
        max_episode_steps: int,
        invariants: Sequence[str] = (),
        verification: Sequence[str] = (),
        failure_modes: Sequence[str] = (),
        deletes: Sequence[str] = (),
        description: str = "",
        target_parameter: Optional[str] = None,
    ) -> None:
        if not task:
            raise ValueError("contract task must be non-empty")
        if not target:
            raise ValueError("contract target must be non-empty")
        contract_type = _normalise_contract_type(contract_type)
        contract_type_name = (
            contract_type.value if isinstance(contract_type, ContractType) else contract_type
        )
        if target_parameter is None:
            target_parameter = _TARGET_PARAMETER.get(contract_type)
        if not target_parameter or not target_parameter.isidentifier():
            raise ValueError(
                "contract target_parameter must be a non-empty identifier"
            )

        self.parameters: Tuple[ContractParameter, ...] = tuple(parameters)
        self.preconditions: Tuple[str, ...] = tuple(preconditions)
        self.effects: Tuple[str, ...] = tuple(effects)
        self.invariants: Tuple[str, ...] = tuple(invariants)
        self.verification: Tuple[str, ...] = tuple(verification) or self.effects
        self.failure_modes: Tuple[str, ...] = tuple(failure_modes)
        self.deletes: Tuple[str, ...] = tuple(deletes)

        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("contract has duplicate parameter names")
        if not self.effects:
            raise ValueError("contract must declare at least one effect")
        for group_name in _PREDICATE_GROUPS:
            if any(not predicate.strip() for predicate in getattr(self, group_name)):
                raise ValueError("{} cannot contain empty predicates".format(group_name))
        _check_predicate_consistency(
            self.effects, self.invariants, self.deletes, grounded=False
        )
        if target_parameter not in set(names):
            raise ValueError(
                "target parameter {!r} is not declared by the contract".format(
                    target_parameter
                )
            )
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        self.name = "{}.{}".format(contract_type_name, target)
        self.task = task
        self.description = description
        self.contract_type = contract_type
        self.target_parameter = target_parameter
        self.target = target
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps

    @property
    def id(self) -> str:
        return "mshab.{}.{}".format(self.task, self.name)

    @property
    def contract_type_name(self) -> str:
        return (
            self.contract_type.value
            if isinstance(self.contract_type, ContractType)
            else self.contract_type
        )

    # -- grounding ----------------------------------------------------------

    def bind(self, arguments: Mapping[str, Any]) -> GroundedSkill:
        """Ground this contract for one skill node's symbolic arguments.

        Grounding involves no policy: which bound policy runs the result is a
        Layer-4 decision the library makes at execution time.
        """

        values = dict(arguments)
        if self.target != "all":
            supplied = values.get(self.target_parameter)
            if supplied is not None and supplied != self.target:
                raise ValueError(
                    "contract {} is specialized for {!r}, not {!r}".format(
                        self.id, self.target, supplied
                    )
                )
            values[self.target_parameter] = self.target

        known = {parameter.name: parameter for parameter in self.parameters}
        missing = sorted(
            parameter.name
            for parameter in self.parameters
            if parameter.required and parameter.name not in values
        )
        unknown = sorted(set(values) - set(known))
        if missing or unknown:
            raise ValueError(
                "invalid contract arguments: missing={}, unknown={}".format(missing, unknown)
            )
        for name, value in values.items():
            _validate_parameter(known[name], value)

        def ground(predicates: Sequence[str]) -> Tuple[str, ...]:
            try:
                return tuple(predicate.format(**values) for predicate in predicates)
            except KeyError as exc:
                raise ValueError(
                    "predicate template references undeclared/unbound parameter {!r}".format(
                        exc.args[0]
                    )
                ) from exc

        return GroundedSkill(
            contract=self,
            arguments=values,
            preconditions=ground(self.preconditions),
            effects=ground(self.effects),
            invariants=ground(self.invariants),
            verification=ground(self.verification),
            failure_modes=self.failure_modes,
            deletes=ground(self.deletes),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task": self.task,
            "contract_type": self.contract_type_name,
            "target": self.target,
            "target_parameter": self.target_parameter,
            "description": self.description,
            "parameters": [
                {
                    "name": parameter.name,
                    "type": parameter.type.value,
                    "description": parameter.description,
                    "required": parameter.required,
                }
                for parameter in self.parameters
            ],
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "invariants": list(self.invariants),
            "verification": list(self.verification),
            "failure_modes": list(self.failure_modes),
            "deletes": list(self.deletes),
            "execution": {
                "env_id": self.env_id,
                "max_episode_steps": self.max_episode_steps,
            },
        }


def _normalise_contract_type(contract_type: Union[ContractType, str]) -> Union[ContractType, str]:
    if isinstance(contract_type, ContractType):
        return contract_type
    if not isinstance(contract_type, str) or not contract_type.strip():
        raise ValueError("contract_type must be a ContractType or non-empty string")
    value = contract_type.strip().lower()
    try:
        return ContractType(value)
    except ValueError:
        if not value.replace("_", "").isalnum():
            raise ValueError(
                "custom contract_type must contain only letters, numbers, or underscores"
            )
        return value


class NavigateContract(Contract):
    def __init__(self, task: str, target: str = "all") -> None:
        super().__init__(
            ContractType.NAVIGATE,
            task,
            target,
            parameters=(ContractParameter("goal", ParameterType.ENTITY),),
            preconditions=("present({goal})",),
            effects=("reachable({goal})",),
            verification=("reachable({goal})",),
            failure_modes=("navigation_timeout", "collision_limit"),
            env_id="NavigateSubtaskTrain-v0",
            max_episode_steps=1000,
            description="Move to a pose from which the grounded goal is reachable.",
        )


class PickContract(Contract):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            ContractType.PICK,
            task,
            target,
            parameters=(ContractParameter("object", ParameterType.ENTITY),),
            preconditions=("reachable({object})", "gripper_empty()"),
            effects=("holding({object})",),
            invariants=("collision_safe()",),
            verification=("holding({object})",),
            failure_modes=("grasp_failed", "object_dropped", "force_limit"),
            deletes=("gripper_empty()",),
            env_id="PickSubtaskTrain-v0",
            max_episode_steps=200,
            description="Grasp and hold one grounded object.",
        )


class PlaceContract(Contract):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            ContractType.PLACE,
            task,
            target,
            parameters=(
                ContractParameter("object", ParameterType.ENTITY),
                ContractParameter("destination", ParameterType.LOCATION),
            ),
            preconditions=("holding({object})", "reachable({destination})"),
            effects=("at({object},{destination})", "gripper_empty()"),
            invariants=("collision_safe()",),
            verification=("at({object},{destination})",),
            failure_modes=("placement_failed", "object_dropped", "force_limit"),
            deletes=("holding({object})",),
            env_id="PlaceSubtaskTrain-v0",
            max_episode_steps=200,
            description="Place a held object at a grounded destination.",
        )


class OpenContract(Contract):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            ContractType.OPEN,
            task,
            target,
            parameters=(
                ContractParameter("articulation", ParameterType.ARTICULATION),
            ),
            preconditions=("reachable({articulation})", "closed({articulation})"),
            effects=("open({articulation})",),
            invariants=("collision_safe()",),
            verification=("open({articulation})",),
            failure_modes=("handle_not_grasped", "joint_blocked", "force_limit"),
            deletes=("closed({articulation})",),
            env_id="OpenSubtaskTrain-v0",
            max_episode_steps=200,
            description="Open one grounded articulation such as a fridge or drawer.",
        )


class CloseContract(Contract):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            ContractType.CLOSE,
            task,
            target,
            parameters=(
                ContractParameter("articulation", ParameterType.ARTICULATION),
            ),
            preconditions=("reachable({articulation})", "open({articulation})"),
            effects=("closed({articulation})",),
            invariants=("collision_safe()",),
            verification=("closed({articulation})",),
            failure_modes=("handle_not_grasped", "joint_blocked", "force_limit"),
            deletes=("open({articulation})",),
            env_id="CloseSubtaskTrain-v0",
            max_episode_steps=200,
            description="Close one grounded articulation such as a fridge or drawer.",
        )


CONTRACT_CLASSES = {
    ContractType.NAVIGATE: NavigateContract,
    ContractType.PICK: PickContract,
    ContractType.PLACE: PlaceContract,
    ContractType.OPEN: OpenContract,
    ContractType.CLOSE: CloseContract,
}
