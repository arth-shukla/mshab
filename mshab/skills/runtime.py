"""Layer-3/4 grounding, contract monitoring, and policy dispatch interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

from mshab.skills.environment import EnvironmentAdapter, EnvironmentSnapshot
from mshab.skills.graph import SkillGraph, SkillNode
from mshab.skills.library import ContractLibrary
from mshab.skills.model import (
    ArtifactStatus,
    GroundedSkill,
    Policy,
)


class ContractViolation(RuntimeError):
    """Raised when a grounded skill cannot safely start or continue."""


@dataclass(frozen=True)
class PolicyExecution:
    """Environment-specific result returned by a Layer-4 policy executor."""

    success: bool
    steps: int
    failure_mode: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("policy execution steps cannot be negative")
        if self.success and self.failure_mode is not None:
            raise ValueError("successful execution cannot declare a failure mode")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


ContractMonitor = Callable[[EnvironmentSnapshot], None]


class PolicyExecutor(ABC):
    """Executor interface for RL/BC/DP/VLA/controller policies.

    A concrete executor loads or caches its model, repeatedly obtains actions,
    calls ``environment.step(action)``, and calls ``monitor(snapshot)`` after
    every step.  The executor owns tensor/model details; the runtime owns
    contract admission and final verification.
    """

    @abstractmethod
    def execute(
        self,
        grounded: GroundedSkill,
        policy: Policy,
        environment: EnvironmentAdapter,
        monitor: ContractMonitor,
    ) -> PolicyExecution:
        pass


@dataclass(frozen=True)
class SkillExecutionResult:
    """Auditable outcome combining policy status with contract evidence."""

    grounded: GroundedSkill
    policy_id: str
    environment_id: str
    before: EnvironmentSnapshot
    after: EnvironmentSnapshot
    policy_execution: PolicyExecution
    missing_effects: Tuple[str, ...]
    missing_verification: Tuple[str, ...]
    unretracted_deletes: Tuple[str, ...] = ()
    violated_invariants: Tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return (
            self.policy_execution.success
            and not self.missing_effects
            and not self.missing_verification
            and not self.unretracted_deletes
            and not self.violated_invariants
        )

    @property
    def failure_mode(self) -> Optional[str]:
        if self.policy_execution.failure_mode is not None:
            return self.policy_execution.failure_mode
        if self.violated_invariants:
            return "invariant_violated"
        if self.missing_effects or self.missing_verification:
            return "contract_verification_failed"
        if self.unretracted_deletes:
            return "state_not_retracted"
        return None


class SkillGrounder:
    """Layer-2 skill node -> Layer-3 GroundedSkill boundary.

    Grounding binds a node to its contract only.  The policy that will run
    the contract is chosen later, by the library, at execution time.
    """

    def __init__(self, library: ContractLibrary) -> None:
        self.library = library

    def ground(self, node: SkillNode) -> GroundedSkill:
        contract = self.library.get(node.contract_id)
        return contract.bind(node.arguments)

    def grounded_skills(
        self, graph: SkillGraph
    ) -> Mapping[str, GroundedSkill]:
        return MappingProxyType(
            {
                node_id: self.ground(node)
                for node_id, node in graph.nodes.items()
            }
        )


class SkillRuntime:
    """Coordinates Layer 3 contracts with Layer 4 environment execution."""

    def __init__(
        self,
        library: ContractLibrary,
        environment: EnvironmentAdapter,
    ) -> None:
        self.library = library
        self.environment = environment
        self.grounder = SkillGrounder(library)

    def ready_nodes(
        self,
        graph: SkillGraph,
        completed: Tuple[str, ...] = (),
        policy_ids: Optional[Mapping[str, str]] = None,
    ) -> Tuple[SkillNode, ...]:
        """Dependency-, artifact-, environment-, and contract-ready nodes.

        ``policy_ids`` optionally pins one policy per node id; such a node is
        ready only if that policy is bound to its contract and its artifacts
        are available.  Otherwise any ready bound policy suffices.
        """

        facts = self.environment.snapshot().facts
        policy_ids = policy_ids or {}
        ready = []
        for node in graph.ready_nodes(completed):
            try:
                grounded = self.grounder.ground(node)
            except (KeyError, ValueError):
                # An unregistered or unbindable contract makes the node
                # unexecutable, not the whole schedule unanswerable.
                continue
            contract = grounded.contract
            requested = policy_ids.get(node.id)
            if requested is None:
                if not self.library.ready(contract.id):
                    continue
            else:
                try:
                    policy = self.library.select_policy(contract.id, requested)
                except KeyError:
                    continue
                if policy.status != ArtifactStatus.READY:
                    continue
            if not self.environment.supports_contract_env(contract.env_id):
                continue
            if not grounded.can_start(facts):
                continue
            if not set(grounded.invariants).issubset(facts):
                continue
            ready.append(node)
        return tuple(ready)

    def execute_node(
        self,
        graph: SkillGraph,
        node_id: str,
        executor: PolicyExecutor,
        policy_id: Optional[str] = None,
    ) -> SkillExecutionResult:
        """Ground one node, admit it, and run it with a policy bound to its contract.

        ``policy_id`` selects one of the contract's bound policies explicitly;
        by default the first ready one in binding order is used.
        """

        try:
            node = graph.nodes[node_id]
        except KeyError as exc:
            raise KeyError("unknown skill node {!r}".format(node_id)) from exc
        grounded = self.grounder.ground(node)
        contract = grounded.contract
        if not self.environment.supports_contract_env(contract.env_id):
            raise RuntimeError(
                "environment {!r} is not compatible with contract environment {!r}".format(
                    self.environment.description.environment_id, contract.env_id
                )
            )
        policy = self.library.select_policy(contract.id, policy_id)
        before = self.environment.snapshot()
        self._admit(grounded, before)

        def monitor(snapshot: EnvironmentSnapshot) -> None:
            missing = _missing(grounded.invariants, snapshot.facts)
            if missing:
                raise ContractViolation(
                    "invariants violated during {}: {}".format(
                        grounded.id, list(missing)
                    )
                )

        execution = executor.execute(
            grounded=grounded,
            policy=policy,
            environment=self.environment,
            monitor=monitor,
        )
        after = self.environment.snapshot()
        # A post-execution invariant breach is reported as evidence on the
        # result, not raised: a failed node must stay auditable so the planner
        # can route to its fallback instead of unwinding the whole rollout.
        return SkillExecutionResult(
            grounded=grounded,
            policy_id=policy.id,
            environment_id=self.environment.description.environment_id,
            before=before,
            after=after,
            policy_execution=execution,
            missing_effects=_missing(grounded.effects, after.facts),
            missing_verification=_missing(
                grounded.verification, after.facts
            ),
            unretracted_deletes=tuple(
                predicate
                for predicate in grounded.deletes
                if predicate in after.facts
            ),
            violated_invariants=_missing(grounded.invariants, after.facts),
        )

    @staticmethod
    def _admit(grounded: GroundedSkill, snapshot: EnvironmentSnapshot) -> None:
        missing_preconditions = _missing(grounded.preconditions, snapshot.facts)
        missing_invariants = _missing(grounded.invariants, snapshot.facts)
        if missing_preconditions or missing_invariants:
            raise ContractViolation(
                "contract cannot start: missing_preconditions={}, "
                "missing_invariants={}".format(
                    list(missing_preconditions), list(missing_invariants)
                )
            )


def _missing(required: Tuple[str, ...], facts: frozenset) -> Tuple[str, ...]:
    return tuple(predicate for predicate in required if predicate not in facts)
