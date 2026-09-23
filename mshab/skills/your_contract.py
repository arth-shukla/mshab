"""Copyable extension point for adding a new environment-specific contract."""

from __future__ import annotations

from typing import Optional, Sequence

from mshab.skills.model import Contract, ContractParameter, ParameterType


class YourContract(Contract):
    """Minimal example of a user-defined contract.

    Replace ``your_contract`` and the default predicates with domain ones,
    then provide a ``Policy`` and a ``PolicyExecutor`` implementation.  No
    change to the Layer-1/2 graph classes is required.
    """

    def __init__(
        self,
        task: str,
        target: str = "all",
        *,
        env_id: str = "YourContractEnv-v0",
        max_episode_steps: int = 200,
        preconditions: Optional[Sequence[str]] = None,
        effects: Optional[Sequence[str]] = None,
        invariants: Sequence[str] = ("collision_safe()",),
        verification: Sequence[str] = (),
        failure_modes: Sequence[str] = (
            "target_not_found",
            "execution_timeout",
            "force_limit",
        ),
        deletes: Sequence[str] = (),
    ) -> None:
        super().__init__(
            contract_type="your_contract",
            task=task,
            target=target,
            target_parameter="target",
            parameters=(
                ContractParameter(
                    "target",
                    ParameterType.ENTITY,
                    "Symbolic target resolved by the environment adapter.",
                ),
            ),
            preconditions=(
                ("ready_for_your_contract({target})",)
                if preconditions is None
                else preconditions
            ),
            effects=(
                ("your_contract_done({target})",) if effects is None else effects
            ),
            invariants=invariants,
            verification=verification,
            failure_modes=failure_modes,
            deletes=deletes,
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            description="Template for a user-defined environment-specific contract.",
        )
