"""Registry of contracts, policies, and the bindings between them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from mshab.skills.model import (
    CONTRACT_CLASSES,
    ArtifactStatus,
    CheckpointPolicy,
    Contract,
    ContractType,
    Policy,
)


class ContractLibrary:
    """In-memory owner of contracts, policies, and which policy executes which contract.

    Contracts and policies are registered independently and then *bound*.
    The binding relation is many-to-many:

    - one contract may be executed by several policies, for example the RL,
      BC and DP checkpoints that all execute ``pick.all``;
    - one policy may execute several contracts, for example the ``pick.all``
      checkpoint, which can pick any object and therefore also executes
      ``pick.013_apple`` and ``pick.024_bowl``.

    A contract's bindings are ordered.  :meth:`select_policy` returns the
    first ready one unless a policy id is requested explicitly, so binding
    order is the default preference order.
    """

    def __init__(
        self,
        contracts: Iterable[Contract] = (),
        policies: Iterable[Policy] = (),
        bindings: Iterable[Tuple[str, str]] = (),
    ) -> None:
        self._contracts: Dict[str, Contract] = {}
        self._policies: Dict[str, Policy] = {}
        # contract id -> policy ids in preference order
        self._bindings: Dict[str, List[str]] = {}
        for contract in contracts:
            self.register(contract)
        for policy in policies:
            self.register_policy(policy)
        for contract_id, policy_id in bindings:
            self.bind(contract_id, policy_id)

    # -- contracts ----------------------------------------------------------

    def register(self, contract: Contract) -> None:
        if contract.id in self._contracts:
            raise ValueError("duplicate contract id {!r}".format(contract.id))
        self._contracts[contract.id] = contract
        self._bindings[contract.id] = []

    def get(self, contract_id: str) -> Contract:
        try:
            return self._contracts[contract_id]
        except KeyError as exc:
            raise KeyError(
                "unknown contract {!r}; available={}".format(
                    contract_id, sorted(self._contracts)
                )
            ) from exc

    def find(
        self,
        task: Optional[str] = None,
        contract_type: Optional[Union[ContractType, str]] = None,
        target: Optional[str] = None,
        ready: Optional[bool] = None,
    ) -> List[Contract]:
        result = []
        for contract in self._contracts.values():
            if task is not None and contract.task != task:
                continue
            if ready is not None and self.ready(contract.id) != ready:
                continue
            if contract_type is not None:
                requested_type = (
                    contract_type.value
                    if isinstance(contract_type, ContractType)
                    else str(contract_type)
                )
                if contract.contract_type_name != requested_type:
                    continue
            if target is not None and contract.target != target:
                continue
            result.append(contract)
        return sorted(result, key=lambda item: item.id)

    # -- policies -----------------------------------------------------------

    def register_policy(self, policy: Policy) -> None:
        if policy.id in self._policies:
            raise ValueError("duplicate policy id {!r}".format(policy.id))
        self._policies[policy.id] = policy

    def policy(self, policy_id: str) -> Policy:
        try:
            return self._policies[policy_id]
        except KeyError as exc:
            raise KeyError(
                "unknown policy {!r}; available={}".format(
                    policy_id, sorted(self._policies)
                )
            ) from exc

    @property
    def policies(self) -> Tuple[Policy, ...]:
        """Every registered policy, sorted by id."""

        return tuple(self._policies[policy_id] for policy_id in sorted(self._policies))

    # -- bindings -----------------------------------------------------------

    def bind(self, contract_id: str, policy_id: str) -> None:
        """Record that ``policy_id`` executes ``contract_id``.

        Both must be registered.  Order matters: a contract's first binding is
        its default policy.
        """

        contract = self.get(contract_id)
        policy = self.policy(policy_id)
        bound = self._bindings[contract.id]
        if policy.id in bound:
            raise ValueError(
                "policy {!r} is already bound to contract {!r}".format(
                    policy.id, contract.id
                )
            )
        bound.append(policy.id)

    def policies_for(self, contract_id: str) -> Tuple[Policy, ...]:
        """The policies bound to one contract, in preference order."""

        contract = self.get(contract_id)
        return tuple(
            self._policies[policy_id] for policy_id in self._bindings[contract.id]
        )

    def contracts_for(self, policy_id: str) -> Tuple[Contract, ...]:
        """Every contract one policy is bound to, sorted by contract id."""

        policy = self.policy(policy_id)
        return tuple(
            self._contracts[contract_id]
            for contract_id in sorted(self._bindings)
            if policy.id in self._bindings[contract_id]
        )

    def ready(self, contract_id: str) -> bool:
        """Whether at least one policy bound to the contract is fully available."""

        return any(
            policy.status == ArtifactStatus.READY
            for policy in self.policies_for(contract_id)
        )

    def select_policy(
        self, contract_id: str, policy_id: Optional[str] = None
    ) -> Policy:
        """Choose the policy that will execute one contract.

        An explicit ``policy_id`` must be bound to the contract.  Otherwise
        the first ready policy in binding order is returned.
        """

        contract = self.get(contract_id)
        bound = self.policies_for(contract.id)
        if policy_id is not None:
            for policy in bound:
                if policy.id == policy_id:
                    return policy
            raise KeyError(
                "policy {!r} is not bound to contract {!r}; bound={}".format(
                    policy_id, contract.id, [policy.id for policy in bound]
                )
            )
        for policy in bound:
            if policy.status == ArtifactStatus.READY:
                return policy
        raise RuntimeError(
            "contract {} has no ready policy; bound={}".format(
                contract.id, [policy.id for policy in bound]
            )
        )

    def bind_generic_policies(self) -> None:
        """Bind each ``all``-target contract's policies to its specialized siblings.

        MS-HAB's ``<type>/all`` checkpoints are trained over every object of
        the task, so a policy that executes ``pick.all`` also executes
        ``pick.013_apple``.  The generic policies are appended *after* a
        contract's own bindings: a specialized checkpoint stays the default
        and the generic one is the Layer-4 fallback when it is missing.
        Calling this twice is harmless.
        """

        for generic in self.find(target="all"):
            siblings = self.find(
                task=generic.task, contract_type=generic.contract_type_name
            )
            for contract in siblings:
                if contract.id == generic.id:
                    continue
                for policy in self.policies_for(generic.id):
                    if policy.id not in self._bindings[contract.id]:
                        self.bind(contract.id, policy.id)

    # -- export and discovery -----------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        contracts = []
        for contract in self.find():
            record = contract.as_dict()
            record["policies"] = [
                policy.id for policy in self.policies_for(contract.id)
            ]
            record["ready"] = self.ready(contract.id)
            contracts.append(record)
        policies = []
        for policy in self.policies:
            record = policy.as_dict()
            record["contracts"] = [
                contract.id for contract in self.contracts_for(policy.id)
            ]
            policies.append(record)
        return {
            "schema_version": "mshab.contract-library.v1",
            "contracts": contracts,
            "policies": policies,
        }

    def save_index(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_checkpoint_root(cls, root: Path) -> "ContractLibrary":
        """Discover ``family/task/type/target/{config.yml,policy.pt}`` leaves.

        Every leaf becomes one :class:`CheckpointPolicy` bound to the contract
        its path names.  A leaf is registered even when just one artifact is
        present, allowing callers to report ``partial`` downloads instead of
        silently hiding them.  Afterwards :meth:`bind_generic_policies` binds
        each ``all`` checkpoint to the specialized contracts of its type.
        """

        checkpoint_root = Path(root)
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(
                "checkpoint root does not exist: {}".format(checkpoint_root)
            )

        library = cls()
        leaves = sorted(
            {
                path.parent
                for artifact_name in ("policy.pt", "config.yml")
                for path in checkpoint_root.glob("*/*/*/*/{}".format(artifact_name))
            }
        )
        for leaf in leaves:
            family, task, raw_type, target = leaf.relative_to(checkpoint_root).parts
            try:
                contract_type = ContractType(raw_type)
            except ValueError:
                continue
            contract_id = "mshab.{}.{}.{}".format(task, contract_type.value, target)
            if contract_id not in library._contracts:
                contract_class = CONTRACT_CLASSES[contract_type]
                library.register(contract_class(task=task, target=target))
            policy = CheckpointPolicy.from_leaf(
                checkpoint_root, family, task, contract_type, target
            )
            library.register_policy(policy)
            library.bind(contract_id, policy.id)
        library.bind_generic_policies()
        return library
