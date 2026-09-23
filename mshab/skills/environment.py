"""Environment boundary for grounded contracts and the policies that execute them.

Layers 1 and 2 must be usable without importing a simulator.  This module is
the explicit Layer-3/4 bridge: an adapter converts environment observations
and ``info`` dictionaries into canonical symbolic facts, and exposes reset /
step without leaking a concrete Gym or ManiSkill class into the graph model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, FrozenSet, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class EnvironmentEntity:
    """A symbolic entity and its environment-specific name/metadata."""

    name: str
    kind: str
    environment_name: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.kind or not self.environment_name:
            raise ValueError("environment entity fields must be non-empty")
        object.__setattr__(
            self, "attributes", MappingProxyType(dict(self.attributes))
        )


@dataclass(frozen=True)
class EnvironmentDescription:
    """Environment identity and the symbolic-to-simulator entity mapping."""

    environment_id: str
    scene_id: Optional[str] = None
    entities: Mapping[str, EnvironmentEntity] = field(default_factory=dict)
    compatible_contract_env_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.environment_id:
            raise ValueError("environment_id must be non-empty")
        entities = dict(self.entities)
        if any(key != entity.name for key, entity in entities.items()):
            raise ValueError("environment entity keys must match entity.name")
        object.__setattr__(self, "entities", MappingProxyType(entities))
        compatible = tuple(dict.fromkeys(self.compatible_contract_env_ids))
        if self.environment_id not in compatible:
            compatible = (self.environment_id,) + compatible
        object.__setattr__(self, "compatible_contract_env_ids", compatible)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def supports_contract_env(self, env_id: str) -> bool:
        return env_id in self.compatible_contract_env_ids


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """One adapter-normalized environment state used by contract checks."""

    observation: Any = field(repr=False)
    info: Mapping[str, Any]
    facts: FrozenSet[str]
    step_index: int
    reward: Any = None
    terminated: Any = False
    truncated: Any = False

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))
        object.__setattr__(self, "facts", frozenset(self.facts))


class EnvironmentAdapter(ABC):
    """Simulator-neutral interface consumed by Layer 3 and Layer 4.

    A concrete adapter owns four translations:

    1. symbolic entity name -> simulator object/name;
    2. observation/info -> canonical predicate facts;
    3. reset/step -> normalized :class:`EnvironmentSnapshot`;
    4. contract environment id -> compatibility decision.
    """

    @property
    @abstractmethod
    def description(self) -> EnvironmentDescription:
        pass

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> EnvironmentSnapshot:
        pass

    @abstractmethod
    def snapshot(self) -> EnvironmentSnapshot:
        pass

    @abstractmethod
    def step(self, action: Any) -> EnvironmentSnapshot:
        pass

    def supports_contract_env(self, env_id: str) -> bool:
        return self.description.supports_contract_env(env_id)

    def resolve_entity(self, symbolic_name: str) -> EnvironmentEntity:
        try:
            return self.description.entities[symbolic_name]
        except KeyError as exc:
            raise KeyError(
                "environment {!r} has no entity {!r}; available={}".format(
                    self.description.environment_id,
                    symbolic_name,
                    sorted(self.description.entities),
                )
            ) from exc

    def close(self) -> None:
        """Release environment resources when the adapter owns any."""


FactExtractor = Callable[
    [Any, Mapping[str, Any], EnvironmentDescription], Iterable[str]
]
EntityExtractor = Callable[
    [Any, Any, Mapping[str, Any]], Iterable[EnvironmentEntity]
]


class MSHabEnvironmentAdapter(EnvironmentAdapter):
    """Thin adapter around an already-created MS-HAB/Gym environment.

    This class deliberately does not import Gym, Torch, or ManiSkill.  The
    caller injects environment-specific entity and fact extractors.  That
    keeps graph construction CPU-only while still allowing a real vectorized
    MS-HAB environment to provide tensors in ``observation`` and ``info``.
    """

    def __init__(
        self,
        env: Any,
        environment_id: str,
        fact_extractor: FactExtractor,
        *,
        scene_id: Optional[str] = None,
        entities: Iterable[EnvironmentEntity] = (),
        entity_extractor: Optional[EntityExtractor] = None,
        compatible_contract_env_ids: Iterable[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.env = env
        self._fact_extractor = fact_extractor
        self._entity_extractor = entity_extractor
        entity_map = {entity.name: entity for entity in entities}
        self._description = EnvironmentDescription(
            environment_id=environment_id,
            scene_id=scene_id,
            entities=entity_map,
            compatible_contract_env_ids=tuple(compatible_contract_env_ids),
            metadata=metadata or {},
        )
        self._snapshot: Optional[EnvironmentSnapshot] = None
        self._step_index = 0

    @property
    def description(self) -> EnvironmentDescription:
        return self._description

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> EnvironmentSnapshot:
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = dict(options)
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        self._step_index = 0
        self._refresh_entities(observation, info)
        self._snapshot = self._make_snapshot(observation, info)
        return self._snapshot

    def snapshot(self) -> EnvironmentSnapshot:
        if self._snapshot is None:
            raise RuntimeError("environment adapter must be reset before snapshot()")
        return self._snapshot

    def step(self, action: Any) -> EnvironmentSnapshot:
        result = self.env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("env.step(action) must return a Gymnasium 5-tuple")
        observation, reward, terminated, truncated, info = result
        self._step_index += 1
        self._snapshot = self._make_snapshot(
            observation,
            info,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        return self._snapshot

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def _refresh_entities(
        self, observation: Any, info: Mapping[str, Any]
    ) -> None:
        if self._entity_extractor is None:
            return
        entities = {
            entity.name: entity
            for entity in self._entity_extractor(self.env, observation, info)
        }
        self._description = EnvironmentDescription(
            environment_id=self._description.environment_id,
            scene_id=self._description.scene_id,
            entities=entities,
            compatible_contract_env_ids=self._description.compatible_contract_env_ids,
            metadata=self._description.metadata,
        )

    def _make_snapshot(
        self,
        observation: Any,
        info: Mapping[str, Any],
        *,
        reward: Any = None,
        terminated: Any = False,
        truncated: Any = False,
    ) -> EnvironmentSnapshot:
        facts = self._fact_extractor(observation, info, self._description)
        return EnvironmentSnapshot(
            observation=observation,
            info=info,
            facts=frozenset(facts),
            step_index=self._step_index,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
