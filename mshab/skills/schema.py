"""Strict deserialization primitives for untrusted graph documents.

Layer 1 and Layer 2 objects are constructed from dictionaries in two cases: a
human editing a checked-in catalog, and a future VLM emitting a structured
``SkillGraphPatch``.  Neither source is trusted, so every ``from_dict`` in this
package funnels through the helpers below instead of calling ``dict.get`` and
hoping.  The rules are deliberately narrow:

- a payload must be a mapping with no unknown keys;
- identifiers use a closed character set, so they cannot smuggle path
  separators or predicate punctuation into a graph;
- skill-node arguments are scalars only, which keeps ``SkillNode.arguments``
  genuinely immutable and keeps grounded predicates well formed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "mshab.skill-graph.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SKILL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")

#: Scalars that may appear as a ``SkillNode`` argument value.
SCALAR_TYPES = (str, int, float, bool)


class SchemaError(ValueError):
    """Raised when a serialized document does not match the expected schema."""


def require_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(
            "{} must be an object, got {}".format(where, type(value).__name__)
        )
    bad_keys = sorted(key for key in value if not isinstance(key, str))
    if bad_keys:
        raise SchemaError("{} has non-string keys {}".format(where, bad_keys))
    return value


def require_keys(
    payload: Mapping[str, Any],
    *,
    where: str,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
) -> None:
    """Reject both missing required keys and any key outside the schema."""

    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing or unknown:
        raise SchemaError(
            "{} has missing={} unknown={} (allowed={})".format(
                where, missing, unknown, sorted(allowed)
            )
        )


def require_str(payload: Mapping[str, Any], key: str, *, where: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SchemaError(
            "{}.{} must be a non-empty string, got {!r}".format(where, key, value)
        )
    return value


def optional_str(
    payload: Mapping[str, Any], key: str, *, where: str, default: str = ""
) -> str:
    if key not in payload or payload[key] is None:
        return default
    value = payload[key]
    if not isinstance(value, str):
        raise SchemaError(
            "{}.{} must be a string, got {}".format(where, key, type(value).__name__)
        )
    return value


def require_identifier(payload: Mapping[str, Any], key: str, *, where: str) -> str:
    value = require_str(payload, key, where=where)
    if not _IDENTIFIER.match(value):
        raise SchemaError(
            "{}.{} {!r} is not a valid identifier "
            "(letters, digits, '_', '.', '-'; max 128 chars)".format(where, key, value)
        )
    return value


def optional_identifier(
    payload: Mapping[str, Any], key: str, *, where: str
) -> Optional[str]:
    """An identifier that may be absent or explicitly ``null``."""

    if key not in payload or payload[key] is None:
        return None
    return require_identifier(payload, key, where=where)


def require_contract_id(payload: Mapping[str, Any], key: str, *, where: str) -> str:
    value = require_str(payload, key, where=where)
    if not _SKILL_ID.match(value):
        raise SchemaError(
            "{}.{} {!r} is not a valid contract id".format(where, key, value)
        )
    return value


def require_sequence(
    payload: Mapping[str, Any], key: str, *, where: str
) -> Sequence[Any]:
    value = payload.get(key, ())
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise SchemaError(
            "{}.{} must be an array, got {}".format(where, key, type(value).__name__)
        )
    return value


def require_str_tuple(
    payload: Mapping[str, Any], key: str, *, where: str
) -> Tuple[str, ...]:
    items = require_sequence(payload, key, where=where)
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item:
            raise SchemaError(
                "{}.{}[{}] must be a non-empty string, got {!r}".format(
                    where, key, index, item
                )
            )
    return tuple(items)


def require_arguments(
    payload: Mapping[str, Any], key: str, *, where: str
) -> Dict[str, Any]:
    """Skill-node arguments: a flat mapping of identifier -> scalar.

    Nested containers are rejected.  ``MappingProxyType`` is shallow, so a
    nested list inside a frozen ``SkillNode`` would still be mutable; refusing
    them at the boundary is what makes Layer-2 nodes actually immutable.
    """

    raw = payload.get(key, {})
    values = require_mapping(raw, where="{}.{}".format(where, key))
    result: Dict[str, Any] = {}
    for name, value in values.items():
        if not _IDENTIFIER.match(name):
            raise SchemaError(
                "{}.{} key {!r} is not a valid identifier".format(where, key, name)
            )
        if not isinstance(value, SCALAR_TYPES):
            raise SchemaError(
                "{}.{}.{} must be a scalar ({}), got {}".format(
                    where,
                    key,
                    name,
                    "/".join(item.__name__ for item in SCALAR_TYPES),
                    type(value).__name__,
                )
            )
        result[name] = value
    return result


def require_schema_version(payload: Mapping[str, Any], *, where: str) -> None:
    """Accept a document with no version, but reject a version we do not know."""

    version = payload.get("schema_version")
    if version is None:
        return
    if version != SCHEMA_VERSION:
        raise SchemaError(
            "{} declares unsupported schema_version {!r}; expected {!r}".format(
                where, version, SCHEMA_VERSION
            )
        )
