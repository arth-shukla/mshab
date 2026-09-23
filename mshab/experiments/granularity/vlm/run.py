"""Run DeepSeek over the four-layer graph and compare with official MS-HAB GT.

The first model call receives no ground-truth-derived data. It returns only
strategy occurrences and bindings. This module validates graph references,
compiles those choices through the authoritative graph, renders a flowchart,
and only then loads GT for deterministic metrics and an optional judge call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import string
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

from mshab.experiments.granularity.vlm.prompt import (
    DECISION_SCHEMA_VERSION,
    FLOW_SCHEMA_VERSION,
    JUDGE_SCHEMA_VERSION,
    build_judge_messages,
    build_planner_messages,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GRAPH = PACKAGE_DIR.parent / "artifacts" / "coarse_four_layers.json"
DEFAULT_CONFIG = PACKAGE_DIR / "config.yaml"
DEFAULT_OUTPUT_ROOT = PACKAGE_DIR / "outputs"
DEFAULT_ASSET_ROOT = Path(
    os.environ.get(
        "MS_ASSET_DIR",
        str(REPOSITORY_ROOT.parent / "mshab-assets"),
    )
)

TASK_FAMILIES = ("set_table", "prepare_groceries", "tidy_house")
EDGE_RELATIONS = frozenset(("next", "on_failure", "rejoins"))
SUBGRAPH_RELATIONS = frozenset(
    ("enables", "requires", "fallback_to", "alternative_to")
)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("{} must contain a JSON object".format(path))
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _object_category(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return re.sub(r"-\d+$", "", str(value))


def _entity_aliases(value: Optional[str]) -> frozenset[str]:
    """Return conservative aliases for one MS-HAB entity identifier.

    Official plans use dataset categories such as ``024_bowl`` and runtime
    instances such as ``024_bowl-4``. Natural-language planners commonly bind
    the same entity as ``bowl``. Grounding owns this dataset-specific
    translation, so all three forms compare as the same semantic entity.
    """

    category = _object_category(value)
    if category is None:
        return frozenset()
    normalized = re.sub(r"[\s-]+", "_", category.strip().lower())
    aliases = {normalized}
    semantic = re.sub(r"^\d+_", "", normalized)
    if semantic:
        aliases.add(semantic)
    return frozenset(aliases)


def _entities_match(left: Optional[str], right: Optional[str]) -> bool:
    """Whether two exact or semantic MS-HAB entity names denote one entity."""

    return bool(_entity_aliases(left) & _entity_aliases(right))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be an object".format(label))
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(label))
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            "{} fields mismatch; missing={} unexpected={}".format(
                label,
                sorted(expected - actual),
                sorted(actual - expected),
            )
        )


@dataclass(frozen=True)
class DeepSeekConfig:
    """Validated local API configuration; its secret is never serialized."""

    base_url: str
    model: str
    api_key_env: str
    api_key: str = field(repr=False)
    thinking: str
    reasoning_effort: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    retries: int

    @classmethod
    def from_yaml(cls, path: Path) -> "DeepSeekConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        root = _require_mapping(payload, "configuration")
        data = _require_mapping(root.get("deepseek"), "deepseek configuration")
        config = cls(
            base_url=str(data.get("base_url", "https://api.deepseek.com")),
            model=str(data.get("model", "deepseek-flash")),
            api_key_env=str(data.get("api_key_env", "DEEPSEEK_API_KEY")),
            api_key=str(data.get("api_key", "")),
            thinking=str(data.get("thinking", "enabled")),
            reasoning_effort=str(data.get("reasoning_effort", "high")),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 8192)),
            timeout_seconds=float(data.get("timeout_seconds", 120)),
            retries=int(data.get("retries", 2)),
        )
        if not config.base_url.startswith("https://"):
            raise ValueError("DeepSeek base_url must use HTTPS")
        if not config.model or not config.api_key_env:
            raise ValueError("model and api_key_env must be non-empty")
        if config.thinking not in ("enabled", "disabled"):
            raise ValueError("thinking must be 'enabled' or 'disabled'")
        if config.reasoning_effort not in ("low", "high", "max"):
            raise ValueError("reasoning_effort must be low, high, or max")
        if config.max_tokens <= 0 or config.timeout_seconds <= 0:
            raise ValueError("max_tokens and timeout_seconds must be positive")
        if config.retries < 0:
            raise ValueError("retries cannot be negative")
        inline_key = config.api_key.strip()
        if (
            inline_key
            and not inline_key.startswith("PASTE_")
            and Path(path).stat().st_mode & 0o077
        ):
            raise PermissionError(
                "config with an inline API key must be private; run chmod 600 {}".format(
                    path
                )
            )
        return config

    def resolved_api_key(self) -> str:
        """Prefer the environment so a token need not be written to disk."""

        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            value = self.api_key.strip()
        if not value or value.startswith("PASTE_"):
            raise RuntimeError(
                "DeepSeek API key is missing; set {} or edit the ignored config.yaml".format(
                    self.api_key_env
                )
            )
        return value

    def request_payload(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [dict(item) for item in messages],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
            "thinking": {"type": self.thinking},
        }
        if self.thinking == "enabled":
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = self.temperature
        return payload


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so a bearer token can never cross origins."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


class DeepSeekClient:
    """Small OpenAI-compatible client implemented with the standard library."""

    def __init__(self, config: DeepSeekConfig) -> None:
        self.config = config
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload = self.config.request_payload(messages)
        body = json.dumps(payload).encode("utf-8")
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        last_error: Optional[Exception] = None
        for attempt in range(self.config.retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": "Bearer {}".format(
                        self.config.resolved_api_key()
                    ),
                    "Content-Type": "application/json",
                },
            )
            try:
                with self._opener.open(
                    request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    raw_response = _require_mapping(
                        json.loads(response.read().decode("utf-8")),
                        "DeepSeek response",
                    )
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt >= self.config.retries:
                    break
                time.sleep(2**attempt)
                continue
            try:
                result, metadata = _parse_chat_completion(raw_response)
                if not result:
                    raise ValueError("DeepSeek returned an empty JSON object")
                return result, metadata
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                # A server response is one experimental sample. Retrying a
                # truncated or schema-invalid model response would silently
                # sample until success and bias validity metrics upward.
                raise RuntimeError(
                    "DeepSeek response was unusable: {}".format(exc)
                ) from exc
        raise RuntimeError("DeepSeek request failed: {}".format(last_error))


def _parse_json_content(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("model response must be one JSON object")
    return payload


def _parse_chat_completion(
    response: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("DeepSeek response must contain a non-empty choices list")
    choice = _require_mapping(choices[0], "DeepSeek choice")
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise ValueError("DeepSeek JSON was truncated at max_tokens")
    message = _require_mapping(choice.get("message"), "DeepSeek message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek returned empty content")
    return _parse_json_content(content), {
        "id": response.get("id"),
        "model": response.get("model"),
        "finish_reason": finish_reason,
        "usage": response.get("usage", {}),
    }


def load_response_file(path: Path) -> Dict[str, Any]:
    """Accept either a direct model JSON object or a saved API response."""

    payload = _read_json(path)
    if "choices" in payload:
        return _parse_chat_completion(payload)[0]
    return payload


class FourLayerGraphIndex:
    """Read-only compiler view over the existing four-layer JSON artifact."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        if document.get("scope") != "layers_1_through_4":
            raise ValueError("expected a complete four-layer graph")
        self.document = document
        self.subgraphs: Dict[str, Mapping[str, Any]] = {}
        for item in document["layer2"]["subgraphs"]:
            subgoal_id = item["subgoal_id"]
            if subgoal_id in self.subgraphs:
                raise ValueError("duplicate SkillSubgraph {}".format(subgoal_id))
            self.subgraphs[subgoal_id] = item

        layer1_subgoals = [item["id"] for item in document["layer1"]["subgoals"]]
        if len(layer1_subgoals) != len(set(layer1_subgoals)):
            raise ValueError("Layer 1 SubGoal ids must be unique")
        if set(layer1_subgoals) != set(self.subgraphs):
            raise ValueError("every Layer 1 SubGoal must own exactly one SkillSubgraph")

        self.nodes: Dict[str, Mapping[str, Any]] = {}
        self.node_subgoal: Dict[str, str] = {}
        for subgoal_id, subgraph in self.subgraphs.items():
            local_node_ids = [item["id"] for item in subgraph["nodes"]]
            if len(local_node_ids) != len(set(local_node_ids)):
                raise ValueError(
                    "duplicate SkillNode inside {}".format(subgoal_id)
                )
            if set(subgraph["execution_order"]) != set(local_node_ids):
                raise ValueError(
                    "execution_order must contain every {} SkillNode once".format(
                        subgoal_id
                    )
                )
            if len(subgraph["execution_order"]) != len(local_node_ids):
                raise ValueError("execution_order cannot contain duplicates")
            for node in subgraph["nodes"]:
                node_id = node["id"]
                if node_id in self.nodes:
                    raise ValueError("duplicate graph node {}".format(node_id))
                self.nodes[node_id] = node
                self.node_subgoal[node_id] = subgoal_id

            local_nodes = set(local_node_ids)
            seen_edges = set()
            for edge in subgraph["edges"]:
                relation = edge["relation"]
                source = edge["source"]
                target = edge["target"]
                if relation not in SUBGRAPH_RELATIONS:
                    raise ValueError(
                        "unknown Layer 2 relation {}".format(relation)
                    )
                if source not in local_nodes or target not in local_nodes:
                    raise ValueError(
                        "Layer 2 edge endpoints must stay inside {}".format(
                            subgoal_id
                        )
                    )
                edge_key = (source, target, relation)
                if edge_key in seen_edges:
                    raise ValueError("duplicate Layer 2 edge {}".format(edge_key))
                seen_edges.add(edge_key)

        self.contracts: Dict[str, Mapping[str, Any]] = {}
        for item in document["layer3"]["contracts"]:
            contract_id = item["id"]
            if contract_id in self.contracts:
                raise ValueError("duplicate Contract {}".format(contract_id))
            self.contracts[contract_id] = item
        for node_id, node in self.nodes.items():
            if node["contract_id"] not in self.contracts:
                raise ValueError(
                    "SkillNode {} references an unknown Contract".format(node_id)
                )

        explicit_references: Dict[str, str] = {}
        for reference in document["layer2_to_layer3"]["references"]:
            if reference.get("relation") != "references":
                raise ValueError("Layer 2 to Layer 3 relation must be references")
            source = reference["source"]
            target = reference["target"]
            if source in explicit_references:
                raise ValueError(
                    "SkillNode {} has more than one Contract reference".format(
                        source
                    )
                )
            if source not in self.nodes or target not in self.contracts:
                raise ValueError("invalid SkillNode-to-Contract reference")
            if self.nodes[source]["contract_id"] != target:
                raise ValueError(
                    "embedded and explicit Contract references disagree for {}".format(
                        source
                    )
                )
            explicit_references[source] = target
        if set(explicit_references) != set(self.nodes):
            raise ValueError("every SkillNode must reference exactly one Contract")
        self.node_contract = explicit_references

        self.strategies: Dict[str, Mapping[str, Any]] = {}
        self.strategies_by_subgoal: Dict[str, List[Mapping[str, Any]]] = {
            subgoal_id: [] for subgoal_id in self.subgraphs
        }
        for subgoal_id, items in document["layer2"][
            "semantic_strategies"
        ].items():
            if subgoal_id not in self.subgraphs:
                raise ValueError("strategy group has no matching SkillSubgraph")
            for strategy in items:
                if strategy["subgoal_id"] != subgoal_id:
                    raise ValueError("strategy is stored under the wrong SubGoal")
                strategy_id = strategy["id"]
                if strategy_id in self.strategies:
                    raise ValueError(
                        "duplicate SemanticStrategy {}".format(strategy_id)
                    )
                node_ids = strategy["node_ids"]
                if not node_ids or len(node_ids) != len(set(node_ids)):
                    raise ValueError(
                        "strategy {} must select unique SkillNodes".format(
                            strategy_id
                        )
                    )
                if any(self.node_subgoal.get(item) != subgoal_id for item in node_ids):
                    raise ValueError(
                        "strategy {} contains a foreign SkillNode".format(strategy_id)
                    )
                primary = strategy["primary_achiever"]
                fallback = strategy["fallback_achiever"]
                if primary not in node_ids or fallback not in node_ids:
                    raise ValueError(
                        "strategy achievers must be selected SkillNodes"
                    )
                fallback_edges = {
                    (edge["source"], edge["target"])
                    for edge in self.subgraphs[subgoal_id]["edges"]
                    if edge["relation"] == "fallback_to"
                }
                if (primary, fallback) not in fallback_edges:
                    raise ValueError(
                        "strategy {} has no matching fallback_to edge".format(
                            strategy_id
                        )
                    )
                self.strategies[strategy_id] = strategy
                self.strategies_by_subgoal[subgoal_id].append(strategy)

        self.alternative_edges: Dict[str, List[Mapping[str, Any]]] = {}
        for subgoal_id, subgraph in self.subgraphs.items():
            edges = [
                edge
                for edge in subgraph["edges"]
                if edge["relation"] == "alternative_to"
            ]
            self.alternative_edges[subgoal_id] = edges
            primary_ids = {
                strategy["primary_achiever"]
                for strategy in self.strategies_by_subgoal[subgoal_id]
            }
            if any(
                edge["source"] not in primary_ids
                or edge["target"] not in primary_ids
                for edge in edges
            ):
                raise ValueError(
                    "alternative_to edges must connect strategy primary achievers"
                )
            if len(primary_ids) > 1 and not self._is_connected_alternative_graph(
                primary_ids,
                edges,
            ):
                raise ValueError(
                    "semantic alternatives for {} are disconnected".format(
                        subgoal_id
                    )
                )

        self.policies: Dict[str, Mapping[str, Any]] = {}
        for item in document["layer4"]["policies"]:
            policy_id = item["id"]
            if policy_id in self.policies:
                raise ValueError("duplicate Policy {}".format(policy_id))
            self.policies[policy_id] = item
        self.policy_contract: Dict[str, str] = {}
        for connection in document["layer4_to_layer3"]["connections"]:
            if connection.get("relation") != "EXECUTES":
                raise ValueError("Layer 4 to Layer 3 relation must be EXECUTES")
            policy_id = connection["source_policy_id"]
            contract_id = connection["target_contract_id"]
            if policy_id in self.policy_contract:
                raise ValueError(
                    "Policy {} has more than one EXECUTES connection".format(
                        policy_id
                    )
                )
            if policy_id not in self.policies or contract_id not in self.contracts:
                raise ValueError("invalid Policy-to-Contract connection")
            if (
                self.policies[policy_id]["contract_type"]
                != self.contracts[contract_id]["contract_type"]
            ):
                raise ValueError(
                    "Policy and Contract types disagree for {}".format(policy_id)
                )
            self.policy_contract[policy_id] = contract_id
        if set(self.policy_contract) != set(self.policies):
            raise ValueError("every Policy must have one EXECUTES connection")

    @staticmethod
    def _is_connected_alternative_graph(
        node_ids: Set[str],
        edges: Sequence[Mapping[str, Any]],
    ) -> bool:
        if not node_ids:
            return True
        adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in node_ids}
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            adjacency[source].add(target)
            adjacency[target].add(source)
        visited: Set[str] = set()
        frontier = [next(iter(node_ids))]
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            frontier.extend(adjacency[current] - visited)
        return visited == node_ids

    def _alternative_path(
        self,
        subgoal_id: str,
        source: str,
        target: str,
    ) -> List[Dict[str, Any]]:
        """Return authoritative Layer-2 provenance between two alternatives."""

        if source == target:
            return []
        adjacency: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = {}
        for edge in self.alternative_edges[subgoal_id]:
            adjacency.setdefault(edge["source"], []).append((edge["target"], edge))
            adjacency.setdefault(edge["target"], []).append((edge["source"], edge))
        frontier: List[Tuple[str, List[Dict[str, Any]]]] = [(source, [])]
        visited: Set[str] = set()
        while frontier:
            current, path = frontier.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for neighbour, edge in sorted(
                adjacency.get(current, []),
                key=lambda item: item[0],
            ):
                step = {
                    "source_skill_node_id": current,
                    "target_skill_node_id": neighbour,
                    "relation": "alternative_to",
                    "stored_layer2_edge": {
                        "source_skill_node_id": edge["source"],
                        "target_skill_node_id": edge["target"],
                        "relation": "alternative_to",
                    },
                }
                if neighbour == target:
                    return path + [step]
                if neighbour not in visited:
                    frontier.append((neighbour, path + [step]))
        raise ValueError(
            "no alternative_to path from {} to {}".format(source, target)
        )

    def required_bindings(self, strategy: Mapping[str, Any]) -> Set[str]:
        required: Set[str] = set()
        for node_id in strategy["node_ids"]:
            for value in self.nodes[node_id]["arguments"].values():
                if not isinstance(value, str):
                    continue
                for _, field_name, _, _ in string.Formatter().parse(value):
                    if field_name:
                        required.add(field_name)
        return required

    def _causal_order(
        self,
        strategy: Mapping[str, Any],
        achiever: str,
    ) -> List[str]:
        selected = set(strategy["node_ids"])
        subgraph = self.subgraphs[strategy["subgoal_id"]]
        prerequisites: Dict[str, Set[str]] = {
            node_id: set() for node_id in selected
        }
        causal_edges = []
        for edge in subgraph["edges"]:
            source = edge["source"]
            target = edge["target"]
            if source not in selected or target not in selected:
                continue
            if edge["relation"] == "enables":
                prerequisites[target].add(source)
                causal_edges.append((source, target))
            elif edge["relation"] == "requires":
                prerequisites[source].add(target)
                causal_edges.append((target, source))

        closure = {achiever}
        frontier = list(prerequisites[achiever])
        while frontier:
            current = frontier.pop()
            if current in closure:
                continue
            closure.add(current)
            frontier.extend(prerequisites[current] - closure)

        rank = {
            node_id: index
            for index, node_id in enumerate(subgraph["execution_order"])
        }
        indegree = {node_id: 0 for node_id in closure}
        successors: Dict[str, Set[str]] = {node_id: set() for node_id in closure}
        for source, target in causal_edges:
            if source in closure and target in closure and target not in successors[source]:
                successors[source].add(target)
                indegree[target] += 1
        ready = sorted(
            (item for item, degree in indegree.items() if degree == 0),
            key=lambda item: (rank.get(item, 10**9), item),
        )
        order = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for target in sorted(successors[current]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=lambda item: (rank.get(item, 10**9), item))
        if len(order) != len(closure):
            raise ValueError("causal cycle in strategy {}".format(strategy["id"]))
        return order

    def _ground_arguments(
        self,
        node_id: str,
        bindings: Mapping[str, str],
    ) -> Dict[str, Any]:
        grounded: Dict[str, Any] = {}
        for name, value in self.nodes[node_id]["arguments"].items():
            if isinstance(value, str):
                grounded[name] = value.format(**bindings)
            else:
                grounded[name] = value
        return grounded

    def _policies_for(
        self,
        task_family: str,
        contract_id: str,
    ) -> List[Mapping[str, Any]]:
        return sorted(
            (
                policy
                for policy_id, policy in self.policies.items()
                if policy["task_family"] == task_family
                and self.policy_contract[policy_id] == contract_id
            ),
            key=lambda item: item["id"],
        )

    def _select_policy(
        self,
        task_family: str,
        contract_id: str,
        contract_type: str,
        arguments: Mapping[str, Any],
    ) -> Tuple[str, List[str]]:
        candidates = self._policies_for(task_family, contract_id)
        if not candidates:
            raise ValueError(
                "no {} Policy executes {} for {}".format(
                    task_family, contract_id, contract_type
                )
            )
        target_name = {
            "navigate": "goal",
            "pick": "object",
            "place": "object",
            "open": "articulation",
            "close": "articulation",
        }[contract_type]
        target = _object_category(str(arguments.get(target_name, "")))
        exact = [item for item in candidates if item["target"] == target]
        generic = [item for item in candidates if item["target"] == "all"]
        selected = (exact or generic or candidates)[0]
        return selected["id"], [item["id"] for item in candidates]

    def _flow_node(
        self,
        *,
        flow_id: str,
        occurrence_id: str,
        strategy: Mapping[str, Any],
        skill_node_id: str,
        bindings: Mapping[str, str],
        task_family: str,
        kind: str,
    ) -> Dict[str, Any]:
        node = self.nodes[skill_node_id]
        contract_id = self.node_contract[skill_node_id]
        contract = self.contracts[contract_id]
        contract_type = contract["contract_type"]
        arguments = self._ground_arguments(skill_node_id, bindings)
        policy_id, candidates = self._select_policy(
            task_family,
            contract_id,
            contract_type,
            arguments,
        )
        return {
            "id": flow_id,
            "occurrence_id": occurrence_id,
            "strategy_id": strategy["id"],
            "subgoal_id": strategy["subgoal_id"],
            "skill_node_id": skill_node_id,
            "contract_id": contract_id,
            "contract_type": contract_type,
            "arguments": arguments,
            "grounded_target": _comparison_target(contract_type, arguments),
            "policy_id": policy_id,
            "candidate_policy_ids": candidates,
            "kind": kind,
            "grounding": {},
        }

    def compile_decision(
        self,
        decision: Mapping[str, Any],
        instruction: str,
        task_family: str,
        scene_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        validate_decision_shape(decision, instruction, task_family)
        occurrences = {
            item["id"]: item for item in decision["strategy_occurrences"]
        }
        order = decision["occurrence_order"]
        if len(occurrences) != len(decision["strategy_occurrences"]):
            raise ValueError("strategy occurrence ids must be unique")
        if len(order) != len(set(order)) or set(order) != set(occurrences):
            raise ValueError("occurrence_order must contain every occurrence once")

        nodes: List[Dict[str, Any]] = []
        occurrence_data: Dict[str, Dict[str, Any]] = {}
        for occurrence_id in order:
            occurrence = occurrences[occurrence_id]
            strategy_id = occurrence["strategy_id"]
            try:
                strategy = self.strategies[strategy_id]
            except KeyError as exc:
                raise ValueError(
                    "unknown strategy_id {!r}".format(strategy_id)
                ) from exc
            bindings_raw = _require_mapping(
                occurrence.get("bindings"),
                "bindings for {}".format(occurrence_id),
            )
            bindings = {
                str(key): _require_string(value, "binding {}".format(key))
                for key, value in bindings_raw.items()
            }
            required = self.required_bindings(strategy)
            if set(bindings) != required:
                raise ValueError(
                    "strategy {} requires bindings {}; got {}".format(
                        strategy_id,
                        sorted(required),
                        sorted(bindings),
                    )
                )

            nominal_skills = self._causal_order(
                strategy,
                strategy["primary_achiever"],
            )
            fallback_closure = self._causal_order(
                strategy,
                strategy["fallback_achiever"],
            )
            recovery_skills = [
                item for item in fallback_closure if item not in nominal_skills
            ]
            if strategy["fallback_achiever"] not in recovery_skills:
                raise ValueError("fallback achiever is not a distinct recovery node")

            nominal_ids = []
            recovery_ids = []
            for kind, skill_ids, destination in (
                ("nominal", nominal_skills, nominal_ids),
                ("recovery", recovery_skills, recovery_ids),
            ):
                for skill_node_id in skill_ids:
                    flow_id = "{}:{}".format(occurrence_id, skill_node_id)
                    destination.append(flow_id)
                    nodes.append(
                        self._flow_node(
                            flow_id=flow_id,
                            occurrence_id=occurrence_id,
                            strategy=strategy,
                            skill_node_id=skill_node_id,
                            bindings=bindings,
                            task_family=task_family,
                            kind=kind,
                        )
                    )
            occurrence_data[occurrence_id] = {
                "strategy": strategy,
                "bindings": bindings,
                "nominal_ids": nominal_ids,
                "recovery_ids": recovery_ids,
                "fallback_route_ids": [
                    "{}:{}".format(occurrence_id, skill_node_id)
                    for skill_node_id in fallback_closure
                ],
            }

        semantic_validation = _validate_semantic_order(
            order,
            occurrence_data,
            scene_context or {},
        )

        nominal_order = [
            node_id
            for occurrence_id in order
            for node_id in occurrence_data[occurrence_id]["nominal_ids"]
        ]
        edges = [
            {"source": source, "target": target, "relation": "next"}
            for source, target in zip(nominal_order, nominal_order[1:])
        ]
        fallback_branches = []
        for index, occurrence_id in enumerate(order):
            item = occurrence_data[occurrence_id]
            strategy = item["strategy"]
            failed_id = "{}:{}".format(
                occurrence_id,
                strategy["primary_achiever"],
            )
            recovery = item["recovery_ids"]
            edges.append(
                {
                    "source": failed_id,
                    "target": recovery[0],
                    "relation": "on_failure",
                }
            )
            edges.extend(
                {
                    "source": source,
                    "target": target,
                    "relation": "next",
                }
                for source, target in zip(recovery, recovery[1:])
            )
            rejoin = None
            if index + 1 < len(order):
                rejoin = occurrence_data[order[index + 1]]["nominal_ids"][0]
                edges.append(
                    {
                        "source": recovery[-1],
                        "target": rejoin,
                        "relation": "rejoins",
                    }
                )
            fallback_branches.append(
                {
                    "failed_node_id": failed_id,
                    "recovery_entry_node_id": recovery[0],
                    "fallback_achiever_node_id": "{}:{}".format(
                        occurrence_id,
                        strategy["fallback_achiever"],
                    ),
                    "source_layer2_relation": {
                        "source_skill_node_id": strategy["primary_achiever"],
                        "target_skill_node_id": strategy["fallback_achiever"],
                        "relation": "fallback_to",
                    },
                    "recovery_order": recovery,
                    "fallback_route_order": item["fallback_route_ids"],
                    "rejoin_node_id": rejoin,
                }
            )

        alternative_choices = []
        for occurrence_id in order:
            selected = occurrence_data[occurrence_id]["strategy"]
            candidates = []
            for candidate in sorted(
                self.strategies_by_subgoal[selected["subgoal_id"]],
                key=lambda value: value["id"],
            ):
                if candidate["id"] == selected["id"]:
                    continue
                candidates.append(
                    {
                        "strategy_id": candidate["id"],
                        "applicable_when": candidate["applicable_when"],
                        "primary_achiever_skill_node_id": candidate[
                            "primary_achiever"
                        ],
                        "source_layer2_relation_path": self._alternative_path(
                            selected["subgoal_id"],
                            selected["primary_achiever"],
                            candidate["primary_achiever"],
                        ),
                    }
                )
            alternative_choices.append(
                {
                    "occurrence_id": occurrence_id,
                    "subgoal_id": selected["subgoal_id"],
                    "selected_strategy_id": selected["id"],
                    "selected_applicable_when": selected["applicable_when"],
                    "selected_achiever_node_id": "{}:{}".format(
                        occurrence_id,
                        selected["primary_achiever"],
                    ),
                    "candidates": candidates,
                }
            )

        return {
            "schema_version": FLOW_SCHEMA_VERSION,
            "source": "deepseek_compiled",
            "instruction": instruction,
            "task_family": task_family,
            "metadata": {
                "decision_schema_version": decision["schema_version"],
                "decision_summary": decision["decision_summary"],
                "semantic_validation": semantic_validation,
            },
            "nodes": nodes,
            "edges": edges,
            "nominal_order": nominal_order,
            "fallback_branches": fallback_branches,
            "alternative_choices": alternative_choices,
            "selected_strategies": [
                {
                    "occurrence_id": occurrence_id,
                    "strategy_id": occurrences[occurrence_id]["strategy_id"],
                    "bindings": dict(occurrence_data[occurrence_id]["bindings"]),
                }
                for occurrence_id in order
            ],
        }


def _scene_object_names(scene_context: Mapping[str, Any]) -> Set[str]:
    values = scene_context.get("objects", [])
    if not isinstance(values, list):
        raise TypeError("scene_context.objects must be a list")
    names: Set[str] = set()
    for index, value in enumerate(values):
        if isinstance(value, str):
            names.add(value)
            category = _object_category(value)
            if category:
                names.add(category)
            continue
        item = _require_mapping(value, "scene object {}".format(index))
        for key in ("id", "instance_id", "category", "object", "name"):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate:
                names.add(candidate)
                category = _object_category(candidate)
                if category:
                    names.add(category)
    return names


def _validate_semantic_order(
    order: Sequence[str],
    occurrence_data: Mapping[str, Mapping[str, Any]],
    scene_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Report semantic plan defects without hiding a model's failed attempt."""

    issues: List[Dict[str, str]] = []
    retrieved: Set[str] = set()
    scene_objects = _scene_object_names(scene_context)
    scene_grounding_checked = bool(scene_objects)
    ordered_strategies = [
        occurrence_data[item]["strategy"] for item in order
    ]

    for index, occurrence_id in enumerate(order):
        item = occurrence_data[occurrence_id]
        strategy = item["strategy"]
        bindings = item["bindings"]
        subgoal_id = strategy["subgoal_id"]
        obj = bindings.get("object")
        if obj and scene_grounding_checked and obj not in scene_objects:
            issues.append(
                {
                    "code": "object_not_in_scene_context",
                    "occurrence_id": occurrence_id,
                    "message": "{} is absent from explicit scene context".format(
                        obj
                    ),
                }
            )
        if subgoal_id == "deliver" and obj not in retrieved:
            issues.append(
                {
                    "code": "deliver_before_retrieve",
                    "occurrence_id": occurrence_id,
                    "message": "{} is delivered before it is retrieved".format(obj),
                }
            )
        if subgoal_id == "retrieve" and obj:
            retrieved.add(obj)

        if subgoal_id == "restore" and strategy["id"].startswith("restore_"):
            container = strategy["id"].removeprefix("restore_")
            later_use = any(
                other["subgoal_id"] != "restore"
                and container in other["id"]
                for other in ordered_strategies[index + 1 :]
            )
            if later_use:
                issues.append(
                    {
                        "code": "restore_before_last_use",
                        "occurrence_id": occurrence_id,
                        "message": "{} is restored before its last use".format(
                            container
                        ),
                    }
                )

    return {
        "valid": not issues,
        "scope": "local_consistency_only",
        "instruction_satisfaction_checked": False,
        "issues": issues,
        "scene_grounding_checked": scene_grounding_checked,
        "checks": [
            "retrieve_before_deliver_per_object",
            "restore_after_last_matching_container_use",
            "object_membership_when_explicit_scene_context_is_supplied",
        ],
    }


def validate_decision_shape(
    decision: Mapping[str, Any],
    instruction: str,
    task_family: str,
) -> None:
    _require_exact_keys(
        decision,
        {
            "schema_version",
            "instruction",
            "task_family",
            "strategy_occurrences",
            "occurrence_order",
            "decision_summary",
        },
        "decision",
    )
    if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError("unsupported decision schema_version")
    if decision.get("instruction") != instruction:
        raise ValueError("model must echo the instruction exactly")
    if decision.get("task_family") != task_family:
        raise ValueError("model returned the wrong task_family")
    occurrences = decision.get("strategy_occurrences")
    order = decision.get("occurrence_order")
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError("strategy_occurrences must be a non-empty list")
    if not isinstance(order, list) or not order:
        raise ValueError("occurrence_order must be a non-empty list")
    _require_string(decision.get("decision_summary"), "decision_summary")
    for index, occurrence in enumerate(occurrences):
        item = _require_mapping(occurrence, "strategy occurrence {}".format(index))
        _require_exact_keys(
            item,
            {"id", "strategy_id", "bindings"},
            "strategy occurrence {}".format(index),
        )
        _require_string(item.get("id"), "occurrence id")
        _require_string(item.get("strategy_id"), "strategy_id")
        _require_mapping(item.get("bindings"), "bindings")
    for item in order:
        _require_string(item, "occurrence_order item")


def validate_judge(judge: Mapping[str, Any]) -> None:
    _require_exact_keys(
        judge,
        {
            "schema_version",
            "verdict",
            "score",
            "instruction_assessment",
            "sequence_assessment",
            "grounding_assessment",
            "graph_assessment",
            "discrepancies",
            "limitations",
        },
        "judge",
    )
    if judge.get("schema_version") != JUDGE_SCHEMA_VERSION:
        raise ValueError("unsupported judge schema_version")
    if judge.get("verdict") not in ("match", "partial_match", "mismatch"):
        raise ValueError("judge verdict is invalid")
    score = judge.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("judge score must be numeric")
    if not 0 <= float(score) <= 1:
        raise ValueError("judge score must be between 0 and 1")
    for key in (
        "instruction_assessment",
        "sequence_assessment",
        "grounding_assessment",
        "graph_assessment",
    ):
        _require_string(judge.get(key), key)
    for key in ("discrepancies", "limitations"):
        if not isinstance(judge.get(key), list):
            raise ValueError("{} must be a list".format(key))
        if not all(isinstance(item, str) for item in judge[key]):
            raise ValueError("{} items must be strings".format(key))


def _comparison_target(
    contract_type: str,
    arguments: Mapping[str, Any],
) -> Optional[str]:
    key = {
        "navigate": "goal",
        "pick": "object",
        "place": "object",
        "open": "articulation",
        "close": "articulation",
    }[contract_type]
    value = arguments.get(key)
    return _object_category(str(value)) if value is not None else None


def official_task_plan_path(
    task_family: str,
    split: str,
    asset_root: Path = DEFAULT_ASSET_ROOT,
) -> Path:
    return (
        Path(asset_root)
        / "data"
        / "scene_datasets"
        / "replica_cad_dataset"
        / "rearrange"
        / "task_plans"
        / task_family
        / "sequential"
        / split
        / "all.json"
    )


def _selected_plan(
    source: Mapping[str, Any],
    plan_index: int,
) -> Mapping[str, Any]:
    plans = source.get("plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError("official PlanData has no plans")
    if plan_index < 0 or plan_index >= len(plans):
        raise IndexError("plan_index is out of range")
    return _require_mapping(plans[plan_index], "official task plan")


def _validate_gt_task_family(
    source: Mapping[str, Any],
    task_family: str,
) -> None:
    dataset = re.sub(r"[^a-z0-9]", "", str(source.get("dataset", "")).lower())
    aliases = {
        "set_table": "settable",
        "prepare_groceries": "preparegroceries",
        "tidy_house": "tidyhouse",
    }
    identified = [name for name, token in aliases.items() if token in dataset]
    if identified and task_family not in identified:
        raise ValueError(
            "GT dataset {!r} does not match task_family {}".format(
                source.get("dataset"),
                task_family,
            )
        )


def _gt_destination(
    subtask: Mapping[str, Any],
    task_family: str,
) -> Optional[str]:
    articulation = subtask.get("articulation_config")
    if isinstance(articulation, Mapping):
        articulation_type = articulation.get("articulation_type")
        if articulation_type:
            return str(articulation_type)
    # These two benchmark task definitions give the otherwise unlabeled goal
    # rectangles a stable semantic destination. TidyHouse mixes several
    # surfaces, so it deliberately remains unknown without extra annotation.
    if task_family == "set_table":
        return "dining_table"
    if task_family == "prepare_groceries":
        return "countertop"
    return None


def _gt_arguments(
    subtasks: Sequence[Mapping[str, Any]],
    index: int,
    task_family: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    subtask = subtasks[index]
    contract_type = subtask["type"]
    if contract_type == "navigate":
        target = None
        if index + 1 < len(subtasks):
            following = subtasks[index + 1]
            following_type = following["type"]
            if following_type in ("open", "close"):
                target = following.get("articulation_type")
            elif following_type == "pick":
                target = _object_category(following.get("obj_id"))
            elif following_type == "place":
                target = _gt_destination(following, task_family)
        return ({"goal": target} if target is not None else {}), target
    if contract_type in ("pick", "place"):
        target = _object_category(subtask.get("obj_id"))
        arguments: Dict[str, Any] = {"object": target}
        if contract_type == "place" and subtask.get("goal_pos") is not None:
            arguments["goal_position"] = subtask["goal_pos"]
        if contract_type == "place":
            destination = _gt_destination(subtask, task_family)
            if destination is not None:
                arguments["destination"] = destination
        return arguments, target
    target = subtask.get("articulation_type")
    return {"articulation": target}, str(target) if target is not None else None


def normalize_ground_truth(
    source: Mapping[str, Any],
    *,
    plan_index: int,
    instruction: str,
    task_family: str,
) -> Dict[str, Any]:
    """Represent official nominal TaskPlan subtasks in the shared flow schema."""

    plan = _selected_plan(source, plan_index)
    subtasks = [
        _require_mapping(item, "official subtask") for item in plan["subtasks"]
    ]
    nodes = []
    for index, subtask in enumerate(subtasks):
        contract_type = _require_string(subtask.get("type"), "subtask type")
        arguments, target = _gt_arguments(subtasks, index, task_family)
        nodes.append(
            {
                "id": "gt_{:03d}".format(index),
                "occurrence_id": None,
                "strategy_id": None,
                "subgoal_id": None,
                "skill_node_id": None,
                "contract_id": None,
                "contract_type": contract_type,
                "arguments": arguments,
                "grounded_target": target,
                "policy_id": None,
                "candidate_policy_ids": [],
                "kind": "nominal",
                "grounding": {
                    key: subtask[key]
                    for key in (
                        "uid",
                        "obj_id",
                        "articulation_type",
                        "articulation_id",
                        "goal_pos",
                    )
                    if key in subtask
                },
            }
        )
    nominal_order = [item["id"] for item in nodes]
    return {
        "schema_version": FLOW_SCHEMA_VERSION,
        "source": "official_mshab_task_plan",
        "instruction": instruction,
        "task_family": task_family,
        "metadata": {
            "dataset": source.get("dataset"),
            "plan_index": plan_index,
            "build_config_name": plan.get("build_config_name"),
            "init_config_name": plan.get("init_config_name"),
            "fallback_annotation_available": False,
        },
        "nodes": nodes,
        "edges": [
            {"source": source_id, "target": target_id, "relation": "next"}
            for source_id, target_id in zip(nominal_order, nominal_order[1:])
        ],
        "nominal_order": nominal_order,
        "fallback_branches": [],
        "alternative_choices": [],
        "selected_strategies": [],
    }


def _ordered_nodes(flow: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    by_id = {item["id"]: item for item in flow["nodes"]}
    return [by_id[item] for item in flow["nominal_order"]]


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _grounding_fields(contract_type: str) -> Tuple[str, ...]:
    return {
        "navigate": ("goal",),
        "pick": ("object",),
        "place": ("object", "destination"),
        "open": ("articulation",),
        "close": ("articulation",),
    }.get(contract_type, ())


def compare_flows(
    prediction: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
) -> Dict[str, Any]:
    predicted_nodes = _ordered_nodes(prediction)
    gt_nodes = _ordered_nodes(ground_truth)
    predicted_types = [item["contract_type"] for item in predicted_nodes]
    gt_types = [item["contract_type"] for item in gt_nodes]
    edit_distance = _edit_distance(predicted_types, gt_types)
    lcs_length = _lcs_length(predicted_types, gt_types)
    grounding_pairs: List[Tuple[Any, Any]] = []
    grounding_mismatches: List[Dict[str, Any]] = []
    expected_grounding_fields = 0
    prediction_grounding_capacity = sum(
        len(_grounding_fields(item["contract_type"])) for item in predicted_nodes
    )
    prediction_grounding_fields = sum(
        item["arguments"].get(field) is not None
        for item in predicted_nodes
        for field in _grounding_fields(item["contract_type"])
    )
    gt_grounding_capacity = sum(
        len(_grounding_fields(item["contract_type"])) for item in gt_nodes
    )
    gt_grounding_fields = sum(
        item["arguments"].get(field) is not None
        for item in gt_nodes
        for field in _grounding_fields(item["contract_type"])
    )
    for step, (predicted, target) in enumerate(zip(predicted_nodes, gt_nodes)):
        target_fields = _grounding_fields(target["contract_type"])
        if predicted["contract_type"] != target["contract_type"]:
            continue
        for field in target_fields:
            expected = target["arguments"].get(field)
            if expected is None:
                continue
            expected_grounding_fields += 1
            actual = predicted["arguments"].get(field)
            if actual is None:
                grounding_mismatches.append(
                    {
                        "step": step,
                        "field": field,
                        "predicted": None,
                        "ground_truth": expected,
                    }
                )
                continue
            grounding_pairs.append((actual, expected))
            if actual != expected:
                grounding_mismatches.append(
                    {
                        "step": step,
                        "field": field,
                        "predicted": actual,
                        "ground_truth": expected,
                    }
                )
    position_matches = sum(
        left == right for left, right in zip(predicted_types, gt_types)
    )
    grounding_matches = sum(left == right for left, right in grounding_pairs)
    gt_grounding_fully_observable = (
        gt_grounding_fields == gt_grounding_capacity
    )
    return {
        "scope": "compiled_prediction_vs_supplied_gt_plan",
        "instruction_gt_binding_checked": False,
        "predicted_length": len(predicted_types),
        "gt_length": len(gt_types),
        "predicted_contract_types": predicted_types,
        "gt_contract_types": gt_types,
        "contract_type_exact_match": predicted_types == gt_types,
        "position_accuracy": position_matches
        / max(len(predicted_types), len(gt_types), 1),
        "edit_distance": edit_distance,
        "normalized_edit_similarity": 1
        - edit_distance / max(len(predicted_types), len(gt_types), 1),
        "lcs_length": lcs_length,
        "lcs_recall": lcs_length / max(len(gt_types), 1),
        "grounded_targets_compared": len(grounding_pairs),
        "grounded_target_pair_coverage": len(grounding_pairs)
        / max(expected_grounding_fields, 1),
        "prediction_target_coverage": prediction_grounding_fields
        / max(prediction_grounding_capacity, 1),
        "gt_target_coverage": gt_grounding_fields
        / max(gt_grounding_capacity, 1),
        "gt_grounding_fully_observable": gt_grounding_fully_observable,
        "grounded_target_accuracy": (
            grounding_matches / len(grounding_pairs)
            if grounding_pairs
            else None
        ),
        "semantic_grounding_exact_match": (
            predicted_types == gt_types
            and gt_grounding_fully_observable
            and expected_grounding_fields == len(grounding_pairs)
            and not grounding_mismatches
        ),
        "grounding_mismatches": grounding_mismatches,
        "grounding_fields_expected": expected_grounding_fields,
        "fallback_gt_available": False,
        "prediction_fallback_branches": len(prediction["fallback_branches"]),
    }


def decision_flow_dot(flow: Mapping[str, Any]) -> str:
    """Create a deterministic Graphviz flowchart from the compiled JSON."""

    nodes = list(flow["nodes"])
    aliases = {node["id"]: "n{}".format(index) for index, node in enumerate(nodes)}

    def quote(value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "digraph execution_flow {",
        "  rankdir=LR;",
        '  graph [bgcolor="white", pad="0.3", nodesep="0.35", ranksep="0.55"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]
    for node in nodes:
        fill = "#eef8f1" if node["kind"] == "nominal" else "#fff1f1"
        label = "{}\\n{}\\n{}\\n{}".format(
            node["id"],
            node.get("skill_node_id") or node["contract_type"],
            node["contract_type"],
            json.dumps(node["arguments"], sort_keys=True, ensure_ascii=False),
        )
        lines.append(
            '  {} [label="{}", fillcolor="{}"] ;'.format(
                aliases[node["id"]],
                quote(label),
                fill,
            )
        )
    styles = {
        "next": ("#2b965d", "solid"),
        "on_failure": ("#c43d3d", "dashed"),
        "rejoins": ("#7354c7", "dashed"),
    }
    for edge in flow["edges"]:
        relation = edge["relation"]
        if relation not in EDGE_RELATIONS:
            raise ValueError("unknown compiled edge relation {}".format(relation))
        color, style = styles[relation]
        lines.append(
            '  {} -> {} [label="{}", color="{}", style="{}"] ;'.format(
                aliases[edge["source"]],
                aliases[edge["target"]],
                relation,
                color,
                style,
            )
        )
    alternative_index = 0
    for choice in flow.get("alternative_choices", []):
        source_id = choice["selected_achiever_node_id"]
        if source_id not in aliases:
            raise ValueError(
                "alternative choice references unknown selected achiever {}".format(
                    source_id
                )
            )
        for candidate in choice["candidates"]:
            alias = "a{}".format(alternative_index)
            alternative_index += 1
            label = "alternative strategy\\n{}\\n{}".format(
                candidate["strategy_id"],
                candidate["applicable_when"],
            )
            lines.append(
                '  {} [label="{}", shape=note, fillcolor="#f3efff", '
                'color="#7354c7", fontcolor="#4f3991"] ;'.format(
                    alias,
                    quote(label),
                )
            )
            lines.append(
                '  {} -> {} [label="alternative_to", color="#7354c7", '
                'style="dotted", constraint=false] ;'.format(
                    aliases[source_id],
                    alias,
                )
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_flowchart(flow: Mapping[str, Any], output_dir: Path) -> Tuple[Path, Optional[Path]]:
    dot_path = output_dir / "predicted_flow.dot"
    svg_path = output_dir / "predicted_flow.svg"
    dot_path.write_text(decision_flow_dot(flow), encoding="utf-8")
    executable = shutil.which("dot")
    if executable is None:
        return dot_path, None
    subprocess.run(
        [executable, "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return dot_path, svg_path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _instruction_from_args(args: argparse.Namespace) -> str:
    if args.instruction is not None:
        return args.instruction.strip()
    return args.instruction_file.read_text(encoding="utf-8").strip()


def _load_scene_context(
    path: Optional[Path],
    task_family: str,
) -> Dict[str, Any]:
    if path is None:
        return {
            "task_family": task_family,
            "objects": [],
            "articulations": [],
            "source": "none",
            "note": (
                "No scene context supplied. The planner must rely only on "
                "the instruction and graph."
            ),
        }
    context = _read_json(path)
    context_task = context.get("task_family")
    if context_task is not None and context_task != task_family:
        raise ValueError("scene context has the wrong task_family")
    context.setdefault("task_family", task_family)
    context.setdefault("objects", [])
    context.setdefault("articulations", [])
    context["source"] = "user_supplied"
    _scene_object_names(context)
    if not isinstance(context["articulations"], list):
        raise TypeError("scene_context.articulations must be a list")
    return context


def _prepare_output_dir(
    requested: Optional[Path],
    instruction: str,
    task_family: str,
) -> Path:
    if requested is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8]
        base = DEFAULT_OUTPUT_ROOT / "{}-{}-{}".format(
            timestamp,
            task_family,
            digest,
        )
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = Path("{}-{}".format(base, suffix))
            suffix += 1
    else:
        candidate = requested
        if candidate.exists() and any(candidate.iterdir()):
            raise FileExistsError(
                "output directory is not empty: {}; choose a fresh path".format(
                    candidate
                )
            )
    candidate.mkdir(parents=True, exist_ok=False if not candidate.exists() else True)
    return candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    instruction = parser.add_mutually_exclusive_group(required=True)
    instruction.add_argument("--instruction")
    instruction.add_argument("--instruction-file", type=Path)
    parser.add_argument("--task-family", choices=TASK_FAMILIES, default="set_table")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--plan-index", type=int, default=0)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument(
        "--scene-context",
        type=Path,
        help="Optional user-supplied scene JSON; never derived from GT.",
    )
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--no-gt", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--response-file", type=Path)
    parser.add_argument("--judge-response-file", type=Path)
    parser.add_argument("--skip-judge", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.no_gt and args.gt is not None:
        raise ValueError("--gt and --no-gt cannot be used together")
    if args.dry_run and (
        args.response_file is not None or args.judge_response_file is not None
    ):
        raise ValueError("response files cannot be used with --dry-run")
    if args.judge_response_file is not None and (
        args.skip_judge or args.no_gt
    ):
        raise ValueError(
            "--judge-response-file requires GT and an enabled judge"
        )
    instruction = _instruction_from_args(args)
    if not instruction:
        raise ValueError("instruction must be non-empty")
    graph = _read_json(args.graph)
    index = FourLayerGraphIndex(graph)
    config = DeepSeekConfig.from_yaml(args.config)
    scene_context = _load_scene_context(args.scene_context, args.task_family)

    gt_path: Optional[Path] = None
    if not args.no_gt:
        gt_path = args.gt or official_task_plan_path(args.task_family, args.split)
        if not gt_path.exists():
            raise FileNotFoundError(
                "official GT not found at {}; pass --gt or --no-gt".format(gt_path)
            )

    output_dir = _prepare_output_dir(
        args.output_dir,
        instruction,
        args.task_family,
    )

    planner_messages = build_planner_messages(
        instruction=instruction,
        task_family=args.task_family,
        scene_context=scene_context,
        graph=graph,
    )
    planner_request = config.request_payload(planner_messages)
    _write_json(output_dir / "scene_context.json", scene_context)
    _write_json(output_dir / "planner_request.json", planner_request)
    manifest: Dict[str, Any] = {
        "instruction": instruction,
        "task_family": args.task_family,
        "split": args.split,
        "graph": str(args.graph),
        "graph_sha256": _file_sha256(args.graph),
        "scene_context": str(args.scene_context) if args.scene_context else None,
        "scene_context_sha256": (
            _file_sha256(args.scene_context) if args.scene_context else None
        ),
        "model": config.model,
        "gt_plan_index": None if args.no_gt else args.plan_index,
        "planner_received_ground_truth_data": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    if args.dry_run:
        print("dry run wrote request preview to {}".format(output_dir))
        return

    if args.response_file is not None:
        decision = load_response_file(args.response_file)
        planner_metadata = {
            "source": "response_file",
            "sha256": _file_sha256(args.response_file),
        }
    else:
        decision, planner_metadata = DeepSeekClient(config).complete(planner_messages)
    # Save the model decision before strict graph compilation so rejected
    # outputs remain available for auditing graph-validity failures.
    _write_json(output_dir / "strategy_decision.json", decision)
    flow = index.compile_decision(
        decision,
        instruction,
        args.task_family,
        scene_context,
    )
    _write_json(output_dir / "predicted_flow.json", flow)
    dot_path, svg_path = render_flowchart(flow, output_dir)

    metrics: Optional[Dict[str, Any]] = None
    judge: Optional[Dict[str, Any]] = None
    judge_metadata: Optional[Dict[str, Any]] = None
    gt_flow: Optional[Dict[str, Any]] = None
    simulator_bundle: Optional[Dict[str, Any]] = None
    if gt_path is not None:
        # This is intentionally after the planning decision and compilation.
        gt_source = _read_json(gt_path)
        _validate_gt_task_family(gt_source, args.task_family)
        gt_flow = normalize_ground_truth(
            gt_source,
            plan_index=args.plan_index,
            instruction=instruction,
            task_family=args.task_family,
        )
        manifest["gt"] = str(gt_path)
        manifest["gt_sha256"] = _file_sha256(gt_path)
        manifest["gt_pairing_assumption"] = (
            "The caller is responsible for pairing the instruction with the "
            "selected official plan index."
        )
        _write_json(output_dir / "ground_truth_flow.json", gt_flow)
        try:
            from mshab.experiments.granularity.vlm.simulator import (
                write_simulator_bundle,
            )

            simulator_bundle = {
                "status": "ready",
                **write_simulator_bundle(
                    flow,
                    gt_source,
                    task_family=args.task_family,
                    plan_index=args.plan_index,
                    output_dir=output_dir / "simulator",
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            simulator_bundle = {
                "status": "not_groundable",
                "error": str(exc),
                "note": (
                    "Planning and GT comparison remain valid, but this flow "
                    "cannot be lowered onto the selected scene plan."
                ),
            }
        _write_json(output_dir / "simulator_compilation.json", simulator_bundle)

    if gt_flow is not None:
        metrics = compare_flows(flow, gt_flow)
        _write_json(output_dir / "deterministic_metrics.json", metrics)
        if not args.skip_judge:
            judge_messages = build_judge_messages(
                prediction=flow,
                ground_truth=gt_flow,
                deterministic_metrics=metrics,
            )
            _write_json(
                output_dir / "judge_request.json",
                config.request_payload(judge_messages),
            )
            if args.judge_response_file is not None:
                judge = load_response_file(args.judge_response_file)
                judge_metadata = {
                    "source": "response_file",
                    "sha256": _file_sha256(args.judge_response_file),
                }
            else:
                judge, judge_metadata = DeepSeekClient(config).complete(
                    judge_messages
                )
            validate_judge(judge)
            _write_json(output_dir / "vlm_judge.json", judge)

    _write_json(output_dir / "run_manifest.json", manifest)

    result = {
        **manifest,
        "planner_api": planner_metadata,
        "deterministic_metrics": metrics,
        "vlm_judge": judge,
        "judge_api": judge_metadata,
        "simulator_compilation": simulator_bundle,
        "artifacts": {
            "decision": "strategy_decision.json",
            "flow_json": "predicted_flow.json",
            "flow_dot": dot_path.name,
            "flow_svg": svg_path.name if svg_path is not None else None,
            "ground_truth": "ground_truth_flow.json" if gt_flow is not None else None,
            "simulator": "simulator/manifest.json"
            if simulator_bundle is not None
            and simulator_bundle.get("status") == "ready"
            else None,
        },
    }
    _write_json(output_dir / "result.json", result)
    print("wrote graph-constrained execution flow to {}".format(output_dir))
    if metrics is not None:
        print(
            "contract_type_exact_match={} edit_distance={} lcs_recall={:.3f}".format(
                metrics["contract_type_exact_match"],
                metrics["edit_distance"],
                metrics["lcs_recall"],
            )
        )


if __name__ == "__main__":
    main()
