"""Task-package boundaries must preserve the stable public API."""

from mshab.skills import build_set_table_graph as public_build_set_table_graph
from mshab.skills.starter import (
    build_set_table_graph as compatibility_build_set_table_graph,
)
from mshab.skills.tasks.set_table import (
    build_set_table_graph as packaged_build_set_table_graph,
)


def test_set_table_package_preserves_public_and_compatibility_imports():
    assert public_build_set_table_graph is packaged_build_set_table_graph
    assert compatibility_build_set_table_graph is packaged_build_set_table_graph


def test_packaged_set_table_graph_keeps_reference_shape():
    subgoals, graph = packaged_build_set_table_graph()

    assert len(subgoals.subgoals) == 8
    assert len(graph.subgraphs) == 8
    assert len(graph.nodes) == 20
