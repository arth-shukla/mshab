"""Packaged SetTable reference implementation.

The graph, contracts, and checkpoint manifest are task-specific, while the
OOP primitives they use remain in :mod:`mshab.skills.graph`,
:mod:`mshab.skills.model`, and :mod:`mshab.skills.library`.
"""

from mshab.skills.tasks.set_table.stack import (
    SetTableAppleGraphBuilder,
    SetTableGraphBuilder,
    StarterSkillStack,
    build_set_table_apple_graph,
    build_set_table_graph,
    build_set_table_library,
    build_set_table_starter,
    build_set_table_stack,
)

__all__ = [
    "SetTableAppleGraphBuilder",
    "SetTableGraphBuilder",
    "StarterSkillStack",
    "build_set_table_apple_graph",
    "build_set_table_graph",
    "build_set_table_library",
    "build_set_table_starter",
    "build_set_table_stack",
]

