"""Backward-compatible imports for the packaged SetTable example.

New task-specific code should import from :mod:`mshab.skills.tasks.set_table`.
This module remains so existing user code and older catalog-generation scripts
do not break after the SetTable implementation was packaged.
"""

from mshab.skills.tasks.set_table import (
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
