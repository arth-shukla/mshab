# SetTable reference package

This package contains the hand-authored SetTable reference implementation for
the four-layer skill library. It is a completed example, not the shared object
model: reusable graph, contract, library, planner, and runtime classes remain
one level above in `mshab.skills`.

## Contents

- `stack.py`: SetTable Layer-1/2 builders, the canonical contract/checkpoint
  manifest, and helpers that assemble all four layers.
- `mshab/skills/catalogs/set_table.json`: generated portable catalog.
- `docs/static/images/set_table_skill_graph.svg`: generated graph rendering.
- `scripts/generate_set_table_skill_graph.py`: regenerates both artifacts.
- `scripts/build_set_table_graph_plan.py`: grounds a selected graph path with
  one official MS-HAB episode plan.
- `scripts/evaluate_set_table_graph_plan.sh`: runs the grounded plan and records
  a video.

The stable public API remains available from both:

```python
from mshab.skills import build_set_table_stack
from mshab.skills.tasks.set_table import build_set_table_stack
```

`mshab.skills.starter` is retained as a compatibility import only.

