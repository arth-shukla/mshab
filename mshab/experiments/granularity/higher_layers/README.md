# Coarse semantic higher layers

The coarse graph is task-agnostic. It does not divide capabilities into
SetTable, PrepareGroceries, or TidyHouse. Layer 1 contains exactly three
already-defined outcome templates and no edges between them.

![Coarse higher layers](artifacts/coarse_higher_layers.svg)

## Layer 1: three independent SubGoals

```text
Retrieve                 Deliver                         Restore
holding({object})        at({object},{destination})      closed({source})
```

There are no Layer-1 dependencies:

```text
dependencies = []
```

The three nodes form a reusable semantic vocabulary. A later task instance or
VLM may select and ground whichever SubGoals it needs; the catalog itself does
not encode a fixed task sequence.

## One-to-one Layer-1/2 ownership

Each SubGoal owns exactly one SkillSubgraph:

```text
Retrieve  1:1  RetrieveSkillSubgraph
Deliver   1:1  DeliverSkillSubgraph
Restore   1:1  RestoreSkillSubgraph
```

There are no cross-subgraph edges in this catalog.

## Alternative semantics

`ALTERNATIVE_TO` means two primary SkillNodes achieve the same SubGoal in
different semantic contexts. It does not mean selecting another checkpoint.

Retrieve alternatives:

- retrieve an exposed object from an open surface;
- retrieve an object from a fridge;
- retrieve an object from a counter drawer (`kitchen_counter`).

Deliver alternatives:

- deliver to a dining table;
- deliver to an open countertop;
- deliver inside a fridge;
- deliver inside a counter drawer.

Restore alternatives:

- restore/close a fridge;
- restore/close a counter drawer.

## Fallback semantics

Every strategy has a directed fallback to a different recovery SkillNode:

```text
primary achiever --FALLBACK_TO--> recovery achiever
```

Examples:

- failed Pick → relocalize the object → another Pick SkillNode;
- failed Place → re-approach the destination → another Place SkillNode;
- failed Close → reposition at the articulation → another Close SkillNode.

Fallback is therefore a semantic recovery transition between skills. No
Policy-to-Policy fallback exists here or in Layer 4.

## SkillNode-to-Contract relationship

Every green atomic box in the SVG is one real `SkillNode`, not a whole route.
Every SkillNode stores one scalar `contract_id`, and the JSON emits exactly one
derived `references` edge for it. Many SkillNodes may therefore converge on
the same Contract, producing an n:1 relationship.

SkillNode IDs are action-first and carry their own context, for example:

```text
navigate_table            -> NavigateTable
place_on_table            -> PlaceOnTable
recover_place_on_table    -> RecoverPlaceOnTable
navigate_fridge_source    -> NavigateFridgeSource
close_fridge              -> CloseFridge
```

All 44 displayed names are unique. Strategy labels such as "Deliver to table"
remain machine-readable alternative metadata, but they are not rendered as
boxes or headings inside a SkillSubgraph. The SVG communicates composition
through SkillNodes and typed edges only.

| SkillSubgraph | Navigate | Open | Pick | Place | Close | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retrieve | 8 | 2 | 6 | 0 | 0 | 16 |
| Deliver | 10 | 2 | 0 | 8 | 0 | 20 |
| Restore | 4 | 0 | 0 | 0 | 4 | 8 |
| **Total** | **22** | **4** | **6** | **8** | **4** | **44** |

The 22 Navigate nodes come from distinct graph roles:

- Retrieve: `2 + 3 + 3 = 8` for surface, fridge, and drawer routes;
- Deliver: `2 + 2 + 3 + 3 = 10` for table, counter, fridge, and drawer routes;
- Restore: `2 + 2 = 4` for fridge and drawer routes.

For example, `Navigate source`, `Navigate object`, `Relocalize object`, and
`Reapproach destination` are different semantic SkillNodes. Each still points
only to:

```text
mshab.granularity.navigate.all
```

The relationship is stored explicitly in the artifact:

```text
layer2_to_layer3.references = [
    SkillNode id --REFERENCES--> exactly one Contract id,
    ...
]
```

The SVG renders those references as thin orange lines that merge into the five
shared Contract boxes. Green solid arrows are sequential `ENABLES` edges,
purple dashed double arrows are `ALTERNATIVE_TO`, and red dashed arrows are
`FALLBACK_TO`. These arrows are generated directly from the OOP graph edges.

Layer 1 and Layer 2 contain no Policy IDs. Policy execution remains in the
separate lower-layer `Policy --EXECUTES--> Contract` connection.

## Semantic metadata

The graph stores a `SemanticStrategy` record alongside each route. It includes:

- human-readable label and description;
- the context in which the alternative applies;
- all participating SkillNode IDs;
- the primary achiever;
- its fallback achiever.

This metadata is included in the canonical JSON so a VLM does not have to
infer the meaning of an alternative only from node names.

## Select an alternative before execution

The complete SkillSubgraph is a **candidate catalog**, not one linear plan.
For example, Retrieve contains the surface, fridge, and drawer alternatives at
the same time. Passing the whole catalog directly to a linear planner would
leave several unrelated primary/fallback chains and therefore be ambiguous.

The intended boundary is:

```text
candidate SkillSubgraph
        ↓ VLM/oracle selects one SemanticStrategy
StrategyExecutionView
        ↓ planner follows ENABLES and FALLBACK_TO
executable skill sequence
```

The experiment-specific projection keeps the production skill OOP unchanged:

```python
from mshab.experiments.granularity.higher_layers import (
    build_coarse_higher_layers,
)
from mshab.skills.plan import SkillPlanner

higher = build_coarse_higher_layers()
view = higher.execution_view("retrieve_from_fridge")
planner = SkillPlanner(view.subgoals, view.skills)

nominal = planner.plan()
recovery = planner.plan(
    failed=(view.strategy.primary_achiever,),
)
```

The projection does not mutate the canonical graph. It removes cross-strategy
`ALTERNATIVE_TO` edges, retains the selected strategy's `ENABLES` and
`FALLBACK_TO` edges, and makes its fallback chain unambiguous. A future VLM
only needs to output the selected `strategy_id`; it does not need to rewrite
the graph.

## Generate artifacts

From the repository root:

```bash
PYTHONPATH=. \
python -m mshab.experiments.granularity.higher_layers.render
```

This generates:

- `artifacts/coarse_graph.json`: canonical VLM-ready graph representation;
- `artifacts/coarse_higher_layers.svg`: human-readable overview.
