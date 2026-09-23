# DeepSeek graph-planning experiment

This package asks DeepSeek to select semantic strategies from the existing
four-layer coarse graph, compiles those choices into a branching execution
flow, compares the nominal flow with one official MS-HAB task plan, and lowers
compatible routes into native MS-HAB `PlanData` files.

The first model call receives:

- the natural-language instruction;
- the four-layer graph JSON;
- optional scene facts supplied independently with `--scene-context`.

It receives **no data derived from the GT file**. The runner does not load GT
content until the planning response has been validated and compiled. A second,
independent judge call receives the compiled prediction and GT afterward.

## API configuration

`config.yaml` is a local, Git-ignored secret file. Either edit its `api_key`
field or, preferably, leave it empty and export the configured environment
variable:

```bash
cd ~/zihan_workspace/physical_harnessing/sims/mshab

cp -n \
  mshab/experiments/granularity/vlm/config.example.yaml \
  mshab/experiments/granularity/vlm/config.yaml
chmod 600 mshab/experiments/granularity/vlm/config.yaml

export DEEPSEEK_API_KEY='your-token-here'
```

Never add `config.yaml` to Git. The runner never writes the token into a
request preview, result, log, or flow artifact, rejects HTTP redirects, and
rejects an inline token if the configuration is readable by group/others.

The template explicitly enables DeepSeek thinking mode with `high` reasoning
effort. Set `thinking: disabled` if you want a non-reasoning ablation; only in
that mode does the configured `temperature` apply.

## Inspect the request without calling the API

```bash
MS_ASSET_DIR=../mshab-assets \
PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.run \
  --instruction \
  'Set the table by moving the bowl from the kitchen counter drawer and the apple from the fridge to the dining table, then close both containers.' \
  --task-family set_table \
  --split val \
  --plan-index 0 \
  --dry-run
```

The dry run builds `planner_request.json`, `scene_context.json`, and
`run_manifest.json`. It parses the local configuration file, but does not
resolve/send the API key, load GT content, or use the network.

If the instruction alone is insufficient, provide scene facts independently
of the GT plan:

```json
{
  "task_family": "set_table",
  "objects": ["024_bowl", "013_apple"],
  "articulations": ["kitchen_counter", "fridge"]
}
```

Pass that file with `--scene-context my_scene.json`. Do not construct it from
the official ordered task-plan file; that would contaminate the experiment.

## Run planning and GT comparison

Remove `--dry-run`:

```bash
MS_ASSET_DIR=../mshab-assets \
PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.run \
  --instruction \
  'Set the table by moving the bowl from the kitchen counter drawer and the apple from the fridge to the dining table, then close both containers.' \
  --task-family set_table \
  --split val \
  --plan-index 0
```

This normally performs two calls:

1. Planner: instruction + graph + optional independent scene context →
   strategy choices.
2. Judge: compiled prediction + normalized GT + deterministic metrics → a
   semantic assessment, including whether the instruction was paired with an
   appropriate GT plan.

Use `--skip-judge` to make only the planner call. Use `--response-file` and
`--judge-response-file` to validate saved responses without network access.

By default, each run gets a new timestamped directory under `vlm/outputs/`;
the runner refuses to reuse a non-empty explicit `--output-dir`. All generated
run directories are ignored by Git:

- `strategy_decision.json`: DeepSeek's constrained strategy selection;
- `predicted_flow.json`: authoritative locally compiled four-layer flow;
- `predicted_flow.dot` and `predicted_flow.svg`: execution flowchart;
- `ground_truth_flow.json`: one official task plan in the shared flow schema;
- `deterministic_metrics.json`: exact sequence, edit distance, LCS, and
  object/articulation/destination grounding metrics where GT semantics are
  observable;
- `vlm_judge.json`: optional second-call semantic comparison;
- `simulator/nominal.json`: simulator-ready nominal `PlanData`;
- `simulator/fallback.<occurrence>.json`: one simulator-ready fallback route
  for each selected semantic strategy;
- `simulator/manifest.json`: available route IDs and their source failure;
- `result.json`: run summary without credentials.

`predicted_flow.json` contains three distinct kinds of planning evidence:

- `nominal_order`: the route selected for normal execution;
- `fallback_branches`: `on_failure` recovery, its full replacement route, and
  its rejoin point, all derived from Layer-2 `fallback_to` edges;
- `alternative_choices`: the other semantic strategies available for the same
  SubGoal, with their `applicable_when` condition and authoritative
  `alternative_to` provenance.

The model selects the semantic alternative. The local compiler—not the
model—derives SkillNodes, Contracts, Policies, fallback nodes, and all graph
relations.

## Execute the generated plan in MS-HAB

The planning run automatically creates the simulator bundle when its selected
flow can be grounded in the chosen official scene plan. List its routes:

```bash
RUN_DIR=mshab/experiments/granularity/vlm/outputs/<your-run-directory>

MS_ASSET_DIR=../mshab-assets \
PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.execute \
  --run-dir "$RUN_DIR" \
  --list-routes
```

Prepare the nominal command without starting the simulator:

```bash
MS_ASSET_DIR=../mshab-assets \
PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.execute \
  --run-dir "$RUN_DIR" \
  --route nominal
```

Add `--run` to execute the route with the existing MS-HAB evaluator and
downloaded policies:

```bash
MS_ASSET_DIR=../mshab-assets \
PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.execute \
  --run-dir "$RUN_DIR" \
  --route nominal \
  --run
```

A fallback route is invoked the same way, for example:

```bash
MS_ASSET_DIR=../mshab-assets \
PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.execute \
  --run-dir "$RUN_DIR" \
  --route fallback.retrieve_bowl_from_drawer \
  --run
```

The executor does not replay demonstrations and does not implement a new
policy. It calls the existing `mshab.evaluate` loop, which loads the downloaded
checkpoint for each grounded Pick/Place/Navigate/Open/Close subtask and steps
`SequentialTask-v0` until that subtask succeeds or fails.

MS-HAB's native `PlanData` schema is linear and has no conditional-edge field.
Consequently, `predicted_flow.json` remains the branching control graph, while
each nominal/fallback choice is emitted as a separate native linear plan.
These fallback plans support controlled failure-injection comparisons; they do
not claim that stock `SequentialTask-v0` dynamically switches branches during
one episode.

## Main SetTable graph-ablation experiment

This is the main causal experiment. It gives DeepSeek the same ten SetTable
queries, SkillNodes, Contracts, model configuration, and output schema under
two conditions:

- `flat_library`: Layer-2 edges are hidden;
- `full_graph`: the same library includes sequential, alternative, and
  fallback edges.

The model must output every SkillNode in its path. No strategy-level compiler
fills in missing steps. Four queries use nominal execution and six inject one
of the six SetTable primary failures. This requires 20 API calls. Each query
is paired with one official scene, and execution adds one deterministic graph
oracle, producing 30 simulator jobs total.

Run the phases separately:

```bash
cd ~/zihan_workspace/physical_harnessing/sims/mshab
export DEEPSEEK_API_KEY='your-token-here'

# 20 API calls: 10 flat-library + 10 full-graph queries.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase planning

# No API and no simulator: ground the valid outputs into 30 paired jobs.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase prepare

# One-job CUDA smoke test.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase execute --max-jobs 1

# Run all remaining groundable jobs. The command is resumable.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase execute

# Rebuild presentation tables without API or simulator calls.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase summarize
```

Use `--condition flat_library` or `--condition full_graph` to run one planning
condition. Use `--case-id N01` to select one query. Planning is resumable and
treats a received but graph-invalid response as a completed experimental
sample rather than silently retrying it.

The PPT-ready report is written to:

```text
vlm/outputs/set_table_graph_ablation_main/tables/report.md
```

It contains three tables:

1. overall planning: Valid Path, Path LCS-F1, Grounding Accuracy;
2. nominal versus fallback planning breakdown;
3. paired MS-HAB Groundable Rate and Simulator Success.

### Reuse the saved planning samples for train-scene execution

The released SetTable checkpoints declare train-split subtask plans and spawn
data in their bundled configurations. The validation split uses disjoint
ReplicaCAD scene builds, so first establish an in-distribution execution
baseline on `train`; treat validation execution as a later scene-generalization
experiment. The saved DeepSeek decisions are scene-independent (instruction +
semantic graph only), so they can be reused without another API call.

Use a separate output directory so the existing validation-grounded artifacts
remain untouched:

```bash
cd ~/zihan_workspace/physical_harnessing/sims/mshab

VAL_OUTPUT=mshab/experiments/granularity/vlm/outputs/set_table_graph_ablation_main
TRAIN_OUTPUT=mshab/experiments/granularity/vlm/outputs/set_table_graph_ablation_train_execution

# Zero API calls: reuse VAL_OUTPUT/planning and ground it to train plans 0–9.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase prepare \
  --planning-source-dir "$VAL_OUTPUT" \
  --execution-split train \
  --output-dir "$TRAIN_OUTPUT"

# First run the deterministic graph-oracle baseline for one train scene.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase execute \
  --planning-source-dir "$VAL_OUTPUT" \
  --execution-split train \
  --output-dir "$TRAIN_OUTPUT" \
  --condition graph_oracle \
  --case-id N01 \
  --max-jobs 1 \
  --record-video

tail -n 20 "$TRAIN_OUTPUT/execution_logs/graph_oracle.N01.log"
```

If that pilot reaches `success_at_end: 1`, run every remaining groundable job
and rebuild the report:

```bash
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase execute \
  --planning-source-dir "$VAL_OUTPUT" \
  --execution-split train \
  --output-dir "$TRAIN_OUTPUT"

MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase summarize \
  --planning-source-dir "$VAL_OUTPUT" \
  --execution-split train \
  --output-dir "$TRAIN_OUTPUT"
```

The resulting report explicitly labels Tables 1–2 as validation planning,
Table 3 as nominal graph-oracle controller calibration, and Table 4 as paired
train execution. An API-truncated response remains one failed model sample
(zero for the aggregate metrics) and is not silently resampled.

### Corrected paired multi-seed execution protocol

Scene grounding resolves semantic names such as `bowl` and `apple` to the
scene's concrete object categories before policy selection. Therefore the
oracle, flat-library, and full-graph conditions all use the same available
object-specific `rl_per_obj` policies. The validator also enforces the six
instruction-level SetTable stages in this order:

```text
retrieve bowl -> deliver bowl -> restore drawer
-> retrieve apple -> deliver apple -> restore fridge
```

Use a new output directory when changing the seed set. First calibrate the
downloaded controllers on four valid nominal oracle plans with five paired
seeds (20 rollouts, zero API calls):

```bash
cd ~/zihan_workspace/physical_harnessing/sims/mshab

VAL_OUTPUT=mshab/experiments/granularity/vlm/outputs/set_table_graph_ablation_main
PAIRED_OUTPUT=mshab/experiments/granularity/vlm/outputs/set_table_graph_ablation_paired_5seed

MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase prepare \
  --planning-source-dir "$VAL_OUTPUT" \
  --execution-split train \
  --execution-seeds 0,1,2,3,4 \
  --output-dir "$PAIRED_OUTPUT"

MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase execute \
  --planning-source-dir "$VAL_OUTPUT" \
  --execution-split train \
  --execution-seeds 0,1,2,3,4 \
  --output-dir "$PAIRED_OUTPUT" \
  --condition graph_oracle \
  --case-type nominal

cat "$PAIRED_OUTPUT/tables/controller_calibration.md"
```

If the oracle calibration is adequate, run the remaining grounded plans with
the same command minus the two filters. Execution is resumable. For any two
conditions that compile to byte-identical plans, the runner reuses the same
rollout at a given seed instead of allowing GPU nondeterminism to create a
false method difference:

```bash
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_graph_ablation \
  --phase execute \
  --planning-source-dir "$VAL_OUTPUT" \
  --execution-split train \
  --execution-seeds 0,1,2,3,4 \
  --output-dir "$PAIRED_OUTPUT"
```

Table 4 reports both conditional simulator success over grounded rollouts and
end-to-end success, where invalid or ungroundable plans count as zero. The
fallback cases remain precompiled recovery-route tests; they are not claims
of online failure detection and branch switching.

## Pilot SetTable strategy-selection experiment

The earlier tracked pilot definition contains ten order-explicit instruction
paraphrases and official SetTable `val` plan indices `0–9`:

```text
set_table_experiment.yaml
```

The pilot runner performs exactly ten DeepSeek calls, grounds every model
decision into ten scene plans, prepares ten GT baselines, and prepares all six
canonical fallback routes. This produces 170 independent simulator jobs:

- 10 GT nominal cases;
- 100 VLM nominal cases (`10 instructions × 10 scenes`);
- 60 fallback cases (`6 routes × 10 scenes`).

Run each phase separately:

```bash
cd ~/zihan_workspace/physical_harnessing/sims/mshab
export DEEPSEEK_API_KEY='your-token-here'

# Ten API calls. Existing successful instruction runs are skipped.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_experiment \
  --phase planning

# No API calls: compile all 170 scene-grounded PlanData jobs.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_experiment \
  --phase prepare

# Requires CUDA. Completed jobs are skipped, so this command is resumable.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_experiment \
  --phase execute

# Rebuild both tables without API or simulator calls.
MS_ASSET_DIR=../mshab-assets PYTHONPATH=. \
python -m mshab.experiments.granularity.vlm.set_table_experiment \
  --phase summarize
```

For one uninterrupted run, use `--phase all`. Videos are disabled by default
for the 170-job experiment. Add `--record-video` only when needed. During a
smoke check, `--max-jobs 1` executes only the next pending job. Filters such as
`--condition vlm_nominal` and `--instruction-id I01` select a subset without
changing the experiment definition.

Results are written under:

```text
vlm/outputs/set_table_full_experiment/
├── planning/I01 ... I10/
├── execution_plans/
├── execution_status/
├── execution_logs/
└── tables/
    ├── planning_results.csv
    ├── planning_table.csv
    ├── execution_results.csv
    ├── execution_table.csv
    └── report.md
```

This pilot demonstrates the strategy-selection and lowering pipeline, but it
is not the main graph-structure ablation: its deterministic compiler expands
selected strategies into SkillNode paths. Use the main experiment above when
claiming that visible graph relations improve explicit path planning.

## What MS-HAB provides as GT

The downloaded benchmark provides official nominal `PlanData` JSON files:

```text
$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/
  task_plans/<task>/sequential/<split>/all.json
```

They contain ordered primitive subtasks and scene grounding, not graph images,
SkillNode IDs, alternatives, or fallback annotations. The standard nominal
lengths are:

- SetTable: 16 primitive steps;
- PrepareGroceries: 12 primitive steps;
- TidyHouse: 20 primitive steps.

Therefore nominal contract order and grounding can be compared to GT.
Fallback branches are checked for graph validity and lowered to executable
routes, but their experimental metric must be failure-injection simulator
success rather than GT exact match.

`--task-family` is checked against identifiable official dataset names, but a
natural-language instruction cannot be proven to describe a particular
`--plan-index`. The experiment caller is responsible for that pairing; its GT
path, split, index, and SHA-256 are recorded in `run_manifest.json`. The
deterministic file therefore marks `instruction_gt_binding_checked: false`;
the independent DeepSeek judge explicitly assesses that relation.

Likewise, `semantic_validation.valid` means only local consistency (for
example retrieve-before-deliver and explicit scene membership), not complete
natural-language task satisfaction. It is labeled `local_consistency_only` in
the flow. For TidyHouse, the official plan does not label non-articulated
destination classes, so `semantic_grounding_exact_match` remains false unless
all required grounding fields are observable; consult the coverage fields
rather than treating unobservable destinations as correct.

Before compilation, the runner checks the actual Layer-2 `fallback_to` and
`alternative_to` topology, every SkillNode→Contract reference, and every
Policy→Contract `EXECUTES` connection. The execution flow uses `on_failure`
for the transition to the first recovery prerequisite; each branch also
retains the original Layer-2 `fallback_to` achiever relation as provenance.

## Important terminology

This first version sends graph **JSON**, so technically it is structured-text
graph reasoning rather than visual-input reasoning. The folder is named
`vlm` for the planned experiment family. An SVG/PNG vision-input ablation can
be added later without changing the output or comparison schema.
