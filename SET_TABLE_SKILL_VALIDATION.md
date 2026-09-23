# SetTable single-policy validation

Environment fixed for the first milestone:

- task: `set_table`
- robot: Fetch (MS-HAB default)
- policy: official per-object RL checkpoint
- simulator: ManiSkill 3.0.0b18 / SAPIEN 3.0.0b1
- runtime: the `mshab:latest` Docker image (Python 3.10, PyTorch 2.5.1+cu121)

Everything runs in the container. See the
[skill-library guide](./mshab/skills/README.md#installation) for building the
image, downloading assets, and mounting checkpoints. `docker compose run`
starts in `/work/mshab` with `MS_ASSET_DIR`, `MSHAB_EXPS_DIR` and the
checkpoint mount already set, so no activation step or `export` is needed.

Run one policy (contract type + target) at a time:

```bash
docker compose run --rm mshab ./scripts/evaluate_set_table_skill.sh navigate
docker compose run --rm mshab ./scripts/evaluate_set_table_skill.sh open fridge
docker compose run --rm mshab ./scripts/evaluate_set_table_skill.sh pick 013_apple
docker compose run --rm mshab ./scripts/evaluate_set_table_skill.sh place 013_apple
docker compose run --rm mshab ./scripts/evaluate_set_table_skill.sh close fridge
```

The runner performs a CUDA preflight check because the official evaluation
configuration uses ManiSkill's GPU simulation backend. If it reports that CUDA
is not visible, the container did not get the GPU: check `nvidia-smi` on the
host and that the NVIDIA Container Toolkit is installed.

Script options are environment variables, passed to the container with `-e`.
Use `MAX_TRAJECTORIES` and `NUM_ENVS` to scale beyond the default smoke test:

```bash
docker compose run --rm -e MAX_TRAJECTORIES=100 -e NUM_ENVS=21 \
  mshab ./scripts/evaluate_set_table_skill.sh pick 013_apple
```

Record a rollout video (keep `NUM_ENVS=1` for a readable smoke test):

```bash
docker compose run --rm -e RECORD_VIDEO=True -e MAX_TRAJECTORIES=1 -e NUM_ENVS=1 \
  mshab ./scripts/evaluate_set_table_skill.sh pick 013_apple
```

Set `INFO_ON_VIDEO=True` only when the green per-step diagnostic overlay is
needed:

```bash
docker compose run --rm \
  -e RECORD_VIDEO=True -e INFO_ON_VIDEO=True -e MAX_TRAJECTORIES=1 -e NUM_ENVS=1 \
  mshab ./scripts/evaluate_set_table_skill.sh pick 013_apple
```

Videos and evaluation logs are written to `MSHAB_EXPS_DIR`, which
`docker-compose.yml` points inside the bind-mounted repository. On the host
they appear under:

```text
./mshab_exps/set_table_single_skill/<skill>/<object>/
```

That directory is gitignored. The container runs as root, so its contents are
root-owned on the host; `sudo chown -R "$(id -u):$(id -g)" mshab_exps` if that
gets in the way.

Record one row per run:

| contract type | object | episodes | success rate | mean steps | failure states | observation shape | action shape | checkpoint/config | termination condition |
|---|---|---:|---:|---:|---|---|---|---|---|
| navigate | all | | | | | | | `rl/set_table/navigate/all` | `success` |
| open | fridge | | | | | | | `rl/set_table/open/fridge` | `articulation_open` |
| pick | 013_apple | | | | | | | `rl/set_table/pick/013_apple` | `is_grasped` |
| place | 013_apple | | | | | | | `rl/set_table/place/013_apple` | `obj_at_goal` |
| close | fridge | | | | | | | `rl/set_table/close/fridge` | `articulation_closed` |

Do not mark a policy validated from video alone. A validated row needs a completed
evaluation run and the environment's programmatic success signal.

## First composed chain

The apple starts inside a closed fridge, so the minimal physically valid chain
includes Open and a second navigation:

```text
Navigate(fridge) -> Open(fridge) -> Navigate(apple)
-> Pick(apple) -> Navigate(table) -> Place(apple)
```

Run one continuous rollout with automatic checkpoint switching:

```bash
docker compose run --rm \
  -e RECORD_VIDEO=True -e INFO_ON_VIDEO=True -e MAX_TRAJECTORIES=1 -e NUM_ENVS=1 \
  mshab ./scripts/evaluate_set_table_apple_chain.sh
```

For a clean video, use `-e INFO_ON_VIDEO=False -e INVISIBLE_GOALS=True`.

The general runner defaults to permissive composition mode
(`CONTINUOUS_TASK=True`) so later policy handoffs remain observable after a
subtask timeout. For strict evaluation, set `CONTINUOUS_TASK=False`; completion
in permissive mode is not a valid strict task success if an earlier `fail`
signal occurred.

## General skill-chain runner (legacy scripts)

```bash
docker compose run --rm mshab ./scripts/evaluate_skill_chain.sh TASK CHAIN_NAME SELECTION
```

`SELECTION` addresses subtasks in an official sequential TaskPlan, either as a
half-open range or as increasing comma-separated indices. Examples:

```bash
# Apple segment from SetTable plan 0
docker compose run --rm -e EXPECTED_TYPES=navigate,open,navigate,pick,navigate,place \
  mshab ./scripts/evaluate_skill_chain.sh set_table apple_to_table 8:14

# Select explicit nodes instead of a contiguous range
docker compose run --rm mshab \
  ./scripts/evaluate_skill_chain.sh set_table selected_nodes 8,9,10,11,12,13

# Use another official episode/scene
docker compose run --rm -e PLAN_INDEX=12 \
  mshab ./scripts/evaluate_skill_chain.sh set_table apple_scene_12 8:14
```

The generated plan prints every selected skill node and grounded target before
execution. A syntactically valid slice is not necessarily physically valid:
the user/planner must preserve prerequisites such as Open before picking an
object from a closed fridge.

The chain runner writes its generated TaskPlan into the assets volume under
`$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/<task>/custom/`,
so plans persist between containers alongside the official ones.
