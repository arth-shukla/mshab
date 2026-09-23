#!/usr/bin/env bash
set -euo pipefail

# Execute a SkillCatalog plan in MS-HAB and record one video.
# Usage: ./scripts/evaluate_set_table_graph_plan.sh [nominal|recovery_all_primaries_failed]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSHAB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$MSHAB_ROOT/../.." && pwd)"

EXECUTION_PLAN="${1:-nominal}"
SPLIT="${SPLIT:-train}"
PLAN_INDEX="${PLAN_INDEX:-0}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
INFO_ON_VIDEO="${INFO_ON_VIDEO:-True}"
INVISIBLE_GOALS="${INVISIBLE_GOALS:-False}"
RUN_NAME="${RUN_NAME:-${EXECUTION_PLAN}_plan${PLAN_INDEX}_seed${SEED}}"
DRY_RUN="${DRY_RUN:-False}"

if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, underscores, and hyphens" >&2
    exit 2
fi

# Docker sets MS_ASSET_DIR=/root/.maniskill; the fallback is ManiSkill's own default.
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
# Evaluation outputs (videos, tensorboard). Docker points this inside the
# bind-mounted repo so results land in ./mshab_exps on the host.
MSHAB_EXPS_DIR="${MSHAB_EXPS_DIR:-$WORKSPACE_ROOT/mshab_exps}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mshab-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

CATALOG="$MSHAB_ROOT/mshab/skills/catalogs/set_table.json"
SOURCE_PLAN="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/sequential/$SPLIT/all.json"
CHAIN_PLAN="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/custom/graph_${RUN_NAME}.json"
CKPT_DIR="$MS_ASSET_DIR/data/mshab_checkpoints/rl/set_table"

for required in "$CATALOG" "$SOURCE_PLAN"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required file: $required" >&2
        exit 1
    fi
done
python "$SCRIPT_DIR/build_set_table_graph_plan.py" \
    "$SOURCE_PLAN" \
    "$CHAIN_PLAN" \
    --catalog "$CATALOG" \
    --execution-plan "$EXECUTION_PLAN" \
    --source-plan-index "$PLAN_INDEX"

if [[ "$DRY_RUN" == "True" ]]; then
    echo "dry run complete; grounded graph plan: $CHAIN_PLAN"
    exit 0
fi

if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is not visible to PyTorch; the official MS-HAB GPU backend cannot run." >&2
    exit 1
fi
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "missing SetTable checkpoints: $CKPT_DIR" >&2
    exit 1
fi

POLICY_TYPE="${POLICY_TYPE:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["recommended_policy_type"])' "$CHAIN_PLAN")}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["estimated_max_episode_steps"])' "$CHAIN_PLAN")}"
VIDEO_DIR="$MSHAB_EXPS_DIR/set_table-graph-plan/$RUN_NAME/eval_videos"

cd "$MSHAB_ROOT"
SAPIEN_NO_DISPLAY=1 python -m mshab.evaluate configs/evaluate.yml \
    seed="$SEED" \
    task=set_table \
    policy_type="$POLICY_TYPE" \
    save_trajectory=False \
    max_trajectories="$MAX_TRAJECTORIES" \
    eval_env.env_id=SequentialTask-v0 \
    eval_env.task_plan_fp="$CHAIN_PLAN" \
    eval_env.spawn_data_fp=null \
    eval_env.num_envs="$NUM_ENVS" \
    eval_env.max_episode_steps="$MAX_EPISODE_STEPS" \
    eval_env.continuous_task=False \
    eval_env.frame_stack=3 \
    eval_env.stack=null \
    eval_env.record_video=True \
    eval_env.info_on_video="$INFO_ON_VIDEO" \
    eval_env.save_video_freq=1 \
    eval_env.extra_stat_keys='<list>success, fail, subtask, subtask_type, subtasks_steps_left, robot_force, robot_cumulative_force</list>' \
    eval_env.env_kwargs.invisible_goals_in_human_render="$INVISIBLE_GOALS" \
    eval_env.env_kwargs.task_cfgs.navigate.ignore_arm_checkers=True \
    logger.workspace="$MSHAB_EXPS_DIR" \
    logger.exp_name="set_table-graph-plan/$RUN_NAME" \
    logger.clear_out=False \
    logger.tensorboard=True \
    logger.wandb=False

echo "video directory: $VIDEO_DIR"
