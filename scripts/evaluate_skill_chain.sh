#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/evaluate_skill_chain.sh TASK CHAIN_NAME SELECTION
# Example:
#   ./scripts/evaluate_skill_chain.sh set_table apple_to_table 8:14
#
# SELECTION is a half-open source-plan range (8:14) or increasing indices
# (8,9,10,11,12,13). All task grounding is copied from the selected official
# sequential TaskPlan.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSHAB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$MSHAB_ROOT/../.." && pwd)"

TASK="${1:-}"
CHAIN_NAME="${2:-}"
SELECTION="${3:-}"

if [[ ! "$TASK" =~ ^(set_table|tidy_house|prepare_groceries)$ ]] || [[ -z "$CHAIN_NAME" ]] || [[ -z "$SELECTION" ]]; then
    echo "usage: $0 {set_table|tidy_house|prepare_groceries} CHAIN_NAME {start:end|i,j,...}" >&2
    exit 2
fi
if [[ ! "$CHAIN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "CHAIN_NAME may contain only letters, numbers, underscores, and hyphens" >&2
    exit 2
fi

SPLIT="${SPLIT:-train}"
PLAN_INDEX="${PLAN_INDEX:-0}"
EXPECTED_TYPES="${EXPECTED_TYPES:-}"
POLICY_TYPE="${POLICY_TYPE:-rl_per_obj}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
RECORD_VIDEO="${RECORD_VIDEO:-True}"
INFO_ON_VIDEO="${INFO_ON_VIDEO:-True}"
INVISIBLE_GOALS="${INVISIBLE_GOALS:-False}"
CONTINUOUS_TASK="${CONTINUOUS_TASK:-True}"

# Docker sets MS_ASSET_DIR=/root/.maniskill; the fallback is ManiSkill's own default.
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
# Evaluation outputs (videos, tensorboard). Docker points this inside the
# bind-mounted repo so results land in ./mshab_exps on the host.
MSHAB_EXPS_DIR="${MSHAB_EXPS_DIR:-$WORKSPACE_ROOT/mshab_exps}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mshab-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is not visible to PyTorch; the official MS-HAB GPU backend cannot run." >&2
    exit 1
fi

SOURCE_PLAN="${SOURCE_PLAN:-$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/sequential/$SPLIT/all.json}"
CHAIN_PLAN="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/custom/${CHAIN_NAME}_${SPLIT}.json"
CKPT_DIR="$MS_ASSET_DIR/data/mshab_checkpoints/${POLICY_TYPE%%_*}/$TASK"

if [[ ! -f "$SOURCE_PLAN" ]]; then
    echo "missing source TaskPlan: $SOURCE_PLAN" >&2
    exit 1
fi
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "missing checkpoints for task/policy: $CKPT_DIR" >&2
    exit 1
fi

BUILD_ARGS=(
    "$SOURCE_PLAN"
    "$CHAIN_PLAN"
    --plan-index "$PLAN_INDEX"
    --select "$SELECTION"
)
if [[ -n "$EXPECTED_TYPES" ]]; then
    BUILD_ARGS+=(--expect-types "$EXPECTED_TYPES")
fi
python "$SCRIPT_DIR/build_skill_chain.py" "${BUILD_ARGS[@]}"

if [[ -z "${MAX_EPISODE_STEPS:-}" ]]; then
    MAX_EPISODE_STEPS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["estimated_max_episode_steps"])' "$CHAIN_PLAN")"
fi

cd "$MSHAB_ROOT"
SAPIEN_NO_DISPLAY=1 python -m mshab.evaluate configs/evaluate.yml \
    seed="$SEED" \
    task="$TASK" \
    policy_type="$POLICY_TYPE" \
    save_trajectory=False \
    max_trajectories="$MAX_TRAJECTORIES" \
    eval_env.env_id=SequentialTask-v0 \
    eval_env.task_plan_fp="$CHAIN_PLAN" \
    eval_env.spawn_data_fp=null \
    eval_env.num_envs="$NUM_ENVS" \
    eval_env.max_episode_steps="$MAX_EPISODE_STEPS" \
    eval_env.continuous_task="$CONTINUOUS_TASK" \
    eval_env.frame_stack=3 \
    eval_env.stack=null \
    eval_env.record_video="$RECORD_VIDEO" \
    eval_env.info_on_video="$INFO_ON_VIDEO" \
    eval_env.save_video_freq=1 \
    eval_env.extra_stat_keys='<list>success, fail, subtask, subtask_type, subtasks_steps_left, robot_force, robot_cumulative_force</list>' \
    eval_env.env_kwargs.invisible_goals_in_human_render="$INVISIBLE_GOALS" \
    eval_env.env_kwargs.task_cfgs.navigate.ignore_arm_checkers=True \
    logger.workspace="$MSHAB_EXPS_DIR" \
    logger.exp_name="$TASK-skill-chains/$CHAIN_NAME" \
    logger.clear_out=False \
    logger.tensorboard=True \
    logger.wandb=False
