#!/usr/bin/env bash
set -euo pipefail

# Reproducible single-skill smoke test for the first physical-harness milestone.
# Run from any directory after activating the dedicated MS-HAB environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSHAB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$MSHAB_ROOT/../.." && pwd)"

SUBTASK="${1:-}"
OBJ="${2:-}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-10}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
RECORD_VIDEO="${RECORD_VIDEO:-False}"
INFO_ON_VIDEO="${INFO_ON_VIDEO:-False}"

if [[ ! "$SUBTASK" =~ ^(navigate|open|pick|place|close)$ ]]; then
    echo "usage: $0 {navigate|open|pick|place|close} [object]" >&2
    exit 2
fi

case "$SUBTASK" in
    navigate)
        OBJ="${OBJ:-all}"
        MAX_EPISODE_STEPS=1000
        EXTRA_STAT_KEYS='<list>success, subtask_type</list>'
        ;;
    pick)
        OBJ="${OBJ:-013_apple}"
        MAX_EPISODE_STEPS=200
        EXTRA_STAT_KEYS='<list>success, subtask_type, is_grasped, robot_target_pairwise_force, robot_force, robot_cumulative_force</list>'
        ;;
    place)
        OBJ="${OBJ:-013_apple}"
        MAX_EPISODE_STEPS=200
        EXTRA_STAT_KEYS='<list>success, subtask_type, is_grasped, obj_at_goal, robot_force, robot_cumulative_force</list>'
        ;;
    open)
        OBJ="${OBJ:-fridge}"
        MAX_EPISODE_STEPS=200
        EXTRA_STAT_KEYS='<list>success, subtask_type, articulation_open, robot_target_pairwise_force, robot_force, robot_cumulative_force, handle_active_joint_qpos, handle_active_joint_qmax, handle_active_joint_qmin</list>'
        ;;
    close)
        OBJ="${OBJ:-fridge}"
        MAX_EPISODE_STEPS=200
        EXTRA_STAT_KEYS='<list>success, subtask_type, articulation_closed, robot_target_pairwise_force, robot_force, robot_cumulative_force, handle_active_joint_qpos, handle_active_joint_qmax, handle_active_joint_qmin</list>'
        ;;
esac

# Docker sets MS_ASSET_DIR=/root/.maniskill; the fallback is ManiSkill's own default.
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
# Evaluation outputs (videos, tensorboard). Docker points this inside the
# bind-mounted repo so results land in ./mshab_exps on the host.
MSHAB_EXPS_DIR="${MSHAB_EXPS_DIR:-$WORKSPACE_ROOT/mshab_exps}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mshab-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is not visible to PyTorch; MS-HAB's official GPU backend cannot run in this session." >&2
    echo "Check nvidia-smi and the container/job GPU allocation, then rerun this command." >&2
    exit 1
fi

TASK_PLAN="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/$SUBTASK/train/$OBJ.json"
SPAWN_DATA="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/set_table/$SUBTASK/train/spawn_data.pt"

for required in "$TASK_PLAN" "$SPAWN_DATA"; do
    if [[ ! -e "$required" ]]; then
        echo "missing MS-HAB asset: $required" >&2
        exit 1
    fi
done

cd "$MSHAB_ROOT"
SAPIEN_NO_DISPLAY=1 python -m mshab.evaluate configs/evaluate.yml \
    seed="$SEED" \
    task=set_table \
    policy_type=rl_per_obj \
    save_trajectory=False \
    max_trajectories="$MAX_TRAJECTORIES" \
    eval_env.env_id="${SUBTASK^}SubtaskTrain-v0" \
    eval_env.task_plan_fp="$TASK_PLAN" \
    eval_env.spawn_data_fp="$SPAWN_DATA" \
    eval_env.num_envs="$NUM_ENVS" \
    eval_env.max_episode_steps="$MAX_EPISODE_STEPS" \
    eval_env.continuous_task=False \
    eval_env.frame_stack=3 \
    eval_env.stack=null \
    eval_env.record_video="$RECORD_VIDEO" \
    eval_env.info_on_video="$INFO_ON_VIDEO" \
    eval_env.save_video_freq=1 \
    eval_env.extra_stat_keys="$EXTRA_STAT_KEYS" \
    logger.workspace="$MSHAB_EXPS_DIR" \
    logger.exp_name="set_table_single_skill/$SUBTASK/$OBJ" \
    logger.clear_out=False \
    logger.tensorboard=True \
    logger.wandb=False
