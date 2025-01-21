#!/usr/bin/bash

SEED=0

MODEL_DIR="mshab_exps/tidy_house-multi_task/tidy_house-rcad-bc_multi_task/bc_multi_task-local-trajs_per_obj=350"
MODEL_CKPT="$MODEL_DIR/models/latest.pt"

TASK=tidy_house

# shellcheck disable=SC2001
WORKSPACE="mshab_exps"
GROUP=$TASK-rcad-bc_multi_task
EXP_NAME="EVAL--$MODEL_DIR"
# shellcheck disable=SC2001
PROJECT_NAME="MS-HAB-RCAD-bc_multi_task"

WANDB=False
TENSORBOARD=False
if [ -z "$MS_ASSET_DIR" ]; then
    MS_ASSET_DIR="$HOME/.maniskill/data"
fi

args=(
    "model_ckpt=$MODEL_CKPT"
    "logger.wandb_cfg.group=$GROUP"
    "logger.exp_name=$EXP_NAME"
    "seed=$SEED"
    "logger.wandb=$WANDB"
    "logger.tensorboard=$TENSORBOARD"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
    "logger.clear_out=False"
    "logger.best_stats_cfg={eval/success_once: 1, eval/return_per_step: 1}"
)

echo "STARTING"
SAPIEN_NO_DISPLAY=1 python -m mshab.evaluate_multi_task configs/bc_multi_task.yml \
    "${args[@]}"
