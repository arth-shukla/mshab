#!/usr/bin/bash

SEED=0

TRAJS_PER_OBJ=350
MAX_IMAGE_CACHE_SIZE=all   # safe num for about 64 GiB system memory
EPOCHS=100_000

TASK=tidy_house

# shellcheck disable=SC2001
WORKSPACE="mshab_exps"
GROUP="$TASK-rcad-bc_multi_task-embed_subtask_one_hot-batch_size=4096-run_longer"
EXP_NAME="$TASK-multi_task/$GROUP/bc_multi_task-local-trajs_per_obj=$TRAJS_PER_OBJ"
# shellcheck disable=SC2001
PROJECT_NAME="MS-HAB-RCAD-bc_multi_task"

WANDB=True
TENSORBOARD=True
if [ -z "$MS_ASSET_DIR" ]; then
    MS_ASSET_DIR="$HOME/.maniskill/data"
fi

# RESUME_LOGDIR="$WORKSPACE/$EXP_NAME"
# RESUME_CONFIG="$RESUME_LOGDIR/config.yml"

DATA_DIR_FP="$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange-dataset/$TASK"

args=(
    "logger.wandb_cfg.group=$GROUP"
    "logger.exp_name=$EXP_NAME"
    "seed=$SEED"
    "algo.epochs=$EPOCHS"
    "algo.trajs_per_obj=$TRAJS_PER_OBJ"
    "algo.data_dir_fp=$DATA_DIR_FP"
    "algo.max_image_cache_size=$MAX_IMAGE_CACHE_SIZE"
    "algo.batch_size=4096"
    "algo.eval_freq=1"
    "algo.log_freq=1"
    "algo.save_freq=1"
    "logger.wandb=$WANDB"
    "logger.tensorboard=$TENSORBOARD"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
)

echo "STARTING"
SAPIEN_NO_DISPLAY=1 python -m mshab.train_bc_multi_task configs/bc_multi_task.yml \
    logger.clear_out="True" \
    logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
    "${args[@]}"

# if [ -f "$RESUME_CONFIG" ] && [ -f "$RESUME_LOGDIR/models/latest.pt" ]; then
#     echo "RESUMING"
#     SAPIEN_NO_DISPLAY=1 python -m mshab.train_bc_multi_task "$RESUME_CONFIG" RESUME_LOGDIR="$RESUME_LOGDIR" \
#         logger.clear_out="False" \
#         logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
#         "${args[@]}"
# else
#     echo "STARTING"
#     SAPIEN_NO_DISPLAY=1 python -m mshab.train_bc_multi_task configs/bc_multi_task.yml \
#         logger.clear_out="True" \
#         logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
#         "${args[@]}"
# fi
