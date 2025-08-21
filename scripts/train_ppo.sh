#!/usr/bin/bash

SEED=0

TASK=tidy_house
SUBTASK=pick
SPLIT=train
OBJ=all

# shellcheck disable=SC2001
ENV_ID="$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"
WORKSPACE="mshab_exps"
GROUP=$TASK-rcad-ppo-$SUBTASK
EXP_NAME="$ENV_ID/$GROUP/ppo-$SUBTASK-$OBJ-local"
# shellcheck disable=SC2001
PROJECT_NAME="MS-HAB-RCAD-$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')-$TASK-ppo"

WANDB=False
TENSORBOARD=True
if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi

resume_logdir="$WORKSPACE/$EXP_NAME"
resume_config="$resume_logdir/config.yml"

if [ "$TASK" = "set_table" ] && [ "$SUBTASK" = "open" ] && [ "$OBJ" = "kitchen_counter" ]; then
    stem="env.env_kwargs.randomly_slightly_open_articulation"
    extra_args=(
        "$stem=True"
        "eval_$stem=False"
    )
else
    extra_args=()
fi

if [ "$SUBTASK" = "pick" ]; then
    extra_stat_keys='["is_grasped", "ee_rest", "robot_rest", "is_static", "cumulative_force_within_limit"]'
elif [ "$SUBTASK" = "navigate" ]; then
    extra_stat_keys='["is_grasped", "oriented_correctly", "distance_from_goal", "navigated_close", "ee_rest", "robot_rest", "is_static"]'
elif [ "$SUBTASK" = "open" ]; then
    extra_stat_keys='["is_grasped", "articulation_open", "ee_rest", "robot_rest", "is_static", "cumulative_force_within_limit"]'
elif [ "$SUBTASK" = "close" ]; then
    extra_stat_keys='["is_grasped", "articulation_closed", "ee_rest", "robot_rest", "is_static", "cumulative_force_within_limit"]'
else
    extra_stat_keys='[]'
fi

if [ "$SUBTASK" = "navigate" ]; then
    train_max_episode_steps=1000
    eval_max_episode_steps=1000
    continuous_task=False
else
    train_max_episode_steps=100
    eval_max_episode_steps=200
    continuous_task=True
fi

# NOTE: the below args are defaults, however the released checkpoints may use different hyperparameters. To train using the same args, check the config.yml files from the released checkpoints.
args=(
    "logger.wandb_cfg.group=$GROUP"
    "logger.exp_name=$EXP_NAME"
    "seed=$SEED"
    "env.env_id=$ENV_ID"
    "env.task_plan_fp=$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json"
    "env.spawn_data_fp=$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"
    "algo.gamma=0.95"
    "algo.gae=0.9"
    "algo.update_epochs=8"
    "algo.num_minibatches=16"
    "algo.total_timesteps=100_000_000"
    "algo.eval_freq=null"
    "algo.log_freq=500_000"
    "algo.save_freq=500_000"
    "algo.save_backup_ckpts=True"
    "eval_env.make_env=True"
    "env.make_env=True"
    "env.num_envs=189"
    "eval_env.num_envs=63"
    "env.max_episode_steps=$train_max_episode_steps"
    "eval_env.max_episode_steps=$eval_max_episode_steps"
    "env.env_kwargs.task_cfgs.${SUBTASK}.horizon=$train_max_episode_steps"
    "eval_env.env_kwargs.task_cfgs.${SUBTASK}.horizon=$eval_max_episode_steps"
    "algo.num_steps=100"
    "env.continuous_task=$continuous_task"
    "env.record_video=False"
    "env.info_on_video=False"
    "eval_env.record_video=False"
    "eval_env.info_on_video=True"
    "eval_env.save_video_freq=10"
    "logger.wandb=$WANDB"
    "logger.tensorboard=$TENSORBOARD"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
    "${extra_args[@]}"
)

echo "RUNNING"

if [ -f "$resume_config" ] && [ -f "$resume_logdir/models/latest.pt" ]; then
    echo "RESUMING"
    SAPIEN_NO_DISPLAY=1 python -m mshab.train_ppo "$resume_config" resume_logdir="$resume_logdir" \
        logger.clear_out="False" \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        env.extra_stat_keys="${extra_stat_keys}" eval_env.extra_stat_keys="${extra_stat_keys}" \
        "${args[@]}"
        
else
    echo "STARTING"
    SAPIEN_NO_DISPLAY=1 python -m mshab.train_ppo configs/ppo_pick.yml \
        logger.clear_out="True" \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        env.extra_stat_keys="${extra_stat_keys}" eval_env.extra_stat_keys="${extra_stat_keys}" \
        "${args[@]}"
        
fi
