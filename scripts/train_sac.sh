#!/usr/bin/bash

SEED=0

TASK=tidy_house
SUBTASK=pick
SPLIT=train
OBJ=all

# shellcheck disable=SC2001
ENV_ID="$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"
WORKSPACE="mshab_exps"
GROUP=$TASK-rcad-sac-$SUBTASK
EXP_NAME="$ENV_ID/$GROUP/sac-$SUBTASK-$OBJ-local"
# shellcheck disable=SC2001
PROJECT_NAME="MS-HAB-RCAD-$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')-$TASK-sac"

WANDB=False
# NOTE (arth): tensorboard=False since there seems to be an issue with tensorboardX crashing on very long runs
if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi

if [ "$SUBTASK" = "navigate" ]; then
    train_max_episode_steps=1000
    eval_max_episode_steps=1000
else
    train_max_episode_steps=100
    eval_max_episode_steps=200
fi

# NOTE: the below args are defaults, however the released checkpoints may use different hyperparameters. To train using the same args, check the config.yml files from the released checkpoints.
SAPIEN_NO_DISPLAY=1 python -m mshab.train_sac configs/sac_pick.yml \
        logger.clear_out="True" \
        logger.wandb_cfg.group="$GROUP" \
        logger.exp_name="$EXP_NAME" \
        seed=$SEED \
        env.env_id="$ENV_ID" \
        env.task_plan_fp="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json" \
        env.spawn_data_fp="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt" \
        \
        env.max_episode_steps=$train_max_episode_steps \
        eval_env.max_episode_steps=$eval_max_episode_steps \
        env.env_kwargs.task_cfgs.${SUBTASK}.horizon=$train_max_episode_steps \
        eval_env.env_kwargs.task_cfgs.${SUBTASK}.horizon=$eval_max_episode_steps \
        \
        algo.gamma=0.95 \
        algo.total_timesteps=1_000_000_000 \
        algo.eval_freq=null \
        algo.log_freq=10_000 \
        algo.save_freq=100_000 \
        algo.batch_size=512 \
        algo.replay_buffer_capacity=1_000_000 \
        \
        eval_env.make_env="True" \
        env.make_env="True" \
        \
        env.num_envs=63 \
        eval_env.num_envs=63 \
        \
        env.record_video="False" \
        env.info_on_video="False" \
        \
        eval_env.record_video="False" \
        eval_env.info_on_video="False" \
        eval_env.save_video_freq=10 \
        \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        logger.wandb="$WANDB" \
        logger.tensorboard="False" \
        logger.project_name="$PROJECT_NAME" \
        logger.workspace="$WORKSPACE" \
