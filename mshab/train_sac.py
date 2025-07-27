# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from dacite import from_dict
from omegaconf import OmegaConf

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
import torch.nn.functional as F

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import common

from mshab.agents.sac.agent import Agent
from mshab.agents.sac.replay import PixelStateBatchReplayBuffer
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_numpy, to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataclasses import default_field
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


@dataclass
class SACConfig:
    name: str = "sac"

    # replay buffer
    replay_buffer_capacity: int = 1_000_000

    # train
    total_timesteps: int = 50_000_000
    num_steps: int = field(init=False)
    init_steps: int = 5000
    batch_size: int = 512

    # (shared) encoders
    critic_encoder_tau: float = 0.005
    cnn_features: List[int] = default_field([32, 64, 128, 256])
    cnn_filters: List[int] = default_field([3, 3, 3, 3])
    cnn_strides: List[int] = default_field([2, 2, 2, 2])
    cnn_padding: Union[str, int] = "valid"
    encoder_pixels_feature_dim: int = 50
    encoder_state_feature_dim: int = 50
    detach_encoder: bool = False

    # critic
    critic_hidden_dims: List[int] = default_field([256, 256, 256])
    critic_lr: float = 3e-4
    critic_layer_norm: bool = True
    critic_dropout: Union[None, float] = None
    critic_beta: float = 0.9
    critic_tau: float = 0.005
    critic_target_update_freq: int = 2

    # actor
    actor_hidden_dims: List[int] = default_field([256, 256, 256])
    actor_lr: float = 3e-4
    actor_beta: float = 0.9
    actor_log_std_min: float = -20
    actor_log_std_max: float = 2
    actor_update_freq: int = 2

    # sac
    gamma: float = 0.9
    init_temperature: float = 0.1
    alpha_lr: float = 3e-4
    alpha_beta: float = 0.9

    log_freq: int = 10_000
    """log frequency in terms of global_step"""
    save_freq: int = 100_000
    """save frequency in terms of global_step"""
    eval_freq: Optional[int] = 100_000
    """evaluation frequency in terms of global_step"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    save_backup_ckpts: bool = False
    """whether to save separate ckpts eace save_freq which are not overwritten"""

    # passed from env/eval_env cfg
    num_steps: int = field(init=False)
    """the number of steps to run in each environment per policy rollout"""
    num_envs: int = field(init=False)
    """the number of parallel environments"""
    num_eval_envs: int = field(init=False)
    """the number of parallel environments"""
    num_iterations: int = field(init=False)
    """the number of iterations (computed in runtime)"""

    def _additional_processing(self):
        assert self.name == "sac", "Wrong algo config"

        self.replay_buffer_capacity = (
            self.replay_buffer_capacity // (self.num_envs * self.num_steps)
        ) * (self.num_envs * self.num_steps)
        assert (
            self.replay_buffer_capacity % self.num_steps == 0
        ), "SAC replay buffer needs capacity divisible by horizon"
        assert (
            self.replay_buffer_capacity // self.num_steps
        ) % self.num_envs == 0, (
            "SAC replay buffer needs max episodes divisible by num_envs"
        )

        self.num_iterations = math.ceil(self.total_timesteps / self.num_envs)


@dataclass
class TrainConfig:
    seed: int
    env: EnvConfig
    eval_env: EnvConfig
    algo: SACConfig
    logger: LoggerConfig

    wandb_id: Optional[str] = None
    resume_logdir: Optional[Union[Path, str]] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    def __post_init__(self):
        assert (
            self.resume_logdir is None or not self.logger.clear_out
        ), "Can't resume to a cleared out logdir!"

        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "if setting resume_logdir, must set logger workspace and exp_name accordingly"
            else:
                assert (
                    old_config_path.exists()
                ), f"Couldn't find old config at path {old_config_path}"
                old_config = get_mshab_train_cfg(
                    parse_cfg(default_cfg_path=old_config_path)
                )
                self.logger.workspace = old_config.logger.workspace
                self.logger.exp_path = old_config.logger.exp_path
                self.logger.log_path = old_config.logger.log_path
                self.logger.model_path = old_config.logger.model_path
                self.logger.train_video_path = old_config.logger.train_video_path
                self.logger.eval_video_path = old_config.logger.eval_video_path

            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo.num_envs = self.env.num_envs
        self.algo.num_steps = self.env.max_episode_steps
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    cfg.eval_env = {**cfg.env, **cfg.eval_env}
    cfg.eval_env.env_kwargs = {**cfg.env.env_kwargs, **cfg.eval_env.env_kwargs}
    cfg = from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))
    return cfg


def store_batch(
    replay_buffer: PixelStateBatchReplayBuffer,
    trunc,
    pixel_obs,
    pixel_next_obs,
    state_obs,
    state_next_obs,
    action,
    rew,
    term,
):
    # NOTE (arth): requre continuous task bc of replay buffer
    if trunc.any():
        return

    save_pixel_obs = to_numpy(pixel_obs, dtype=np.uint16)
    save_pixel_next_obs = to_numpy(pixel_next_obs, dtype=np.uint16)
    save_state_obs = to_numpy(state_obs)
    save_state_next_obs = to_numpy(state_next_obs)
    save_action = to_numpy(action)
    save_rew = to_numpy(rew)
    save_term = to_numpy(term)
    replay_buffer.store_batch(
        save_pixel_obs,
        save_pixel_next_obs,
        save_state_obs,
        save_state_next_obs,
        save_action,
        save_rew,
        save_term,
    )


def train(cfg: TrainConfig):

    # seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    # NOTE (arth): maybe require cuda since we only allow gpu sim anyways
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("making env", flush=True)
    envs = make_env(
        cfg.env,
        video_path=cfg.logger.train_video_path,
    )
    if cfg.algo.eval_freq is not None:
        print("making eval env", flush=True)
        eval_envs = make_env(
            cfg.eval_env,
            video_path=cfg.logger.eval_video_path,
        )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    print("made", flush=True)

    next_obs, _ = envs.reset(seed=cfg.seed)
    envs.action_space.seed(cfg.seed)
    if cfg.algo.eval_freq is not None:
        eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
        eval_envs.action_space.seed(cfg.seed + 1_000_000)

    obs_space = envs.unwrapped.single_observation_space
    pixels_obs_space: spaces.Dict = obs_space["pixels"]
    state_obs_space: spaces.Box = obs_space["state"]
    act_space = envs.unwrapped.single_action_space

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------
    model_pixel_obs_space = dict()
    for k, space in pixels_obs_space.items():
        shape, low, high, dtype = (
            space.shape,
            space.low,
            space.high,
            space.dtype,
        )
        if len(shape) == 4:
            shape = (shape[0] * shape[1], shape[-2], shape[-1])
            low = low.reshape((-1, *low.shape[-2:]))
            high = high.reshape((-1, *high.shape[-2:]))
        model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
    model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)

    agent = Agent(
        model_pixel_obs_space,
        state_obs_space.shape,
        act_space.shape,
        actor_hidden_dims=cfg.algo.actor_hidden_dims,
        critic_hidden_dims=cfg.algo.critic_hidden_dims,
        critic_layer_norm=cfg.algo.critic_layer_norm,
        critic_dropout=cfg.algo.critic_dropout,
        encoder_pixels_feature_dim=cfg.algo.encoder_pixels_feature_dim,
        encoder_state_feature_dim=cfg.algo.encoder_state_feature_dim,
        cnn_features=cfg.algo.cnn_features,
        cnn_filters=cfg.algo.cnn_filters,
        cnn_strides=cfg.algo.cnn_strides,
        cnn_padding=cfg.algo.cnn_padding,
        log_std_min=cfg.algo.actor_log_std_min,
        log_std_max=cfg.algo.actor_log_std_max,
        device=device,
    )

    log_alpha = torch.tensor(np.log(cfg.algo.init_temperature)).to(device)
    log_alpha.requires_grad = True
    target_entropy = -np.prod(act_space.shape)

    # optimizers
    actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=cfg.algo.actor_lr,
        betas=(cfg.algo.actor_beta, 0.999),
    )

    critic_optimizer = torch.optim.Adam(
        agent.critic.parameters(),
        lr=cfg.algo.critic_lr,
        betas=(cfg.algo.critic_beta, 0.999),
    )

    log_alpha_optimizer = torch.optim.Adam(
        [log_alpha], lr=cfg.algo.alpha_lr, betas=(cfg.algo.alpha_beta, 0.999)
    )

    agent.actor.train()
    agent.critic.train()
    agent.critic_target.train()

    def save(save_path):
        torch.save(
            dict(
                agent=agent.state_dict(),
                actor_optimizer=actor_optimizer.state_dict(),
                critic_optimizer=critic_optimizer.state_dict(),
                log_alpha_optimizer=log_alpha_optimizer.state_dict(),
                log_alpha=log_alpha,
            ),
            save_path,
        )

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=save,
    )

    print("og_log_alpha", log_alpha)
    resuming = False
    if cfg.model_ckpt is not None:
        resuming = True

        checkpoint = torch.load(str(cfg.model_ckpt), map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        log_alpha = checkpoint["log_alpha"]
        log_alpha_optimizer = torch.optim.Adam(
            [log_alpha], lr=cfg.algo.alpha_lr, betas=(cfg.algo.alpha_beta, 0.999)
        )
        log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])

        logger.print(f"Sucessfully loaded {cfg.model_ckpt}, {resuming=}", flush=True)
    print("new_log_alpha", log_alpha, flush=True)

    # -------------------------------------------------------------------------------------------------
    # MEMORY
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): we use horizon - 1 since we ignore the data
    #   from the final step of the episode
    replay_horizon = cfg.algo.num_steps - 1
    replay_buffer = PixelStateBatchReplayBuffer(
        pixels_obs_space=pixels_obs_space,
        state_obs_dim=np.prod(state_obs_space.shape),
        act_dim=np.prod(act_space.shape),
        size=((cfg.algo.replay_buffer_capacity // cfg.algo.num_steps) * replay_horizon),
        horizon=replay_horizon,
        num_envs=cfg.algo.num_envs,
    )

    global_step, global_start_step, iteration = (
        logger.last_log_step,
        logger.last_log_step,
        0,
    )

    def check_freq(freq):
        return (global_step % freq < cfg.algo.num_envs) or (
            iteration == cfg.algo.num_iterations - 1
        )

    def store_env_stats(key):
        assert key in ["eval", "train"]
        if key == "eval":
            log_env = eval_envs
        else:
            log_env = envs
        logger.store(
            key,
            return_per_step=common.to_tensor(log_env.return_queue, device=device)
            .float()
            .mean()
            / log_env.max_episode_steps,
            success_once=common.to_tensor(log_env.success_once_queue, device=device)
            .float()
            .mean(),
            success_at_end=common.to_tensor(log_env.success_at_end_queue, device=device)
            .float()
            .mean(),
            len=common.to_tensor(log_env.length_queue, device=device).float().mean(),
        )
        log_env.reset_queues()

    to_tensor_transform_fn = lambda x: to_tensor(x, device=device, dtype="float")
    to_numpy_transfrom_fn = lambda x: to_numpy(x)

    def pixel_state_obs(obs, tensor=False, to_device=False):
        if tensor:
            return to_tensor_transform_fn(obs["pixels"]), to_tensor_transform_fn(
                obs["state"]
            )
        return to_numpy_transfrom_fn(obs["pixels"]), to_numpy_transfrom_fn(obs["state"])

    print("start", flush=True)
    pixel_obs, state_obs = pixel_state_obs(next_obs, tensor=True, to_device=True)
    timer = NonOverlappingTimeProfiler()
    for iteration in range(cfg.algo.num_iterations):
        if global_step > cfg.algo.total_timesteps:
            break

        logger.print(f"Epoch: {iteration}, {global_step=}", flush=True)

        if not resuming and len(replay_buffer) < cfg.algo.init_steps:
            action = envs.action_space.sample()
        else:
            with torch.no_grad():
                _, action, _, _ = agent.actor(
                    pixel_obs, state_obs, compute_log_pi=False
                )

        timer.end(key="get_action")

        # Step the env, get next observation, reward and done signal
        next_obs, rew, term, trunc, info = envs.step(action)

        timer.end(key="sim_sample")

        # store batched data into buffer
        pixel_next_obs, state_next_obs = pixel_state_obs(
            next_obs, tensor=True, to_device=True
        )
        store_batch(
            replay_buffer,
            trunc,
            pixel_obs,
            pixel_next_obs,
            state_obs,
            state_next_obs,
            action,
            rew,
            term,
        )
        global_step += envs.num_envs

        timer.end(key="save_to_buf")

        # let agent update
        loss_logs = dict()
        if len(replay_buffer) >= cfg.algo.init_steps:
            num_updates = 1
            if not resuming and len(replay_buffer) == cfg.algo.init_steps:
                num_updates = cfg.algo.init_steps
            for _ in range(num_updates):
                batch = replay_buffer.sample_batch(batch_size=cfg.algo.batch_size)
                batch = to_tensor(batch, device=device, dtype="float")

                (
                    b_pixel_obs,
                    b_pixel_next_obs,
                    b_state_obs,
                    b_state_next_obs,
                    b_act,
                    b_rew,
                    b_term,
                ) = (
                    batch["pixel_obs"],
                    batch["pixel_next_obs"],
                    batch["state_obs"],
                    batch["state_next_obs"],
                    batch["act"],
                    batch["rew"],
                    batch["term"],
                )

                # Update critic
                with torch.no_grad():
                    not_done = (1 - b_term).unsqueeze(-1)
                    b_rew = b_rew.unsqueeze(-1)
                    _, policy_action, log_pi, _ = agent.actor(
                        b_pixel_next_obs, b_state_next_obs
                    )
                    target_Q1, target_Q2 = agent.critic_target(
                        b_pixel_next_obs, b_state_next_obs, policy_action
                    )
                    target_V = (
                        torch.min(target_Q1, target_Q2)
                        - log_alpha.exp().detach() * log_pi
                    )
                    target_Q = b_rew + (not_done * cfg.algo.gamma * target_V)

                # get current Q estimates
                current_Q1, current_Q2 = agent.critic(
                    b_pixel_obs, b_state_obs, b_act, detach=cfg.algo.detach_encoder
                )
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                    current_Q2, target_Q
                )

                # Optimize the critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                critic_loss_logs = dict(critic_loss=critic_loss)

                # Update actor
                actor_loss_logs = dict()
                if iteration % cfg.algo.actor_update_freq == 0:
                    _, pi, log_pi, log_std = agent.actor(
                        b_pixel_obs, b_state_obs, detach=True
                    )
                    actor_Q1, actor_Q2 = agent.critic(
                        b_pixel_obs, b_state_obs, pi, detach=True
                    )

                    actor_Q = torch.min(actor_Q1, actor_Q2)
                    actor_loss = (log_alpha.exp().detach() * log_pi - actor_Q).mean()

                    entropy = 0.5 * log_std.shape[1] * (
                        1.0 + np.log(2 * np.pi)
                    ) + log_std.sum(dim=-1)

                    # optimize the actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss = (
                        log_alpha.exp() * (-log_pi - target_entropy).detach()
                    ).mean()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    actor_loss_logs = dict(
                        actor_loss=actor_loss,
                        target_entropy=target_entropy,
                        entropy_mean=entropy.mean(),
                        alpha_loss=alpha_loss,
                        alpha=log_alpha.exp(),
                        log_alpha=log_alpha,
                    )

                if iteration % cfg.algo.critic_target_update_freq == 0:
                    agent.soft_update_params(
                        cfg.algo.critic_tau, cfg.algo.critic_encoder_tau
                    )

                loss_logs = dict(**critic_loss_logs, **actor_loss_logs)
            timer.end(key="train")

        # set obs to next obs
        pixel_obs, state_obs = pixel_next_obs, state_next_obs

        # Log
        if check_freq(cfg.algo.log_freq):
            if len(envs.return_queue) > 0:
                store_env_stats("train")
            if iteration > 0:
                logger.store(
                    "time", **timer.get_time_logs(global_step - global_start_step)
                )
            logger.store(tag="losses", **loss_logs)

            logger.log(global_step)
            timer.end(key="log")

        # Evaluation
        if cfg.algo.eval_freq is not None and check_freq(cfg.algo.eval_freq):
            agent.actor.eval()
            agent.critic.eval()
            eval_obs, _ = eval_envs.reset()  # don't seed here

            for _ in range(eval_envs.max_episode_steps):
                with torch.no_grad():
                    eval_pixel_obs, eval_state_obs = pixel_state_obs(
                        eval_obs, tensor=True, to_device=True
                    )
                    action, _, _, _ = agent.actor(
                        eval_pixel_obs,
                        eval_state_obs,
                        compute_pi=False,
                        compute_log_pi=False,
                    )
                eval_obs, _, _, _, _ = eval_envs.step(action)

            agent.actor.train()
            agent.critic.train()

            if len(eval_envs.return_queue) > 0:
                store_env_stats("eval")
            logger.log(global_step)
            timer.end(key="eval")

        # Checkpoint
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{global_step}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end(key="checkpoint")

    save(logger.model_path / "final_ckpt.pt")
    save(logger.model_path / "latest.pt")

    # close everyhting once script is done running
    envs.close()
    if cfg.algo.eval_freq is not None:
        eval_envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
