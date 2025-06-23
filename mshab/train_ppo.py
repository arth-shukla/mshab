# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

from dacite import from_dict
from omegaconf import OmegaConf

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import common

from mshab.agents.ppo import Agent, DictArray
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import recursive_slice
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


@dataclass
class PPOConfig:
    name: str = "ppo"

    total_timesteps: int = 100_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.9
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""

    log_freq: int = 10_000
    """log frequency in terms of global_step"""
    save_freq: int = 100_000
    """save frequency in terms of global_step"""
    eval_freq: Optional[int] = 100_000
    """evaluation frequency in terms of global_step"""
    finite_horizon_gae: bool = True
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    save_backup_ckpts: bool = False
    """whether to save separate ckpts eace save_freq which are not overwritten"""

    num_steps: Optional[int] = None
    """the number of steps to run in each environment per policy rollout"""

    # passed from env/eval_env cfg
    num_envs: int = field(init=False)
    """the number of parallel environments"""
    num_eval_envs: int = field(init=False)
    """the number of parallel environments"""
    # filled in after above passed from cfg.env
    batch_size: int = field(init=False)
    """the batch size (computed in runtime)"""
    minibatch_size: int = field(init=False)
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = field(init=False)
    """the number of iterations (computed in runtime)"""

    def _additional_processing(self):
        assert self.name == "ppo", "Wrong algo config"

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_iterations = math.ceil(self.total_timesteps / self.batch_size)


@dataclass
class TrainConfig:
    seed: int
    env: EnvConfig
    eval_env: EnvConfig
    algo: PPOConfig
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
        if self.algo.num_steps is None:
            self.algo.num_steps = self.env.max_episode_steps
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    cfg.eval_env = {**cfg.env, **cfg.eval_env}
    cfg.eval_env.env_kwargs = {**cfg.env.env_kwargs, **cfg.eval_env.env_kwargs}
    # NOTE (arth): odd OmageConf quirk where wandb_id=null -> wandb_id=True
    if hasattr(cfg, "wandb_id") and isinstance(cfg.wandb_id, bool):
        cfg.wandb_id = None
    cfg = from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))
    return cfg


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
    print("making eval env", flush=True)
    if cfg.algo.eval_freq:
        eval_envs = make_env(
            cfg.eval_env,
            video_path=cfg.logger.eval_video_path,
        )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    print("made", flush=True)

    next_obs, _ = envs.reset(seed=cfg.seed)
    if cfg.algo.eval_freq:
        eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)

    agent = Agent(
        sample_obs=next_obs, single_act_shape=envs.single_action_space.shape
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.algo.learning_rate, eps=1e-5)

    def save(save_path):
        torch.save(
            dict(agent=agent.state_dict(), optimizer=optimizer.state_dict()), save_path
        )

    def load(load_path):
        checkpoint = torch.load(load_path)
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=save,
    )

    if cfg.model_ckpt is not None:
        load(cfg.model_ckpt)

    print("start", flush=True)
    # -------------------------------------------------------------------------------------------------
    # MEMORY
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): we never "reset" this, just overwrite old data
    # -------------------------------------------------------------------------------------------------
    obs = DictArray(
        (cfg.algo.num_steps, cfg.algo.num_envs),
        envs.single_observation_space,
        device=device,
    )
    actions = torch.zeros(
        (cfg.algo.num_steps, cfg.algo.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((cfg.algo.num_steps, cfg.algo.num_envs)).to(device)
    rewards = torch.zeros((cfg.algo.num_steps, cfg.algo.num_envs)).to(device)
    dones = torch.zeros((cfg.algo.num_steps, cfg.algo.num_envs)).to(device)
    values = torch.zeros((cfg.algo.num_steps, cfg.algo.num_envs)).to(device)

    global_step, global_start_step, iteration = (
        logger.last_log_step,
        logger.last_log_step,
        0,
    )
    next_done = torch.zeros(cfg.algo.num_envs, device=device)
    timer = NonOverlappingTimeProfiler()

    def check_freq(freq):
        return (global_step % freq < cfg.algo.batch_size) or (
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
        extra_stat_logs = dict()
        for k, v in log_env.extra_stats.items():
            extra_stat_values = torch.stack(v)
            extra_stat_logs[f"{k}_once"] = torch.mean(
                torch.any(extra_stat_values, dim=1).float()
            )
            extra_stat_logs[f"{k}_at_end"] = torch.mean(
                extra_stat_values[..., -1].float()
            )
        logger.store(f"extra/{key}", **extra_stat_logs)
        log_env.reset_queues()

    for iteration in range(cfg.algo.num_iterations):

        if global_step > cfg.algo.total_timesteps:
            break

        logger.print(f"Epoch: {iteration}, {global_step=}", flush=True)
        final_values = torch.zeros(
            (cfg.algo.num_steps, cfg.algo.num_envs), device=device
        )
        agent.eval()

        # ---------------------------------------------------------------------------------------------
        # MISC
        # ---------------------------------------------------------------------------------------------

        # anneal lr (if option passed)
        if cfg.algo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.algo.num_iterations
            lrnow = frac * cfg.algo.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # checkpoint model
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{global_step}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end("save")

        # log train stats
        if check_freq(cfg.algo.log_freq) and iteration > 0:
            store_env_stats("train")
            if iteration > 0:
                logger.store(
                    "time", **timer.get_time_logs(global_step - global_start_step)
                )
            logger.log(global_step)
            timer.end("log")

        # ---------------------------------------------------------------------------------------------
        # EVAL
        # ---------------------------------------------------------------------------------------------
        if cfg.algo.eval_freq and check_freq(cfg.algo.eval_freq):
            eval_envs.reset()
            for _ in range(eval_envs.max_episode_steps):
                with torch.no_grad():
                    eval_obs, _, _, _, _ = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True)
                    )
            store_env_stats("eval")
            logger.log(global_step)

        timer.end("eval")

        # ---------------------------------------------------------------------------------------------
        # ROLLOUT
        # ---------------------------------------------------------------------------------------------
        for step in range(0, cfg.algo.num_steps):
            global_step += cfg.algo.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                infos["final_observation"] = recursive_slice(
                    infos["final_observation"], done_mask
                )
                final_values[
                    step, torch.arange(cfg.algo.num_envs, device=device)[done_mask]
                ] = agent.get_value(infos["final_observation"]).view(-1)

        timer.end("sim_sample")

        # ---------------------------------------------------------------------------------------------
        # BOOTSTRAP
        # ---------------------------------------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.algo.num_steps)):
                if t == cfg.algo.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = (
                    next_not_done * nextvalues + final_values[t]
                )  # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if cfg.algo.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == cfg.algo.num_steps - 1:  # initialize
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0  # the sum of the second term
                        value_term_sum = 0.0  # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + cfg.algo.gae_lambda * lam_coef_sum
                    reward_term_sum = (
                        cfg.algo.gae_lambda * cfg.algo.gamma * reward_term_sum
                        + lam_coef_sum * rewards[t]
                    )
                    value_term_sum = (
                        cfg.algo.gae_lambda * cfg.algo.gamma * value_term_sum
                        + cfg.algo.gamma * real_next_values
                    )

                    advantages[t] = (
                        reward_term_sum + value_term_sum
                    ) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + cfg.algo.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = (
                        delta
                        + cfg.algo.gamma
                        * cfg.algo.gae_lambda
                        * next_not_done
                        * lastgaelam
                    )  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        timer.end("bootstrap")

        # ---------------------------------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------------------------------
        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(cfg.algo.batch_size)
        clipfracs = []

        for epoch in range(cfg.algo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.algo.batch_size, cfg.algo.minibatch_size):
                end = start + cfg.algo.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.algo.clip_coef).float().mean().item()
                    ]

                if cfg.algo.target_kl is not None and approx_kl > cfg.algo.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if cfg.algo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.algo.clip_coef, 1 + cfg.algo.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.algo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.algo.clip_coef,
                        cfg.algo.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - cfg.algo.ent_coef * entropy_loss
                    + v_loss * cfg.algo.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.algo.max_grad_norm)
                optimizer.step()

            if cfg.algo.target_kl is not None and approx_kl > cfg.algo.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.store(
            "losses",
            value_loss=v_loss.item(),
            policy_loss=pg_loss.item(),
            entropy=entropy_loss.item(),
            old_approx_kl=old_approx_kl.item(),
            approx_kl=approx_kl.item(),
            clipfrac=np.mean(clipfracs),
            explained_variance=explained_var,
        )
        logger.store(
            "charts",
            learning_rate=optimizer.param_groups[0]["lr"],
        )

        timer.end("train")

    save(logger.model_path / "final_ckpt.pt")
    save(logger.model_path / "latest.pt")

    # close everyhting once script is done running
    envs.close()
    if cfg.algo.eval_freq:
        eval_envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
