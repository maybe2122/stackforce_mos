import argparse
import inspect
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained MosOne closed-chain USD policy.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="MosOne-Mos20262ClosedUsd-ClosedUsd-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--load_run", type=str, default=".*")
parser.add_argument("--num_steps", type=int, default=5000, help="Number of steps to run. Use 0 to run until the window is closed.")
parser.add_argument("--torque_limit", type=float, default=30.0, help="Torque limit (N·m) used as the y-axis range / reference line in the torque plots.")
parser.add_argument(
    "--disable_resets",
    action="store_true",
    default=False,
    help="Disable environment reset during visual play so short or unstable policies do not instantly jump back to the start pose.",
)
parser.add_argument(
    "--terrain",
    type=str,
    default="flat",
    choices=["flat", "rough", "curriculum"],
    help=(
        "Ground terrain: 'flat' (plane, default), 'rough' (procedural heightfield), "
        "or 'curriculum' (start flat, progress to rough + stairs as envs succeed)."
    ),
)
parser.add_argument("--no_privileged_critic", action="store_true",
                    help="关闭非对称 critic 特权观测（state_space=0）。回放 2026-07-05 之前"
                         "训练的旧 checkpoint（对称 critic）时必须加此参数，否则维度不匹配。")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
try:
    from rsl_rl.algorithms import PPO
except ImportError:
    PPO = None
try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import mos_one.tasks  # noqa: F401
from mos_one.tasks.direct.mos2026_2_closed_usd.mos2026_2_closed_usd_env_cfg import (
    CURRICULUM_TERRAIN_CFG,
    ROUGH_TERRAIN_CFG,
)


def _runner_uses_obs_groups():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return "resolve_obs_groups" in source or '"obs_groups"' in source or "'obs_groups'" in source


def _runner_uses_split_actor_critic():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return 'cfg["actor"]' in source or "cfg['actor']" in source or 'cfg["critic"]' in source or "cfg['critic']" in source


def _runner_expects_privileged_step():
    try:
        source = inspect.getsource(OnPolicyRunner.learn)
    except OSError:
        return True
    return "privileged_obs" in source or "critic_obs" in source


def _format_rsl_rl_obs(obs_dict, use_obs_groups):
    if not use_obs_groups:
        return obs_dict["policy"]
    if TensorDict is not None and not isinstance(obs_dict, TensorDict):
        first_obs = next(iter(obs_dict.values()))
        return TensorDict(dict(obs_dict), batch_size=[first_obs.shape[0]], device=first_obs.device)
    return obs_dict


class LegacyRslRlVecEnvWrapper:
    def __init__(self, env, clip_actions=None):
        self.env = env
        self.clip_actions = clip_actions
        self.use_obs_groups = _runner_uses_obs_groups()
        self.return_privileged_obs = _runner_expects_privileged_step()
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.cfg = env.unwrapped.cfg
        self.num_actions = gym.spaces.flatdim(env.unwrapped.single_action_space)
        obs_dict, extras = self.env.reset()
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.num_obs = obs_dict["policy"].shape[-1]
        self.num_privileged_obs = self.privileged_obs_buf.shape[-1] if self.privileged_obs_buf is not None else None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = env.unwrapped.episode_length_buf
        self.extras = extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self, env_ids=None):
        del env_ids
        obs_dict, extras = self.env.reset()
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.extras = extras
        if not self.return_privileged_obs:
            return self.obs_buf
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.env.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        if "log" in extras and "episode" not in extras:
            extras["episode"] = extras["log"]
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = extras
        if not self.return_privileged_obs:
            return self.obs_buf, rewards, dones, extras
        return self.obs_buf, self.privileged_obs_buf, rewards, dones, extras

    def close(self):
        return self.env.close()


def _runner_uses_nested_class_name():
    try:
        source = inspect.getsource(OnPolicyRunner)
    except OSError:
        return False
    return (
        'algorithm"]["class_name' in source
        or "algorithm']['class_name" in source
        or 'policy_cfg.pop("class_name")' in source
        or "policy_cfg.pop('class_name')" in source
        or 'self.policy_cfg.pop("class_name")' in source
        or "self.policy_cfg.pop('class_name')" in source
        or "resolve_callable" in source
    )


def _runner_uses_split_actor_critic():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return 'cfg["actor"]' in source or "cfg['actor']" in source or 'cfg["critic"]' in source or "cfg['critic']" in source


def _runner_expects_nested_runner():
    try:
        source = inspect.getsource(OnPolicyRunner)
    except OSError:
        return True
    return 'train_cfg["runner"]' in source or "train_cfg['runner']" in source


def to_compatible_rsl_rl_cfg(agent_cfg, has_critic_obs=False):
    # has_critic_obs：env 输出独立 "critic" 特权观测组时为 True（非对称 critic），
    # obs_groups 的 critic 指向 "critic" 组；否则退回与 actor 共用 "policy"。
    critic_group = ["critic"] if has_critic_obs else ["policy"]
    data = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    allowed_policy_keys = {
        "actor_hidden_dims", "critic_hidden_dims", "activation", "init_noise_std", "clip_actions",
        "actor_obs_normalization", "critic_obs_normalization"
    }
    allowed_algorithm_keys = {
        "num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam", "value_loss_coef",
        "entropy_coef", "learning_rate", "max_grad_norm", "use_clipped_value_loss", "schedule", "desired_kl", "use_spo"
    }
    if "runner" in data and "policy" in data and "algorithm" in data:
        runner_cfg = dict(data["runner"])
        policy_cfg = {key: value for key, value in dict(data["policy"]).items() if key in allowed_policy_keys or key == "class_name"}
        algorithm_cfg = {key: value for key, value in dict(data["algorithm"]).items() if key in allowed_algorithm_keys or key == "class_name"}
    else:
        policy_cfg = {key: value for key, value in dict(data["policy"]).items() if key in allowed_policy_keys}
        algorithm_cfg = {key: value for key, value in dict(data["algorithm"]).items() if key in allowed_algorithm_keys}
        runner_cfg = {key: value for key, value in data.items() if key not in {"policy", "algorithm", "class_name"}}
    runner_cfg.setdefault("num_steps_per_env", getattr(agent_cfg, "num_steps_per_env", 24))
    runner_cfg.setdefault("max_iterations", getattr(agent_cfg, "max_iterations", 1500))
    runner_cfg.setdefault("save_interval", getattr(agent_cfg, "save_interval", 50))
    runner_cfg.setdefault("obs_groups", {"policy": ["policy"], "critic": critic_group})
    runner_cfg.setdefault("experiment_name", getattr(agent_cfg, "experiment_name", "mos_one"))
    runner_cfg.setdefault("run_name", getattr(agent_cfg, "run_name", ""))
    runner_cfg.setdefault("resume", getattr(agent_cfg, "resume", False))
    runner_cfg.setdefault("load_run", getattr(agent_cfg, "load_run", ".*"))
    runner_cfg.setdefault("checkpoint", getattr(agent_cfg, "load_checkpoint", "model_.*.pt"))
    if _runner_uses_nested_class_name():
        policy_cfg.setdefault("class_name", "ActorCritic")
        algorithm_cfg.setdefault("class_name", "PPO")
    else:
        runner_cfg.setdefault("policy_class_name", "ActorCritic")
        runner_cfg.setdefault("algorithm_class_name", "PPO")
    if _runner_expects_nested_runner():
        return {"runner": runner_cfg, "policy": policy_cfg, "algorithm": algorithm_cfg}
    if _runner_uses_split_actor_critic():
        algorithm_cfg.setdefault("class_name", "PPO")
        algorithm_cfg.pop("use_spo", None)
        actor_cfg = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", [256, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": policy_cfg.get("init_noise_std", 1.0),
            },
        }
        critic_cfg = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", [256, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }
        runner_cfg.pop("policy_class_name", None)
        runner_cfg.pop("algorithm_class_name", None)
        runner_cfg["obs_groups"] = {"actor": ["policy"], "critic": critic_group, "policy": ["policy"]}
        runner_cfg.setdefault("multi_gpu", None)
        return {**runner_cfg, "actor": actor_cfg, "critic": critic_cfg, "algorithm": algorithm_cfg}
    return {**runner_cfg, "policy": policy_cfg, "algorithm": algorithm_cfg}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_resets and hasattr(env_cfg, "visual_disable_resets"):
        env_cfg.visual_disable_resets = True
    # 回放按全速命令：关闭训练用的命令速度课程。
    if hasattr(env_cfg, "command_curriculum_steps"):
        env_cfg.command_curriculum_steps = 0
    # 旧 checkpoint（对称 critic）兼容：按需关闭特权观测。
    if args_cli.no_privileged_critic and hasattr(env_cfg, "state_space"):
        env_cfg.state_space = 0
    if args_cli.terrain == "rough":
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = ROUGH_TERRAIN_CFG
        env_cfg.terrain.max_init_terrain_level = None
        env_cfg.terrain_curriculum_enabled = False
    elif args_cli.terrain == "curriculum":
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = CURRICULUM_TERRAIN_CFG
        # On play, no curriculum advancement — keep envs at their initial
        # level so behaviour on every terrain row is observable.
        env_cfg.terrain.max_init_terrain_level = None
        env_cfg.terrain_curriculum_enabled = False
    else:
        env_cfg.terrain.terrain_type = "plane"
        env_cfg.terrain.terrain_generator = None
        env_cfg.terrain_curriculum_enabled = False
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_arg = args_cli.checkpoint
    resume_path = os.path.abspath(checkpoint_arg) if os.path.isfile(checkpoint_arg) else get_checkpoint_path(
        log_root_path, args_cli.load_run, checkpoint_arg
    )
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped_env = LegacyRslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(
        wrapped_env,
        to_compatible_rsl_rl_cfg(agent_cfg, has_critic_obs=wrapped_env.num_privileged_obs is not None),
        log_dir=None,
        device=env.unwrapped.device,
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs_dict, _ = env.reset()
    steps = 0

    # --- 力矩监控初始化 ---
    robot = env.unwrapped._robot
    joint_ids = env.unwrapped._actuated_joint_ids
    joint_names = env.unwrapped._actuated_joint_names
    torque_history = []  # 每个元素: shape (num_joints,) 的 cpu tensor（监控 env 0）

    # --- TensorBoard 实时曲线 ---
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_log_dir = os.path.join(os.path.dirname(resume_path), "play_torque_tb")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[INFO] TensorBoard 力矩日志: {tb_log_dir}", flush=True)
        print(f"[INFO] 实时查看: tensorboard --logdir {tb_log_dir}", flush=True)
    except ImportError:
        print("[WARN] 未安装 tensorboard，跳过实时曲线（pip install tensorboard）", flush=True)

    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs_dict)
            obs_dict, _, _, _, _ = env.step(actions)
            steps += 1

            # 实际施加到各关节的力矩 (num_envs, num_joints) -> 取第 0 个环境
            tau = robot.data.applied_torque[:, joint_ids][0].detach().cpu()
            torque_history.append(tau.clone())

            # 写入 TensorBoard（每步一条，可实时刷新）
            # 每个关节单图叠加 ±力矩上限参考线，使 y 轴范围始终覆盖 ±torque_limit。
            if tb_writer is not None:
                lim = args_cli.torque_limit
                for n, val in zip(joint_names, tau.tolist()):
                    tb_writer.add_scalars(
                        f"torque/{n}",
                        {"tau": val, "+limit": lim, "-limit": -lim},
                        steps,
                    )
                tb_writer.flush()

            if args_cli.num_steps > 0 and steps >= args_cli.num_steps:
                break

    # --- 力矩统计汇总 ---
    if torque_history:
        data = torch.stack(torque_history, dim=0)  # (steps, num_joints)
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        tmin = data.min(dim=0).values
        tmax = data.max(dim=0).values
        absmax = data.abs().max(dim=0).values
        rms = data.pow(2).mean(dim=0).sqrt()
        print("\n================ 各电机力矩统计 (单位 N·m) ================", flush=True)
        header = f"{'joint':<22}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'|max|':>9}{'rms':>9}"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for i, n in enumerate(joint_names):
            print(
                f"{n:<22}{mean[i]:>9.3f}{std[i]:>9.3f}{tmin[i]:>9.3f}"
                f"{tmax[i]:>9.3f}{absmax[i]:>9.3f}{rms[i]:>9.3f}",
                flush=True,
            )
        print("-" * len(header), flush=True)
        print(
            f"{'ALL':<22}{mean.mean():>9.3f}{std.mean():>9.3f}{tmin.min():>9.3f}"
            f"{tmax.max():>9.3f}{absmax.max():>9.3f}{rms.mean():>9.3f}",
            flush=True,
        )
        # 保存到 csv，便于离线分析
        csv_path = os.path.join(os.path.dirname(resume_path), "play_torque_stats.csv")
        try:
            import csv

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["joint", "mean", "std", "min", "max", "abs_max", "rms"])
                for i, n in enumerate(joint_names):
                    writer.writerow(
                        [n, f"{mean[i]:.6f}", f"{std[i]:.6f}", f"{tmin[i]:.6f}",
                         f"{tmax[i]:.6f}", f"{absmax[i]:.6f}", f"{rms[i]:.6f}"]
                    )
            print(f"[INFO] 力矩统计已保存: {csv_path}", flush=True)
        except OSError as exc:
            print(f"[WARN] 力矩统计保存失败: {exc}", flush=True)

        # --- 力矩曲线绘制 (每个关节 力矩 vs 步数) ---
        try:
            import math

            import matplotlib

            matplotlib.use("Agg")  # 无界面后端，避免阻塞 GUI 渲染
            import matplotlib.pyplot as plt

            num_joints = data.shape[1]
            ncols = 3
            nrows = math.ceil(num_joints / ncols)
            t = torch.arange(data.shape[0]).numpy()
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.6 * nrows), squeeze=False)
            lim = args_cli.torque_limit
            for i, n in enumerate(joint_names):
                ax = axes[i // ncols][i % ncols]
                # 颜色分级（仿 robot_web drawTorque）：按本关节峰值 |τ| 相对上限的比例。
                #   绿 = 裕度充足(<80%)，橙 = 接近上限(≥80%)，红 = 达到/超过上限(≥100%)。
                peak = float(absmax[i])
                if lim > 0 and peak >= lim:
                    color, tag = "#d64545", " ≥上限"
                elif lim > 0 and peak >= 0.8 * lim:
                    color, tag = "#e08a1e", " 接近上限"
                else:
                    color, tag = "#2a8a3e", ""
                ax.plot(t, data[:, i].numpy(), color=color, linewidth=0.8)
                ax.axhline(0.0, color="gray", linewidth=0.5)
                ax.axhline(lim, color="red", linestyle="--", linewidth=0.6)
                ax.axhline(-lim, color="red", linestyle="--", linewidth=0.6)
                ax.set_ylim(-lim * 1.1, lim * 1.1)
                ax.set_title(f"{n}  |max|={peak:.1f}{tag}", fontsize=9, color=color)
                ax.set_xlabel("step")
                ax.set_ylabel("N·m")
                ax.grid(True, alpha=0.3)
            # 隐藏多余空子图
            for j in range(num_joints, nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")
            fig.tight_layout()
            png_path = os.path.join(os.path.dirname(resume_path), "play_torque_curves.png")
            fig.savefig(png_path, dpi=120)
            plt.close(fig)
            print(f"[INFO] 力矩曲线已保存: {png_path}", flush=True)
        except ImportError:
            print("[WARN] 未安装 matplotlib，跳过曲线绘制（pip install matplotlib）", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 力矩曲线绘制失败: {exc}", flush=True)

    if tb_writer is not None:
        tb_writer.close()

    print(f"PLAY_COMPLETED steps={steps} checkpoint={resume_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
