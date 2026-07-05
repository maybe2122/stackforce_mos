"""四足闭链 USD 策略离线评估脚本。

在训练环境里以**多并行环境、关闭探索噪声**的方式跑策略，统计一整套验证指标，
并把结果写成 JSON（供 eval_report.py 汇总成验证表）。

设计目标：与 play.py 共享同一套 rsl_rl 版本兼容 shim，但面向**批量定量评估**而非可视化。
单次调用 = 单个测试条件（一组指令 / 地形 / 摩擦 / 噪声 / 延迟 / 推搡）。
要跑完整验证矩阵请用 eval_suite.sh 多次调用本脚本，再用 eval_report.py 汇总。

指标（写入 JSON.metrics）：
  存活/成功     success_rate, fall_rate, mean_survival_s, mean_return
  速度跟踪      vx_mae, vx_rmse, vy_mae, vy_rmse, yaw_mae, yaw_rmse, mean_speed
  姿态          mean_base_height, base_height_target, mean_upright_err
  力矩/能耗     per-joint 力矩 mean/rms/|max|/近上限占比；mean_power_w；mean_cot
  步态          per-foot 占空比 duty_factor、触地频率 touchdown_hz、对角同相位率（trot 检测）
  足端打滑      mean_foot_slip

示例：
  # 平地、默认前进指令、64 个环境、跑 3000 步
  python scripts/rsl_rl/eval.py --headless --num_envs 64 --num_steps 3000 \
      --load_run 2026-06-07_22-11-43 --tag flat_default

  # 侧向 + 转身指令
  python scripts/rsl_rl/eval.py --headless --cmd_vx 0.5 --cmd_vy 0.2 --cmd_wz 0.5 --tag cmd_turn

  # 低摩擦泛化
  python scripts/rsl_rl/eval.py --headless --friction 0.4 --tag friction_0p4

  # 鲁棒性：每 2 秒侧推 0.6 m/s 速度冲量
  python scripts/rsl_rl/eval.py --headless --push_interval_s 2.0 --push_vel 0.6 --tag push_0p6

  # Sim-to-real：观测噪声 + 20ms 控制延迟
  python scripts/rsl_rl/eval.py --headless --obs_noise 0.05 --action_delay 1 --tag noise_delay
"""

import argparse
import inspect
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained MosOne closed-chain USD policy.")
parser.add_argument("--num_envs", type=int, default=64, help="并行评估环境数。越多统计越稳，受显存限制。")
parser.add_argument("--task", type=str, default="MosOne-Mos20262ClosedUsd-ClosedUsd-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--load_run", type=str, default=".*")
parser.add_argument("--num_steps", type=int, default=3000, help="评估控制步数（dt=0.02s；1000 步 = 20s = 一个完整回合）。")
parser.add_argument("--warmup_episodes", type=int, default=1,
                    help="每个环境丢弃前 N 个回合再开始统计（初始回合长度被随机化，是部分回合）。")
parser.add_argument("--torque_limit", type=float, default=None,
                    help="力矩上限 N·m（用于近上限占比）。默认读 cfg 的 effort_limit_sim。")
parser.add_argument("--foot_contact_height", type=float, default=None,
                    help="足端接触判定高度阈值 m（步态/打滑指标用）。默认读 cfg；shank body 偏高时调大（如 0.15）。")
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough", "curriculum"],
                    help="地面类型：flat / rough(随机高度场) / curriculum(平地→楼梯等)。")
# --- 测试条件覆盖 ---
parser.add_argument("--cmd_vx", type=float, default=None, help="覆盖前进指令速度 vx (m/s)。")
parser.add_argument("--cmd_vy", type=float, default=None, help="覆盖侧向指令速度 vy (m/s)。")
parser.add_argument("--cmd_wz", type=float, default=None, help="覆盖偏航指令角速度 wz (rad/s)。")
parser.add_argument("--friction", type=float, default=None, help="覆盖地面静/动摩擦系数（泛化测试）。")
parser.add_argument("--mass_scale", type=float, default=1.0, help="按比例缩放机器人所有刚体质量（负载测试，例 1.1=+10%%）。")
parser.add_argument("--obs_noise", type=float, default=0.0, help="加到 policy 观测上的高斯噪声标准差（传感器噪声测试）。")
parser.add_argument("--action_delay", type=int, default=0, help="动作延迟步数（每步=20ms；延迟测试）。")
parser.add_argument("--push_interval_s", type=float, default=0.0, help=">0 时每隔该秒数对 base 施加一次随机水平速度冲量（推搡测试）。")
parser.add_argument("--push_vel", type=float, default=0.5, help="推搡冲量大小 (m/s)，方向随机水平。")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_privileged_critic", action="store_true",
                    help="关闭非对称 critic 特权观测（state_space=0）。评估 2026-07-05 之前"
                         "训练的旧 checkpoint（对称 critic）时必须加此参数，否则维度不匹配。")
parser.add_argument("--tag", type=str, default="eval", help="结果标签，用于 JSON 文件名与报告分组。")
parser.add_argument("--out_dir", type=str, default=None, help="JSON 输出目录，默认 <checkpoint目录>/eval。")
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


# =============================================================================
# rsl_rl 版本兼容 shim —— 与 play.py 保持一致，用来在不同 rsl_rl 版本下加载策略。
# （play.py 内嵌了同一份；如需统一可抽到共享模块，这里为零风险保持自包含。）
# =============================================================================
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


def _runner_expects_nested_runner():
    try:
        source = inspect.getsource(OnPolicyRunner)
    except OSError:
        return True
    return 'train_cfg["runner"]' in source or "train_cfg['runner']" in source


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


# =============================================================================
# 评估主逻辑
# =============================================================================
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed

    # --- 指令覆盖 ---
    cmd_xy = list(getattr(env_cfg, "commanded_lin_vel_xy", (1.0, 0.0)))
    if args_cli.cmd_vx is not None:
        cmd_xy[0] = args_cli.cmd_vx
    if args_cli.cmd_vy is not None:
        cmd_xy[1] = args_cli.cmd_vy
    env_cfg.commanded_lin_vel_xy = tuple(cmd_xy)
    cmd_wz = float(getattr(env_cfg, "commanded_ang_vel_z", 0.0))
    if args_cli.cmd_wz is not None:
        cmd_wz = args_cli.cmd_wz
        env_cfg.commanded_ang_vel_z = cmd_wz
    # 评估时关掉箭头可视化（headless 下无意义，省开销）。
    if hasattr(env_cfg, "show_velocity_arrows"):
        env_cfg.show_velocity_arrows = False
    # 评估必须按全速命令测量：关闭训练用的命令速度课程。
    if hasattr(env_cfg, "command_curriculum_steps"):
        env_cfg.command_curriculum_steps = 0
    # 旧 checkpoint（对称 critic）兼容：按需关闭特权观测。
    if args_cli.no_privileged_critic and hasattr(env_cfg, "state_space"):
        env_cfg.state_space = 0

    # --- 摩擦覆盖（泛化测试）---
    if args_cli.friction is not None:
        f = float(args_cli.friction)
        for mat in (getattr(env_cfg.sim, "physics_material", None), getattr(env_cfg.terrain, "physics_material", None)):
            if mat is not None:
                mat.static_friction = f
                mat.dynamic_friction = f

    # --- 地形选择（与 play.py 一致）---
    if args_cli.terrain == "rough":
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = ROUGH_TERRAIN_CFG
        env_cfg.terrain.max_init_terrain_level = None
        env_cfg.terrain_curriculum_enabled = False
    elif args_cli.terrain == "curriculum":
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = CURRICULUM_TERRAIN_CFG
        env_cfg.terrain.max_init_terrain_level = None
        env_cfg.terrain_curriculum_enabled = False
    else:
        env_cfg.terrain.terrain_type = "plane"
        env_cfg.terrain.terrain_generator = None
        env_cfg.terrain_curriculum_enabled = False

    # 力矩上限：优先命令行，否则读 actuator cfg。
    torque_limit = args_cli.torque_limit
    if torque_limit is None:
        try:
            torque_limit = float(env_cfg.robot.actuators["main_joints"].effort_limit_sim)
        except (AttributeError, KeyError, TypeError):
            torque_limit = 12.0

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_arg = args_cli.checkpoint
    resume_path = os.path.abspath(checkpoint_arg) if os.path.isfile(checkpoint_arg) else get_checkpoint_path(
        log_root_path, args_cli.load_run, checkpoint_arg
    )
    print(f"[INFO] 加载 checkpoint: {resume_path}", flush=True)

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

    u = env.unwrapped
    device = u.device
    N = u.num_envs
    dt = float(u.step_dt)
    joint_ids = u._actuated_joint_ids
    joint_names = list(u._actuated_joint_names)
    nj = len(joint_ids)
    foot_ids = list(getattr(u, "_foot_body_ids", []) or [])
    foot_names = list(getattr(u, "_foot_body_names", []) or [])
    nf = len(foot_ids)
    robot = u._robot
    terrain_z = u._terrain.env_origins[:, 2]
    base_height_target = float(getattr(u.cfg, "base_height_target", 0.32))
    contact_h = (args_cli.foot_contact_height
                 if args_cli.foot_contact_height is not None
                 else float(getattr(u.cfg, "foot_contact_height_threshold", 0.07)))
    g = 9.81

    # --- 质量缩放（负载测试，可选；失败不致命）---
    total_mass = robot.data.default_mass.sum(dim=1).to(device)  # (N,)
    if abs(args_cli.mass_scale - 1.0) > 1e-6:
        try:
            view = robot.root_physx_view
            masses = view.get_masses()
            indices = torch.arange(masses.shape[0])
            view.set_masses(masses * args_cli.mass_scale, indices)
            total_mass = total_mass * args_cli.mass_scale
            print(f"[INFO] 质量已缩放 x{args_cli.mass_scale}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 质量缩放失败（按原质量评估）: {exc}", flush=True)

    push_interval = int(round(args_cli.push_interval_s / dt)) if args_cli.push_interval_s > 0 else 0

    cmd_xy_t = torch.tensor(env_cfg.commanded_lin_vel_xy, device=device)
    cmd_wz_t = torch.tensor(cmd_wz, device=device)

    # --- 全局逐步累加器（device 上，最后一次性同步）---
    z = lambda: torch.zeros((), device=device)  # noqa: E731
    g_alive = z()                       # 累计纳入统计的 env·step 数
    g_vx_abs, g_vx_sq = z(), z()
    g_vy_abs, g_vy_sq = z(), z()
    g_yaw_abs, g_yaw_sq = z(), z()
    g_speed = z()                       # 实际前向速度 vx 的累计
    g_base_h, g_upright, g_power = z(), z(), z()
    g_foot_slip = z()
    g_lin_err = z()                     # 水平线速度向量误差 ||cmd_xy - v_xy||（对齐 robot_lab lin_err）
    g_act_rate = z()                    # 相邻两步动作差 |a_t - a_{t-1}| 均值（对齐 robot_lab action_rate）
    g_slip_speed = z()                  # 触地足端水平滑移速度（线性 m/s，对齐 robot_lab foot_slip）
    tau_abs_sum = torch.zeros(nj, device=device)
    tau_sq_sum = torch.zeros(nj, device=device)
    tau_absmax = torch.zeros(nj, device=device)
    tau_near = torch.zeros(nj, device=device)
    foot_contact_steps = torch.zeros(max(nf, 1), device=device)
    foot_touchdown = torch.zeros(max(nf, 1), device=device)
    diag_inphase = z()                  # 对角对同相位（trot）计数
    lat_inphase = z()                   # 同端对同相位（bound）计数
    gait_pair_steps = z()

    # --- 每环境回合累加器 ---
    ep_steps = torch.zeros(N, dtype=torch.long, device=device)
    ep_energy = torch.zeros(N, device=device)
    ep_path = torch.zeros(N, device=device)
    ep_return = torch.zeros(N, device=device)
    episodes_done = torch.zeros(N, dtype=torch.long, device=device)  # 已完成回合计数（含 warmup）

    # --- 回合级记录（python list）---
    rec_time, rec_died, rec_return, rec_cot, rec_speed = [], [], [], [], []

    prev_xy = robot.data.root_pos_w[:, :2].clone()
    prev_contact = torch.zeros(N, max(nf, 1), dtype=torch.bool, device=device)
    prev_applied = None  # action_rate 的上一步动作
    delay_buf = []  # 动作延迟缓冲

    obs_dict, _ = env.reset()
    print(
        f"[INFO] 开始评估 | tag={args_cli.tag} terrain={args_cli.terrain} "
        f"cmd=({env_cfg.commanded_lin_vel_xy[0]:.2f},{env_cfg.commanded_lin_vel_xy[1]:.2f},{cmd_wz:.2f}) "
        f"friction={args_cli.friction} mass_x{args_cli.mass_scale} "
        f"obs_noise={args_cli.obs_noise} delay={args_cli.action_delay} "
        f"push={args_cli.push_interval_s}s/{args_cli.push_vel} | N={N} steps={args_cli.num_steps}",
        flush=True,
    )

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            # 观测噪声注入
            if args_cli.obs_noise > 0.0 and "policy" in obs_dict:
                obs_dict["policy"] = obs_dict["policy"] + torch.randn_like(obs_dict["policy"]) * args_cli.obs_noise

            action = policy(obs_dict)

            # 动作延迟：用 delay 步之前的动作
            if args_cli.action_delay > 0:
                delay_buf.append(action.clone())
                if len(delay_buf) > args_cli.action_delay:
                    applied = delay_buf.pop(0)
                else:
                    applied = torch.zeros_like(action)
            else:
                applied = action

            obs_dict, rewards, terminated, truncated, _ = env.step(applied)
            dones = terminated | truncated
            alive = (~dones).float()  # done 的 env 本步状态已被自动 reset，逐步指标排除

            # 动作平滑度（排除 done 步：reset 后动作清零会造成虚假跳变）
            if prev_applied is not None:
                g_act_rate += ((applied - prev_applied).abs().mean(dim=1) * alive).sum()
            prev_applied = applied.clone()

            # --- 读取本步状态 ---
            root_lin_b = torch.nan_to_num(robot.data.root_lin_vel_b)
            root_ang_b = torch.nan_to_num(robot.data.root_ang_vel_b)
            grav_b = robot.data.projected_gravity_b
            root_xy = robot.data.root_pos_w[:, :2]
            root_h = robot.data.root_pos_w[:, 2] - terrain_z
            tau = torch.nan_to_num(robot.data.applied_torque[:, joint_ids])   # (N, nj)
            qd = torch.nan_to_num(robot.data.joint_vel[:, joint_ids])         # (N, nj)

            # 回合累加（所有 env）
            ep_return += rewards
            ep_steps += 1
            power = (tau * qd).abs().sum(dim=1)        # (N,) 机械功率 W
            ep_energy += power * dt * alive
            delta = torch.norm(root_xy - prev_xy, dim=1)
            ep_path += delta * alive                    # done 步的 teleport 不计入

            # 逐步全局统计（排除 done 步）
            vx_err = (cmd_xy_t[0] - root_lin_b[:, 0]).abs()
            vy_err = (cmd_xy_t[1] - root_lin_b[:, 1]).abs()
            yaw_err = (cmd_wz_t - root_ang_b[:, 2]).abs()
            g_lin_err += (torch.norm(cmd_xy_t - root_lin_b[:, :2], dim=1) * alive).sum()
            upright = (grav_b[:, :2] ** 2).sum(dim=1)
            g_alive += alive.sum()
            g_vx_abs += (vx_err * alive).sum(); g_vx_sq += ((vx_err ** 2) * alive).sum()
            g_vy_abs += (vy_err * alive).sum(); g_vy_sq += ((vy_err ** 2) * alive).sum()
            g_yaw_abs += (yaw_err * alive).sum(); g_yaw_sq += ((yaw_err ** 2) * alive).sum()
            g_speed += (root_lin_b[:, 0] * alive).sum()
            g_base_h += (root_h * alive).sum()
            g_upright += (upright * alive).sum()
            g_power += (power * alive).sum()

            am = alive.unsqueeze(1)
            tau_abs_sum += (tau.abs() * am).sum(dim=0)
            tau_sq_sum += ((tau ** 2) * am).sum(dim=0)
            tau_absmax = torch.maximum(tau_absmax, (tau.abs() * am).amax(dim=0))
            tau_near += ((tau.abs() >= 0.9 * torque_limit).float() * am).sum(dim=0)

            # 步态：足端接触（用高度近似，本 USD 无接触传感器）
            if nf >= 4:
                foot_pos_w = torch.nan_to_num(robot.data.body_pos_w[:, foot_ids, :])
                foot_vel_w = torch.nan_to_num(robot.data.body_lin_vel_w[:, foot_ids, :])
                foot_h = foot_pos_w[..., 2] - terrain_z.unsqueeze(1)     # (N, nf)
                contact = foot_h < contact_h                             # (N, nf) bool
                cf = contact.float() * am
                foot_contact_steps += cf.sum(dim=0)
                # 触地事件：上一步未接触、本步接触（排除 done 步）
                td = contact & (~prev_contact)
                foot_touchdown += (td.float() * am).sum(dim=0)
                # 足端打滑：接触脚的水平速度平方和
                slip = (foot_vel_w[..., :2] ** 2).sum(-1) * contact.float()
                g_foot_slip += (slip.sum(dim=1) * alive).sum()
                # 线性版打滑速度 (m/s)，按接触步取均值（对齐 robot_lab foot_slip 口径）
                slip_speed = torch.norm(foot_vel_w[..., :2], dim=-1) * contact.float()
                g_slip_speed += (slip_speed.sum(dim=1) * alive).sum()
                # 相位（trot 检测）：对角对 (FL,RR)/(FR,RL) 同相位，同端对 (FL,FR)/(RL,RR) 同相位
                fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]
                diag = ((fl == rr).float() + (fr == rl).float()) * 0.5
                lat = ((fl == fr).float() + (rl == rr).float()) * 0.5
                diag_inphase += (diag * alive).sum()
                lat_inphase += (lat * alive).sum()
                gait_pair_steps += alive.sum()
                prev_contact = contact  # 下一步触地检测的基准；done 步触地已被 alive 掩掉

            # --- 推搡扰动 ---
            if push_interval > 0 and step > 0 and step % push_interval == 0:
                vel = robot.data.root_vel_w.clone()  # (N, 6) lin+ang
                ang = torch.rand(N, device=device) * 2 * math.pi
                vel[:, 0] += torch.cos(ang) * args_cli.push_vel
                vel[:, 1] += torch.sin(ang) * args_cli.push_vel
                robot.write_root_velocity_to_sim(vel)

            # --- 回合结束处理 ---
            if dones.any():
                d_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                episodes_done[d_idx] += 1
                # 只记录 warmup 之后的回合
                record_mask = dones & (episodes_done > args_cli.warmup_episodes)
                if record_mask.any():
                    r = record_mask.nonzero(as_tuple=False).squeeze(-1)
                    t_s = (ep_steps[r].float() * dt)
                    path = ep_path[r].clamp_min(1e-6)
                    cot = ep_energy[r] / (total_mass[r] * g * path)
                    spd = path / t_s.clamp_min(1e-6)
                    rec_time += t_s.cpu().tolist()
                    rec_died += terminated[r].cpu().tolist()
                    rec_return += ep_return[r].cpu().tolist()
                    rec_cot += cot.cpu().tolist()
                    rec_speed += spd.cpu().tolist()
                # 重置回合累加器
                ep_steps[d_idx] = 0
                ep_energy[d_idx] = 0.0
                ep_path[d_idx] = 0.0
                ep_return[d_idx] = 0.0

            prev_xy = root_xy.clone()

    # =========================================================================
    # 汇总
    # =========================================================================
    n_alive = max(float(g_alive.item()), 1.0)
    n_ep = len(rec_time)

    def _mean(x):
        return float(sum(x) / len(x)) if x else float("nan")

    metrics = {
        # 存活 / 成功
        "episodes": n_ep,
        "fall_rate": (_mean([1.0 if d else 0.0 for d in rec_died]) if n_ep else float("nan")),
        "success_rate": (_mean([0.0 if d else 1.0 for d in rec_died]) if n_ep else float("nan")),
        "mean_survival_s": _mean(rec_time),
        "mean_return": _mean(rec_return),
        # 速度跟踪
        "vx_mae": g_vx_abs.item() / n_alive,
        "vx_rmse": math.sqrt(g_vx_sq.item() / n_alive),
        "vy_mae": g_vy_abs.item() / n_alive,
        "vy_rmse": math.sqrt(g_vy_sq.item() / n_alive),
        "yaw_mae": g_yaw_abs.item() / n_alive,
        "yaw_rmse": math.sqrt(g_yaw_sq.item() / n_alive),
        "mean_speed_vx": g_speed.item() / n_alive,
        "mean_episode_speed": _mean(rec_speed),
        # 姿态
        "mean_base_height": g_base_h.item() / n_alive,
        "base_height_target": base_height_target,
        "mean_upright_err": g_upright.item() / n_alive,
        # 能耗
        "mean_power_w": g_power.item() / n_alive,
        "mean_cot": _mean(rec_cot),
        # 足端打滑
        "mean_foot_slip": g_foot_slip.item() / n_alive,
        # --- 与 robot_lab eval.py 对齐口径的指标 ---
        "lin_err": g_lin_err.item() / n_alive,
        "mean_action_rate": g_act_rate.item() / max(n_alive - 1.0, 1.0),
        "mean_foot_slip_speed": (
            g_slip_speed.item() / max(float(foot_contact_steps.sum().item()), 1.0) if nf >= 4 else float("nan")
        ),
    }
    # 步态相位
    if gait_pair_steps.item() > 0:
        metrics["diag_inphase_rate"] = diag_inphase.item() / gait_pair_steps.item()
        metrics["lateral_inphase_rate"] = lat_inphase.item() / gait_pair_steps.item()

    # 每关节力矩
    tau_mean = (tau_abs_sum / n_alive).cpu().tolist()
    tau_rms = (tau_sq_sum / n_alive).sqrt().cpu().tolist()
    tau_max = tau_absmax.cpu().tolist()
    tau_nearf = (tau_near / n_alive).cpu().tolist()
    per_joint = []
    for i, name in enumerate(joint_names):
        per_joint.append({
            "joint": name, "abs_mean": tau_mean[i], "rms": tau_rms[i],
            "abs_max": tau_max[i], "near_limit_frac": tau_nearf[i],
        })
    metrics["torque_limit"] = torque_limit
    metrics["torque_abs_mean_all"] = sum(tau_mean) / len(tau_mean) if tau_mean else float("nan")
    metrics["torque_abs_max_all"] = max(tau_max) if tau_max else float("nan")
    metrics["torque_near_limit_frac_all"] = sum(tau_nearf) / len(tau_nearf) if tau_nearf else float("nan")
    metrics["per_joint_torque"] = per_joint

    # 每脚步态
    if nf >= 1 and n_alive > 0:
        duty = (foot_contact_steps / n_alive).cpu().tolist()
        # touchdown 频率 = 触地事件数 / 总接触统计时间。总时间 ≈ n_alive*dt / N（平均每 env），
        # 用 touchdown / (n_alive*dt) * N 估计单脚平均触地频率。
        total_time = n_alive * dt
        td_hz = [(foot_touchdown[i].item() / total_time) for i in range(nf)]
        metrics["per_foot"] = [
            {"foot": (foot_names[i] if i < len(foot_names) else f"foot{i}"),
             "duty_factor": duty[i], "touchdown_hz": td_hz[i]}
            for i in range(nf)
        ]
        metrics["mean_duty_factor"] = sum(duty[:nf]) / nf
        metrics["mean_touchdown_hz"] = sum(td_hz) / nf

    condition = {
        "tag": args_cli.tag,
        "checkpoint": resume_path,
        "task": args_cli.task,
        "num_envs": N,
        "num_steps": args_cli.num_steps,
        "terrain": args_cli.terrain,
        "cmd_vx": env_cfg.commanded_lin_vel_xy[0],
        "cmd_vy": env_cfg.commanded_lin_vel_xy[1],
        "cmd_wz": cmd_wz,
        "friction": args_cli.friction,
        "mass_scale": args_cli.mass_scale,
        "obs_noise": args_cli.obs_noise,
        "action_delay": args_cli.action_delay,
        "push_interval_s": args_cli.push_interval_s,
        "push_vel": args_cli.push_vel if args_cli.push_interval_s > 0 else 0.0,
    }
    result = {"condition": condition, "metrics": metrics}

    # --- 控制台输出 ---
    _print_report(result)

    # --- 保存 JSON ---
    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(resume_path), "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args_cli.tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    md_path = os.path.join(out_dir, f"{args_cli.tag}.md")
    with open(md_path, "w") as f:
        f.write(_format_unified_md(result))
    print(f"\n[INFO] 评估结果已保存: {out_path}", flush=True)
    print(f"[INFO] 统一格式报告（与 robot_lab eval.py 同口径，可直接并排对比）: {md_path}", flush=True)
    print(f"EVAL_COMPLETED tag={args_cli.tag} json={out_path}", flush=True)
    env.close()


# 与 robot_lab scripts/reinforcement_learning/rsl_rl/eval.py 输出对齐的指标说明。
METRIC_LEGEND = """
## 指标说明（怎样算好）

| 指标 | 含义 | 怎样算好 |
|---|---|---|
| cmd (vx, vy, wz) | 固定速度命令：前向 [m/s]、侧向 [m/s]、偏航角速度 [rad/s] | — |
| lin_err | 水平线速度跟踪误差的均值 [m/s] | 越小越好：<0.15 优秀，0.15~0.3 一般，>0.3 基本没跟上命令 |
| ang_err | 偏航角速度跟踪误差的均值 [rad/s] | 越小越好，<0.2 良好 |
| fall_rate | 已结束的 episode 中非超时终止（摔倒/机身过低/翻倒）的占比 | 越低越好，好策略接近 0% |
| episodes | 统计窗口内结束的 episode 数（摔倒+超时） | fall_rate 的样本量；为 0 时 fall_rate 无意义 |
| CoT | 运输成本 = 关节能耗 / (重量 x 距离)，无量纲 | 越小越省电：四足正常行走约 0.3~0.8，>2 说明大量能量浪费 |
| tau_mean | 关节力矩绝对值的均值 [Nm] | 越小越省电 |
| tau_peak | 关节力矩绝对值的峰值 [Nm] | 越低越安全；贴近电机上限说明真机部署有风险 |
| action_rate | 相邻两步动作差的均值 | 越小越平滑；过大意味着真机上高频抖动、电机发热 |
| foot_slip | 足端触地期间的水平滑移速度 [m/s] | 越小越好，<0.1 说明落脚干净不打滑 |

判读顺序：先看 fall_rate 和 lin_err（能不能走、跟不跟得上），再看 CoT / action_rate / foot_slip
（走得好不好），tau_peak 用于评估真机部署的安全裕度。跨项目对比时确保命令、地形、时长一致。
"""


def _format_unified_md(result) -> str:
    """输出与 robot_lab eval.py 相同表头的 markdown 报告，方便两个项目并排对比。"""
    c, m = result["condition"], result["metrics"]
    slip = m.get("mean_foot_slip_speed", float("nan"))
    fall = m.get("fall_rate", float("nan"))
    cot = m.get("mean_cot", float("nan"))
    fmt = lambda v, spec=".3f": (format(v, spec) if v == v else "-")  # noqa: E731  NaN → "-"
    lines = [
        f"# Evaluation: {c['checkpoint']}",
        "",
        f"- tag: {c['tag']}, terrain: {c['terrain']}, num_envs: {c['num_envs']}, num_steps: {c['num_steps']}",
        f"- friction: {c['friction']}, mass_scale: {c['mass_scale']}, obs_noise: {c['obs_noise']},"
        f" action_delay: {c['action_delay']}, push: {c['push_interval_s']}s/{c['push_vel']}",
        "",
        "| cmd (vx, vy, wz) | lin_err [m/s] | ang_err [rad/s] | fall_rate | episodes | CoT |"
        " tau_mean [Nm] | tau_peak [Nm] | action_rate | foot_slip [m/s] |",
        "|---" * 10 + "|",
        f"| ({c['cmd_vx']:.1f}, {c['cmd_vy']:.1f}, {c['cmd_wz']:.1f}) | {fmt(m['lin_err'])} |"
        f" {fmt(m['yaw_mae'])} | {fmt(fall * 100, '.1f')}% | {m['episodes']} | {fmt(cot)} |"
        f" {fmt(m['torque_abs_mean_all'], '.2f')} | {fmt(m['torque_abs_max_all'], '.2f')} |"
        f" {fmt(m['mean_action_rate'])} | {fmt(slip)} |",
        METRIC_LEGEND,
    ]
    return "\n".join(lines)


def _print_report(result):
    c, m = result["condition"], result["metrics"]
    line = "=" * 64
    print(f"\n{line}\n 评估报告  tag={c['tag']}\n{line}", flush=True)
    print(f"  条件: terrain={c['terrain']} cmd=({c['cmd_vx']:.2f},{c['cmd_vy']:.2f},{c['cmd_wz']:.2f}) "
          f"friction={c['friction']} mass_x{c['mass_scale']} noise={c['obs_noise']} "
          f"delay={c['action_delay']} push={c['push_interval_s']}s", flush=True)
    print(f"\n  [存活]   回合数={m['episodes']}  成功率={m['success_rate']*100:5.1f}%  "
          f"跌倒率={m['fall_rate']*100:5.1f}%  平均存活={m['mean_survival_s']:.2f}s  "
          f"平均回报={m['mean_return']:.1f}", flush=True)
    print(f"  [跟踪]   vx_MAE={m['vx_mae']:.3f}  vx_RMSE={m['vx_rmse']:.3f}  "
          f"vy_MAE={m['vy_mae']:.3f}  yaw_MAE={m['yaw_mae']:.3f}  "
          f"实际vx均值={m['mean_speed_vx']:.3f} m/s", flush=True)
    print(f"  [姿态]   base高度均值={m['mean_base_height']:.3f}（目标 {m['base_height_target']:.2f}）  "
          f"倾斜误差={m['mean_upright_err']:.4f}", flush=True)
    print(f"  [能耗]   平均功率={m['mean_power_w']:.1f} W  CoT={m['mean_cot']:.3f}", flush=True)
    if "mean_duty_factor" in m:
        print(f"  [步态]   平均占空比={m['mean_duty_factor']:.2f}  平均触地频率={m['mean_touchdown_hz']:.2f} Hz  "
              f"对角同相位={m.get('diag_inphase_rate', float('nan')):.2f}  "
              f"同端同相位={m.get('lateral_inphase_rate', float('nan')):.2f}", flush=True)
    print(f"  [打滑]   foot_slip={m['mean_foot_slip']:.4f}", flush=True)
    print(f"\n  [力矩]   上限={m['torque_limit']:.1f} N·m  |τ|均值={m['torque_abs_mean_all']:.2f}  "
          f"|τ|峰值={m['torque_abs_max_all']:.2f}  近上限占比={m['torque_near_limit_frac_all']*100:.1f}%", flush=True)
    hdr = f"    {'joint':<20}{'|τ|mean':>9}{'rms':>8}{'|τ|max':>9}{'近上限%':>9}"
    print(hdr, flush=True)
    for j in m["per_joint_torque"]:
        flag = "  <==" if j["abs_max"] >= m["torque_limit"] else ("  ~" if j["abs_max"] >= 0.8 * m["torque_limit"] else "")
        print(f"    {j['joint']:<20}{j['abs_mean']:>9.2f}{j['rms']:>8.2f}{j['abs_max']:>9.2f}"
              f"{j['near_limit_frac']*100:>8.1f}%{flag}", flush=True)
    print(line, flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
