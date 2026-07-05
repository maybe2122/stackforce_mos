import argparse
import inspect
import os
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a MosOne closed-chain USD task with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=8192,
                    help="并行环境数。默认 16384（=4096×4，按 RTX 5090 32GB 估算）；小卡请显式调低。")
parser.add_argument("--task", type=str, default="MosOne-Mos20262ClosedUsd-ClosedUsd-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
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
parser.add_argument(
    "--max_gpu_mem",
    type=float,
    default=32.0,
    help=(
        "Target GPU memory budget in GB. The default PhysX broad-phase caps in "
        "the env cfg are sized for a 32 GB card; pass a smaller value (e.g. "
        "--max_gpu_mem 16, --max_gpu_mem 8) to scale every gpu_* capacity down "
        "proportionally so the simulator fits on smaller GPUs. Does NOT change "
        "num_envs — pair it with --num_envs if you also need fewer envs."
    ),
)
# --- 域随机化 / sim2real（默认关闭；开启后需先小 env 冒烟验证）---
parser.add_argument("--domain_rand", action="store_true",
                    help="启用域随机化事件（摩擦/质量/增益/关节零位偏置/周期推搡）。默认关闭。")
parser.add_argument("--obs_noise_std", type=float, default=0.0,
                    help="观测高斯噪声标准差（加在 45 维 obs 上模拟传感器噪声）。默认 0=关闭。")
# --- SwanLab 实验跟踪（通过 TensorBoard 同步镜像 rsl_rl 的所有标量）---
parser.add_argument("--no_swanlab", action="store_true",
                    help="关闭 SwanLab 实验跟踪（默认开启；未安装 swanlab 时自动跳过）。")
parser.add_argument("--swanlab_project", type=str, default="mos_one-mos",
                    help="SwanLab 项目名。")
parser.add_argument("--swanlab_mode", type=str, default="cloud",
                    choices=["cloud", "local", "offline", "disabled"],
                    help="SwanLab 运行模式：cloud 上传云端 / local 本地看板 / offline 离线缓存 / disabled 关闭。")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
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
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import mos_one.tasks  # noqa: F401
from mos_one.tasks.direct.mos2026_2_closed_usd.mos2026_2_closed_usd_env_cfg import (
    CURRICULUM_TERRAIN_CFG,
    ROUGH_TERRAIN_CFG,
    EventCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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


class LegacyRslRlVecEnvWrapper(VecEnv):
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
    """转换 agent cfg 到当前 rsl_rl 版本的格式。

    has_critic_obs：env 是否输出独立的 "critic" 特权观测组（非对称 critic，
    见 env cfg 的 state_space）。True 时 obs_groups 的 critic 指向 "critic"
    组，False 时退回与 actor 共用 "policy" 组（对称 critic / 旧行为）。
    """
    critic_group = ["critic"] if has_critic_obs else ["policy"]
    data = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    allowed_policy_keys = {
        "actor_hidden_dims",
        "critic_hidden_dims",
        "activation",
        "init_noise_std",
        "clip_actions",
        "actor_obs_normalization",
        "critic_obs_normalization",
    }
    allowed_algorithm_keys = {
        "num_learning_epochs",
        "num_mini_batches",
        "clip_param",
        "gamma",
        "lam",
        "value_loss_coef",
        "entropy_coef",
        "learning_rate",
        "max_grad_norm",
        "use_clipped_value_loss",
        "schedule",
        "desired_kl",
        "use_spo",
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


# PhysX broad-phase caps that consume GPU memory roughly linearly. Defaults
# in the env cfg are sized for a 32 GB card; scaling these is what lets the
# same task fit on a smaller GPU. Floors below are the smallest values PhysX
# tolerates without throwing capacity-exceeded errors during contact-rich steps.
_PHYSX_GPU_CAP_FIELDS = {
    "gpu_found_lost_pairs_capacity":            2**18,   # 262144 floor
    "gpu_found_lost_aggregate_pairs_capacity":  2**20,   # 1048576 floor
    "gpu_total_aggregate_pairs_capacity":       2**18,   # 262144 floor
    "gpu_max_rigid_contact_count":              2**18,   # 262144 floor
    "gpu_max_rigid_patch_count":                2**16,   # 65536 floor
}


def _scale_physx_gpu_caps(env_cfg, max_gpu_mem_gb: float) -> None:
    # 32 GB is the baseline the env cfg was authored for; scale linearly below
    # that. Above 32 GB we leave the configured caps alone — they're already
    # generous and growing them further wastes VRAM the policy could use.
    if max_gpu_mem_gb is None or max_gpu_mem_gb >= 32.0:
        return
    physx = getattr(env_cfg.sim, "physx", None)
    if physx is None:
        return
    scale = max(max_gpu_mem_gb / 32.0, 1.0 / 32.0)  # never scale below 1/32× of authored values
    scaled = {}
    for field, floor in _PHYSX_GPU_CAP_FIELDS.items():
        current = getattr(physx, field, None)
        if current is None:
            continue
        new_value = max(int(current * scale), floor)
        setattr(physx, field, new_value)
        scaled[field] = (current, new_value)
    if scaled:
        print(
            f"[MosOne] Scaling PhysX GPU caps for {max_gpu_mem_gb:.1f} GB budget "
            f"(scale={scale:.3f}):",
            flush=True,
        )
        for field, (old, new) in scaled.items():
            print(f"  {field}: {old} -> {new}", flush=True)


def _init_swanlab(args_cli, agent_cfg, env_cfg, log_dir):
    """启动 SwanLab 并把 rsl_rl 的 TensorBoard 标量自动镜像过去。

    用 SwanLab 的 tensorboard 同步（patch SummaryWriter 的 add_scalar 等方法），
    所以无需改动 rsl_rl 的 OnPolicyRunner —— 它照常写 TensorBoard，标量会同时
    进入 SwanLab。必须在 runner 调用 add_scalar 之前调用（这里在建 runner 前调）。

    任何失败都只打印告警、返回 None，绝不影响训练本身。
    """
    if args_cli.no_swanlab or args_cli.swanlab_mode == "disabled":
        return None
    try:
        import swanlab
    except ImportError:
        print("[SwanLab] 未安装，跳过实验跟踪。安装：pip install swanlab", flush=True)
        return None
    try:
        config = {
            "task": args_cli.task,
            "num_envs": env_cfg.scene.num_envs,
            "max_iterations": agent_cfg.max_iterations,
            "seed": getattr(agent_cfg, "seed", None),
            "terrain": args_cli.terrain,
            "max_gpu_mem": args_cli.max_gpu_mem,
            # 关键环境/奖励超参（便于在 SwanLab 里跨实验对比）
            "commanded_lin_vel_xy": getattr(env_cfg, "commanded_lin_vel_xy", None),
            "commanded_ang_vel_z": getattr(env_cfg, "commanded_ang_vel_z", None),
            "base_height_target": getattr(env_cfg, "base_height_target", None),
            "action_scale": getattr(env_cfg, "action_scale", None),
            "tracking_sigma": getattr(env_cfg, "tracking_sigma", None),
            "reward_scales": getattr(env_cfg, "reward_scales", None),
        }
        try:
            actuator = env_cfg.robot.actuators["main_joints"]
            config["effort_limit_sim"] = getattr(actuator, "effort_limit_sim", None)
            config["velocity_limit_sim"] = getattr(actuator, "velocity_limit_sim", None)
        except (AttributeError, KeyError, TypeError):
            pass
        swanlab.init(
            project=args_cli.swanlab_project,
            experiment_name=os.path.basename(log_dir),
            config=config,
            logdir=log_dir,
            mode=args_cli.swanlab_mode,
        )
        # 镜像 rsl_rl 的 TensorBoard 标量（不同 swanlab 版本函数名略有差异）
        synced = False
        for fn_name in ("sync_tensorboard_torch", "sync_tensorboardX", "sync_tensorboard"):
            fn = getattr(swanlab, fn_name, None)
            if callable(fn):
                fn()
                synced = True
                break
        if not synced:
            print("[SwanLab] 未找到 tensorboard 同步函数，标量可能不会镜像（升级 swanlab 可修复）。", flush=True)
        print(
            f"[SwanLab] 已启用 project={args_cli.swanlab_project} "
            f"exp={os.path.basename(log_dir)} mode={args_cli.swanlab_mode}",
            flush=True,
        )
        return swanlab
    except Exception as exc:  # noqa: BLE001
        print(f"[SwanLab] 初始化失败，继续训练（仅 TensorBoard）：{exc}", flush=True)
        return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    _scale_physx_gpu_caps(env_cfg, args_cli.max_gpu_mem)

    if args_cli.terrain == "rough":
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = ROUGH_TERRAIN_CFG
        env_cfg.terrain.max_init_terrain_level = None
        env_cfg.terrain_curriculum_enabled = False
    elif args_cli.terrain == "curriculum":
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = CURRICULUM_TERRAIN_CFG
        # Start every env on the easiest row (flat) so the curriculum begins
        # at the bottom and walks up only after the policy succeeds.
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain_curriculum_enabled = True
    else:
        env_cfg.terrain.terrain_type = "plane"
        env_cfg.terrain.terrain_generator = None
        env_cfg.terrain_curriculum_enabled = False

    # 域随机化 / 观测噪声（sim2real）。默认关闭；--domain_rand 时挂载 EventCfg，
    # DirectRLEnv 检测到 cfg.events 非空会自动建 EventManager。eval.py 不走这里，
    # 保持「干净测量」。
    if args_cli.domain_rand:
        env_cfg.events = EventCfg()
        print("[INFO] 域随机化已启用（摩擦/质量/增益/关节零位/推搡）。"
              "首次启用请先小 env 冒烟，并核对 USD body/joint 名是否匹配 EventCfg 正则。")
    else:
        env_cfg.events = None
    env_cfg.obs_noise_std = args_cli.obs_noise_std
    if args_cli.obs_noise_std > 0.0:
        print(f"[INFO] 观测高斯噪声已启用：std={args_cli.obs_noise_std}")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    env_cfg.log_dir = log_dir

    # 启动 SwanLab（在建 runner / 写 TensorBoard 之前），失败不影响训练。
    swan = _init_swanlab(args_cli, agent_cfg, env_cfg, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped_env = LegacyRslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(
        wrapped_env,
        to_compatible_rsl_rl_cfg(agent_cfg, has_critic_obs=wrapped_env.num_privileged_obs is not None),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    checkpoint_path = os.path.join(log_dir, "model_final.pt")
    runner.save(checkpoint_path)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds", flush=True)
    print(f"TRAINING_COMPLETED checkpoint={checkpoint_path}", flush=True)
    if swan is not None:
        try:
            swan.finish()
        except Exception:  # noqa: BLE001
            pass
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
