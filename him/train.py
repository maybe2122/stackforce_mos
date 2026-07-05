# Train the HIM (Hybrid Internal Model) policy on the mos2026_2 closed-chain
# quadruped in Isaac Lab.
#
# Run with the shared Isaac Lab env (the one that has isaaclab + torch):
#   GUI:      env_isaaclab/bin/python him/train.py --num_envs 256
#   headless: env_isaaclab/bin/python him/train.py --headless --num_envs 4096
#
# HIM only changes the policy/critic/algorithm and the observation layout; the
# robot, actuators, rewards, terminations and terrain are the base Mos env's.

import argparse
import os
import sys
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
# make `him_rl` + sibling modules and the (possibly un-installed) mos_one
# package importable without relying on a pip install.
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "source", "mos_one"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train HIM on mos2026_2 (Isaac Lab).")
parser.add_argument("--num_envs", type=int, default=8192, help="Number of parallel envs.")
parser.add_argument("--max_iterations", type=int, default=200000, help="Policy update iterations.")
parser.add_argument("--num_steps_per_env", type=int, default=24, help="Rollout length per iteration.")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--experiment_name", type=str, default="mos2026_2_him")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument(
    "--terrain",
    type=str,
    default="flat",
    choices=["flat", "rough", "curriculum"],
    help="Ground: 'flat' (plane), 'rough' (heightfield), or 'curriculum' (flat->rough+stairs).",
)
# --- 域随机化 / sim2real（与 scripts/rsl_rl/train.py 同款；默认关闭）---
# 注意：HIM 自带 actor 观测噪声（him_env_cfg.him_actor_noise，默认开、分块均匀噪声），
# 基类的 --obs_noise_std 路径被 HIM 的 _get_observations 覆写绕开，故这里不提供该参数。
parser.add_argument("--domain_rand", default=True, action="store_true",
                    help="启用域随机化事件（摩擦/质量/增益/关节零位偏置/周期推搡）。默认关闭。")
parser.add_argument("--no_actor_noise", action="store_true",
                    help="关闭 HIM 的 actor 观测噪声（him_actor_noise，默认开启）。")
parser.add_argument("--payload", type=float, default=5.0,
                    help="训练时给基座额外附加的固定负载质量（kg），模拟背负载荷。"
                         "默认 %(default)s；设为 0 关闭。开 --domain_rand 时并入质量随机化区间。")
# --- SwanLab 实验跟踪（镜像 HIM runner 的 TensorBoard 标量）---
parser.add_argument("--no_swanlab", action="store_true",
                    help="关闭 SwanLab 实验跟踪（默认开启；未安装 swanlab 时自动跳过）。")
parser.add_argument("--swanlab_project", type=str, default="mos_one-mos",
                    help="SwanLab 项目名（与 PPO 训练同项目便于对比）。")
parser.add_argument("--swanlab_mode", type=str, default="cloud",
                    choices=["cloud", "local", "offline", "disabled"],
                    help="SwanLab 运行模式：cloud / local / offline / disabled。")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# launch Isaac Sim (GUI unless --headless was passed)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- imports that require the running app (carb/isaaclab) ----
import torch  # noqa: E402

import isaaclab.envs.mdp as mdp  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from mos_one.tasks.direct.mos2026_2_closed_usd.mos2026_2_closed_usd_env_cfg import (  # noqa: E402
    CURRICULUM_TERRAIN_CFG,
    ROUGH_TERRAIN_CFG,
    EventCfg,
)
from him_env_cfg import Mos20262ClosedUsdHIMEnvCfg  # noqa: E402
from him_env import Mos20262ClosedUsdHIMEnv  # noqa: E402
from adapter import HIMVecEnvAdapter  # noqa: E402
from him_cfg import get_him_train_cfg  # noqa: E402
from him_rl.runners.him_on_policy_runner import HIMOnPolicyRunner  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@configclass
class PayloadOnlyEventCfg:
    """仅含基座负载项的事件容器（未开 --domain_rand 时挂载用）。

    DirectRLEnv 在 cfg.events 非空时自动建 EventManager；负载项由
    `_make_payload_term` 在 main() 里按 --payload 动态填入。
    """

    pass


def _make_payload_term(payload_kg: float) -> EventTerm:
    """固定负载：startup 时给 base 刚体一次性 +payload_kg（上下限相等=确定值）。

    走 mdp.randomize_rigid_body_mass（operation="add"），惯量按质量比例自动重算。
    注意该函数每次调用都先把质量重置回 default 再施加，同一刚体上多个质量事件
    **不叠加**（后者覆盖前者）——所以 --domain_rand 开启时不要用本项，而是把
    负载平移进 EventCfg.add_base_mass 的随机区间（见 main()）。
    """
    return EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (payload_kg, payload_kg),
            "operation": "add",
        },
    )


def _init_swanlab(args, env_cfg, train_cfg, log_dir):
    """启动 SwanLab 并镜像 HIM runner 的 TensorBoard 标量。

    HIMOnPolicyRunner 在 learn() 里建 SummaryWriter；swanlab 的 tensorboard 同步
    是给 SummaryWriter 类方法打补丁，所以只要在 runner.learn() 之前调用即可生效，
    无需改 him_rl。任何失败只告警、返回 None，不影响训练。
    """
    if args.no_swanlab or args.swanlab_mode == "disabled":
        return None
    try:
        import swanlab
    except ImportError:
        print("[SwanLab] 未安装，跳过实验跟踪。安装：pip install swanlab", flush=True)
        return None
    try:
        algo = train_cfg.get("algorithm", {})
        config = {
            "algo": "HIM",
            "task": "mos2026_2_him",
            "num_envs": args.num_envs,
            "max_iterations": args.max_iterations,
            "num_steps_per_env": args.num_steps_per_env,
            "seed": args.seed,
            "terrain": args.terrain,
            "domain_rand": bool(args.domain_rand),
            "payload": args.payload,
            "him_actor_noise": not args.no_actor_noise,
            # HIM 观测/网络
            "num_actor_obs": 270,
            "num_privileged_obs": 51,
            "num_one_step_obs": 45,
            "actor_hidden_dims": train_cfg.get("policy", {}).get("actor_hidden_dims"),
            # PPO 超参
            "learning_rate": algo.get("learning_rate"),
            "gamma": algo.get("gamma"),
            "entropy_coef": algo.get("entropy_coef"),
            # 环境/奖励（与 PPO 训练同字段，便于对比）
            "commanded_lin_vel_xy": getattr(env_cfg, "commanded_lin_vel_xy", None),
            "commanded_ang_vel_z": getattr(env_cfg, "commanded_ang_vel_z", None),
            "base_height_target": getattr(env_cfg, "base_height_target", None),
            "reward_scales": getattr(env_cfg, "reward_scales", None),
        }
        try:
            actuator = env_cfg.robot.actuators["main_joints"]
            config["effort_limit_sim"] = getattr(actuator, "effort_limit_sim", None)
            config["velocity_limit_sim"] = getattr(actuator, "velocity_limit_sim", None)
        except (AttributeError, KeyError, TypeError):
            pass
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=os.path.basename(log_dir),
            config=config,
            logdir=log_dir,
            mode=args.swanlab_mode,
        )
        for fn_name in ("sync_tensorboard_torch", "sync_tensorboardX", "sync_tensorboard"):
            fn = getattr(swanlab, fn_name, None)
            if callable(fn):
                fn()
                break
        print(f"[SwanLab] 已启用 project={args.swanlab_project} "
              f"exp={os.path.basename(log_dir)} mode={args.swanlab_mode}", flush=True)
        return swanlab
    except Exception as exc:  # noqa: BLE001
        print(f"[SwanLab] 初始化失败，继续训练（仅 TensorBoard）：{exc}", flush=True)
        return None


def main():
    device = "cuda:0"

    env_cfg = Mos20262ClosedUsdHIMEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = device

    # terrain selection (mirrors scripts/rsl_rl/train.py)
    if args.terrain == "rough":
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = ROUGH_TERRAIN_CFG
        env_cfg.terrain.max_init_terrain_level = None
        env_cfg.terrain_curriculum_enabled = False
    elif args.terrain == "curriculum":
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = CURRICULUM_TERRAIN_CFG
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain_curriculum_enabled = True
    else:
        env_cfg.terrain.terrain_type = "plane"
        env_cfg.terrain.terrain_generator = None
        env_cfg.terrain_curriculum_enabled = False

    # 域随机化（sim2real）。默认关闭；--domain_rand 时挂载 EventCfg，DirectRLEnv
    # 检测到 cfg.events 非空会自动建 EventManager（与 scripts/rsl_rl/train.py 同机制）。
    if args.domain_rand:
        env_cfg.events = EventCfg()
        print("[HIM-Mos] 域随机化已启用（摩擦/质量/增益/关节零位/推搡）。"
              "首次启用请先小 env 冒烟。")
    else:
        env_cfg.events = None

    # 基座固定负载（kg）。randomize_rigid_body_mass 每次调用都基于 default 质量
    # 重算，同一刚体上多个质量事件互相覆盖，所以分两种挂法：
    #   - domain_rand 开：把负载平移进已有 add_base_mass 的随机区间（单事件完成
    #     "payload + 随机 ±kg"）；
    #   - domain_rand 关：挂仅含固定负载项的事件容器（EventManager 解析
    #     events.__dict__，运行时 setattr 即可生效）。
    if args.payload > 0.0:
        if env_cfg.events is not None:
            lo, hi = env_cfg.events.add_base_mass.params["mass_distribution_params"]
            env_cfg.events.add_base_mass.params["mass_distribution_params"] = (
                lo + args.payload, hi + args.payload)
            print(f"[HIM-Mos] 基座负载已并入质量随机化：base +({lo + args.payload:.2f}, "
                  f"{hi + args.payload:.2f}) kg。")
        else:
            env_cfg.events = PayloadOnlyEventCfg()
            env_cfg.events.add_payload = _make_payload_term(args.payload)
            print(f"[HIM-Mos] 基座负载已启用：base +{args.payload:.2f} kg（startup 一次性附加）。")
    elif args.payload < 0.0:
        raise ValueError(f"--payload 不能为负数，收到 {args.payload}")

    # HIM actor 观测噪声（默认开）：--no_actor_noise 关闭，训干净 obs 基线用。
    env_cfg.him_actor_noise = not args.no_actor_noise
    if args.no_actor_noise:
        print("[HIM-Mos] HIM actor 观测噪声已关闭（him_actor_noise=False）。")

    env = Mos20262ClosedUsdHIMEnv(cfg=env_cfg)
    vec_env = HIMVecEnvAdapter(env)

    log_root = os.path.join(_THIS_DIR, "logs", args.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + args.run_name)
    os.makedirs(log_dir, exist_ok=True)

    train_cfg = get_him_train_cfg(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        num_steps_per_env=args.num_steps_per_env,
        max_iterations=args.max_iterations,
    )

    runner = HIMOnPolicyRunner(vec_env, train_cfg, log_dir=log_dir, device=device)
    print(f"[HIM-Mos] obs={vec_env.num_obs} priv={vec_env.num_privileged_obs} "
          f"one_step={vec_env.num_one_step_obs} actions={vec_env.num_actions} "
          f"num_envs={vec_env.num_envs} terrain={args.terrain} device={device}")

    # 启动 SwanLab（在 runner.learn 建 TensorBoard writer 之前），失败不影响训练。
    swan = _init_swanlab(args, env_cfg, train_cfg, log_dir)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    if swan is not None:
        try:
            swan.finish()
        except Exception:  # noqa: BLE001
            pass
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
