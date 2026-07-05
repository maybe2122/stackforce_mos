"""把训练好的 rsl_rl PPO actor 部署到 mos2026_2 机器人的 MuJoCo MJCF 上。

MJCF 需先由 `tools/asset/usd_to_mjcf.py` 生成。本脚本做的事：
  - 加载 .pt checkpoint，取出 actor 权重，按 rsl_rl RslRlPpoActorCriticCfg
    重建 [256, 256, 128]-ELU 的 MLP（层结构从权重形状自动推导）。
  - 复刻 Isaac Lab 的 45 维观测：root_lin_vel_b、root_ang_vel_b、
    projected_gravity_b、joint_pos_rel、joint_vel、prev_actions。
  - 复刻 Isaac Lab 的动作管线：clip(action_clip=1.5) →
    processed = action_scale * action + default_pos → MuJoCo position
    执行器（kp=25 / kv=0.5）。
  - 50 Hz 策略控制跑在高频 MuJoCo 物理上（软闭链需要 1000 Hz，见 SIM_HZ 注释）。
  - 默认开交互 viewer；--headless 静默跑 N 步做定量验证（输出摔倒/里程/步态报告）。
  - 支持部署条件仿真：--lin-vel-source/--imu-source 复刻真机受限观测、
    --kp/--kv 模拟部署增益、--settle 站姿保持、--action-lpf 动作低通、--slowmo 慢放。

示例：
    # 裸跑——自动挑 logs/rsl_rl 下最新的 model_*.pt
    python deploy/mujoco/play_mujoco.py

    python deploy/mujoco/play_mujoco.py \
        --checkpoint logs/rsl_rl/mos2026_2_closed_usd/2026-05-14_11-23-51/model_600.pt

    python deploy/mujoco/play_mujoco.py --headless --duration 5 \
        --checkpoint logs/rsl_rl/mos2026_2_closed_usd/2026-05-14_11-23-51/model_600.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = REPO_ROOT / "deploy/mujoco/assets/mos2026_2.xml"
LOG_ROOT = REPO_ROOT / "logs/rsl_rl"


def find_latest_checkpoint() -> Path | None:
    """挑 logs/rsl_rl 下最新的 model_*.pt（先按 run 目录 mtime，再按迭代号）。"""
    candidates = list(LOG_ROOT.rglob("model_*.pt"))
    if not candidates:
        return None

    def sort_key(p: Path):
        try:
            it = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            it = -1
        return (p.parent.stat().st_mtime, it)

    return max(candidates, key=sort_key)

# Isaac Lab 训练环境配置的镜像——抄在这里是为了让部署脚本自包含
# （直接 import 环境配置会把整个 Isaac Lab 栈拖进来）。
ACTUATED_JOINTS = [
    "fl_hip", "fr_hip", "rl_hip", "rr_hip",
    "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
    "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a",
]
# 必须与 Mos20262ClosedUsdEnvCfg.init_state.joint_pos（env_cfg.py）一致。
# 前后髋从 0.06 加宽到 0.15 以扩大站距；大腿/小腿静止在 0。环境的
# _capture_usd_default_joint_state 就是用这组值给 12 个受驱关节写 default_joint_pos。
DEFAULT_JOINT_POS = np.array([
    0.15, 0.15, -0.15, -0.15,  # 髋（fl, fr, rl, rr）
    0.0, 0.0, 0.0, 0.0,        # 大腿
    0.0, 0.0, 0.0, 0.0,        # 小腿连杆
])
# 逐关节动作缩放——与 env_cfg.action_scale 完全一致。如果用旧的统一 0.5，
# 大腿/小腿动作幅度会被压到 0.5/0.8145 = 61%，机器人抬不起腿、走不动。
ACTION_SCALE = np.array([
    0.5, 0.5, 0.5, 0.5,            # 髋
    0.8145, 0.8145, 0.8145, 0.8145,  # 大腿
    0.8145, 0.8145, 0.8145, 0.8145,  # 小腿连杆
])
ACTION_CLIP = 1.5
INIT_HEIGHT = 0.35
INIT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
COMMANDED_LIN_VEL_XY = np.array([1.0, 0.0])
COMMANDED_ANG_VEL_Z = 0.0
CONTROL_HZ = 50
# 训练用的是 200 Hz 物理，但 Isaac 的闭链腿是刚性 PhysX 精简坐标关节；
# MuJoCo 用软 <connect> 等式约束近似，需要小步长才能既刚又稳
# （约束 timeconst 必须 >= 2*dt）。1000 Hz 让闭链足够刚、下肢能承重
# 而不软塌；50 Hz 的策略控制频率不变。
SIM_HZ = 1000


class ActorMLP(nn.Module):
    """复刻 rsl_rl ActorCritic 的 actor：Linear-ELU 堆叠，末层输出动作均值。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256, 128)):
        super().__init__()
        dims = [obs_dim, *hidden, action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def load_actor(checkpoint_path: Path, obs_dim: int, action_dim: int) -> ActorMLP:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # logs/ 里存在两种磁盘格式：
    #   旧（改名前的 rsl_rl）: {"actor_state_dict": {"mlp.0.weight", ...}}
    #   新（env_isaaclab rsl_rl）: {"model_state_dict": {"actor.0.weight", ...}}
    # 两种都归一化成本地 ActorMLP 的 "mlp.*" 键。
    if "actor_state_dict" in ckpt:
        mlp_state = {
            k: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")
        }
    elif "model_state_dict" in ckpt:
        mlp_state = {
            "mlp." + k[len("actor."):]: v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("actor.")
        }
    else:
        raise KeyError(
            "checkpoint missing 'actor_state_dict'/'model_state_dict' key. "
            f"available: {list(ckpt.keys())}"
        )
    # 隐藏层尺寸从权重形状推导而不是硬编码 (256, 256, 128)——
    # 网络结构一变就在形状层面大声报错，绝不静默错下去。
    weight_keys = sorted(
        (k for k in mlp_state if k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    dims = [mlp_state[weight_keys[0]].shape[1]] + [
        mlp_state[k].shape[0] for k in weight_keys
    ]
    if dims[0] != obs_dim or dims[-1] != action_dim:
        raise ValueError(
            f"checkpoint actor is {dims[0]}->{dims[-1]} but deploy expects "
            f"{obs_dim}->{action_dim}: {checkpoint_path}"
        )
    actor = ActorMLP(obs_dim, action_dim, hidden=tuple(dims[1:-1]))
    # 丢弃高斯 std 参数——部署用确定性均值。
    actor.load_state_dict(mlp_state, strict=True)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor


class MujocoDeployer:
    """lin_vel_source / imu_source 复刻 deploy/real/rl_deploy.py 的观测降级：
    - lin_vel_source: "sim"=真值（特权观测，Isaac 训练同款）；"zero"=obs[0:3]=0
      （真机无线速度传感器时 rl_deploy 的实际输入）。
    - imu_source: "sim"=真值角速度+投影重力；"stub"=ang_vel=0, gravity=[0,0,-1]
      （真机 IMU 未接入时 rl_deploy 的实际输入）。
    """

    def __init__(self, mjcf_path: Path, lin_vel_source: str = "sim",
                 imu_source: str = "sim", kp: float = 0.0, kv: float = 0.0,
                 action_lpf: float = 1.0):
        self.lin_vel_source = lin_vel_source
        self.imu_source = imu_source
        # 动作低通滤波系数 α∈(0,1]：target 用 α*新动作+(1-α)*上次，1=不滤波。
        # 只滤执行路径；obs 里的 prev_action 仍是策略原始输出（契约不变）。
        self.action_lpf = float(action_lpf)
        self._filt_action = np.zeros(12, dtype=np.float64)
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        # 覆盖 12 个 position 执行器的 PD 增益（0 = 用 MJCF 里的训练值 25/0.5）。
        # 用途：模拟真机 rl_deploy 实际下发的增益（motor_kp=8 转子侧 ≈ 关节侧
        # 8*6.33^2≈320）——训练增益 kp=25 在 MuJoCo 软闭链下撑不住 11.7 kg 自重，
        # 四连杆会穿过奇异点翻进折叠装配分支（真机有物理限位不会）。
        if kp > 0 or kv > 0:
            for i in range(self.model.nu):
                if kp > 0:
                    self.model.actuator_gainprm[i, 0] = kp
                    self.model.actuator_biasprm[i, 1] = -kp
                if kv > 0:
                    self.model.actuator_biasprm[i, 2] = -kv
        # 仿真步长按 SIM_HZ 设置（软闭链需要 1000 Hz，见顶部 SIM_HZ 注释）。
        self.model.opt.timestep = 1.0 / SIM_HZ
        self.data = mujoco.MjData(self.model)

        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.base_id < 0:
            raise RuntimeError("body 'base' not found in MJCF")

        # 12 个受驱关节在 qpos/qvel 里的地址。
        self.qpos_idx = []
        self.qvel_idx = []
        for name in ACTUATED_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"actuated joint '{name}' not in MJCF")
            self.qpos_idx.append(self.model.jnt_qposadr[jid])
            self.qvel_idx.append(self.model.jnt_dofadr[jid])
        self.qpos_idx = np.array(self.qpos_idx, dtype=int)
        self.qvel_idx = np.array(self.qvel_idx, dtype=int)

        # 按名字映射执行器，确保 ctrl 顺序与 ACTUATED_JOINTS 一致。
        self.ctrl_idx = []
        for jname in ACTUATED_JOINTS:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
            if aid < 0:
                raise RuntimeError(f"actuator 'act_{jname}' not in MJCF")
            self.ctrl_idx.append(aid)
        self.ctrl_idx = np.array(self.ctrl_idx, dtype=int)

        self.prev_actions = np.zeros(12, dtype=np.float32)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # 自由关节 qpos 布局：[px py pz, qw qx qy qz]
        self.data.qpos[0:3] = [0.0, 0.0, INIT_HEIGHT]
        self.data.qpos[3:7] = INIT_QUAT_WXYZ
        # 受驱关节摆到默认姿态；其余（被动）关节保持 0。
        self.data.qpos[self.qpos_idx] = DEFAULT_JOINT_POS
        self.data.qvel[:] = 0.0
        self.prev_actions[:] = 0.0
        self._filt_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def observe(self) -> np.ndarray:
        # base 机体在自身坐标系下的速度（线速度、角速度）。
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                  self.base_id, vel6, 1)  # 1 = 机体局部系
        ang_body = vel6[:3]
        lin_body = vel6[3:6]

        # 机体系下的投影重力：R^T * (0, 0, -1)，R 为 base→世界的旋转。
        base_rot = self.data.xmat[self.base_id].reshape(3, 3)
        gravity_body = base_rot.T @ np.array([0.0, 0.0, -1.0])

        if self.lin_vel_source == "zero":
            lin_body = np.zeros(3)
        if self.imu_source == "stub":
            ang_body = np.zeros(3)
            gravity_body = np.array([0.0, 0.0, -1.0])

        joint_pos = self.data.qpos[self.qpos_idx]
        joint_vel = self.data.qvel[self.qvel_idx]
        joint_pos_rel = joint_pos - DEFAULT_JOINT_POS

        obs = np.concatenate([
            lin_body, ang_body, gravity_body,
            joint_pos_rel, joint_vel, self.prev_actions,
        ]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def apply_action(self, action: np.ndarray) -> None:
        # 复刻 Isaac Lab：先 clip，再 `processed = scale * action + default_pos`。
        clipped = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        self._filt_action = (self.action_lpf * clipped
                             + (1.0 - self.action_lpf) * self._filt_action)
        target = ACTION_SCALE * self._filt_action + DEFAULT_JOINT_POS
        # ctrl_idx[i] 是 MuJoCo 执行器下标，target[i] 是该执行器所驱关节的
        # 目标 qpos——两者顺序一致。
        for i, aid in enumerate(self.ctrl_idx):
            self.data.ctrl[aid] = target[i]
        self.prev_actions[:] = clipped


class GaitTracker:
    """基于足-地接触的步态统计（50 Hz 控制步粒度）。

    评价基准（0.3 m 级 trot 四足的健康区间,供读数参考）:
      duty factor 0.5~0.65 | 步频 1.5~3 Hz | 抬脚 3~6 cm | 对角同步率 >0.8
    """

    FEET = ["fl_shank", "fr_shank", "rl_shank", "rr_shank"]

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.FEET]
        self.contact_hist = [[] for _ in self.FEET]   # 每控制步一个 bool（是否触地）
        self.z_hist = [[] for _ in self.FEET]
        self.z0 = [0.0] * 4                            # 站立基准足高（settle 后标定）
        self.action_hist = []

    def calibrate(self):
        self.z0 = [float(self.data.xpos[b][2]) for b in self.bids]

    def record(self, action):
        touching = set()
        for c in self.data.contact[: self.data.ncon]:
            for g_self, g_other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g_other == self.floor:
                    touching.add(self.model.geom_bodyid[g_self])
        for i, b in enumerate(self.bids):
            self.contact_hist[i].append(b in touching)
            self.z_hist[i].append(float(self.data.xpos[b][2]))
        self.action_hist.append(np.asarray(action, dtype=np.float32).copy())

    def report(self, control_dt):
        n = len(self.action_hist)
        if n < 10:
            return
        t_total = n * control_dt
        print(f"[gait] {'foot':>9} {'duty':>6} {'步频Hz':>7} {'抬脚cm':>7} {'摆动ms':>7}")
        print(f"[gait] {'(健康区间)':>9} {'.50-.65':>6} {'1.5-3':>7} {'3-6':>7} {'150-250':>7}"
              f"   <- 0.3m 级 trot 四足参考值,四脚应接近")
        for i, name in enumerate(self.FEET):
            c = np.array(self.contact_hist[i])
            z = np.array(self.z_hist[i]) - self.z0[i]
            duty = c.mean()
            touchdowns = int(np.sum(c[1:] & ~c[:-1]))
            freq = touchdowns / t_total
            # 每段摆动相（连续离地）的峰值抬高与时长
            lifts, airs = [], []
            start = None
            for k in range(n):
                if not c[k] and start is None:
                    start = k
                elif c[k] and start is not None:
                    lifts.append(z[start:k].max() if k > start else 0.0)
                    airs.append((k - start) * control_dt)
                    start = None
            lift_cm = 100 * float(np.mean(lifts)) if lifts else 0.0
            air_ms = 1000 * float(np.mean(airs)) if airs else 0.0
            print(f"[gait] {name:>9} {duty:6.2f} {freq:7.2f} {lift_cm:7.1f} {air_ms:7.0f}")
        cf = [np.array(h) for h in self.contact_hist]
        diag_sync = float(np.mean((cf[0] == cf[3]) & (cf[1] == cf[2])))  # fl-rr, fr-rl
        acts = np.stack(self.action_hist)
        act_rate = float(np.abs(np.diff(acts, axis=0)).mean())
        print(f"[gait] 对角同步率(trot)={diag_sync:.2f} (>0.8 好) | "
              f"动作变化率 mean|Δa|={act_rate:.3f}/步 (越小越平滑)")


def run_loop(args) -> int:
    mjcf = Path(args.mjcf)
    if not mjcf.exists():
        print(f"[error] MJCF not found at {mjcf}. Run tools/asset/usd_to_mjcf.py first.", file=sys.stderr)
        return 1

    deployer = MujocoDeployer(mjcf, lin_vel_source=args.lin_vel_source,
                              imu_source=args.imu_source, kp=args.kp, kv=args.kv,
                              action_lpf=args.action_lpf)
    deployer.reset()
    actor = load_actor(Path(args.checkpoint), obs_dim=45, action_dim=12)

    decimation = SIM_HZ // CONTROL_HZ
    control_dt = 1.0 / CONTROL_HZ

    # 站姿保持阶段：对齐 rl_deploy.py 的 hold_secs——先用默认姿态伺服让机器人
    # 落地稳定，再切策略。不加这段策略会在自由落体/触地反弹中启动，开局即摔。
    if args.settle > 0:
        for _ in range(int(args.settle * CONTROL_HZ)):
            deployer.apply_action(np.zeros(12))
            for _ in range(decimation):
                mujoco.mj_step(deployer.model, deployer.data)
        z0 = float(deployer.data.qpos[2])
        print(f"[info] settle {args.settle:.1f}s done, standing height = {z0:.3f} m")

    tracker = GaitTracker(deployer.model, deployer.data)
    tracker.calibrate()
    max_control_steps = int(args.duration * CONTROL_HZ) if args.duration > 0 else None

    def step_control(step_i: int) -> bool:
        obs = deployer.observe()
        with torch.no_grad():
            action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        deployer.apply_action(action)
        for _ in range(decimation):
            mujoco.mj_step(deployer.model, deployer.data)
        tracker.record(action)
        if not np.all(np.isfinite(deployer.data.qpos)):
            print(f"[warn] non-finite qpos at control step {step_i}; resetting.")
            deployer.reset()
            return False
        return True

    if args.headless:
        # 摔倒判定与训练 env 一致（fall_height_threshold / fall_cos_threshold），
        # 摔倒即停——真机此时 rl_deploy 已经触发急停，没有继续跑的意义。
        FALL_HEIGHT, FALL_COS = 0.22, 0.85
        print(f"[info] headless run for {args.duration:.1f}s @ {CONTROL_HZ} Hz control "
              f"(lin_vel={args.lin_vel_source}, imu={args.imu_source})")
        t0 = time.time()
        heights, tilts = [], []
        fall_step = None
        steps_run = 0
        for k in range(max_control_steps or 0):
            step_control(k)
            steps_run = k + 1
            z = float(deployer.data.qpos[2])
            cos_tilt = float(deployer.data.xmat[deployer.base_id].reshape(3, 3)[2, 2])
            heights.append(z)
            tilts.append(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))
            if fall_step is None and (z < FALL_HEIGHT or cos_tilt < FALL_COS):
                fall_step = k
                if not args.no_fall_stop:
                    break
        elapsed = time.time() - t0
        base = deployer.data.qpos[:3]
        sim_time = steps_run * control_dt
        print(f"[info] {steps_run} steps in {elapsed:.2f}s "
              f"(realtime ratio {sim_time / elapsed:.2f})")
        print(f"[info] base pos = ({base[0]:+.3f}, {base[1]:+.3f}, {base[2]:+.3f})")
        if fall_step is not None:
            print(f"[result] FELL at t={fall_step * control_dt:.2f}s "
                  f"(height={heights[fall_step]:.3f} m, tilt={tilts[fall_step]:.1f}°)"
                  + (f"; final height={heights[-1]:.3f} m, tilt={tilts[-1]:.1f}°"
                     if args.no_fall_stop else ""))
        else:
            print(f"[result] SURVIVED {sim_time:.1f}s")
        vx_mean = base[0] / sim_time if sim_time > 0 else 0.0
        print(f"[result] distance x={base[0]:+.3f} m, y={base[1]:+.3f} m | "
              f"mean vx={vx_mean:+.3f} m/s (command {COMMANDED_LIN_VEL_XY[0]:.1f}) | "
              f"mean height={np.mean(heights):.3f} m (target 0.32) | "
              f"max tilt={max(tilts):.1f}°")
        tracker.report(control_dt)
        return 0 if fall_step is None else 3

    with mujoco.viewer.launch_passive(deployer.model, deployer.data) as viewer:
        print("[info] viewer launched. press Esc to quit.")
        k = 0
        next_tick = time.time()
        while viewer.is_running():
            if max_control_steps is not None and k >= max_control_steps:
                break
            step_control(k)
            viewer.sync()
            k += 1
            next_tick += control_dt * args.slowmo
            sleep = next_tick - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.time()
    tracker.report(control_dt)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF),
                    help=f"MJCF 路径（默认: {DEFAULT_MJCF}）。")
    ap.add_argument("--checkpoint", default=None,
                    help="rsl_rl 的 .pt（含 actor_state_dict 或 model_state_dict 均可）。"
                         "不给则自动挑 logs/rsl_rl 下最新的 model_*.pt。")
    ap.add_argument("--headless", action="store_true",
                    help="不开 viewer 静默跑（SSH/无显示环境用），结束输出定量报告。")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="跑多少秒；0 = 不限时（仅 viewer 模式），headless 必须 > 0。")
    ap.add_argument("--lin-vel-source", choices=["sim", "zero"], default="sim",
                    help="obs[0:3] 线速度来源：sim=真值（训练同款特权观测）；"
                         "zero=喂 0（真机 rl_deploy 无传感器时的实际输入）。")
    ap.add_argument("--kp", type=float, default=0.0,
                    help="覆盖执行器 kp（关节侧 N·m/rad；0=用 MJCF 的训练值 25）。"
                         "真机 rl_deploy 当前增益 ≈ 8*6.33^2 ≈ 320。")
    ap.add_argument("--kv", type=float, default=0.0,
                    help="覆盖执行器 kv（0=用 MJCF 的 0.5）。真机 ≈ 0.08*6.33^2 ≈ 3.2。")
    ap.add_argument("--slowmo", type=float, default=1.0,
                    help="viewer 慢放倍数（5 = 比真实慢 5 倍，看清步态用；"
                         "只影响播放节奏，不影响物理和策略）。")
    ap.add_argument("--action-lpf", type=float, default=1.0,
                    help="动作低通滤波 α∈(0,1]，1=不滤波。0.3~0.5 可明显压高频抖动；"
                         "只滤执行路径，obs 的 prev_action 契约不变。注意这改变了"
                         "闭环动力学，真机采用前需在此验证稳定性。")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="切策略前用默认姿态伺服站稳的秒数（对齐 rl_deploy 的 "
                         "hold_secs；0 = 关闭，旧行为：落地瞬间即开策略）。")
    ap.add_argument("--no-fall-stop", action="store_true",
                    help="headless 下摔倒不提前终止，跑满 duration 观察后续行为"
                         "（MuJoCo 闭链站高上限 ~0.26 m，0.22 阈值可能被起始震荡误触发）。")
    ap.add_argument("--imu-source", choices=["sim", "stub"], default="sim",
                    help="obs[3:9] 来源：sim=真值角速度+投影重力；"
                         "stub=ang_vel=0, gravity=[0,0,-1]（真机 IMU 未接入时的输入）。")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.headless and args.duration <= 0:
        print("[error] --headless requires --duration > 0", file=sys.stderr)
        sys.exit(2)
    if args.checkpoint is None:
        latest = find_latest_checkpoint()
        if latest is None:
            print(f"[error] no model_*.pt found under {LOG_ROOT}; pass --checkpoint.",
                  file=sys.stderr)
            sys.exit(2)
        args.checkpoint = str(latest)
        print(f"[info] no --checkpoint given; using latest {latest}")
    sys.exit(run_loop(args))
