"""mos2026_2 四足「机身移动速度 ↔ 各电机角速度」IsaacSim 实时可视化。

做什么
------
在 IsaacSim 里以**运动学播放**方式让机器人按可配置的机身速度前进、腿按 trot 步态
循环，并用 omni.ui 实时 HUD 显示：

  · 当前机身速度（m/s）
  · 12 个驱动电机的角速度（rad/s 与 rpm），图形横条 + 颜色按载荷（绿/橙/红）
  · 按控制步滚动的**角速度曲线**（hip/thigh/shank 三组图，4 腿四色叠加，
    叠训练软限 15 / 电机物理 30 rad/s 两条参考线）
  · 每个电机的**峰值保持**（peak-hold）+ 峰值数值表（rad/s 关节侧 / 电机轴 / rpm）

外加复用 env 的速度箭头 marker（绿=命令速度方向，蓝=实际）。

为什么是「运动学播放」
----------------------
关节角与角速度由 `传统步态/代码/speed_map.py` 的解析解算器逐帧给出（足端轨迹→IK→
解析 Jacobian→q̇），机身按目标速度平移。这是**几何/运动学真值**，不含接触动力学——
正好直接把「机身速度 ↔ 每个电机要转多快」画在机器人身上。HUD 上的电机角速度是权威
解析值；sim 里的机器人姿态是可视化载体（膝为平行四连杆闭链，shank 电机↔等效膝角的
传动比未标定，见 kinematics.py 文首与 todo §E，故腿姿为近似）。

运行（在 IsaacLab/isaac 环境）：
  ../env_isaaclab/bin/python tools/isaac/speed_viz_isaac.py --speed 1.0
  ../env_isaaclab/bin/python tools/isaac/speed_viz_isaac.py --speed 2.5 --step-length 0.18
  ../env_isaaclab/bin/python tools/isaac/speed_viz_isaac.py --ramp --v-max 3.0   # 周期性扫速度
  # 验证脚本能跑（无显示器/CI）：加 --headless，HUD 自动退化为控制台表
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# deploy/common 是纯 numpy 解算器，加进 path 直接复用（不依赖被重命名的 source/ 包）
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "传统步态" / "代码"))

from isaaclab.app import AppLauncher

# --- CLI（AppLauncher 的参数也挂上：--headless / --device 等）---------------------
parser = argparse.ArgumentParser(description="mos2026_2 速度↔电机角速度 IsaacSim 实时可视化")
parser.add_argument("--speed", type=float, default=1.0, help="目标机身速度 m/s（恒定模式）")
parser.add_argument("--step-length", type=float, default=0.12, help="步幅 m")
parser.add_argument("--duty", type=float, default=0.5, help="支撑相占空比（trot=0.5）")
parser.add_argument("--step-height", type=float, default=0.04, help="摆动相抬腿高度 m")
parser.add_argument("--body-height", type=float, default=0.30, help="站立足端深度 m")
parser.add_argument("--ramp", action="store_true", help="周期性三角扫速度 0→v_max→0，直观看转速随速度变化")
parser.add_argument("--ramp-period", type=float, default=10.0, help="--ramp 一个来回的时长 s")
parser.add_argument("--v-max", type=float, default=3.0, help="--ramp 的峰值速度 m/s")
parser.add_argument("--num-steps", type=int, default=0, help="控制步数上限（0=直到关窗）")
parser.add_argument("--base-z", type=float, default=0.36, help="机身离地高度 m（仅可视化）")
parser.add_argument("--console-hud", action="store_true", help="即便有 GUI 也打印控制台表")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 以下导入必须在 SimulationApp 起来之后 -------------------------------------
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from pxr import Sdf  # noqa: E402

from gait import TrotGait  # noqa: E402
from kinematics import LEG_NAMES  # noqa: E402
import speed_map as sm  # noqa: E402

# --- 机器人常量（与 env_cfg.py / play_mujoco.py 对齐）---------------------------
USD_PATH = REPO_ROOT / "source/mos_one/mos_one/assets/robots/mos2026_2_closed_usd/usd/mos2026_2.usd"
if not USD_PATH.exists():
    # 包重命名兜底：扫描 source/ 找 mos2026_2.usd
    cands = list((REPO_ROOT / "source").rglob("mos2026_2.usd"))
    if cands:
        USD_PATH = cands[0]

ACTUATED_JOINTS = [
    "fl_hip", "fr_hip", "rl_hip", "rr_hip",
    "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
    "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a",
]
DEFAULT_HIP = np.array([0.15, 0.15, -0.15, -0.15])      # fl,fr,rl,rr
THIGH_AXIS_SIGN = np.array([-1.0, +1.0, -1.0, +1.0])    # thigh 关节 y 轴符号
SHANK_AXIS_SIGN = np.array([-1.0, +1.0, -1.0, +1.0])    # shank_link 关节 y 轴符号
LOOP_JOINTS = {"fl_close_loop", "fr_close_loop", "rl_close_loop", "rr_close_loop"}

GEAR = sm.GEAR                       # 6.33
RAD_S_TO_RPM = sm.RAD_S_TO_RPM
V_TRAIN, V_PHYS = sm.JOINT_VEL_TRAIN, sm.JOINT_VEL_PHYS   # 15 / 30 rad/s (关节侧)


def make_robot_cfg() -> ArticulationCfg:
    """内联 ArticulationCfg（从 env_cfg 抄关键字段，prim_path 单机器人）。"""
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,           # 运动学播放：关重力，不让它掉下去
                retain_accelerations=False,
                max_linear_velocity=50.0,
                max_angular_velocity=400.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, args_cli.base_z),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "fl_hip": 0.15, "fr_hip": 0.15, "rl_hip": -0.15, "rr_hip": -0.15,
                "fl_thigh": 0.0, "fr_thigh": 0.0, "rl_thigh": 0.0, "rr_thigh": 0.0,
                "fl_shank_link": 0.0, "fr_shank_link_a": 0.0,
                "rl_shank_link_a": 0.0, "rr_shank_link_a": 0.0,
            },
            joint_vel={},
        ),
        actuators={
            "main": ImplicitActuatorCfg(
                joint_names_expr=ACTUATED_JOINTS,
                stiffness=200.0,    # 运动学播放：刚一点，让 written 目标稳稳保持
                damping=5.0,
                effort_limit_sim=80.0,
                velocity_limit_sim=400.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def patch_loop_joints(stage) -> None:
    """给 *_close_loop 闭环约束开 projection（照搬 env._patch_projected_loop_joints），
    否则纯运动学步进时平行四连杆 anchor 会漂、被动 shank 不跟随。"""
    patched = []
    for prim in stage.TraverseAll():
        if prim.GetName() not in LOOP_JOINTS:
            continue
        prim.CreateAttribute("physxJoint:enableProjection", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxJoint:projectionLinearTolerance", Sdf.ValueTypeNames.Float).Set(0.002)
        prim.CreateAttribute("physxJoint:projectionAngularTolerance", Sdf.ValueTypeNames.Float).Set(0.05)
        patched.append(prim.GetName())
    print(f"[viz] loop-joint projection patched: {sorted(set(patched)) or '(none found)'}", flush=True)


def convention_to_sim(q: np.ndarray, qd: np.ndarray):
    """约定角 (4,3)[ab,hip,knee] → 12 个 sim 关节的 (pos, vel)，顺序同 ACTUATED_JOINTS。

    hip 电机 ← q_ab（gait 基本不外摆，叠加到默认外撇上）；thigh ← q_hip；
    shank_link ← q_knee（闭链等效，近似映射，见文件头）。轴符号镜像四腿同向。
    """
    hip_pos = DEFAULT_HIP + q[:, 0]
    thigh_pos = THIGH_AXIS_SIGN * q[:, 1]
    shank_pos = SHANK_AXIS_SIGN * q[:, 2]
    pos = np.concatenate([hip_pos, thigh_pos, shank_pos])
    hip_vel = qd[:, 0]
    thigh_vel = THIGH_AXIS_SIGN * qd[:, 1]
    shank_vel = SHANK_AXIS_SIGN * qd[:, 2]
    vel = np.concatenate([hip_vel, thigh_vel, shank_vel])
    return pos, vel


def motor_speeds_abs(qd: np.ndarray) -> np.ndarray:
    """12 个电机的角速度大小 (rad/s, 关节侧)，顺序同 ACTUATED_JOINTS = |q̇|。"""
    return np.concatenate([np.abs(qd[:, 0]), np.abs(qd[:, 1]), np.abs(qd[:, 2])])


# ============================ HUD（omni.ui，GUI 才建；否则控制台）============================
HIST_LEN = 400          # 滚动曲线窗口：400 控制步 ≈ 8 s @ 50 Hz
PLOT_MAX = V_PHYS * 1.1  # 曲线 y 轴上限 (rad/s 关节侧)
# 四腿曲线配色（omni.ui 颜色为 0xAABBGGRR）：fl 绿 / fr 青 / rl 紫 / rr 白
LEG_COLORS = (0xFF60D060, 0xFFE0C040, 0xFFD060B0, 0xFFE8E8E8)
COLOR_TRAIN = 0xFF20A0F0   # 橙：训练软限 15 rad/s
COLOR_PHYS = 0xFF3040E0    # 红：电机物理限 30 rad/s


class MotorSpeedHUD:
    """实时显示机身速度 + 12 电机角速度条/滚动曲线/峰值表。omni.ui 失败则退化控制台表。

    曲线按「控制步」推进（与 update() 调用同频，50 Hz），每关节一条，按 hip/thigh/
    shank 分三组图，腿用颜色区分；峰值表随 peak-hold 实时刷新。
    """

    def __init__(self, use_gui: bool, console: bool):
        self.peak = np.zeros(12)
        self.hist = np.zeros((12, HIST_LEN))   # 滚动角速度历史（关节侧 rad/s）
        self.use_omni = False
        self.console = console or (not use_gui)
        self._last_print = -1.0
        self._speed_label = None
        self._rows = []      # [(bar_rect, value_label)] × 12
        self._plots = []     # ui.Plot × 12，顺序同 ACTUATED_JOINTS
        self._tbl = []       # [[joint_rad_s, motor_rad_s, rpm] labels] × 12
        self._plot_warned = False
        if use_gui:
            try:
                self._build_omni()
                self.use_omni = True
            except Exception as e:  # noqa: BLE001
                print(f"[viz] omni.ui HUD 构建失败，退化为控制台表: {e}", flush=True)
                self.console = True

    def _build_omni(self):
        import omni.ui as ui
        self._ui = ui
        self._win = ui.Window("Motor angular speed", width=640, height=980)
        with self._win.frame:
            with ui.ScrollingFrame(
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=3):
                    self._speed_label = ui.Label("body speed: -- m/s", height=28,
                                                 style={"font_size": 22, "color": 0xFFFFFFFF})
                    ui.Label(f"motor = joint x {GEAR:.2f} | caps: train {V_TRAIN:.0f} / phys "
                             f"{V_PHYS:.0f} rad/s (joint)", height=18,
                             style={"font_size": 12, "color": 0xFF9AA0A6})
                    ui.Spacer(height=4)

                    # --- ① 瞬时角速度：图形条 + 数值 ---------------------------------
                    for name in ACTUATED_JOINTS:
                        with ui.HStack(height=20, spacing=6):
                            ui.Label(name, width=110,
                                     style={"font_size": 13, "color": 0xFFE8EAED})
                            with ui.ZStack(width=150, height=14):
                                ui.Rectangle(style={"background_color": 0xFF303030,
                                                    "border_radius": 2})
                                with ui.HStack():
                                    bar = ui.Rectangle(
                                        width=ui.Fraction(0.003),
                                        style={"background_color": 0xFF60D060,
                                               "border_radius": 2})
                                    ui.Spacer()
                            val = ui.Label("", style={"font_size": 13, "color": 0xFF60D060})
                            self._rows.append((bar, val))

                    # --- ② 按控制步滚动的角速度曲线 ----------------------------------
                    ui.Spacer(height=6)
                    ui.Label(f"speed curves — last {HIST_LEN} ctrl steps (50 Hz), "
                             f"y: 0–{PLOT_MAX:.0f} rad/s joint-side",
                             height=16, style={"font_size": 12, "color": 0xFF9AA0A6})
                    with ui.HStack(height=16, spacing=12):
                        for li, leg in enumerate(("fl", "fr", "rl", "rr")):
                            ui.Label(f"— {leg}", width=44,
                                     style={"font_size": 12, "color": LEG_COLORS[li]})
                        ui.Label(f"— train {V_TRAIN:.0f}", width=70,
                                 style={"font_size": 12, "color": COLOR_TRAIN})
                        ui.Label(f"— phys {V_PHYS:.0f}", width=70,
                                 style={"font_size": 12, "color": COLOR_PHYS})
                    self._plots = [None] * 12
                    self._cur_vals = []   # 每组 4 条腿的当前值 Label（曲线右上角数值读数）
                    PLOT_H = 96
                    for gname, idxs in (("hip (ab)", range(0, 4)),
                                        ("thigh", range(4, 8)),
                                        ("shank / knee", range(8, 12))):
                        ui.Label(gname, height=14,
                                 style={"font_size": 12, "color": 0xFFE8EAED})
                        with ui.HStack(height=PLOT_H, spacing=4):
                            # 左侧 y 轴刻度：0 / 15 / 30 rad/s（色与参考线一致）
                            with ui.ZStack(width=26):
                                for tick, tc in ((V_PHYS, COLOR_PHYS),
                                                 (V_TRAIN, COLOR_TRAIN),
                                                 (0.0, 0xFF9AA0A6)):
                                    off = round((1.0 - tick / PLOT_MAX) * PLOT_H) - 6
                                    off = max(0, min(PLOT_H - 12, off))
                                    with ui.Placer(offset_y=off):
                                        ui.Label(f"{tick:.0f}", height=12,
                                                 style={"font_size": 11, "color": tc})
                            with ui.ZStack():
                                ui.Rectangle(style={"background_color": 0xFF1A1A1A,
                                                    "border_radius": 3})
                                # 两条水平参考线：用 2 点常值曲线画
                                ui.Plot(ui.Type.LINE, 0.0, PLOT_MAX, V_TRAIN, V_TRAIN,
                                        style={"color": COLOR_TRAIN, "background_color": 0x0})
                                ui.Plot(ui.Type.LINE, 0.0, PLOT_MAX, V_PHYS, V_PHYS,
                                        style={"color": COLOR_PHYS, "background_color": 0x0})
                                for k in idxs:
                                    self._plots[k] = ui.Plot(
                                        ui.Type.LINE, 0.0, PLOT_MAX, *self.hist[k],
                                        style={"color": LEG_COLORS[k % 4],
                                               "background_color": 0x0})
                                # 右上角：四条腿当前角速度数值（按腿配色）
                                with ui.VStack():
                                    with ui.HStack(height=14):
                                        ui.Spacer()
                                        cur = []
                                        for li in range(4):
                                            cur.append(ui.Label(
                                                "", width=42,
                                                style={"font_size": 11,
                                                       "color": LEG_COLORS[li]}))
                                        ui.Label("rad/s", width=36,
                                                 style={"font_size": 11,
                                                        "color": 0xFF9AA0A6})
                                        self._cur_vals.append(cur)
                                    ui.Spacer()

                    # --- ③ 峰值角速度数值表 ------------------------------------------
                    ui.Spacer(height=6)
                    ui.Label("max |angular speed| since start (peak-hold)",
                             height=16, style={"font_size": 12, "color": 0xFF9AA0A6})
                    hdr_style = {"font_size": 12, "color": 0xFF9AA0A6}
                    with ui.HStack(height=16, spacing=6):
                        ui.Label("joint", width=110, style=hdr_style)
                        ui.Label("joint rad/s", width=90, style=hdr_style)
                        ui.Label("motor rad/s", width=90, style=hdr_style)
                        ui.Label("motor rpm", width=90, style=hdr_style)
                    for name in ACTUATED_JOINTS:
                        with ui.HStack(height=16, spacing=6):
                            ui.Label(name, width=110,
                                     style={"font_size": 12, "color": 0xFFE8EAED})
                            cells = [ui.Label("--", width=90,
                                              style={"font_size": 12, "color": 0xFF60D060})
                                     for _ in range(3)]
                            self._tbl.append(cells)

    @staticmethod
    def _color(joint_rad_s: float) -> int:
        # omni.ui 颜色是 0xAABBGGRR。绿<50% / 橙<85% / 红，阈值按物理限 30 rad/s。
        load = joint_rad_s / V_PHYS
        if load < 0.5:
            return 0xFF60D060
        if load < 0.85:
            return 0xFF20A0F0
        return 0xFF3040E0

    def accumulate(self, motor_joint_rad_s: np.ndarray):
        """每物理步调用：峰值保持（比按控制率采样更准，能抓到峰值瞬间）。"""
        self.peak = np.maximum(self.peak, np.nan_to_num(motor_joint_rad_s))

    def update(self, v_body: float, motor_joint_rad_s: np.ndarray, t: float):
        """motor_joint_rad_s: 12 个关节侧角速度 (rad/s)。电机轴 = ×GEAR。

        每调用一次 = 一个控制步：滚动历史推进一格，曲线/条/峰值表同步刷新。
        """
        js_all = np.nan_to_num(motor_joint_rad_s, nan=0.0, posinf=0.0, neginf=0.0)
        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = js_all
        if self.use_omni:
            try:
                self._speed_label.text = f"body speed: {v_body:5.2f} m/s    (t={t:5.1f}s)"
                ui = self._ui
                for i in range(12):
                    js = float(js_all[i])
                    rpm = js * GEAR * RAD_S_TO_RPM
                    color = self._color(js)
                    bar, val = self._rows[i]
                    bar.width = ui.Fraction(max(min(js / V_PHYS, 1.0), 0.003))
                    bar.style = {"background_color": color, "border_radius": 2}
                    val.text = f"{js:6.2f} rad/s  {rpm:6.0f} rpm   pk {self.peak[i]:6.2f}"
                    val.style = {"font_size": 13, "color": color}
                    # 峰值表
                    pk = float(self.peak[i])
                    pk_color = self._color(pk)
                    cells = self._tbl[i]
                    cells[0].text = f"{pk:6.2f}"
                    cells[1].text = f"{pk * GEAR:6.1f}"
                    cells[2].text = f"{pk * GEAR * RAD_S_TO_RPM:6.0f}"
                    for c in cells:
                        c.style = {"font_size": 12, "color": pk_color}
                # 滚动曲线（set_data 个别版本缺失时只警告一次，不拖垮整个 HUD）
                try:
                    for i in range(12):
                        self._plots[i].set_data(*self.hist[i])
                except AttributeError:
                    if not self._plot_warned:
                        self._plot_warned = True
                        print("[viz] 本版 omni.ui.Plot 不支持 set_data，曲线不更新"
                              "（条/峰值表不受影响）", flush=True)
                # 曲线右上角当前值读数（与曲线同源同帧）
                for gi in range(3):
                    for li in range(4):
                        self._cur_vals[gi][li].text = f"{js_all[gi * 4 + li]:5.1f}"
            except Exception as e:  # noqa: BLE001
                print(f"[viz] omni.ui 更新失败，转控制台: {e}", flush=True)
                self.use_omni = False
                self.console = True
        if self.console and (t - self._last_print) >= 0.5:
            self._last_print = t
            peak_motor = self.peak.max()
            j = int(np.argmax(self.peak))
            print(f"[viz] t={t:5.1f}s  v={v_body:4.2f} m/s | "
                  f"peak joint {self.peak[j]:5.2f} rad/s ({ACTUATED_JOINTS[j]}) "
                  f"= {self.peak[j]*GEAR*RAD_S_TO_RPM:.0f} rpm motor | "
                  f"thigh~{motor_joint_rad_s[4]:.2f} knee~{motor_joint_rad_s[8]:.2f} rad/s",
                  flush=True)


# ============================ 速度日程 ============================
def speed_at(t: float) -> float:
    if not args_cli.ramp:
        return args_cli.speed
    # 三角波 0→v_max→0
    p = (t % args_cli.ramp_period) / args_cli.ramp_period
    tri = 1.0 - abs(2.0 * p - 1.0)
    return max(0.02, args_cli.v_max * tri)


# ============================ 主流程 ============================
def main() -> int:
    if not USD_PATH.exists():
        print(f"[viz][error] 找不到机器人 USD：{USD_PATH}", file=sys.stderr)
        return 1

    sim_cfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=4,
        gravity=(0.0, 0.0, 0.0),     # 运动学播放：全局关重力
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2 ** 21,
            gpu_found_lost_aggregate_pairs_capacity=2 ** 23,
            gpu_total_aggregate_pairs_capacity=2 ** 22,
            gpu_max_rigid_contact_count=2 ** 22,
            gpu_max_rigid_patch_count=2 ** 19,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.2, -2.6, 1.4), target=(0.0, 0.0, 0.4))

    # 地面 + 灯光（纯视觉参考）
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2400.0, color=(0.78, 0.82, 0.9)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2400.0, color=(0.78, 0.82, 0.9)))

    robot = Articulation(make_robot_cfg())
    patch_loop_joints(get_current_stage())

    sim.reset()
    print(f"[viz] joints in articulation: {robot.num_joints} | "
          f"USD: {USD_PATH.name}", flush=True)

    # 解析驱动关节顺序 → articulation 下标
    j_ids, j_names = robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
    j_ids_t = torch.tensor(j_ids, device=sim.device, dtype=torch.long)
    if len(j_ids) != 12:
        print(f"[viz][error] 期望 12 个驱动关节，匹配到 {len(j_ids)}: {j_names}", file=sys.stderr)
        return 1

    use_gui = bool(getattr(sim, "has_gui", lambda: not args_cli.headless)())
    hud = MotorSpeedHUD(use_gui=use_gui, console=args_cli.console_hud)

    # 速度箭头 marker（绿=命令方向，蓝=实际），GUI 才建
    cmd_arrow = act_arrow = None
    if use_gui:
        try:
            ccfg = GREEN_ARROW_X_MARKER_CFG.copy(); ccfg.prim_path = "/Visuals/cmd_vel"
            ccfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
            cmd_arrow = VisualizationMarkers(ccfg)
            acfg = BLUE_ARROW_X_MARKER_CFG.copy(); acfg.prim_path = "/Visuals/act_vel"
            acfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
            act_arrow = VisualizationMarkers(acfg)
        except Exception as e:  # noqa: BLE001
            print(f"[viz] 速度箭头 marker 构建失败，跳过: {e}", flush=True)

    def show_arrow(marker, speed, base_pos, z_off):
        if marker is None:
            return
        scale = torch.tensor((0.4, 0.4, 0.4), device=sim.device).repeat(1, 1)
        scale[:, 0] *= abs(speed) * 3.0 + 1e-3
        heading = torch.zeros(1, device=sim.device)  # +x 前进
        quat = quat_from_euler_xyz(heading * 0, heading * 0, heading * 0)
        pos = base_pos.clone(); pos[:, 2] += z_off
        marker.visualize(pos, quat, scale)

    sim_dt = sim.get_physics_dt()
    decim = 4
    phase = 0.0           # 步态相位累积器（[0,1)），跨速度变化保持连续
    root_x = 0.0
    base_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)
    step = 0
    ctrl_steps = 0

    print(f"[viz] 开始播放：{'ramp 0→%.1f m/s' % args_cli.v_max if args_cli.ramp else 'v=%.2f m/s' % args_cli.speed}"
          f" | 步幅 {args_cli.step_length*100:.0f}cm β={args_cli.duty} | "
          f"{'GUI' if use_gui else 'headless(控制台 HUD)'}", flush=True)

    while simulation_app.is_running():
        t = step * sim_dt
        v = speed_at(t)
        gait = sm.gait_for_speed(
            v, step_length=args_cli.step_length, duty=args_cli.duty,
            step_height=args_cli.step_height, body_height=args_cli.body_height)

        # 相位推进（用当前周期，速度变化时相位连续不跳）
        phase = (phase + sim_dt / gait.period) % 1.0
        q, qd = sm.instant(gait, phase * gait.period)
        pos, vel = convention_to_sim(q, qd)

        # 写关节状态（精确）+ 设目标（让伺服稳稳保持，不在子步内漂）
        pos_t = torch.tensor(pos, device=sim.device, dtype=torch.float32).unsqueeze(0)
        vel_t = torch.tensor(vel, device=sim.device, dtype=torch.float32).unsqueeze(0)
        robot.write_joint_state_to_sim(pos_t, vel_t, joint_ids=j_ids_t)
        robot.set_joint_position_target(pos_t, joint_ids=j_ids_t)

        # 机身按速度平移
        root_x += v * sim_dt
        root_pose = torch.tensor([[root_x, 0.0, args_cli.base_z]], device=sim.device, dtype=torch.float32)
        root_pose = torch.cat([root_pose, base_quat], dim=1)   # (1,7) pos+quat
        try:
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(
                torch.tensor([[v, 0, 0, 0, 0, 0]], device=sim.device, dtype=torch.float32))
        except Exception:  # noqa: BLE001 — 某些版本 API 名不同
            robot.write_root_state_to_sim(
                torch.cat([root_pose,
                           torch.tensor([[v, 0, 0, 0, 0, 0]], device=sim.device, dtype=torch.float32)], dim=1))

        robot.write_data_to_sim()
        sim.step(render=(use_gui and step % decim == 0))
        robot.update(sim_dt)

        # 峰值每物理步累积（更准）；HUD/箭头按控制率 50Hz 呈现
        motor_js = motor_speeds_abs(qd)         # 12 关节侧 rad/s
        hud.accumulate(motor_js)
        if step % decim == 0:
            hud.update(v, motor_js, t)
            if use_gui:
                base_pos = robot.data.root_pos_w.clone()
                show_arrow(cmd_arrow, v, base_pos, 0.40)
                show_arrow(act_arrow, v, base_pos, 0.48)
            ctrl_steps += 1
            if args_cli.num_steps and ctrl_steps >= args_cli.num_steps:
                break
        step += 1

    print(f"[viz] 结束。各电机峰值角速度（rad/s 关节侧 / rpm 电机轴）：", flush=True)
    for i, name in enumerate(ACTUATED_JOINTS):
        print(f"   {name:18s} {hud.peak[i]:6.2f} rad/s  {hud.peak[i]*GEAR*RAD_S_TO_RPM:6.0f} rpm", flush=True)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rc = 1
    # simulation_app.close() 在 headless 批处理下常挂起。用看门狗给它 15s 机会，
    # 超时就硬退出让 OS 回收资源（GUI 下用户关窗后同样走这里）。
    import threading
    _done = threading.Event()

    def _close():
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        _done.set()

    threading.Thread(target=_close, daemon=True).start()
    _done.wait(timeout=15.0)
    sys.stdout.flush()
    os._exit(rc)
