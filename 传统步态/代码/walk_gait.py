#!/usr/bin/env python3
"""让 mos2026_2 在 MuJoCo 里走起来 —— 不用 RL，纯运动学 + 步态 + 位置伺服。

这是 tools/learn/go2_walk.py 在 Mos-One 上的对应实现。链路一样：

    速度指令 vx/vy/wz
        -> 步态相位（trot）
        -> 足端轨迹 p_des（腿系）
        -> 闭链 IK（closed_chain_kin）
        -> 三个受驱关节角 (hip, thigh, crank)
        -> position 执行器
        -> MuJoCo

与 Go2 的三处关键差异（都是被 MJCF 逼出来的，不是风格选择）
------------------------------------------------------------
1. **闭链五杆**：Go2 是串联 3-DOF，解析 IK 一行搞定；本机每条腿是共轴双输入
   平面四杆，膝角是 f(thigh, crank) 的双输入函数。IK 在 `closed_chain_kin`，
   已与 MuJoCo 约束解交叉验证到 1e-8。

2. **position 执行器**：MJCF 用 `<position>`，`data.ctrl` 是目标角不是力矩，
   所以没有 go2_walk 里的 PD + τ=JᵀF 重力前馈那一段。重力静差靠提高 kp 解决
   （kp=25 撑不住自重，见下）。

3. **要用足够硬的位置增益**：XML 里的 kp=25 是软伺服，静差 ≈τ_g/kp≈0.3 rad，
   机身会下沉、穿过四杆的奇异位形翻进折叠装配分支，站高只剩 ~0.17 m
   （doc 里记的「MuJoCo 闭链翻分支」）。本脚本**默认 kp=500**：trot 的机身竖直
   起伏（2×步频、双支撑颠簸）在 kp=320 下峰峰 ~20 mm，kp=500 降到 ~14 mm、
   std 减半，且不损速度。做 sim2real 保真（对齐真机关节侧增益 ~320）时用
   `--kp 320 --kv 3.2`，会颠得更明显。脚本逐步监控奇异余量 margin，翻分支立刻报。
   还想更平：`--bob-comp 0.002` 加竖直前馈，默认工况可压到 ~10 mm（见文末与代码）。

4. **站姿要前后配平（x_offset，默认 −55 mm）**：整机 CoM 在四足支撑中心
   **后方 23.6 mm**，不配平的话 trot 会持续点头，后腿离地只有 7 mm、全程拖地，
   速度达成率 58%。把足端整体后移后 pitch 从 ±0.80° 降到 ±0.55°、roll 从
   ±2.57° 降到 ±1.13°、达成率升到 88%。最优值 −55~−60 mm 比静态配平量大，
   多出来的部分是在压 trot 的动态点头。

性能参照：同口径下 tools/learn/go2_walk.py 在 Go2 上实测达成率 82%，本实现
默认工况 88%。开环步态的足端打滑是固有的，不是实现缺陷。

设置速度（三个方向都是机体系速度指令）：
    --vx  前进 m/s（按 步幅=vx·占空·周期 反推步幅；不给则用 --step-length）
    --vy  侧移 m/s（左正）      --wz  转向 rad/s（左转正）
  也可越过 --vx 直接调 --step-length / --period / --step-height 精细控制步态。

  ⚠️ 速度上限：vx 太大 ⇒ 步幅超出腿工作域（默认周期下步幅上限 ~0.40 m ⇒ vx≈1.6 m/s），
     会被开走前的可行性预检拦下并提示改法（腿伸不长，要提步频=缩短 --period，而非加大
     步幅）。且开环步态 >~1 m/s 常会动态失稳摔倒——这是物理，不是脚本 bug。

用法：
    python 传统步态/代码/walk_gait.py --headless -T 10           # 无窗跑，默认 vx=0.4
    python 传统步态/代码/walk_gait.py --headless --vx 0.6        # 指定前进速度
    python 传统步态/代码/walk_gait.py --headless --vx 0.4 --wz 0.3   # 边走边左转
    python 传统步态/代码/walk_gait.py --headless --vx 0 --wz 0.5     # 原地转
    MUJOCO_GL=glx python 传统步态/代码/walk_gait.py --viewer --vx 0.5  # 开窗看
    python 传统步态/代码/walk_gait.py --headless --sweep         # 六工况验收
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
# closed_chain_kin / gait 与本文件同在「传统步态/代码」里，加自身目录即可互导
sys.path.insert(0, str(Path(__file__).resolve().parent))

from closed_chain_kin import LEGS, LEG_NAMES, leg_ik, solve_linkage  # noqa: E402
from gait import NOMINAL_X, TrotGait  # noqa: E402

DEFAULT_SCENE = REPO_ROOT / "deploy/mujoco/assets/scene.xml"

# 执行器/关节顺序（与 stand_balance.py、play_mujoco.py 一致）
ACT = ["fl_hip", "fr_hip", "rl_hip", "rr_hip",
       "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
       "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a"]

SIM_HZ = 1000
CONTROL_HZ = 200

# 竖直起伏前馈的相位（rad），在默认工况下网格搜索得到（见 --bob-comp）
BOB_PHASE = 3.25


def _yaw_matrix(R: np.ndarray) -> np.ndarray:
    """从旋转阵取出纯 yaw 部分（绕世界 z）。"""
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class GaitController:
    """步态 -> 闭链 IK -> 12 关节目标角。

    IK 用上一控制周期的解做延拓（seed），既快又保证解连续、不会在两个装配
    分支之间跳。

    level_feet：摆动腿走重力对齐水平系（**实验性，默认关**）
    --------------------------------------------------------
    动机：trot 任一时刻只有对角两条腿支撑，机身天然会摇。未配平时实测 pitch
    在 0.4°~2.4° 之间摆、roll ±2.6°，后腿离地峰值只有 7 mm（前腿 44 mm），
    后腿全程拖地、占空比 0.75，速度只到理论值的 58%。

    做法：把摆动腿的足端轨迹解释为重力对齐（yaw 跟随机身）的水平系坐标，再用
    实测姿态转回机身系下发  p_body = R_bodyᵀ · R_yaw · p_level，
    使机身怎么歪、摆动腿在世界里的离地高度都不变。

    实测结论：**没用，反而更差**（vx 从 0.233 掉到 −0.013 m/s）。原因是支撑/
    摆动切换的瞬间目标位置会阶跃约 11 mm，对 kp=320 的硬伺服等于每步敲两下。
    真正有效的是静态配平 `x_offset`（见下），它把摇摆从源头压掉：pitch 降到
    ±0.55°、roll ±1.13°，四腿占空比回到对称的 0.59，达成率 58% → 88%。
    保留此开关是为了后续接姿态 PD 时能平滑混合（需要先解决切换阶跃）。

    ⚠️ 无论如何**不能用在支撑腿上**：支撑腿的脚踩在地上，对它下发「保持世界
    高度」等于让机身去适应脚——机身前倾时反而把前腿收短、把机身压得更低，
    形成正反馈，实测 0.4 s 内就发散到 IK 无解。支撑腿的姿态修正是反号的
    （低的那侧要伸长），那是姿态 PD 的活，见 stand_balance.py。
    """

    def __init__(self, gait: TrotGait, level_feet: bool = False,
                 bob_comp: float = 0.0):
        self.gait = gait
        self.level_feet = level_feet
        self.bob_comp = bob_comp          # 竖直起伏前馈幅值 m（见 targets 与 BOB_PHASE）
        self.seed = {n: (0.0, 0.0) for n in LEG_NAMES}
        self.min_margin = np.inf

    def targets(self, t: float, R_body: np.ndarray | None = None) -> np.ndarray:
        """返回 12 维目标角，顺序与 ACT 一致。

        R_body: 机身姿态阵（世界←机身）。level_feet 打开时必须给。
        """
        q = np.zeros(12)
        M = None
        if self.level_feet and R_body is not None:
            M = R_body.T @ _yaw_matrix(R_body)

        # 竖直起伏前馈：trot 的机身起伏是**确定性的**、锁定在 2×步频（每半周期一次
        # 对角腿交替就颠一下）。这里按全局步态相位给全部足端一个同相 z 补偿，直接抵消
        # 那个颠簸。相位/幅值是在默认工况（周期 0.5、步幅 0.1、kp≈500）调出来的，
        # 换速度后不一定最优，故默认关，作专家旋钮。
        zff = 0.0
        if self.bob_comp:
            pg = (t / self.gait.period) % 1.0
            zff = self.bob_comp * np.sin(2 * 2 * np.pi * pg + BOB_PHASE)

        for i, name in enumerate(LEG_NAMES):
            leg = LEGS[name]
            foot = self.gait.foot_target(name, t)
            if zff:
                foot = foot.copy()
                foot[2] += zff
            if M is not None and not self.gait.is_stance(name, t):
                # 仅摆动腿：连挂载点一起转（目标是在 base 系里给的）
                p_base = np.asarray(leg.leg_pos) + foot
                foot = M @ p_base - np.asarray(leg.leg_pos)
            h, th, cr, info = leg_ik(foot, leg, frame="leg", seed=self.seed[name])
            if not info["converged"]:
                raise RuntimeError(
                    f"{name} 腿 t={t:.3f}s 足端 {np.round(foot, 4)} IK 不收敛"
                    f"（残差 {info['residual']:.2e} m）")
            self.seed[name] = (th, cr)
            self.min_margin = min(self.min_margin, info["margin"])
            q[i], q[4 + i], q[8 + i] = h, th, cr
        return q


def _cycle_feasible(gait: TrotGait, step_length: float, *, n: int = 30) -> bool:
    """给定步幅，整周期每腿 IK 是否都可解且离奇异位形够远。"""
    saved = gait.step_length
    gait.step_length = step_length
    try:
        for t in np.linspace(0.0, gait.period, n, endpoint=False):
            for name in LEG_NAMES:
                f = gait.foot_target(name, t)
                _, _, _, info = leg_ik(f, LEGS[name], frame="leg")
                if not info["converged"] or info["margin"] < 0.010:
                    return False
        return True
    finally:
        gait.step_length = saved


def _max_feasible_step(gait: TrotGait) -> float:
    """二分求当前周期/配平下最大可行步幅（工作域上限）。"""
    if _cycle_feasible(gait, gait.step_length):
        return gait.step_length
    lo, hi = 0.0, gait.step_length
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if _cycle_feasible(gait, mid):
            lo = mid
        else:
            hi = mid
    return lo


def preflight(gait: TrotGait, v_cmd: float) -> str | None:
    """开走前的可行性预检。可行返回 None；否则返回一段可读的诊断+建议。

    在启动 viewer / 落地之前调用——不可行的指令（如 --vx 太大导致步幅超出腿工作域）
    会在这里被拦下并给出办法，而不是等到主循环里 IK 崩、再连累 viewer 段错误。
    """
    if _cycle_feasible(gait, gait.step_length):
        return None
    max_step = _max_feasible_step(gait)
    max_vx = max_step / (gait.duty * gait.period)
    # 达到期望速度所需的周期（留 10% 余量）
    sug_period = 0.9 * max_step / (gait.duty * abs(v_cmd)) if abs(v_cmd) > 1e-6 else gait.period
    lines = [
        "❌ 步态不可行：足端目标超出腿工作域，无法开走（未启动仿真）。",
        f"   指令 vx {v_cmd:+.2f} m/s ⇒ 步幅 {gait.step_length:.3f} m"
        f"（半步 ±{gait.step_length / 2:.3f} m），已超过工作域上限。",
        f"   本周期({gait.period:.2f}s)下最大可行步幅 ≈ {max_step:.3f} m"
        f" ⇒ 最大 vx ≈ {max_vx:.2f} m/s。",
        "   办法（腿伸不了那么长，只能提高步频而非加大步幅）：",
        f"     ① 降速：--vx {max_vx:.2f} 或更低；",
        f"     ② 要更快就缩短周期：--period {sug_period:.2f}（配 --vx {v_cmd:.1f}）；",
        "     ③ 也可小步幅高步频组合：--step-length ≤ "
        f"{max_step:.2f} 并相应减小 --period。",
        "   注：即便运动学可解，>1.5 m/s 开环步态也可能因电机转速上限/动态失稳而摔"
        "（见 传统步态/代码/speed_map.py）。",
    ]
    return "\n".join(lines)


def build(scene: Path, kp: float, kv: float):
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 1.0 / SIM_HZ
    for i in range(model.nu):        # XML 的 kp=25 撑不住自重，见文首
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kv
    return model


def run(args) -> int:
    model = build(Path(args.scene), args.kp, args.kv)
    data = mujoco.MjData(model)

    ci = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + n)
                   for n in ACT])
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    qadr = np.array([model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ACT])

    gait = TrotGait(body_height=args.body_height, step_length=args.step_length,
                    step_height=args.step_height, period=args.period,
                    x_offset=args.x_offset, vy=args.vy, wz=args.wz)
    # 前进速度可直接给 --vx：无打滑下 v = 步幅/(占空·周期)，反推步幅。
    # 不给 --vx 时沿用 --step-length（向后兼容）。
    if args.vx is not None:
        gait.step_length = args.vx * gait.duty * gait.period
    v_cmd = gait.step_length / (gait.duty * gait.period)   # 无打滑理论前进速度
    ctrl = GaitController(gait, level_feet=args.level_feet, bob_comp=args.bob_comp)

    def R_body():
        return data.xmat[base_id].reshape(3, 3)

    if not args.quiet:
        vx_src = "指定" if args.vx is not None else "由步幅×步频推得"
        print("━" * 60)
        print("  mos2026_2 · MuJoCo 步态行走（无 RL，闭链 IK + 位置伺服）")
        print("━" * 60)
        print(f"  指令   vx {v_cmd:+.2f} m/s（{vx_src}）"
              f"   vy {gait.vy:+.2f} m/s   wz {gait.wz:+.2f} rad/s")
        print(f"  步态   步幅 {gait.step_length:.3f} m   步高 {gait.step_height:.3f} m"
              f"   周期 {gait.period:.2f} s   占空 {gait.duty:.2f}   trot 对角")
        print(f"  姿态   body_height {gait.body_height:.3f} m"
              f"   x_offset {gait.x_offset:+.3f} m"
              f"   摆动型线 {gait.swing_profile}"
              f"   水平系摆动 {'开' if args.level_feet else '关'}")
        bob = f"   起伏前馈 {args.bob_comp * 1e3:.1f} mm" if args.bob_comp else ""
        print(f"  伺服   kp {args.kp:.0f}   kv {args.kv:.1f}"
              f"   仿真 {SIM_HZ} Hz / 控制 {CONTROL_HZ} Hz"
              f"   站立 {args.settle:.1f} s + 行走 {args.duration:.1f} s{bob}")
        print("━" * 60)

    # --- 可行性预检：不可行就在启动 viewer/落地之前拦下，避免主循环崩 + viewer 段错误 ---
    msg = preflight(gait, v_cmd)
    if msg is not None:
        print(msg)
        return 2

    # --- 先站起来再走：把足端从初始位形平滑压到名义站姿 ---
    data.qpos[2] = args.spawn_height
    mujoco.mj_forward(model, data)   # 先算一次 xmat，否则姿态阵是全 0
    data.ctrl[ci] = ctrl.targets(0.0, R_body())
    for _ in range(int(args.settle * SIM_HZ)):
        mujoco.mj_step(model, data)

    if not args.quiet:
        up0 = float(data.xmat[base_id].reshape(3, 3)[2, 2])
        print(f"  站立完成：base 高度 {data.qpos[2]:.4f} m   upright {up0:.3f}"
              f"   → 开始行走 {args.duration:.0f}s …")

    def upright():
        """base 局部 z 轴在世界系的 z 分量：1=直立，0=侧翻 90°。"""
        return float(data.xmat[base_id].reshape(3, 3)[2, 2])

    def world_yaw():
        w, x, y, z = data.qpos[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))

    # 机体系速度必须逐步积分：转向后世界系净位移不代表机体前进量。
    body_vel_sum = np.zeros(2)
    yaw_total = 0.0
    prev_yaw = world_yaw()
    t_gait = 0.0
    steps_per_ctrl = SIM_HZ // CONTROL_HZ
    n_steps = int(args.duration * SIM_HZ)
    z_hist = []

    viewer = None
    if args.viewer:
        # 注意用 from-import：`import mujoco.viewer` 会把 mujoco 变成本函数的局部名
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    try:
        for k in range(n_steps):
            if k % steps_per_ctrl == 0:
                t_gait += steps_per_ctrl / SIM_HZ
                try:
                    data.ctrl[ci] = ctrl.targets(t_gait, R_body())
                except RuntimeError as e:
                    # 预检过了仍中途 IK 崩（通常是被扰动推出工作域）——干净退出，
                    # 不让异常穿过 viewer 上下文（会段错误）。
                    print(f"❌ t={data.time:.2f}s 足端跟踪失败，停机：{e}")
                    return 5
            mujoco.mj_step(model, data)

            yaw = world_yaw()
            yaw_total += np.arctan2(np.sin(yaw - prev_yaw), np.cos(yaw - prev_yaw))
            prev_yaw = yaw
            c, s = np.cos(yaw), np.sin(yaw)
            vw = data.qvel[0:2]
            body_vel_sum += np.array([c * vw[0] + s * vw[1],
                                      -s * vw[0] + c * vw[1]]) / SIM_HZ
            z_hist.append(data.qpos[2])

            # 摔倒判据用姿态而非绝对高度（爬坡时高度本就会变）
            up = upright()
            if up < 0.5 and not args.no_fall_stop:
                print(f"❌ t={data.time:.2f}s 摔倒"
                      f"（机身倾角 {np.degrees(np.arccos(np.clip(up, -1, 1))):.0f}°）")
                return 3

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(1.0 / SIM_HZ)
    finally:
        if viewer is not None:
            viewer.close()

    dur = data.time - args.settle
    vx, vy = body_vel_sum / dur
    z = np.array(z_hist)   # v_cmd 已在配置横幅处算出（无打滑理论前进速度）

    def rate(actual, cmd):
        """达成率；指令为 0 时百分比无意义，只报绝对残差。"""
        return f"达成率 {100 * actual / cmd:.0f}%" if abs(cmd) > 1e-9 else "指令 0"

    if not args.quiet:
        print(f"✅ 站住并走完 {dur:.1f}s（机体系平均速度）")
        print(f"   vx {vx:+.3f} m/s（无打滑理论值 {v_cmd:+.3f}，{rate(vx, v_cmd)}）")
        print(f"   vy {vy:+.3f} m/s（指令 {gait.vy:+.3f}，{rate(vy, gait.vy)}）")
        print(f"   wz {yaw_total / dur:+.3f} rad/s（指令 {gait.wz:+.3f}，"
              f"{rate(yaw_total / dur, gait.wz)}；累计转过 {np.degrees(yaw_total):+.1f}°）")
        print(f"   base 高度 {z.mean():.4f} ± {z.std() * 1000:.1f} mm"
              f"   起伏峰峰 {(z.max() - z.min()) * 1000:.1f} mm"
              f"（末值 {data.qpos[2]:.4f}）")
        print(f"   姿态 upright = {upright():.4f}")
        print(f"   步态最小奇异余量 {ctrl.min_margin * 1e3:.1f} mm"
              f"（<10 mm 有翻分支风险）")

    # 翻分支检测：用实际关节角复算余量，与规划值比对
    q_act = data.qpos[qadr]
    m_act = min(solve_linkage(q_act[4 + i], q_act[8 + i], LEGS[n])["margin"]
                for i, n in enumerate(LEG_NAMES))
    if not args.quiet:
        print(f"   实际位形最小余量 {m_act * 1e3:.1f} mm")
    if m_act < 0.005:
        print("❌ 实际位形贴近奇异位形，可能已翻装配分支")
        return 4

    return 0


def sweep(args) -> int:
    """验收：扫一组步幅/周期，检查速度达成率、站高稳定性、不摔。"""
    # (step_length, period, wz, 说明)
    # 达成率门槛 70%：参照实现 tools/learn/go2_walk.py 在 Go2 上同口径实测 82%，
    # 开环步态的足端打滑是固有的，不是本实现的缺陷。
    cases = [
        (0.06, 0.60, 0.0, "慢走"),
        (0.10, 0.50, 0.0, "默认"),
        (0.14, 0.45, 0.0, "中速"),
        (0.16, 0.40, 0.0, "快走"),
        (0.10, 0.50, 0.3, "走+左转"),
        (0.10, 0.50, -0.3, "走+右转"),
    ]
    print(f"{'工况':<9}{'步幅':>6}{'周期':>6}{'wz':>6}{'理论v':>7}{'实测v':>7}"
          f"{'达成率':>7}{'实测wz':>8}{'站高':>8}{'余量':>7}  结果")
    print("-" * 88)
    ok_all = True
    for L, T, WZ, label in cases:
        a = argparse.Namespace(**vars(args))
        a.step_length, a.period, a.quiet, a.sweep, a.viewer = L, T, True, False, False
        model = build(Path(a.scene), a.kp, a.kv)
        data = mujoco.MjData(model)
        ci = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + n)
                       for n in ACT])
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

        gait = TrotGait(body_height=a.body_height, step_length=L,
                        step_height=a.step_height, period=T, x_offset=a.x_offset,
                        wz=WZ)
        ctrl = GaitController(gait, level_feet=a.level_feet, bob_comp=a.bob_comp)
        data.qpos[2] = a.spawn_height
        mujoco.mj_forward(model, data)   # 先算一次 xmat，否则姿态阵是全 0
        data.ctrl[ci] = ctrl.targets(0.0, data.xmat[base_id].reshape(3, 3))
        for _ in range(int(a.settle * SIM_HZ)):
            mujoco.mj_step(model, data)

        body_vel_sum = np.zeros(2)
        yaw_tot = 0.0
        prev_yaw = 0.0
        t_gait = 0.0
        spc = SIM_HZ // CONTROL_HZ
        z_hist = []
        fell = False
        for k in range(int(a.duration * SIM_HZ)):
            if k % spc == 0:
                t_gait += spc / SIM_HZ
                data.ctrl[ci] = ctrl.targets(t_gait, data.xmat[base_id].reshape(3, 3))
            mujoco.mj_step(model, data)
            w, x, y, z_ = data.qpos[3:7]
            yaw = np.arctan2(2 * (w * z_ + x * y), 1 - 2 * (y ** 2 + z_ ** 2))
            yaw_tot += np.arctan2(np.sin(yaw - prev_yaw), np.cos(yaw - prev_yaw))
            prev_yaw = yaw
            c, s = np.cos(yaw), np.sin(yaw)
            vw = data.qvel[0:2]
            body_vel_sum += np.array([c * vw[0] + s * vw[1],
                                      -s * vw[0] + c * vw[1]]) / SIM_HZ
            z_hist.append(data.qpos[2])
            if data.xmat[base_id].reshape(3, 3)[2, 2] < 0.5:
                fell = True
                break

        dur = data.time - a.settle
        vx = body_vel_sum[0] / dur
        wz_act = yaw_tot / dur
        v_th = L / (gait.duty * T)
        rate = 100 * vx / v_th
        zz = np.array(z_hist)
        wz_ok = abs(wz_act - WZ) < 0.15
        good = (not fell) and rate > 70 and zz.std() < 0.02 and wz_ok
        ok_all &= good
        print(f"{label:<9}{L:>6.2f}{T:>6.2f}{WZ:>6.1f}{v_th:>7.2f}{vx:>7.2f}{rate:>6.0f}%"
              f"{wz_act:>8.2f}{zz.mean():>7.3f}m{ctrl.min_margin * 1e3:>6.0f}mm  "
              f"{'❌摔倒' if fell else ('✅' if good else '⚠️')}")

    print("-" * 88)
    print("✅ 全部工况通过" if ok_all else "❌ 有工况未达标")
    return 0 if ok_all else 1


def main() -> int:
    p = argparse.ArgumentParser(description="mos2026_2 MuJoCo 步态行走（无 RL）")
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    p.add_argument("--viewer", action="store_true", help="开窗可视化")
    p.add_argument("--headless", action="store_true", help="无窗（默认）")
    p.add_argument("--sweep", action="store_true", help="扫多组步态参数做验收")
    p.add_argument("-T", "--duration", type=float, default=10.0, help="行走时长 s")
    p.add_argument("--settle", type=float, default=1.0, help="起步前站立稳定时长 s")
    p.add_argument("--spawn-height", type=float, default=0.30)
    p.add_argument("--body-height", type=float, default=0.26)
    p.add_argument("--step-length", type=float, default=0.10)
    p.add_argument("--step-height", type=float, default=0.04)
    p.add_argument("--period", type=float, default=0.5)
    p.add_argument("--vx", type=float, default=None,
                   help="前进速度 m/s。给了就按 步幅=vx·占空·周期 反推步幅（覆盖 "
                        "--step-length）；不给则用 --step-length。默认步幅 0.10/周期 0.5 ⇒ 0.4 m/s")
    p.add_argument("--vy", type=float, default=0.0, help="侧移速度 m/s（左为正）")
    p.add_argument("--wz", type=float, default=0.0, help="转向角速度 rad/s（左转为正）")
    p.add_argument("--kp", type=float, default=500.0,
                   help="关节侧位置增益。默认 500 抑制机身起伏（峰峰 20→14 mm）；"
                        "要与真机部署增益一致做 sim2real 保真，用 --kp 320 --kv 3.2")
    p.add_argument("--kv", type=float, default=5.0)
    p.add_argument("--bob-comp", type=float, default=0.0,
                   help="竖直起伏前馈幅值 m（默认 0=关）。默认工况下 0.002 可把峰峰"
                        "再压到 ~10 mm；换速度后需重调，属专家旋钮")
    p.add_argument("--no-fall-stop", action="store_true")
    p.add_argument("--x-offset", type=float, default=-0.055,
                   help="站姿前后配平（足端整体后移）。CoM 在支撑中心后 23.6 mm，"
                        "但实测最优在 −55~−60 mm：除静态配平外还要压住 trot 的点头振荡")
    p.add_argument("--level-feet", action="store_true",
                   help="实验：摆动腿走重力对齐水平系（实测反而更差，见 GaitController 文档）")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    return sweep(args) if args.sweep else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
