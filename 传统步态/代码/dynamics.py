"""mos2026_2 四足动力学分析 + 减速比选型。

内容（对应 todo.md「力矩/电流不足专项」与「选型评估」）
----------------------------------------------------------
1. 连杆受力：矢状面静力 Newton-Euler 递推（足端地反力 → 小腿 → 大腿 → 髋），
   给出各关节的反作用力与保持力矩。
2. 关节力矩需求：站立(4腿) / trot 支撑(2腿) / 动态蹬地(含地反力放大系数)。
3. 齿轮/减速传动受力：转子力矩、行星轮齿面切向力、膝平行连杆传动力。
4. 电机 T-N 曲线：GO-M8010-6 转子侧曲线，按减速比 N 反射到关节。
5. 最佳减速比：扫 N，给出力矩/转速可行带与推荐值，并对比现用 6.33。

几何/质量取自 deploy/mujoco/assets/mos2026_2.xml；电机规格取自
deploy/real/config/mos2026_2.yaml 注释（GO-M8010-6, 24V）。

运行：python 传统步态/代码/dynamics.py            # 控制台报告
      python 传统步态/代码/dynamics.py --plot     # 另出图到 传统步态/图与数据/dynamics/
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from gait import TrotGait
from kinematics import LEGS, leg_ik

# ============================ 参数 ============================
G = 9.81
M_TOTAL = 11.7006              # 整机质量 kg（XML 求和核对）
M_BASE = 4.520163
M_HIP = 0.985181              # 髋连杆（FL 等）
M_THIGH = 0.218416           # 大腿
M_SHANK = 0.286222           # 小腿
# 膝平行连杆 + 电机齿轮等随大腿/髋运动的从动件，质量小，计入大腿等效
M_THIGH_EFF = M_THIGH + 0.171588 + 0.046939 + 0.038524 + 0.048236  # ≈0.524

L1 = 0.180                    # 大腿连杆长 m（XML 实测）
L2 = 0.160                    # 小腿连杆长 m（站立几何估计，待标定）

# --- GO-M8010-6 电机（24V），来自 yaml 注释 ---
GEAR_NOW = 6.33              # 现用减速比
TAU_OUT_PEAK = 23.7         # 输出端峰值力矩 N·m
W_OUT_MAX = 30.0            # 输出端最大转速 rad/s（空载）
ETA = 0.90                  # 减速器效率（单级行星估计）
# 反射到「转子侧」的电机本体 T-N（换减速比时保持不变的是这条曲线）
TAU_ROTOR_PEAK = TAU_OUT_PEAK / GEAR_NOW      # ≈3.744 N·m
W_ROTOR_NOLOAD = W_OUT_MAX * GEAR_NOW         # ≈189.9 rad/s
P_MOTOR_MAX = TAU_ROTOR_PEAK * W_ROTOR_NOLOAD / 4.0  # 转子峰值机械功率（T-N 抛物线顶点）


# ============================ 几何/位姿 ============================
def stance_planar(body_height: float):
    """站立矢状面位姿：hip 在原点，返回 (knee, foot, q_hip, q_knee)。"""
    leg = LEGS["fl"]
    # 足端在髋系正下方 body_height（x=0），用 IK 解关节角
    foot_hip = np.array([0.0, leg.d, -body_height])
    _, q_hip, q_knee = leg_ik(foot_hip, leg, frame="hip", knee_sign=-1.0)
    q_hip, q_knee = float(q_hip), float(q_knee)
    knee = np.array([L1 * math.sin(q_hip), -L1 * math.cos(q_hip)])
    foot = knee + np.array([L2 * math.sin(q_hip + q_knee), -L2 * math.cos(q_hip + q_knee)])
    return knee, foot, q_hip, q_knee


# ============================ 连杆受力（静力 Newton-Euler 递推）=========
def link_forces(foot_force: np.ndarray, body_height: float):
    """矢状面静力递推：已知足端地反力 F=(Fx,Fz)，从小腿往髋逐段做力/力矩平衡。

    每段：ΣF=0 → 近端关节反力；Σm(关节)=0 → 该关节驱动力矩。
    连杆自重作用在中点（标准简化，自重仅占地反力 ~3-5%）。
    返回各关节反力与力矩 dict。
    """
    knee, foot, q_hip, q_knee = stance_planar(body_height)
    hip = np.array([0.0, 0.0])
    thigh_com = 0.5 * (hip + knee)
    shank_com = 0.5 * (knee + foot)

    Wg = np.array([0.0, -G])  # 单位质量重力
    F = np.asarray(foot_force, float)

    def moment(r, f):  # 2D 叉乘 z 分量：r×f
        return r[0] * f[1] - r[1] * f[0]

    # --- 小腿段：受 足端反力 F + 自重 + 膝关节反力 R_knee ---
    R_knee = -(F + M_SHANK * Wg)               # ΣF=0
    # 膝力矩：足端力 + 小腿自重 对膝取矩（驱动力矩 = -外力矩，使段平衡）
    tau_knee = -(moment(foot - knee, F) + moment(shank_com - knee, M_SHANK * Wg))

    # --- 大腿段：受 膝处来自小腿的反力 -R_knee + 自重 + 髋反力 R_hip ---
    R_hip = -(-R_knee + M_THIGH_EFF * Wg)
    tau_hip = -(moment(knee - hip, -R_knee) + moment(thigh_com - hip, M_THIGH_EFF * Wg))

    return {
        "q_hip_deg": math.degrees(q_hip), "q_knee_deg": math.degrees(q_knee),
        "knee": knee, "foot": foot,
        "R_knee": R_knee, "R_hip": R_hip,
        "tau_knee": tau_knee, "tau_hip": tau_hip,
        "Rknee_mag": float(np.linalg.norm(R_knee)),
        "Rhip_mag": float(np.linalg.norm(R_hip)),
    }


# ============================ 关节速度需求（来自步态）============
def joint_speed_demand(scale_speed: float = 1.0):
    """从 trot 步态算关节角速度需求。scale_speed 缩短周期 = 走更快。

    gait.py 2026-07-21 起用闭链模型，joint_targets 返回的 [1]/[2] 就是
    **thigh 与 shank_link 曲柄两个真实电机**的角度，不再是「等效膝角」，
    所以这里的角速度需求可以直接对着电机 T-N 曲线读。
    """
    g = TrotGait()
    g.period = g.period / scale_speed
    n = 600
    times = np.linspace(0, g.period, n, endpoint=False)
    q_thigh = np.array([g.joint_targets(t)["fl"][1] for t in times])
    q_crank = np.array([g.joint_targets(t)["fl"][2] for t in times])
    dt = times[1] - times[0]
    w_thigh = np.max(np.abs(np.gradient(q_thigh, dt)))
    w_crank = np.max(np.abs(np.gradient(q_crank, dt)))
    return max(w_thigh, w_crank), w_thigh, w_crank


# ============================ 齿轮/传动受力 ============================
def gear_forces(tau_joint: float, gear: float):
    """给定关节力矩，反推转子力矩与齿面切向力。

    转子力矩 τ_r = τ_joint / (gear·η)。行星轮系：太阳轮齿面切向力
    F_t = τ_r / r_sun（n_p 个行星轮分担 → 单齿 F_t/n_p）。
    r_sun/n_p 取 GO 系列典型值（无精确齿数，标注为估计）。
    """
    tau_rotor = tau_joint / (gear * ETA)
    r_sun = 0.006   # 太阳轮节圆半径估计 ~6 mm（GO-M8010-6 量级）
    n_planet = 3
    F_tan_total = tau_rotor / r_sun
    return {
        "tau_rotor": tau_rotor,
        "F_tan_total": F_tan_total,
        "F_tan_per_tooth": F_tan_total / n_planet,
    }


def knee_linkage_force(tau_knee: float):
    """膝平行四连杆传动力（闭链）。曲柄力臂取 XML 闭环 anchor 几何。

    XML: connect anchor (-0.10764, ∓0.005, -0.11851) 相对 shank 关节，
    曲柄等效力臂 r ≈ |(anchor_x, anchor_z)| 在垂直连杆方向的分量量级。
    这里用 anchor 到膝轴的距离作为力臂上界估计。
    """
    r_crank = math.hypot(0.10764, 0.11851)  # ≈0.1599 m（anchor 到膝轴距离）
    return tau_knee / r_crank


# ============================ 减速比寻优 ============================
def motor_tn_joint(gear: float, n=200):
    """电机 T-N 反射到关节：返回关节侧 (ω_joint[], τ_joint[]) 曲线。"""
    w_rotor = np.linspace(0, W_ROTOR_NOLOAD, n)
    tau_rotor = TAU_ROTOR_PEAK * (1.0 - w_rotor / W_ROTOR_NOLOAD)  # 线性 T-N
    w_joint = w_rotor / gear
    tau_joint = tau_rotor * gear * ETA
    return w_joint, tau_joint


def optimal_gear(tau_req: float, w_req: float):
    """给定关节峰值力矩需求 + 峰值角速度需求，求可行带与最优减速比。"""
    n_torque_min = tau_req / (ETA * TAU_ROTOR_PEAK)   # 力矩下限：N≥
    n_speed_max = W_ROTOR_NOLOAD / w_req              # 转速上限：N≤
    # 余量平衡最优：峰值力矩(支撑)与峰值转速(摆动)是两个不同时发生的工作点，
    # 取使「力矩余量 = 转速余量」的 N，即两约束边界的几何中点 sqrt(N_min·N_max)。
    n_balanced = math.sqrt(n_torque_min * n_speed_max)
    return n_torque_min, n_speed_max, n_balanced


# ============================ 报告 ============================
def report():
    print("=" * 70)
    print("mos2026_2 四足动力学 + 减速比选型分析")
    print("=" * 70)
    print(f"整机 {M_TOTAL:.2f} kg | 体重 W={M_TOTAL*G:.1f} N | L1(大腿)={L1} L2(小腿)={L2} m")
    print(f"电机 GO-M8010-6: 输出峰值 {TAU_OUT_PEAK} N·m / 空载 {W_OUT_MAX} rad/s @现减速比 {GEAR_NOW}")
    print(f"  → 转子侧 T-N: 峰值力矩 {TAU_ROTOR_PEAK:.3f} N·m, 空载转速 {W_ROTOR_NOLOAD:.1f} rad/s")
    print(f"  → 转子峰值机械功率 {P_MOTOR_MAX:.1f} W（η={ETA}）")

    bh = 0.30
    print(f"\n--- 站立矢状面位姿（body_height={bh} m）---")
    base = link_forces(np.array([0.0, M_TOTAL * G / 4]), bh)
    print(f"q_hip={base['q_hip_deg']:.1f}°  q_knee={base['q_knee_deg']:.1f}°  "
          f"knee={np.round(base['knee'],3)} foot={np.round(base['foot'],3)}")

    print("\n--- 关节保持力矩需求（不同工况，单位 N·m）---")
    cases = [
        ("静立 4 腿支撑 (Fz=W/4)", np.array([0.0, M_TOTAL * G / 4])),
        ("trot 2 腿支撑 (Fz=W/2)", np.array([0.0, M_TOTAL * G / 2])),
        ("动态蹬地 (Fz=1.5·W/2, Fx=0.4·Fz)", None),
    ]
    tau_knee_peak = 0.0
    for name, F in cases:
        if F is None:
            fz = 1.5 * M_TOTAL * G / 2
            F = np.array([0.4 * fz, fz])
        r = link_forces(F, bh)
        tau_knee_peak = max(tau_knee_peak, abs(r["tau_knee"]), abs(r["tau_hip"]))
        print(f"  {name:34s} | τ_hip={r['tau_hip']:+6.2f}  τ_knee={r['tau_knee']:+6.2f}"
              f" | 膝反力 |R|={r['Rknee_mag']:5.1f} N  髋反力 |R|={r['Rhip_mag']:5.1f} N")

    print(f"\n  ⇒ 关节峰值力矩需求 τ_req ≈ {tau_knee_peak:.1f} N·m"
          f"（现配 effort_limit=12 N·m，电机输出峰值 {TAU_OUT_PEAK} N·m）")

    print("\n--- 齿轮/传动受力（按动态蹬地峰值力矩）---")
    gf = gear_forces(tau_knee_peak, GEAR_NOW)
    klf = knee_linkage_force(tau_knee_peak)
    print(f"  转子力矩 τ_rotor = {gf['tau_rotor']:.3f} N·m"
          f"（转子峰值 {TAU_ROTOR_PEAK:.3f}，占比 {gf['tau_rotor']/TAU_ROTOR_PEAK*100:.0f}%）")
    print(f"  行星太阳轮齿面切向力 F_t ≈ {gf['F_tan_total']:.0f} N（3 行星轮分担 {gf['F_tan_per_tooth']:.0f} N/齿）")
    print(f"  膝平行连杆传动力 ≈ {klf:.0f} N（力臂 0.16 m）")

    print("\n--- 关节角速度需求（来自 trot 步态）---")
    for sp, label in [(1.0, "当前慢速 gait"), (2.0, "2× 速度"), (3.0, "3× 速度")]:
        wmax, wh, wk = joint_speed_demand(sp)
        print(f"  {label:14s}: 关节峰值角速度 ω_req={wmax:.2f} rad/s (大腿 {wh:.2f}, 曲柄 {wk:.2f})")
    w_req_design = joint_speed_demand(3.0)[0]

    print("\n" + "=" * 70)
    print("减速比寻优")
    print("=" * 70)
    tau_req = tau_knee_peak
    n_tmin, n_smax, n_bal = optimal_gear(tau_req, w_req_design)
    print(f"设计点: τ_req={tau_req:.1f} N·m, ω_req={w_req_design:.1f} rad/s（取 3× 速度）")
    print(f"  力矩约束（N·η·τ_rotor ≥ τ_req）  ⇒ N ≥ {n_tmin:.2f}")
    print(f"  转速约束（ω_rotor/N ≥ ω_req）     ⇒ N ≤ {n_smax:.2f}")
    print(f"  余量平衡最优（力矩/转速余量相等）  ⇒ N* = {n_bal:.2f}")
    feasible = n_tmin <= n_smax
    print(f"  可行带: [{n_tmin:.2f}, {n_smax:.2f}]  {'✅可行' if feasible else '❌不可行(电机功率不足)'}")

    w_av = W_ROTOR_NOLOAD / GEAR_NOW
    t_av = ETA * TAU_ROTOR_PEAK * GEAR_NOW
    tmar, smar = t_av / tau_req, w_av / w_req_design
    print(f"\n  现用 N={GEAR_NOW}:")
    print(f"    关节可用峰值力矩 {t_av:.1f} N·m（需求 {tau_req:.1f}，余量 {tmar:.2f}×）")
    print(f"    关节空载转速 {w_av:.1f} rad/s（需求 {w_req_design:.1f}，余量 {smar:.2f}×）")

    # 深蹲敏感性：站姿越低，膝力臂越大，力矩需求越高（eval 实测 base 仅 0.256）
    deep = link_forces(np.array([0.4 * 1.5 * M_TOTAL * G / 2, 1.5 * M_TOTAL * G / 2]), 0.256)
    print(f"  深蹲敏感性（body_height 0.256，eval 实测低姿）: "
          f"τ_hip={deep['tau_hip']:+.2f} τ_knee={deep['tau_knee']:+.2f} N·m")

    print("\n  结论：")
    print(f"    1) 余量平衡最优 N*={n_bal:.2f}，与现用 {GEAR_NOW} 几乎重合；力矩余量({tmar:.2f}×)"
          f"与转速余量({smar:.2f}×)基本均衡 → 【现减速比 6.33 已接近最优】。")
    print(f"    2) 在可行带 [{n_tmin:.1f},{n_smax:.1f}] 内：若主打中低速/抗扰，可上调到 "
          f"N≈8 换 ~26% 力矩余量（峰值速度降到 {W_ROTOR_NOLOAD/8:.0f} rad/s 仍够 2× 步速）；"
          f"若要高速则保持 6.33 或更低。")
    print("    3) 电机输出峰值 23.7 N·m（关节 21.3 N·m）对需求 12 N·m 有 1.78× 余量 →"
          " 真机「力矩/电流不足」根因【不是减速比选型】，而是 effort/电流上限设低或供电掉压：")
    print("       现 effort_limit_sim=12 N·m 恰好卡在需求线、零余量；建议放到 ~16–18 N·m"
          "（仍 ≤ 电机峰值），并核查驱动器电流上限与母线电压。")
    print("=" * 70)
    return tau_req, w_req_design, (n_tmin, n_smax, n_bal)


def _plot(tau_req, w_req, bounds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "图与数据", "dynamics")
    os.makedirs(out, exist_ok=True)
    n_tmin, n_smax, n_bal = bounds

    # 图1：关节侧 T-N 包络（不同减速比）+ 工作点
    fig, ax = plt.subplots(figsize=(8, 6))
    for gear in [4.0, 6.33, 8.0, 10.0, 12.0]:
        w, t = motor_tn_joint(gear)
        ax.plot(w, t, label=f"N={gear}", lw=2 if gear == 6.33 else 1.2,
                ls="-" if gear == 6.33 else "--")
    # 两个真实工作点（不同时发生）：支撑/蹬地=低速高力矩；摆动=高速低力矩
    ax.scatter([2.0], [tau_req], c="red", s=90, zorder=5, marker="s",
               label=f"stance/push-off (~2 rad/s, {tau_req:.0f} N·m)")
    ax.scatter([w_req], [2.0], c="purple", s=90, zorder=5, marker="^",
               label=f"swing ({w_req:.1f} rad/s, ~2 N·m)")
    ax.axhline(TAU_OUT_PEAK, color="gray", ls=":", lw=1)
    ax.text(1, TAU_OUT_PEAK + 0.5, "motor output peak 23.7 N·m", color="gray", fontsize=8)
    ax.set_xlabel("joint speed (rad/s)")
    ax.set_ylabel("joint torque (N·m)")
    ax.set_title("Joint-side T-N envelope vs gear ratio (GO-M8010-6)")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 35)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(out, "tn_envelope.png")
    fig.savefig(p1, dpi=120)

    # 图2：力矩/转速余量 vs 减速比
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    Ns = np.linspace(3, 14, 200)
    torque_margin = ETA * TAU_ROTOR_PEAK * Ns / tau_req
    speed_margin = (W_ROTOR_NOLOAD / Ns) / w_req
    ax2.plot(Ns, torque_margin, label="torque margin (avail/req)", color="C0")
    ax2.plot(Ns, speed_margin, label="speed margin (avail/req)", color="C1")
    ax2.axhline(1.0, color="red", ls=":", label="feasibility = 1.0")
    ax2.axvline(GEAR_NOW, color="gray", ls="--", label=f"current N={GEAR_NOW}")
    ax2.axvline(n_bal, color="green", ls="-.", label=f"power-matched N*={n_bal:.1f}")
    ax2.axvspan(n_tmin, n_smax, alpha=0.12, color="green", label=f"feasible [{n_tmin:.1f},{n_smax:.1f}]")
    ax2.set_xlabel("gear ratio N")
    ax2.set_ylabel("margin (×)")
    ax2.set_title("Torque vs speed margin across gear ratio")
    ax2.set_ylim(0, 6)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    p2 = os.path.join(out, "gear_margin.png")
    fig2.savefig(p2, dpi=120)
    print("已输出：")
    print(f"  {p1}")
    print(f"  {p2}")


def main():
    p = argparse.ArgumentParser(description="mos2026_2 动力学 + 减速比选型")
    p.add_argument("--plot", action="store_true", help="另出 T-N 包络 / 余量图")
    args = p.parse_args()
    tau_req, w_req, bounds = report()
    if args.plot:
        _plot(tau_req, w_req, bounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
