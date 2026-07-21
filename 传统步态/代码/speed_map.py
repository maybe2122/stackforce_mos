"""mos2026_2 机身移动速度 ↔ 关节/电机角速度 解算器。

回答的问题
----------
「四足要走 3 m/s，每个驱动电机得转多快？」——以及反过来「电机转速上限决定了能走
多快？」。结论先行：**这不是唯一对应**。同一个机身速度，可以用「大步幅 + 低步频」
（电机慢）或「小步幅 + 高步频」（电机快）实现。所以答案是一张映射/一条曲线，步幅、
占空比、步频是旋钮。本模块把这套关系严格算出来并出图。

物理链条（全部可在 --selftest 里自洽验证）
------------------------------------------
1. 无打滑约束：支撑相足端贴地不动，机身前进的速度 = 足端在机体系里向后扫的速度。
   对 `gait.TrotGait` 的支撑相直线扫掠，得
        v_body = step_length / (duty · period)                         …(1)
   给定目标 v，固定 step_length/duty，反解 period（= 1/步频）。
2. 足端速度剖面：对步态足端轨迹（支撑相贴地直线 + 摆动相摆线/(1−cos)）解析求导，
   得腿系足端速度 foot_vel(t)。摆动相峰值前向足速 = 2·v_body·β/(1−β)（β=0.5 时 = 2v）。
3. 足端速度 → 关节角速度：经解析 Jacobian
        q_dot(t) = J(q(t))⁻¹ · foot_vel(t)                            …(2)
   （`closed_chain_kin.leg_jacobian`）。
4. 关节 → 电机：直驱 + 行星减速，电机轴速 = 关节角速 × 减速比 N。
        ω_motor = N · q_dot,  N = 6.33                                …(3)

速度上限（两条参考线）
----------------------
- 电机物理上限：关节侧空载 30 rad/s（GO-M8010-6 @24V，见 dynamics.py / 电机 yaml）。
- 训练 sim cap：`velocity_limit_sim = 15 rad/s`（env_cfg.py，sim2real 友好的软限）。
峰值关节角速度超过它即「该步态在该速度不可行」，需要加大步幅/降步频。

闭链传动（2026-07-21 起已解决）：本模块改用 `closed_chain_kin`，Jacobian 直接建在
真实四杆机构上，输出的第三列就是 **shank_link 曲柄电机**自己的角速度，不再是
「等效膝角」加一个未标定的传动比。三个关节都是直驱，映射干净。

运行：
  python 传统步态/代码/speed_map.py --speed 3.0      # 单点：3 m/s 需要多少电机转速 + 可行性
  python 传统步态/代码/speed_map.py --speed 1.0 --step-length 0.18
  python 传统步态/代码/speed_map.py --sweep          # 机身速度→电机转速 曲线 + CSV → 传统步态/图与数据/speed_map/
  python 传统步态/代码/speed_map.py --selftest
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np

from closed_chain_kin import LEG_NAMES, LEGS, leg_jacobian, leg_ik
from gait import TrotGait

# --- 传动 / 限位常量（取自 传统步态/代码/dynamics.py 与 env_cfg.py）---------------
GEAR = 6.33                 # 减速比：ω_motor = GEAR · ω_joint
JOINT_VEL_TRAIN = 15.0      # 训练 sim 软限 velocity_limit_sim (rad/s, 关节侧)
JOINT_VEL_PHYS = 30.0       # 电机物理空载上限 (rad/s, 关节侧)
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)

# 三个受驱关节（闭链模型直接输出 MJCF/电机角，无「约定角」中间层）
JOINT_LABELS = ("hip", "thigh", "crank(shank_link)")


# ============================ 步态构造 ============================
def gait_for_speed(
    v_body: float,
    *,
    step_length: float = 0.10,
    duty: float = 0.5,
    step_height: float = 0.04,
    body_height: float = 0.30,
    x_offset: float = 0.0,
) -> TrotGait:
    """按目标机身速度构造 trot 步态：由式(1) 反解 period = L/(duty·v)。

    固定步幅/占空比、用步频承载速度（高速 = 高步频）。v<=0 时给一个很慢的步态。
    """
    if v_body <= 0:
        period = 1e3
    else:
        period = step_length / (duty * v_body)
    return TrotGait(
        body_height=body_height,
        step_length=step_length,
        step_height=step_height,
        period=period,
        duty=duty,
        x_offset=x_offset,
    )


def body_speed_of(gait: TrotGait) -> float:
    """步态隐含的机身速度（式 1）。"""
    return gait.step_length / (gait.duty * gait.period)


# ============================ 足端速度（解析求导）============================
def foot_vel_hip(gait: TrotGait, leg_name: str, t: float) -> np.ndarray:
    """腿系足端速度 (vx, vy, vz) —— 对 gait.foot_target 解析求导。

    gait.foot_target 的水平部分是 (x, y) = (x0, y0) + (Lx, Ly)·f(相位)，
    x/y 共用同一个归一化型线 f，所以这里只要对 f 求一次导再乘步向量即可，
    转向/侧移（Ly≠0）自动带上。

    支撑相：f = 0.5 − s, s = p/β  →  df/dt = −1/(β·T)
            水平速度 = −(Lx,Ly)/(β·T)，前进时即 −v_body（足端向后=机身向前）。
    摆动相：随 swing_profile 取摆线或三次多项式的导数。
    竖直方向恒为 (1−cos) 抬腿的导数，与水平型线无关。
    """
    T, h, beta = gait.period, gait.step_height, gait.duty
    Lx, Ly = gait.step_vec(leg_name)
    p = (t / T + gait.phase_offset[leg_name]) % 1.0

    if p < beta:                      # 支撑相
        df_dt = -1.0 / (beta * T)
        vz = 0.0
    else:                             # 摆动相
        s = (p - beta) / (1.0 - beta)
        ds_dt = 1.0 / ((1.0 - beta) * T)
        if gait.swing_profile == "cycloid":
            # f = −1/2 + s − sin(2πs)/(2π)  → df/ds = 1 − cos(2πs)
            df_ds = 1.0 - math.cos(2 * math.pi * s)
        elif gait.swing_profile == "matched":
            # f = −1/2 − r·s + 3(1+r)s² − 2(1+r)s³ → df/ds = −r + 6(1+r)s − 6(1+r)s²
            r = (1.0 - beta) / beta
            df_ds = -r + 6.0 * (1.0 + r) * s - 6.0 * (1.0 + r) * s ** 2
        else:
            raise ValueError(f"未知 swing_profile: {gait.swing_profile!r}")
        df_dt = df_ds * ds_dt
        # z = z0 + h(1 − cos(2πs))/2  → dz/ds = h·π·sin(2πs)
        vz = h * math.pi * math.sin(2 * math.pi * s) * ds_dt

    return np.array([Lx * df_dt, Ly * df_dt, vz])


# ============================ 一个周期的运动学解算 ============================
@dataclass
class CycleResult:
    times: np.ndarray          # (n,)
    q: np.ndarray              # (n,4,3) 受驱角 [hip,thigh,crank]×[fl,fr,rl,rr]
    qd: np.ndarray             # (n,4,3) 关节角速度 rad/s
    foot: np.ndarray           # (n,4,3) 腿系足端
    foot_vel: np.ndarray       # (n,4,3) 腿系足端速度
    motor_omega: np.ndarray    # (n,4,3) 电机轴角速度 rad/s = qd·GEAR
    v_body: float
    gait: TrotGait


def cycle_kinematics(gait: TrotGait, n: int = 400) -> CycleResult:
    """覆盖一个完整步态周期，逐时刻解出每条腿每个关节的角速度。

    foot_vel 经解析求导，q_dot = J(q)⁻¹·foot_vel（解析 Jacobian），motor = q_dot·GEAR。
    """
    times = np.linspace(0.0, gait.period, n, endpoint=False)
    q = np.zeros((n, 4, 3))
    qd = np.zeros((n, 4, 3))
    foot = np.zeros((n, 4, 3))
    foot_vel = np.zeros((n, 4, 3))

    for ti, t in enumerate(times):
        for li, name in enumerate(LEG_NAMES):
            leg = LEGS[name]
            f = gait.foot_target(name, t)
            a, hh, k, _ = leg_ik(f, leg, frame="leg")
            fv = foot_vel_hip(gait, name, t)
            J = leg_jacobian(a, hh, k, leg)
            # q_dot = J⁻¹ fv（奇异位形 pinv 兜底）
            try:
                qdot = np.linalg.solve(J, fv)
            except np.linalg.LinAlgError:
                qdot = np.linalg.pinv(J) @ fv
            q[ti, li] = (a, hh, k)
            qd[ti, li] = qdot
            foot[ti, li] = f
            foot_vel[ti, li] = fv

    return CycleResult(
        times=times, q=q, qd=qd, foot=foot, foot_vel=foot_vel,
        motor_omega=qd * GEAR, v_body=body_speed_of(gait), gait=gait,
    )


def instant(gait: TrotGait, t: float) -> tuple[np.ndarray, np.ndarray]:
    """单时刻 4 腿受驱角与关节角速度（供实时可视化逐帧调用）。

    返回 q (4,3) 与 qd (4,3)，列 = (θ_hip, θ_thigh, θ_crank)，行 = (fl,fr,rl,rr)。
    电机轴角速度 = qd · GEAR。
    """
    q = np.zeros((4, 3))
    qd = np.zeros((4, 3))
    for li, name in enumerate(LEG_NAMES):
        leg = LEGS[name]
        f = gait.foot_target(name, t)
        a, hh, k, _ = leg_ik(f, leg, frame="leg")
        fv = foot_vel_hip(gait, name, t)
        J = leg_jacobian(a, hh, k, leg)
        try:
            qdot = np.linalg.solve(J, fv)
        except np.linalg.LinAlgError:
            qdot = np.linalg.pinv(J) @ fv
        q[li] = (a, hh, k)
        qd[li] = qdot
    return q, qd


# ============================ 统计 / 峰值 ============================
def summarize(res: CycleResult) -> dict:
    """每关节峰值/RMS 角速度（关节侧 rad/s），以及整机峰值与命中关节。"""
    absqd = np.abs(res.qd)                       # (n,4,3)
    peak = absqd.max(axis=0)                      # (4,3) 每关节峰值
    rms = np.sqrt((res.qd ** 2).mean(axis=0))     # (4,3)
    flat = peak.reshape(-1)
    idx = int(np.argmax(flat))
    li, ji = divmod(idx, 3)
    return {
        "peak_joint": peak,                       # rad/s
        "rms_joint": rms,
        "peak_overall": float(flat[idx]),          # rad/s
        "peak_leg": LEG_NAMES[li],
        "peak_jtype": JOINT_LABELS[ji],
        "peak_motor_rpm": float(flat[idx] * GEAR * RAD_S_TO_RPM),
        "feasible_train": bool(flat[idx] <= JOINT_VEL_TRAIN),
        "feasible_phys": bool(flat[idx] <= JOINT_VEL_PHYS),
    }


def best_stride_for(v_body: float, limit: float, *, duty=0.5, step_height=0.04,
                    body_height=0.30, n_scan=40):
    """在给定 v 下扫描步幅，找「峰值关节角速度 ≤ limit」的最小可行步幅。

    峰值随步幅是 U 形：步幅太小 → 步频高 → 峰值高；步幅太大 → 足端超出可达域、腿
    接近伸直奇异 → 峰值暴涨。故不能二分。可达域上界：足端在深度 body_height 处的
    最大水平半幅 = √((L1+L2)² − body_height²)，留 5% 余量。

    返回 (L_min_feasible, peak_there) 若有可行步幅；否则 (None, (L_best, peak_best))
    给出能做到的最低峰值及对应步幅。
    """
    reach = math.hypot(LEGS["fl"].l1 + LEGS["fl"].l2, 0.0)
    half_max = 0.95 * math.sqrt(max(reach ** 2 - body_height ** 2, 1e-6))
    L_hi = min(0.30, 2.0 * half_max)
    Ls = np.linspace(0.06, max(L_hi, 0.08), n_scan)

    def peak(L):
        g = gait_for_speed(v_body, step_length=L, duty=duty, step_height=step_height,
                           body_height=body_height)
        return summarize(cycle_kinematics(g, n=200))["peak_overall"]

    peaks = np.array([peak(L) for L in Ls])
    feasible = Ls[peaks <= limit]
    if len(feasible):
        Lmin = float(feasible.min())
        return Lmin, float(peak(Lmin))
    i = int(np.argmin(peaks))
    return None, (float(Ls[i]), float(peaks[i]))


# ============================ 单点查询 ============================
def query(v_body: float, *, step_length=0.10, duty=0.5, step_height=0.04,
          body_height=0.30, n=400, verbose=True) -> tuple[CycleResult, dict]:
    gait = gait_for_speed(v_body, step_length=step_length, duty=duty,
                          step_height=step_height, body_height=body_height)
    res = cycle_kinematics(gait, n=n)
    s = summarize(res)
    if verbose:
        f = 1.0 / gait.period
        print("=" * 68)
        print(f"目标机身速度 v = {v_body:.3f} m/s")
        print("=" * 68)
        print(f"实现步态：步幅 L={step_length*100:.1f} cm | 占空比 β={duty} | "
              f"周期 T={gait.period*1000:.1f} ms | 步频 f={f:.2f} Hz | 抬腿 {step_height*100:.0f} cm")
        print(f"  无打滑约束 v = L/(β·T) = {body_speed_of(gait):.3f} m/s (核对)")
        print(f"  摆动相峰值前向足速 ≈ 2·v·β/(1−β) = {2*v_body*duty/(1-duty):.2f} m/s")
        print()
        print("每关节峰值 / RMS 角速度（关节侧 rad/s；电机轴 = ×{:.2f}）：".format(GEAR))
        print(f"  {'leg':<4} {'hip':>22} {'thigh':>22} {'crank(shank_link)':>22}")
        for li, name in enumerate(LEG_NAMES):
            cells = []
            for ji in range(3):
                pk = s["peak_joint"][li, ji]
                cells.append(f"{pk:5.2f}/{s['rms_joint'][li,ji]:4.2f}({pk*GEAR*RAD_S_TO_RPM:4.0f}rpm)")
            print(f"  {name:<4} " + " ".join(f"{c:>22}" for c in cells))
        print(f"\n  ⇒ 峰值关节角速度 {s['peak_overall']:.2f} rad/s "
              f"@ {s['peak_leg'].upper()} {s['peak_jtype']}")
        print(f"     → 峰值电机轴速 {s['peak_overall']*GEAR:.1f} rad/s = "
              f"{s['peak_motor_rpm']:.0f} rpm")
        print()
        tr = "✅" if s["feasible_train"] else "❌"
        ph = "✅" if s["feasible_phys"] else "❌"
        print(f"  可行性： 训练软限 {JOINT_VEL_TRAIN} rad/s {tr}   "
              f"电机物理 {JOINT_VEL_PHYS} rad/s {ph}")
        if not s["feasible_train"] or not s["feasible_phys"]:
            for lab, lim in (("训练软限", JOINT_VEL_TRAIN), ("电机物理", JOINT_VEL_PHYS)):
                Lmin, info = best_stride_for(v_body, lim, duty=duty, step_height=step_height,
                                             body_height=body_height)
                if Lmin is not None:
                    print(f"  · 要 ≤ {lab} {lim} rad/s：步幅至少 {Lmin*100:.1f} cm"
                          f"（当前 {step_length*100:.1f} cm；步幅再大会超可达域、腿伸直奇异反而更糟）")
                else:
                    L_best, pk_best = info
                    print(f"  · 要 ≤ {lab} {lim} rad/s：trot(β={duty}) 下任意步幅都不可行；"
                          f"最低峰值 {pk_best:.1f} rad/s @ 步幅 {L_best*100:.1f} cm（需更大 β 或降速）")
        print("=" * 68)
    return res, s


# ============================ 速度扫描 + 出图 ============================
def _out_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "图与数据", "speed_map")
    os.makedirs(d, exist_ok=True)
    return d


def sweep(v_max=4.0, n_v=41, step_length=0.10, duty=0.5, step_height=0.04,
          body_height=0.30) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _out_dir()
    vs = np.linspace(0.05, v_max, n_v)
    peak_joint = np.zeros(n_v)
    rms_joint = np.zeros(n_v)
    for i, v in enumerate(vs):
        g = gait_for_speed(v, step_length=step_length, duty=duty,
                           step_height=step_height, body_height=body_height)
        res = cycle_kinematics(g, n=240)
        s = summarize(res)
        peak_joint[i] = s["peak_overall"]
        rms_joint[i] = np.sqrt((res.qd ** 2).mean())

    # 最大可行机身速度（峰值首次越过 15 / 30）
    def v_at(limit):
        over = np.where(peak_joint > limit)[0]
        return float(vs[over[0]]) if len(over) else None
    v_train, v_phys = v_at(JOINT_VEL_TRAIN), v_at(JOINT_VEL_PHYS)

    # --- 图1：峰值/RMS 关节角速度 vs 机身速度（左轴 rad/s，右轴电机 rpm）---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(vs, peak_joint, "C0-", lw=2, label="peak joint speed")
    ax.plot(vs, rms_joint, "C0--", lw=1.3, label="rms joint speed")
    ax.axhline(JOINT_VEL_TRAIN, color="orange", ls=":", lw=1.4, label=f"train cap {JOINT_VEL_TRAIN} rad/s")
    ax.axhline(JOINT_VEL_PHYS, color="red", ls=":", lw=1.4, label=f"motor max {JOINT_VEL_PHYS} rad/s")
    if v_train:
        ax.axvline(v_train, color="orange", ls="-", lw=0.8, alpha=0.6)
        ax.annotate(f"{v_train:.2f} m/s", (v_train, JOINT_VEL_TRAIN), color="orange", fontsize=8)
    if v_phys:
        ax.axvline(v_phys, color="red", ls="-", lw=0.8, alpha=0.6)
        ax.annotate(f"{v_phys:.2f} m/s", (v_phys, JOINT_VEL_PHYS), color="red", fontsize=8)
    ax.set_xlabel("body speed (m/s)")
    ax.set_ylabel("joint angular speed (rad/s)")
    ax.set_title(f"mos2026_2: body speed → joint/motor angular speed\n"
                 f"(L={step_length*100:.0f}cm, β={duty}, h={step_height*100:.0f}cm; "
                 f"motor = joint × {GEAR})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ax2.set_ylabel("motor shaft speed (rpm)")
    lo, hi = ax.get_ylim()
    ax2.set_ylim(lo * GEAR * RAD_S_TO_RPM, hi * GEAR * RAD_S_TO_RPM)
    fig.tight_layout()
    p1 = os.path.join(out, "speed_vs_motor_speed.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # --- 图2：2D 设计图 —— (机身速度 × 步幅) → 峰值关节角速度，叠 15/30 等高线 ---
    Ls = np.linspace(0.06, 0.30, 25)
    Z = np.zeros((len(Ls), n_v))
    for r, L in enumerate(Ls):
        for c, v in enumerate(vs):
            g = gait_for_speed(v, step_length=L, duty=duty, step_height=step_height,
                               body_height=body_height)
            Z[r, c] = summarize(cycle_kinematics(g, n=160))["peak_overall"]
    fig2, axc = plt.subplots(figsize=(9, 6))
    pcm = axc.pcolormesh(vs, Ls * 100, Z, shading="auto", cmap="viridis", vmin=0, vmax=40)
    cs = axc.contour(vs, Ls * 100, Z, levels=[JOINT_VEL_TRAIN, JOINT_VEL_PHYS],
                     colors=["orange", "red"], linewidths=2)
    axc.clabel(cs, fmt={JOINT_VEL_TRAIN: "15 rad/s", JOINT_VEL_PHYS: "30 rad/s"}, fontsize=9)
    fig2.colorbar(pcm, ax=axc, label="peak joint angular speed (rad/s)")
    axc.set_xlabel("body speed (m/s)")
    axc.set_ylabel("step length (cm)")
    axc.set_title("Design map: feasible region is LEFT of each limit contour\n"
                  "(longer stride → lower stride freq → lower joint speed, until reach limit bends it back)")
    fig2.tight_layout()
    p2 = os.path.join(out, "design_map_speed_steplength.png")
    fig2.savefig(p2, dpi=120)
    plt.close(fig2)

    # --- CSV ---
    p3 = os.path.join(out, "speed_sweep.csv")
    hdr = "v_body_mps,peak_joint_rad_s,rms_joint_rad_s,peak_motor_rad_s,peak_motor_rpm"
    data = np.stack([vs, peak_joint, rms_joint, peak_joint * GEAR,
                     peak_joint * GEAR * RAD_S_TO_RPM], axis=1)
    np.savetxt(p3, data, delimiter=",", header=hdr, comments="")

    print("已输出：")
    for p in (p1, p2, p3):
        print(f"  {p}")
    print(f"\n固定步幅 L={step_length*100:.0f}cm / β={duty} 下的最大可行机身速度：")
    print(f"  ≤ 训练软限 {JOINT_VEL_TRAIN} rad/s : "
          + (f"{v_train:.2f} m/s" if v_train else "全程可行"))
    print(f"  ≤ 电机物理 {JOINT_VEL_PHYS} rad/s : "
          + (f"{v_phys:.2f} m/s" if v_phys else "全程可行"))


# ============================ 自测 ============================
def _selftest() -> int:
    ok = True
    print("== 1. 无打滑约束 v = L/(β·T) 与构造一致 ==")
    for v in (0.4, 1.0, 2.5):
        g = gait_for_speed(v)
        err = abs(body_speed_of(g) - v)
        print(f"  v={v}: 反解 period={g.period*1000:.2f} ms → 复算 v={body_speed_of(g):.4f} (err {err:.1e})")
        ok &= err < 1e-9

    print("\n== 2. 解析足端速度 vs 有限差分（gait.foot_target 数值导数）==")
    g = gait_for_speed(1.0)
    n = 4000
    ts = np.linspace(0, g.period, n, endpoint=False)
    dt = ts[1] - ts[0]
    max_err = 0.0
    for name in LEG_NAMES:
        fpos = np.array([g.foot_target(name, t) for t in ts])
        fnum = np.gradient(fpos, dt, axis=0)                      # 数值导数
        fana = np.array([foot_vel_hip(g, name, t) for t in ts])
        # 摆动/支撑切换点数值导数会有尖峰，比中段；用 95 分位避免边界毛刺
        err = float(np.percentile(np.abs(fana - fnum), 99))
        print(f"  {name}: |v_analytic − v_numeric| p99 = {err:.2e} m/s")
        max_err = max(max_err, err)
    ok &= max_err < 1e-2   # 切换点附近差分本身有限精度，1cm/s 量级足够

    print("\n== 3. 支撑相足端向后速度 ≈ −v_body（无打滑核心）==")
    g = gait_for_speed(1.5)
    # 取 FL 支撑相中点
    t_mid = 0.25 * g.period      # β=0.5，支撑相在 [0,0.5T]
    vx = foot_vel_hip(g, "fl", t_mid)[0]
    print(f"  v_body=1.5：支撑相中点 FL 足端 vx = {vx:.4f} m/s（期望 −1.5）")
    ok &= abs(vx + 1.5) < 1e-9

    print("\n== 4. Jacobian 解出的 q_dot vs 直接差分 q(t) ==")
    g = gait_for_speed(1.0)
    res = cycle_kinematics(g, n=2000)
    dt = res.times[1] - res.times[0]
    # 周期信号：用 np.gradient 并对首尾做周期延拓减小边界误差
    qd_num = np.gradient(res.q, dt, axis=0)
    diff = np.abs(res.qd - qd_num)
    p99 = float(np.percentile(diff, 99))
    print(f"  |qd_jacobian − qd_numeric| p99 = {p99:.2e} rad/s")
    ok &= p99 < 0.2     # 切换点差分尖峰；中段应极小

    print("\n== 5. 已知量级核对：低速 0.4 m/s 应远在限位内、3 m/s 小步幅应超限 ==")
    _, s04 = query(0.4, verbose=False)
    _, s30 = query(3.0, step_length=0.10, verbose=False)
    print(f"  0.4 m/s 峰值 {s04['peak_overall']:.2f} rad/s → 训练限内={s04['feasible_train']}")
    print(f"  3.0 m/s @10cm步幅 峰值 {s30['peak_overall']:.2f} rad/s → 训练限内={s30['feasible_train']}")
    ok &= s04["feasible_train"] and (not s30["feasible_train"])

    print("\n" + ("✅ 全部通过" if ok else "❌ 有用例失败"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="mos2026_2 机身速度 ↔ 关节/电机角速度 解算")
    p.add_argument("--speed", type=float, default=None, help="目标机身速度 m/s（单点查询）")
    p.add_argument("--step-length", type=float, default=0.10, help="步幅 m（默认 0.10）")
    p.add_argument("--duty", type=float, default=0.5, help="支撑相占空比（trot 默认 0.5）")
    p.add_argument("--step-height", type=float, default=0.04, help="摆动相抬腿高度 m")
    p.add_argument("--body-height", type=float, default=0.30, help="站立足端深度 m")
    p.add_argument("--sweep", action="store_true", help="扫描机身速度，出曲线/设计图/CSV")
    p.add_argument("--v-max", type=float, default=4.0, help="--sweep 的最大机身速度")
    p.add_argument("--selftest", action="store_true", help="自洽性自测")
    args = p.parse_args()

    if args.selftest:
        return _selftest()
    if args.sweep:
        sweep(v_max=args.v_max, step_length=args.step_length, duty=args.duty,
              step_height=args.step_height, body_height=args.body_height)
        return 0
    if args.speed is not None:
        query(args.speed, step_length=args.step_length, duty=args.duty,
              step_height=args.step_height, body_height=args.body_height)
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
