"""mos2026_2 足端轨迹步态生成器（Phase 2 传统控制 baseline）。

思路（对应 todo.md Phase 2「足端轨迹 gait」）
---------------------------------------------
传统四足步态在「足端笛卡尔轨迹」里规划，再用 IK 转成关节目标——比在关节空间硬凑
正弦直观得多，也能直接对齐真机的步幅/步高/离地高度。本模块实现一个对角小跑（trot）：

  · 相位：FL+RR 同相，FR+RL 反相（差半个周期）——对角腿成对，trot 的定义。
  · 支撑相（stance, 占空比 β）：足端贴地，相对机身从 +L/2 向后扫到 −L/2，
    机身因此前进（足端不动、机身动）。
  · 摆动相（swing, 1−β）：足端抬起前摆，水平用摆线（cycloid）、竖直用 (1−cos)/2，
    保证离地/触地瞬间水平与竖直速度都为 0 → 平滑、少打滑。

运动学后端：闭链解析模型（2026-07-21 起）
------------------------------------------
本模块原先用 `kinematics.py` 的 3-DOF 串联近似，输出「约定角」，还需要一层
**从未标定过**的仿射映射才能下发——实际上无法部署。现已换成
`closed_chain_kin.py`：直接对真实的共轴双输入平面四杆建模，
**输出就是 MJCF/电机的关节角**（θ_hip, θ_thigh, θ_crank），没有中间映射。

随之而来的两处接口变更（调用方注意）：

  1. `foot_target` 返回的是**腿系**（leg frame，原点在腿根 body / 髋轴上）坐标，
     不再是原先折入挂载偏移的「髋系」。零位站姿在腿系里并不在原点正下方：
     前腿 x ≈ −0.112 m（足端在髋后方），后腿 x ≈ +0.131 m。故 `x_offset` 是
     **相对名义站姿**的增量，名义站姿 x 由 `NOMINAL_X` 从零位 FK 算出。
  2. `joint_targets` / `rollout` 返回 (θ_hip, θ_thigh, θ_crank)，
     不再是 (q_ab, q_hip, q_knee)。

⚠️ `speed_map.py` / `dynamics.py` / `tools/isaac/speed_viz_isaac.py` 仍在用旧的
`kinematics.py` 串联模型自行做 IK，尚未同步（见文末 TODO）。

自测：``python 传统步态/代码/gait.py --selftest``
出图/CSV：``python 传统步态/代码/gait.py --demo``
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field

import numpy as np

from closed_chain_kin import (
    JOINT_LIMIT,
    LEG_NAMES,
    LEGS,
    R_A,
    _rot2,
    leg_fk,
    leg_ik,
    solve_linkage,
)

# 名义站姿：零位（全关节 0）时足端在腿系的 x。四杆机构前后镜像，故前后腿不同号。
NOMINAL_X = {n: float(leg_fk(0.0, 0.0, 0.0, LEGS[n], frame="leg")[0])
             for n in LEG_NAMES}
# 零位站高（足端深度），作为 body_height 的量程参考
NOMINAL_DEPTH = -float(leg_fk(0.0, 0.0, 0.0, LEGS["fl"], frame="leg")[2])

# 步态规划应与奇异位形（装配分支折叠边界）保持的最小余量
MIN_MARGIN = 0.010


@dataclass
class TrotGait:
    """对角小跑步态参数（单位：m / s）。

    默认值都已用 `--selftest` 验过落在工作域内、且奇异余量 > 10 mm。
    实测可用范围：深度 0.24–0.29 m，前后 ±0.12 m。
    """

    # 足端在腿系的深度（−z）。零位站高 0.2729 m；默认略蹲一点留控制余量。
    body_height: float = 0.26
    step_length: float = 0.10      # 单步前后总位移
    step_height: float = 0.04      # 摆动相离地高度
    period: float = 0.5            # 一个步态周期时长（s）
    duty: float = 0.5              # 支撑相占空比（trot 取 0.5）
    x_offset: float = 0.0          # 相对名义站姿的前后偏置
    y_offset: float = 0.0          # 站距加宽（左腿 +、右腿 −，由髋外摆实现）

    # 侧移 / 转向指令。前进量由 step_length 给（保持与既有调用方兼容），
    # 侧移与转向按速度给，换算成各腿自己的支撑相扫掠量：
    #   刚体上一点的速度 v_i = v + ω × r_i，r_i 是该足端相对机身中心的名义位置。
    #   (ω × r)_x = −wz·r_y,  (ω × r)_y = +wz·r_x
    # 所以转向时左右腿步长不同——这正是四足能原地转的原因。
    vy: float = 0.0                # 侧移速度 m/s（左为正）
    wz: float = 0.0                # 转向角速度 rad/s（左转为正）

    # 摆动相水平轨迹：
    #   "cycloid" 摆线，离地/触地瞬间**机体系**水平速度为 0；
    #   "matched" 三次多项式，端点速度匹配支撑相的 −L/(βT)。
    # 触地时脚相对**地面**的速度才决定打不打滑：cycloid 落地瞬间脚随机身以 +v
    # 前移，一触地就被摩擦刹住，每步损失一截，实测速度达成率只有 ~50%。
    # matched 让脚落地时相对地面速度≈0，达成率提到 ~95%。默认用 matched。
    swing_profile: str = "matched"

    # 对角相位偏移：FL,FR,RL,RR
    phase_offset: dict = field(default_factory=lambda: {
        "fl": 0.0, "fr": 0.5, "rl": 0.5, "rr": 0.0})

    def step_vec(self, leg_name: str) -> np.ndarray:
        """该腿支撑相要扫过的 (dx, dy)。转向时逐腿不同（v + ω × r）。"""
        leg = LEGS[leg_name]
        t_stance = self.duty * self.period
        # 足端名义位置相对机身中心（base 原点）
        r_x = leg.leg_pos[0] + NOMINAL_X[leg_name] + self.x_offset
        r_y = leg.leg_pos[1] + leg.y_foot
        return np.array([
            self.step_length + (-self.wz * r_y) * t_stance,
            (self.vy + self.wz * r_x) * t_stance,
        ])

    def foot_target(self, leg_name: str, t: float) -> np.ndarray:
        """给定腿名与时刻 t，返回**腿系**足端目标 (x, y, z)。"""
        leg = LEGS[leg_name]
        p = (t / self.period + self.phase_offset[leg_name]) % 1.0
        x0 = NOMINAL_X[leg_name] + self.x_offset
        z0 = -self.body_height
        y0 = leg.y_foot + math.copysign(self.y_offset, leg.y_foot)
        Lx, Ly = self.step_vec(leg_name)
        h, beta = self.step_height, self.duty

        if p < beta:  # 支撑相：+L/2 → −L/2，贴地
            s = p / beta
            f = 0.5 - s
            z = z0
        else:         # 摆动相：前摆 + (1−cos) 抬腿（抬腿两端速度均为 0）
            s = (p - beta) / (1.0 - beta)
            if self.swing_profile == "cycloid":
                f = -0.5 + (s - math.sin(2 * math.pi * s) / (2 * math.pi))
            elif self.swing_profile == "matched":
                # 三次多项式 f(s)=a+bs+cs²+ds³，边界条件（s 归一到摆动相时长）：
                #   f(0)=−1/2, f(1)=+1/2, f'(0)=f'(1)=−r,  r=(1−β)/β
                # 解得 a=−1/2, b=−r, c=3(1+r), d=−2(1+r)
                # β=0.5(r=1) ⇒ f(s)=−1/2 − s + 6s² − 4s³
                r = (1.0 - beta) / beta
                f = (-0.5 - r * s + 3.0 * (1.0 + r) * s ** 2
                     - 2.0 * (1.0 + r) * s ** 3)
            else:
                raise ValueError(f"未知 swing_profile: {self.swing_profile!r}")
            z = z0 + h * (1.0 - math.cos(2 * math.pi * s)) / 2.0
        # 同一个归一化型线 f 同时驱动 x 与 y，转向/侧移才不会与前进相位错开
        return np.array([x0 + Lx * f, y0 + Ly * f, z])

    def phase_of(self, leg_name: str, t: float) -> float:
        """该腿在 t 时刻的相位 [0,1)，< duty 为支撑相。"""
        return (t / self.period + self.phase_offset[leg_name]) % 1.0

    def is_stance(self, leg_name: str, t: float) -> bool:
        return self.phase_of(leg_name, t) < self.duty

    def joint_targets(self, t: float, seed: dict | None = None) -> dict:
        """时刻 t 的 4 腿关节角 {leg: (θ_hip, θ_thigh, θ_crank)}。

        seed: {leg: (θ_thigh, θ_crank)} 上一周期解，用于延拓（更快、解连续）。
        """
        out = {}
        for name in LEG_NAMES:
            foot = self.foot_target(name, t)
            s = None if seed is None else seed.get(name)
            h, th, cr, info = leg_ik(foot, LEGS[name], frame="leg", seed=s)
            if not info["converged"]:
                raise ValueError(
                    f"{name} 腿在 t={t:.3f}s 的足端目标 {np.round(foot, 4)} 不可解"
                    f"（残差 {info['residual']:.2e} m，奇异余量 {info['margin']:.4f} m）"
                    " —— 检查 body_height / step_length 是否超出工作域")
            out[name] = (h, th, cr)
        return out

    def rollout(self, n: int = 200):
        """返回 (times[n], feet[n,4,3], q[n,4,3])，覆盖一个完整周期。

        q 的列顺序 = (θ_hip, θ_thigh, θ_crank)；feet 为腿系坐标。
        逐步延拓上一时刻的解，保证轨迹在关节空间也连续。
        """
        times = np.linspace(0.0, self.period, n, endpoint=False)
        feet = np.zeros((n, 4, 3))
        q = np.zeros((n, 4, 3))
        seed = None
        for ti, t in enumerate(times):
            tgt = self.joint_targets(t, seed=seed)
            for li, name in enumerate(LEG_NAMES):
                feet[ti, li] = self.foot_target(name, t)
                q[ti, li] = tgt[name]
            seed = {n_: (tgt[n_][1], tgt[n_][2]) for n_ in LEG_NAMES}
        return times, feet, q

    def margins(self, n: int = 200) -> np.ndarray:
        """整周期每腿的奇异余量 (n, 4)，用于确认步态没贴着分支折叠边界走。"""
        _, _, q = self.rollout(n)
        return np.array([[solve_linkage(q[ti, li, 1], q[ti, li, 2], LEGS[name])["margin"]
                          for li, name in enumerate(LEG_NAMES)]
                         for ti in range(len(q))])


# --- 自测 ---------------------------------------------------------------------
def _selftest() -> int:
    ok = True
    gait = TrotGait()
    times, feet, q = gait.rollout(400)

    print("== 0. 名义站姿（腿系）==")
    print(f"  零位站高 = {NOMINAL_DEPTH:.4f} m   本步态 body_height = {gait.body_height} m")
    for n in LEG_NAMES:
        print(f"  {n}: 名义 x = {NOMINAL_X[n]:+.4f} m   y = {LEGS[n].y_foot:+.4f} m")

    # 1) IK 可解且足端 FK 往返一致
    print("\n== 1. 步态 IK 足端往返 ==")
    max_rt = 0.0
    for li, name in enumerate(LEG_NAMES):
        foot_fk = np.stack([leg_fk(q[k, li, 0], q[k, li, 1], q[k, li, 2],
                                   LEGS[name], frame="leg") for k in range(len(times))])
        max_rt = max(max_rt, float(np.max(np.linalg.norm(foot_fk - feet[:, li], axis=-1))))
    print(f"  足端 FK(IK) 往返 max_err = {max_rt:.2e} m")
    ok &= max_rt < 1e-9

    # 2) 关节角在限位内
    print("\n== 2. 关节限位（±1.57 rad）==")
    qmax = float(np.max(np.abs(q)))
    print(f"  |q|_max = {qmax:.3f} rad（限位 {JOINT_LIMIT}）")
    for li, name in enumerate(LEG_NAMES):
        print(f"  {name}: hip [{q[:, li, 0].min():+.3f},{q[:, li, 0].max():+.3f}] "
              f"thigh [{q[:, li, 1].min():+.3f},{q[:, li, 1].max():+.3f}] "
              f"crank [{q[:, li, 2].min():+.3f},{q[:, li, 2].max():+.3f}]")
    ok &= qmax < JOINT_LIMIT

    # 3) 奇异余量：整周期都要离分支折叠边界足够远
    print(f"\n== 3. 奇异余量（阈值 {MIN_MARGIN * 1e3:.0f} mm）==")
    mg = gait.margins(400)
    print(f"  整周期最小余量 = {mg.min() * 1e3:.1f} mm   最大 = {mg.max() * 1e3:.1f} mm")
    ok &= mg.min() > MIN_MARGIN

    # 4) 轨迹连续（相邻采样足端位移、关节角增量有界）
    print("\n== 4. 轨迹连续性（周期首尾相接、无跳变）==")
    close = max(float(np.linalg.norm(gait.foot_target(n, 0.0)
                                     - gait.foot_target(n, gait.period)))
                for n in LEG_NAMES)
    dfoot = float(np.max(np.linalg.norm(np.diff(feet, axis=0), axis=-1)))
    dq = float(np.max(np.abs(np.diff(q, axis=0))))
    print(f"  周期闭合误差 = {close:.2e}  相邻足端位移 = {dfoot * 1000:.2f} mm/步"
          f"  关节增量 = {math.degrees(dq):.2f}°")
    ok &= close < 1e-9 and dfoot < 0.01 and dq < 0.1

    # 5) 对角相位：取 t=period/4（支撑相中点 vs 摆动相中点），此时两组 z 必须分开。
    #    t=0 处两组 z 恰好都等于 z0（摆动相起点抬腿量为 0），验不出任何东西。
    t5 = gait.period * 0.25
    print(f"\n== 5. 对角 trot 相位检查（t={t5:.3f}s，支撑/摆动应分成两组）==")
    z5 = {n: gait.foot_target(n, t5)[2] for n in LEG_NAMES}
    print("  " + "  ".join(f"{n}:z={z5[n]:.4f}" for n in LEG_NAMES))
    diag_ok = (abs(z5["fl"] - z5["rr"]) < 1e-12 and abs(z5["fr"] - z5["rl"]) < 1e-12
               and abs(z5["fl"] - z5["fr"]) > gait.step_height * 0.5)
    print(f"  对角同相且两组分离（FL≈RR, FR≈RL, 组间 Δz="
          f"{abs(z5['fl'] - z5['fr']) * 1e3:.1f} mm）: {diag_ok}")
    ok &= diag_ok

    # 5b) 转向/侧移：v + ω×r 的符号与量级
    print("\n== 5b. 转向 / 侧移指令 ==")
    gt = TrotGait(step_length=0.0, wz=0.5)      # 原地左转
    sv = {n: gt.step_vec(n) for n in LEG_NAMES}
    print("  原地左转 wz=+0.5：各腿支撑相扫掠 (dx, dy) mm")
    for n in LEG_NAMES:
        print(f"    {n}: ({sv[n][0] * 1e3:+6.1f}, {sv[n][1] * 1e3:+6.1f})")
    # 左转时左侧腿应向后扫得少、右侧腿多（dx 反号于 r_y·wz）
    turn_ok = (sv["fl"][0] < 0 < sv["fr"][0] and sv["rl"][0] < 0 < sv["rr"][0]
               and sv["fl"][1] > 0 and sv["rl"][1] < 0)
    print(f"  左右扫掠反号、前后腿侧向反号（能形成绕 z 的旋转）: {turn_ok}")
    ok &= turn_ok

    gs = TrotGait(step_length=0.0, vy=0.2)      # 纯侧移
    sy = [gs.step_vec(n)[1] for n in LEG_NAMES]
    side_ok = all(abs(v - 0.2 * gs.duty * gs.period) < 1e-12 for v in sy)
    print(f"  纯侧移 vy=+0.2：四腿 dy 一致 = {sy[0] * 1e3:+.1f} mm  {side_ok}")
    ok &= side_ok

    # 6) 工作域边界：极端参数应被明确拒绝而不是悄悄给个错解
    print("\n== 6. 超工作域时应报错 ==")
    try:
        TrotGait(body_height=0.42).joint_targets(0.0)
        print("  ❌ body_height=0.42 未报错")
        ok = False
    except ValueError as e:
        print(f"  ✅ 已拒绝：{str(e)[:70]}…")

    print("\n" + ("✅ 全部通过" if ok else "❌ 有用例失败"))
    return 0 if ok else 1


def _demo() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "图与数据", "gait_demo")
    os.makedirs(out_dir, exist_ok=True)

    gait = TrotGait()
    times, feet, q = gait.rollout(300)

    # 关节角曲线（matplotlib 默认字体无中文，标签用英文避免乱码）
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    labels = ["q_hip", "q_thigh", "q_crank (shank_link)"]
    for li, name in enumerate(LEG_NAMES):
        ax = axes[li // 2][li % 2]
        for j in range(3):
            ax.plot(times, np.degrees(q[:, li, j]), label=labels[j])
        ax.set_title(f"leg {name.upper()}")
        ax.set_ylabel("deg")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1][0].set_xlabel("t (s)")
    axes[1][1].set_xlabel("t (s)")
    fig.suptitle("mos2026_2 trot gait: actuated joint targets from closed-chain IK")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "joint_targets.png")
    fig.savefig(p1, dpi=120)

    # 足端矢状面轨迹（x-z），看到典型的 "D" 形 trot 轨迹
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for li, name in enumerate(LEG_NAMES):
        ax2.plot(feet[:, li, 0], feet[:, li, 2], label=name.upper())
    ax2.set_xlabel("x forward (m)")
    ax2.set_ylabel("z up (m)")
    ax2.set_title("Foot sagittal trajectory (leg frame): flat stance + cycloid swing")
    ax2.axis("equal")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "foot_trajectory.png")
    fig2.savefig(p2, dpi=120)

    # CSV：time + 12 关节角（leg×[hip,thigh,crank]），可直接对齐部署侧顺序
    header = "t," + ",".join(f"{n}_{j}" for n in LEG_NAMES
                             for j in ("hip", "thigh", "crank"))
    data = np.concatenate([times[:, None], q.reshape(len(times), -1)], axis=1)
    p3 = os.path.join(out_dir, "joint_targets.csv")
    np.savetxt(p3, data, delimiter=",", header=header, comments="")

    p4 = _plot_linkage("fl", out_dir)

    print("已输出：")
    for p in (p1, p2, p3, p4):
        print(f"  {p}")
    return 0


def _plot_linkage(leg_name: str, out_dir: str) -> str:
    """画矢状面四杆机构姿态序列 + 足端轨迹（腿系，腿根固定）。

    画的是真实闭链：曲柄 O→P1、连杆 P1→A、大腿 O→K、小腿 K→A→足端。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    gait = TrotGait()
    leg = LEGS[leg_name]
    times, feet, q = gait.rollout(240)
    li = LEG_NAMES.index(leg_name)

    O = np.array(leg.pivot_xz)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    n_pose = 16
    idx = np.linspace(0, len(times) - 1, n_pose, dtype=int)
    colors = cm.viridis(np.linspace(0, 1, n_pose))
    for c, i in zip(colors, idx):
        s = solve_linkage(q[i, li, 1], q[i, li, 2], leg)
        K, A, toe = s["K"], s["A"], s["toe"]
        # 曲柄侧 P1：A 沿连杆反推不便，直接由曲柄角算
        P1 = O + _rot2(leg.sgn_c * q[i, li, 2]) @ R_A
        ax.plot([O[0], K[0]], [O[1], K[1]], "-", color=c, lw=2.4, alpha=0.8)   # 大腿
        ax.plot([K[0], toe[0]], [K[1], toe[1]], "-", color=c, lw=2.4, alpha=0.8)  # 小腿→足
        ax.plot([K[0], A[0]], [K[1], A[1]], "-", color=c, lw=1.2, alpha=0.5)   # 小腿支耳
        ax.plot([O[0], P1[0]], [O[1], P1[1]], "--", color=c, lw=1.6, alpha=0.8)  # 曲柄
        ax.plot([P1[0], A[0]], [P1[1], A[1]], ":", color=c, lw=1.6, alpha=0.8)   # 连杆
        ax.plot(*K, "o", color=c, ms=4)
        ax.plot(*A, "^", color=c, ms=4)
        ax.plot(*toe, ".", color=c, ms=7)

    ax.plot(feet[:, li, 0], feet[:, li, 2], "k-", lw=1.2, label="foot trajectory")
    ax.plot(*O, "rs", ms=10, label="pivot O (thigh & crank, coaxial)")
    ax.annotate("O", O, textcoords="offset points", xytext=(8, 6), color="r")

    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("z up (m)")
    ax.set_title(f"leg {leg_name.upper()} closed-chain four-bar over one trot cycle\n"
                 "solid=thigh/shank  dashed=crank  dotted=coupler  (color = phase)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    sm = cm.ScalarMappable(cmap="viridis")
    sm.set_array([0, gait.period])
    fig.colorbar(sm, ax=ax, label="phase t (s)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "leg_linkage.png")
    fig.savefig(path, dpi=120)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="mos2026_2 足端轨迹 trot 步态")
    p.add_argument("--selftest", action="store_true", help="运行步态自洽性自测")
    p.add_argument("--demo", action="store_true", help="输出关节曲线/足端轨迹图 + CSV + 连杆图")
    p.add_argument("--linkage", action="store_true", help="只画矢状面四杆机构 + 足端轨迹")
    p.add_argument("--leg", type=str, default="fl", choices=list(LEG_NAMES),
                   help="连杆图选哪条腿（默认 fl）")
    args = p.parse_args()
    if args.selftest:
        return _selftest()
    if args.demo:
        return _demo()
    if args.linkage:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "图与数据", "gait_demo")
        os.makedirs(out_dir, exist_ok=True)
        print(f"已输出：{_plot_linkage(args.leg, out_dir)}")
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
