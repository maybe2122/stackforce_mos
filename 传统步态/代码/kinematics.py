"""mos2026_2 四足腿部运动学（正/逆运动学，FK/IK）。

为什么需要它（对应 todo.md §E）
---------------------------------
纯 RL 端到端策略直接输出 12 个关节目标，可以不显式用运动学；但下面三件事必须有 FK/IK：

  1. Phase 2 传统控制 baseline：手写 gait 在「足端笛卡尔轨迹」里规划（抬腿→前摆→
     落地），再用 IK 转成关节目标。没有 IK 只能在关节空间硬凑正弦。
  2. 足端接触/打滑判定：用 FK 算真实足端点高度/速度，替代当前用 shank body 中心
     高度做的粗略近似（见 eval.py 的 foot_contact_height）。
  3. 腿式里程计（leg odometry）：足端 FK + 接触相位估机身线速度，给真机
     lin_vel_source=zero 的降级问题兜底。

模型与坐标约定
---------------
每条腿抽象为标准 3-DOF 串联腿（与 Unitree/MIT-Cheetah 同构）：

    q_ab   : 髋外摆（abduction），绕机体 x 轴
    q_hip  : 大腿俯仰（thigh），绕 y 轴
    q_knee : 膝（knee），绕 y 轴，相对大腿

腿局部「髋系」：x 前、y 左、z 上，原点在外摆轴上。几何参数取自
``deploy/mujoco/assets/mos2026_2.xml``：

    L1 (大腿) = |(Δx, Δz)_{thigh→shank}| = √(0.144992² + 0.106664²) = 0.180 m
    d  (髋侧向偏移) = thigh joint 的 y 偏移（左腿 +、右腿 −）
    L2 (小腿/足端) ≈ 0.16 m  —— XML 无 foot site，按站立几何估计，需标定

约定角定义（便于解析 IK，与 motor/sim 零位差一个仿射偏移，见下「标定」）：
    q_hip 自 −z（正下方）量起，正方向使足端前摆（+x）；
    q_knee 相对大腿，knee_sign=-1 时膝向后弯（四足常见后膝）。

⚠️ 闭链特殊性
-------------
本机器人膝由平行四连杆驱动：MuJoCo 里 actuator 实际驱动 ``*_shank_link``，真实
``*_shank`` 经 equality/connect 闭环跟随。所以「shank 电机轴角 ↔ 本模块的等效
q_knee」存在一个连杆传动关系，需单独标定（todo.md §E「闭链传动标定」）。

标定（约定角 → sim/motor 角）
----------------------------
本模块产出的是「约定角」，下发真机前需做每关节仿射映射：
    q_motor = sign · q_convention + offset
sign/offset 取决于 USD/电机的轴向与零位（见 config/mos2026_2.yaml 的
default_dof_pos 与 stand_config.json）。映射标定后即可接 rl_deploy.py。

自测：``python 传统步态/代码/kinematics.py --selftest``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

# --- 几何常量（取自 deploy/mujoco/assets/mos2026_2.xml，单位 m） ---------------
L1_THIGH = 0.180          # 大腿连杆长（矢状面内 thigh→shank），XML 实测
L2_SHANK = 0.160          # 小腿/足端长，XML 无 foot site，按站立几何估计（待标定）

# 腿顺序与 config/mos2026_2.yaml、joint_map.default.json 一致
LEG_NAMES = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class LegGeom:
    """单腿几何（髋系，base-aligned）。"""

    name: str
    hip_mount: tuple[float, float, float]  # 外摆轴原点在 base 系坐标（已折入 thigh 的 x/z 偏移）
    d: float                               # 髋侧向偏移（左 +，右 −）
    ab_sign: float                         # 外摆轴方向（FL/FR=−x → −1；RL/RR=+x → +1）
    pitch_sign: float                      # 俯仰轴方向（左腿 −y → −1；右腿 +y → +1）
    l1: float = L1_THIGH
    l2: float = L2_SHANK


# 直接来自 XML：leg body pos、thigh offset、hip/thigh 轴向
# hip_mount = leg_body_pos + (thigh_off_x, 0, thigh_off_z)（把矢状面常量偏移折进挂载点，
# 使外摆解算只剩 y 向偏移 d，IK 保持解析且可严格往返）。
LEGS: dict[str, LegGeom] = {
    "fl": LegGeom("fl", (0.378300 - 0.116000, 0.098000, -0.010100), d=+0.068150, ab_sign=-1.0, pitch_sign=-1.0),
    "fr": LegGeom("fr", (0.378300 - 0.116000, -0.098000, -0.010100), d=-0.068150, ab_sign=-1.0, pitch_sign=+1.0),
    "rl": LegGeom("rl", (-0.389329 + 0.127029, 0.098000, -0.010100), d=+0.087150, ab_sign=+1.0, pitch_sign=-1.0),
    "rr": LegGeom("rr", (-0.372300 + 0.110000, -0.098000, -0.010100), d=-0.068150, ab_sign=+1.0, pitch_sign=+1.0),
}


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def leg_fk(q_ab, q_hip, q_knee, leg: LegGeom, *, frame: str = "hip") -> np.ndarray:
    """单腿正运动学：约定角 → 足端坐标。

    参数可为标量或等长数组（向量化）。返回 shape (..., 3)。
    frame="hip" 返回髋系（外摆轴为原点）足端；frame="base" 加挂载点返回 base 系。
    """
    q_ab = _as_array(q_ab)
    q_hip = _as_array(q_hip)
    q_knee = _as_array(q_knee)
    l1, l2, d = leg.l1, leg.l2, leg.d

    # 矢状面 2 连杆（x 前、z 上），大腿角自 −z 量起
    x = l1 * np.sin(q_hip) + l2 * np.sin(q_hip + q_knee)
    z = -(l1 * np.cos(q_hip) + l2 * np.cos(q_hip + q_knee))

    # 外摆绕 x：旋转 (y, z)，外摆前足端 y = d
    ca, sa = np.cos(q_ab), np.sin(q_ab)
    y_h = d * ca - z * sa
    z_h = d * sa + z * ca
    foot = np.stack([x, y_h, z_h], axis=-1)

    if frame == "base":
        foot = foot + _as_array(leg.hip_mount)
    elif frame != "hip":
        raise ValueError(f"frame 必须为 'hip' 或 'base'，收到 {frame!r}")
    return foot


def leg_ik(foot, leg: LegGeom, *, frame: str = "hip", knee_sign: float = -1.0):
    """单腿逆运动学：足端坐标 → 约定角 (q_ab, q_hip, q_knee)。

    解析解（外摆 + 矢状面 2 连杆）。超出可达域时对中间量做 clamp，返回的角仍是
    「最接近目标」的可行解（不抛异常，便于实时控制）。
    knee_sign=-1 选膝向后弯的解（四足常见），+1 选向前弯。
    返回三个与输入同 shape 的数组。
    """
    foot = _as_array(foot)
    if frame == "base":
        foot = foot - _as_array(leg.hip_mount)
    elif frame != "hip":
        raise ValueError(f"frame 必须为 'hip' 或 'base'，收到 {frame!r}")

    px, py, pz = foot[..., 0], foot[..., 1], foot[..., 2]
    l1, l2, d = leg.l1, leg.l2, leg.d

    # --- 外摆：y-z 平面内半径守恒，腿平面点为 (d, z_p) ---
    r_yz_sq = py ** 2 + pz ** 2
    z_p = -np.sqrt(np.clip(r_yz_sq - d ** 2, 0.0, None))  # 足端在下方
    q_ab = np.arctan2(pz, py) - np.arctan2(z_p, d)
    q_ab = np.arctan2(np.sin(q_ab), np.cos(q_ab))  # wrap 到 (−π, π]

    # --- 矢状面 2 连杆，目标 (px, z_p) ---
    l_sq = px ** 2 + z_p ** 2
    c2 = (l_sq - l1 ** 2 - l2 ** 2) / (2.0 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)
    q_knee = knee_sign * np.arccos(c2)

    q_hip = np.arctan2(px, -z_p) - np.arctan2(l2 * np.sin(q_knee), l1 + l2 * np.cos(q_knee))
    q_hip = np.arctan2(np.sin(q_hip), np.cos(q_hip))
    return q_ab, q_hip, q_knee


def leg_jacobian(q_ab, q_hip, q_knee, leg: LegGeom) -> np.ndarray:
    """单腿解析 Jacobian J = ∂foot/∂q，foot=(x,y,z)，q=(q_ab,q_hip,q_knee)。

    满足  foot_dot = J · q_dot，反之 q_dot = solve(J, foot_dot)。这是「足端速度 ↔
    关节角速度」的核心映射——给定机身/足端速度需求即可解出每个电机要转多快。

    由 `leg_fk` 直接微分得到（含外摆耦合）。frame 不影响 Jacobian：base 系只比 hip
    系多一个常量挂载点偏移，对 q 求导为 0。

    参数可为标量或等长数组（向量化）。返回 shape (..., 3, 3)，行=foot(x,y,z)、
    列=q(ab,hip,knee)。在奇异位形（腿完全伸直 q_knee=0）附近 J 接近奇异，求逆请用
    `np.linalg.solve` / `pinv` 并留意。
    """
    qa = _as_array(q_ab)
    qh = _as_array(q_hip)
    qk = _as_array(q_knee)
    l1, l2, d = leg.l1, leg.l2, leg.d

    ca, sa = np.cos(qa), np.sin(qa)
    # 矢状面量（与 leg_fk 一致）：x 前、z_sag 为外摆前的竖直分量
    z_sag = -(l1 * np.cos(qh) + l2 * np.cos(qh + qk))
    dx_dqh = l1 * np.cos(qh) + l2 * np.cos(qh + qk)
    dx_dqk = l2 * np.cos(qh + qk)
    # dz_sag/dqh == x；dz_sag/dqk == L2 sin(qh+qk)
    dzs_dqh = l1 * np.sin(qh) + l2 * np.sin(qh + qk)
    dzs_dqk = l2 * np.sin(qh + qk)

    zeros = np.zeros_like(dx_dqh)
    # 行 x：外摆不改变 x
    jx = np.stack([zeros, dx_dqh, dx_dqk], axis=-1)
    # 行 y_h = d cos(qa) − z_sag sin(qa)
    jy = np.stack([-d * sa - z_sag * ca, -sa * dzs_dqh, -sa * dzs_dqk], axis=-1)
    # 行 z_h = d sin(qa) + z_sag cos(qa)
    jz = np.stack([d * ca - z_sag * sa, ca * dzs_dqh, ca * dzs_dqk], axis=-1)
    return np.stack([jx, jy, jz], axis=-2)


def quad_jacobian(q: np.ndarray) -> np.ndarray:
    """整机 Jacobian。q: (..., 4, 3) 顺序 [ab,hip,knee]×[fl,fr,rl,rr]。
    返回 (..., 4, 3, 3)，每腿一个 3×3 J。"""
    q = _as_array(q)
    out = []
    for i, name in enumerate(LEG_NAMES):
        out.append(leg_jacobian(q[..., i, 0], q[..., i, 1], q[..., i, 2], LEGS[name]))
    return np.stack(out, axis=-3)


def leg_foot_vel_to_qdot(foot_vel, q_ab, q_hip, q_knee, leg: LegGeom) -> np.ndarray:
    """足端速度 → 关节角速度：q_dot = J(q)⁻¹ · foot_vel。

    foot_vel: (..., 3)（髋系/base 系同值，速度与平移无关）。返回 (..., 3) 的
    (q_ab_dot, q_hip_dot, q_knee_dot)。奇异位形用最小二乘（pinv）兜底，不抛异常。
    """
    J = leg_jacobian(q_ab, q_hip, q_knee, leg)          # (..., 3, 3)
    fv = _as_array(foot_vel)[..., None]                 # (..., 3, 1)
    try:
        qd = np.linalg.solve(J, fv)
    except np.linalg.LinAlgError:
        qd = np.linalg.pinv(J) @ fv
    return qd[..., 0]


def quad_fk(q: np.ndarray, *, frame: str = "base") -> np.ndarray:
    """整机 FK。q: shape (..., 4, 3)，顺序 [ab, hip, knee] × [fl,fr,rl,rr]。
    返回 (..., 4, 3) 足端坐标。"""
    q = _as_array(q)
    feet = []
    for i, name in enumerate(LEG_NAMES):
        feet.append(leg_fk(q[..., i, 0], q[..., i, 1], q[..., i, 2], LEGS[name], frame=frame))
    return np.stack(feet, axis=-2)


def quad_ik(feet: np.ndarray, *, frame: str = "base", knee_sign: float = -1.0) -> np.ndarray:
    """整机 IK。feet: (..., 4, 3) → q: (..., 4, 3)。"""
    feet = _as_array(feet)
    out = []
    for i, name in enumerate(LEG_NAMES):
        a, h, k = leg_ik(feet[..., i, :], LEGS[name], frame=frame, knee_sign=knee_sign)
        out.append(np.stack([a, h, k], axis=-1))
    return np.stack(out, axis=-2)


# --- 自测 ---------------------------------------------------------------------
def _selftest() -> int:
    rng = np.random.default_rng(0)
    ok = True

    # 1) 零位 FK：约定角全 0 → 足端应在挂载点正下方 (L1+L2)
    print("== 1. 零位 FK（约定角全 0，腿竖直向下）==")
    for name, leg in LEGS.items():
        foot = leg_fk(0.0, 0.0, 0.0, leg, frame="base")
        exp = np.array(leg.hip_mount) + np.array([0.0, leg.d, -(leg.l1 + leg.l2)])
        err = float(np.linalg.norm(foot - exp))
        print(f"  {name}: foot={np.round(foot,4)}  期望={np.round(exp,4)}  err={err:.2e}")
        ok &= err < 1e-12

    # 2) FK∘IK 往返：随机可达足端 → IK → FK，应复现
    # 采样限定在「足端位于髋之下」的物理可行域（站立/步态实际工作区）——这是 IK
    # 选取的规范分支；非物理分支（深屈膝使足端翻到髋上方）会被外摆角吸收，足端仍
    # 复现但关节角走另一分支，不在本测试断言范围内。
    print("\n== 2. FK→IK→FK 往返（每腿 2000 个随机可达点，足端在髋下方）==")
    max_err = 0.0
    for name, leg in LEGS.items():
        q_ab = rng.uniform(-0.5, 0.5, 2000)
        q_hip = rng.uniform(-0.4, 0.9, 2000)
        q_knee = rng.uniform(-1.6, -0.4, 2000)  # 膝向后弯（knee_sign=-1）
        foot = leg_fk(q_ab, q_hip, q_knee, leg, frame="hip")
        a2, h2, k2 = leg_ik(foot, leg, frame="hip", knee_sign=-1.0)
        foot2 = leg_fk(a2, h2, k2, leg, frame="hip")
        err = float(np.max(np.linalg.norm(foot - foot2, axis=-1)))
        # 角度也应复现（在唯一解域内）
        ang_err = float(np.max(np.abs(np.stack([a2 - q_ab, h2 - q_hip, k2 - q_knee]))))
        print(f"  {name}: 足端往返 max_err={err:.2e}  关节角 max_err={ang_err:.2e}")
        max_err = max(max_err, err, ang_err)
        ok &= err < 1e-9 and ang_err < 1e-7

    # 3) 整机 API 往返
    print("\n== 3. 整机 quad_fk / quad_ik 往返（batch=512）==")
    q = np.stack([
        rng.uniform(-0.5, 0.5, (512, 4)),
        rng.uniform(-1.0, 1.0, (512, 4)),
        rng.uniform(-2.2, -0.4, (512, 4)),
    ], axis=-1)
    feet = quad_fk(q, frame="base")
    q2 = quad_ik(feet, frame="base", knee_sign=-1.0)
    feet2 = quad_fk(q2, frame="base")
    err = float(np.max(np.linalg.norm(feet - feet2, axis=-1)))
    print(f"  整机足端往返 max_err={err:.2e}")
    ok &= err < 1e-9

    # 3b) Jacobian 对比有限差分：解析 J 应与 FK 的中心差分一致
    print("\n== 3b. 解析 Jacobian vs 有限差分（每腿 1000 个随机位形）==")
    eps = 1e-6
    max_j_err = 0.0
    for name, leg in LEGS.items():
        qa = rng.uniform(-0.5, 0.5, 1000)
        qh = rng.uniform(-0.4, 0.9, 1000)
        qk = rng.uniform(-1.6, -0.4, 1000)
        J = leg_jacobian(qa, qh, qk, leg)                 # (1000, 3, 3)
        # 对每个关节做中心差分，拼出数值 Jacobian
        cols = []
        for j, (da, dh, dk) in enumerate([(eps, 0, 0), (0, eps, 0), (0, 0, eps)]):
            fp = leg_fk(qa + da, qh + dh, qk + dk, leg, frame="hip")
            fm = leg_fk(qa - da, qh - dh, qk - dk, leg, frame="hip")
            cols.append((fp - fm) / (2 * eps))            # ∂foot/∂q_j, (1000, 3)
        J_num = np.stack(cols, axis=-1)                   # (1000, 3, 3)
        jerr = float(np.max(np.abs(J - J_num)))
        # 往返：foot_vel = J·qd 任取 qd，再 solve 回来应复现
        qd = rng.uniform(-2.0, 2.0, (1000, 3))
        fv = np.einsum("nij,nj->ni", J, qd)
        qd2 = np.stack([leg_foot_vel_to_qdot(fv[k], qa[k], qh[k], qk[k], leg) for k in range(0, 1000, 200)])
        rt = float(np.max(np.abs(qd2 - qd[0:1000:200])))
        print(f"  {name}: |J−J_num|_max={jerr:.2e}  foot_vel→qd→ 往返 max_err={rt:.2e}")
        max_j_err = max(max_j_err, jerr, rt)
    ok &= max_j_err < 1e-6

    # 4) 可达性 clamp：给一个远超臂展的目标，不应崩、解有界
    print("\n== 4. 不可达目标的 clamp 行为 ==")
    far = np.array([5.0, 0.0, -5.0])
    a, h, k = leg_ik(far, LEGS["fl"], frame="hip")
    finite = bool(np.all(np.isfinite([a, h, k])))
    print(f"  远点 IK → (ab,hip,knee)=({a:.3f},{h:.3f},{k:.3f}) finite={finite}")
    ok &= finite

    print("\n" + ("✅ 全部通过" if ok else "❌ 有用例失败"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="mos2026_2 腿部 FK/IK")
    p.add_argument("--selftest", action="store_true", help="运行自洽性自测")
    args = p.parse_args()
    if args.selftest:
        return _selftest()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
