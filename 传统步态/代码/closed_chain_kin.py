"""mos2026_2 闭链腿运动学：解析 FK + 数值 IK（纯 numpy）。

为什么需要它
------------
`kinematics.py` 把每条腿近似成标准 3-DOF 串联腿（q_ab, q_hip, q_knee），几何里
L2=0.16 是「XML 无 foot site，按站立几何估计」的猜测值，且「shank 电机轴角 ↔ 等效
膝角」的闭链传动关系一直没标定（见该文件 §⚠️ 闭链特殊性）。结果是：IK 自洽（往返
误差 1e-12），但和真实机构不是同一个映射——用它规划足端轨迹，落到真机/MuJoCo 上
足端并不在你以为的位置。

本模块直接对真实机构建模，不做串联近似。

机构：共轴双输入平面四杆
------------------------
每条腿在矢状面（x 前、z 上）内是一个四杆闭链，受驱的两个关节 **同轴**
（thigh 与 shank_link 的枢轴在腿系里 x/z 完全重合，仅 y 相差一个装配间隙），
所以机架杆长为 0，退化成「从同一原点 O 引出的两条二杆链」：

    链 A（曲柄侧，受驱 θ_c）：O --r_a--> P1 --r_b--> A
    链 B（大腿侧，受驱 θ_t）：O --r_t--> K  --r_s--> A

    O   两个受驱关节的公共枢轴（腿系原点偏移）
    r_a 曲柄 shank_link_a，|r_a| = 0.15638 m（指向后上方）
    r_b 连杆 shank_link_b，|r_b| = 0.16010 m（被动）
    r_t 大腿 thigh，膝点 K，|r_t| = 0.18000 m
    r_s 小腿上的连杆支耳（膝 K → 闭链锚点 A），|r_s| = 0.12417 m

A 是 MuJoCo `<equality><connect>` 的锚点，闭链约束就是「两条链算出的 A 重合」。

⚠️ 该约束严格落在矢状面内：两条链的 y 偏移之和相等（实测残差 1.4e-17 m），
所以可以放心降到 2D 求解，不损失精度。

FK 是闭式的：给定 (θ_t, θ_c)，K 与 P1 已知，A 落在「以 P1 为心 |r_b| 为半径」和
「以 K 为心 |r_s| 为半径」两圆的交点上——两个交点就是两个**装配分支**。
本模块固定取 branch=-1（qpos=0 处与 MJCF 装配一致的那支）。

    ⚠️ 分支即 doc 里记的「MuJoCo 闭链翻分支」：低增益下机身下沉穿过奇异位形
    （两圆相切，h=0），软 connect 会滑到另一支，机身塌到 ~0.17 m。本模块解析式
    永远停在 branch=-1，不会翻——所以它同时是「判断仿真有没有翻分支」的裁判：
    把 MuJoCo 的被动角和本模块的解一比，对不上就是翻了。

足端点
------
MJCF 里没有 foot site，足端取小腿碰撞网格上离膝最远的顶点（凸包尖点），
在小腿系约 (0.1689, ·, -0.1469)，|r| = 0.2244 m。这是几何估计，**未经真机标定**，
是本模块目前唯一的软肋（见文末 TODO）。矢状面链路本身是精确的。

关节角约定
----------
直接用 MJCF/电机的关节角，不再引入「约定角」——省掉 kinematics.py 那层需要标定的
仿射映射。θ_t = XML 里 `*_thigh` 的 qpos，θ_c = `*_shank_link[_a]` 的 qpos，
θ_hip = `*_hip` 的 qpos。轴向符号已折进 `sgn_*` 常量。

自测：``python 传统步态/代码/closed_chain_kin.py --selftest``
（带 mujoco 时会逐腿与 MuJoCo 的约束解交叉验证；没有 mujoco 只跑自洽性检查）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

LEG_NAMES = ("fl", "fr", "rl", "rr")

# --- 矢状面连杆向量（(x, z)，四条腿完全相同，取自 mos2026_2.xml）---------------
R_A = np.array([-0.104652, 0.116201])   # 曲柄 shank_link_a → shank_link_b 原点
R_B = np.array([-0.107640, -0.118510])  # 连杆 shank_link_b → connect 锚点（body1 侧）
R_T = np.array([-0.144992, -0.106664])  # 大腿 thigh → shank 原点（膝 K）
R_S = np.array([-0.067300, 0.104355])   # 小腿 shank → connect 锚点（body2 侧）

# 足端在小腿系的位置：零位（小腿竖直）时碰撞网格的最低顶点，即实际触地点。
# 左右腿 y 略有差异（网格不对称），x/z 四条腿一致。
#
# ⚠️ 足底是圆弧面不是尖点：最低点 2 mm 内有 173 个顶点、x 向跨度 19 mm。小腿转动时
# 触地点会沿弧面迁移，本模块按定点近似 —— 小腿转角 ±0.3 rad 内误差约 ±2 mm。
# 需要更准就得给足底拟合一个圆弧半径（TODO）。
R_TOE_XZ = np.array([0.148960, -0.156158])

# 装配分支：qpos=0 处与 MJCF 一致的那一支（见模块 docstring）
BRANCH = -1.0


@dataclass(frozen=True)
class LegGeom:
    """单腿闭链几何。

    sgn_* 把 MJCF 的关节轴向折算成「矢状面 (x,z) 内的逆时针转角」：
    绕 +y 轴转 q 等于平面内转 −q，故 sgn = −axis_y。
    hip_ax 是髋关节轴的 x 分量（FL/FR = −1，RL/RR = +1，前后镜像而非左右镜像）。
    """

    name: str
    leg_pos: tuple[float, float, float]  # 腿根 body 在 base 系的位置（髋轴原点）
    pivot_xz: tuple[float, float]        # 公共枢轴 O 在腿系的 (x, z)
    y_foot: float                        # 足端在腿系的 y（矢状链不改变 y，故为常量）
    sgn_t: float                         # 大腿
    sgn_c: float                         # 曲柄
    sgn_s: float                         # 小腿（被动）
    sgn_b: float                         # 连杆（被动）
    hip_ax: float


# y_foot = thigh 挂载 y + shank 挂载 y + 足端顶点 y（三段常量偏移之和）
LEGS: dict[str, LegGeom] = {
    "fl": LegGeom("fl", (0.378300, 0.098, 0.0), (-0.116000, -0.0101),
                  y_foot=0.068150 + 0.008 + 0.003496,
                  sgn_t=+1, sgn_c=+1, sgn_s=+1, sgn_b=-1, hip_ax=-1),
    "fr": LegGeom("fr", (0.378300, -0.098, 0.0), (-0.116000, -0.0101),
                  y_foot=-0.068150 - 0.008 + 0.012500,
                  sgn_t=-1, sgn_c=-1, sgn_s=-1, sgn_b=+1, hip_ax=-1),
    "rl": LegGeom("rl", (-0.389329, 0.098, 0.0), (0.127029, -0.0101),
                  y_foot=0.087150 - 0.011 + 0.003496,
                  sgn_t=+1, sgn_c=+1, sgn_s=+1, sgn_b=-1, hip_ax=+1),
    "rr": LegGeom("rr", (-0.372300, -0.098, 0.0), (0.110000, -0.0101),
                  y_foot=-0.068150 - 0.008 + 0.012500,
                  sgn_t=-1, sgn_c=-1, sgn_s=-1, sgn_b=+1, hip_ax=+1),
}

# XML 里每个 hinge 的 range 都是 ±1.57 rad
JOINT_LIMIT = 1.57


def _rot2(a: float) -> np.ndarray:
    """矢状面 (x, z) 内的逆时针旋转矩阵。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


# --- 矢状面四杆求解 -----------------------------------------------------------
def solve_linkage(theta_t: float, theta_c: float, leg: LegGeom):
    """解闭链：受驱角 (大腿, 曲柄) → 矢状面各点与被动角。

    返回 dict：
        K      膝点（腿系 x-z）
        A      闭链锚点
        toe    足端（腿系 x-z）
        a_t    大腿在平面内的绝对转角
        a_s    小腿在平面内的绝对转角
        theta_s / theta_b   被动关节角（可直接与 MuJoCo qpos 对比）
        margin 两圆交点的半弦长 h —— 越接近 0 越靠近奇异位形（分支翻转边界）

    机构无法装配（两圆不相交）时 h 取 0，返回的是最接近的退化解，不抛异常。
    """
    O = np.asarray(leg.pivot_xz, dtype=np.float64)
    a_t = leg.sgn_t * theta_t
    a_c = leg.sgn_c * theta_c

    K = O + _rot2(a_t) @ R_T
    P1 = O + _rot2(a_c) @ R_A

    lb, ls = np.linalg.norm(R_B), np.linalg.norm(R_S)
    D = K - P1
    dd = float(np.linalg.norm(D))
    # 两圆交点：沿 P1→K 走 a，再沿法向偏 ±h
    a = (dd ** 2 + lb ** 2 - ls ** 2) / (2.0 * dd)
    h = np.sqrt(max(lb ** 2 - a ** 2, 0.0))
    u = D / dd
    perp = np.array([-u[1], u[0]])
    A = P1 + a * u + BRANCH * h * perp

    # 小腿绝对转角：把 r_s 转到 (A − K) 上
    v = A - K
    a_s = np.arctan2(v[1], v[0]) - np.arctan2(R_S[1], R_S[0])
    a_s = np.arctan2(np.sin(a_s), np.cos(a_s))

    # 连杆绝对转角：把 r_b 转到 (A − P1) 上
    w = A - P1
    a_b = np.arctan2(w[1], w[0]) - np.arctan2(R_B[1], R_B[0])
    a_b = np.arctan2(np.sin(a_b), np.cos(a_b))

    toe = K + _rot2(a_s) @ R_TOE_XZ

    return {
        "K": K, "A": A, "toe": toe, "a_t": a_t, "a_s": a_s,
        # 被动关节是相对父体的角，故减去父体绝对转角再除符号
        "theta_s": (a_s - a_t) / leg.sgn_s,
        "theta_b": (a_b - a_c) / leg.sgn_b,
        "margin": float(h),
    }


def leg_fk(theta_hip: float, theta_t: float, theta_c: float, leg: LegGeom,
           *, frame: str = "base") -> np.ndarray:
    """单腿正运动学：三个受驱关节角 → 足端位置 (x, y, z)。

    frame="leg" 返回腿系（髋轴原点、机身姿态对齐）；"base" 再加腿根挂载点。
    """
    sol = solve_linkage(theta_t, theta_c, leg)
    x, z = sol["toe"]
    y = leg.y_foot

    # 髋绕 x 轴：轴向为 (hip_ax, 0, 0)，等效绕 +x 转 hip_ax·θ
    phi = leg.hip_ax * theta_hip
    c, s = np.cos(phi), np.sin(phi)
    foot = np.array([x, c * y - s * z, s * y + c * z])

    if frame == "base":
        foot = foot + np.asarray(leg.leg_pos)
    elif frame != "leg":
        raise ValueError(f"frame 必须为 'leg' 或 'base'，收到 {frame!r}")
    return foot


# --- 数值 IK ------------------------------------------------------------------
def leg_ik(foot, leg: LegGeom, *, frame: str = "base",
           seed: tuple[float, float] | None = None,
           tol: float = 1e-10, max_iter: int = 60):
    """单腿逆运动学：足端位置 → (theta_hip, theta_t, theta_c)。

    髋角是解析的：矢状链不改变腿系 y（恒为 leg.y_foot），所以绕 x 的旋转由
    足端的 (y, z) 唯一确定；剩下 (θ_t, θ_c) 是 2 元 2 式，用阻尼牛顿迭代解，
    Jacobian 取解析 FK 的中心差分（FK 很便宜，差分足够准且不易写错）。

    返回 (theta_hip, theta_t, theta_c, info)，info 含 `converged` / `residual` /
    `margin`。不可达时返回最后一次迭代结果并置 converged=False，不抛异常。
    """
    p = np.asarray(foot, dtype=np.float64)
    if frame == "base":
        p = p - np.asarray(leg.leg_pos)
    elif frame != "leg":
        raise ValueError(f"frame 必须为 'leg' 或 'base'，收到 {frame!r}")

    px, py, pz = p
    y0 = leg.y_foot

    # --- 髋：绕 x 旋转保持 x 与半径不变 ---
    r_sq = py ** 2 + pz ** 2
    z_s = -np.sqrt(max(r_sq - y0 ** 2, 0.0))          # 矢状面内的 z（足端在下方）
    phi = np.arctan2(pz, py) - np.arctan2(z_s, y0)
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    theta_hip = phi / leg.hip_ax

    # --- 矢状面：牛顿解 (θ_t, θ_c) 命中 (px, z_s) ---
    target = np.array([px, z_s])
    eps = 1e-7

    def newton(q0):
        q = np.array(q0, dtype=np.float64)
        res = np.inf
        for _ in range(max_iter):
            err = target - solve_linkage(q[0], q[1], leg)["toe"]
            res = float(np.linalg.norm(err))
            if res < tol:
                return q, res, True
            J = np.empty((2, 2))
            for j in range(2):
                dq = np.zeros(2)
                dq[j] = eps
                J[:, j] = (solve_linkage(*(q + dq), leg)["toe"]
                           - solve_linkage(*(q - dq), leg)["toe"]) / (2 * eps)
            # Levenberg 阻尼：奇异位形（两圆相切，h→0）附近 J 病态，避免步长炸掉
            lam = 1e-9 + 1e-3 * res
            step = np.linalg.solve(J.T @ J + lam * np.eye(2), J.T @ err)
            q = np.clip(q + step, -JOINT_LIMIT, JOINT_LIMIT)
        return q, res, False

    # 多起点：靠近奇异位形时单起点会卡在 h=0 边界上（该方向梯度为 0），
    # 换个初值几乎总能绕开。第一个起点用调用方给的延拓值。
    starts = [seed if seed is not None else (0.0, 0.0),
              (0.0, 0.0), (0.3, -0.3), (-0.3, 0.3), (0.5, 0.5), (-0.5, -0.5)]
    best_q, best_res = None, np.inf
    for s in starts:
        q, res, done = newton(s)
        if res < best_res:
            best_q, best_res = q, res
        if done:
            break

    sol = solve_linkage(best_q[0], best_q[1], leg)
    info = {"converged": best_res < tol * 1e3, "residual": best_res,
            "margin": sol["margin"]}
    return float(theta_hip), float(best_q[0]), float(best_q[1]), info


def leg_jacobian(theta_hip: float, theta_t: float, theta_c: float,
                 leg: LegGeom, *, eps: float = 1e-7) -> np.ndarray:
    """足端 Jacobian J = ∂foot/∂(hip, thigh, crank)，3×3，中心差分。

    满足 foot_dot = J · q_dot；τ = Jᵀ F 可用于重力前馈（go2_walk.py 的做法）。
    """
    q = np.array([theta_hip, theta_t, theta_c], dtype=np.float64)
    J = np.empty((3, 3))
    for j in range(3):
        dq = np.zeros(3)
        dq[j] = eps
        J[:, j] = (leg_fk(*(q + dq), leg, frame="leg")
                   - leg_fk(*(q - dq), leg, frame="leg")) / (2 * eps)
    return J


# --- 整机封装 -----------------------------------------------------------------
def quad_fk(q: np.ndarray, *, frame: str = "base") -> np.ndarray:
    """整机 FK。q: (4, 3)，顺序 [hip, thigh, crank] × [fl, fr, rl, rr]。→ (4, 3)"""
    q = np.asarray(q, dtype=np.float64)
    return np.stack([leg_fk(*q[i], LEGS[n], frame=frame)
                     for i, n in enumerate(LEG_NAMES)])


def quad_ik(feet: np.ndarray, *, frame: str = "base",
            seed: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    """整机 IK。feet: (4, 3) → (q (4,3), [info×4])。

    seed: (4, 2) 的 (θ_t, θ_c) 初值。步态里应传上一控制周期的解做延拓——
    既快，又保证解连续（不会在两个分支/解之间跳）。
    """
    feet = np.asarray(feet, dtype=np.float64)
    out, infos = [], []
    for i, n in enumerate(LEG_NAMES):
        s = None if seed is None else tuple(seed[i])
        h, t, c, info = leg_ik(feet[i], LEGS[n], frame=frame, seed=s)
        out.append([h, t, c])
        infos.append(info)
    return np.array(out), infos


# --- 自测 ---------------------------------------------------------------------
def _mujoco_reference():
    """用 MuJoCo 的约束解做独立参照：给定受驱角，最小二乘解被动角，返回各点。

    与本模块完全独立（3D 数值 vs 2D 闭式），因此是有效的交叉验证。
    没装 mujoco/scipy 时返回 None。
    """
    try:
        import mujoco
        from scipy.optimize import least_squares
    except ImportError:
        return None

    import pathlib
    mjcf = pathlib.Path(__file__).resolve().parents[2] / "deploy/mujoco/assets/mos2026_2.xml"
    if not mjcf.exists():
        return None
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)

    def adr(n):
        return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]

    def ref(leg, hip, thigh, crank):
        crank_j = "fl_shank_link" if leg == "fl" else f"{leg}_shank_link_a"
        ia = [adr(f"{leg}_hip"), adr(f"{leg}_thigh"), adr(crank_j),
              adr(f"{leg}_shank_link_b"), adr(f"{leg}_shank")]
        e = [i for i in range(m.neq)
             if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) == f"{leg}_close_loop"][0]
        b1, b2 = m.eq_obj1id[e], m.eq_obj2id[e]

        def resid(passive):
            d.qpos[:] = 0
            d.qpos[3:7] = [1, 0, 0, 0]
            d.qpos[ia[0]], d.qpos[ia[1]], d.qpos[ia[2]] = hip, thigh, crank
            d.qpos[ia[3]], d.qpos[ia[4]] = passive
            mujoco.mj_kinematics(m, d)
            p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[e][0:3]
            p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[e][3:6]
            return p1 - p2

        r = least_squares(resid, np.zeros(2), xtol=1e-14, ftol=1e-14)
        resid(r.x)  # 用解重置一次，保证 d 里是收敛位形
        shank_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_shank")
        return {
            "theta_b": r.x[0], "theta_s": r.x[1],
            "knee": d.xpos[shank_bid].copy(),
            "R_shank": d.xmat[shank_bid].reshape(3, 3).copy(),
            "eq_resid": float(np.linalg.norm(resid(r.x))),
        }

    return ref, m


def _settle_check():
    """MuJoCo 里以零位姿站稳，用稳态实际关节角做 FK，返回每腿足端的世界 z。

    理想值 0（刚好触地）。返回 None 表示环境不具备。
    """
    try:
        import mujoco
    except ImportError:
        return None
    import pathlib
    scene = pathlib.Path(__file__).resolve().parents[2] / "deploy/mujoco/assets/scene.xml"
    if not scene.exists():
        return None

    m = mujoco.MjModel.from_xml_path(str(scene))
    m.opt.timestep = 1e-3
    for i in range(m.nu):        # 部署增益：kp=25 的软伺服撑不住自重
        m.actuator_gainprm[i, 0] = 320
        m.actuator_biasprm[i, 1] = -320
        m.actuator_biasprm[i, 2] = -3.2
    d = mujoco.MjData(m)
    d.qpos[2] = 0.35

    act = ["fl_hip", "fr_hip", "rl_hip", "rr_hip",
           "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
           "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a"]
    ci = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + n) for n in act]
    qi = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in act]
    d.ctrl[ci] = 0.0
    for _ in range(8000):
        mujoco.mj_step(m, d)

    q = d.qpos[qi]
    return {n: float(d.qpos[2] + leg_fk(q[i], q[4 + i], q[8 + i],
                                        LEGS[n], frame="base")[2])
            for i, n in enumerate(LEG_NAMES)}


def _selftest() -> int:
    rng = np.random.default_rng(0)
    ok = True
    np.set_printoptions(precision=6, suppress=True)

    print("== 0. 连杆尺寸 ==")
    for n, v in [("r_a 曲柄", R_A), ("r_b 连杆", R_B), ("r_t 大腿", R_T), ("r_s 小腿支耳", R_S)]:
        print(f"  {n}: {v}  |·| = {np.linalg.norm(v):.6f} m")
    print(f"  足端(小腿系 x-z): {R_TOE_XZ}  离膝 {np.linalg.norm(R_TOE_XZ):.6f} m")

    ref_pack = _mujoco_reference()
    if ref_pack is None:
        print("\n⚠️ 未找到 mujoco/scipy 或 MJCF，跳过交叉验证，只跑自洽性检查")
    else:
        ref, _ = ref_pack
        print("\n== 1. 解析 FK vs MuJoCo 约束解（被动角 + 膝点 + 小腿姿态）==")
        max_pa, max_kn, max_ro = 0.0, 0.0, 0.0
        for name, leg in LEGS.items():
            for _ in range(40):
                hip = rng.uniform(-0.4, 0.4)
                th = rng.uniform(-0.6, 0.6)
                cr = rng.uniform(-0.6, 0.6)
                mine = solve_linkage(th, cr, leg)
                r = ref(name, hip, th, cr)
                if r["eq_resid"] > 1e-8:      # 参照本身没收敛，跳过
                    continue
                max_pa = max(max_pa, abs(mine["theta_s"] - r["theta_s"]),
                             abs(mine["theta_b"] - r["theta_b"]))
                # 膝点：本模块在腿系，MuJoCo 在世界系（base 在原点）→ 加腿根、绕髋
                x, z = mine["K"]
                phi = leg.hip_ax * hip
                c, s = np.cos(phi), np.sin(phi)
                yk = KNEE_Y[name]   # 膝在腿系的 y = thigh 挂载 y + shank 挂载 y
                k_mine = np.array([x, c * yk - s * z, s * yk + c * z]) + np.asarray(leg.leg_pos)
                max_kn = max(max_kn, float(np.linalg.norm(k_mine - r["knee"])))
                # 小腿姿态：世界系 R = R_x(髋) · R_y(−a_s)
                # （平面 CCW 角 a_s 对应绕 +y 转 −a_s；髋不能漏，否则差 ~3e-2 rad）
                ca, sa = np.cos(-mine["a_s"]), np.sin(-mine["a_s"])
                Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
                Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                max_ro = max(max_ro, float(np.abs(Rx @ Ry - r["R_shank"]).max()))
        print(f"  被动关节角  max_err = {max_pa:.3e} rad")
        print(f"  膝点位置    max_err = {max_kn:.3e} m")
        print(f"  小腿姿态阵  max_err = {max_ro:.3e}")
        ok &= max_pa < 1e-6 and max_kn < 1e-9 and max_ro < 1e-6

    print("\n== 2. FK→IK→FK 往返（每腿 500 个随机位形，margin > 10 mm）==")
    # margin 是两圆交点的半弦长：→0 即奇异位形（装配分支的折叠边界）。
    # 这一带 FK 有 sqrt 型分支点、导数发散，IK 本身病态，不属于可用工作域，
    # 故按 margin 筛掉后再断言；边界行为单列到 2b 量化。
    max_p, max_q, n_bad = 0.0, 0.0, 0
    for name, leg in LEGS.items():
        n = 0
        while n < 500:
            hip = rng.uniform(-0.4, 0.4)
            th = rng.uniform(-0.5, 0.5)
            cr = rng.uniform(-0.5, 0.5)
            if solve_linkage(th, cr, leg)["margin"] < 0.010:
                continue
            n += 1
            foot = leg_fk(hip, th, cr, leg, frame="base")
            h2, t2, c2, info = leg_ik(foot, leg, frame="base", seed=(th * 0.5, cr * 0.5))
            if not info["converged"]:
                n_bad += 1
                continue
            foot2 = leg_fk(h2, t2, c2, leg, frame="base")
            max_p = max(max_p, float(np.linalg.norm(foot - foot2)))
            max_q = max(max_q, abs(h2 - hip), abs(t2 - th), abs(c2 - cr))
    print(f"  足端往返 max_err = {max_p:.3e} m   关节角 max_err = {max_q:.3e} rad"
          f"   未收敛 {n_bad}/2000")
    ok &= max_p < 1e-8 and max_q < 1e-6 and n_bad == 0

    print("\n== 2b. 奇异边界附近的 IK 成功率（按 margin 分档）==")
    bands = [(0.000, 0.005), (0.005, 0.010), (0.010, 0.020), (0.020, 0.050)]
    for lo, hi in bands:
        tot = fail = 0
        for name, leg in LEGS.items():
            n = 0
            while n < 150:
                th, cr = rng.uniform(-0.6, 0.6, 2)
                if not (lo <= solve_linkage(th, cr, leg)["margin"] < hi):
                    continue
                n += 1
                tot += 1
                foot = leg_fk(0.0, th, cr, leg, frame="base")
                *_, info = leg_ik(foot, leg, frame="base")
                fail += not info["converged"]
        print(f"  margin ∈ [{lo * 1e3:5.1f}, {hi * 1e3:5.1f}) mm: "
              f"失败 {fail:3d}/{tot}  ({100 * fail / tot:.1f}%)")

    print("\n== 3. 零位与工作域 ==")
    for name, leg in LEGS.items():
        f0 = leg_fk(0, 0, 0, leg, frame="base")
        s0 = solve_linkage(0, 0, leg)
        print(f"  {name}: 零位足端(base) = {np.round(f0, 4)}  站高 = {-f0[2]:.4f} m"
              f"  奇异余量 h = {s0['margin']:.4f} m")

    if ref_pack is not None:
        print("\n== 3b. 足端点标定：MuJoCo 落地静置，用实际受驱角做 FK 对比地面 ==")
        # 这是唯一能验证 R_TOE_XZ 的检查（矢状链本身已被 §1 证明是精确的）。
        # 必须用稳态的**实际**关节角，不能用指令角——位置伺服有重力静差。
        err = _settle_check()
        if err is None:
            print("  （跳过：无 scene.xml）")
        else:
            for leg, e in err.items():
                print(f"  {leg}: 足端预测 z_world = {e:+.4f} m（地面=0，理想≈0）")
            worst = max(abs(e) for e in err.values())
            print(f"  最大偏差 {worst * 1e3:.1f} mm")
            ok &= worst < 0.004

    print("\n== 4. Jacobian vs 有限差分一致性（Jacobian 本身即差分，查对称性/量级）==")
    for name, leg in LEGS.items():
        J = leg_jacobian(0.0, 0.1, -0.1, leg)
        print(f"  {name}: ∂foot/∂thigh = {np.round(J[:, 1], 4)}  "
              f"∂foot/∂crank = {np.round(J[:, 2], 4)} m/rad")

    print("\n" + ("✅ 全部通过" if ok else "❌ 有用例失败"))
    return 0 if ok else 1


# 膝点在腿系的 y（thigh 挂载 y + shank 挂载 y），仅自测交叉验证用
KNEE_Y = {
    "fl": 0.068150 + 0.008,
    "fr": -0.068150 - 0.008,
    "rl": 0.087150 - 0.011,
    "rr": -0.068150 - 0.008,
}


def main() -> int:
    p = argparse.ArgumentParser(description="mos2026_2 闭链腿运动学（解析 FK + 数值 IK）")
    p.add_argument("--selftest", action="store_true", help="自洽性检查 + 与 MuJoCo 交叉验证")
    args = p.parse_args()
    if args.selftest:
        return _selftest()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
