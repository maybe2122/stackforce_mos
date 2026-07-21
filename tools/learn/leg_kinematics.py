"""Go2 单腿三连杆运动学：FK / Jacobian / IK。

坐标系约定（hip 关节原点，与 base 姿态对齐）：
    x 前，y 左，z 上

关节顺序 q = [q0, q1, q2]
    q0  abduction  绕 x 轴（外摆）
    q1  hip pitch  绕 y 轴（大腿）
    q2  knee       绕 y 轴（小腿，相对大腿）

几何取自 mujoco_menagerie/unitree_go2/go2.xml：
    thigh 相对 hip     (0, ±0.0955, 0)   -> d
    calf  相对 thigh   (0, 0, -0.213)    -> l1
    foot  相对 calf    (0, 0, -0.213)    -> l2
"""

import numpy as np

L1 = 0.213  # 大腿长
L2 = 0.213  # 小腿长
D = 0.0955  # abduction 横向偏置（左腿 +D，右腿 -D）


def fk(q, d=D, l1=L1, l2=L2):
    """正运动学：关节角 -> 足端位置。q -> x"""
    q0, q1, q2 = q
    s0, c0 = np.sin(q0), np.cos(q0)
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)

    # 先在腿平面（x-z）里解两连杆，此时 y 就是偏置 d
    x = -l1 * s1 - l2 * s12
    z_leg = -l1 * c1 - l2 * c12

    # 再绕 x 轴转 q0，把 (d, z_leg) 摆出去
    return np.array([
        x,
        c0 * d - s0 * z_leg,
        s0 * d + c0 * z_leg,
    ])


def jacobian(q, d=D, l1=L1, l2=L2):
    """解析雅可比 J(q)，3x3。

    正向：足端速度   ẋ = J q̇
    反向：关节力矩   τ = Jᵀ F     （F 是足端受到的力）
    """
    q0, q1, q2 = q
    s0, c0 = np.sin(q0), np.cos(q0)
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)

    z_leg = -l1 * c1 - l2 * c12

    # 腿平面内对 q1 / q2 的偏导
    dx_dq1 = -l1 * c1 - l2 * c12
    dz_dq1 = l1 * s1 + l2 * s12
    dx_dq2 = -l2 * c12
    dz_dq2 = l2 * s12

    J = np.zeros((3, 3))
    # 对 q0：只影响绕 x 的旋转，x 分量不变
    J[:, 0] = [0.0, -s0 * d - c0 * z_leg, c0 * d - s0 * z_leg]
    # 对 q1 / q2：腿平面内的量再经 R_x(q0) 旋转
    J[:, 1] = [dx_dq1, -s0 * dz_dq1, c0 * dz_dq1]
    J[:, 2] = [dx_dq2, -s0 * dz_dq2, c0 * dz_dq2]
    return J


def ik(p, d=D, l1=L1, l2=L2):
    """解析逆运动学：足端位置 -> 关节角。x -> q

    分支选择（两处，都是有意为之）：
      1. 膝关节向后弯 q2 < 0        —— 与 go2.xml 中 knee range 一致
      2. 腿平面内 z_leg < 0          —— 即腿挂在水平面以下

    第 2 条是必要的：给定足端位置 p，q0 有两个解（腿朝下 / 腿翻到上方），
    单靠 p 无法区分。行走时腿恒在身体下方，所以固定取朝下的分支。
    若腿真的折到水平面以上，本函数会返回另一个分支的解。
    """
    px, py, pz = p

    # --- q0：yz 平面内解外摆角 ---
    r_yz = np.hypot(py, pz)
    z_leg = -np.sqrt(max(r_yz**2 - d**2, 0.0))  # 腿平面内的 z，恒为负
    # py = r*cos(q0+φ), pz = r*sin(q0+φ)，其中 φ = atan2(z_leg, d)
    q0 = np.arctan2(pz, py) - np.arctan2(z_leg, d)
    q0 = np.arctan2(np.sin(q0), np.cos(q0))  # wrap 到 [-π, π]

    # --- q2：余弦定理 ---
    reach2 = px**2 + z_leg**2
    cos_q2 = (reach2 - l1**2 - l2**2) / (2 * l1 * l2)
    q2 = -np.arccos(np.clip(cos_q2, -1.0, 1.0))

    # --- q1：目标方向角 减去 两连杆夹角 ---
    alpha = np.arctan2(-px, -z_leg)
    beta = np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q1 = alpha - beta

    return np.array([q0, q1, q2])


def jacobian_numeric(q, d=D, l1=L1, l2=L2, eps=1e-6):
    """数值雅可比，仅用于验证解析式。"""
    J = np.zeros((3, 3))
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        J[:, i] = (fk(q + dq, d, l1, l2) - fk(q - dq, d, l1, l2)) / (2 * eps)
    return J


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1) home 位姿自检：q=[0, 0.9, -1.8]，足端应在正下方
    q_home = np.array([0.0, 0.9, -1.8])
    print("FK(home) =", fk(q_home))

    # 2) 解析雅可比 vs 数值雅可比
    max_err = 0.0
    for _ in range(200):
        q = np.array([
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.5, 3.4),
            rng.uniform(-2.7, -0.84),
        ])
        max_err = max(max_err, np.abs(jacobian(q) - jacobian_numeric(q)).max())
    print(f"Jacobian 解析 vs 数值  最大误差 = {max_err:.3e}")

    # 3) IK 往返：q -> FK -> IK -> q
    #    只在「腿朝下」分支内采样（见 ik 的 docstring），这是行走的工作区间
    max_err = 0.0
    for _ in range(200):
        q = np.array([
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.5, 2.0),
            rng.uniform(-2.7, -0.84),
        ])
        # 分支条件在腿平面的 z_leg 上，不是世界系的 pz
        z_leg = -L1 * np.cos(q[1]) - L2 * np.cos(q[1] + q[2])
        if z_leg > 0:  # 跳过腿翻到水平面以上的位形
            continue
        max_err = max(max_err, np.abs(ik(fk(q)) - q).max())
    print(f"IK 往返 q->x->q       最大误差 = {max_err:.3e}")

    # 4) τ = Jᵀ F：站立时足端向上支撑 ~65N（整机 ~15kg / 4 腿 * g）
    F = np.array([0.0, 0.0, 65.0])
    print("τ = JᵀF (home) =", jacobian(q_home).T @ F)
