#!/usr/bin/env python3
"""传统控制 baseline · 最小闭环:姿态 PD 主动站立平衡(MuJoCo 验证版)。

做什么:
  机身倾角(投影重力)→ 姿态 PD → 四条腿伸缩修正 → 12 关节目标 → 位置伺服。
  这是 todo「运控缺口 3」的第一步:白盒控制器打通 IMU→控制→执行整条链路,
  真机 IMU 接入后同一控制逻辑可直接搬到 rl_deploy 底层(M2)。

控制律(50 Hz):
  g_b = R^T·[0,0,-1]                      # 机身系投影重力(仿真 IMU;真机=姿态融合输出)
  Δz_i = -(kp·g_b + kd·ġ_b)·(sx_i, sy_i)  # 每腿足端高度修正:低的那侧腿伸长
  Δq_i = J_i⁺ · Δz_i                       # 数值雅可比伪逆 → (thigh, 曲柄) 修正量
  target = default_pose + Δq               # 叠加到站姿目标,clip 后下发

雅可比 J_i = ∂(足端z)/∂(thigh, 曲柄) 在站姿处数值求取:给定受驱角,闭链被动角用
connect 锚点重合的最小二乘解算(延拓自闭链分支扫描),再有限差分。不拍脑袋。

验证(--push 扫描):侧向速度冲量从小到大推机身,对比 开/关 平衡控制器的
最大倾角与是否倾覆,给出抗扰增益的量化结论。

用法:
  # headless 推搡对比实验(默认):无控制器 vs 有控制器 各扫一遍
  ../env_isaaclab/bin/python deploy/mujoco/stand_balance.py
  # 只跑有控制器 + viewer 直观看
  MUJOCO_GL=glx ../env_isaaclab/bin/python deploy/mujoco/stand_balance.py --viewer
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = REPO_ROOT / "deploy/mujoco/assets/mos2026_2.xml"

# 关节顺序与训练/部署一致(见 play_mujoco.py 注释)。
# 髋角与训练不同:MJCF/USD 里 hip 轴是前后镜像(fl/fr=-X, rl/rr=+X)而非左右镜像,
# 训练用的 [0.15,0.15,-0.15,-0.15] 实际是四足整体右移 ~17mm(策略自己补偿了);
# 这里是白盒控制器,直接用左右对称外撇的站姿:左腿符号取反。
ACTUATED = ["fl_hip", "fr_hip", "rl_hip", "rr_hip",
            "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
            "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a"]
DEFAULT_POSE = np.array([-0.15, 0.15, 0.15, -0.15, 0, 0, 0, 0, 0, 0, 0, 0])
LEGS = ["fl", "fr", "rl", "rr"]
FOOT_BODY = {leg: f"{leg}_shank" for leg in LEGS}          # 触地的足端 body
THIGH_IDX = {leg: 4 + i for i, leg in enumerate(LEGS)}     # DEFAULT_POSE 下标
CRANK_IDX = {leg: 8 + i for i, leg in enumerate(LEGS)}
CONTROL_HZ = 50
SIM_HZ = 1000


class ClosedChainKin:
    """闭链腿运动学:给定 (thigh, 曲柄) 受驱角,最小二乘解被动角,返回足端位置。

    被动角 = f(thigh, crank) 双输入(五杆机构),MuJoCo 的 connect 锚点重合即约束。
    以站姿解为初值做局部求解,保证停留在正确装配分支。
    """

    def __init__(self, model):
        from scipy.optimize import least_squares
        self._lsq = least_squares
        self.model = model
        self.data = mujoco.MjData(model)
        self.jadr = {}
        for leg in LEGS:
            crank = f"{leg}_shank_link" if leg == "fl" else f"{leg}_shank_link_a"
            names = (f"{leg}_hip", f"{leg}_thigh", crank,
                     f"{leg}_shank_link_b", f"{leg}_shank")
            self.jadr[leg] = [self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        self.eq_id = {leg: [i for i in range(model.neq)
                            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
                            == f"{leg}_close_loop"][0] for leg in LEGS}
        self.foot_bid = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                                FOOT_BODY[leg]) for leg in LEGS}
        self._passive_seed = {leg: np.zeros(2) for leg in LEGS}

    def _anchor_sep(self, leg, hip, thigh, crank, passive):
        a_hip, a_thigh, a_crank, a_pb, a_ps = self.jadr[leg]
        self.data.qpos[:] = 0
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[a_hip] = hip
        self.data.qpos[a_thigh] = thigh
        self.data.qpos[a_crank] = crank
        self.data.qpos[a_pb], self.data.qpos[a_ps] = passive
        mujoco.mj_kinematics(self.model, self.data)
        e = self.eq_id[leg]
        b1, b2 = self.model.eq_obj1id[e], self.model.eq_obj2id[e]
        a1, a2 = self.model.eq_data[e][0:3], self.model.eq_data[e][3:6]
        p1 = self.data.xpos[b1] + self.data.xmat[b1].reshape(3, 3) @ a1
        p2 = self.data.xpos[b2] + self.data.xmat[b2].reshape(3, 3) @ a2
        return p1 - p2

    def foot_pos(self, leg, hip, thigh, crank):
        """求解闭链后,足端 body 在机身系的位置(base 在原点、姿态单位阵)。"""
        r = self._lsq(lambda p: self._anchor_sep(leg, hip, thigh, crank, p),
                      self._passive_seed[leg], xtol=1e-12, ftol=1e-12)
        self._passive_seed[leg] = r.x        # 延拓:下次以本次解为初值
        return self.data.xpos[self.foot_bid[leg]].copy()

    def foot_z_jacobian(self, leg, hip, thigh, crank, h=0.02):
        """J = [∂z/∂thigh, ∂z/∂crank](中心差分)。"""
        jz = []
        for dq in ((h, 0), (0, h)):
            zp = self.foot_pos(leg, hip, thigh + dq[0], crank + dq[1])[2]
            zm = self.foot_pos(leg, hip, thigh - dq[0], crank - dq[1])[2]
            jz.append((zp - zm) / (2 * h))
        return np.array(jz)


class StandBalance:
    """姿态 PD 站立平衡控制器(白盒,~30 行控制逻辑)。

    抗抖三件套(2026-07-05,修「站立高频震动」):
      g_alpha  姿态一阶低通(50 Hz 下 0.25 ≈ 截止 2.3 Hz),触地冲击不进控制;
      kd 作用在滤波后的 g 上,不再放大帧间跳变;
      slew     输出限斜率(rad/控制步),关节指令不允许突跳。
    """

    def __init__(self, model, kp_att=0.3, kd_att=0.05, max_dq=0.25,
                 g_alpha=0.25, slew=0.02):
        self.kp_att, self.kd_att, self.max_dq = kp_att, kd_att, max_dq
        self.g_alpha, self.slew = g_alpha, slew
        kin = ClosedChainKin(model)
        # 站姿处:每腿足端 (x,y) 符号(前/左=+)与 z 雅可比
        self.sx, self.sy, self.J, self.Jpinv = {}, {}, {}, {}
        for i, leg in enumerate(LEGS):
            hip = DEFAULT_POSE[i]
            p = kin.foot_pos(leg, hip, 0.0, 0.0)
            self.sx[leg] = np.sign(p[0])
            self.sy[leg] = np.sign(p[1])
            J = kin.foot_z_jacobian(leg, hip, 0.0, 0.0)
            self.J[leg] = J
            self.Jpinv[leg] = J / (J @ J)        # 最小范数伪逆(1x2)
        self.reset()
        print("[balance] 足端符号/雅可比标定:")
        for leg in LEGS:
            print(f"  {leg}: sx={self.sx[leg]:+.0f} sy={self.sy[leg]:+.0f} "
                  f"J=[{self.J[leg][0]:+.3f}, {self.J[leg][1]:+.3f}] m/rad")

    def reset(self):
        self._g_filt = np.array([0.0, 0.0, -1.0])
        self._g_prev = self._g_filt.copy()
        self._dq_prev = np.zeros(12)

    def compute(self, g_body, dt):
        """g_body: 机身系单位投影重力 → 返回 12 维关节目标修正量。"""
        self._g_filt = self._g_filt + self.g_alpha * (g_body - self._g_filt)
        g_dot = (self._g_filt - self._g_prev) / dt
        self._g_prev = self._g_filt.copy()
        u = self.kp_att * self._g_filt[:2] + self.kd_att * g_dot[:2]  # (ux, uy) 倾斜量
        dq = np.zeros(12)
        for i, leg in enumerate(LEGS):
            # 低的那侧伸腿:机头低(g_x>0)→前腿伸;左侧低(g_y>0)→左腿伸。
            # 伸腿 = 足端 z 更低(机身系),故 Δz 取负。
            dz = -(u[0] * self.sx[leg] + u[1] * self.sy[leg])
            d = self.Jpinv[leg] * dz                              # (Δthigh, Δcrank)
            dq[THIGH_IDX[leg]] = d[0]
            dq[CRANK_IDX[leg]] = d[1]
        dq = np.clip(dq, -self.max_dq, self.max_dq)
        # 分区限斜率:静立时温柔(高频分量到不了伺服,不抖);倾角大(>~3°)时
        # 放开到 4 倍斜率快速回中——抗扰靠这里,不靠拉高 kp(kp>0.3 会起振)。
        tilt = float(np.hypot(self._g_filt[0], self._g_filt[1]))
        slew = self.slew * (4.0 if tilt > 0.05 else 1.0)
        dq = np.clip(dq, self._dq_prev - slew, self._dq_prev + slew)
        self._dq_prev = dq
        return dq


def run(args):
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    model.opt.timestep = 1.0 / SIM_HZ
    # 部署增益(kp=25 的软伺服撑不住自重,见 play_mujoco.py 注释)
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = args.kp
        model.actuator_biasprm[i, 1] = -args.kp
        model.actuator_biasprm[i, 2] = -args.kv
    data = mujoco.MjData(model)
    qi = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in ACTUATED])
    ci = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}")
                   for n in ACTUATED])
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    ctrl = StandBalance(model, args.kp_att, args.kd_att) if args.balance else None

    data.qpos[2] = 0.30
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qi] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    decim = SIM_HZ // CONTROL_HZ
    dt = 1.0 / CONTROL_HZ

    def gravity_body():
        R = data.xmat[base].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def step_50hz():
        g = gravity_body()
        dq = ctrl.compute(g, dt) if ctrl else np.zeros(12)
        data.ctrl[ci] = DEFAULT_POSE + dq
        for _ in range(decim):
            mujoco.mj_step(model, data)
        return g

    # ── settle ──
    for _ in range(int(1.5 * CONTROL_HZ)):
        step_50hz()
    z0 = data.qpos[2]
    print(f"[info] settle 完成,站高 {z0:.3f} m,balance={'on' if ctrl else 'off'}")

    # ── 稳态抖动量化(2 s 静立):关节指令步变化量 + 倾角波动 ──
    ctrl_prev, dq_deltas, tilts = data.ctrl[ci].copy(), [], []
    for _ in range(int(2.0 * CONTROL_HZ)):
        g = step_50hz()
        dq_deltas.append(np.abs(data.ctrl[ci] - ctrl_prev).max())
        ctrl_prev = data.ctrl[ci].copy()
        tilts.append(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))
    print(f"[info] 稳态抖动:max|Δq_cmd|={np.mean(dq_deltas)*1000:.1f} mrad/步 "
          f"(>10 目测明显抖) | 倾角 {np.mean(tilts):.2f}°±{np.std(tilts):.2f}°")

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            print("[info] viewer:每 3 秒一次侧推(0.4 m/s),Esc 退出")
            k, tick = 0, time.time()
            while v.is_running():
                if k % (3 * CONTROL_HZ) == 0 and k > 0:
                    data.qvel[1] += 0.4
                step_50hz()
                v.sync()
                k += 1
                tick += dt
                if (s := tick - time.time()) > 0:
                    time.sleep(s)
        return 0

    # ── 推搡扫描:逐级加大侧向冲量,记录峰值倾角与是否倾覆 ──
    print(f"[push] {'冲量m/s':>8} {'峰值倾角°':>10} {'2s后倾角°':>10} {'结果':>6}")
    fell_at = None
    for push in args.pushes:
        # 每级重置到站姿再推,级间独立
        data.qpos[:] = 0
        data.qpos[2] = 0.30
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[qi] = DEFAULT_POSE
        data.qvel[:] = 0
        if ctrl:
            ctrl.reset()
        mujoco.mj_forward(model, data)
        for _ in range(int(1.0 * CONTROL_HZ)):
            step_50hz()
        data.qvel[1] += push                     # 侧向速度冲量
        max_tilt = 0.0
        for _ in range(int(2.0 * CONTROL_HZ)):
            g = step_50hz()
            tilt = np.degrees(np.arccos(np.clip(-g[2], -1, 1)))
            max_tilt = max(max_tilt, tilt)
        end_tilt = np.degrees(np.arccos(np.clip(-gravity_body()[2], -1, 1)))
        fell = data.qpos[2] < 0.12 or end_tilt > 45
        print(f"[push] {push:>8.2f} {max_tilt:>10.1f} {end_tilt:>10.1f} "
              f"{'倾覆' if fell else 'ok':>6}")
        if fell and fell_at is None:
            fell_at = push
    print(f"[result] balance={'on' if ctrl else 'off'}  "
          f"倾覆阈值 = {fell_at if fell_at is not None else f'>{args.pushes[-1]}'} m/s")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--kp", type=float, default=320.0, help="伺服 kp(关节侧,部署等效)")
    ap.add_argument("--kv", type=float, default=3.2, help="伺服 kv")
    ap.add_argument("--kp_att", type=float, default=0.3, help="姿态 PD 比例(m/单位重力分量)")
    ap.add_argument("--kd_att", type=float, default=0.05, help="姿态 PD 微分")
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                    help="开/关平衡控制器(--no-balance 跑对照组)")
    ap.add_argument("--viewer", action="store_true", help="开窗观看(每 3s 自动侧推)")
    ap.add_argument("--pushes", type=float, nargs="+",
                    default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
                    help="推搡扫描的侧向冲量序列 m/s")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
