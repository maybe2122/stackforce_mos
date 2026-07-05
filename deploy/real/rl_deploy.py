#!/usr/bin/env python3
"""50 Hz RL 推理主控节点 — mos2026_2 实机部署

从 motor_ctrl servo 进程的 FB 行读取关节状态，组建 45 维观测，运行 TorchScript
policy，把动作转换为旋子目标角并写回 servo stdin，实现闭环 RL 控制。

前置条件：
  1. 机器人已通过 robot_web.py 站立稳定（servo 进程已由 robot_web 释放，或从未启动）
  2. policy.pt 已用 policy_export.py 导出
  3. stand_config.json 已标定（motor_control/config/stand_config.json）

用法：
  python deploy/real/rl_deploy.py
  python deploy/real/rl_deploy.py --policy deploy/real/policy/policy.pt
  python deploy/real/rl_deploy.py --hold_secs 5.0          # 站姿保持 5s 再切 RL
  python deploy/real/rl_deploy.py --no_rl                  # 只保持站姿，不跑 policy（调试用）

观测合约（45 维，与 mos2026_2_closed_usd_env 完全一致）：
  lin_vel(3) + ang_vel(3) + gravity_vec(3) + dof_pos(12) + dof_vel(12) + prev_action(12)

坐标转换（per joint）：
  rotor_target = stand_rotor + GEAR * dir * sim_sign * (q_sim_target - default_dof_pos)
  q_sim_meas   = default_dof_pos + (rotor_meas - stand_rotor) / (GEAR * dir * sim_sign)
  qdot_sim     = rotor_vel / (GEAR * dir * sim_sign)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# ── 路径 ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_POLICY = HERE / "policy" / "policy.pt"
DEFAULT_CONFIG = HERE / "config" / "mos2026_2.yaml"
DEFAULT_STAND_CONFIG = REPO_ROOT / "motor_control" / "config" / "stand_config.json"
MOTOR_CTRL = str(REPO_ROOT / "motor_control" / "Linux" / "build" / "motor_ctrl")

GEAR_RATIO = 6.33
CTRL_HZ = 50
CTRL_DT = 1.0 / CTRL_HZ
MAX_MOTOR_TEMP = 80  # °C，GO-M8010-6 过温急停阈值

# policy 关节顺序（与 mos2026_2.yaml joint_names 一致）
JOINT_NAMES_POLICY = [
    "fl_hip", "fr_hip", "rl_hip", "rr_hip",
    "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
    "fl_shank", "fr_shank", "rl_shank", "rr_shank",
]
NUM_DOFS = 12

# ── FB 行解析 ─────────────────────────────────────────────────────────────────
_FB_PAT = re.compile(
    r"FB\s+id=(\d+)"
    r"\s+pos=(-?[\d.eE+\-]+)"
    r"\s+vel=(-?[\d.eE+\-]+)"
    r"\s+tau=(-?[\d.eE+\-]+)"
    r"\s+temp=(\d+)"
    r"\s+merr=(\d+)"
    r"\s+ok=([01])"
    r"\s+errd=(-?[\d.eE+\-]+)"
)


def _parse_fb(line: str) -> Optional[dict]:
    m = _FB_PAT.search(line)
    if not m:
        return None
    return {
        "id":   int(m.group(1)),
        "pos":  float(m.group(2)),   # 旋子角 rad
        "vel":  float(m.group(3)),   # 旋子角速度 rad/s
        "tau":  float(m.group(4)),
        "temp": int(m.group(5)),
        "merr": int(m.group(6)),
        "ok":   m.group(7) == "1",
        "errd": float(m.group(8)),   # 跟踪误差（关节°）
    }


# ── 关节元数据 ─────────────────────────────────────────────────────────────────
class JointMeta:
    __slots__ = ("name", "port", "motor_id", "dir", "stand_rotor", "sim_sign")

    def __init__(self, name: str, port: str, motor_id: int,
                 dir_: int, stand_rotor: float, sim_sign: int = 1):
        self.name = name
        self.port = port
        self.motor_id = motor_id
        self.dir = dir_
        self.stand_rotor = stand_rotor
        self.sim_sign = sim_sign

    def _factor(self) -> float:
        return GEAR_RATIO * self.dir * self.sim_sign

    def rotor_to_sim(self, rotor: float, default_q: float) -> float:
        return default_q + (rotor - self.stand_rotor) / self._factor()

    def vel_rotor_to_sim(self, rotor_vel: float) -> float:
        return rotor_vel / self._factor()

    def sim_to_rotor(self, q_sim: float, default_q: float) -> float:
        return self.stand_rotor + self._factor() * (q_sim - default_q)


# ── servo 子进程封装 ──────────────────────────────────────────────────────────
class ServoProc:
    """封装单路串口的 motor_ctrl servo 进程，异步读取 FB 反馈。"""

    def __init__(self, port: str, kp: float, kw: float, fb_ms: int = 20):
        self._port = port
        self._lock = threading.Lock()
        self._latest: Dict[int, dict] = {}   # motor_id → 最新 FB 字典
        self._alive = True

        env = dict(os.environ)
        env["MOTOR_CTRL_FB_MS"] = str(fb_ms)
        cmd = [MOTOR_CTRL, port, "servo", f"{kp:.4f}", f"{kw:.4f}"]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        print(f"[servo] port={port} kp={kp:.4f} kw={kw:.4f} pid={self._proc.pid}")

    def _reader(self):
        for raw in self._proc.stdout:
            line = raw.rstrip()
            fb = _parse_fb(line)
            if fb:
                with self._lock:
                    self._latest[fb["id"]] = fb
            elif line and not line.startswith("FB"):
                print(f"  [{self._port.split('/')[-1]}] {line}", flush=True)

    def get(self, motor_id: int) -> Optional[dict]:
        with self._lock:
            return self._latest.get(motor_id)

    def send(self, cmds: List[Tuple[int, float, float]]) -> bool:
        """写一行指令：[(motor_id, rotor_pos_rad, t_ff_Nm), ...]"""
        if self._proc.poll() is not None:
            return False
        line = " ".join(f"{mid} {pos:.5f} {tff:.4f}" for mid, pos, tff in cmds)
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def stop(self, timeout: float = 1.0):
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write("stop\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        # 等待 servo 发完 mode=0 停止脉冲
        try:
            self._proc.wait(timeout=timeout + 0.4)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
        self._alive = False


# ── RL 部署主类 ───────────────────────────────────────────────────────────────
class RLDeploy:
    def __init__(self, args: argparse.Namespace):
        # 加载 yaml 配置
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        cfg = raw["mos2026_2"]
        hw = raw["hardware"]

        self.default_dof_pos = np.array(cfg["default_dof_pos"], dtype=np.float32)
        self.action_scale = np.array(cfg["action_scale"], dtype=np.float32)
        self.clip_lo = np.array(cfg["clip_actions_lower"], dtype=np.float32)
        self.clip_hi = np.array(cfg["clip_actions_upper"], dtype=np.float32)
        self.clip_obs = float(cfg.get("clip_obs", 100.0))
        # 关节侧力矩上限（= 训练 effort_limit_sim）。超限说明真机动力学已经
        # 偏离训练分布，只告警不截断——位置伺服无法直接限力矩，截断目标角反而危险。
        self.torque_limits = np.array(
            cfg.get("torque_limits", [16.0] * NUM_DOFS), dtype=np.float32)

        # K_P/K_W 必须用 hardware.motor_kp/motor_kw（转子侧），不能用 rl_kp/rl_kd！
        # motor_ctrl servo 的 kp/kw 写进 Unitree SDK 的 cmd.K_P/K_W，作用在【转子侧】：
        # 若直接下发关节侧 rl_kp=25，等效关节刚度 = 25*6.33^2 ≈ 1000 N·m/rad，
        # 是训练值(25)的 40 倍，极硬且危险。换算关系见 yaml hardware 块注释：
        #   精确复现训练关节刚度: motor_kp = rl_kp/N^2 ≈ 0.62（可能软到撑不住）
        #   已验证可站立的保守值: motor_kp = 8.0（默认）
        self._kp = float(hw.get("motor_kp", 8.0))
        self._kw = float(hw.get("motor_kw", 0.08))

        self._imu_source = hw.get("imu_source", "stub")
        self._lin_vel_source = hw.get("lin_vel_source", "zero")

        # 加载 stand_config.json，构建 policy 顺序的关节元数据列表
        with open(args.stand_config) as f:
            sc = json.load(f)
        sc_by_name = {j["name"]: j for j in sc["joints"]}
        hw_by_name = {j["name"]: j for j in hw.get("joints", [])}

        self.joints: List[JointMeta] = []
        for name in JOINT_NAMES_POLICY:
            jd = sc_by_name[name]
            hw_j = hw_by_name.get(name, {})
            self.joints.append(JointMeta(
                name=name,
                port=jd["port"],
                motor_id=jd["id"],
                dir_=jd["dir"],
                stand_rotor=jd["stand_rotor"],
                sim_sign=hw_j.get("sim_sign", 1),
            ))

        # 每路串口对应的关节列表
        self._port_joints: Dict[str, List[JointMeta]] = {}
        for jm in self.joints:
            self._port_joints.setdefault(jm.port, []).append(jm)

        # 加载 TorchScript policy
        print(f"[rl_deploy] 加载 policy: {args.policy}")
        self._policy = torch.jit.load(str(args.policy), map_location="cpu")
        self._policy.eval()
        print(f"[rl_deploy] policy 已加载")

        # 动作低通滤波 α∈(0,1]：执行目标用 α*新动作+(1-α)*上次，1=不滤波。
        # 只滤执行路径；obs 里的 prev_action 仍是策略原始输出（契约不变）。
        # MuJoCo 实测 α=0.3 可明显压高频抖动（kp=320 下存活 0.12s→1.32s）。
        self._action_lpf = float(args.action_lpf)
        self._filt_action = np.zeros(NUM_DOFS, dtype=np.float32)

        # 运行时状态
        self._prev_action = np.zeros(NUM_DOFS, dtype=np.float32)
        self._shutdown = threading.Event()
        self._servos: Dict[str, ServoProc] = {}
        self._motor_fault: Optional[str] = None   # 过温/驱动错误 → 主循环急停
        self._last_tau_warn = 0.0                 # 力矩超限告警限频（1 Hz）

        self._hold_secs = float(args.hold_secs)
        self._no_rl = args.no_rl
        self._max_pr = float(args.max_pitch_roll)  # 姿态超限阈值 rad
        self._verbose = args.verbose

    # ── servo 进程管理 ────────────────────────────────────────────────────────
    def _start_servos(self):
        for port in self._port_joints:
            self._servos[port] = ServoProc(port, self._kp, self._kw)
        time.sleep(0.5)   # 等待进程启动 + sudo 鉴权

    def _stop_servos(self):
        for sp in self._servos.values():
            sp.stop()
        self._servos.clear()

    def _send_stand_hold(self):
        """所有关节保持在 stand_rotor 位置（零前馈）。"""
        for port, jms in self._port_joints.items():
            sp = self._servos.get(port)
            if sp:
                sp.send([(jm.motor_id, jm.stand_rotor, 0.0) for jm in jms])

    def _send_rotor_targets(self, rotor_targets: np.ndarray):
        """按 policy 顺序的旋子目标角下发给对应 servo 进程。"""
        port_cmds: Dict[str, List[Tuple[int, float, float]]] = {p: [] for p in self._port_joints}
        for i, jm in enumerate(self.joints):
            port_cmds[jm.port].append((jm.motor_id, float(rotor_targets[i]), 0.0))
        for port, cmds in port_cmds.items():
            sp = self._servos.get(port)
            if sp:
                sp.send(cmds)

    # ── 状态读取 ──────────────────────────────────────────────────────────────
    def _read_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """从 servo FB 读取最新关节位置/速度（policy 顺序，sim 坐标系）。
        未收到数据的关节使用 default_dof_pos / 0 速度（安全保守值）。
        """
        dof_pos = self.default_dof_pos.copy()
        dof_vel = np.zeros(NUM_DOFS, dtype=np.float32)
        no_data = []
        for i, jm in enumerate(self.joints):
            sp = self._servos.get(jm.port)
            if sp is None:
                no_data.append(jm.name)
                continue
            fb = sp.get(jm.motor_id)
            if fb is None:
                no_data.append(jm.name)
                continue
            # 电机健康监护：过温 / 驱动错误码 → 置故障标志，主循环急停
            if fb["temp"] >= MAX_MOTOR_TEMP or fb["merr"] != 0:
                self._motor_fault = (
                    f"{jm.name}(id={jm.motor_id}): temp={fb['temp']}°C merr={fb['merr']}")
            # 关节侧力矩 = 转子侧 tau × 减速比；超训练上限说明已偏离训练分布
            tau_joint = abs(fb["tau"]) * GEAR_RATIO
            if tau_joint > self.torque_limits[i]:
                now = time.monotonic()
                if now - self._last_tau_warn > 1.0:
                    self._last_tau_warn = now
                    print(f"[warn] {jm.name} 关节力矩 {tau_joint:.1f} N·m "
                          f"超训练上限 {self.torque_limits[i]:.0f} N·m")
            dof_pos[i] = jm.rotor_to_sim(fb["pos"], self.default_dof_pos[i])
            dof_vel[i] = jm.vel_rotor_to_sim(fb["vel"])
        if no_data and self._verbose:
            print(f"[warn] 暂无 FB 数据（用默认值）: {no_data}")
        return dof_pos, dof_vel

    # ── 观测构建 ──────────────────────────────────────────────────────────────
    def _build_obs(self, dof_pos: np.ndarray, dof_vel: np.ndarray) -> np.ndarray:
        """构建 45 维观测向量（与训练 env 合约完全一致）。"""
        # lin_vel (3): 真机无直接传感器，用 zero 作安全初值
        lin_vel = np.zeros(3, dtype=np.float32)

        if self._imu_source == "stub":
            # stub：假设机体近似水平，零角速度，重力方向向下
            ang_vel = np.zeros(3, dtype=np.float32)
            gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            raise NotImplementedError(f"imu_source='{self._imu_source}' 尚未实现")

        obs = np.concatenate([
            lin_vel,                           # [0:3]
            ang_vel,                           # [3:6]
            gravity_vec,                       # [6:9]
            dof_pos - self.default_dof_pos,    # [9:21]  dof_pos_scale=1.0
            dof_vel,                           # [21:33] dof_vel_scale=1.0
            self._prev_action,                 # [33:45]
        ])
        return np.clip(obs, -self.clip_obs, self.clip_obs).astype(np.float32)

    # ── policy 推理 ───────────────────────────────────────────────────────────
    def _infer(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)  # [1, 45]
            act_t = self._policy(obs_t)                  # [1, 12]
        action = act_t.squeeze(0).numpy().astype(np.float32)
        return np.clip(action, self.clip_lo, self.clip_hi)

    # ── 动作 → 旋子目标 ───────────────────────────────────────────────────────
    def _action_to_rotor(self, action: np.ndarray) -> np.ndarray:
        """policy 动作 → sim 关节目标角 → 旋子目标角（policy 顺序）。"""
        q_sim_target = self.action_scale * action + self.default_dof_pos
        rotor_targets = np.array([
            jm.sim_to_rotor(float(q_sim_target[i]), float(self.default_dof_pos[i]))
            for i, jm in enumerate(self.joints)
        ], dtype=np.float32)
        return rotor_targets

    # ── 安全检查 ──────────────────────────────────────────────────────────────
    def _is_safe(self, obs: np.ndarray) -> bool:
        """检查姿态是否在安全范围内（stub 模式跳过）。"""
        if self._imu_source == "stub":
            return True
        gravity_vec = obs[6:9]
        pitch = abs(math.asin(float(np.clip(gravity_vec[0], -1.0, 1.0))))
        roll = abs(math.asin(float(np.clip(gravity_vec[1], -1.0, 1.0))))
        if pitch > self._max_pr or roll > self._max_pr:
            print(f"[SAFETY] 姿态超限！pitch={math.degrees(pitch):.1f}° "
                  f"roll={math.degrees(roll):.1f}° 阈值={math.degrees(self._max_pr):.1f}°")
            return False
        return True

    # ── 信号处理 ──────────────────────────────────────────────────────────────
    def _on_signal(self, signum, _frame):
        print(f"\n[rl_deploy] 收到信号 {signum}，正在停止...")
        self._shutdown.set()

    # ── 主控循环 ──────────────────────────────────────────────────────────────
    def run(self):
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        print(f"[rl_deploy] 启动 servo 进程 (kp={self._kp}, kw={self._kw})")
        self._start_servos()

        # ── 站姿保持阶段 ──────────────────────────────────────────────────────
        print(f"[rl_deploy] 站姿保持 {self._hold_secs:.1f}s ...")
        t_end = time.monotonic() + self._hold_secs
        while time.monotonic() < t_end and not self._shutdown.is_set():
            self._send_stand_hold()
            time.sleep(CTRL_DT)

        if self._no_rl or self._shutdown.is_set():
            print("[rl_deploy] 退出（no_rl 或收到停止信号）")
            self._stop_servos()
            return

        # ── RL 控制阶段 ───────────────────────────────────────────────────────
        print(f"[rl_deploy] 开始 RL 控制（{CTRL_HZ} Hz）")
        print(f"[rl_deploy] 注意：imu_source={self._imu_source}，"
              f"lin_vel_source={self._lin_vel_source}")
        if self._imu_source == "stub":
            print(f"[rl_deploy] [警告] IMU=stub：ang_vel=0, gravity_vec=[0,0,-1]，"
                  f"policy 运行降级但安全；真机首次测试时先吊线/低增益验证")

        frame = 0
        t_next = time.monotonic()
        while not self._shutdown.is_set():
            t_start = time.monotonic()

            dof_pos, dof_vel = self._read_joint_state()
            obs = self._build_obs(dof_pos, dof_vel)

            if self._motor_fault is not None:
                print(f"[SAFETY] 电机故障急停！{self._motor_fault}")
                break
            if not self._is_safe(obs):
                print("[rl_deploy] 安全急停！")
                break

            action = self._infer(obs)
            self._prev_action = action.copy()

            self._filt_action = (self._action_lpf * action
                                 + (1.0 - self._action_lpf) * self._filt_action)
            rotor_targets = self._action_to_rotor(self._filt_action)
            self._send_rotor_targets(rotor_targets)

            frame += 1
            if frame % CTRL_HZ == 0 or (self._verbose and frame % 10 == 0):
                dpos_err = dof_pos - self.default_dof_pos
                print(f"[rl_deploy] frame={frame:5d} "
                      f"|Δq|_max={np.abs(dpos_err).max():.3f} "
                      f"|act|_max={np.abs(action).max():.3f} "
                      f"dt_ms={1000*(time.monotonic()-t_start):.1f}")

            # 精确定频
            t_next += CTRL_DT
            slack = t_next - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            elif self._verbose:
                print(f"[rl_deploy] 控制环超时 {-slack*1000:.1f}ms")

        print("[rl_deploy] 停止 servo 进程...")
        self._stop_servos()
        print("[rl_deploy] 完成。")


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--policy", default=str(DEFAULT_POLICY),
                    help=f"TorchScript policy .pt（默认: {DEFAULT_POLICY}）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help=f"YAML 配置文件（默认: {DEFAULT_CONFIG}）")
    ap.add_argument("--stand_config", default=str(DEFAULT_STAND_CONFIG),
                    help=f"stand_config.json（默认: {DEFAULT_STAND_CONFIG}）")
    ap.add_argument("--hold_secs", type=float, default=3.0,
                    help="RL 启动前站姿保持秒数（默认: 3.0）")
    ap.add_argument("--no_rl", action="store_true",
                    help="只保持站姿，不运行 policy（功能调试用）")
    ap.add_argument("--action_lpf", type=float, default=1.0,
                    help="动作低通滤波 α∈(0,1]，1=不滤波（默认）。0.3~0.5 压高频抖动；"
                         "与 play_mujoco.py --action-lpf 同语义，先在 MuJoCo 验证再上机。")
    ap.add_argument("--max_pitch_roll", type=float, default=0.5,
                    help="姿态超限急停阈值 rad（默认: 0.5 ≈ 28°，stub IMU 下不生效）")
    ap.add_argument("--verbose", action="store_true",
                    help="输出每帧调试信息")
    args = ap.parse_args()

    # 前置检查
    if not Path(args.policy).exists():
        print(f"[error] policy 文件不存在: {args.policy}")
        print(f"[hint]  先运行: python deploy/real/policy_export.py")
        sys.exit(1)
    if not Path(args.stand_config).exists():
        print(f"[error] stand_config 不存在: {args.stand_config}")
        print(f"[hint]  先用 robot_web.py 完成趴/站标定并保存")
        sys.exit(1)

    RLDeploy(args).run()


if __name__ == "__main__":
    main()
