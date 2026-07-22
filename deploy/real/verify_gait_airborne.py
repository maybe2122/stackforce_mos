#!/usr/bin/env python3
"""传统步态真机部署 ① —— 吊空验证脚本。

方案见 传统步态/文档/真机部署-01-角度映射标定与吊空验证.md。把「gait → 帧对齐 → rotor
映射 → 顺序 → 限速 → 预检 → 下发」整条链搭起来，全程吊空、软增益、限速、在线预检、
退出即松力。

子命令：
  check-frame              映射自洽 + 帧对齐（纯计算，不碰硬件）
  play-gait --dry-run      SimMotorBus 上实时彩排一个步态周期（整段预检+顺序双射+限速）
  read                     读 12 电机当前姿态 → 关节角 → FK（不驱动）
  calib-dir                逐关节 nudge 核 sim_sign（看足端是否朝 FK 预测方向动）
  pose-stand               缓慢摆到模型静态站姿并保持（吊空核对对称/别劲）

  硬件三命令(read/calib-dir/pose-stand) 均可加 --dry-run 在 SimMotorBus 上冒烟流程；
  真机路径已搭好但**尚未在真实电机上跑过**——上真机前务必吊空、软增益、行程内无人无物。

关键事实（详见方案文档 §0、§3、§4）：
  · 映射已存在于 deploy/real/motor_bus.py：rotor = stand_rotor + gear·dir·sim_sign·(q−default)。
  · 传统步态模型(closed_chain_kin, MJCF) 与 Isaac 训练环境(USD) 同资产 ⇒ θ_model == q_sim。
  · policy 顺序 = [hip×4, thigh×4, shank×4]（leg 序 fl,fr,rl,rr）；**shank 槽 = 模型 crank**。

用法示例：
  python deploy/real/verify_gait_airborne.py check-frame
  python deploy/real/verify_gait_airborne.py play-gait --dry-run
  python deploy/real/verify_gait_airborne.py read --dry-run          # 冒烟
  python deploy/real/verify_gait_airborne.py pose-stand --dry-run --hold 2
  python deploy/real/verify_gait_airborne.py calib-dir --dry-run
  # 真机（吊空后）：去掉 --dry-run，先 read，再 calib-dir，再 pose-stand
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config" / "mos2026_2.yaml"

# 纯 numpy 解算器（无 mujoco 依赖），来自传统步态 bundle
sys.path.insert(0, str(REPO_ROOT / "传统步态" / "代码"))
from gait import TrotGait  # noqa: E402
from closed_chain_kin import LEG_NAMES, LEGS, leg_ik, quad_fk, solve_linkage  # noqa: E402

from motor_bus import build_joint_maps, make_bus  # noqa: E402

JOINT_LIMIT = 1.57        # XML 里每个 hinge 的 range ±1.57 rad
MIN_MARGIN = 0.010        # 离奇异位形（分支折叠）至少留 10 mm
CONTROL_HZ = 100          # 真机伺服下发频率（本阶段干跑同频）
# 关节角速度安全上限（rad/s）。限速器的职责是拦「跳变」（初始逼近、glitch），
# 不是限步态自身的平滑运动——所以要设在步态峰值角速度之上、危险跳变之下。
# 吊空首测 1.0 rad/s 已很温柔；步态更快就得放慢 --period 或显式提 --vmax。
JOINT_VMAX = 1.0


# --------------------------------------------------------------------------- #
# 帧转换：gait 的 (4,3) [hip,thigh,crank]×[fl,fr,rl,rr]  <->  policy 顺序 12 维
# --------------------------------------------------------------------------- #
def _name_index(maps) -> dict[str, int]:
    return {jm.name: i for i, jm in enumerate(maps)}


def gait_to_policy(q_gait: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """q_gait: (4,3) 列=(θ_hip,θ_thigh,θ_crank)，行=fl,fr,rl,rr → policy 顺序 12 维。

    ⚠️ crank 放进 `*_shank` 槽（真机 shank 电机驱动的是四连杆曲柄）。
    """
    q12 = np.zeros(12)
    for li, leg in enumerate(LEG_NAMES):
        q12[idx[f"{leg}_hip"]] = q_gait[li, 0]
        q12[idx[f"{leg}_thigh"]] = q_gait[li, 1]
        q12[idx[f"{leg}_shank"]] = q_gait[li, 2]   # crank → shank 槽
    return q12


def policy_to_gait(q12: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """policy 顺序 12 维 → gait 的 (4,3)（逆变换，供 FK 自检）。"""
    q_gait = np.zeros((4, 3))
    for li, leg in enumerate(LEG_NAMES):
        q_gait[li, 0] = q12[idx[f"{leg}_hip"]]
        q_gait[li, 1] = q12[idx[f"{leg}_thigh"]]
        q_gait[li, 2] = q12[idx[f"{leg}_shank"]]
    return q_gait


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# 轨迹生成（纯运动学，无 mujoco）
# --------------------------------------------------------------------------- #
def gait_static_stand(x_offset: float = -0.055, body_height: float = 0.26) -> np.ndarray:
    """步态静态站姿的 (4,3) 关节角（相位冻结在支撑相起点）。"""
    g = TrotGait(x_offset=x_offset, body_height=body_height)
    q = np.zeros((4, 3))
    for li, leg in enumerate(LEG_NAMES):
        foot = g.foot_target(leg, 0.0)
        h, th, cr, info = leg_ik(foot, LEGS[leg], frame="leg")
        if not info["converged"]:
            raise RuntimeError(f"{leg} 静态站姿 IK 不收敛")
        q[li] = (h, th, cr)
    return q


def build_gait_trajectory(cycles: float, period: float, scale: float,
                          x_offset: float = -0.055):
    """返回 (times, q_traj[N,4,3])：放慢/缩幅的步态，逐帧 IK（延拓 seed）。

    scale 缩放步幅与步高（首次吊空先给小幅）。period 拉长即整体放慢。
    """
    g = TrotGait(period=period, step_length=0.10 * scale,
                 step_height=0.04 * scale, x_offset=x_offset)
    n = int(round(cycles * period * CONTROL_HZ))
    times = np.arange(n) / CONTROL_HZ
    q_traj = np.zeros((n, 4, 3))
    seed = {leg: (0.0, 0.0) for leg in LEG_NAMES}
    for k, t in enumerate(times):
        for li, leg in enumerate(LEG_NAMES):
            foot = g.foot_target(leg, t)
            h, th, cr, info = leg_ik(foot, LEGS[leg], frame="leg", seed=seed[leg])
            if not info["converged"]:
                raise RuntimeError(
                    f"{leg} 腿 t={t:.3f}s 足端 {np.round(foot,4)} IK 不收敛"
                    f"（残差 {info['residual']:.2e}）——步幅/周期超工作域")
            seed[leg] = (th, cr)
            q_traj[k, li] = (h, th, cr)
    return times, q_traj


# --------------------------------------------------------------------------- #
# 预检（整段，开跑前）
# --------------------------------------------------------------------------- #
def preflight_trajectory(q_traj: np.ndarray, maps, idx: dict[str, int],
                         vmax: float = JOINT_VMAX) -> list[str]:
    """检查整段轨迹：关节限位、奇异余量、逐帧关节步长、rotor 有限。返回问题清单。"""
    problems = []
    # 1) 关节限位
    if np.any(np.abs(q_traj) > JOINT_LIMIT):
        bad = np.argwhere(np.abs(q_traj) > JOINT_LIMIT)
        problems.append(f"超关节限位 ±{JOINT_LIMIT}: {len(bad)} 处，"
                        f"最大 |q|={np.abs(q_traj).max():.3f}")
    # 2) 奇异余量（曲柄侧四连杆折叠边界）
    mmin = np.inf
    for k in range(len(q_traj)):
        for li, leg in enumerate(LEG_NAMES):
            m = solve_linkage(q_traj[k, li, 1], q_traj[k, li, 2], LEGS[leg])["margin"]
            mmin = min(mmin, m)
    if mmin < MIN_MARGIN:
        problems.append(f"奇异余量过小 {mmin*1e3:.1f} mm (<{MIN_MARGIN*1e3:.0f})")
    # 3) 步态峰值角速度须在安全上限之下（否则限速器会削、轨迹失真）
    if len(q_traj) > 1:
        vpeak = np.abs(np.diff(q_traj, axis=0)).max() * CONTROL_HZ
        if vpeak > vmax:
            problems.append(
                f"步态峰值关节角速度 {vpeak:.2f} rad/s 超安全上限 {vmax:.2f}"
                f"——放慢 --period 或显式提 --vmax（吊空首测别贪快）")
    # 4) rotor 映射有限 + 顺序双射自洽（catch crank↔shank 槽位/顺序错）
    for k in (0, len(q_traj)//2, len(q_traj)-1):
        q12 = gait_to_policy(q_traj[k], idx)
        rotor = np.array([maps[i].q_to_rotor(q12[i]) for i in range(12)])
        if not np.all(np.isfinite(rotor)):
            problems.append(f"帧 {k} 存在非有限 rotor 目标")
        if np.abs(policy_to_gait(q12, idx) - q_traj[k]).max() > 1e-12:
            problems.append(f"帧 {k} 顺序映射非双射（gait↔policy 槽位错）")
    return problems


def rate_limit(cur: np.ndarray, tgt: np.ndarray, dt: float,
               vmax: float = JOINT_VMAX) -> np.ndarray:
    """关节空间限速：单步位移不超过 vmax·dt。"""
    step = np.clip(tgt - cur, -vmax * dt, vmax * dt)
    return cur + step


# --------------------------------------------------------------------------- #
# 硬件安全底座（read / calib-dir / pose-stand 共用）
# --------------------------------------------------------------------------- #
SAFE_KP = 2.0             # 吊空默认软增益（转子侧），远低于走路的 ~12.5
SAFE_KW = 0.05
MAX_ONLINE_ERR = 0.5      # 在线预检：命令↔反馈偏差超此值(rad) 立即松力停机


class _Abort(Exception):
    """在线预检/安全触发，需立即松力停机。"""


def _open_bus(cfg: dict, args):
    """按软增益打开总线（真机或 --dry-run 的 SimMotorBus）。"""
    kp = getattr(args, "kp", SAFE_KP)
    kw = getattr(args, "kw", SAFE_KW)
    cfg = dict(cfg)
    cfg["hardware"] = dict(cfg["hardware"], motor_kp=kp, motor_kw=kw)
    dry = getattr(args, "dry_run", False)
    try:
        bus = make_bus(cfg, REPO_ROOT, dry_run=dry)
    except FileNotFoundError as e:
        print(f"❌ 打开总线失败：{e}")
        raise SystemExit(2)
    tag = "SimMotorBus(dry-run)" if dry else "真机 MotorBus"
    print(f"[bus] {tag}  软增益 kp={kp} kw={kw}"
          + ("" if dry else "  ⚠️ 会驱动真实电机——确认已吊空、行程内无人无物！"))
    return bus


def _ramp_to(bus, cmd: np.ndarray, target: np.ndarray, dt: float, vmax: float,
             maps, label: str = "") -> np.ndarray:
    """限速把命令从 cmd 平滑推到 target，每步在线预检；返回到达后的命令。"""
    steps, limit = 0, int(20 * CONTROL_HZ)
    while np.abs(target - cmd).max() > 1e-4 and steps < limit:
        t0 = time.time()
        cmd = rate_limit(cmd, target, dt, vmax)
        bus.write_joint_targets(cmd)
        st = bus.read_joint_state()
        err = np.abs(cmd - st.q)
        if float(err.max()) > MAX_ONLINE_ERR:
            j = int(err.argmax())
            raise _Abort(f"{label} 在线预检失败：{maps[j].name} 命令↔反馈偏差 "
                         f"{err.max():.2f} rad 超 {MAX_ONLINE_ERR}（卡住/增益/机构异常）")
        steps += 1
        lag = dt - (time.time() - t0)
        if lag > 0:
            time.sleep(lag)
    return cmd


def _current_q(bus, maps) -> np.ndarray:
    """start() 之前读一次各电机转子角 → policy 帧关节角（端口独占，必须先读再起伺服）。"""
    rotor = bus.read_initial_rotor(attempts=2)
    return np.array([maps[i].rotor_to_q(rotor[i]) for i in range(len(maps))])


def _print_pose(maps, q12: np.ndarray, idx: dict[str, int]) -> None:
    """打印 policy 帧关节角 + FK 四脚 + 站高/对称性自检。"""
    feet = quad_fk(policy_to_gait(q12, idx), frame="base")
    print("    关节角（policy 顺序, rad）：")
    for i, jm in enumerate(maps):
        print(f"      {jm.name:10s} {q12[i]:+.4f}")
    print("    FK 四脚（base 系, m）：")
    for li, leg in enumerate(LEG_NAMES):
        print(f"      {leg}: {feet[li]}")
    zspread = feet[:, 2].max() - feet[:, 2].min()
    y_ok = feet[0, 1] > 0 and feet[2, 1] > 0 and feet[1, 1] < 0 and feet[3, 1] < 0
    print(f"    站高 {(-feet[:,2].mean()):.4f} m   四脚站高差 {zspread*1e3:.1f} mm"
          f"   y 符号(左+右−) {'✅' if y_ok else '❌'}")


# --------------------------------------------------------------------------- #
# read：只读当前姿态（不驱动）
# --------------------------------------------------------------------------- #
def cmd_read(args) -> int:
    cfg = load_cfg(Path(args.config))
    maps = build_joint_maps(cfg)
    idx = _name_index(maps)
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 68)
    print("read —— 读 12 电机当前转子角 → 关节角 → FK（不驱动）")
    print("=" * 68)
    bus = _open_bus(cfg, args)
    try:
        q12 = _current_q(bus, maps)
    finally:
        bus.close()   # read_initial_rotor 不起伺服，但保持对称
    _print_pose(maps, q12, idx)
    return 0


# --------------------------------------------------------------------------- #
# pose-stand：缓慢摆到模型静态站姿并保持（软增益、限速、在线预检）
# --------------------------------------------------------------------------- #
def cmd_pose_stand(args) -> int:
    cfg = load_cfg(Path(args.config))
    maps = build_joint_maps(cfg)
    idx = _name_index(maps)
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 68)
    print("pose-stand —— 缓慢摆到模型静态站姿并保持（吊空核对用）")
    print("=" * 68)

    target = gait_to_policy(gait_static_stand(), idx)
    if np.any(np.abs(target) > JOINT_LIMIT):
        print("❌ 目标站姿超关节限位，拒绝。")
        return 2

    bus = _open_bus(cfg, args)
    dt = 1.0 / CONTROL_HZ
    try:
        q0 = _current_q(bus, maps)
        print(f"[1] 当前姿态读取完成；将从当前位限速({args.vmax} rad/s)摆到模型站姿。")
        bus.start(initial_q_sim=q0)          # 先按当前位保持，不snap
        cmd = _ramp_to(bus, q0.copy(), target, dt, args.vmax, maps, "pose-stand")
        print(f"[2] 已到站姿，保持 {args.hold:.0f}s（人眼核对对称/别劲，Ctrl-C 提前松力）")
        t_end = time.time() + args.hold
        while time.time() < t_end:
            t0 = time.time()
            bus.write_joint_targets(cmd)
            bus.read_joint_state()
            lag = dt - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
        st = bus.read_joint_state()
        print("[3] 读回实测姿态：")
        _print_pose(maps, st.q, idx)
    except (_Abort, KeyboardInterrupt) as e:
        print(f"\n⚠️ 中止（{type(e).__name__}: {e}）——松力停机。")
        return 4
    finally:
        bus.close()
        print("[bus] 已发 stop 松力。")
    return 0


# --------------------------------------------------------------------------- #
# calib-dir：逐关节 nudge，核 sim_sign（观察足端是否朝 FK 预测方向动）
# --------------------------------------------------------------------------- #
def cmd_calib_dir(args) -> int:
    cfg = load_cfg(Path(args.config))
    maps = build_joint_maps(cfg)
    idx = _name_index(maps)
    dry = getattr(args, "dry_run", False)

    print("=" * 68)
    print("calib-dir —— 逐关节 nudge 核 sim_sign（观察足端朝 FK 预测方向动否）")
    print("=" * 68)
    print(f"每关节：命令 q_sim +{args.nudge} rad，看对应足端是否朝预测方向移动。")
    print("一致=当前 sim_sign 对；相反=应翻转。一次只动一个关节，动完回中。\n")

    bus = _open_bus(cfg, args)
    dt = 1.0 / CONTROL_HZ
    proposed: dict[str, int] = {}
    try:
        q0 = _current_q(bus, maps)
        bus.start(initial_q_sim=q0)
        base = q0.copy()
        for i, jm in enumerate(maps):
            # 预测：仅该关节 +nudge 时，该腿足端在 base 系的位移
            leg = jm.name.split("_")[0]
            q_g = policy_to_gait(base, idx)
            f0 = quad_fk(q_g, frame="base")[LEG_NAMES.index(leg)]
            q_g2 = q_g.copy()
            slot = {"hip": 0, "thigh": 1, "shank": 2}[jm.name.split("_")[1]]
            q_g2[LEG_NAMES.index(leg), slot] += args.nudge
            f1 = quad_fk(q_g2, frame="base")[LEG_NAMES.index(leg)]
            d = (f1 - f0) * 1e3
            pred = (f"前{d[0]:+.0f} 左{d[1]:+.0f} 上{d[2]:+.0f} mm")

            ans = "s"
            if not dry:
                ans = input(f"[{i+1:2d}/12] {jm.name}: 回车=nudge / s=跳过 / q=退出 > ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    continue
            print(f"    预测：{leg} 足端应 {pred}")
            tgt = base.copy(); tgt[i] += args.nudge
            _ramp_to(bus, base.copy(), tgt, dt, args.vmax, maps, jm.name)
            time.sleep(0.3)
            if dry:
                same = True   # 干跑无从观察，默认「一致」走流程
            else:
                r = input(f"    {jm.name} 足端实际是否朝上面预测方向动？ y=一致 / n=相反 / r=重看 > ").strip().lower()
                while r == "r":
                    _ramp_to(bus, tgt.copy(), base.copy(), dt, args.vmax, maps, jm.name)
                    _ramp_to(bus, base.copy(), tgt, dt, args.vmax, maps, jm.name)
                    r = input("    y/n/r > ").strip().lower()
                same = (r == "y")
            proposed[jm.name] = jm.sim_sign if same else -jm.sim_sign
            if not same:
                print(f"    ⚠️ {jm.name} 方向相反 ⇒ sim_sign 应 {jm.sim_sign} → {-jm.sim_sign}")
            _ramp_to(bus, tgt.copy(), base.copy(), dt, args.vmax, maps, jm.name)   # 回中
    except (_Abort, KeyboardInterrupt) as e:
        print(f"\n⚠️ 中止（{type(e).__name__}: {e}）——松力停机。")
        return 4
    finally:
        bus.close()
        print("[bus] 已发 stop 松力。")

    print("\n== sim_sign 核验结果 ==")
    changed = {n: s for n, s in proposed.items()
               if s != next(j.sim_sign for j in maps if j.name == n)}
    for jm in maps:
        if jm.name in proposed:
            flag = "  ← 翻转" if jm.name in changed else ""
            print(f"  {jm.name:10s} sim_sign {proposed[jm.name]:+d}{flag}")
    if changed:
        print(f"\n{len(changed)} 个关节 sim_sign 需翻转。**未自动改 YAML**（安全起见）；"
              "确认后手动改 deploy/real/config/mos2026_2.yaml 的 hardware.joints[*].sim_sign。")
    elif not dry:
        print("\n全部方向正确，无需改动。")
    if dry:
        print("\n（--dry-run：以上为流程冒烟，未真实观察方向。）")
    return 0


# --------------------------------------------------------------------------- #
# check-frame：映射自洽 + 帧对齐（纯计算）
# --------------------------------------------------------------------------- #
def cmd_check_frame(args) -> int:
    cfg = load_cfg(Path(args.config))
    maps = build_joint_maps(cfg)
    idx = _name_index(maps)
    ok = True
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 68)
    print("check-frame —— 映射自洽 + 帧对齐（纯计算，不碰硬件）")
    print("=" * 68)

    print("\n[0] 映射参数表  rotor = stand_rotor + k·(q − default),  k = gear·dir·sim_sign")
    print(f"    {'joint':10s}{'dir':>4}{'sim_sign':>9}{'k':>9}{'stand_rotor':>13}{'default':>9}")
    for jm in maps:
        print(f"    {jm.name:10s}{jm.dir:>4}{jm.sim_sign:>9}{jm.k:>9.3f}"
              f"{jm.stand_rotor:>13.4f}{jm.default:>9.3f}")

    # 1) 正逆往返
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.5, 1.5, (2000, 12))
    err = 0.0
    for i, jm in enumerate(maps):
        r = jm.q_to_rotor(q[:, i])
        back = jm.rotor_to_q(r)
        err = max(err, float(np.abs(back - q[:, i]).max()))
    print(f"\n[1] 正逆往返 q→rotor→q  max_err = {err:.2e}", "✅" if err < 1e-9 else "❌")
    ok &= err < 1e-9

    # 2) 锚点自洽：q=default ⇒ rotor=stand_rotor
    aerr = max(abs(jm.q_to_rotor(jm.default) - jm.stand_rotor) for jm in maps)
    print(f"[2] 锚点自洽 q_to_rotor(default)==stand_rotor  max_err = {aerr:.2e}",
          "✅" if aerr < 1e-9 else "❌")
    ok &= aerr < 1e-9

    # 3) k == gear·dir·sim_sign
    kerr = max(abs(jm.k - jm.gear * jm.dir * jm.sim_sign) for jm in maps)
    print(f"[3] k == gear·dir·sim_sign  max_err = {kerr:.2e}", "✅" if kerr < 1e-12 else "❌")
    ok &= kerr < 1e-12

    # 4) 帧对齐：步态静态站姿 → policy → rotor → 逆 → FK
    print("\n[4] 帧对齐（步态静态站姿穿过映射 + FK 自检）")
    q_stand = gait_static_stand()
    q12 = gait_to_policy(q_stand, idx)
    rotor = np.array([maps[i].q_to_rotor(q12[i]) for i in range(12)])
    q12_back = np.array([maps[i].rotor_to_q(rotor[i]) for i in range(12)])
    rt = float(np.abs(q12_back - q12).max())
    feet = quad_fk(policy_to_gait(q12_back, idx), frame="base")
    print(f"    穿映射往返 max_err = {rt:.2e}", "✅" if rt < 1e-9 else "❌")
    print("    站姿关节角（policy 顺序）与对应转子目标：")
    for i, jm in enumerate(maps):
        print(f"      {jm.name:10s} q={q12[i]:+.4f} rad  → rotor={rotor[i]:+.4f} rad")
    print("    FK 四脚（base 系, m）：")
    for li, leg in enumerate(LEG_NAMES):
        print(f"      {leg}: {feet[li]}")
    # 「站姿是否合理」的真正判据（映射算术已由 [1][2] 证明精确）：
    #   a) 四脚共面（同一站高）——某腿符号错会让它 z 明显偏离；
    #   b) y 符号对（左腿 +、右腿 −）——不比 |y| 量级：足端网格左右本就不对称
    #      （y_foot 左 +0.0796 / 右 −0.0636，见 gait 自测），16 mm 差是资产属性非 bug；
    #   c) 关节角保留模型的**对角反对称**（fl↔rr、fr↔rl 反号）。
    zspread = feet[:, 2].max() - feet[:, 2].min()
    y_sign_ok = feet[0, 1] > 0 and feet[2, 1] > 0 and feet[1, 1] < 0 and feet[3, 1] < 0
    diag = (abs(q_stand[0, 1] + q_stand[3, 1]) + abs(q_stand[1, 1] + q_stand[2, 1])
            + abs(q_stand[0, 2] + q_stand[3, 2]) + abs(q_stand[1, 2] + q_stand[2, 2]))
    print(f"    四脚站高差 = {zspread*1e3:.1f} mm   y 符号(左+右−) = {'✅' if y_sign_ok else '❌'}"
          f"   对角反对称残差 = {diag:.2e}")
    ok &= rt < 1e-9 and zspread < 0.015 and y_sign_ok and diag < 1e-6

    print("\n" + ("✅ check-frame 全部通过" if ok else "❌ check-frame 有失败"))
    print("注：θ_model==q_sim 的**物理**确认仍需吊空站姿核对（方案 §5.3），本步只证自洽。")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# play-gait --dry-run：整条管线跑一遍（SimMotorBus）
# --------------------------------------------------------------------------- #
def cmd_play_gait(args) -> int:
    if not args.dry_run:
        print("❌ 本阶段 play-gait 仅支持 --dry-run（真机路径待硬件就绪后在 ② 补）。")
        return 2

    cfg = load_cfg(Path(args.config))
    maps = build_joint_maps(cfg)
    idx = _name_index(maps)
    default = np.array([jm.default for jm in maps])

    print("=" * 68)
    print(f"play-gait --dry-run  cycles={args.cycles} period={args.period}s "
          f"scale={args.scale}  ({CONTROL_HZ} Hz, SimMotorBus)")
    print("=" * 68)

    # 1) 生成轨迹
    try:
        times, q_traj = build_gait_trajectory(args.cycles, args.period, args.scale)
    except RuntimeError as e:
        print(f"[1] ❌ 轨迹不可行：{e}")
        print("    办法：减小 --scale（步幅）或用默认参数；步幅×0.5≈0.05m 半步。")
        return 2
    print(f"[1] 轨迹 {len(q_traj)} 帧 / {times[-1]:.2f}s")

    # 2) 整段预检
    problems = preflight_trajectory(q_traj, maps, idx, vmax=args.vmax)
    if problems:
        print("[2] ❌ 预检不通过：")
        for p in problems:
            print("      -", p)
        return 2
    print("[2] ✅ 整段预检通过（限位/奇异余量/步长/rotor 有限）")

    # 3) 起 SimMotorBus，从 default 平滑逼近轨迹起点，再流式播放
    bus = make_bus(cfg, REPO_ROOT, dry_run=True)
    bus.start(initial_q_sim=default)
    dt = 1.0 / CONTROL_HZ
    cmd_prev = default.copy()

    # 逼近相：限速从 default 到 traj[0]（实时节拍，SimMotorBus 按墙钟积分）
    tgt0 = gait_to_policy(q_traj[0], idx)
    approach = 0
    while np.abs(tgt0 - cmd_prev).max() > 1e-4 and approach < 5 * CONTROL_HZ:
        t0 = time.time()
        cmd_prev = rate_limit(cmd_prev, tgt0, dt, args.vmax)
        bus.write_joint_targets(cmd_prev)
        bus.read_joint_state()
        approach += 1
        lag = dt - (time.time() - t0)
        if lag > 0:
            time.sleep(lag)
    print(f"[3] 逼近轨迹起点用 {approach/CONTROL_HZ:.2f}s（限速 {args.vmax} rad/s）")

    # 播放相：逐帧限速下发，记录命令 vs 反馈
    # 实时节拍：SimMotorBus 按墙钟积分，必须实时喂它才有意义（也是真机的忠实彩排）
    max_track = 0.0
    max_temp = 0
    clipped = 0
    for k in range(len(q_traj)):
        t0 = time.time()
        tgt = gait_to_policy(q_traj[k], idx)
        limited = rate_limit(cmd_prev, tgt, dt, args.vmax)
        if np.abs(limited - tgt).max() > 1e-6:
            clipped += 1
        cmd_prev = limited
        bus.write_joint_targets(cmd_prev)
        st = bus.read_joint_state()
        max_track = max(max_track, float(np.abs(st.q - cmd_prev).max()))
        max_temp = max(max_temp, int(st.temp.max()))
        lag = dt - (time.time() - t0)
        if lag > 0:
            time.sleep(lag)
    bus.close()

    print(f"[4] 播放 {len(q_traj)} 帧完成（实时 {len(q_traj)/CONTROL_HZ:.1f}s）")
    print(f"    被限速削帧数 = {clipped}/{len(q_traj)}（步态平滑且 vmax 够 ⇒ 应为 0）")
    print(f"    命令↔反馈 最大跟踪误差 = {max_track:.4f} rad"
          f"（SimMotorBus 一阶跟随，仅粗看管线；真机跟踪看 ② 的实机 FB）")
    print(f"    最高温度 = {max_temp}℃（仿真恒 30）")
    print(f"    结束已发 stop 松力：{'是' if not bus._running else '否'}")

    # 通过判据只看**管线**：预检过（含顺序双射、限位、奇异余量）、限速器不与步态
    # 打架（clipped=0）、跑完、已松力。跟踪误差仅信息（SimMotorBus 是一阶替身，非物理）。
    ok = clipped == 0 and not bus._running
    print("\n" + ("✅ dry-run 管线通过：gait→帧对齐→映射→顺序→限速→预检→下发→松力 全链无异常"
                  if ok else "⚠️ dry-run 有告警（见上）——排查后再上真机"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="传统步态真机部署 ① 吊空验证（check-frame + dry-run）")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="部署 YAML")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_hw_args(sp):
        sp.add_argument("--dry-run", action="store_true",
                        help="用 SimMotorBus 冒烟流程（不连真机）")
        sp.add_argument("--kp", type=float, default=SAFE_KP, help="转子侧软增益 kp")
        sp.add_argument("--kw", type=float, default=SAFE_KW, help="转子侧软增益 kw")
        sp.add_argument("--vmax", type=float, default=JOINT_VMAX, help="关节角速度上限 rad/s")

    sub.add_parser("check-frame", help="映射自洽 + 帧对齐（纯计算，不碰硬件）")

    pg = sub.add_parser("play-gait", help="干跑一个步态周期走完整管线")
    pg.add_argument("--dry-run", action="store_true", help="用 SimMotorBus（本阶段必需）")
    pg.add_argument("--cycles", type=float, default=1.0, help="播放周期数")
    pg.add_argument("--period", type=float, default=2.0, help="步态周期 s（拉长=放慢）")
    pg.add_argument("--scale", type=float, default=0.5, help="步幅/步高缩放（首次吊空给小）")
    pg.add_argument("--vmax", type=float, default=JOINT_VMAX, help="关节角速度安全上限 rad/s")

    rd = sub.add_parser("read", help="读当前 12 电机姿态 + FK（不驱动）")
    add_hw_args(rd)

    cd = sub.add_parser("calib-dir", help="逐关节 nudge 核 sim_sign 方向")
    add_hw_args(cd)
    cd.add_argument("--nudge", type=float, default=0.15, help="每关节试探角 rad（q_sim 空间）")

    ps = sub.add_parser("pose-stand", help="缓慢摆到模型静态站姿并保持")
    add_hw_args(ps)
    ps.add_argument("--hold", type=float, default=8.0, help="到位后保持时长 s")

    args = p.parse_args()
    dispatch = {
        "check-frame": cmd_check_frame, "play-gait": cmd_play_gait,
        "read": cmd_read, "calib-dir": cmd_calib_dir, "pose-stand": cmd_pose_stand,
    }
    fn = dispatch.get(args.cmd)
    if fn is None:
        p.print_help()
        return 0
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
