"""站起来（蹲→站）过程的关节/电机力矩分析与可视化。

做什么
------
在 MuJoCo（``mos2026_2.xml``）里让机器人从蹲姿插值站起来，逐控制步记录：
  - 关节驱动力矩  τ_joint  = ``data.actuator_force``（position 伺服实际输出，
    已含重力 + 惯性 + 地反力，正是「电机要加在关节上的力」）；
  - 电机转子侧力矩 τ_rotor = τ_joint /(N·η)（用 dynamics.py 的减速比/效率反射）；
  - 电机负载率   = |τ_joint| / 输出峰值，超 effort_limit / 输出峰值即报警；
  - 估算相电流   I = τ_rotor / Kt（Kt 待标定，--kt 传入，0 则不算）。

为什么用这条路（对应 todo.md「力矩/电流不足专项」）
--------------------------------------------------
12 个关节都是直驱 hinge + <position> 伺服，所以 ``actuator_force`` 就是关节净
驱动力矩，不需要 mj_inverse（本机膝部是 <connect> 闭链，逆动力学不好用）。
站起来是「带加速度 + 足端地反力切换」的动态过程，只有物理仿真能算准，纯静力
（dynamics.py）只覆盖单姿态保持。

蹲姿几何
--------
站姿 = env 默认关节角（thigh/shank=0，腿竖直）。蹲姿 = 站姿 + 折叠量。thigh/shank
轴向左右相反（FL/RL 的 thigh 轴 −y，FR/RR 为 +y），所以同一「物理俯仰」要按轴
符号镜像，保证四条腿朝同一方向折叠。折叠量 --fold-thigh/--fold-shank 可调，建议先
``--viewer`` 看一眼站起来动作对不对，再跑 headless 记录。

输出
----
  outputs/standup_torque/torque_curves.png   每关节力矩曲线 + 极限线
  outputs/standup_torque/torque_series.csv   逐步全量序列
  outputs/standup_torque/torque_stats.csv    每关节 mean/std/min/max/abs_max/rms
  （--swanlab 时）实时 log 到 swanlab，面板可看每关节 τ/电机负载/电流

运行（env_isaaclab，装了 mujoco/matplotlib/swanlab）：
  python deploy/mujoco/standup_torque.py                 # headless 记录 + 出图
  python deploy/mujoco/standup_torque.py --viewer        # 边看边记
  python deploy/mujoco/standup_torque.py --swanlab       # 同时实时上报 swanlab
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = REPO_ROOT / "deploy/mujoco/assets/mos2026_2.xml"
OUT_DIR = REPO_ROOT / "outputs/standup_torque"

# --- 12 个驱动关节（顺序：hips, thighs, shank_links；腿序 fl,fr,rl,rr）-----------
ACTUATED_JOINTS = [
    "fl_hip", "fr_hip", "rl_hip", "rr_hip",
    "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
    "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a",
]
# 髋外摆角（= env init_state / play_mujoco.DEFAULT_JOINT_POS 的 hips），站/蹲一致
HIP_POS = np.array([0.15, 0.15, -0.15, -0.15])
# thigh / shank_link 轴的 y 分量符号（+y→+1，−y→−1），来自 XML axis。
# 物理俯仰 P 映射到关节值 q = P * sign，保证四腿同向折叠。
THIGH_AXIS_SIGN = np.array([-1.0, +1.0, -1.0, +1.0])   # fl,fr,rl,rr: -y,+y,-y,+y
SHANK_AXIS_SIGN = np.array([-1.0, +1.0, -1.0, +1.0])

# 站/蹲姿用「物理俯仰」(rad) 描述，再按轴符号镜像成 12 关节目标。
# 蹲姿受大腿-小腿机械限位约束：MJCF 已加 thigh↔shank 凸分解碰撞对
# （tools/asset/add_knee_self_collision.py），实测限位在 thigh≈0.34 /
# shank≈-0.39，蹲更深大腿就会顶在小腿上（真机同样顶死）。默认蹲姿取限位内
# 留余量的 0.30/-0.35；注意 reset 是直接置入蹲姿 qpos，若给超限值会从
# 已穿透状态开始、接触把杆件越推越深（穿模）。
STAND_THIGH_PHYS = 0.0
STAND_SHANK_PHYS = 0.4
CROUCH_THIGH_PHYS = 0.30
CROUCH_SHANK_PHYS = -0.35

# --- 电机/传动常数（取自 传统步态/代码/dynamics.py，GO-M8010-6 @24V）-----------
GEAR = 6.33                 # 减速比
ETA = 0.90                  # 减速器效率
TAU_OUT_PEAK = 23.7         # 输出端峰值力矩 N·m（电机物理上限）
EFFORT_LIMIT = 12.0         # 现配 effort_limit N·m（控制软限幅）

# --- 控制/仿真频率（与 play_mujoco 一致）---------------------------------------
CONTROL_HZ = 50
SIM_HZ = 1000


_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_PANEL_W = 400          # 瞬时力矩条面板宽
_CHART_W = 464          # 所有关节力矩对比曲线面板宽
# swanlab 合并图 all_joint_torque 的 12 关节配色（无法分线型，用 12 个可区分的颜色）
_SW_PALETTE = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
               "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9A6324", "#808000"]
_LEG_COLOR = {"fl": (31, 119, 180), "fr": (255, 127, 14),
              "rl": (44, 160, 44), "rr": (214, 39, 40)}
_TYPE_THICK = (1, 2, 3)   # hip / thigh / shank 线粗，编码关节类型


def _bars_panel(tau, t, h):
    """瞬时力矩条：每关节一条，长度∝|τ|，绿/橙/红按负载率，叠 effort_limit/电机峰值竖线。"""
    from PIL import Image, ImageDraw, ImageFont
    panel = Image.new("RGB", (_PANEL_W, h), (245, 245, 245))
    d = ImageDraw.Draw(panel)
    f = ImageFont.truetype(_FONT_PATH, 13); fs = ImageFont.truetype(_FONT_PATH, 11)
    d.text((8, 6), "joint torque (N*m)", font=f, fill=(0, 0, 0))
    if t is not None:
        d.text((_PANEL_W - 92, 6), f"t={t:5.2f}s", font=fs, fill=(60, 60, 60))
    x0 = 118
    scale = (_PANEL_W - x0 - 48) / TAU_OUT_PEAK
    top, bot = 28, h - 8
    for lim, col in [(EFFORT_LIMIT, (255, 150, 0)), (TAU_OUT_PEAK, (210, 0, 0))]:
        x = x0 + lim * scale
        d.line([(x, top), (x, bot)], fill=col, width=1)
        d.text((x - 8, top - 14), f"{lim:g}", font=fs, fill=col)
    rowh = (bot - top) / 12.0
    for i, name in enumerate(ACTUATED_JOINTS):
        y = top + i * rowh
        val = float(tau[i]); a = abs(val); load = a / TAU_OUT_PEAK
        col = (40, 170, 60) if load < 0.5 else (230, 150, 0) if load < 0.85 else (210, 40, 40)
        d.text((6, y + 1), name[:13], font=fs, fill=(0, 0, 0))
        d.rectangle([x0, y + 2, x0 + a * scale, y + rowh - 3], fill=col)
        d.text((x0 + a * scale + 4, y + 1), f"{val:+5.1f}", font=fs, fill=(30, 30, 30))
    return np.asarray(panel)


def _chart_panel(hist_t, hist_tau, t_total, h):
    """所有 12 个关节力矩随时间增长的对比曲线：色=腿，线粗=关节类型，竖线=当前时刻。"""
    from PIL import Image, ImageDraw, ImageFont
    panel = Image.new("RGB", (_CHART_W, h), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    f = ImageFont.truetype(_FONT_PATH, 12); fs = ImageFont.truetype(_FONT_PATH, 10)
    d.text((8, 4), "all joint torques (N*m)", font=f, fill=(0, 0, 0))
    L, R, T, B = 44, _CHART_W - 10, 24, h - 44
    HT = np.asarray(hist_tau) if len(hist_tau) else np.zeros((0, 12))
    ymax = max(EFFORT_LIMIT, float(np.abs(HT).max()) if HT.size else EFFORT_LIMIT) * 1.1

    def X(tt): return L + (R - L) * (tt / max(t_total, 1e-6))
    def Y(v): return (T + B) / 2 - (v / ymax) * ((B - T) / 2)

    d.rectangle([L, T, R, B], outline=(180, 180, 180))
    d.line([(L, Y(0)), (R, Y(0))], fill=(150, 150, 150))
    for s in (1, -1):                              # effort_limit 参考线
        d.line([(L, Y(s * EFFORT_LIMIT)), (R, Y(s * EFFORT_LIMIT))], fill=(255, 170, 0))
    for v in (ymax, 0, -ymax):
        d.text((6, Y(v) - 6), f"{v:+.0f}", font=fs, fill=(120, 120, 120))

    if len(hist_t) >= 2:
        ht = np.asarray(hist_t)
        for gi in range(3):                        # hip/thigh/shank
            for li, leg in enumerate(("fl", "fr", "rl", "rr")):
                j = gi * 4 + li
                pts = [(X(ht[k]), Y(HT[k, j])) for k in range(len(ht))]
                d.line(pts, fill=_LEG_COLOR[leg], width=_TYPE_THICK[gi])
        xn = X(ht[-1])
        d.line([(xn, T), (xn, B)], fill=(0, 0, 0))  # 当前时刻

    lx = L                                          # 图例：腿色
    for leg in ("fl", "fr", "rl", "rr"):
        d.rectangle([lx, B + 8, lx + 11, B + 18], fill=_LEG_COLOR[leg])
        d.text((lx + 14, B + 7), leg, font=fs, fill=(0, 0, 0)); lx += 52
    lx = L                                           # 图例：类型线粗
    for typ, w in (("hip", 1), ("thigh", 2), ("shank", 3)):
        d.line([(lx, B + 30), (lx + 22, B + 30)], fill=(80, 80, 80), width=w)
        d.text((lx + 26, B + 24), typ, font=fs, fill=(0, 0, 0)); lx += 66
    return np.asarray(panel)


def overlay_torque(img, tau, t=None, hist_t=None, hist_tau=None, t_total=None):
    """渲染帧右侧拼：瞬时力矩条 +（给了历史时）所有关节力矩对比曲线。"""
    h = img.shape[0]
    parts = [img, _bars_panel(tau, t, h)]
    if hist_t is not None:
        parts.append(_chart_panel(hist_t, hist_tau, t_total, h))
    return np.concatenate(parts, axis=1)


def build_alltau_echarts(times, tau):
    """构建 swanlab ECharts 折线图：12 关节力矩同图。色=12 色板，线型=关节类型
    (hip 实/thigh 虚/shank 点)。供 swanlab.log 实时上报（cloud 下随步增长）。"""
    import swanlab
    import pyecharts.options as opts

    line = swanlab.echarts.Line(init_opts=opts.InitOpts(width="820px", height="440px"))
    line.add_xaxis([f"{t:.2f}" for t in times])
    ls_by_type = {"hip": "solid", "thigh": "dashed", "shank": "dotted"}
    for i, name in enumerate(ACTUATED_JOINTS):
        typ = name.split("_")[1]            # fl_hip→hip / fr_shank_link_a→shank
        line.add_yaxis(
            name, [round(float(v), 3) for v in tau[:, i]],
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, type_=ls_by_type.get(typ, "solid")),
            itemstyle_opts=opts.ItemStyleOpts(color=_SW_PALETTE[i % len(_SW_PALETTE)]),
        )
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="all joint torques (N*m)"),
        xaxis_opts=opts.AxisOpts(name="t (s)", type_="category"),
        yaxis_opts=opts.AxisOpts(name="N*m"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="6%", type_="scroll"),
    )
    return line


def make_pose(thigh_phys: float, shank_phys: float) -> np.ndarray:
    """由「物理俯仰」(thigh, shank) 生成 12 关节目标，按轴符号镜像四腿同向。"""
    q = np.empty(12)
    q[0:4] = HIP_POS
    q[4:8] = THIGH_AXIS_SIGN * thigh_phys
    q[8:12] = SHANK_AXIS_SIGN * shank_phys
    return q


class StandupRig:
    def __init__(self, mjcf_path: Path, init_height: float):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.model.opt.timestep = 1.0 / SIM_HZ
        self.data = mujoco.MjData(self.model)
        self.init_height = init_height

        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        self.qpos_idx, self.ctrl_idx = [], []
        for jname in ACTUATED_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
            if jid < 0 or aid < 0:
                raise RuntimeError(f"joint/actuator for '{jname}' not found in MJCF")
            self.qpos_idx.append(self.model.jnt_qposadr[jid])
            self.ctrl_idx.append(aid)
        self.qpos_idx = np.array(self.qpos_idx, dtype=int)
        self.ctrl_idx = np.array(self.ctrl_idx, dtype=int)

    def reset(self, pose: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, self.init_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_idx] = pose
        self.data.ctrl[self.ctrl_idx] = pose
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_targets(self, pose: np.ndarray) -> None:
        self.data.ctrl[self.ctrl_idx] = pose

    def step(self, decimation: int) -> None:
        for _ in range(decimation):
            mujoco.mj_step(self.model, self.data)

    def joint_torque(self) -> np.ndarray:
        """各关节驱动力矩（N·m），顺序同 ACTUATED_JOINTS。"""
        return self.data.actuator_force[self.ctrl_idx].copy()

    def base_height(self) -> float:
        return float(self.data.qpos[2])


def smoothstep(t: float) -> float:
    """0→1 的 C1 平滑插值（避免阶跃造成力矩尖峰）。"""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def run(args) -> int:
    mjcf = Path(args.mjcf)
    if not mjcf.exists():
        print(f"[error] MJCF not found: {mjcf}", file=sys.stderr)
        return 1

    rig = StandupRig(mjcf, args.init_height)
    if args.kp > 0:  # 覆盖 MJCF 的 kp=25（伺服偏软会站不稳；调硬看真实跟踪力矩）
        rig.model.actuator_gainprm[:, 0] = args.kp
        rig.model.actuator_biasprm[:, 1] = -args.kp
    stand = make_pose(args.stand_thigh, args.stand_shank)
    crouch = make_pose(args.crouch_thigh, args.crouch_shank)
    rig.reset(crouch)

    decimation = SIM_HZ // CONTROL_HZ
    control_dt = 1.0 / CONTROL_HZ
    n_settle = int(args.settle * CONTROL_HZ)
    n_rise = int(args.rise * CONTROL_HZ)
    n_hold = int(args.hold * CONTROL_HZ)

    # swanlab（可选，实时上报）。mode=cloud 才有「图自己流式长出、不用刷新」；
    # local 看板不主动轮询，需手动刷新页面才拉新点。
    sw = None
    if args.swanlab:
        try:
            import swanlab
            init_kw = dict(
                project="standup_torque",
                experiment_name=time.strftime("%Y%m%d_%H%M%S"),
                mode=args.swanlab_mode,
                config={"stand": [args.stand_thigh, args.stand_shank],
                        "crouch": [args.crouch_thigh, args.crouch_shank],
                        "rise_s": args.rise, "kp": args.kp or 25.0, "gear": GEAR},
            )
            if args.swanlab_mode != "cloud":   # 本地/离线才指定落盘目录
                init_kw["logdir"] = str(REPO_ROOT / "swanlog_local")
            sw = swanlab.init(**init_kw)
            if args.swanlab_mode != "cloud":
                print("[swanlab] local 看板不自动刷新，需手动刷新页面；要真·实时不刷新用 "
                      "--swanlab-mode cloud", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] swanlab 初始化失败，跳过实时上报: {e}", file=sys.stderr)
            sw = None

    times, heights = [], []
    tau_log = []   # 每行 12 个关节力矩

    viewer_cm = None
    if args.viewer:
        import mujoco.viewer as mj_viewer
        viewer_cm = mj_viewer.launch_passive(rig.model, rig.data)
        viewer = viewer_cm.__enter__()

    # 离屏渲染成视频（GUI 窗口在无 GPU 上下文的环境里开不起来时用这个看动作）。
    # 慢放：在 sim 步级抽帧，输出 vid_fps；slowmo=播放慢于真实的倍数。
    renderer = cam = frames = None
    cap_every = 1
    if args.video:
        renderer = mujoco.Renderer(rig.model, args.vid_h, args.vid_w)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 120.0, -15.0, 1.2
        frames = []
        cap_every = max(1, round(SIM_HZ / (args.vid_fps * args.slowmo)))
        real_slowmo = SIM_HZ / (args.vid_fps * cap_every)
        print(f"[video] slowmo≈{real_slowmo:.1f}× (每 {cap_every} sim步抽1帧, 输出 {args.vid_fps}fps)")
    vid_hist_t, vid_hist_tau = [], []   # 视频右侧对比曲线的历史

    def grab_frame(t: float) -> None:
        tau = rig.joint_torque()
        vid_hist_t.append(t)
        vid_hist_tau.append(tau)
        cam.lookat[:] = rig.data.qpos[:3]
        renderer.update_scene(rig.data, cam)
        frames.append(overlay_torque(renderer.render(), tau, t,
                                     vid_hist_t, vid_hist_tau, n_cycle * control_dt))

    chart_warned: list = []   # 合并图上报失败只警告一次

    def record(k: int) -> None:
        tau = rig.joint_torque()
        t = k * control_dt
        times.append(t)
        heights.append(rig.base_height())
        tau_log.append(tau)
        if sw is not None:
            tau_rotor = tau / (GEAR * ETA)
            load_pct = np.abs(tau) / TAU_OUT_PEAK * 100.0
            metrics = {"base_height": rig.base_height()}
            for i, name in enumerate(ACTUATED_JOINTS):
                metrics[f"torque/{name}"] = float(tau[i])   # → 每关节独立 scalar 图
                metrics[f"motor_torque/{name}"] = float(tau_rotor[i])
                metrics[f"load_pct/{name}"] = float(load_pct[i])
                if args.kt > 0:
                    metrics[f"current/{name}"] = float(tau_rotor[i] / args.kt)
            sw.log(metrics, step=k)
            # 合并对比图：每 chart_every 步带全历史重 log 一次（cloud 下随步增长）
            if k % max(args.swanlab_chart_every, 1) == 0:
                try:
                    chart = build_alltau_echarts(np.array(times), np.array(tau_log))
                    sw.log({"all_joint_torque": chart}, step=k)
                except Exception as e:  # noqa: BLE001
                    if not chart_warned:
                        print(f"[warn] 合并 echarts 图上报失败: {e}", file=sys.stderr)
                        chart_warned.append(True)

    n_lower = int(args.lower * CONTROL_HZ)
    n_cycle = n_settle + n_rise + n_hold + n_lower

    def cycle_target(j: int) -> np.ndarray:
        # 一个循环：蹲姿稳定 → 平滑站起 → 站定保持 → 平滑蹲下
        if j < n_settle:
            return crouch
        if j < n_settle + n_rise:
            a = smoothstep((j - n_settle) / max(n_rise, 1))
            return (1 - a) * crouch + a * stand
        if j < n_settle + n_rise + n_hold:
            return stand
        a = smoothstep((j - n_settle - n_rise - n_hold) / max(n_lower, 1))
        return (1 - a) * stand + a * crouch

    # viewer 模式：循环往复直到关窗；否则跑 args.repeat 个循环。只记录第一个循环。
    t_wall = time.time()
    k = sim_step = 0
    while True:
        target = cycle_target(k % n_cycle)
        rig.set_targets(target)
        for _ in range(decimation):       # 逐 sim 步，按 cap_every 抽帧（慢放）
            mujoco.mj_step(rig.model, rig.data)
            sim_step += 1
            if renderer is not None and k < n_cycle and sim_step % cap_every == 0:
                grab_frame(sim_step / SIM_HZ)
        if k < n_cycle:
            record(k)
        k += 1

        if viewer_cm is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            sleep = t_wall + k * control_dt - time.time()
            if sleep > 0:
                time.sleep(sleep)
        else:
            if args.realtime:  # 按真实节奏上报，便于在 swanlab 看板看曲线逐步长出
                sleep = t_wall + k * control_dt - time.time()
                if sleep > 0:
                    time.sleep(sleep)
            if k >= n_cycle * max(args.repeat, 1):
                break

    if viewer_cm is not None:
        viewer_cm.__exit__(None, None, None)
    if sw is not None:
        sw.finish()
    if frames:
        import imageio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fps = args.vid_fps
        try:
            imageio.mimsave(args.video, frames, fps=fps)
        except Exception:  # noqa: BLE001 — 无 ffmpeg 时退回 GIF
            gif = str(Path(args.video).with_suffix(".gif"))
            imageio.mimsave(gif, frames, fps=fps)
            args.video = gif
        print(f"[out] {args.video}  ({len(frames)} 帧 @ {fps:.0f}fps)")

    times = np.array(times)
    tau_log = np.array(tau_log)           # (T, 12)
    write_outputs(times, np.array(heights), tau_log, args)
    return 0


def write_outputs(times, heights, tau, args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tau_rotor = tau / (GEAR * ETA)

    # --- 逐步序列 CSV ---
    series_csv = OUT_DIR / "torque_series.csv"
    with series_csv.open("w", newline="") as f:
        w = csv.writer(f)
        head = ["t", "base_height"]
        head += [f"tau_{n}" for n in ACTUATED_JOINTS]
        head += [f"motor_tau_{n}" for n in ACTUATED_JOINTS]
        w.writerow(head)
        for i in range(len(times)):
            w.writerow([f"{times[i]:.4f}", f"{heights[i]:.4f}",
                        *[f"{v:.4f}" for v in tau[i]],
                        *[f"{v:.4f}" for v in tau_rotor[i]]])

    # --- 每关节统计 CSV（列与现有 play_torque_stats.csv 一致）---
    stats_csv = OUT_DIR / "torque_stats.csv"
    with stats_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "mean", "std", "min", "max", "abs_max", "rms"])
        for i, name in enumerate(ACTUATED_JOINTS):
            c = tau[:, i]
            w.writerow([name, f"{c.mean():.6f}", f"{c.std():.6f}", f"{c.min():.6f}",
                        f"{c.max():.6f}", f"{np.abs(c).max():.6f}",
                        f"{np.sqrt((c**2).mean()):.6f}"])

    # --- 出图 ---
    png = OUT_DIR / "torque_curves.png"
    png_all = OUT_DIR / "torque_all.png"
    try:
        plot_curves(times, heights, tau, png)
        plot_all(times, tau, png_all)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 出图失败（缺 matplotlib?）: {e}", file=sys.stderr)
        png = png_all = None

    peak = np.abs(tau).max()
    over = [ACTUATED_JOINTS[i] for i in range(12) if np.abs(tau[:, i]).max() > EFFORT_LIMIT]
    print(f"[done] {len(times)} 控制步, 关节力矩峰值 {peak:.2f} N·m "
          f"(effort_limit {EFFORT_LIMIT}, 电机输出峰值 {TAU_OUT_PEAK})")
    if over:
        print(f"[warn] 超 effort_limit 的关节: {', '.join(over)}")
    print(f"[out] {series_csv}")
    print(f"[out] {stats_csv}")
    if png:
        print(f"[out] {png}")
    if png_all:
        print(f"[out] {png_all}")


def plot_all(times, tau, png_path) -> None:
    """所有 12 个关节力矩叠在同一张图：颜色=腿(fl/fr/rl/rr)，线型=关节类型
    (hip 实线/thigh 虚线/shank 点线)，方便横向对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    legs = ["fl", "fr", "rl", "rr"]
    leg_color = {"fl": "#1f77b4", "fr": "#ff7f0e", "rl": "#2ca02c", "rr": "#d62728"}
    type_style = {"hip": "-", "thigh": "--", "shank": ":"}
    types = ["hip", "thigh", "shank"]   # 与 ACTUATED_JOINTS 的 3×4 分组对应

    fig, ax = plt.subplots(figsize=(13, 7))
    for gi, typ in enumerate(types):
        for li, leg in enumerate(legs):
            j = gi * 4 + li
            ax.plot(times, tau[:, j], color=leg_color[leg], ls=type_style[typ],
                    lw=1.4, label=f"{leg}_{typ}")
    for lim, c, lab in [(EFFORT_LIMIT, "orange", "effort_limit"), (TAU_OUT_PEAK, "red", "motor peak")]:
        ax.axhline(lim, color=c, ls="-.", lw=1.0, label=lab)
        ax.axhline(-lim, color=c, ls="-.", lw=1.0)
    ax.set_title("all joint torques (color=leg, style: hip - / thigh -- / shank :)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("torque (N*m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_curves(times, heights, tau, png_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("hip", slice(0, 4)), ("thigh", slice(4, 8)), ("shank_link", slice(8, 12))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    legs = ["fl", "fr", "rl", "rr"]

    for ax, (gname, sl) in zip(axes.flat[:3], groups):
        for j, leg in zip(range(sl.start, sl.stop), legs):
            ax.plot(times, tau[:, j], label=leg, lw=1.2)
        for lim, c, ls, lab in [(EFFORT_LIMIT, "orange", "--", "effort_limit"),
                                 (TAU_OUT_PEAK, "red", ":", "motor peak")]:
            ax.axhline(lim, color=c, ls=ls, lw=0.8, label=lab)
            ax.axhline(-lim, color=c, ls=ls, lw=0.8)
        ax.set_title(f"{gname} joint torque (tau_joint)")
        ax.set_xlabel("t (s)"); ax.set_ylabel("N*m")
        ax.legend(loc="upper right", fontsize=7); ax.grid(alpha=0.3)

    # 4th panel: base height + peak motor-load envelope
    ax = axes.flat[3]
    ax.plot(times, heights, color="k", lw=1.5, label="base height (m)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("height (m)"); ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    load = np.abs(tau).max(axis=1) / TAU_OUT_PEAK * 100.0
    ax2.plot(times, load, color="purple", lw=1.0, alpha=0.7, label="peak load %")
    ax2.axhline(EFFORT_LIMIT / TAU_OUT_PEAK * 100, color="orange", ls="--", lw=0.8)
    ax2.set_ylabel("motor load %")
    ax.set_title("base height & peak motor load")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="站起来过程关节/电机力矩分析")
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--viewer", action="store_true", help="启动 GUI 窗口看站起来动作（需真实显示/GPU）")
    ap.add_argument("--video", default=None, help="离屏渲染成视频文件（如 outputs/standup_torque/standup.mp4）")
    ap.add_argument("--vid-h", type=int, default=480, help="视频高")
    ap.add_argument("--vid-w", type=int, default=640, help="视频宽")
    ap.add_argument("--vid-fps", type=int, default=30, help="视频输出帧率")
    ap.add_argument("--slowmo", type=float, default=1.0, help="慢放倍数（10=比真实慢10倍）")
    ap.add_argument("--swanlab", action="store_true", help="上报 swanlab")
    ap.add_argument("--swanlab-mode", default="local", choices=["local", "cloud", "offline"],
                    help="cloud=云端看板真·实时不用刷新（需登录）；local=本地看板需手动刷新")
    ap.add_argument("--realtime", action="store_true", help="按真实节奏跑（headless 下让 swanlab 曲线逐步长出）")
    ap.add_argument("--swanlab-chart-every", type=int, default=5,
                    help="每几步重 log 一次合并对比图 all_joint_torque（越小越流畅、开销越大）")
    ap.add_argument("--init-height", type=float, default=0.16, help="蹲姿机身初始高度 m")
    ap.add_argument("--stand-thigh", type=float, default=STAND_THIGH_PHYS, help="站姿大腿俯仰 rad")
    ap.add_argument("--stand-shank", type=float, default=STAND_SHANK_PHYS, help="站姿小腿俯仰 rad")
    ap.add_argument("--crouch-thigh", type=float, default=CROUCH_THIGH_PHYS, help="蹲姿大腿俯仰 rad")
    ap.add_argument("--crouch-shank", type=float, default=CROUCH_SHANK_PHYS, help="蹲姿小腿俯仰 rad")
    ap.add_argument("--kp", type=float, default=0.0, help="覆盖伺服 kp（0=用 MJCF 的 25；调硬看真实跟踪力矩）")
    ap.add_argument("--settle", type=float, default=0.5, help="蹲姿稳定时长 s")
    ap.add_argument("--rise", type=float, default=1.5, help="站起来时长 s")
    ap.add_argument("--hold", type=float, default=1.0, help="站定保持时长 s")
    ap.add_argument("--lower", type=float, default=1.5, help="蹲下时长 s（循环用）")
    ap.add_argument("--repeat", type=int, default=1, help="headless 跑几个循环（viewer 模式忽略，循环到关窗）")
    ap.add_argument("--kt", type=float, default=0.0, help="转子力矩常数 N·m/A（>0 才算电流，待标定）")
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
