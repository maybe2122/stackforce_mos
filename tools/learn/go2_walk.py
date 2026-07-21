"""让 Go2 在 MuJoCo 里走起来 —— 不用 RL，纯运动学 + 步态 + PD。

完整链路（对应 doc/顺序.md 里那张图）：
    速度指令 vx
        -> 步态相位（trot）
        -> 足端轨迹 p_des (hip 系)
        -> IK
        -> 关节目标角 q_des
        -> PD
        -> 力矩 τ
        -> MuJoCo 电机

用法：
    python tools/learn/go2_walk.py -T 600           # 开窗，用键盘遥控
    python tools/learn/go2_walk.py --vx 0.4 --wz 0.5  # 指定固定指令
    python tools/learn/go2_walk.py --headless -T 8 --vx 0.4  # 无窗跑，打印达成率

窗口里的键盘遥控（先点一下 3D 窗口）：
    W / S   前进 / 后退      ±0.1 m/s
    A / D   左转 / 右转      ±0.2 rad/s
    Q / E   左移 / 右移      ±0.1 m/s
    空格    全部指令清零
"""

import argparse
import pathlib
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import terrain  # noqa: E402
from leg_kinematics import D, ik, jacobian  # noqa: E402

SCENE = pathlib.Path.home() / "mujoco_menagerie/unitree_go2/scene.xml"

# ctrl / qpos 里的腿顺序、abduction 偏置符号（左 +，右 -）、以及 hip 在 base 系的位置。
# hip 位置取自 go2.xml，转向时算 ω × r 要用。
LEGS = [
    ("FL", +1, +0.1934),
    ("FR", -1, +0.1934),
    ("RL", +1, -0.1934),
    ("RR", -1, -0.1934),
]
HIP_Y = 0.0465  # hip 关节相对 base 中线的横向距离

# trot：对角腿同相。相位偏移 0 / 0.5
PHASE_OFFSET = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}


def foot_target(phase, v_foot, stand_h, step_h, t_stance):
    """给定腿的相位 [0,1) 和该腿支撑期应有的水平速度，返回足端偏移 (dx, dy, z)。

    duty = 0.5：前半周期支撑（脚贴地往后扫），后半周期摆动（抬起来送回去）。
    v_foot 是「身体要往哪走」，足端在支撑期就要朝反方向扫同样的量。
    """
    step = v_foot * t_stance  # (step_x, step_y)

    if phase < 0.5:  # 支撑相 stance
        s = phase / 0.5
        dx, dy = step * (0.5 - s)  # 从 +step/2 扫到 -step/2
        z = -stand_h
    else:  # 摆动相 swing
        s = (phase - 0.5) / 0.5
        dx, dy = step * (s - 0.5)  # 从 -step/2 送回 +step/2
        z = -stand_h + step_h * np.sin(np.pi * s)  # 抬腿，落地时归零

    return dx, dy, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.3, help="前进速度指令 m/s")
    ap.add_argument("--vy", type=float, default=0.0, help="侧移速度指令 m/s（左为正）")
    ap.add_argument("--wz", type=float, default=0.0, help="转向角速度指令 rad/s（左转为正）")
    ap.add_argument("--period", type=float, default=0.5, help="步态周期 s")
    ap.add_argument("--stand-h", type=float, default=0.28, help="站立高度 m")
    ap.add_argument("--step-h", type=float, default=0.06, help="抬腿高度 m")
    ap.add_argument("--mass", type=float, default=15.2, help="整机质量 kg（重力前馈用）")
    ap.add_argument("--kp", type=float, default=200.0)
    ap.add_argument("--kd", type=float, default=3.0)
    ap.add_argument("--terrain", default="flat",
                    choices=list(terrain.BUILDERS), help="内置地形")
    ap.add_argument("--terrain-png", default=None,
                    help="从 PNG 灰度图读高度场，覆盖 --terrain")
    ap.add_argument("--scene", default=None,
                    help="直接指定场景 XML，整个换掉（默认用 menagerie 的 go2 scene）")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("-T", "--duration", type=float, default=10.0)
    args = ap.parse_args()

    model = terrain.build(args.scene or SCENE, args.terrain, png=args.terrain_png)
    data = mujoco.MjData(model)

    # 从 home keyframe 起步。地形有起伏时要先抬高出生点，否则脚埋在地里。
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] += terrain.spawn_clearance(model) + 0.02

    dt = model.opt.timestep
    t_stance = args.period * 0.5
    # 力矩上限直接读 XML 里的 ctrlrange，别自己瞎写
    tau_lim = model.actuator_ctrlrange[:, 1].copy()

    # 落定：先原地站 0.5s 让机器人踩实地面，再开始走。
    # 否则起步瞬间的自由落体会被算进步态里，也容易一上来就摔。
    q_home = data.qpos[7:19].copy()
    for _ in range(int(0.5 / dt)):
        tau = args.kp * (q_home - data.qpos[7:19]) - args.kd * data.qvel[6:18]
        data.ctrl[:] = np.clip(tau, -tau_lim, tau_lim)
        mujoco.mj_step(model, data)
    data.time = 0.0

    n_steps = int(args.duration / dt)

    # 机体系速度要逐步积分：一旦转向，世界系净位移就不代表机体前进量了
    # （绕圈走会回到原点，测出来 vx≈0）。
    body_vel_sum = np.zeros(2)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    def model_up(d):
        """base 局部 z 轴在世界系的 z 分量：1=直立，0=侧翻90°。"""
        return d.xmat[base_id].reshape(3, 3)[2, 2]

    def world_yaw():
        w, x, y, z = data.qpos[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # yaw 必须逐步累积。直接比较首尾的 arctan2 只能分辨 ±180°，
    # 转一圈以上就会混叠成完全错误（甚至反号）的读数。
    yaw_total = 0.0
    prev_yaw = world_yaw()

    def control(t):
        """一个控制周期：相位 -> 足端 -> IK -> PD(+重力前馈) -> τ"""
        q = data.qpos[7:19]   # 前 7 个是 base 的 free joint
        qd = data.qvel[6:18]

        q_des = np.zeros(12)
        stance = np.zeros(4, dtype=bool)
        for i, (leg, side, hip_x) in enumerate(LEGS):
            phase = (t / args.period + PHASE_OFFSET[leg]) % 1.0
            stance[i] = phase < 0.5

            # 每条腿的落脚点速度不一样：刚体上一点的速度 = v + ω × r。
            # r 是该足端名义位置（相对 base 中心），ω = (0,0,wz)：
            #   (ω × r)_x = -wz * r_y      (ω × r)_y = +wz * r_x
            r_y = side * (HIP_Y + D)
            v_foot = np.array([
                args.vx - args.wz * r_y,
                args.vy + args.wz * hip_x,
            ])

            dx, dy, z = foot_target(phase, v_foot, args.stand_h, args.step_h, t_stance)
            q_des[3 * i:3 * i + 3] = ik(np.array([dx, side * D + dy, z]), d=side * D)

        tau = args.kp * (q_des - q) + args.kd * (0.0 - qd)

        # 重力前馈：支撑腿平分体重，用 τ = JᵀF 换算成关节力矩。
        # 没有这一项，纯 PD 的稳态误差 = τ_重力/Kp，机器人会明显下沉。
        n_stance = max(stance.sum(), 1)
        F = np.array([0.0, 0.0, -args.mass * 9.81 / n_stance])
        for i, (leg, side, _) in enumerate(LEGS):
            if stance[i]:
                sl = slice(3 * i, 3 * i + 3)
                tau[sl] += jacobian(q[sl], d=side * D).T @ F

        return np.clip(tau, -tau_lim, tau_lim)

    if args.headless:
        for k in range(n_steps):
            data.ctrl[:] = control(data.time)
            mujoco.mj_step(model, data)
            yaw = world_yaw()
            yaw_total += np.arctan2(np.sin(yaw - prev_yaw), np.cos(yaw - prev_yaw))
            prev_yaw = yaw
            # 世界系线速度投影回机体系（只用 yaw，忽略俯仰/横滚）
            c, s = np.cos(yaw), np.sin(yaw)
            vw = data.qvel[0:2]
            body_vel_sum += np.array([c * vw[0] + s * vw[1],
                                      -s * vw[0] + c * vw[1]]) * dt
            # 摔倒判据用姿态而不是绝对高度：爬楼梯时高度本来就会上升，
            # 用高度会误判。这里看 base 的 z 轴还有多「朝上」。
            upright = model_up(data)
            if upright < 0.5:  # 倾斜超过 60°
                print(f"❌ 在 t={data.time:.2f}s 摔倒"
                      f"（机身倾角 {np.degrees(np.arccos(np.clip(upright,-1,1))):.0f}°）")
                return 1
        vx_avg, vy_avg = body_vel_sum / args.duration
        d_yaw = yaw_total

        print(f"✅ 站住并走完 {args.duration}s（机体系平均速度）")
        print(f"   vx {vx_avg:+.3f} m/s（指令 {args.vx:+.2f}）")
        print(f"   vy {vy_avg:+.3f} m/s（指令 {args.vy:+.2f}）")
        print(f"   wz {d_yaw / args.duration:+.3f} rad/s（指令 {args.wz:+.2f}）"
              f"  累计转过 {np.degrees(d_yaw):+.1f}°")
        print(f"   base 高度 {data.qpos[2]:.3f} m")
        return 0

    # --- 键盘遥控 ---
    # viewer 的 key_callback 只在按键「按下」时触发一次，所以做成增量调节，
    # 不是「按住才走」。指令直接改 args，control() 每周期都会重新读。
    KEYS = {
        87: ("vx", +0.1), 83: ("vx", -0.1),   # W / S  前后
        65: ("wz", +0.2), 68: ("wz", -0.2),   # A / D  左右转
        81: ("vy", +0.1), 69: ("vy", -0.1),   # Q / E  左右平移
    }
    LIMITS = {"vx": 0.8, "vy": 0.4, "wz": 1.2}

    def show():
        print(f"\r指令  vx {args.vx:+.2f}  vy {args.vy:+.2f}  wz {args.wz:+.2f}   ",
              end="", flush=True)

    def on_key(keycode):
        if keycode == 32:  # 空格：急停
            args.vx = args.vy = args.wz = 0.0
        elif keycode in KEYS:
            name, delta = KEYS[keycode]
            lim = LIMITS[name]
            setattr(args, name, float(np.clip(getattr(args, name) + delta, -lim, lim)))
        else:
            return
        show()

    print("W/S 前后   A/D 转向   Q/E 平移   空格 停   （点一下 3D 窗口再按键）")
    show()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running() and data.time < args.duration:
            step_start = time.time()
            data.ctrl[:] = control(data.time)
            mujoco.mj_step(model, data)
            viewer.sync()
            # 实时对齐
            lag = dt - (time.time() - step_start)
            if lag > 0:
                time.sleep(lag)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
