#!/usr/bin/env python3
"""绘制渐开线变位齿轮(profile-shifted involute gear)。

原理:
  变位齿轮与标准齿轮共用同一把齿条刀具,只是切齿时刀具沿径向移开 x*m
  (x 为变位系数,正变位远离轮心)。因此:
    - 分度圆 r = m*z/2、基圆 rb = r*cos(alpha) 不变;
    - 齿顶圆 ra = r + m*(ha* + x)
    - 齿根圆 rf = r - m*(ha* + c* - x)
    - 分度圆齿厚 s = m*(pi/2 + 2*x*tan(alpha))  -> 正变位齿根变厚
  齿廓仍是同一基圆的渐开线,只是取用的区段和齿槽宽度变了。

齿廓生成:
  渐开线上半径 rho 处的压力角 a_rho = arccos(rb/rho),
  该点相对齿中心线的角坐标 theta(rho) = psi + inv(alpha) - inv(a_rho),
  其中 psi = (pi/2 + 2*x*tan(alpha))/z 是分度圆上半齿厚角,
  inv(a) = tan(a) - a 为渐开线函数。

用法示例:
  python3 involute_gear.py -m 2 -z 17 -x 0.5
  python3 involute_gear.py -m 2 -z 10 -x 0.4 --compare -o gear.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt


def inv(a):
    """渐开线函数 inv(a) = tan(a) - a"""
    return np.tan(a) - a


def gear_outline(m, z, alpha_deg=20.0, x=0.0, ha_star=1.0, c_star=0.25,
                 n_flank=60, n_arc=12):
    """生成变位齿轮完整轮廓坐标 (N,2),并返回主要尺寸字典。"""
    alpha = np.deg2rad(alpha_deg)
    r = m * z / 2.0                     # 分度圆半径
    rb = r * np.cos(alpha)              # 基圆半径
    ra = r + m * (ha_star + x)          # 齿顶圆半径
    rf = r - m * (ha_star + c_star - x) # 齿根圆半径

    # 根切检查:x_min = ha* - z*sin^2(alpha)/2
    x_min = ha_star - z * np.sin(alpha) ** 2 / 2.0
    if x < x_min - 1e-9:
        print(f"警告: x={x:.3f} < x_min={x_min:.3f},会发生根切"
              f"(绘图按理论渐开线,不含根切曲线)")

    # 分度圆上半齿厚角
    psi = (np.pi / 2 + 2 * x * np.tan(alpha)) / z

    def theta(rho):
        """半径 rho 处齿廓相对齿中心线的角坐标(取正侧)"""
        a_rho = np.arccos(np.clip(rb / rho, -1.0, 1.0))
        return psi + inv(alpha) - inv(a_rho)

    # 齿顶变尖检查:theta(ra) < 0 说明两侧渐开线在 ra 之前已相交
    theta_a = theta(ra)
    if theta_a < 0:
        # 二分求齿廓交点半径,把齿顶削平到该处
        lo, hi = rb * (1 + 1e-9), ra
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if theta(mid) > 0:
                lo = mid
            else:
                hi = mid
        ra = lo
        theta_a = max(theta(ra), 0.0)
        print(f"警告: 齿顶变尖,齿顶圆削减到 ra={ra:.4f}"
              f"(可减小 x 或减小 ha*)")

    # 渐开线起始半径:基圆以下没有渐开线,用径向线连到齿根圆
    rho0 = max(rb, rf)
    theta_b = theta(rho0)

    tau = 2 * np.pi / z  # 齿距角

    # --- 单个齿的轮廓(齿中心线在角度 0,逆时针走向) ---
    pts = []

    # 1) 左半齿槽根部圆弧: -tau/2 -> -theta_b
    ang = np.linspace(-tau / 2, -theta_b, n_arc)
    pts.append(np.column_stack([rf * np.cos(ang), rf * np.sin(ang)]))

    # 2) 若 rf < rb,径向段 rf -> rb(近似代替过渡圆角)
    if rf < rb:
        rr = np.linspace(rf, rb, 4)
        pts.append(np.column_stack([rr * np.cos(-theta_b),
                                    rr * np.sin(-theta_b)]))

    # 3) 左侧渐开线齿廓: rho0 -> ra
    rho = np.linspace(rho0, ra, n_flank)
    th = theta(rho)
    pts.append(np.column_stack([rho * np.cos(-th), rho * np.sin(-th)]))

    # 4) 齿顶圆弧: -theta_a -> +theta_a
    ang = np.linspace(-theta_a, theta_a, n_arc)
    pts.append(np.column_stack([ra * np.cos(ang), ra * np.sin(ang)]))

    # 5) 右侧渐开线齿廓: ra -> rho0
    pts.append(np.column_stack([rho[::-1] * np.cos(th[::-1]),
                                rho[::-1] * np.sin(th[::-1])]))

    # 6) 径向段 rb -> rf
    if rf < rb:
        rr = np.linspace(rb, rf, 4)
        pts.append(np.column_stack([rr * np.cos(theta_b),
                                    rr * np.sin(theta_b)]))

    # 7) 右半齿槽根部圆弧: theta_b -> tau/2
    ang = np.linspace(theta_b, tau / 2, n_arc)
    pts.append(np.column_stack([rf * np.cos(ang), rf * np.sin(ang)]))

    tooth = np.vstack(pts)

    # --- 旋转复制出 z 个齿 ---
    outline = []
    for k in range(z):
        a = k * tau
        rot = np.array([[np.cos(a), -np.sin(a)],
                        [np.sin(a),  np.cos(a)]])
        outline.append(tooth @ rot.T)
    outline.append(outline[0][:1])  # 闭合
    outline = np.vstack(outline)

    dims = dict(r=r, rb=rb, ra=ra, rf=rf, x_min=x_min,
                s=m * (np.pi / 2 + 2 * x * np.tan(alpha)))
    return outline, dims


def draw_gear(ax, m, z, alpha_deg, x, ha_star, c_star,
              color="tab:blue", label=None, show_circles=True):
    outline, d = gear_outline(m, z, alpha_deg, x, ha_star, c_star)
    ax.plot(outline[:, 0], outline[:, 1], color=color, lw=1.2,
            label=label or f"x = {x}")

    if show_circles:
        t = np.linspace(0, 2 * np.pi, 400)
        for radius, style, name in [(d["ra"], ":", "tip"),
                                    (d["r"], "-.", "pitch"),
                                    (d["rb"], "--", "base"),
                                    (d["rf"], ":", "root")]:
            ax.plot(radius * np.cos(t), radius * np.sin(t),
                    style, color="gray", lw=0.6)
            ax.annotate(name, (radius * np.cos(np.pi / 4),
                               radius * np.sin(np.pi / 4)),
                        fontsize=7, color="gray")
    return d


def main():
    p = argparse.ArgumentParser(description="绘制渐开线变位齿轮")
    p.add_argument("-m", "--module", type=float, default=2.0, help="模数 m")
    p.add_argument("-z", "--teeth", type=int, default=17, help="齿数 z")
    p.add_argument("-a", "--alpha", type=float, default=20.0,
                   help="压力角(度),默认 20")
    p.add_argument("-x", "--shift", type=float, default=0.5,
                   help="变位系数 x,正变位为正")
    p.add_argument("--ha", type=float, default=1.0, help="齿顶高系数 ha*")
    p.add_argument("--c", type=float, default=0.25, help="顶隙系数 c*")
    p.add_argument("--compare", action="store_true",
                   help="叠加画出 x=0 的标准齿轮作对比")
    p.add_argument("-o", "--output", default="gear.png", help="输出图片路径")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 8))
    d = draw_gear(ax, args.module, args.teeth, args.alpha, args.shift,
                  args.ha, args.c, color="tab:blue",
                  label=f"shifted  x = {args.shift}")
    if args.compare:
        draw_gear(ax, args.module, args.teeth, args.alpha, 0.0,
                  args.ha, args.c, color="tab:red",
                  label="standard  x = 0", show_circles=False)

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"m={args.module}  z={args.teeth}  "
                 f"alpha={args.alpha}deg  x={args.shift}")
    ax.grid(alpha=0.2)

    print(f"分度圆 r  = {d['r']:.4f}")
    print(f"基圆   rb = {d['rb']:.4f}")
    print(f"齿顶圆 ra = {d['ra']:.4f}")
    print(f"齿根圆 rf = {d['rf']:.4f}")
    print(f"分度圆齿厚 s = {d['s']:.4f}")
    print(f"不根切最小变位系数 x_min = {d['x_min']:.4f}")

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
