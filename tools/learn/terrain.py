"""给 Go2 场景加地形。

MuJoCo 里加载地形有三条路，这个文件把三条都用上了：

1. box geom 拼装        —— 楼梯、障碍块这类有棱有角的规则几何
2. heightfield (hfield) —— 连续起伏地面，MuJoCo 的原生地形机制，
                           高度数据可以来自数组，也可以来自 PNG 灰度图
3. 直接读 XML           —— 见 go2_walk.py 的 --scene，整个场景换掉

前两种都用 MjSpec 在内存里改模型，不生成 XML 文件。
这样做的原因：go2.xml 里 meshdir="assets" 是相对主模型文件目录解析的，
把生成的 scene.xml 写到别的目录会导致 mesh 全部找不到。
"""

import mujoco
import numpy as np

GRAY = [0.55, 0.55, 0.60, 1.0]
WARN = [0.75, 0.55, 0.25, 1.0]


def _box(spec, name, size, pos, rgba=GRAY):
    g = spec.worldbody.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = size          # 半长
    g.pos = pos            # 中心
    g.rgba = rgba
    return g


def flat(spec):
    """什么都不加，基准。"""
    return spec


def stairs(spec, n=8, rise=0.08, run=0.30, width=1.5, x0=1.0):
    """上行楼梯。rise 每级高度，run 每级进深。

    每级做成从地面长到该高度的实心块，避免踏面之间出现悬空缝隙。
    """
    for i in range(n):
        h = rise * (i + 1)
        _box(spec, f"step{i}",
             size=[run / 2, width / 2, h / 2],
             pos=[x0 + run * (i + 0.5), 0.0, h / 2])
    # 顶部平台，走上去有地方站
    top = rise * n
    _box(spec, "landing",
         size=[0.8, width / 2, top / 2],
         pos=[x0 + run * n + 0.8, 0.0, top / 2])
    return spec


def boxes(spec, n=14, seed=0, area=(1.0, 6.0, -1.2, 1.2), h=(0.03, 0.10)):
    """随机散布的矮障碍块，模拟碎石地。"""
    rng = np.random.default_rng(seed)
    x_lo, x_hi, y_lo, y_hi = area
    for i in range(n):
        hi = rng.uniform(*h)
        _box(spec, f"rock{i}",
             size=[rng.uniform(0.08, 0.20), rng.uniform(0.08, 0.20), hi / 2],
             pos=[rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), hi / 2],
             rgba=WARN)
    return spec


def slope(spec, angle_deg=10.0, length=4.0, width=1.5, x0=1.0):
    """斜坡。用一个绕 y 轴转过 angle 的扁box。"""
    a = np.radians(angle_deg)
    g = _box(spec, "ramp",
             size=[length / 2, width / 2, 0.02],
             pos=[x0 + length / 2 * np.cos(a), 0.0, length / 2 * np.sin(a)])
    # 绕 y 轴负向转 -> 沿 +x 上坡
    g.quat = [np.cos(-a / 2), 0.0, np.sin(-a / 2), 0.0]
    return spec


def _hfield(spec, name, elev, half_x, half_y, z_max, center_x, base=0.05):
    """把一个 (nrow, ncol) 高度数组挂成 hfield geom。

    elev 会被归一化到 [0,1]，真实高度 = elev_norm * z_max。
    size = [x半径, y半径, 最大高度, 底座厚度]，底座是地面以下的实体部分，
    必须 > 0，否则薄地形会被脚捅穿。
    """
    e = np.asarray(elev, dtype=float)
    e -= e.min()
    if e.max() > 0:
        e /= e.max()

    # 必须先删掉原场景的无限大 floor 平面。
    # hfield 的实体从 -base 延伸到 +z_max，而 floor 在 z=0 横穿过去，
    # 两个碰撞面重叠会把脚夹住 —— 表现为机器人站得好好的但原地零位移。
    try:
        spec.delete(spec.geom("floor"))
    except (KeyError, ValueError):
        pass

    hf = spec.add_hfield()
    hf.name = name
    hf.nrow, hf.ncol = e.shape
    hf.size = [half_x, half_y, z_max, base]
    hf.userdata = e.flatten().tolist()

    g = spec.worldbody.add_geom()
    g.name = f"{name}_geom"
    g.type = mujoco.mjtGeom.mjGEOM_HFIELD
    g.hfieldname = name
    g.pos = [center_x, 0.0, 0.0]
    g.rgba = [0.45, 0.42, 0.38, 1.0]
    return g


def rough(spec, n=96, z_max=0.06, seed=0, half_x=8.0, half_y=3.0, center_x=0.0):
    """连续起伏的粗糙地面（hfield）。z_max 是最大起伏高度。"""
    rng = np.random.default_rng(seed)
    # 叠几层不同尺度的噪声，比纯白噪声更像真实地形
    e = np.zeros((n, n))
    for scale, amp in ((4, 1.0), (8, 0.5), (16, 0.25), (32, 0.12)):
        coarse = rng.random((scale, scale))
        idx = (np.linspace(0, scale - 1, n)).astype(int)
        e += amp * coarse[np.ix_(idx, idx)]
    _hfield(spec, "rough", e, half_x, half_y, z_max, center_x)
    return spec


def waves(spec, n=128, z_max=0.08, periods=5.0, half_x=8.0, half_y=3.0, center_x=0.0):
    """规则正弦波纹地面（hfield）。起伏是确定的，便于复现对比。"""
    u = np.linspace(0, 2 * np.pi * periods, n)
    e = np.sin(u)[None, :] * np.ones((n, 1))
    _hfield(spec, "waves", e, half_x, half_y, z_max, center_x)
    return spec


def from_png(spec, path, z_max=0.15, half_x=8.0, half_y=3.0, center_x=0.0):
    """从 PNG 灰度图读高度（hfield 最常见的用法）。

    这是 MuJoCo 原生支持的方式：也可以在 XML 里写
        <hfield name="t" file="terrain.png" size="5 3 0.15 0.05"/>
    这里用 Python 读是为了能先做归一化和裁剪。
    """
    import PIL.Image
    img = PIL.Image.open(path).convert("L")
    _hfield(spec, "png", np.asarray(img, dtype=float), half_x, half_y, z_max, center_x)
    return spec


BUILDERS = {
    "flat": flat,
    "stairs": stairs,
    "boxes": boxes,
    "slope": slope,
    "rough": rough,
    "waves": waves,
}


def build(scene_path, kind="flat", png=None, **kw):
    """加载基础场景 + 叠加地形，返回编译好的 MjModel。

    scene_path 换成别的 XML 就等于整个场景换掉（第 3 条路）。
    """
    spec = mujoco.MjSpec.from_file(str(scene_path))
    if png:
        from_png(spec, png, **kw)
    elif kind in BUILDERS:
        BUILDERS[kind](spec, **kw)
    else:
        raise SystemExit(f"未知地形 {kind!r}，可选：{list(BUILDERS)}")
    return spec.compile()


def spawn_clearance(model):
    """出生点需要额外抬高多少，才不会一开始就埋在地形里。

    keyframe 里的 base 高度是按平地写的（脚刚好触地）。地形一旦有起伏，
    脚就会生成在地面以下，腿被卡在网格里 —— 表现为腿摆不开、原地不动。
    """
    if model.nhfield == 0:
        return 0.0
    return float(model.hfield_size[:, 2].max())  # 各 hfield 的最大起伏高度
