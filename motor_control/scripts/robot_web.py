#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四足机器人操控网页 —— 站立（趴 → 站）。

仿手柄的操控面板，本期只实现「站立」一个动作的完整闭环：
  1. 趴姿标定：机器人自然趴下，读取全部 12 关节转子角 -> prone
  2. 站姿标定：人手扶撑成站立姿态，读取全部 12 关节转子角 -> stand
  3. 计算 趴->站 每关节转动量 delta=stand-prone（带符号=方向），存配置文件
  4. 站立：先读当前角，逐关节与配置趴姿比 |Δ|<阈值 才放行；放行后目标=当前+delta，
     按关节限速（默认 0.1 rad/s）做平滑插值，通过 motor_ctrl servo 流式发位置指令。

底层复用 motor_control/Linux/build/motor_ctrl：
  - 读角度：motor_ctrl <port> <id> read
  - 流式位置伺服：motor_ctrl <port> servo <kp> <kw>（stdin 喂目标位置，插值在本脚本里算）

纯标准库（http.server），默认仅监听 127.0.0.1，避免把机器人控制暴露到局域网。
运行：python3 scripts/robot_web.py  然后浏览器开 http://127.0.0.1:8000
设计文档见 motor_control/doc/四足站立控制设计.md。
"""

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# --- 路径与常量（与 motor_web.py 对齐） ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SDK_ROOT = os.path.normpath(os.path.join(_HERE, "..", "Linux"))
SDK_ROOT = os.environ.get("UNITREE_MOTOR_SDK", _DEFAULT_SDK_ROOT)
MOTOR_CTRL = os.path.join(SDK_ROOT, "build", "motor_ctrl")

CONFIG_DIR = os.path.normpath(os.path.join(_HERE, "..", "config"))
JOINT_MAP_DEFAULT = os.path.join(CONFIG_DIR, "joint_map.default.json")
STAND_CONFIG = os.path.join(CONFIG_DIR, "stand_config.json")

GEAR_RATIO = 6.33
# 分段站立顺序：先小腿(shank)→再大腿(thigh)→再 hip。每段只驱动一组关节，
# 上一组到位并保持后再启动下一组，降低同时抬升的峰值力矩/电流。
STAGE_ORDER = ("shank", "thigh", "hip")
SUDO_PASSWORD = "1"
NEED_SUDO = os.geteuid() != 0
MAX_LOG = 4000

# 站立保持时「单关节微调」：每次最多 ±JOG_STEP_MAX 度（关节角），两次微调间隔至少
# JOG_COOLDOWN 秒（安全间隔，给人观察/急停的时间）。微调本身按 JOG_VMAX 限速缓慢插值。
JOG_STEP_MAX = 10.0    # 每次微调最大角度（关节°）
JOG_COOLDOWN = 1.0     # 两次微调最小安全间隔（s）
JOG_VMAX = 0.1         # 微调限速（关节角速度 rad/s），越小越慢越稳

# 「重锚趴姿」静止校验：间隔 REANCHOR_SETTLE 秒读两次，任一关节漂移超过
# REANCHOR_DRIFT_MAX（关节°）判定仍在动（如小腿重力下垂未靠住限位），拒绝写入。
REANCHOR_SETTLE = 5.0
REANCHOR_DRIFT_MAX = 0.5


# ----------------------------------------------------------------- 工具函数
def parse_angle_line(line):
    """解析 motor_ctrl read 的输出，取转子角(rotor=data.Pos)。
        ANGLE id=0 ok=1 rotor=1.234 joint=0.195 deg=11.17 temp=30 err=0
    """
    m = re.match(r"\s*ANGLE\s+id=(\d+)\s+ok=([01])(.*)", line)
    if not m:
        return None
    mid = int(m.group(1))
    ok = m.group(2) == "1"
    rest = m.group(3)

    def grab(key):
        mm = re.search(rf"{key}=(-?\d+(?:\.\d+)?)", rest)
        return float(mm.group(1)) if mm else None

    if ok:
        return {"id": mid, "ok": True, "rotor": grab("rotor"),
                "temp": grab("temp"), "err": grab("err")}
    return {"id": mid, "ok": False}


def load_joint_template():
    """读默认映射模板，返回 (joints, execute)。"""
    with open(JOINT_MAP_DEFAULT, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["joints"], j.get("execute", {})


def joints_by_port(joints):
    """按串口分组，保持顺序。返回 {port: [joint, ...]}。"""
    groups = {}
    for jt in joints:
        groups.setdefault(jt["port"], []).append(jt)
    return groups


def parse_opt_gain(raw):
    """解析前端传来的可选增益（K_P / K_W）：
       空 -> None（表示沿用配置 execute 里的默认值）；
       非数字 -> "ERR"（调用方据此报错）；
       合法 -> clamp 到 [0, 25.599]（手册 K_P/K_W 上限）。
    """
    s = str(raw).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return "ERR"
    return max(0.0, min(25.599, v))


def parse_opt_num(raw, lo, hi):
    """解析前端传来的可选数值（速度 / 时长 / 频率 / 阈值等）：
       空 -> None（沿用配置默认）；非数字 -> "ERR"；合法 -> clamp 到 [lo, hi]。
    """
    s = str(raw).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return "ERR"
    return max(lo, min(hi, v))


# ----------------------------------------------------------------- 控制器
class RobotController:
    def __init__(self):
        self.lock = threading.RLock()
        self.log = []
        self.dropped = 0
        self.busy = False           # 标定 / 读取等短任务进行中
        self.standing_thread = None  # 站立执行线程
        self.abort = threading.Event()
        self.servo_procs = {}        # port -> Popen（流式伺服进程）
        self.hold_scope = ""         # 当前 servo 锁位保持的范围描述（空=已松开）；供前端「是否松开」显示
        self.hold_targets = {}       # name -> 当前锁位保持的目标转子角(rad)，供「单关节微调」做增量基准
        self.hold_tff = {}           # name -> 保持时施加的前馈力矩(转子侧 N·m)
        self.jog_last = {}           # name -> 上次微调的单调钟时间戳，用于 2s 安全间隔
        self.notice = None           # 供前端弹窗的后台通知 {id,text,level}；轮询到新 id 弹一次
        self.notice_id = 0

        # 标定与配置数据（关节角均以「转子角 rad」存储，与 motor_ctrl 的 Pos 一致）
        self.joints, self.execute = load_joint_template()
        self.prone = {}              # name -> rotor
        self.stand = {}              # name -> rotor
        self.current = {}            # name -> rotor（最近一次读取）
        self.current_ok = {}         # name -> bool
        self.last_dev = {}           # name -> 与配置趴姿的偏差（校验用）
        self.feedback = {}           # name -> {vel,tau,errd,temp,...} servo 实时反馈（保持/运动时）
        self.tau_peak = {}           # name -> 峰值 |力矩|（转子侧 N·m），运动/保持期间持续更新，可清零
        self._id2name = {int(jt["id"]): jt["name"] for jt in self.joints}  # 电机 id -> 关节名
        self._config_mtime = None    # stand_config.json 的 mtime，用于检测外部更新后自动重载
        self.config = self._load_config()
        self.phase = "IDLE"          # IDLE/PRONE_DONE/STAND_DONE/CONFIGURED/EXECUTING/STANDING/ERROR
        self.status = "就绪"
        if self.config:
            self.phase = "CONFIGURED"

    # -- 日志 --
    def append(self, line):
        with self.lock:
            self.log.append(str(line))
            over = len(self.log) - MAX_LOG
            if over > 0:
                del self.log[:over]
                self.dropped += over

    def set_status(self, s):
        with self.lock:
            self.status = s

    def _notify(self, text, level="bad"):
        """记录一条供前端弹窗的通知。后台线程里的拒绝/异常无法通过 api() 返回值
        告知页面（动作早已异步返回 ok），改为写入 notice，前端轮询到新的 id 就 alert 一次。"""
        with self.lock:
            self.notice_id += 1
            self.notice = {"id": self.notice_id, "text": str(text), "level": level}

    def _load_config(self):
        if os.path.isfile(STAND_CONFIG):
            try:
                with open(STAND_CONFIG, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                try:
                    self._config_mtime = os.path.getmtime(STAND_CONFIG)
                except OSError:
                    self._config_mtime = None
                return cfg
            except Exception as e:
                self.append(f"[警告] 读取 {STAND_CONFIG} 失败: {e}")
        self._config_mtime = None
        return None

    def _reload_config_now(self):
        """手动从磁盘重新加载 stand_config.json，并打印一行摘要（供页面「重新加载配置」按钮）。"""
        cfg = self._load_config()   # 内部会更新 self._config_mtime
        with self.lock:
            self.config = cfg
            if cfg is not None and self.phase == "IDLE":
                self.phase = "CONFIGURED"
        if cfg is None:
            self.append(f"[配置] 未找到或无法读取 {STAND_CONFIG}（请先在 motor_web 标定并保存）。")
            return False, f"未找到/无法读取 {STAND_CONFIG}"
        js = cfg.get("joints", [])
        n_dir = sum(1 for j in js if j.get("dir") is not None)
        n_ver = sum(1 for j in js if j.get("verified"))
        self.append(f"[配置] 已重新加载 {STAND_CONFIG}：机器人={cfg.get('robot')} "
                    f"减速比={cfg.get('gear_ratio')} 关节={len(js)} 含方向={n_dir} "
                    f"已验证={n_ver} 生成于={cfg.get('_generated', '?')}")
        return True, "ok"

    def _refresh_config_if_changed(self):
        """若磁盘上的 stand_config.json 比内存里的新（被 motor_web 标定 / 手动改过），
        自动重载，保证关节角表里的趴姿/站姿始终跟配置文件一致。"""
        try:
            mtime = os.path.getmtime(STAND_CONFIG) if os.path.isfile(STAND_CONFIG) else None
        except OSError:
            mtime = None
        if mtime == self._config_mtime:
            return
        cfg = self._load_config()   # 内部会更新 self._config_mtime
        self.config = cfg
        if cfg is not None:
            if self.phase in ("IDLE",):
                self.phase = "CONFIGURED"
            self.append(f"[配置] 检测到 {STAND_CONFIG} 已更新，关节角表已按最新配置刷新。")

    # -- 状态快照 --
    def snapshot(self, since):
        with self.lock:
            self._refresh_config_if_changed()
            total = self.dropped + len(self.log)
            start = max(since, self.dropped)
            lines = self.log[start - self.dropped:]
            cfg_joints = {}
            if self.config:
                for j in self.config.get("joints", []):
                    cfg_joints[j["name"]] = j
            rows = []
            for jt in self.joints:
                name = jt["name"]
                cj = cfg_joints.get(name, {})
                fb = self.feedback.get(name, {})
                rows.append({
                    "name": name, "port": jt["port"], "id": jt["id"],
                    "prone": cj.get("prone_rotor", self.prone.get(name)),
                    "stand": cj.get("stand_rotor", self.stand.get(name)),
                    "delta": cj.get("delta_rotor"),
                    "current": self.current.get(name),
                    "ok": self.current_ok.get(name),
                    "dev": self.last_dev.get(name),
                    "verified": cj.get("verified", False),
                    "t_ff": cj.get("t_ff", 0.0),   # 前馈力矩(转子侧 N·m)，站立/保持时补偿重力

                    "fb_vel": fb.get("vel"),     # 实测转子角速度 rad/s
                    "fb_tau": fb.get("tau"),     # 实测力矩 N·m（转子侧）
                    "fb_errd": fb.get("errd"),   # 跟踪误差（关节°）
                    "fb_temp": fb.get("temp"),   # 温度 ℃
                    "tau_peak": self.tau_peak.get(name),   # 峰值 |力矩|（转子侧 N·m）
                })
            holding = False
            for _p in self.servo_procs.values():
                try:
                    if _p.poll() is None:   # 进程还活着 = 电机仍带电锁位保持（未松开）
                        holding = True
                        break
                except Exception:
                    pass
            return {
                "phase": self.phase,
                "status": self.status,
                "busy": self.busy,
                "standing": self.standing_thread is not None
                            and self.standing_thread.is_alive(),
                "configured": self.config is not None,
                "holding": holding,
                "hold_scope": self.hold_scope,
                "notice": self.notice,
                "has_prone": bool(self.prone),
                "has_stand": bool(self.stand),
                "execute": (self.config or {}).get("execute", self.execute),
                "gear_ratio": (self.config or {}).get("gear_ratio", GEAR_RATIO),
                "joints": rows,
                "log": lines,
                "log_index": total,
            }

    # -- 子进程：一次性捕获（读角度） --
    def _run_capture(self, cmd, timeout=8):
        full = (["sudo", "-S", "-p", ""] + cmd) if NEED_SUDO else cmd
        try:
            r = subprocess.run(
                full, input=(SUDO_PASSWORD + "\n") if NEED_SUDO else None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout,
            )
            return r.stdout or ""
        except FileNotFoundError as e:
            self.append(f"[错误] {e}")
            return ""
        except subprocess.TimeoutExpired:
            self.append(f"[超时] {' '.join(cmd)}")
            return ""

    # -- 读取全部关节转子角（按串口并行） --
    def _read_all(self, joints=None):
        if not os.path.isfile(MOTOR_CTRL):
            self.append(f"[错误] 未找到 {MOTOR_CTRL}，请先编译 motor_ctrl")
            return {}, {}
        results = {}
        oks = {}
        groups = joints_by_port(joints if joints is not None else self.joints)
        threads = []

        def worker(port, jts):
            for jt in jts:
                if self.abort.is_set():
                    return
                out = self._run_capture([MOTOR_CTRL, port, str(jt["id"]), "read"])
                rotor, ok = None, False
                for line in out.splitlines():
                    r = parse_angle_line(line)
                    if r and r["id"] == jt["id"]:
                        ok = r["ok"]
                        if ok:
                            rotor = r["rotor"]
                        break
                with self.lock:
                    results[jt["name"]] = rotor
                    oks[jt["name"]] = ok
                tag = f"rotor={rotor:.4f}" if ok and rotor is not None else "无响应"
                self.append(f"  [{jt['name']}] {port} id={jt['id']}: {tag}")

        for port, jts in groups.items():
            t = threading.Thread(target=worker, args=(port, jts), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results, oks

    # -- 标定：读趴姿 / 读站姿 --
    def _calib(self, which):
        self.append("\n" + "=" * 56)
        label = "趴姿" if which == "prone" else "站姿"
        self.append(f"[标定-{label}] 读取全部 12 关节当前转子角 ...")
        self.set_status(f"标定中：读取{label}角度")
        res, oks = self._read_all()
        missing = [n for n, ok in oks.items() if not ok]
        with self.lock:
            self.current = dict(res)
            self.current_ok = dict(oks)
            if which == "prone":
                self.prone = {n: v for n, v in res.items() if oks.get(n)}
            else:
                self.stand = {n: v for n, v in res.items() if oks.get(n)}
            if missing:
                self.append(f"[警告] 以下关节无响应，未记入{label}: {', '.join(missing)}")
            else:
                self.append(f"[完成] {label}标定记录了全部 12 关节。")
            self._refresh_phase()
        self._finish(f"标定{label}")

    def _refresh_phase(self):
        if self.standing_thread is not None and self.standing_thread.is_alive():
            return
        if self.config is not None:
            self.phase = "CONFIGURED"
        elif self.prone and self.stand:
            self.phase = "STAND_DONE"
        elif self.prone:
            self.phase = "PRONE_DONE"
        else:
            self.phase = "IDLE"

    # -- 标定：计算 delta 并保存配置 --
    def _save_config(self):
        with self.lock:
            prone, stand = dict(self.prone), dict(self.stand)
        names = [jt["name"] for jt in self.joints]
        miss = [n for n in names if n not in prone or n not in stand]
        if miss:
            return False, f"以下关节缺少趴姿或站姿标定，无法保存: {', '.join(miss)}"

        # 重新标定时保留已调好的前馈力矩（按关节名继承旧配置）
        old_tff = {}
        if self.config:
            for j in self.config.get("joints", []):
                if j.get("t_ff") is not None:
                    old_tff[j.get("name")] = j["t_ff"]

        joints_out = []
        for jt in self.joints:
            n = jt["name"]
            p, s = prone[n], stand[n]
            joints_out.append({
                "name": n, "port": jt["port"], "id": jt["id"],
                "prone_rotor": round(p, 6),
                "stand_rotor": round(s, 6),
                "delta_rotor": round(s - p, 6),
                "t_ff": old_tff.get(n, 0.0),
            })
        cfg = {
            "robot": "mos2026_2",
            "gear_ratio": GEAR_RATIO,
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "encoder_note": "单圈绝对值编码器，绝对位置断电不变，配置跨上电有效；机械/零点变动后才需重标",
            "joints": joints_out,
            "execute": self.execute,
        }
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(STAND_CONFIG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        with self.lock:
            self.config = cfg
            try:
                self._config_mtime = os.path.getmtime(STAND_CONFIG)
            except OSError:
                self._config_mtime = None
            self.phase = "CONFIGURED"
        self.append(f"\n[保存] 配置已写入 {STAND_CONFIG}")
        for j in joints_out:
            arrow = "↑" if j["delta_rotor"] >= 0 else "↓"
            pj = math.degrees(j["prone_rotor"] / GEAR_RATIO)
            sj = math.degrees(j["stand_rotor"] / GEAR_RATIO)
            dj = math.degrees(j["delta_rotor"] / GEAR_RATIO)
            self.append(f"  {j['name']}: 趴={pj:.2f}° 站={sj:.2f}° Δ={dj:+.2f}° {arrow}")
        return True, "配置已保存"

    # -- 重锚趴姿：prone=当前读数，delta 不变，stand=prone+delta --
    # 适用场景：编码器零点相对关节平移了（换电机、摇臂错齿等），趴→站行程本身没变。
    # 若连杆几何变过（delta 失效），必须走完整标定（①趴 ②站 ③保存）。
    def _reanchor_prone(self):
        try:
            self._reanchor_impl()
        finally:
            self._finish("重锚趴姿")

    def _reanchor_impl(self):
        self.append("\n" + "=" * 56)
        self.append("[重锚] 趴姿重锚：prone=当前读数，delta 保持，stand=prone+delta")
        cfg = self._load_config()
        if not cfg or not cfg.get("joints"):
            self.append(f"[拒绝] 没有可用的 {STAND_CONFIG}；重锚需要已有 delta，请先走完整标定。")
            return
        gear = cfg.get("gear_ratio", GEAR_RATIO)
        names = [jt["name"] for jt in self.joints]
        cj = {j.get("name"): j for j in cfg["joints"]}
        lack = [n for n in names if n not in cj or cj[n].get("delta_rotor") is None]
        if lack:
            self.append(f"[拒绝] 以下关节缺 delta_rotor，无法重锚: {', '.join(lack)}")
            return

        self.set_status("重锚：第 1 次读取")
        r1, ok1 = self._read_all()
        if self.abort.is_set():
            return
        self.append(f"[重锚] 等待 {REANCHOR_SETTLE:.0f}s 后复读，校验姿势静止 ...")
        self.set_status("重锚：等待姿势静止")
        if self.abort.wait(REANCHOR_SETTLE):
            self.append("[重锚] 收到急停/中止。")
            return
        self.set_status("重锚：第 2 次读取")
        r2, ok2 = self._read_all()

        dead = [n for n in names if not (ok1.get(n) and ok2.get(n))]
        if dead:
            self.append(f"[拒绝] 以下关节无响应: {', '.join(dead)}；请检查电机模式/接线后重试。")
            return
        moving = []
        for n in names:
            d = abs(r2[n] - r1[n]) / gear * 180.0 / math.pi
            if d > REANCHOR_DRIFT_MAX:
                moving.append(f"{n}({d:.2f}°/{REANCHOR_SETTLE:.0f}s)")
        if moving:
            self.append(f"[拒绝] 以下关节仍在动（重力下垂/被触碰）: {', '.join(moving)}")
            self.append("       请让关节完全靠住限位/支撑面、手离开后重试。")
            return

        bak = STAND_CONFIG.replace(".json", time.strftime(".bak.%Y%m%d_%H%M%S.json"))
        shutil.copyfile(STAND_CONFIG, bak)
        self.append(f"[重锚] 旧配置已备份: {os.path.basename(bak)}")
        two_pi = 2 * math.pi
        for j in cfg["joints"]:
            n = j["name"]
            old_p = j.get("prone_rotor")
            j["prone_rotor"] = round(r2[n], 6)
            j["stand_rotor"] = round(r2[n] + j["delta_rotor"], 6)
            if old_p is not None:
                raw = r2[n] - old_p
                dev = raw - round(raw / two_pi) * two_pi   # 整圈归一后的锚点移动量
                self.append(f"  [{n}] 锚点移动 {math.degrees(dev / gear):+.2f}°(关节)")
        cfg["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        cfg["_recalibrated_at"] = time.strftime("%Y-%m-%d %H:%M") + \
            " 网页重锚: prone=当前确认趴姿, delta不变, stand=prone+delta"
        with open(STAND_CONFIG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        with self.lock:
            self.config = cfg
            try:
                self._config_mtime = os.path.getmtime(STAND_CONFIG)
            except OSError:
                self._config_mtime = None
            self._refresh_phase()
        self.append(f"[重锚] 完成，已写入 {STAND_CONFIG}。")
        self.append("       注意：deploy/real/config/mos2026_2.yaml 的 hardware.stand_rotor 需同步更新。")

    # -- 单关节方向：confirm 确认 / unconfirm 取消确认 / invert 取反(翻转 delta/dir 并存盘) --
    def _set_dir(self, name, mode):
        cfg = self._load_config()
        if not cfg:
            return False, "没有配置文件，请先标定保存"
        target = next((j for j in cfg["joints"] if j.get("name") == name), None)
        if target is None:
            return False, f"配置里没有关节 {name}"

        if mode == "invert":
            dr = target.get("delta_rotor")
            if dr is None:
                return False, f"{name} 缺 delta_rotor，无法取反"
            target["delta_rotor"] = round(-dr, 6)
            pr = target.get("prone_rotor")
            if pr is not None:
                target["stand_rotor"] = round(pr + target["delta_rotor"], 6)
            if target.get("dir") in (1, -1):
                target["dir"] = -target["dir"]
            # 同步 motor_web 的关节角字段（站姿关于趴姿镜像）
            pra, sta = target.get("prone_rad"), target.get("stand_rad")
            if pra is not None and sta is not None:
                target["stand_rad"] = round(2 * pra - sta, 6)
            prd, std = target.get("prone_deg"), target.get("stand_deg")
            if prd is not None and std is not None:
                target["stand_deg"] = round(2 * prd - std, 3)
            target["verified"] = False
            msg = f"{name} 方向已取反并存盘（请重新做一次方向验证确认）"
        elif mode == "confirm":
            target["verified"] = True
            msg = f"{name} 方向已确认"
        elif mode == "unconfirm":
            target["verified"] = False
            msg = f"{name} 已取消确认"
        else:
            return False, f"未知方向操作: {mode}"

        try:
            with open(STAND_CONFIG, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except OSError as e:
            return False, f"保存失败: {e}"
        with self.lock:
            self.config = cfg
            try:
                self._config_mtime = os.path.getmtime(STAND_CONFIG)
            except OSError:
                self._config_mtime = None
        self.append("[标定] " + msg)
        return True, msg

    # -- 手动修改单关节某个姿态(趴/站)的目标角（页面直接改一个角度，写回配置并重算 Δ）--
    # deg 为关节角(度)，与表格「趴姿/站姿(°)」同坐标 = rotor / gear * 180/π。
    def _set_pose_angle(self, name, pose, deg):
        cfg = self._load_config()
        if not cfg:
            return False, "没有配置文件，请先标定保存"
        target = next((j for j in cfg["joints"] if j.get("name") == name), None)
        if target is None:
            return False, f"配置里没有关节 {name}"
        if pose not in ("prone", "stand"):
            return False, f"未知姿态: {pose}（应为 prone/stand）"
        gear = float(cfg.get("gear_ratio", GEAR_RATIO))
        deg = max(-360.0, min(360.0, float(deg)))
        joint_rad = math.radians(deg)            # 关节角 rad
        rotor = round(joint_rad * gear, 6)       # 转子角 rad（控制/校验用）
        label = "趴姿" if pose == "prone" else "站姿"

        old_rotor = target.get(f"{pose}_rotor")
        target[f"{pose}_rotor"] = rotor
        target[f"{pose}_rad"] = round(joint_rad, 6)
        target[f"{pose}_deg"] = round(deg, 3)

        # 重算 Δ(站−趴) 与方向；姿态变了，原方向验证不再可信
        pr, st = target.get("prone_rotor"), target.get("stand_rotor")
        if pr is not None and st is not None:
            target["delta_rotor"] = round(st - pr, 6)
            if target.get("dir") in (1, -1):
                dr = target["delta_rotor"]
                if dr != 0:
                    target["dir"] = 1 if dr > 0 else -1
        target["verified"] = False

        try:
            with open(STAND_CONFIG, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except OSError as e:
            return False, f"保存失败: {e}"
        with self.lock:
            self.config = cfg
            try:
                self._config_mtime = os.path.getmtime(STAND_CONFIG)
            except OSError:
                self._config_mtime = None
            if self.phase == "IDLE":
                self.phase = "CONFIGURED"
        old_txt = (f"{math.degrees(old_rotor / gear):.2f}° -> "
                   if old_rotor is not None else "")
        dj = (math.degrees(target["delta_rotor"] / gear)
              if target.get("delta_rotor") is not None else None)
        self.append(f"[手动修改] {name} {label}: {old_txt}{deg:.2f}°"
                    f"（转子 {rotor:.4f} rad）"
                    + (f"；Δ(站−趴)重算为 {dj:+.2f}°" if dj is not None else "")
                    + "，已清除该关节方向验证标记。")
        return True, f"{name} {label}已改为 {deg:.2f}°"

    # -- 手动设置单关节前馈力矩 t_ff（转子侧 N·m，写回配置；站立/保持时补偿重力减小下沉）--
    def _set_feedforward(self, name, tff):
        cfg = self._load_config()
        if not cfg:
            return False, "没有配置文件，请先标定保存"
        target = next((j for j in cfg["joints"] if j.get("name") == name), None)
        if target is None:
            return False, f"配置里没有关节 {name}"
        tff = max(-8.0, min(8.0, float(tff)))   # 限幅 ±8 N·m，防误填大值蹬腿
        old = target.get("t_ff", 0.0)
        target["t_ff"] = round(tff, 4)
        try:
            with open(STAND_CONFIG, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except OSError as e:
            return False, f"保存失败: {e}"
        with self.lock:
            self.config = cfg
            try:
                self._config_mtime = os.path.getmtime(STAND_CONFIG)
            except OSError:
                self._config_mtime = None
        self.append(f"[前馈力矩] {name}: {old:+.3f} -> {tff:+.3f} N·m(转子侧)。"
                    "站立时按抬起进度 0→该值 施加，保持阶段维持该值。")
        return True, f"{name} 前馈力矩已设为 {tff:+.3f} N·m"

    # -- 流式伺服进程 --
    def _spawn_servo(self, port, kp, kw):
        cmd = [MOTOR_CTRL, port, "servo", f"{kp}", f"{kw}"]
        full = (["sudo", "-S", "-p", ""] + cmd) if NEED_SUDO else cmd
        proc = subprocess.Popen(
            full, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        if NEED_SUDO:
            try:
                proc.stdin.write(SUDO_PASSWORD + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

        def reader(p, tag):
            for line in p.stdout:
                s = line.rstrip()
                if s.startswith("FB "):
                    self._update_feedback(s)   # 实时反馈行喂给表格，不进日志
                else:
                    self.append(f"[{tag}] {s}")

        threading.Thread(target=reader, args=(proc, os.path.basename(port)),
                         daemon=True).start()
        return proc

    # 解析 servo 子进程的 "FB id=.. pos=.. vel=.. tau=.. temp=.. merr=.. ok=.. errd=.." 行，
    # 存成每关节实时反馈供前端表格显示；同时用 pos 实时刷新「当前角」。
    def _update_feedback(self, line):
        def g(key):
            m = re.search(rf"\b{key}=(-?\d+(?:\.\d+)?)", line)
            return m.group(1) if m else None
        sid = g("id")
        if sid is None:
            return
        try:
            mid = int(float(sid))
        except ValueError:
            return
        name = self._id2name.get(mid)
        if not name:
            return

        def fnum(key):
            v = g(key)
            return float(v) if v is not None else None

        ok = g("ok") == "1"
        fb = {"vel": fnum("vel"), "tau": fnum("tau"), "errd": fnum("errd"),
              "temp": int(float(g("temp"))) if g("temp") is not None else None,
              "merr": int(float(g("merr"))) if g("merr") is not None else None,
              "ok": ok}
        pos = fnum("pos")
        tau = fb.get("tau")
        with self.lock:
            self.feedback[name] = fb
            if tau is not None:                # 峰值 |力矩| 保持（取最大绝对值）
                a = abs(tau)
                if a > self.tau_peak.get(name, 0.0):
                    self.tau_peak[name] = a
            if pos is not None:                # 保持/运动时也实时刷新当前角
                self.current[name] = pos
                self.current_ok[name] = ok

    def _stop_servos(self, brake=True):
        with self.lock:
            procs = dict(self.servo_procs)
            self.servo_procs = {}
            self.hold_scope = ""      # 电机即将释放，清除「保持中」标记
            self.feedback = {}        # 松开后不再有实时反馈，清空表格反馈列
            self.hold_targets = {}    # 松开后没有保持目标，禁止微调
            self.hold_tff = {}
        for port, proc in procs.items():
            try:
                if brake and proc.poll() is None:
                    proc.stdin.write("stop\n")
                    proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        time.sleep(0.4)  # 给 servo 进程发完 mode=0 停止脉冲的时间
        for port, proc in procs.items():
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    def _stop_stale_servos(self, reason=""):
        """开始新动作前调用：若上一动作的 servo 进程仍在运行（电机还锁位保持），
        先干净地停掉它们再继续，避免新旧进程在同一总线上双重驱动同一批电机
        （会导致反馈乱跳、力矩异常）。无运行进程时直接返回，不引入额外延时。"""
        with self.lock:
            running = any(p.poll() is None for p in self.servo_procs.values())
        if running:
            self.append(f"[清理] 检测到仍在运行的旧 servo（{reason}），先停掉再开始新动作，避免双重驱动。")
            self._stop_servos()

    # -- 站立执行（安全校验 + 限速插值 + 流式伺服） --
    def _do_stand(self, legs=None, kp=None, kw=None, skip_verify=False, ex_override=None, descend=False):
        verb = "坐下" if descend else "站立"
        try:
            self._stand_impl(legs, kp, kw, skip_verify=skip_verify, ex_override=ex_override, descend=descend)
        except Exception as e:
            self.append(f"[异常] {verb}执行出错: {e}")
            self.set_status(f"{verb}异常: {e}")
            self._notify(f"{verb}执行出错：{e}")
            self._stop_servos()
            with self.lock:
                self.phase = "ERROR"

    def _stand_impl(self, legs=None, kp_override=None, kw_override=None, skip_verify=False, ex_override=None, descend=False):
        # descend=True：坐下（站立的反向）——目标 = 当前 − Δ（回到趴姿），分段顺序反过来，
        # 前馈随下放进度 满→0，并以「站姿」而非「趴姿」做基准。
        cfg = self._load_config()
        if not cfg:
            self.append("[错误] 没有配置文件，请先完成标定并保存。")
            self._notify("没有配置文件，无法执行：请先在标定页完成 ①②③ 并保存。")
            with self.lock:
                self.phase = "IDLE"
            return
        move_word = "下放" if descend else "抬升"
        # legs=None 表示整机 12 关节；否则只取这些腿（前缀 fl/fr/rl/rr）
        if legs:
            active = [jt for jt in self.joints
                      if jt["name"].split("_")[0] in legs]
            scope = f"单腿{'坐下' if descend else '测试'} [" + ", ".join(sorted(legs)) + "]"
        else:
            active = list(self.joints)
            scope = "整机坐下" if descend else "整机站立"
        if not active:
            self.append(f"[错误] 没有匹配的关节: {legs}")
            self._notify(f"没有匹配的关节：{legs}")
            with self.lock:
                self.phase = "CONFIGURED"
            return
        ex = cfg.get("execute", self.execute)
        exo = ex_override or {}   # 页面对整机站立的临时覆盖（不写回配置）
        thr = float(exo.get("verify_threshold_rad", ex.get("verify_threshold_rad", 0.20)))
        vmax_joint = float(exo.get("max_joint_vel_rad_s", ex.get("max_joint_vel_rad_s", 0.1)))
        min_dur = float(exo.get("min_duration_s", ex.get("min_duration_s", 1.0)))
        rate = float(exo.get("rate_hz", ex.get("rate_hz", 100)))
        # 站立方式 style：staged=分段抬升(小腿→大腿→hip，降峰值电流) / sync=整机同时线性 /
        # natural=自然站立（全身同步 + 最小加加速度 S 曲线，起停缓入缓出，更接近动物起身）。
        # 未传 style 时退回旧的 staged 布尔开关（兼容旧配置/旧页面）。
        style = str(exo.get("style", ex.get("style", "")) or "").strip().lower()
        if style not in ("staged", "sync", "natural"):
            style = "staged" if bool(exo.get("staged", ex.get("staged", True))) else "sync"
        staged = style == "staged"
        natural = style == "natural"
        kp = float(ex.get("k_p", 8.0)) if kp_override is None else float(kp_override)
        kw = float(ex.get("k_w", 0.1)) if kw_override is None else float(kw_override)
        # 整机统一基准前馈（页面留空则用配置 execute.t_ff，默认 0）；与每关节 t_ff 叠加
        tff_base = (float(exo["t_ff"]) if exo.get("t_ff") is not None
                    else float(ex.get("t_ff", 0.0) or 0.0))
        if kp_override is not None or kw_override is not None:
            self.append(f"[{scope}] 使用页面指定增益 K_P={kp} K_W={kw}（覆盖配置默认）")
        if exo:
            self.append(f"[{scope}] 使用页面指定参数覆盖配置默认："
                        + "，".join(f"{k}={v}" for k, v in exo.items()))

        self._stop_stale_servos(f"开始{scope}前")   # 先停掉上一动作仍在保持的 servo，避免双重驱动/读数冲突
        self.append("\n" + "=" * 56)
        self.append(f"[{scope}] 读取当前角度做安全校验 ...")
        self.set_status(f"{scope}：安全校验中")
        with self.lock:
            self.phase = "EXECUTING"
        res, oks = self._read_all(active)
        with self.lock:
            self.current = dict(res)
            self.current_ok = dict(oks)

        # 站立以「趴姿」为基准、坐下以「站姿」为基准；坐下是向下+重力辅助+按相对量(cur−Δ)走，
        # 落点最差只是离趴姿差一点、不危险，故坐下默认不因姿势偏差拒绝（硬校验仍生效）。
        base_key = "stand_rotor" if descend else "prone_rotor"
        pose_word = "站姿" if descend else "趴姿"
        verify_pose = not (skip_verify or descend)
        cfg_joints = {j["name"]: j for j in cfg["joints"]}
        cur, prone, delta, tff_t = {}, {}, {}, {}
        hard_bad = []   # 配置缺失 / 无响应：无法执行，「关闭安全校验」也不能绕过
        dev_bad = []    # 姿势偏差超阈值：安全校验项，可被「关闭安全校验」/坐下跳过
        with self.lock:
            self.last_dev = {}
        for jt in active:
            n = jt["name"]
            cj = cfg_joints.get(n)
            if cj is None:
                hard_bad.append(f"{n}(配置缺失)")
                continue
            if not oks.get(n) or res.get(n) is None:
                hard_bad.append(f"{n}(无响应)")
                continue
            # 单圈绝对编码器：圈数掉电清零，同一物理位置的读数可能差整数个 2π(转子，≈56.9°关节)。
            # 站立按相对量(当前+Δ)走、不受影响；偏差按整圈归一到 ±π 再比阈值，避免被误判拒绝。
            two_pi = 2 * math.pi
            raw_dev = res[n] - cj[base_key]
            turns = round(raw_dev / two_pi)
            dev = abs(raw_dev - turns * two_pi)
            with self.lock:
                self.last_dev[n] = round(dev, 4)
            if dev > thr:
                dev_bad.append(f"{n}(偏差 {dev:.3f} > {thr})")
            elif turns != 0:
                self.append(f"  [{n}] 读数比{pose_word}差 {turns:+d} 个转子整圈"
                            f"(≈{turns*360.0/GEAR_RATIO:+.0f}°关节)，已按整圈归一校验（单圈编码器掉电圈数清零所致，正常）。")
            cur[n] = res[n]
            prone[n] = cj["prone_rotor"]
            delta[n] = cj["delta_rotor"]
            tff_t[n] = tff_base + float(cj.get("t_ff", 0.0) or 0.0)

        if hard_bad:
            msg = "无法执行（缺配置 / 无响应）：" + "，".join(hard_bad)
            self.append(f"[拒绝] {msg}")
            self.append("       请确认电机处于电机模式、接线正常，再重试。")
            self.set_status("执行被拒绝（缺配置/无响应）")
            self._notify("⛔ 无法执行：\n\n" + "，".join(hard_bad)
                         + "\n\n请检查电机模式 / 接线后重试。")
            with self.lock:
                self.phase = "CONFIGURED"
            return

        if dev_bad and verify_pose:
            msg = "安全校验未通过，拒绝执行：" + "，".join(dev_bad)
            self.append(f"[拒绝] {msg}")
            self.append(f"       请确认机器人已自然{pose_word[0]}好、电机处于电机模式，再重试。")
            self.append("       （如确需在非标定姿态下测试，可勾选「关闭安全校验」后再点「只测这条腿」。）")
            self.set_status("执行被拒绝（安全校验未通过）")
            self._notify("⛔ 安全校验未通过，已拒绝执行：\n\n" + "，".join(dev_bad)
                         + f"\n\n请确认这条腿各关节贴近标定{pose_word}、电机处于电机模式后重试；"
                           "若只想验证电机方向，可改用「方向验证」（不要求姿态）。")
            with self.lock:
                self.phase = "CONFIGURED"
            return

        sgn = -1.0 if descend else 1.0   # 坐下 = 当前 − Δ 回趴姿；站立 = 当前 + Δ 到站姿
        if dev_bad and descend:
            self.append(f"[坐下] 与标定站姿有偏差（{'，'.join(dev_bad)}），仍按『当前角 − Δ』向下回趴姿；"
                        "坐下是重力辅助、按相对量移动，落点最差只是离趴姿差一点，正常。")
            self.set_status(f"{scope}：执行中（下放）")
        elif dev_bad and skip_verify:
            self.append("[⚠️ 安全校验已关闭] 跳过姿势偏差检查，强行按『当前角 + Δ』移动："
                        + "，".join(dev_bad))
            self.append("       危险：若该腿离趴姿较远会大幅运动！请盯住急停。")
            self.set_status(f"{scope}：⚠️ 已关闭安全校验，执行中")
        else:
            self.append(f"[通过] {len(active)} 个关节贴近配置{pose_word}（阈值 {thr} rad）。")

        # 目标 = 当前 + sgn·delta（站立 +Δ 到站姿 / 坐下 −Δ 回趴姿）
        targets = {n: cur[n] + sgn * delta[n] for n in cur}
        vmax_rotor = vmax_joint * GEAR_RATIO

        # 分段：站立按 STAGE_ORDER(小腿→大腿→hip) 抬升；坐下按反序(hip→大腿→小腿)下放。
        # 每段单独限速插值，上一段到位并保持后再启动下一段，降低同时驱动的峰值力矩/电流。
        # staged=False 时退化为「全关节同时插值」（一段）。
        stage_seq = tuple(reversed(STAGE_ORDER)) if descend else STAGE_ORDER
        if staged:
            stages = [(jt, grp) for jt in stage_seq
                      if (grp := [n for n in cur if n.split("_")[1] == jt])]
            other = [n for n in cur if n.split("_")[1] not in STAGE_ORDER]
            if other:
                stages.append(("other", other))
        else:
            stages = [("all", list(cur.keys()))]

        # 每段时长：本段位移最大的关节恰好以 vmax 运动（不足 min_dur 则取 min_dur）。
        # 自然站立走 S 曲线，峰值速度是均速的 15/8=1.875 倍，按此放大时长保证瞬时不超限速。
        vel_scale = 1.875 if natural else 1.0
        stage_plan = []
        for jt, grp in stages:
            smax = max((abs(targets[n] - cur[n]) for n in grp), default=0.0)
            sdur = max(min_dur, vel_scale * smax / vmax_rotor if vmax_rotor > 0 else min_dur)
            stage_plan.append((jt, grp, smax, sdur, max(1, int(sdur * rate))))
        total_dur = sum(s[3] for s in stage_plan)

        nz_tff = {n: v for n, v in tff_t.items() if abs(v) > 1e-9}
        ramp_note = "（按下放进度 满→0 施加）" if descend else "（按抬起进度 0→满 施加）"
        tff_note = ("，前馈力矩(转子N·m): " + "，".join(f"{n}={v:+.2f}" for n, v in nz_tff.items())
                    + ramp_note) if nz_tff else "，无前馈力矩"
        if staged:
            seq = " → ".join(f"{jt}({len(grp)})" for jt, grp, *_ in stage_plan)
            self.append(f"[轨迹] {scope}：分段{move_word} {seq}，限速 {vmax_joint} rad/s(关节) "
                        f"=> 总时长 {total_dur:.2f}s, K_P={kp} K_W={kw}{tff_note}")
        else:
            max_move = max((abs(targets[n] - cur[n]) for n in cur), default=0.0)
            way = ("自然站立（全身同步 + S曲线缓入缓出，峰值速度≤限速）" if natural
                   else "整机同时（线性匀速）")
            self.append(f"[轨迹] {scope}：{way}，最大位移 {max_move:.3f} rad(转子)，限速 {vmax_joint} rad/s(关节) "
                        f"=> 时长 {total_dur:.2f}s, K_P={kp} K_W={kw}{tff_note}")
        self.set_status(f"{scope}执行中：缓慢移动（约 {total_dur:.1f}s）")

        # 每路总线开一个 servo 进程（只开 active 涉及的总线）
        groups = joints_by_port(active)
        with self.lock:
            self.servo_procs = {}
            self.tau_peak = {}   # 本次站立重新统计峰值力矩
        for port in groups:
            with self.lock:
                self.servo_procs[port] = self._spawn_servo(port, kp, kw)
        time.sleep(0.2)  # 等 servo 进程起来（含 sudo 鉴权）
        with self.lock:
            self.hold_scope = scope   # 标记电机已锁位保持，供「是否松开」显示

        dt = 1.0 / rate
        done = set()   # 已完成抬升、保持在目标位的关节集合
        for jt_name, grp, smax, sdur, Ns in stage_plan:
            grpset = set(grp)
            if staged:
                self.append(f"[分段] {move_word} {jt_name}（{len(grp)} 关节，最大位移 {smax:.3f} rad，约 {sdur:.2f}s），其余关节保持。")
                self.set_status(f"{scope}：{move_word} {jt_name} 中（约 {sdur:.1f}s）")
            for k in range(1, Ns + 1):
                if self.abort.is_set():
                    self.append("[中止] 收到急停/中止，停止插值。")
                    self._stop_servos()
                    with self.lock:
                        self.phase = "CONFIGURED"
                    self.set_status(f"{'坐下' if descend else '站立'}已中止")
                    return
                a = k / Ns  # 本段时间进度 0->1
                # 自然站立用最小加加速度(min-jerk) S 曲线 s=10a³−15a⁴+6a⁵：
                # 起步/收尾的速度和加速度都为 0，先缓→快→缓，无顿挫，起身更像动物。
                s = a * a * a * (10.0 + a * (6.0 * a - 15.0)) if natural else a
                for port, jts in groups.items():
                    parts = []
                    for j in jts:
                        n = j["name"]
                        if n not in cur:
                            continue
                        # 站立：未轮到=趴位无前馈 / 在动=0→目标、前馈 0→满 / 完成=目标位+满前馈
                        # 坐下：未轮到=仍站位+满前馈(还撑着) / 在动=站→趴、前馈 满→0 / 完成=趴位无前馈
                        if n in grpset:                 # 本段正在移动
                            pos = cur[n] + s * (targets[n] - cur[n])
                            tff = ((1.0 - s) if descend else s) * tff_t.get(n, 0.0)
                        elif n in done:                 # 本段已完成
                            pos = targets[n]
                            tff = 0.0 if descend else tff_t.get(n, 0.0)
                        else:                           # 还没轮到
                            pos = cur[n]
                            tff = tff_t.get(n, 0.0) if descend else 0.0
                        parts.append(f"{j['id']} {pos:.5f} {tff:.4f}")
                    proc = self.servo_procs.get(port)
                    if proc and proc.poll() is None and parts:
                        try:
                            proc.stdin.write(" ".join(parts) + "\n")
                            proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            self.append(f"[警告] servo {port} 管道已断开")
                time.sleep(dt)
            done.update(grpset)

        # -- 闭环到位判定：插值发完 ≠ 电机真到位。等保持稳定后读实测跟踪误差(errd, 关节°)，
        #    超容差就明确报「未到位」，避免「显示到位、实则力矩/电流不足没顶上去」的误导。--
        reach_tol = float(exo.get("reach_tol_deg", ex.get("reach_tol_deg", 3.0)))
        time.sleep(0.6)  # 等 PD 稳态 + 至少刷新一帧新反馈
        with self.lock:
            fb = {jt["name"]: dict(self.feedback.get(jt["name"], {})) for jt in active}
        over, miss, errs = [], [], []
        for jt in active:
            e = fb[jt["name"]].get("errd")
            if e is None:
                miss.append(jt["name"])
                continue
            errs.append(abs(e))
            if abs(e) > reach_tol:
                over.append((jt["name"], e))
        max_err = max(errs, default=None)

        if over:
            over.sort(key=lambda x: -abs(x[1]))
            detail = "，".join(f"{n} 差{e:+.1f}°" for n, e in over[:6])
            more = f" 等{len(over)}个" if len(over) > 6 else ""
            self.append(f"[未到位] {scope} 指令已发完并锁位保持，但 {len(over)} 个关节实测未到位"
                        f"（容差 {reach_tol:.1f}°）：{detail}{more}。")
            self.append("        多半是力矩/电流不足顶不上去：看扭矩监控 fb_tau 是否已饱和；"
                        "可加重力前馈 t_ff / 提 kp / 上调驱动器电流上限。")
            self.set_status(f"{scope} 保持中（⚠️ {len(over)} 关节未到位，最大 {max_err:.1f}°）")
            self._notify(f"⚠️ {scope} 显示保持但实测未到位：{len(over)} 个关节超 {reach_tol:.1f}°，"
                         f"最大 {max_err:.1f}°。\n\n{detail}{more}\n\n疑似力矩/电流不足，请看扭矩监控。")
        else:
            tail = (f"（最大跟踪误差 {max_err:.1f}° < 容差 {reach_tol:.1f}°）"
                    if max_err is not None else "（⚠️ 无反馈数据，未能核对实测误差）")
            self.append(f"[完成] {scope}到位并保持{tail}，servo 进程持续保持位置（点「急停/松开」释放）。")
            self.set_status(f"{scope}到位（保持中）")
        if miss:
            self.append(f"        注意：{len(miss)} 个关节无反馈(errd)，未纳入核对：{', '.join(miss)}。")
        if legs and not descend:
            self.append("       请检查这条腿：是否朝『站起来』方向收拢？哪个关节方向不对就反它的标定。")
        with self.lock:
            self.hold_targets = dict(targets)   # 记录保持目标，供「单关节微调」做增量基准
            # 坐下后落在趴姿、重量在地面，保持前馈归零；站立保持满前馈顶住重力。
            self.hold_tff = {n: 0.0 for n in targets} if descend else dict(tff_t)
            self.phase = "PRONE_HOLD" if descend else "STANDING"

    # -- 方向验证：单腿，每个关节从当前位置朝「站立方向」各转固定角度（默认 10°）--
    # 与站立不同：不要求处于趴姿（不做 prone 偏差校验），只做小幅相对位移验证方向符号。
    def _do_dir_test(self, leg, deg=10.0, joint=None, kp=None, kw=None):
        try:
            self._dir_test_impl(leg, deg, joint, kp, kw)
        except Exception as e:
            self.append(f"[异常] 方向验证出错: {e}")
            self.set_status(f"方向验证异常: {e}")
            self._notify(f"方向验证出错：{e}")
            self._stop_servos()
            with self.lock:
                self.phase = "ERROR"

    def _dir_test_impl(self, leg, deg=10.0, joint=None, kp_override=None, kw_override=None):
        cfg = self._load_config()
        if not cfg:
            self.append("[错误] 没有配置文件，请先完成标定并保存。")
            self._notify("没有配置文件，无法验证：请先在标定页完成 ①②③ 并保存。")
            with self.lock:
                self.phase = "IDLE"
            return
        ex = cfg.get("execute", self.execute)
        vmax_joint = float(ex.get("max_joint_vel_rad_s", 0.05))
        min_dur = float(ex.get("min_duration_s", 2.0))
        rate = float(ex.get("rate_hz", 100))
        kp = float(ex.get("k_p", 2.0)) if kp_override is None else float(kp_override)
        kw = float(ex.get("k_w", 0.1)) if kw_override is None else float(kw_override)
        if kp_override is not None or kw_override is not None:
            self.append(f"[方向验证] 使用页面指定增益 K_P={kp} K_W={kw}（覆盖配置默认）")

        if joint in ("hip", "thigh", "shank"):
            active = [jt for jt in self.joints if jt["name"] == f"{leg}_{joint}"]
            scope = f"方向验证 [{leg}_{joint}] 单电机 +{deg:.0f}°"
        else:
            active = [jt for jt in self.joints if jt["name"].split("_")[0] == leg]
            scope = f"方向验证 [{leg}] 各关节 +{deg:.0f}°"
        if not active:
            self.append(f"[错误] 没有匹配的关节: {leg}_{joint or '*'}")
            self._notify(f"没有匹配的关节：{leg}_{joint or '*'}")
            with self.lock:
                self.phase = "CONFIGURED"
            return
        port = active[0]["port"]
        cfg_joints = {j["name"]: j for j in cfg["joints"]}

        self._stop_stale_servos(f"开始{scope}前")   # 先停掉上一动作仍在保持的 servo，避免双重驱动/读数冲突
        self.append("\n" + "=" * 56)
        self.append(f"[{scope}] 读取当前角度（不要求趴姿）...")
        self.set_status(f"{scope}：读取中")
        with self.lock:
            self.phase = "EXECUTING"
        res, oks = self._read_all(active)
        with self.lock:
            self.current = dict(res)
            self.current_ok = dict(oks)

        step_rotor = math.radians(deg) * GEAR_RATIO   # 关节角 deg -> 转子 rad
        cur, targets, plan, bad = {}, {}, [], []
        for jt in active:
            n = jt["name"]
            if not oks.get(n) or res.get(n) is None:
                bad.append(f"{n}(无响应)")
                continue
            cj = cfg_joints.get(n, {})
            dr = cj.get("delta_rotor")
            d = cj.get("dir")
            sign = d if d in (1, -1) else (1 if (dr or 0) > 0 else (-1 if (dr or 0) < 0 else 0))
            if sign == 0:
                self.append(f"  [{n}] 配置方向为 0/缺失，跳过（趴↔站几乎不动）")
                continue
            cur[n] = res[n]
            targets[n] = res[n] + sign * step_rotor
            plan.append((n, jt["id"], sign))

        if bad:
            self.append(f"[拒绝] 有关节无响应：{'，'.join(bad)}；检查接线/电机模式后重试。")
            self.set_status("方向验证被拒绝（有关节无响应）")
            self._notify("⛔ 方向验证被拒绝：有关节无响应：\n\n" + "，".join(bad)
                         + "\n\n请检查该腿接线、供电，并确认电机已进入电机模式后重试。")
            with self.lock:
                self.phase = "CONFIGURED"
            return
        if not targets:
            self.append("[结束] 没有可验证的关节（方向都为 0）。")
            self._notify("没有可验证的关节：所选关节在配置里方向都为 0（趴↔站几乎不动），无法验证方向。",
                         level="warn")
            with self.lock:
                self.phase = "CONFIGURED"
            return

        for n, i, sign in plan:
            self.append(f"  {n}: 朝站立方向 {'＋' if sign > 0 else '－'} 转 {deg:.0f}°(关节)")

        vmax_rotor = vmax_joint * GEAR_RATIO
        duration = max(min_dur, step_rotor / vmax_rotor if vmax_rotor > 0 else min_dur)
        N = max(1, int(duration * rate))
        self.append(f"[轨迹] {scope}：每关节 {step_rotor:.3f} rad(转子) => 时长 {duration:.2f}s, "
                    f"{N} 步, K_P={kp} K_W={kw}")
        self.set_status(f"{scope}执行中（约 {duration:.1f}s）")

        servo = self._spawn_servo(port, kp, kw)
        with self.lock:
            self.servo_procs = {port: servo}
            self.hold_scope = scope   # 标记电机已锁位保持，供「是否松开」显示
            self.tau_peak = {}        # 本次方向验证重新统计峰值力矩
        time.sleep(0.2)

        dt = 1.0 / rate
        for k in range(1, N + 1):
            if self.abort.is_set():
                self.append("[中止] 收到急停/中止。")
                self._stop_servos()
                with self.lock:
                    self.phase = "CONFIGURED"
                self.set_status("方向验证已中止")
                return
            a = k / N
            parts = []
            for n, i, sign in plan:
                pos = cur[n] + a * (targets[n] - cur[n])
                parts.append(f"{i} {pos:.5f} 0.0000")   # 方向验证不加前馈，保持判断干净
            if servo.poll() is None and parts:
                try:
                    servo.stdin.write(" ".join(parts) + "\n")
                    servo.stdin.flush()
                except (BrokenPipeError, OSError):
                    self.append(f"[警告] servo {port} 管道已断开")
            time.sleep(dt)

        self.append(f"[完成] {scope}到位，servo 保持中。请核对：每个关节是否朝『站起来』方向转了？")
        self.append("       哪个关节往反方向转 = 那个关节标定方向标反了。点「急停/松开」释放。")
        self.set_status(f"{scope}到位（保持中）")
        with self.lock:
            self.hold_targets = dict(targets)   # 记录保持目标，供「单关节微调」做增量基准
            self.hold_tff = {n: 0.0 for n in targets}   # 方向验证不加前馈
            self.phase = "STANDING"

    # -- 单关节微调：站立/测试到位、电机锁位保持时，对某个关节缓慢加/减一个小角度 --
    # 在已有的保持目标 hold_targets 上增量，限速插值；其余关节维持原保持位+前馈不动。
    # 调用前 run() 已做：处于保持中、未忙、过了 2s 安全间隔的校验，并已设 busy=True。
    def _do_jog(self, name, sign, deg):
        try:
            self._jog_impl(name, sign, deg)
        except Exception as e:
            self.append(f"[异常] 单关节微调出错: {e}")
            self.set_status(f"微调异常: {e}")
            self._notify(f"单关节微调出错：{e}")
        finally:
            with self.lock:
                self.busy = False

    def _jog_impl(self, name, sign, deg):
        jt = next((j for j in self.joints if j["name"] == name), None)
        if jt is None:
            self.append(f"[微调] 未知关节 {name}")
            return
        port = jt["port"]
        with self.lock:
            proc = self.servo_procs.get(port)
            alive = proc is not None and proc.poll() is None
            targets = dict(self.hold_targets)
            tffs = dict(self.hold_tff)
        if not alive or name not in targets:
            self.append(f"[微调] 拒绝：{name} 当前未处于锁位保持（{port} 无存活 servo 或无保持目标）。")
            self._notify("该关节当前未在锁位保持，无法微调：请先站立/单腿测试到位（电机保持）后再调。", "warn")
            return

        # 该总线上、当前在保持的所有关节都要每行重发，否则其余关节会丢失锁位
        port_joints = [j for j in self.joints
                       if j["port"] == port and j["name"] in targets]
        step_rotor = sign * math.radians(deg) * GEAR_RATIO
        start = targets[name]
        end = start + step_rotor
        rate = 100.0
        vmax_rotor = JOG_VMAX * GEAR_RATIO
        duration = max(0.4, abs(step_rotor) / vmax_rotor if vmax_rotor > 0 else 0.4)
        N = max(1, int(duration * rate))
        dt = 1.0 / rate
        s0 = math.degrees(start / GEAR_RATIO)
        s1 = math.degrees(end / GEAR_RATIO)
        arrow = "＋" if sign > 0 else "－"
        self.append(f"[微调] {name} {arrow}{deg:.1f}°(关节)：{s0:.2f}° → {s1:.2f}°，限速 "
                    f"{JOG_VMAX} rad/s、约 {duration:.1f}s；其余关节保持不动。")
        self.set_status(f"微调 {name} {arrow}{deg:.1f}°（约 {duration:.1f}s）")

        for k in range(1, N + 1):
            if self.abort.is_set():
                self.append("[微调] 收到急停/中止，停止微调。")
                return
            with self.lock:
                proc = self.servo_procs.get(port)
            if proc is None or proc.poll() is not None:
                self.append(f"[微调] {port} 的 servo 已结束（可能已松开），停止微调。")
                return
            a = k / N
            parts = []
            for j in port_joints:
                n = j["name"]
                pos = (start + a * step_rotor) if n == name else targets[n]
                parts.append(f"{j['id']} {pos:.5f} {tffs.get(n, 0.0):.4f}")
            try:
                proc.stdin.write(" ".join(parts) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.append(f"[微调] servo {port} 管道已断开，停止微调。")
                return
            time.sleep(dt)

        with self.lock:
            self.hold_targets[name] = end   # 新位置成为后续微调/保持的基准
        self.append(f"[微调] 完成：{name} 现保持在 {s1:.2f}°(关节)。{JOG_COOLDOWN:.0f}s 后可继续微调。")
        self.set_status(f"微调完成：{name} = {s1:.2f}°（保持中）")

    # -- 急停 / 释放 --
    def _estop(self):
        self.abort.set()
        self.append("\n[急停] 中止动作并释放电机 ...")
        self.set_status("急停")
        self._stop_servos()
        with self.lock:
            self._refresh_phase()
        self.append("[急停] 已停止。")

    def _finish(self, title):
        with self.lock:
            self.busy = False
        self.set_status(f"完成: {title}")

    def _start_bg(self, fn, *args):
        with self.lock:
            self.busy = True
        self.abort.clear()
        threading.Thread(target=fn, args=args, daemon=True).start()

    # -- 入口 --
    def run(self, action, p):
        with self.lock:
            busy = self.busy
            standing = (self.standing_thread is not None
                        and self.standing_thread.is_alive())

        if action == "estop":
            threading.Thread(target=self._estop, daemon=True).start()
            return True, "ok"

        if action == "read":
            if busy or standing:
                return False, "忙：请等待当前操作完成"
            self._start_bg(self._read_current)
            return True, "ok"

        if action == "reload_config":
            return self._reload_config_now()

        if action == "reset_peak":
            with self.lock:
                self.tau_peak = {}
            self.append("[扭矩] 峰值力矩已清零。")
            return True, "峰值力矩已清零"

        if action in ("calib_prone", "calib_stand"):
            if busy or standing:
                return False, "忙：请等待当前操作完成"
            self._start_bg(self._calib, "prone" if action == "calib_prone" else "stand")
            return True, "ok"

        if action == "save_config":
            if busy or standing:
                return False, "忙：请等待当前操作完成"
            ok, msg = self._save_config()
            return ok, msg

        if action == "reanchor_prone":
            if busy or standing:
                return False, "忙：请等待当前操作完成"
            self._start_bg(self._reanchor_prone)
            return True, "ok"

        if action == "sit":
            # 坐下功能暂时禁用（代码有问题）。即便绕过前端直接调 API 也一律拒绝。
            return False, "坐下功能已禁用（代码有问题，暂不可用）"

        if action in ("stand", "sit"):
            descend = action == "sit"
            verb = "坐下" if descend else "站立"
            if busy:
                return False, "忙：请等待当前操作完成"
            if standing:
                return False, f"已在执行动作（{verb}）"
            if self.config is None:
                return False, f"尚未标定保存配置，无法{verb}"
            kp = parse_opt_gain(p.get("kp", ""))
            kw = parse_opt_gain(p.get("kw", ""))
            if kp == "ERR" or kw == "ERR":
                return False, "K_P / K_W 必须是数字（留空表示用配置默认值）"
            vmax = parse_opt_num(p.get("vmax", ""), 0.001, 1.0)
            mindur = parse_opt_num(p.get("min_dur", ""), 0.1, 60.0)
            rate = parse_opt_num(p.get("rate", ""), 1.0, 500.0)
            thr = parse_opt_num(p.get("thr", ""), 0.0, 1.5)
            tff = parse_opt_num(p.get("tff", ""), -8.0, 8.0)
            if "ERR" in (vmax, mindur, rate, thr, tff):
                return False, "速度 / 时长 / 频率 / 阈值 / 前馈 必须是数字（留空表示用配置默认值）"
            exo = {}
            if vmax is not None:
                exo["max_joint_vel_rad_s"] = vmax
            if mindur is not None:
                exo["min_duration_s"] = mindur
            if rate is not None:
                exo["rate_hz"] = rate
            if thr is not None:
                exo["verify_threshold_rad"] = thr
            if tff is not None:
                exo["t_ff"] = tff
            # 站立方式：staged=分段(默认，站立 小腿→大腿→hip / 坐下反序) / sync=整机同时线性 /
            # natural=自然站立(全身同步+S曲线缓入缓出)。未传 style 时兼容旧的 staged=0/1 开关。
            style = str(p.get("style", "")).strip().lower()
            if style in ("staged", "sync", "natural"):
                exo["style"] = style
            else:
                exo["staged"] = str(p.get("staged", "1")).strip() in ("1", "true", "True", "on")
            self.abort.clear()
            t = threading.Thread(target=self._do_stand,
                                 kwargs={"kp": kp, "kw": kw, "ex_override": exo, "descend": descend},
                                 daemon=True)
            with self.lock:
                self.standing_thread = t
            t.start()
            return True, "ok"

        if action == "test_leg":
            if busy:
                return False, "忙：请等待当前操作完成"
            if standing:
                return False, "已在执行动作"
            if self.config is None:
                return False, "尚未标定保存配置，无法测试"
            leg = str(p.get("leg", "")).strip().lower()
            if leg not in ("fl", "fr", "rl", "rr"):
                return False, "腿名必须是 fl/fr/rl/rr"
            kp = parse_opt_gain(p.get("kp", ""))
            kw = parse_opt_gain(p.get("kw", ""))
            if kp == "ERR" or kw == "ERR":
                return False, "K_P / K_W 必须是数字（留空表示用配置默认值）"
            skip_verify = str(p.get("skip_verify", "")).strip() in ("1", "true", "True", "on")
            if skip_verify:
                self.append("[请求] 「只测这条腿」已勾选『关闭安全校验』，将跳过趴姿偏差检查。")
            self.abort.clear()
            t = threading.Thread(target=self._do_stand, args=({leg},),
                                 kwargs={"kp": kp, "kw": kw, "skip_verify": skip_verify},
                                 daemon=True)
            with self.lock:
                self.standing_thread = t
            t.start()
            return True, "ok"

        if action == "dir_test":
            if busy:
                return False, "忙：请等待当前操作完成"
            if standing:
                return False, "已在执行动作"
            if self.config is None:
                return False, "尚未标定保存配置，无法验证"
            leg = str(p.get("leg", "")).strip().lower()
            if leg not in ("fl", "fr", "rl", "rr"):
                return False, "腿名必须是 fl/fr/rl/rr"
            joint = str(p.get("joint", "")).strip().lower() or None
            if joint not in (None, "hip", "thigh", "shank", "all"):
                return False, "关节必须是 hip/thigh/shank（或 all/留空表示整腿）"
            if joint == "all":
                joint = None
            try:
                deg = float(p.get("deg", 10.0))
            except (TypeError, ValueError):
                deg = 10.0
            deg = max(1.0, min(30.0, deg))   # 限幅 1~30°，防误填大角度
            kp = parse_opt_gain(p.get("kp", ""))
            kw = parse_opt_gain(p.get("kw", ""))
            if kp == "ERR" or kw == "ERR":
                return False, "K_P / K_W 必须是数字（留空表示用配置默认值）"
            self.abort.clear()
            t = threading.Thread(target=self._do_dir_test, args=(leg, deg, joint),
                                 kwargs={"kp": kp, "kw": kw}, daemon=True)
            with self.lock:
                self.standing_thread = t
            t.start()
            return True, "ok"

        if action in ("confirm_dir", "unconfirm_dir", "invert_dir"):
            if busy or standing:
                return False, "忙：请等待当前动作完成再确认/取反"
            if self.config is None:
                return False, "尚无配置文件"
            name = str(p.get("name", "")).strip()
            if not name:
                return False, "缺少关节名"
            mode = {"confirm_dir": "confirm", "unconfirm_dir": "unconfirm",
                    "invert_dir": "invert"}[action]
            return self._set_dir(name, mode)

        if action == "set_pose_angle":
            if busy or standing:
                return False, "忙：请等待当前动作完成再修改"
            if self.config is None:
                return False, "尚无配置文件，请先标定保存"
            name = str(p.get("name", "")).strip()
            if not name:
                return False, "缺少关节名"
            pose = str(p.get("pose", "")).strip().lower()
            if pose not in ("prone", "stand"):
                return False, "姿态必须是 prone(趴) 或 stand(站)"
            try:
                deg = float(p.get("deg", ""))
            except (TypeError, ValueError):
                return False, "角度必须是数字（单位：度）"
            return self._set_pose_angle(name, pose, deg)

        if action == "set_feedforward":
            if busy or standing:
                return False, "忙：请等待当前动作完成再修改"
            if self.config is None:
                return False, "尚无配置文件，请先标定保存"
            name = str(p.get("name", "")).strip()
            if not name:
                return False, "缺少关节名"
            try:
                tff = float(p.get("tff", ""))
            except (TypeError, ValueError):
                return False, "前馈力矩必须是数字（单位：N·m）"
            return self._set_feedforward(name, tff)

        if action == "jog":
            if busy:
                return False, "忙：请等待当前操作完成"
            if standing:
                return False, "正在执行动作，请等到位保持后再微调"
            name = str(p.get("name", "")).strip()
            if not name:
                return False, "缺少关节名"
            direction = str(p.get("dir", "")).strip()
            if direction in ("+", "plus", "up"):
                sign = 1
            elif direction in ("-", "minus", "down"):
                sign = -1
            else:
                return False, "方向必须是 + 或 -"
            try:
                deg = float(p.get("deg", JOG_STEP_MAX))
            except (TypeError, ValueError):
                return False, "角度必须是数字（单位：度）"
            deg = max(0.1, min(JOG_STEP_MAX, deg))   # 每次最多 5°，防误填大角度
            # 必须处于锁位保持，且过了安全间隔
            with self.lock:
                holding = any(pp.poll() is None for pp in self.servo_procs.values())
                held = name in self.hold_targets
                last = self.jog_last.get(name, 0.0)
            if not holding or not held:
                return False, "请先站立/单腿测试到位（电机锁位保持）后，再微调该关节"
            now = time.monotonic()
            remain = JOG_COOLDOWN - (now - last)
            if remain > 0:
                return False, f"安全间隔：请 {remain:.1f}s 后再微调 {name}"
            with self.lock:
                self.jog_last[name] = now
                self.busy = True
            self.abort.clear()
            threading.Thread(target=self._do_jog, args=(name, sign, deg),
                             daemon=True).start()
            return True, "ok"

        return False, f"未知操作: {action}"

    def _read_current(self):
        self.append("\n[读取] 读取全部 12 关节当前转子角 ...")
        self.set_status("读取当前角度")
        res, oks = self._read_all()
        with self.lock:
            self.current = dict(res)
            self.current_ok = dict(oks)
        self._finish("读取角度")


CTRL = RobotController()


def check_env():
    if not os.path.isfile(MOTOR_CTRL):
        CTRL.append(f"[警告] 未找到 {MOTOR_CTRL}；请在 SDK 目录 build/ 执行 cmake .. && make motor_ctrl")
    else:
        CTRL.append(f"[就绪] motor_ctrl: {MOTOR_CTRL}")
    if CTRL.config:
        CTRL.append(f"[就绪] 已加载配置 {STAND_CONFIG}（标定于 {CTRL.config.get('calibrated_at','?')}）")
        CTRL.append("[提示] 单圈绝对值编码器，绝对位置断电不变，配置跨上电有效；仅在机械改动/重新装配/换电机后才需重标。")
    else:
        CTRL.append("[提示] 尚无配置文件，请先按 ①②③ 完成标定。")


# ----------------------------------------------------------------- 前端页面
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>四足机器人操控</title>
<style>
  /* ── 设计变量（只改样式,不改任何 id/class/JS 逻辑）─────────────────── */
  :root {
    --bd:#e2e7ef; --bd2:#d2d9e5; --fg:#1c2433; --muted:#7c8698;
    --warn:#b07a10; --pri:#2f6bea; --pri2:#1e56cf; --ok:#1f8a4c; --bad:#d64545;
    --card:#ffffff; --bg:#eef1f6;
    --shadow:0 1px 2px rgba(16,24,40,.05), 0 6px 20px rgba(16,24,40,.06);
    --r:14px;
  }
  * { box-sizing: border-box; }
  html{ scroll-behavior:smooth; }
  body { margin:0; padding:20px 22px 30px; color:var(--fg);
    background:
      radial-gradient(1100px 420px at 88% -120px, rgba(47,107,234,.10), transparent 60%),
      radial-gradient(900px 380px at -8% -90px, rgba(31,138,76,.08), transparent 55%),
      var(--bg);
    background-attachment: fixed;
    font-family: system-ui, -apple-system, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    -webkit-font-smoothing: antialiased; }
  h1 { font-size:21px; letter-spacing:.3px; margin:2px 0 16px; }
  .muted { color:var(--muted); }

  /* ── 卡片 ── */
  fieldset { border:1px solid var(--bd); border-radius:var(--r); margin:0 0 16px;
    padding:14px 16px 15px; background:var(--card); box-shadow:var(--shadow); }
  legend { font-weight:700; font-size:13px; padding:3px 12px; color:#3d4c68;
    background:linear-gradient(180deg,#f7f9fc,#eef2f8); border:1px solid var(--bd);
    border-radius:999px; letter-spacing:.2px; }
  .row { display:flex; flex-wrap:wrap; align-items:center; gap:8px 10px; }

  /* ── 按钮 ── */
  button { padding:7px 14px; border:1px solid var(--bd2); border-radius:9px; background:#fff;
    cursor:pointer; font-size:14px; color:var(--fg); box-shadow:0 1px 1.5px rgba(16,24,40,.06);
    transition:border-color .15s, color .15s, box-shadow .15s, transform .06s, background .15s, opacity .15s; }
  button:hover:not(:disabled){ border-color:var(--pri); color:var(--pri);
    box-shadow:0 2px 10px rgba(47,107,234,.18); }
  button:active:not(:disabled){ transform:translateY(1px); }
  button:disabled{ opacity:.45; cursor:not-allowed; }
  button.primary{ background:linear-gradient(180deg,var(--pri),var(--pri2)); color:#fff; border-color:var(--pri2); }
  button.primary:hover:not(:disabled){ color:#fff; filter:brightness(1.07); box-shadow:0 3px 12px rgba(47,107,234,.35); }
  button.stand{ background:linear-gradient(180deg,#27a35c,#1f8a4c); color:#fff; border-color:#1a7a42;
    font-size:16px; padding:12px 26px; border-radius:12px; font-weight:600;
    box-shadow:0 2px 10px rgba(31,138,76,.28); }
  button.stand:hover:not(:disabled){ color:#fff; filter:brightness(1.06); box-shadow:0 4px 14px rgba(31,138,76,.38); }
  button.estop{ background:linear-gradient(180deg,#e0524f,#cf3d3d); color:#fff; border-color:#c23636; font-weight:700; }
  button.estop:hover:not(:disabled){ color:#fff; filter:brightness(1.06); box-shadow:0 3px 12px rgba(214,69,69,.4); }
  button.mini{ padding:3px 9px; font-size:12px; border-radius:7px; }

  /* ── 表单控件（输入框多为内联样式,这里补 select 基础样式与统一焦点态）── */
  select { padding:6px 9px; border:1px solid var(--bd2); border-radius:8px; background:#fff;
    font-size:13.5px; color:var(--fg); cursor:pointer; }
  input[type=number]:focus, select:focus, input[type=checkbox]:focus-visible {
    outline:none; border-color:var(--pri) !important;
    box-shadow:0 0 0 3px rgba(47,107,234,.15); }
  input[type=checkbox]{ accent-color:var(--pri); }
  label{ font-size:13.5px; }

  /* ── 徽标 ── */
  .pill{ font-size:12px; padding:4px 12px; border-radius:999px; background:#eef1f6;
    color:#5a6577; border:1px solid var(--bd); font-weight:500; }
  .pill.ok{ background:#e5f6ec; color:var(--ok); border-color:#c4e8d2; }
  .pill.warn{ background:#fdf5e0; color:var(--warn); border-color:#f1e2b6; }

  /* ── 表格 ── */
  table{ border-collapse:separate; border-spacing:0; width:100%; font-size:13px;
    border:1px solid var(--bd); border-radius:10px; overflow:hidden; }
  th,td{ border:none; border-bottom:1px solid var(--bd); padding:6px 8px; text-align:center; }
  th{ background:linear-gradient(180deg,#f4f6fa,#ecf0f6); color:#3d4c68; font-weight:600;
    border-bottom:1px solid var(--bd2); white-space:nowrap; }
  td + td, th + th { border-left:1px solid #eef1f6; }
  th + th { border-left-color:#e2e7ef; }
  tbody tr:last-child td{ border-bottom:none; }
  tbody tr:nth-child(even) td{ background:#fafbfd; }
  tbody tr:hover td{ background:#f1f5fd; }
  td.bad{ background:#fdecec !important; color:var(--bad); font-weight:600; }

  /* ── 方向手柄 ── */
  .pad{ display:grid; grid-template-columns:repeat(3,52px); grid-template-rows:repeat(3,52px); gap:6px; }
  .pad button{ width:52px; height:52px; padding:0; font-size:18px; border-radius:12px; }
  .grow{ flex:1; }

  /* ── 日志与状态栏 ── */
  pre#log{ height:240px; overflow:auto; margin:0; padding:10px 12px; background:#161b24; color:#d5dbe5;
    border-radius:10px; font-family:"Noto Sans Mono CJK SC", ui-monospace, monospace; font-size:12.5px;
    line-height:1.5; white-space:pre-wrap; word-break:break-all;
    box-shadow:inset 0 2px 8px rgba(0,0,0,.35); }
  pre#log::-webkit-scrollbar{ width:9px; }
  pre#log::-webkit-scrollbar-thumb{ background:#3a4356; border-radius:5px; }
  #status{ flex:1; padding:7px 12px; background:var(--card); border:1px solid var(--bd);
    border-radius:9px; font-size:13px; color:#5a6577; box-shadow:var(--shadow); }

  /* ── 提示横幅 ── */
  .banner{ display:none; margin:0 0 14px; padding:9px 14px; border-radius:10px; font-size:13px; }
  .banner.bad{ display:block; border:1px solid #f3c1c1; background:#fdecec; color:var(--bad);
    box-shadow:0 2px 8px rgba(214,69,69,.10); }
  .banner.hold{ display:block; border:1px solid #f1e2b6; background:#fdf5e0; color:var(--warn);
    box-shadow:0 2px 8px rgba(176,122,16,.10); }

  /* ── 右上角固定面板：驱动状态 + 大急停（结构/行为不变）── */
  #driveBar{ position:fixed; top:12px; right:14px; z-index:1000; width:256px;
    display:flex; flex-direction:column; align-items:stretch; gap:8px;
    padding:10px; background:rgba(255,255,255,.82); backdrop-filter:blur(10px) saturate(1.3);
    -webkit-backdrop-filter:blur(10px) saturate(1.3);
    border:1px solid var(--bd); border-radius:14px;
    box-shadow:0 4px 24px rgba(16,24,40,.14);
    transition:background .15s ease, border-color .15s ease; }
  #driveBar.on{ background:rgba(253,232,232,.92); border-color:#d64545; }
  #driveState{ text-align:center; font-size:28px; font-weight:700; padding:10px 12px; border-radius:10px;
    white-space:nowrap; background:linear-gradient(180deg,#e9f7ef,#ddf2e5); color:var(--ok); }
  #driveBar.on #driveState{ background:#d64545; color:#fff; animation:dpulse 1s steps(1,end) infinite; }
  @keyframes dpulse{ 50%{ opacity:.45; } }
  /* 图标（● / ⏹）单独放大：相对各自所在文字再 2 倍 */
  .bigico{ font-size:2em; vertical-align:middle; }
  #btnEstop{ width:100%; background:linear-gradient(180deg,#e0524f,#c73a3a); color:#fff;
    border:2px solid #fff; border-radius:11px;
    font-size:36px; font-weight:800; letter-spacing:1px; line-height:1.25; padding:16px 8px; cursor:pointer;
    box-shadow:0 2px 10px rgba(190,40,40,.35); }
  #btnEstop:hover{ background:linear-gradient(180deg,#d24543,#b53232); color:#fff; border-color:#fff;
    box-shadow:0 3px 14px rgba(190,40,40,.5); }
  /* 驱动中：整页红色边框警示（不挡点击）*/
  #driveEdge{ position:fixed; inset:0; border:14px solid #d64545; box-shadow:inset 0 0 0 2px #fff; pointer-events:none; z-index:999; display:none; }
  #driveEdge.on{ display:block; animation:edgeglow 1s ease-in-out infinite; }
  @keyframes edgeglow{
    0%,100%{ box-shadow:inset 0 0 0 2px #fff, inset 0 0 22px rgba(214,69,69,.55), 0 0 22px rgba(214,69,69,.55); }
    50%{ box-shadow:inset 0 0 0 2px #fff, inset 0 0 60px rgba(214,69,69,.95), 0 0 60px rgba(214,69,69,.95); }
  }
</style>
</head>
<body>
  <div id="driveBar">
    <span id="driveState"><span class="bigico">●</span> 未驱动</span>
    <button id="btnEstop" type="button" onclick="estopNow()" title="快捷键：空格"><span class="bigico">⏹</span> 急停<br>STOP<br><span style="font-size:11px;font-weight:400;opacity:.9">空格</span></button>
  </div>
  <div id="driveEdge"></div>
  <h1>🐕 四足机器人操控 <span class="muted" style="font-size:13px">（站立标定 / 执行）</span></h1>

  <div id="warnBanner" class="banner bad"></div>
  <div id="holdBanner" class="banner"></div>

  <fieldset>
    <legend>状态</legend>
    <div class="row">
      <span class="pill" id="phasePill">阶段: -</span>
      <span class="pill" id="cfgPill">配置: -</span>
      <span class="pill" id="holdPill" title="电机是否仍带电锁位保持（未松开）">松开状态: -</span>
      <button onclick="api('reload_config')" title="从磁盘重新读取 config/stand_config.json（在 motor_web 标定保存后用它刷新趴姿/站姿）">📂 重新加载配置</button>
      <button class="primary" onclick="api('read')">📐 读取当前 12 关节角</button>
      <span class="grow"></span>
      <button class="estop" onclick="api('estop')">⏹ 急停 / 松开</button>
    </div>
  </fieldset>

  <fieldset>
    <legend>站立标定（单圈绝对值编码器：标定一次跨上电有效，机械改动后再重标）</legend>
    <div class="row">
      <button onclick="api('calib_prone')">① 记录趴姿</button>
      <button onclick="api('calib_stand')">② 记录站姿（手扶撑起后点）</button>
      <button onclick="confirmSave()">③ 计算并保存配置</button>
      <button onclick="confirmReanchor()" title="prone=当前读数、Δ保持不变、stand=prone+Δ。仅适用于零点平移（换电机/摇臂错齿）而趴→站行程未变的情况；连杆几何变过请走 ①②③ 完整标定">🔁 重锚趴姿（保持Δ）</button>
      <span class="pill" id="pronePill">趴姿: 未记录</span>
      <span class="pill" id="standPill">站姿: 未记录</span>
    </div>
    <p class="muted" style="font-size:12.5px;margin:8px 0 0">
      流程：让机器人自然趴下点①；手扶把它撑成站立姿态点②；点③算出每关节趴→站的转动量(Δ)与方向并存盘。
    </p>
  </fieldset>

  <fieldset>
    <legend>操控（手柄）</legend>
    <div class="row" style="align-items:flex-start; gap:28px">
      <div class="pad">
        <span></span><button disabled title="后续">▲</button><span></span>
        <button disabled title="后续">◀</button><button disabled title="后续">●</button><button disabled title="后续">▶</button>
        <span></span><button disabled title="后续">▼</button><span></span>
      </div>
      <div>
        <button class="stand" id="btnStand" onclick="confirmStand()">🧍 站立</button>
        <button class="stand" id="btnSit" style="background:#777; border-color:#777"
                disabled title="坐下功能暂时禁用（代码有问题）">🪑 坐下（已禁用）</button>
        <div id="standParams" style="margin:10px 0 0;max-width:430px">
          <div class="muted" style="font-size:12px;margin-bottom:6px">
            <b style="color:var(--fg)">整机站立参数</b>（留空=用配置 <code>execute</code> 默认值，灰字即当前默认；填了则临时覆盖，<b>不</b>写回配置）：
          </div>
          <div style="display:grid;grid-template-columns:auto 90px auto 90px;gap:6px 10px;align-items:center;font-size:13px">
            <label title="位置环刚度，越大越硬。范围 0~25.599">K_P</label>
            <input id="stKp" type="number" min="0" max="25.599" step="0.1" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="padding:4px 6px;border:1px solid var(--bd);border-radius:6px">
            <label title="速度阻尼。范围 0~25.599">K_W</label>
            <input id="stKw" type="number" min="0" max="25.599" step="0.01" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="padding:4px 6px;border:1px solid var(--bd);border-radius:6px">
            <label title="起身速度上限（关节角速度）。范围 0.001~1.0">速度 rad/s</label>
            <input id="stVmax" type="number" min="0.001" max="1.0" step="0.01" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="padding:4px 6px;border:1px solid var(--bd);border-radius:6px">
            <label title="最短运动时长，过短会变快。范围 0.1~60">最小时长 s</label>
            <input id="stMinDur" type="number" min="0.1" max="60" step="0.5" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="padding:4px 6px;border:1px solid var(--bd);border-radius:6px">
            <label title="伺服控制频率。范围 1~500">频率 Hz</label>
            <input id="stRate" type="number" min="1" max="500" step="10" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="padding:4px 6px;border:1px solid var(--bd);border-radius:6px">
            <label title="趴姿安全校验阈值，越大越宽松。范围 0~1.5">校验阈值 rad</label>
            <input id="stThr" type="number" min="0" max="1.5" step="0.01" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="padding:4px 6px;border:1px solid var(--bd);border-radius:6px">
            <label title="给所有关节的统一基准前馈力矩(转子侧N·m)，叠加每关节表里的 t_ff。范围 -8~8">前馈力矩 N·m</label>
            <input id="stTff" type="number" min="-8" max="8" step="0.05" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="padding:4px 6px;border:1px solid var(--bd);border-radius:6px">
          </div>
          <div style="display:flex;align-items:center;gap:7px;margin-top:9px;font-size:13px">
            <label for="stStyle" title="分段抬升：小腿→大腿→hip 逐组抬，降低同时驱动的峰值力矩/电流，但段间有停顿、观感偏机械。&#10;自然站立：全身 12 关节同步、按 S 曲线缓入缓出（起停速度/加速度为 0），动作连贯流畅、更像动物起身；瞬时峰值速度不超上面的限速，但同时驱动的峰值电流更高。&#10;整机同时：全关节同步匀速直线插值（旧行为）。">站立方式</label>
            <select id="stStyle" style="flex:1">
              <option value="staged">分段抬升（小腿→大腿→hip，峰值电流低，稳妥）</option>
              <option value="natural">自然站立（全身同步 + S曲线缓入缓出，动作流畅）</option>
              <option value="sync">整机同时（线性匀速）</option>
            </select>
          </div>
        </div>
        <p class="muted" style="font-size:12.5px;margin:8px 0 0;max-width:430px">
          点「站立」会先读当前角并与配置趴姿比对，全部贴近（在「校验阈值」内）才放行，然后按上面的
          「速度」限速缓慢起身。整机站立用上面这组参数，<b>不</b>读单腿验证的 K_P/K_W 输入框。这里填的值
          只对本次生效、不写回配置；要永久修改请编辑 <code>stand_config.json</code> 的 <code>execute</code> 段后点「📂 重新加载配置」。
          <b>前馈力矩</b>是给所有关节的统一基准，<b>叠加</b>下方表里每关节单独调的 t_ff（统一基准 + 每关节微调）；用来在“误差小却下沉”时直接补偿自重。方向键为后续行走动作占位。
        </p>
      </div>
      <div style="flex:1; min-width:440px">
        <div style="font-weight:600;color:#555;margin-bottom:8px">📊 扭矩监控 <span class="muted" style="font-weight:400;font-size:12px">实时各关节力矩 + 峰值 + 上限线，判断扭矩是否不足</span></div>
        <div class="row" style="margin-bottom:8px">
          <label title="力矩上限参考线。手册未给物理额定值，请按电机规格/堵转实测填写">力矩上限参考</label>
          <input id="tqLimit" type="number" min="0" step="0.5" value="23.7" oninput="tqOnChange()" style="width:84px; padding:5px 7px; border:1px solid var(--bd); border-radius:6px; font-size:14px">
          <select id="tqSide" onchange="tqOnChange()">
            <option value="joint">关节输出端 N·m（转子×6.33）</option>
            <option value="rotor">电机转子侧 N·m</option>
          </select>
          <button class="mini" onclick="resetPeak()" title="把所有关节的峰值力矩清零，重新开始统计">↺ 峰值清零</button>
          <span class="grow"></span>
          <span id="tqVerdict" style="font-size:13px;font-weight:600"></span>
        </div>
        <canvas id="tqChart" width="940" height="240" style="width:100%;border:1px solid var(--bd);border-radius:6px;background:#fff;display:block"></canvas>
        <p class="muted" style="font-size:12px;margin:8px 0 0">
          实心柱=当前 |τ|，柱顶横杠=本次<b>峰值</b> |τ|，水平虚线=力矩上限参考。柱接近/超过上限、同时上表「跟踪误差」仍较大 → 该关节<b>扭矩不足</b>（出满力也顶不住、会下沉）。
          站立/方向验证开始时峰值自动清零。手册只给协议编码上限 ±127.99 N·m(转子)，<b>非</b>物理额定；默认 23.7 仅为常见参考，请按实际电机规格修改。温度近 90℃ 会触发过热保护(MERROR)。
        </p>
        <div class="row" style="margin:12px 0 6px; border-top:1px dashed var(--bd); padding-top:10px">
          <label>曲线显示</label>
          <select id="tqLineSel" onchange="tqOnChange()">
            <option value="all">全部 12 关节</option>
            <optgroup label="按腿（3 关节）">
              <option value="fl">左前 FL</option>
              <option value="fr">右前 FR</option>
              <option value="rl">左后 RL</option>
              <option value="rr">右后 RR</option>
            </optgroup>
            <optgroup label="单关节">
              <option value="fl_hip">fl_hip</option><option value="fl_thigh">fl_thigh</option><option value="fl_shank">fl_shank</option>
              <option value="fr_hip">fr_hip</option><option value="fr_thigh">fr_thigh</option><option value="fr_shank">fr_shank</option>
              <option value="rl_hip">rl_hip</option><option value="rl_thigh">rl_thigh</option><option value="rl_shank">rl_shank</option>
              <option value="rr_hip">rr_hip</option><option value="rr_thigh">rr_thigh</option><option value="rr_shank">rr_shank</option>
            </optgroup>
          </select>
          <label>时间窗</label>
          <select id="tqWin" onchange="tqOnChange()">
            <option value="30">30s</option><option value="60">1min</option><option value="180">3min</option>
            <option value="300">5min</option><option value="600" selected>10min</option><option value="0">全部</option>
          </select>
          <label title="关闭后曲线冻结、停止记录，可静态翻看（调时间窗/关节）已记录的数据；重新开启继续实时记录" style="user-select:none">
            <input type="checkbox" id="tqLive" checked onchange="tqOnChange()" style="vertical-align:middle"> 实时显示
          </label>
          <span id="tqLiveState" style="font-size:12.5px;color:var(--warn);font-weight:600"></span>
          <button class="mini" onclick="tqClearHistory();tqRedraw()" title="清空曲线历史，重新记录">↺ 清曲线</button>
        </div>
        <canvas id="tqLine" width="940" height="200" style="width:100%;border:1px solid var(--bd);border-radius:6px;background:#fff;display:block"></canvas>
        <p class="muted" style="font-size:12px;margin:6px 0 0">
          起身全过程的力矩随时间变化（带<b>符号</b>，看方向与峰值出现时刻）；±上限为红虚线，0 为中线。新动作开始自动清历史；关闭「实时显示」即冻结，可慢慢看前 10 分钟。
        </p>
      </div>
    </div>
    <div class="row" style="margin-top:12px; border-top:1px dashed var(--bd); padding-top:10px">
      <label>单腿验证</label>
      <select id="legSel">
        <option value="fl">左前 FL (ttyUSB0 / id 1,2,3)</option>
        <option value="fr">右前 FR (ttyUSB1 / id 4,5,6)</option>
        <option value="rl">左后 RL (ttyUSB2 / id 7,8,9)</option>
        <option value="rr">右后 RR (ttyUSB3 / id 10,11,12)</option>
      </select>
      <label title="位置环刚度。留空=用配置 execute.k_p；越大越硬、力矩越大。范围 0~25.599">K_P</label>
      <input id="legKp" type="number" min="0" max="25.599" step="0.1" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="width:80px; padding:5px 7px; border:1px solid var(--bd); border-radius:6px; font-size:14px">
      <label title="速度阻尼。留空=用配置 execute.k_w。范围 0~25.599">K_W</label>
      <input id="legKw" type="number" min="0" max="25.599" step="0.1" placeholder="默认" oninput="boundInput(this)" onchange="boundClamp(this)" style="width:80px; padding:5px 7px; border:1px solid var(--bd); border-radius:6px; font-size:14px">
      <button id="btnTestLeg" onclick="confirmTestLeg()">🦵 只测这条腿（趴→站）</button>
      <label id="skipWrap" title="勾选后『只测这条腿』将跳过『当前是否≈趴姿』的安全校验，直接按 当前角+Δ 运动。危险：若不在趴姿附近会大幅运动！" style="color:#c0392b; font-size:12.5px; user-select:none">
        <input type="checkbox" id="legSkipVerify" style="vertical-align:middle"> 关闭安全校验
      </label>
      <span style="width:10px"></span>
      <label>电机</label>
      <select id="dirJoint">
        <option value="hip">hip 髋</option>
        <option value="thigh">thigh 大腿</option>
        <option value="shank">shank 小腿</option>
        <option value="all">全部(3个)</option>
      </select>
      <label>角度°</label>
      <input id="dirDeg" type="number" min="1" max="30" step="1" value="10" oninput="boundInput(this)" onchange="boundClamp(this)" style="width:64px; padding:5px 7px; border:1px solid var(--bd); border-radius:6px; font-size:14px">
      <button id="btnDirTest" onclick="confirmDirTest()">🧭 方向验证（单电机转设定角度）</button>
      <span class="muted" style="font-size:12.5px">只动选中那条腿、只在它那一路总线发指令，其它腿不碰。请机器人架空、该腿周围清空。</span>
    </div>
    <p class="muted" style="font-size:12.5px;margin:8px 0 0">
      「趴→站」从趴姿走到站姿（要求当前≈趴姿）；「方向验证」不要求趴姿——选一个电机(hip/thigh/shank)，从当前位置朝配置里的站立方向转设定角度，**一次只驱一个电机**、专门核对该电机方向符号对不对。
      上面的 <b>K_P / K_W</b> 同时作用于这两个单腿动作：留空则各自沿用配置 <code>execute</code> 里的默认值（输入框灰字即当前默认值）；填了就临时覆盖（不写回配置文件）。调小 K_P 更软更安全，调大更硬跟随更紧。
    </p>
  </fieldset>

  <fieldset>
    <legend>单关节微调（站立保持时，缓慢调整某关节角度以找更好站姿）</legend>
    <div class="row">
      <label>关节</label>
      <select id="jogJoint">
        <option value="fl_hip">左前 FL · hip 髋</option>
        <option value="fl_thigh">左前 FL · thigh 大腿</option>
        <option value="fl_shank">左前 FL · shank 小腿</option>
        <option value="fr_hip">右前 FR · hip 髋</option>
        <option value="fr_thigh">右前 FR · thigh 大腿</option>
        <option value="fr_shank">右前 FR · shank 小腿</option>
        <option value="rl_hip">左后 RL · hip 髋</option>
        <option value="rl_thigh">左后 RL · thigh 大腿</option>
        <option value="rl_shank">左后 RL · shank 小腿</option>
        <option value="rr_hip">右后 RR · hip 髋</option>
        <option value="rr_thigh">右后 RR · thigh 大腿</option>
        <option value="rr_shank">右后 RR · shank 小腿</option>
      </select>
      <label title="每次微调的角度（关节°），最大 10°">步长°</label>
      <input id="jogDeg" type="number" min="0.1" max="10" step="0.5" value="5" oninput="boundInput(this)" onchange="boundClamp(this)" style="width:64px; padding:5px 7px; border:1px solid var(--bd); border-radius:6px; font-size:14px">
      <button id="btnJogMinus" class="primary" onclick="jog(-1)" disabled style="font-size:18px;min-width:48px">−</button>
      <button id="btnJogPlus" class="primary" onclick="jog(1)" disabled style="font-size:18px;min-width:48px">＋</button>
      <span id="jogHint" style="font-size:12.5px;color:var(--warn);font-weight:600"></span>
      <span class="muted" id="jogTip" style="font-size:12.5px">需先站立/单腿测试到位、电机锁位保持后才可微调。</span>
    </div>
    <p class="muted" style="font-size:12.5px;margin:8px 0 0">
      在当前保持位上对所选关节缓慢加(＋)/减(−)一个角度，<b>每次最多 10°</b>、限速移动，其余关节维持不动。
      为安全，两次微调间隔至少 <b>1 秒</b>（按钮会短暂禁用并倒计时）；想多调就一次次点。随时可按<b>空格/急停</b>松开。
      微调只是临时改保持位、<b>不</b>写回站姿配置；调到满意后若要存为新站姿，用下方「手动修改姿态角度」的「↺ 取当前电机角」写入 stand。
    </p>
  </fieldset>

  <fieldset>
    <legend>关节角（单位：度°，= 转子角 ÷ 6.33 × 180/π；内部控制/校验仍用转子角 rad）</legend>
    <table>
      <thead><tr>
        <th>关节</th><th>串口</th><th>ID</th><th>当前(°)</th><th>趴姿(°)</th><th>站姿(°)</th>
        <th>Δ(站−趴,°)</th><th>方向</th><th>校验偏差(°)</th>
        <th title="实测转子角速度，保持时应≈0；来回跳动=震动">实测ω(rad/s)</th>
        <th title="实测力矩(N·m)，保持时应为较小常值；大幅摆动=震动">实测τ(N·m)</th>
        <th title="目标−实测 的跟踪误差(关节°)，保持/运动时应趋近 0">跟踪误差(°)</th>
        <th title="前馈力矩 t_ff(转子侧 N·m)，站立/保持时补偿重力以减小下沉">前馈τ(N·m)</th>
        <th>确认/取反</th>
      </tr></thead>
      <tbody id="jbody"><tr><td colspan="14" class="muted">（点「读取当前 12 关节角」）</td></tr></tbody>
    </table>
  </fieldset>

  <fieldset>
    <legend>角度跟踪（当前角 vs 站姿目标，看各关节是否到位 / 被掰动 / 下沉）</legend>
    <div class="row" style="margin-bottom:8px">
      <label title="到位判定阈值：|当前−站姿|≤此值算到位(绿)；越大越宽松">到位阈值°</label>
      <input id="angThr" type="number" min="0" max="60" step="0.5" value="5" oninput="boundInput(this)" onchange="boundClamp(this)" style="width:72px; padding:5px 7px; border:1px solid var(--bd); border-radius:6px; font-size:14px">
      <span class="grow"></span>
      <span id="angVerdict" style="font-size:13px;font-weight:600"></span>
    </div>
    <canvas id="angChart" width="940" height="220" style="width:100%;border:1px solid var(--bd);border-radius:6px;background:#fff;display:block"></canvas>
    <p class="muted" style="font-size:12px;margin:8px 0 0">
      柱 = 当前角 − 站姿目标（关节°，已按整圈归一消除单圈编码器圈数歧义），<b>0 = 正好到位</b>，柱越长离站姿越远。柱下方标「当前/站姿」实际角度。
      站立到位后若某关节柱明显偏离 0、或你一掰它**柱子变大且不回弹** → 该关节<b>没出力/刚度不足</b>（调大 K_P）。
    </p>
  </fieldset>

  <fieldset>
    <legend>手动修改姿态角度（直接改某关节的趴姿/站姿目标角，写回 stand_config.json 并重算 Δ）</legend>
    <div class="row">
      <label>关节</label>
      <select id="poseJoint" onchange="fillPoseCurrent()">
        <option value="fl_hip">左前 FL · hip 髋</option>
        <option value="fl_thigh">左前 FL · thigh 大腿</option>
        <option value="fl_shank">左前 FL · shank 小腿</option>
        <option value="fr_hip">右前 FR · hip 髋</option>
        <option value="fr_thigh">右前 FR · thigh 大腿</option>
        <option value="fr_shank">右前 FR · shank 小腿</option>
        <option value="rl_hip">左后 RL · hip 髋</option>
        <option value="rl_thigh">左后 RL · thigh 大腿</option>
        <option value="rl_shank">左后 RL · shank 小腿</option>
        <option value="rr_hip">右后 RR · hip 髋</option>
        <option value="rr_thigh">右后 RR · thigh 大腿</option>
        <option value="rr_shank">右后 RR · shank 小腿</option>
      </select>
      <label>姿态</label>
      <select id="poseSel">
        <option value="prone">趴姿 prone</option>
        <option value="stand">站姿 stand</option>
      </select>
      <label>角度°</label>
      <input id="poseDeg" type="number" step="0.1" min="-360" max="360" placeholder="度" oninput="boundInput(this)" onchange="boundClamp(this)" style="width:96px; padding:5px 7px; border:1px solid var(--bd); border-radius:6px; font-size:14px">
      <button class="mini" onclick="fillPoseCurrent()" title="把输入框填成该关节『当前电机实测角』（需先点「读取当前 12 关节角」）">↺ 取当前电机角</button>
      <button class="primary" onclick="confirmSetPose()">✎ 写入该姿态角</button>
    </div>
    <p class="muted" style="font-size:12.5px;margin:8px 0 0">
      流程：把这条腿手扶到目标姿态 → 点上方「📐 读取当前 12 关节角」→ 选好「关节 + 姿态」→ 点「↺ 取当前电机角」把实测角填进来 → 「✎ 写入该姿态角」。
      也可直接手填度数（单位与上表「趴姿/站姿(°)」一致）。保存后自动重算 Δ(站−趴) 并清除该关节方向验证标记。
      适合微调某条腿标错的趴/站姿（例如右后腿 rr_*）。<b>本操作不驱动电机</b>；改完用「方向验证」或「只测这条腿」复核。
    </p>
  </fieldset>

  <fieldset>
    <legend>前馈力矩 τ_ff（给关节加恒定保持力矩补偿重力，解决“误差小却下沉”。需重新编译 motor_ctrl）</legend>
    <div class="row">
      <label>关节</label>
      <select id="ffJoint" onchange="fillFfCurrent()">
        <option value="fl_hip">左前 FL · hip 髋</option>
        <option value="fl_thigh">左前 FL · thigh 大腿</option>
        <option value="fl_shank">左前 FL · shank 小腿</option>
        <option value="fr_hip">右前 FR · hip 髋</option>
        <option value="fr_thigh">右前 FR · thigh 大腿</option>
        <option value="fr_shank">右前 FR · shank 小腿</option>
        <option value="rl_hip">左后 RL · hip 髋</option>
        <option value="rl_thigh">左后 RL · thigh 大腿</option>
        <option value="rl_shank">左后 RL · shank 小腿</option>
        <option value="rr_hip">右后 RR · hip 髋</option>
        <option value="rr_thigh">右后 RR · thigh 大腿</option>
        <option value="rr_shank">右后 RR · shank 小腿</option>
      </select>
      <label title="转子侧前馈力矩 cmd.T，范围 ±8">前馈力矩 N·m</label>
      <input id="ffVal" type="number" step="0.05" min="-8" max="8" placeholder="0" oninput="boundInput(this)" onchange="boundClamp(this)" style="width:110px; padding:5px 7px; border:1px solid var(--bd); border-radius:6px; font-size:14px">
      <button class="mini" onclick="fillFfCurrent()" title="填回该关节配置里的前馈力矩">↺ 取当前值</button>
      <button class="primary" onclick="confirmSetFf()">✎ 写入前馈力矩</button>
    </div>
    <p class="muted" style="font-size:12.5px;margin:8px 0 0">
      作用：站立/保持时给该关节额外加一个恒定力矩，直接抵消自重，<b>不靠位置误差出力</b>，所以能在“误差很小”时也顶住、不下沉。
      <b>正负号</b>要与“顶住自重”的方向一致——<b>符号反了会帮倒忙</b>（更快下沉甚至蹬腿），可先按上表「Δ方向/实测τ」判断或小幅试出来。
      建议<b>从小往大调</b>（±0.2 → 0.5 → 1.0…），边看上表「跟踪误差」边加，误差趋近 0 即合适，范围 ±8。
      站立时按抬起进度从 0 线性加到设定值、保持阶段维持。改完<b>需重新编译 motor_ctrl</b> 才生效。本操作不驱动电机。
    </p>
  </fieldset>

  <fieldset>
    <legend>输出</legend>
    <pre id="log"></pre>
  </fieldset>

  <div class="row"><span id="status">就绪</span></div>

<script>
  let logIndex = 0;
  let lastNoticeId = null;   // 已弹过的 notice id；null=尚未建立基线（首次轮询只记基线，不回放旧通知）
  let lastExecute = null;    // 最近一次 poll 拿到的 execute 配置（含 K_P/K_W 默认值），供 placeholder / 确认框显示
  let lastJoints = [];       // 最近一次 poll 的关节行（供「手动修改姿态角度」取当前值）
  let lastGear = 6.33;       // 最近一次 poll 的减速比
  let poseInit = false;      // 是否已在首次拿到关节数据后预填「手动修改姿态」输入框
  async function api(action, params){
    params = Object.assign({action}, params||{});
    try{
      const r = await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
      const j = await r.json();
      if(!j.ok) alert(j.message||'操作失败');
    }catch(e){ alert('请求失败: '+e); }
  }
  function estopNow(){ api('estop'); }   // 急停：立即中止动作并松开电机，无需确认
  // 空格键 = 急停（全局快捷键，任何位置按下都触发；阻止默认滚动/按钮重复触发）
  window.addEventListener('keydown', function(e){
    if (e.code === 'Space' || e.key === ' ') {
      e.preventDefault();
      if (e.repeat) return;   // 长按只触发一次
      estopNow();
    }
  });
  function confirmSave(){
    if(confirm('将根据已记录的趴姿/站姿计算各关节Δ并写入 stand_config.json？')) api('save_config');
  }
  function confirmReanchor(){
    if(confirm('重锚趴姿（保持Δ）：把 12 个关节的 prone 重设为当前读数，Δ 不变，stand=prone+Δ，写回 stand_config.json（自动备份旧配置）。\n\n前提：\n· 机器人已摆好标定趴姿，各关节（尤其小腿）完全靠住限位，不再滑动；\n· 机械只发生了零点平移（换电机/摇臂错齿），趴→站行程没有变。\n若连杆几何变过，请改走 ①②③ 完整标定。\n\n执行时会间隔 5s 读两次做静止校验，任一关节漂移超 0.5° 将拒绝写入。')) api('reanchor_prone');
  }
  // 读整机站立的覆盖参数：留空则不传（后端沿用配置默认值）
  function standParams(){
    const m={stKp:'kp', stKw:'kw', stVmax:'vmax', stMinDur:'min_dur', stRate:'rate', stThr:'thr', stTff:'tff'};
    const o={};
    for(const id in m){ const v=document.getElementById(id).value.trim(); if(v!=='') o[m[id]]=v; }
    o.style = document.getElementById('stStyle').value;   // staged / natural / sync
    o.staged = o.style==='staged' ? '1' : '0';            // 兼容旧后端参数
    return o;
  }
  const STYLE_NAMES = {staged:'分段抬升（小腿→大腿→hip）', natural:'自然站立（全身同步 + S曲线缓入缓出）', sync:'整机同时（线性匀速）'};
  function confirmStand(){
    const o=standParams();
    const dv=(id,def)=>{ const el=document.getElementById(id); return el.value.trim()!=='' ? el.value.trim() : (el.placeholder||def); };
    const note='本次参数：K_P='+dv('stKp','默认')+'  K_W='+dv('stKw','默认')
      +'  速度='+dv('stVmax','默认')+'rad/s  最小时长='+dv('stMinDur','默认')+'s'
      +'  频率='+dv('stRate','默认')+'Hz  校验阈值='+dv('stThr','默认')+'rad'
      +'  前馈='+dv('stTff','默认')+'N·m(叠加每关节t_ff)'
      +'\n站立方式：'+(STYLE_NAMES[o.style]||o.style)
      +(Object.keys(o).filter(k=>k!=='staged'&&k!=='style').length?'\n（含页面覆盖值，仅本次生效，不写回配置）':'\n（数值参数全部沿用配置默认值）');
    if(confirm('确认执行整机站立？\n\n'+note+'\n\n请确保：机器人已自然趴好、周围无人无障碍、可随时按急停。'))
      api('stand', o);
  }
  // 坐下：站立的反向，复用同一套整机参数（速度/时长/频率/前馈/分段）；阈值对坐下不拦截
  function confirmSit(){
    const o=standParams();
    const dv=(id,def)=>{ const el=document.getElementById(id); return el.value.trim()!=='' ? el.value.trim() : (el.placeholder||def); };
    const note='本次参数：K_P='+dv('stKp','默认')+'  K_W='+dv('stKw','默认')
      +'  速度='+dv('stVmax','默认')+'rad/s  最小时长='+dv('stMinDur','默认')+'s'
      +'  频率='+dv('stRate','默认')+'Hz  前馈='+dv('stTff','默认')+'N·m'
      +'\n下放方式：'+(o.style==='staged'?'分段反序（hip→大腿→小腿）':(STYLE_NAMES[o.style]||o.style))+'，前馈随下放 满→0';
    if(confirm('确认执行整机坐下（回趴姿）？\n\n'+note+'\n\n请确保：机器人正站立保持、落点处无障碍、可随时按急停。'))
      api('sit', o);
  }
  // 读单腿验证用的 K_P / K_W：留空则不传，后端沿用配置默认值
  function legGains(){
    const g = {};
    const kp = document.getElementById('legKp').value.trim();
    const kw = document.getElementById('legKw').value.trim();
    if (kp !== '') g.kp = kp;
    if (kw !== '') g.kw = kw;
    return g;
  }
  function gainNote(){
    const g = legGains();
    const dk = (lastExecute && gfmt(lastExecute.k_p)) || '配置';
    const dw = (lastExecute && gfmt(lastExecute.k_w)) || '配置';
    if (g.kp === undefined && g.kw === undefined)
      return '增益：用配置默认值（K_P=' + dk + ' K_W=' + dw + '）。';
    return '增益：K_P=' + (g.kp ?? ('默认'+dk)) + '  K_W=' + (g.kw ?? ('默认'+dw)) + '（覆盖配置）。';
  }
  function confirmTestLeg(){
    const leg=document.getElementById('legSel').value;
    const skip=document.getElementById('legSkipVerify').checked;
    let msg='只测试 '+leg.toUpperCase()+' 这一条腿（趴→站），其它腿不动。\n'+gainNote();
    if(skip) msg+='\n\n⚠️⚠️ 已【关闭安全校验】：不检查是否在趴姿，直接按 当前角+Δ 运动。\n若该腿当前离趴姿较远，会发生大幅、可能危险的运动！';
    msg+='\n\n请确保：机器人已架空、该腿周围无障碍、可随时按急停。';
    if(confirm(msg)) api('test_leg', Object.assign({leg, skip_verify: skip?1:0}, legGains()));
  }
  function confirmInvert(name){
    if(confirm('取反 '+name+' 的标定方向？\n\n会翻转该电机的 delta_rotor / dir 并写回 stand_config.json。建议取反后再做一次「方向验证」确认。')) api('invert_dir',{name});
  }
  function confirmDirTest(){
    const leg=document.getElementById('legSel').value;
    const joint=document.getElementById('dirJoint').value;
    let deg=parseFloat(document.getElementById('dirDeg').value);
    if(isNaN(deg)||deg<1||deg>30){ alert('角度需在 1~30° 之间'); return; }
    const tgt = (joint==='all') ? (leg.toUpperCase()+' 全部 3 个电机') : (leg.toUpperCase()+'_'+joint+' 这一个电机');
    if(confirm('方向验证：只驱动 '+tgt+'，从当前位置朝『站立方向』转约 '+deg+'°（不要求趴姿），其它电机不动。\n'+gainNote()+'\n\n请确保：机器人已架空、该腿周围无障碍、可随时按急停。')) api('dir_test', Object.assign({leg, joint, deg}, legGains()));
  }
  // 手动修改姿态角度：把输入框填成选中关节「当前电机实测角」（转子角 -> 关节°）
  // 用法：先把腿摆到目标姿态、点「读取当前 12 关节角」，再点这里取实测角写入该姿态。
  function fillPoseCurrent(){
    const name=document.getElementById('poseJoint').value;
    const el=document.getElementById('poseDeg');
    const j=lastJoints.find(x=>x.name===name);
    const v=j ? j.current : null;   // 当前电机实测转子角(rad)
    if(v===null||v===undefined){ el.value=''; el.placeholder='先点「读取当前 12 关节角」'; return; }
    el.value=(parseFloat(v)/(lastGear||6.33)*180/Math.PI).toFixed(2);
  }
  function confirmSetPose(){
    const name=document.getElementById('poseJoint').value;
    const pose=document.getElementById('poseSel').value;
    const deg=parseFloat(document.getElementById('poseDeg').value);
    if(isNaN(deg)){ alert('请输入角度（单位：度）'); return; }
    const label=pose==='prone'?'趴姿':'站姿';
    const j=lastJoints.find(x=>x.name===name);
    const toDeg=(v)=>(v===null||v===undefined)?'未记录':((parseFloat(v)/(lastGear||6.33)*180/Math.PI).toFixed(2)+'°');
    const orig=j?(pose==='prone'?j.prone:j.stand):null;    // 配置里该姿态的原值
    const cur=j?j.current:null;                            // 当前电机实测角
    if(confirm('把 '+name+' 的'+label+'角改为 '+deg.toFixed(2)+'°？\n\n配置原值：'+toDeg(orig)+'\n当前电机实测：'+toDeg(cur)+'\n\n会写回 stand_config.json、重算 Δ(站−趴)、并清除该关节方向验证标记。\n本操作不驱动电机。'))
      api('set_pose_angle',{name,pose,deg});
  }
  // 前馈力矩：取/写该关节配置里的 t_ff（转子侧 N·m）
  function fillFfCurrent(){
    const name=document.getElementById('ffJoint').value;
    const el=document.getElementById('ffVal');
    const j=lastJoints.find(x=>x.name===name);
    const v=j?j.t_ff:null;
    el.value=(v===null||v===undefined)?'':parseFloat(v).toFixed(3);
  }
  function confirmSetFf(){
    const name=document.getElementById('ffJoint').value;
    const tff=parseFloat(document.getElementById('ffVal').value);
    if(isNaN(tff)){ alert('请输入前馈力矩（单位：N·m，转子侧）'); return; }
    const j=lastJoints.find(x=>x.name===name);
    const cur=(j&&j.t_ff!==undefined&&j.t_ff!==null)?parseFloat(j.t_ff).toFixed(3):'0';
    if(confirm('把 '+name+' 的前馈力矩设为 '+tff.toFixed(3)+' N·m（转子侧）？\n\n当前值：'+cur+' N·m\n\n会写回 stand_config.json；站立时按抬起进度 0→该值 施加、保持阶段维持。\n\n⚠ 符号要与“顶住自重”的方向一致；先用较小值试，改完需重新编译 motor_ctrl。'))
      api('set_feedforward',{name,tff});
  }
  // 单关节微调：在保持位上对所选关节加/减步长（≤5°）。客户端 2s 冷却 + 后端 2s 安全间隔双保险。
  let jogCooldownUntil = 0;       // 客户端冷却截止时间戳(ms)
  function jog(sign){
    const name=document.getElementById('jogJoint').value;
    let deg=parseFloat(document.getElementById('jogDeg').value);
    if(isNaN(deg)||deg<=0) deg=5;
    deg=Math.min(10, deg);
    api('jog', {name, dir: sign>0?'+':'-', deg});
    jogCooldownUntil = Date.now() + 1000;   // 1s 安全间隔，期间禁用按钮并倒计时
    document.getElementById('btnJogPlus').disabled = true;
    document.getElementById('btnJogMinus').disabled = true;
    tickJogCooldown();
  }
  function tickJogCooldown(){
    const remain = jogCooldownUntil - Date.now();
    const hint=document.getElementById('jogHint');
    if(remain>0){
      hint.textContent = '安全间隔：'+(remain/1000).toFixed(1)+'s 后可继续';
      setTimeout(tickJogCooldown, 100);
    } else {
      hint.textContent='';   // 按钮可用性由 render() 按 holding 状态恢复
    }
  }
  // 扭矩监控：峰值/历史、设置持久化、柱状图 + 力矩-时间曲线
  const tqHistory={};          // name -> [{t, v(signed 转子N·m, 或 null)}]
  const TQ_CAP=2400;           // 每关节最多样本数（约 16 分钟@400ms，够装满 10min 窗口）
  let lastStandingFlag=false;  // 上一帧是否在运动（用于新动作开始清历史）
  const TQ_COLORS=['#1769d6','#d64545','#2a8a3e','#e08a1e','#8e44ad','#0fb5ae','#c0399b','#7f8c1f','#2c3e50','#c0392b','#16a085','#d35400'];
  function jointColor(name){ const i=lastJoints.findIndex(j=>j.name===name); return TQ_COLORS[((i>=0?i:0)%TQ_COLORS.length)]; }
  function tqIsLive(){ const el=document.getElementById('tqLive'); return el? el.checked : true; }
  function winLabel(w){ return (w<=0)?'全部':(w>=60?((w%60===0)?(w/60+'min'):(w/60).toFixed(1)+'min'):(w+'s')); }
  function resetPeak(){ api('reset_peak'); tqClearHistory(); tqRedraw(); }
  function tqClearHistory(){ for(const k in tqHistory) delete tqHistory[k]; }
  // 控件变化：保存设置并立即重绘（即使在冻结状态，也能换时间窗/关节静态翻看已记录数据）
  function tqOnChange(){ tqSave(); tqRedraw(); }
  function tqRedraw(){ if(lastJoints && lastJoints.length){ drawTorque(lastJoints, lastGear); drawTorqueLine(lastGear); } }
  function tqPush(joints){
    const t=Date.now();
    for(const j of joints){
      const x=parseFloat(j.fb_tau);
      const v=(j.fb_tau===null||j.fb_tau===undefined||isNaN(x))?null:x;
      let arr=tqHistory[j.name]; if(!arr) arr=tqHistory[j.name]=[];
      arr.push({t,v});
      if(arr.length>TQ_CAP) arr.splice(0, arr.length-TQ_CAP);
    }
  }
  function tqSelected(){
    const sel=document.getElementById('tqLineSel').value;
    const names=lastJoints.map(j=>j.name);
    if(sel==='all') return names;
    if(['fl','fr','rl','rr'].includes(sel)) return names.filter(n=>n.split('_')[0]===sel);
    return names.filter(n=>n===sel);
  }
  function tqSave(){
    try{
      localStorage.setItem('tqLimit', document.getElementById('tqLimit').value);
      localStorage.setItem('tqSide', document.getElementById('tqSide').value);
      localStorage.setItem('tqLineSel', document.getElementById('tqLineSel').value);
      localStorage.setItem('tqWin', document.getElementById('tqWin').value);
      localStorage.setItem('tqLive', document.getElementById('tqLive').checked?'1':'0');
    }catch(e){}
  }
  function tqLoad(){
    try{
      const g=(k,id)=>{ const v=localStorage.getItem(k); if(v!==null) document.getElementById(id).value=v; };
      g('tqLimit','tqLimit'); g('tqSide','tqSide'); g('tqLineSel','tqLineSel'); g('tqWin','tqWin');
      const lv=localStorage.getItem('tqLive'); if(lv!==null) document.getElementById('tqLive').checked=(lv==='1');
    }catch(e){}
  }
  const shortName=(n)=>{ const p=n.split('_'); const j={hip:'h',thigh:'t',shank:'s'}[p[1]]||(p[1]||'').slice(0,1); return (p[0]||'').toUpperCase()+j; };
  // 力矩-时间曲线（带符号；±上限红虚线，0 中线）
  function drawTorqueLine(gear){
    const cv=document.getElementById('tqLine'); if(!cv) return;
    const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height; ctx.clearRect(0,0,W,H);
    const side=document.getElementById('tqSide').value; const factor=(side==='rotor')?1:(gear||6.33);
    const L=Math.max(0, parseFloat(document.getElementById('tqLimit').value)||0);
    const winS=parseInt(document.getElementById('tqWin').value);
    const sel=tqSelected();
    const ml=46,mr=12,mt=12,mb=22,plotW=W-ml-mr,plotH=H-mt-mb;
    const now=Date.now();
    let t0;
    if(winS>0){ t0=now-winS*1000; } else { t0=now; for(const n of sel){ const a=tqHistory[n]; if(a&&a.length&&a[0].t<t0)t0=a[0].t; } }
    const t1=now;
    let maxAbs=L>0?L:0.1;
    for(const n of sel){ const a=tqHistory[n]||[]; for(const p of a){ if(p.t>=t0&&p.v!==null){ const x=Math.abs(p.v*factor); if(x>maxAbs)maxAbs=x; } } }
    const ymax=maxAbs*1.15;
    const xOf=t=>ml+plotW*((t-t0)/Math.max(1,(t1-t0)));
    const yOf=v=>mt+plotH*(1-(v+ymax)/(2*ymax));
    ctx.strokeStyle='#ccc'; ctx.beginPath(); ctx.moveTo(ml,mt); ctx.lineTo(ml,mt+plotH); ctx.lineTo(W-mr,mt+plotH); ctx.stroke();
    ctx.strokeStyle='#ddd'; ctx.beginPath(); ctx.moveTo(ml,yOf(0)); ctx.lineTo(W-mr,yOf(0)); ctx.stroke();
    ctx.fillStyle='#999'; ctx.font='10px sans-serif'; ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(ymax.toFixed(1),ml-4,yOf(ymax)); ctx.fillText('0',ml-4,yOf(0)); ctx.fillText((-ymax).toFixed(1),ml-4,yOf(-ymax));
    if(L>0){ ctx.save(); ctx.strokeStyle='#d64545'; ctx.setLineDash([6,4]); for(const sgn of [L,-L]){ const y=yOf(sgn); ctx.beginPath(); ctx.moveTo(ml,y); ctx.lineTo(W-mr,y); ctx.stroke(); } ctx.restore(); }
    sel.forEach((n)=>{
      const arr=tqHistory[n]||[]; ctx.strokeStyle=jointColor(n); ctx.lineWidth=1.5; ctx.beginPath(); let pen=false, prevT=0;
      for(const p of arr){
        if(p.t<t0){ prevT=p.t; continue; }
        if(p.v===null){ pen=false; prevT=p.t; continue; }
        if(pen && (p.t-prevT)>2000) pen=false;   // 间隔>2s（如曾暂停记录）断开，不连虚假直线
        const x=xOf(p.t), y=yOf(p.v*factor);
        if(!pen){ ctx.moveTo(x,y); pen=true; } else ctx.lineTo(x,y);
        prevT=p.t;
      }
      ctx.stroke();
    });
    ctx.lineWidth=1;
    ctx.textAlign='left'; ctx.textBaseline='middle'; ctx.font='11px sans-serif'; let lx=ml+6, ly=mt+9;
    sel.forEach(n=>{ ctx.fillStyle=jointColor(n); ctx.fillRect(lx,ly-5,10,10); ctx.fillStyle='#333'; const lbl=shortName(n); ctx.fillText(lbl,lx+13,ly); lx+=13+ctx.measureText(lbl).width+12; if(lx>W-90){ lx=ml+6; ly+=14; } });
    ctx.fillStyle='#999'; ctx.textAlign='right'; ctx.textBaseline='alphabetic'; ctx.fillText('最近 '+winLabel(winS),W-mr,H-6);
  }
  // 角度跟踪柱状图：每关节 (当前角 − 站姿目标)，关节°，按整圈归一；0=到位
  function drawAngleTrack(joints, gear){
    const cv=document.getElementById('angChart'); if(!cv) return;
    const ctx=cv.getContext('2d'); const W=cv.width,H=cv.height; ctx.clearRect(0,0,W,H);
    const g=gear||6.33, two_pi=2*Math.PI;
    const thr=Math.max(0, parseFloat(document.getElementById('angThr').value)||0);
    const ml=46,mr=12,mt=16,mb=50,plotW=W-ml-mr,plotH=H-mt-mb;
    const n=joints.length||1;
    const rows=joints.map(j=>{
      const cur=parseFloat(j.current), st=parseFloat(j.stand);
      let diffDeg=null,curDeg=null,stDeg=null;
      if(!isNaN(st)) stDeg=st/g*180/Math.PI;
      if(!isNaN(cur)) curDeg=cur/g*180/Math.PI;
      if(!isNaN(cur)&&!isNaN(st)){ let d=cur-st; d=d-Math.round(d/two_pi)*two_pi; diffDeg=d/g*180/Math.PI; }
      return {name:j.name, diffDeg, curDeg, stDeg};
    });
    let maxAbs=thr*1.5||1; rows.forEach(r=>{ if(r.diffDeg!==null){ const a=Math.abs(r.diffDeg); if(a>maxAbs)maxAbs=a; } });
    const ymax=maxAbs*1.1, yOf=v=>mt+plotH*(1-(v+ymax)/(2*ymax)); // 0 居中
    const yZ=yOf(0);
    ctx.strokeStyle='#ccc'; ctx.beginPath(); ctx.moveTo(ml,mt); ctx.lineTo(ml,mt+plotH); ctx.lineTo(W-mr,mt+plotH); ctx.stroke();
    ctx.strokeStyle='#bbb'; ctx.beginPath(); ctx.moveTo(ml,yZ); ctx.lineTo(W-mr,yZ); ctx.stroke();
    ctx.fillStyle='#999'; ctx.font='10px sans-serif'; ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText('+'+ymax.toFixed(0),ml-4,yOf(ymax)); ctx.fillText('0',ml-4,yZ); ctx.fillText('-'+ymax.toFixed(0),ml-4,yOf(-ymax));
    if(thr>0){ ctx.save(); ctx.strokeStyle='#e08a1e'; ctx.setLineDash([5,4]); for(const s of [thr,-thr]){ const y=yOf(s); ctx.beginPath(); ctx.moveTo(ml,y); ctx.lineTo(W-mr,y); ctx.stroke(); } ctx.restore(); }
    const gap=plotW/n, bw=Math.min(46, gap*0.6), flags=[];
    rows.forEach((r,i)=>{
      const x=ml+i*gap+(gap-bw)/2;
      ctx.textAlign='center'; ctx.textBaseline='alphabetic';
      ctx.fillStyle='#555'; ctx.font='11px sans-serif'; ctx.fillText(shortName(r.name), x+bw/2, mt+plotH+15);
      if(r.curDeg!==null&&r.stDeg!==null){ ctx.fillStyle='#999'; ctx.font='9px sans-serif'; ctx.fillText(r.curDeg.toFixed(0)+'/'+r.stDeg.toFixed(0), x+bw/2, mt+plotH+27); }
      if(r.diffDeg===null) return;
      const a=Math.abs(r.diffDeg);
      let col='#2a8a3e'; if(thr>0&&a>2*thr)col='#d64545'; else if(thr>0&&a>thr)col='#e08a1e';
      const yv=yOf(r.diffDeg), top=Math.min(yv,yZ), hgt=Math.max(1,Math.abs(yv-yZ));
      ctx.fillStyle=col; ctx.globalAlpha=0.85; ctx.fillRect(x, top, bw, hgt); ctx.globalAlpha=1;
      ctx.fillStyle=col; ctx.font='bold 10px sans-serif'; ctx.textBaseline='middle'; ctx.fillText((r.diffDeg>=0?'+':'')+r.diffDeg.toFixed(1), x+bw/2, yv+(r.diffDeg>=0?-9:9));
      if(thr>0&&a>thr) flags.push(shortName(r.name)+(r.diffDeg>=0?'+':'')+r.diffDeg.toFixed(0)+'°');
    });
    const vd=document.getElementById('angVerdict');
    if(rows.every(r=>r.diffDeg===null)){ vd.textContent='（点「读取当前 12 关节角」或站立后显示）'; vd.style.color='#888'; }
    else if(flags.length){ vd.textContent='⚠ 偏离站姿: '+flags.join('，'); vd.style.color='#d64545'; }
    else { vd.textContent='✓ 各关节均在 ±'+thr+'° 内到位'; vd.style.color='#2a8a3e'; }
  }
  function drawTorque(joints, gear){
    const cv=document.getElementById('tqChart'); if(!cv) return;
    const ctx=cv.getContext('2d'); const W=cv.width, H=cv.height;
    ctx.clearRect(0,0,W,H);
    const side=document.getElementById('tqSide').value;
    const factor=(side==='rotor')?1:(gear||6.33);
    const L=Math.max(0, parseFloat(document.getElementById('tqLimit').value)||0);
    const ml=46, mr=12, mt=18, mb=44, plotW=W-ml-mr, plotH=H-mt-mb, base=mt+plotH;
    const n=joints.length||1;
    let maxPeak=0;
    const rows=joints.map(j=>{
      const cur=Math.abs(parseFloat(j.fb_tau)||0)*factor;
      let peak=Math.abs(parseFloat(j.tau_peak)||0)*factor;
      if(cur>peak) peak=cur;   // 峰值至少不小于当前（防后端峰值刚清零时的瞬态，避免当前柱高过峰值线）
      const err=Math.abs(parseFloat(j.fb_errd)||0);
      if(peak>maxPeak) maxPeak=peak; if(cur>maxPeak) maxPeak=cur;
      return {name:j.name, cur, peak, err};
    });
    const ymax=Math.max(L*1.15, maxPeak*1.1, 0.001);
    const yOf=v=>mt+plotH*(1-v/ymax);
    // y 轴 + 刻度
    ctx.strokeStyle='#ccc'; ctx.fillStyle='#888'; ctx.font='10px sans-serif'; ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.beginPath(); ctx.moveTo(ml,mt); ctx.lineTo(ml,base); ctx.lineTo(W-mr,base); ctx.stroke();
    for(let t=0;t<=4;t++){ const v=ymax*t/4, y=yOf(v); ctx.strokeStyle='#eee'; ctx.beginPath(); ctx.moveTo(ml,y); ctx.lineTo(W-mr,y); ctx.stroke(); ctx.fillStyle='#999'; ctx.fillText(v.toFixed(1),ml-4,y); }
    // 上限参考虚线
    if(L>0){ const y=yOf(L); ctx.save(); ctx.strokeStyle='#d64545'; ctx.setLineDash([6,4]); ctx.beginPath(); ctx.moveTo(ml,y); ctx.lineTo(W-mr,y); ctx.stroke(); ctx.restore();
      ctx.fillStyle='#d64545'; ctx.textAlign='left'; ctx.fillText('上限 '+L.toFixed(1),ml+4,y-7); }
    const gap=plotW/n, bw=Math.min(46, gap*0.62);
    const flags=[];
    ctx.textAlign='center';
    rows.forEach((r,i)=>{
      const x=ml+i*gap+(gap-bw)/2;
      let col='#2a8a3e';                              // 绿=正常
      if(L>0 && r.peak>=L){ col='#d64545'; }          // 红=峰值超上限
      else if(L>0 && r.peak>=0.8*L){ col='#e08a1e'; } // 橙=接近上限
      // 当前柱
      ctx.fillStyle=col; ctx.globalAlpha=0.85; ctx.fillRect(x, yOf(r.cur), bw, base-yOf(r.cur)); ctx.globalAlpha=1;
      // 峰值横杠 + 峰值数值（红，柱顶上方，醒目）
      if(r.peak>0){ const yp=yOf(r.peak); ctx.strokeStyle='#7a1f1f'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(x-2,yp); ctx.lineTo(x+bw+2,yp); ctx.stroke(); ctx.lineWidth=1;
        ctx.fillStyle='#7a1f1f'; ctx.font='bold 10px sans-serif'; ctx.fillText(r.peak.toFixed(1), x+bw/2, yp-7); }
      // 关节短名
      ctx.fillStyle='#555'; ctx.font='11px sans-serif'; ctx.fillText(shortName(r.name), x+bw/2, base+15);
      // 当前值（灰，名字下方）
      ctx.fillStyle='#999'; ctx.font='10px sans-serif'; ctx.fillText('当前 '+r.cur.toFixed(1), x+bw/2, base+28);
      if(L>0 && r.peak>=0.8*L){
        flags.push(r.name+'峰值'+r.peak.toFixed(1)+(r.peak>=L?('≥上限'+(r.err>=3?'且误差'+r.err.toFixed(0)+'°→扭矩不足':'')):'接近上限'));
      }
    });
    let top={name:'',v:-1}; rows.forEach(r=>{ if(r.peak>top.v) top={name:r.name, v:r.peak}; });
    const topTxt=(top.v>0)?('峰值最大：'+shortName(top.name)+' '+top.v.toFixed(1)+' N·m'):'尚无峰值数据';
    const vd=document.getElementById('tqVerdict');
    if(!L){ vd.textContent=topTxt+'（填入上限参考后给出是否不足的判断）'; vd.style.color='#888'; }
    else if(flags.length){ vd.textContent='⚠ '+flags.join('；')+'　|　'+topTxt; vd.style.color='#d64545'; }
    else { vd.textContent='✓ 各关节峰值均在上限 80% 以内，扭矩裕度充足　|　'+topTxt; vd.style.color='#2a8a3e'; }
  }
  const f=(v,n)=>{ if(v===null||v===undefined) return '—'; const x=parseFloat(v); return isNaN(x)?'—':x.toFixed(n); };
  // 显示用：转子角(rad) -> 关节角(度)。仅换显示单位；校验偏差标红、方向符号等内部判断仍按转子角 rad。
  const jdeg=(v,gear)=>{ if(v===null||v===undefined) return '—'; const x=parseFloat(v); return isNaN(x)?'—':(x/(gear||6.33)*180/Math.PI).toFixed(2); };
  // 增益(K_P/K_W)显示：整数补一位小数(2 -> 2.0)，其余原样；非数字返回 null
  const gfmt=(v)=>{ const x=parseFloat(v); return isNaN(x)?null:(Number.isInteger(x)?x.toFixed(1):String(x)); };
  // 通用数值范围校验：按输入框自身的 min/max 判定（K_P/K_W 0~25.599、前馈 ±8、角度 1~30 等）。
  // 输入中越界即标红提示；离开输入框自动夹到边界并把框里的数改成真实生效值。
  const _rng=(el)=>({lo: el.min!==''?parseFloat(el.min):-Infinity, hi: el.max!==''?parseFloat(el.max):Infinity});
  const _rngTxt=(lo,hi)=>(isFinite(lo)?lo:'')+'~'+(isFinite(hi)?hi:'');
  function boundInput(el){   // 输入中：越界（含非数字）即标红，提示离开后会被夹
    const s=el.value.trim();
    if(s===''){ el.style.borderColor=''; el.title=''; return; }
    const v=parseFloat(s); const {lo,hi}=_rng(el);
    const bad = isNaN(v) || v<lo || v>hi;
    el.style.borderColor = bad ? 'var(--bad)' : '';
    el.title = bad ? ('超出范围 '+_rngTxt(lo,hi)+'；离开输入框将自动夹到边界，当前填的值不会按原值生效。') : '';
  }
  function boundClamp(el){   // 失焦/change：夹到 [min,max]，把框里的数改成真实生效值
    const s=el.value.trim();
    if(s===''){ el.style.borderColor=''; el.title=''; return; }   // 空=用配置默认，不动
    const v=parseFloat(s);
    if(isNaN(v)){ el.style.borderColor='var(--bad)'; el.title='不是数字'; return; }
    const {lo,hi}=_rng(el); const c=Math.max(lo, Math.min(hi, v));
    if(c!==v){
      el.value = Number.isInteger(c) ? c : parseFloat(c.toFixed(3));
      el.style.borderColor='var(--warn)';
      el.title='已自动夹到 '+el.value+'（范围 '+_rngTxt(lo,hi)+'）—— 这才是实际生效值。';
    } else { el.style.borderColor=''; el.title=''; }
  }

  function render(s){
    document.getElementById('phasePill').textContent = '阶段: ' + s.phase;
    const driving = !!(s.holding || s.standing);
    document.getElementById('driveState').innerHTML = driving ? '<span class="bigico">●</span> 驱动中' : '<span class="bigico">●</span> 未驱动';
    document.getElementById('driveBar').classList.toggle('on', driving);
    document.getElementById('driveEdge').classList.toggle('on', driving);
    const cp=document.getElementById('cfgPill');
    cp.textContent = '配置: ' + (s.configured?'已标定':'未标定');
    cp.className = 'pill ' + (s.configured?'ok':'warn');
    document.getElementById('pronePill').textContent = '趴姿: ' + (s.has_prone?'已记录':'未记录');
    document.getElementById('standPill').textContent = '站姿: ' + (s.has_stand?'已记录':'未记录');
    document.getElementById('btnStand').disabled = !s.configured || s.busy || s.standing;
    // 坐下功能暂时禁用（代码有问题）：始终保持禁用，不随状态恢复
    document.getElementById('btnSit').disabled = true;
    document.getElementById('btnTestLeg').disabled = !s.configured || s.busy || s.standing;
    document.getElementById('btnDirTest').disabled = !s.configured || s.busy || s.standing;
    document.getElementById('status').textContent = s.status;

    // 单关节微调：按钮始终可点（仅 1s 安全间隔内短暂禁用）；未保持时点击由后端返回明确提示，
    // 避免「点了没反应、也不知道为什么」。
    const inCooldown = Date.now() < jogCooldownUntil;
    document.getElementById('btnJogPlus').disabled = inCooldown;
    document.getElementById('btnJogMinus').disabled = inCooldown;
    document.getElementById('jogTip').textContent = s.holding
      ? '电机保持中，可微调；每次≤10°、间隔1s。'
      : '⚠ 当前未在锁位保持：请先站立 / 单腿测试到位（电机保持）后再微调，否则点击会被拒绝。';

    // K_P/K_W 留空时实际生效的是配置 execute 里的默认值；把默认值显示到输入框灰字 placeholder
    if (s.execute){
      lastExecute = s.execute;
      const pk=gfmt(s.execute.k_p), pw=gfmt(s.execute.k_w);
      document.getElementById('legKp').placeholder = (pk!==null) ? ('默认 '+pk) : '配置';
      document.getElementById('legKw').placeholder = (pw!==null) ? ('默认 '+pw) : '配置';
      // 整机站立各输入框的 placeholder = 当前配置默认值（缺省值与后端 _stand_impl 一致）
      const e=s.execute;
      const ph=(id,v,d,n)=>{ const x=parseFloat(v); document.getElementById(id).placeholder='默认 '+(isNaN(x)?d.toFixed(n):x.toFixed(n)); };
      ph('stKp', e.k_p, 8.0, 1);
      ph('stKw', e.k_w, 0.1, 2);
      ph('stVmax', e.max_joint_vel_rad_s, 0.1, 3);
      ph('stMinDur', e.min_duration_s, 1.0, 2);
      ph('stRate', e.rate_hz, 100, 0);
      ph('stThr', e.verify_threshold_rad, 0.20, 2);
      ph('stTff', e.t_ff, 0.0, 2);
    }

    // 是否松开：后端按 servo 进程是否存活判定（holding=电机仍带电锁位）
    const holding = !!s.holding;
    const hp=document.getElementById('holdPill');
    if (holding && s.standing){ hp.textContent='松开状态: 🔒 电机运行中'; hp.className='pill warn'; }
    else if (holding){ hp.textContent='松开状态: 🔒 保持锁位（未松开）'; hp.className='pill warn'; }
    else { hp.textContent='松开状态: 🔓 已松开'; hp.className='pill ok'; }
    hp.title = holding ? ('电机仍带电锁位'+(s.hold_scope?('：'+s.hold_scope):'')+'；点「急停/松开」释放')
                       : '电机自由、未上电锁位';
    // 动作到位但电机仍锁位（未松开）时，给一条醒目横幅 + 就地「松开」按钮
    const hb=document.getElementById('holdBanner');
    if (holding && !s.standing){
      hb.className='banner hold';
      hb.innerHTML='🔒 动作已到位，电机<b>仍在锁位保持</b>'+(s.hold_scope?('（'+s.hold_scope+'）'):'')
        +'，<b>尚未松开</b>。检查完毕后请点 '
        +'<button class="mini" style="margin-left:6px" onclick="api(\'estop\')">🔓 松开电机</button>';
    } else {
      hb.className='banner'; hb.innerHTML='';
    }

    const thr = (s.execute && s.execute.verify_threshold_rad) || 0.2;  // 转子角 rad，与 j.dev 同坐标，仅用于标红
    const gear = s.gear_ratio || 6.33;
    lastGear = gear;
    if (s.joints && s.joints.length){
      lastJoints = s.joints;
      if (!poseInit){ poseInit = true; fillPoseCurrent(); fillFfCurrent(); }  // 首次拿到关节数据时预填输入框
    }
    const tb=document.getElementById('jbody');
    if(!s.joints||!s.joints.length){ } else {
      tb.innerHTML='';
      for(const j of s.joints){
        const tr=document.createElement('tr');
        const devBad = (j.dev!==null&&j.dev!==undefined&&j.dev>thr);
        const dir = (j.delta===null||j.delta===undefined)?'—':(j.delta>=0?'↑ +':'↓ −');
        const okMark = (j.ok===false)?' <span style="color:var(--bad)">⚠</span>':'';
        tr.innerHTML =
          '<td>'+j.name+'</td><td>'+j.port.replace('/dev/','')+'</td><td>'+j.id+'</td>'+
          '<td>'+jdeg(j.current,gear)+okMark+'</td><td>'+jdeg(j.prone,gear)+'</td><td>'+jdeg(j.stand,gear)+'</td>'+
          '<td>'+(j.delta>=0?'+':'')+jdeg(j.delta,gear)+'</td><td>'+dir+'</td>'+
          '<td class="'+(devBad?'bad':'')+'">'+jdeg(j.dev,gear)+'</td>'+
          '<td>'+f(j.fb_vel,3)+'</td><td>'+f(j.fb_tau,3)+'</td><td>'+f(j.fb_errd,2)+'</td>'+
          '<td>'+f(j.t_ff,3)+'</td>'+
          '<td>'+
          (j.verified
            ? '<span style="color:var(--ok)">✅</span> <button class="mini" title="撤销确认，恢复待验证" onclick="api(\'unconfirm_dir\',{name:\''+j.name+'\'})">↩取消确认</button>'
            : '<button class="mini" title="方向正确，标记已确认" onclick="api(\'confirm_dir\',{name:\''+j.name+'\'})">✓确认</button>')+
          ' <button class="mini" title="方向反了，翻转该电机方向并存盘" onclick="confirmInvert(\''+j.name+'\')">⇄取反</button>'+
          '</td>';
        tb.appendChild(tr);
      }
    }
    const wb=document.getElementById('warnBanner');
    if(s.phase==='ERROR'){ wb.className='banner bad'; wb.textContent='⚠ 上次站立执行出错，请查看日志并急停后重试。'; }
    else { wb.className='banner'; wb.textContent=''; }

    if(lastJoints && lastJoints.length){       // 扭矩监控：柱状图 + 时间曲线
      const live=tqIsLive();
      document.getElementById('tqLiveState').textContent = live ? '' : '⏸ 已冻结（关闭了实时显示）';
      if(live){
        if(s.standing && !lastStandingFlag) tqClearHistory();   // 新动作开始清历史
        lastStandingFlag = s.standing;
        tqPush(lastJoints);
        drawTorque(lastJoints, gear);
        drawTorqueLine(gear);
      } else {
        lastStandingFlag = s.standing;   // 冻结：不记录、不刷新，保留画面供静态查看
      }
      drawAngleTrack(lastJoints, gear);  // 角度跟踪图（实时，不受扭矩“实时显示”开关影响）
    }
  }

  async function poll(){
    try{
      const r=await fetch('/api/state?since='+logIndex);
      const s=await r.json();
      const log=document.getElementById('log');
      if(s.log&&s.log.length){
        const atBottom = log.scrollTop+log.clientHeight >= log.scrollHeight-24;
        log.textContent += s.log.join('\n')+'\n';
        if(atBottom) log.scrollTop=log.scrollHeight;
      }
      logIndex=s.log_index;
      // 后台线程的拒绝/异常通过 notice 通道弹窗（首次轮询只记基线，避免回放旧通知）
      const nid = s.notice ? s.notice.id : 0;
      if (lastNoticeId === null) lastNoticeId = nid;
      else if (s.notice && nid !== lastNoticeId){ lastNoticeId = nid; alert(s.notice.text); }
      render(s);
    }catch(e){}
  }
  tqLoad();   // 恢复上次的力矩上限参考 / 显示侧设置
  poll();
  setInterval(poll, 400);
</script>
</body>
</html>
"""


# ----------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            q = parse_qs(u.query)
            try:
                since = int(q.get("since", ["0"])[0])
            except (ValueError, TypeError):
                since = 0
            self._json(CTRL.snapshot(since))
        elif u.path == "/api/config":
            self._json(CTRL.config or {})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/run":
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                p = json.loads(body.decode("utf-8") or "{}")
            except Exception:
                p = {}
            ok, msg = CTRL.run(p.get("action", ""), p)
            self._json({"ok": ok, "message": msg})
        else:
            self._json({"error": "not found"}, 404)


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    check_env()
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"四足操控网页已启动:  http://{host}:{port}")
    if host not in ("127.0.0.1", "localhost"):
        print("⚠️  正在监听非本机地址，局域网内任何人都能操控机器人，请注意安全。")
    print("按 Ctrl+C 退出。")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止 ...")
        CTRL._estop()
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
