# MuJoCo 部署脚本

在 MuJoCo（`assets/mos2026_2.xml`）里做关节/电机力矩分析与动作可视化。需要 `mujoco`、
`matplotlib`（出图）、`imageio`（导视频）、`swanlab`（可选实时上报），用装了这些的
`env_isaaclab` 跑（下文命令在仓库根目录执行，`../env_isaaclab` 指向同级的环境目录）。

| 脚本 | 作用 |
|:---|:---|
| `standup_torque.py` | 蹲→站插值，逐控制步记录每关节驱动力矩 / 电机负载率，出图 + CSV |
| `play_mujoco.py` | 在 MuJoCo 里播放 / 调试动作 |

---

## 🪑 `standup_torque.py` — 站起来力矩分析

让机器人从蹲姿平滑插值站起来（settle → rise → hold → lower 一个循环），逐控制步记录：

- **关节驱动力矩** `τ_joint = data.actuator_force`（position 伺服实际输出，已含重力 + 惯性 + 地反力）
- **电机转子侧力矩** `τ_rotor = τ_joint /(N·η)`（用 `传统步态/代码/dynamics.py` 的减速比/效率反射）
- **电机负载率** `|τ_joint| / 输出峰值`，超 `effort_limit` / 输出峰值即报警
- 估算相电流 `I = τ_rotor / Kt`（`--kt` 传入，0 则不算）

### 运行

```bash
# 🖥️ 带 GUI 看站起来动作（需真实显示/GPU；显式指定 GL 后端避免开窗失败）
MUJOCO_GL=glx ../env_isaaclab/bin/python deploy/mujoco/standup_torque.py \
    --viewer --settle 1.0 --rise 2.5 --hold 2.0 --lower 2.0

# 🤖 headless 记录 + 出图（无显示环境，结果写 outputs/standup_torque/）
../env_isaaclab/bin/python deploy/mujoco/standup_torque.py

# 🎞️ 无 GPU 窗口时离屏渲染成视频看动作（画面右侧叠加每关节实时力矩条）
MUJOCO_GL=glx ../env_isaaclab/bin/python deploy/mujoco/standup_torque.py \
    --video outputs/standup_torque/standup.mp4

# 🐢 慢放 10×（太快看不清时用；sim 步级抽帧，平滑不卡）
MUJOCO_GL=glx ../env_isaaclab/bin/python deploy/mujoco/standup_torque.py \
    --video outputs/standup_torque/standup_slow10x.mp4 --slowmo 10 \
    --settle 0.3 --rise 2.0 --hold 0.6 --lower 0.6

# 📈 同时实时上报 swanlab（每关节 τ / 电机负载 / 电流）
../env_isaaclab/bin/python deploy/mujoco/standup_torque.py --swanlab --realtime
```

> [!TIP]
> 无真实显示/GPU 的环境里 `--viewer` 开 GL 窗口会被杀（signal 16 / exit 144），改用
> headless 或 `--video`（`--video` 也要带 `MUJOCO_GL=glx DISPLAY=:1` 走离屏渲染）。

> 🎞️ **视频说明**（三面板）：① 左=机器人蹲→站；② 中=12 个关节**瞬时力矩条**（绿 <50% /
> 橙 50–85% / 红 >85% 输出峰值，叠 `effort_limit`12 与电机峰值 23.7 竖线 + 时间戳）；
> ③ 右=**所有关节力矩随时间增长的对比曲线**（色=腿 fl/fr/rl/rr，线粗=类型 hip细/thigh中/
> shank粗，黑竖线=当前时刻，橙线=effort_limit）。`--slowmo N` 把真实 1 秒拉成 ~N 秒；想更慢
> 加大 N，只看上升段就把 `--hold/--lower` 调小。

### 📊 在 SwanLab 里实时查看力矩

`--swanlab` 把每关节 `torque/* / motor_torque/* / load_pct/*`（含 `--kt` 时还有 `current/*`）
和 `base_height` 逐步写到本地 `swanlog_local/`。**本地看板**两步看：

```bash
# 1️⃣ 先起本地看板服务（保持运行；浏览器开 http://127.0.0.1:5092）
../env_isaaclab/bin/python -m swanlab watch swanlog_local

# 2️⃣ 另开一个终端跑脚本，加 --realtime 让曲线按真实节奏逐步长出
../env_isaaclab/bin/python deploy/mujoco/standup_torque.py --swanlab --realtime \
    --rise 5 --hold 3 --lower 4          # 拉长便于盯着看
```

在看板里选 `standup_torque` 项目 → 进 run，能看到：

- **`all_joint_torque`（合并对比图，ECharts）** —— **12 个关节力矩同图**对比：色=12 色板，
  线型=关节类型（hip 实 / thigh 虚 / shank 点）。每 `--swanlab-chart-every`(默认 5) 步带全
  历史重画一次，**cloud 下随步增长**实时长出；local 下刷新后用步滑块拖到最新看全程。
- `torque/* / motor_torque/* / load_pct/*` —— 每关节独立的原生 scalar 图（含 `--kt` 时加 `current/*`）。
- `base_height` —— 机身高度。

（不加 `--realtime` 数据会瞬间灌完，仍能看，只是不是“流式长出”。`--swanlab-chart-every`
越小合并图越流畅、写盘开销越大。）

> [!IMPORTANT]
> **本地看板（`swanlab watch`）不会自动刷新**——它只在打开/刷新页面时从磁盘读一次，所以
> 跑着的 run 你得**手动刷新浏览器**才看到新点。数据本身是逐步写盘的（`--realtime` 控制写入
> 节奏），问题只在本地看板没有向浏览器推流。
>
> **要「图自己流式长出、完全不用刷新」→ 用云端模式**（swanlab.cn 用长连接实时推）：
> ```bash
> ../env_isaaclab/bin/python deploy/mujoco/standup_torque.py \
>     --swanlab --swanlab-mode cloud --realtime --rise 5 --hold 3 --lower 4
> ```
> 首次需登录一次（`swanlab login`，在 https://swanlab.cn 账户设置拿 API Key）；之后云端 run
> 页面自动实时刷新。`--swanlab-mode`：`local`（默认，需手刷）/ `cloud`（真实时）/ `offline`。

> [!NOTE]
> 训练侧（rsl_rl）默认走 `cloud`；本脚本默认 `local` 只写本地、不受登录/外网影响，与训练的
> SwanLab 配置（见根 README「📈 SwanLab 实验跟踪」）互不干扰。

### 输出（`outputs/standup_torque/`）

| 文件 | 内容 |
|:---|:---|
| `torque_curves.png` | 分组(hip/thigh/shank)力矩曲线 + 极限线，base 高度 & 峰值电机负载 |
| `torque_all.png` | **12 个关节力矩叠在一张图**对比（色=腿 fl/fr/rl/rr，线型=类型 hip 实/thigh 虚/shank 点） |
| `torque_series.csv` | 逐控制步全量序列（关节力矩 + 电机转子力矩） |
| `torque_stats.csv` | 每关节 mean/std/min/max/abs_max/rms |
| `*.mp4`（`--video`） | 机器人动作 + 右侧实时力矩条；`--slowmo` 慢放 |

### 常用参数

| 参数 | 默认 | 说明 |
|:---|:---:|:---|
| `--viewer` | off | 启动 GUI 窗口看动作（需真实显示/GPU） |
| `--video <path>` | — | 离屏渲染成视频，右侧叠加力矩条（无 ffmpeg 时退回 GIF） |
| `--slowmo` | `1.0` | 慢放倍数（10 = 比真实慢 10×） |
| `--vid-fps` | `30` | 视频输出帧率 |
| `--swanlab` / `--realtime` | off | 上报 swanlab / 按真实节奏跑让曲线逐步长出 |
| `--settle/--rise/--hold/--lower` | `0.5/1.5/1.0/1.5` | 蹲姿稳定 / 站起 / 站定保持 / 蹲下 各时长 (s) |
| `--stand-thigh/--stand-shank` | `0.0/0.4` | 站姿大腿/小腿物理俯仰 (rad) |
| `--crouch-thigh/--crouch-shank` | `0.6/-0.6` | 蹲姿大腿/小腿物理俯仰 (rad) |
| `--kp` | `0`（用 MJCF 的 25） | 覆盖伺服 kp，调硬看真实跟踪力矩 |
| `--kt` | `0` | 转子力矩常数 N·m/A（>0 才算电流，待标定） |
| `--repeat` | `1` | headless 跑几个循环（viewer 模式忽略，循环到关窗） |

> [!NOTE]
> 本模型膝部是 `<connect>` 闭链，逆动力学不好用，所以直接读 `actuator_force` 作关节净
> 驱动力矩。闭链 shank 无法完全伸直，position 控制下站高上限约 0.26 m。

---

## 🎮 `play_mujoco.py` — 把训练好的策略放到 MuJoCo 上跑

加载 rsl_rl 的 `.pt` checkpoint（**新旧两种格式都认**：老训练的
`actor_state_dict`/`mlp.*` 与 env_isaaclab 新 rsl_rl 的 `model_state_dict`/`actor.*`），
actor MLP 的层结构**直接从权重形状推导**（不再硬编码 `[256,256,128]`，结构不匹配会在
加载时报错而非静默错），**完整复刻 Isaac Lab 的 45 维观测与动作管线**（root 线/角速度、投影重力、
joint_pos_rel、joint_vel、prev_actions；clip 1.5 → `scale * action + default_pos` → position
伺服 kp=25/kv=0.5），50 Hz 控制跑在 1000 Hz 物理上，用来在 MuJoCo 里验证 sim-to-sim 一致性。

> [!IMPORTANT]
> MJCF 需先由 `tools/asset/usd_to_mjcf.py` 生成。闭链腿在 Isaac 是刚性 PhysX 关节，MuJoCo 用
> 软 `<connect>` 等式近似，故用 1000 Hz 小步长让闭链够刚、下肢能承重而不软塌。

### 运行

```bash
# 🖥️ 带 GUI 播放，自动挑 logs/rsl_rl 下最新的 model_*.pt（需真实显示/GPU）
MUJOCO_GL=glx ../env_isaaclab/bin/python deploy/mujoco/play_mujoco.py

# 🎯 指定 checkpoint
../env_isaaclab/bin/python deploy/mujoco/play_mujoco.py \
    --checkpoint logs/rsl_rl/mos2026_2_closed_usd/2026-05-14_11-23-51/model_600.pt

# 🤖 headless 验证（无显示/SSH 下用；必须给 --duration > 0）
../env_isaaclab/bin/python deploy/mujoco/play_mujoco.py --headless --duration 5
```

### 常用参数

| 参数 | 默认 | 说明 |
|:---|:---:|:---|
| `--checkpoint` | 最新 `model_*.pt` | rsl_rl 的 `.pt`（含 `actor_state_dict` 或 `model_state_dict` 均可），不给则自动挑 `logs/rsl_rl` 下最新 |
| `--headless` | off | 不开 viewer，静默跑 N 步做验证（SSH 下用），输出存活/里程/高度/倾角指标 |
| `--duration` | `0` | 跑多少秒；0 = viewer 模式不限时；headless 必须 > 0 |
| `--mjcf` | `assets/mos2026_2.xml` | MJCF 路径 |
| `--lin-vel-source` | `sim` | `zero` = obs[0:3] 喂 0，复刻真机 rl_deploy 无线速度传感器的输入 |
| `--imu-source` | `sim` | `stub` = ang_vel=0、gravity=[0,0,-1]，复刻真机 IMU 未接入的输入 |
| `--kp` / `--kv` | `0`（MJCF 的 25/0.5） | 覆盖执行器 PD；真机 rl_deploy 当前 ≈ 320/3.2（=8/0.08 × 6.33²） |
| `--settle` | `1.0` | 切策略前默认姿态伺服站稳秒数（对齐 rl_deploy 的 hold_secs） |
| `--no-fall-stop` | off | headless 摔倒（高度<0.22 或倾角 cos<0.85，与训练 env 同判据）不提前终止 |

> [!WARNING]
> **2026-07-04 sim2sim 测试结论**：训练增益 kp=25 下 MuJoCo 软闭链撑不住 11.7 kg
> 自重（静差 ≈ τ_gravity/kp ≈ 0.3 rad），四连杆在下沉过程中穿过奇异位形翻进**折叠
> 装配分支**（connect 残差仍 <0.2 mm，属合法第二解；真机有物理限位不会发生），
> 之后即使关节回到默认角机身也只有 ~0.17 m——刚化 solref/solimp、提高仿真频率均
> 无效，被动角活动区间与折叠分支重叠，限位也挡不住。用部署增益 kp=320 站立正常
> （0.266 m 站高上限），但策略是按 kp=25 训练的，接管后立即摔倒（增益失配）。
> 结论：当前 checkpoint 无法通过 MuJoCo sim2sim 验证，需按可部署增益重训。

> [!TIP]
> 无真实显示/GPU 的环境里直接开 viewer 会被杀，改用 `--headless --duration <秒>`。
