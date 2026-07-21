# 🦿 控制栈：运动学 · 步态 · 动力学 · 域随机化 · Sim2Real

> 本文档汇总 mos2026_2 在「纯 RL 端到端」之外补齐的**经典运控层**与 **sim2real 必需项**。
> 这些模块是 RL 策略上真机、做传统控制 baseline、以及执行器选型的基础。
>
> 相关文档：[动力学+减速比选型](./dynamics_gear_ratio_analysis.md) ｜ [GO-M8010-6 电机](./8010-6motor.md)
> 任务清单与进度见根目录 [`todo.md`](../../todo.md)。

---

## 0. 模块总览

| 模块 | 文件 | 依赖 | 验证 |
|---|---|---|---|
| **闭链腿 FK/IK/Jacobian** ⭐ | `传统步态/代码/closed_chain_kin.py` | numpy | `--selftest` |
| 足端轨迹 trot 步态（全向） | `传统步态/代码/gait.py` | numpy, matplotlib | `--selftest` / `--demo` |
| **MuJoCo 无 RL 行走** ⭐ | `传统步态/代码/walk_gait.py` | numpy, mujoco | `--sweep` |
| 动力学 + 减速比选型 | `传统步态/代码/dynamics.py` | numpy, matplotlib | `--plot` |
| 腿部串联近似 FK/IK（已退役） | `传统步态/代码/kinematics.py` | numpy | `--selftest` |
| 域随机化 / 观测噪声 | `source/.../mos2026_2_closed_usd_env_cfg.py`（`EventCfg`）、`..._env.py` | Isaac Lab | 冒烟 √ |
| 力矩惩罚奖励 | `source/.../custom_rewards.py` | Isaac Lab | — |
| Policy 导出（TorchScript+ONNX） | `deploy/real/policy_export.py` | torch, onnx(可选 onnxruntime) | 数值一致性门 |

> `deploy/common/*` 不依赖 Isaac/torch，纯 numpy，可在任意机器用 `.venv/bin/python` 直接跑。
> 改 `source/` 下训练代码需在 Isaac 环境（`env_isaaclab`）运行。

---

## 1. 腿部正/逆运动学（FK/IK）

> ⭐ **2026-07-21 起改用精确闭链模型 `传统步态/代码/closed_chain_kin.py`**，并已在
> MuJoCo 里 trot 走起来（达成率 88%，go2 基准 82%）。完整技术实现思路见
> [闭链运动学与 MuJoCo 行走](./闭链运动学与MuJoCo行走.md)。下面的串联近似
> （`kinematics.py`）已退役——它的 foot 长度是猜的、闭链传动比从未标定，实际无法部署。

`传统步态/代码/kinematics.py`（已退役，保留作对照）

每条腿抽象为标准 3-DOF 串联腿（与 Unitree/MIT-Cheetah 同构）：髋外摆 `q_ab`（绕 x）+
大腿 `q_hip`（绕 y）+ 膝 `q_knee`（绕 y）。几何取自 `deploy/mujoco/assets/mos2026_2.xml`：
**L_thigh = 0.180 m**（实测），L_shank ≈ 0.16 m（站立几何估计，待 foot site 标定）；
hip 外摆轴 FL/FR=−x、RL/RR=+x；俯仰轴 左腿=−y、右腿=+y。

- **FK** `leg_fk / quad_fk`：约定角 → 足端 (x,y,z)（髋系/base 系），numpy 向量化覆盖 4 腿。
- **IK** `leg_ik / quad_ik`：足端 → 约定角，解析解（外摆 + 矢状面 2 连杆），含可达性 clamp、
  `knee_sign` 分支选择。

```bash
.venv/bin/python 传统步态/代码/kinematics.py --selftest
# 足端 FK(IK) 往返 1e-16；关节角往返 3e-15；零位 FK 与 XML 累计偏移精确一致
```

> ⚠️ **闭链**：膝由平行四连杆驱动（MuJoCo 里 actuator 实际驱动 `*_shank_link`，真实
> `*_shank` 经 equality 闭环跟随）。本模块输出「约定角」，下发 sim/真机前需做每关节仿射映射
> `q_motor = sign·q + offset` 并标定「shank 电机轴↔等效膝角」传动（见 todo.md §E）。

---

## 2. 足端轨迹步态（trot）

`传统步态/代码/gait.py`

对角小跑：FL+RR 同相、FR+RL 反相。在**足端笛卡尔空间**规划——支撑相贴地直线 +
摆动相**摆线（cycloid）抬腿**（离地/触地瞬间水平与竖直速度均为 0，少打滑），经 IK 转关节目标。

```bash
.venv/bin/python 传统步态/代码/gait.py --selftest          # 足端往返 1e-16、膝角 1.40<1.57 限位、对角相位正确
.venv/bin/python 传统步态/代码/gait.py --demo              # → 传统步态/图与数据/gait_demo/ 关节曲线 + 足端轨迹 + CSV
.venv/bin/python 传统步态/代码/gait.py --linkage --leg fl  # 矢状面 2 连杆（大腿+小腿，髋固定）+ 足端轨迹
```

产物（`outputs/` 已 gitignore，需本地生成）：
- `foot_trajectory.png` 足端矢状面轨迹（教科书 D 形）
- `joint_targets.png` 4 腿关节角曲线（对角同相可见）
- `leg_linkage.png` 髋固定、仅大腿+小腿连杆，多相位叠画 + 足端轨迹
- `joint_targets.csv` 时间 × 12 关节约定角

---

## 2.5 机身速度 ↔ 电机角速度映射（speed_map）

`传统步态/代码/speed_map.py` ｜ 可视化 `scripts/tools/speed_viz_isaac.py`

回答「四足要走 v m/s，每个电机得转多快？」。**核心：不是唯一对应**——同一机身速度，
大步幅低步频（电机慢）或小步幅高步频（电机快）都能实现，步幅/占空比/步频是旋钮。

物理链条（均在 `--selftest` 自洽验证）：
1. 无打滑约束：`v_body = step_length / (duty · period)`，给定 v 反解步频。
2. 足端速度剖面：对步态轨迹解析求导（支撑相 vx=−v；摆动相摆线/(1−cos) 导数）。
3. 足端速度 → 关节角速度：解析 **Jacobian** `q̇ = J(q)⁻¹·foot_vel`（`kinematics.leg_jacobian`）。
4. 关节 → 电机：`ω_motor = N · q̇`，**N = 6.33**。

两条速度上限：训练软限 `velocity_limit_sim = 15 rad/s`、电机物理空载 `30 rad/s`（关节侧）。

```bash
# 单点：3 m/s 需要多少电机转速 + 可行性 + 需要多大步幅才进限位
.venv/bin/python 传统步态/代码/speed_map.py --speed 3.0
# 扫描出「机身速度→电机转速」曲线 + (速度×步幅)→峰值 设计图 + CSV
.venv/bin/python 传统步态/代码/speed_map.py --sweep      # → 传统步态/图与数据/speed_map/
.venv/bin/python 传统步态/代码/speed_map.py --selftest
```

**关键结论**（步幅 10cm / β=0.5 默认 trot）：
- **1.0 m/s** 峰值关节角速度 14.8 rad/s ≈ 897 rpm，恰好贴训练软限 15 —— 与策略在
  `commanded_lin_vel_xy=(1.0,0)` 下训练、`velocity_limit_sim=15` 完全吻合。
- 最大可行机身速度：≤ 训练软限 **1.04 m/s**、≤ 电机物理 **2.02 m/s**。
- **3.0 m/s** 需求峰值 44.5 rad/s ≈ 2691 rpm，远超上限；即便优化步幅到 ~23cm（再大就
  超足端可达域、腿伸直奇异反而更糟），最低也要 ~30 rad/s —— **3 m/s 基本到了此机
  trot 的电机物理天花板**，要更快得加大占空比/换步态或提高电机转速余量。

> ⚠️ 膝为平行四连杆闭链：输出的是「等效膝关节」角速度，shank 电机轴 ≈ ×连杆传动比
> （≈1，未标定，见 §1 与 todo §E）。hip/thigh 直驱、映射干净。

### IsaacSim 实时可视化（运动学播放 + HUD）

`scripts/tools/speed_viz_isaac.py`：在 IsaacSim 里让机器人按**可配置机身速度**前进、腿按
trot 循环（运动学播放：关重力 + 步进物理解闭环），omni.ui 实时 HUD 显示机身速度 +
12 电机角速度条（rad/s/rpm）+ **峰值保持** + 15/30 限位，并复用速度箭头 marker。
关节角速度由 `speed_map` 解析给出（权威值），sim 机器人是可视化载体。

```bash
# 用 IsaacLab 环境（注意是 env_isaaclab，不是纯 numpy 的 .venv）
~/code/RL/env_isaaclab/bin/python scripts/tools/speed_viz_isaac.py --speed 1.0
~/code/RL/env_isaaclab/bin/python scripts/tools/speed_viz_isaac.py --ramp --v-max 3.0  # 周期扫速度
# 无显示器/CI 验证（HUD 退化为控制台表）：加 --headless --num-steps 60
```
单环境 + 平地，PhysX buffer 已按 8GB 笔记本收紧；实测 RTX 4060 Laptop 峰值显存 ~2.6 GB。

---

## 3. 动力学 + 减速比选型

`传统步态/代码/dynamics.py` ｜ 详见 [dynamics_gear_ratio_analysis.md](./dynamics_gear_ratio_analysis.md)

- 连杆受力：矢状面静力 Newton-Euler 递推（足端地反力 → 小腿 → 大腿 → 髋）。
- 关节力矩需求：静立/trot/动态蹬地，**峰值 ≈ 12 N·m**。
- 齿轮传动力：转子力矩、行星齿面切向力、膝连杆传动力。
- 减速比寻优：可行带 [3.56, 10.66]，**余量平衡最优 N\*=6.16**。

```bash
.venv/bin/python 传统步态/代码/dynamics.py --plot   # → 传统步态/图与数据/dynamics/{tn_envelope,gear_margin}.png
```

**核心结论**：现减速比 **6.33 已接近最优**；真机「力矩/电流不足」根因**不是减速比**，
而是 `effort_limit_sim=12` 卡需求线零余量 + 真机驱动器电流上限/母线掉压。建议 effort 放到 16–18 N·m。

---

## 4. 域随机化 / 观测噪声（Sim2Real）

`source/.../mos2026_2_closed_usd_env_cfg.py` 的 `EventCfg` + `..._env.py` 的观测噪声。

| 类别 | 内容 |
|---|---|
| startup | 随机地面摩擦、base ±质量、腿 ±20% 质量 |
| reset | 随机 Kp/Kd ±20%、关节零位偏置 ±0.05 rad |
| interval | 周期推搡 ±0.5 m/s |
| 观测 | 高斯噪声 `obs_noise_std` |

```bash
# 训练时开启（默认关闭，eval 保持「干净测量」不受影响）
python scripts/rsl_rl/train.py --task ... --domain_rand --obs_noise_std 0.01
```

> 开关：`train.py --domain_rand` / `--obs_noise_std`。已在 env_isaaclab + RTX5090 冒烟通过
> （EventManager 加载 6 term、各 body/joint 正则命中真实 USD、reward 正常上升）。

---

## 5. 力矩惩罚奖励

`source/.../custom_rewards.py` 增加 `sum(τ²)`（取 `applied_torque` 的 12 受控关节）。
权重 `reward_scales["torque"]` 默认 **0.0（opt-in，不改变现有训练）**，建议起步 −2e-4，
重训后用 `eval_plot.py` 对比 `near_limit_frac` / CoT。目的：消除策略「贴力矩上限硬走」。

---

## 6. Policy 导出（TorchScript + ONNX）

`deploy/real/policy_export.py`：从 rsl_rl checkpoint 重建 actor，导出 TorchScript + ONNX 双格式，
带 **onnxruntime vs torch 数值一致性门**（`max|Δ| < 1e-4`）。产物 `deploy/real/policy/policy.{pt,onnx}`，
真机 `rl_deploy.py` 用 `jit.load` 推理。

---

## 运行环境备忘

- `deploy/common/*`：根目录 `.venv`（仅 numpy/matplotlib），命令用 `.venv/bin/python`。
- `source/` 训练 + 导出：Isaac 环境 `env_isaaclab`（torch/Isaac Lab/ONNX，RTX5090）。
