# deploy/real — sar_rl 适配进度

> 更新于 2026-07-04

## 2026-07-04 修复记录

- **[修复] rl_deploy.py 增益语义 bug**：原实现把关节侧 `rl_kp=25` 直接当 servo 的
  转子侧 K_P 下发（等效关节刚度 25×6.33²≈1000 N·m/rad，训练值的 40 倍，危险）。
  现改读 `hardware.motor_kp/motor_kw`（默认 8.0/0.08，与 motor_bus.py 一致）。
- **[新增] 电机健康监护**：FB 过温（≥80°C）/ merr≠0 → 急停；关节侧力矩
  （FB tau × 6.33）超 `torque_limits` → 1 Hz 限频告警。
- **[同步] torque_limits 12→16**：训练环境 `effort_limit_sim` 已上调为 16。
- **[MuJoCo 部署测试结论]（play_mujoco.py 新增 --lin-vel-source/--imu-source/--kp/--kv/--settle）**：
  - 训练增益 kp=25 在 MuJoCo 下撑不住 11.7 kg 自重（静差≈τ/kp≈0.3 rad），四连杆
    穿过奇异点翻进折叠装配分支（真机有物理限位不会），无法用于验证策略；
  - 部署增益 kp≈320（=8×6.33²）站立正常（0.266 m），但策略按 kp=25 动力学训练，
    接管后 0.12 s 内摔倒——**增益失配是硬阻塞**，需按可部署增益重训；
  - lin_vel=0 / IMU stub 的受限观测使结果进一步恶化。三种观测配置均无法行走。

---

## 已完成的工作

### 1. `rl_deploy.py` — 50 Hz 主控进程

- **ServoProc 封装**：每路串口启动一个 `motor_ctrl servo` 子进程，后台线程异步读取 FB 行，线程安全缓存最新反馈
- **坐标转换 `JointMeta`**：封装旋子 ↔ sim 双向转换（`stand_rotor + GEAR * dir * sim_sign`），12 个关节元数据从 `stand_config.json` 加载
- **45 维 obs 构建**：`lin_vel(3) + ang_vel(3) + gravity_vec(3) + dof_pos(12) + dof_vel(12) + prev_action(12)`，与训练 env 合约完全一致，clip_obs=100
- **动作管线**：clip(-1.5, 1.5) → action_scale → default_dof_pos → sim_to_rotor，per-joint action_scale（0.5 hip / 0.8145 thigh/shank）与训练对齐
- **安全急停**：姿态超限（默认 0.5 rad ≈ 28°）立即停机；Ctrl-C / SIGTERM 干净退出
- **`--no_rl` 调试模式**：只保持站姿，用于上机前验证 obs 读数
- **精确 50 Hz 定频**：`t_next += CTRL_DT + sleep(slack)` 实现，超时时有警告

### 2. `policy_export.py` — TorchScript 导出

- 重建 `[256, 256, 128]-ELU` actor MLP，丢弃 Gaussian std（确定性部署）
- `torch.jit.trace` + `jit.freeze`，64 轮随机输入数值一致性校验（max|Δ| < 1e-6）
- 自动查找最新 checkpoint，输出路径兼容 rl_sar

### 3. `config/mos2026_2.yaml` — rl_sar 兼容配置

- `mos2026_2:` 块：obs/action 合约、PD 增益（kp=25 / kd=0.5）、action_scale、default_dof_pos、torque_limits 全部与训练 env 对齐
- `hardware:` 块：12 个关节的 `/dev/ttyUSBx` 映射、motor_id、dir、stand_rotor（已从 stand_config.json 导入）

---

## 还需要做的工作

### 近期 / 上机必须

| 项 | 说明 |
|---|---|
| **`sim_sign` 实机验证** | yaml 中所有关节 `sim_sign=1` 是默认值，**未在真机上逐关节确认方向正确**，这是上机前必做步骤 |
| **IMU 接入**（`imu_source="stub"`） | 当前 ang_vel=0，gravity_vec=[0,0,-1]，policy 以降级模式运行，无姿态感知，**不能真正平衡**；需在 `_build_obs` 里扩展真实 IMU 分支（serial/udp） |
| **首次上机流程** | 先吊线 + `--no_rl` 验证 `q_sim ≈ 0`（在站姿时），再切 RL 低增益低速测试 |

### 中期 / 影响部署质量

| 项 | 说明 |
|---|---|
| **线速度估计**（`lin_vel_source="zero"`） | obs[0:3]=0，行走性能受限；根本解法是去掉 lin_vel 重训，或接入 HIM 速度估计器 |
| **域随机化** | 训练侧无任何扰动注入（质量/摩擦/噪声/push），上真机策略泛化能力极弱，上机大概率不稳 |
| **路线定线** | stock PPO 含特权量 lin_vel（真机拿不到），需决策：加非对称 critic + 速度估计器，还是转 `him/` |

### MuJoCo 侧（影响 sim2real 验证闭环）

- MuJoCo 部署仍用 `data.qvel` 直接读机身速度（特权观测），未验证「真机同款受限观测 + 估计器」的闭环

---

## 用法速查

```bash
# 1. 导出 policy（从最新 checkpoint 自动检测）
python deploy/real/policy_export.py

# 2. 指定 checkpoint 导出
python deploy/real/policy_export.py \
    --checkpoint logs/rsl_rl/mos2026_2_closed_usd/2026-05-16_22-22-39/model_1500.pt

# 3. 只保持站姿（不跑 policy，验证 obs 读数）
python deploy/real/rl_deploy.py --no_rl

# 4. 正常 RL 控制（站姿保持 5s 后切 RL）
python deploy/real/rl_deploy.py --hold_secs 5.0

# 5. 调试模式（每帧输出）
python deploy/real/rl_deploy.py --verbose
```

**前置条件**：

1. 机器人已通过 `robot_web.py` 完成趴/站标定，`stand_config.json` 已保存
2. `deploy/real/policy/policy.pt` 已由 `policy_export.py` 导出
3. 各串口 `/dev/ttyUSB0~3` 可访问（或已配置 udev 权限）
