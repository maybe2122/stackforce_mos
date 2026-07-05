# 2026-07-05 训练改进五项与 checkpoint 兼容性

- **日期**:2026-07-05
- **背景**:Mos-One 训练效果不如 robot_lab(Go2)。对比分析结论:PPO 超参两边几乎一致,差距在环境设计——缺非对称 critic、奖励惩罚主导、无课程机制、NaN 坏样本污染训练。
- **改动文件**:
  - `source/mos_one/mos_one/tasks/direct/mos2026_2_closed_usd/mos2026_2_closed_usd_env.py`
  - `source/mos_one/mos_one/tasks/direct/mos2026_2_closed_usd/mos2026_2_closed_usd_env_cfg.py`
  - `scripts/rsl_rl/train.py` / `eval.py` / `play.py`

## 五项改动

### 1. 非对称 critic(特权观测)

- cfg:`state_space` 0 → **66**。critic 观测 = 干净版 45 维 policy obs(无噪声)+ base 相对地形高度 (1) + 足端相对地形高度 (4) + 足端接触状态 (4) + 受控关节实际力矩 `applied_torque` (12)。
- env:`_get_observations` 输出 `"critic"` 键;观测噪声只加 policy obs。
- train/eval/play 的 `to_compatible_rsl_rl_cfg` 新增 `has_critic_obs` 参数:检测到特权观测时 `obs_groups` 的 critic 指向 `["critic"]` 组,否则退回 `["policy"]`(旧行为)。

### 2. 正向步态奖励(移植自 robot_lab)

- `feet_air_time`(权重 **+0.25**):脚落地那步结算 `(腾空时长 − feet_air_time_threshold)`,阈值 0.4s。奖励迈出有腾空期的步子,拖地滑步拿不到分。
- `feet_gait`(权重 **+1.0**):对角脚 (FL,RR)/(FR,RL) 接触状态一致且前左/前右反相时每步 +1,正向奖励 trot。robot_lab Go2 权重 0.5 / track 3.0,此处 track 6.0 等比放大。
- 补丁式惩罚同步减半:`gait_symmetry` -0.5 → **-0.25**,`anti_bound` -1.0 → **-0.5**。思路:好步态由正向项引导,坏步态惩罚只做兜底,避免惩罚主导压制探索。

### 3. NaN 样本按 truncation 处理

- `_get_dones`:NaN/Inf 状态从 termination 改为 **truncation(time_out)**。闭链 PhysX 爆炸是模拟器数值问题不是策略失败,不应给终止惩罚信号;现在 PPO 按超时做价值 bootstrap。reward 置零逻辑保留(`_get_rewards`)。

### 4. 命令速度课程

- cfg 新增 `command_curriculum_steps = 30000`(env 步数口径,约默认训练量 5000 iter × 24 steps 的 1/4)、`command_curriculum_start_scale = 0.4`。
- 命令从 0.4 倍线性升到 1.0 倍;地形课程的期望距离按缩放后命令计算(否则低速期全员被误判降级)。
- eval.py / play.py 自动设 `command_curriculum_steps = 0`,按全速命令评估/回放。
- 注意:resume 训练时 `common_step_counter` 从 0 重新计数,课程会重新爬坡一次。

### 5. 评估口径与 robot_lab 统一

- eval.py 新增指标:`lin_err`(水平速度向量误差 m/s)、`mean_action_rate`(相邻两步动作差)、`mean_foot_slip_speed`(触地足端滑移速度,线性 m/s)。
- 除 JSON 外额外输出 `<tag>.md`:表头/指标名/指标说明表与 robot_lab `scripts/reinforcement_learning/rsl_rl/eval.py` 的报告完全一致,可直接并排对比两个项目。

## ⚠️ checkpoint 兼容性(必读)

**2026-07-05 之前训练的旧 checkpoint(对称 critic,45 维)用 eval.py / play.py 回放时必须加 `--no_privileged_critic`**,否则新建的 66 维 critic 网络与旧权重维度不匹配、`runner.load` 直接报错:

```bash
# 旧 checkpoint 回放/评估
python scripts/rsl_rl/play.py --no_privileged_critic --load_run <旧run>
python scripts/rsl_rl/eval.py --no_privileged_critic --load_run <旧run> --tag baseline_old

# 新 checkpoint 正常用,不加参数
python scripts/rsl_rl/eval.py --load_run <新run> --tag baseline_new
```

## 下一步操作

1. **冒烟测试**:`python scripts/rsl_rl/train.py --num_envs 16 --max_iterations 20`,确认非对称 critic / 新奖励项 / 课程能跑通;
2. **放量重训**(默认参数即可);
3. **新旧对比**:
   ```bash
   python scripts/rsl_rl/eval.py --load_run <新run> --tag baseline_new
   python scripts/rsl_rl/eval.py --no_privileged_critic --load_run <旧run> --tag baseline_old
   ```
   对比统一报告(`<checkpoint目录>/eval/<tag>.md`)里的 **lin_err / fall_rate / CoT / foot_slip**;
4. **注意**:奖励定义变了(新增正向项、惩罚权重调整),**新旧 run 的 reward 曲线不可比**,只看任务指标;SwanLab/TensorBoard 里新增 `Episode_Reward/feet_air_time`、`feet_gait` 两条曲线可用于确认正向步态奖励在生效(应从 0 逐步上升)。
