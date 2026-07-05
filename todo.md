# 🐾 Mos 开发流程

> 总体路线：**建模 → 仿真 → 控制 → RL 平地 → 地形泛化 → 动态/Hybrid Internal Model → Sim2Real → 真机**
>

> 📌 **本文件 = 向前看的计划**。已完成的能力里程碑见 [`CHANGELOG.md`](CHANGELOG.md)；
> 每次训练/调参的因果记录见 [`doc/experiments/EXPERIMENTS.md`](doc/experiments/EXPERIMENTS.md)。
> 三层文档关系见 [`doc/version_tracking.md`](doc/version_tracking.md)。
> 下方"下一步"开干时，去台账立一个 `EXP-XXXX`（可用 `python3 tools/exp/log_run.py`）。

---

## 📍 当前状态（更新于 2026-07-05）

⚠️ **2026-07-04 MuJoCo 部署测试实锤了两件事**（详见 `deploy/real/README.md` 修复记录、
`deploy/mujoco/README.md` 测试结论）：
- **增益失配是硬阻塞**：部署增益（转子 8.0 ≈ 关节侧 320）下站立正常，但按 kp=25 训练的
  策略接管后 0.12 s 摔；训练增益 kp=25 下 MuJoCo 软闭链撑不住自重、四连杆翻折叠分支。
  → 里程碑 2 重训必须**连增益一起改**。
- **步态本身也有病理**：对角同步率 0.16~0.30（无 trot 结构）、步频 5~9 Hz 高频颤步、
  无 air time 奖励——整改对照表见 [`doc/步态评价指标.md`](doc/步态评价指标.md) §六。
- 同批修复：`rl_deploy.py` 增益语义 bug（rl_kp=25 被当转子侧 K_P 下发，40 倍刚度）、
  电机过温/merr 急停、torque_limits 12→16 漂移；`play_mujoco.py` 新增受限观测/部署
  增益/settle/action-lpf/步态报告，部署条件仿真闸门（里程碑 3）的通路已就绪。

✅ **全链路环境冒烟通过（2026-06-11 修复改名遗留后）**：
- 改名 `mos_one` 后 env_isaaclab 里残留旧包 editable 安装、新包未装，
  训练/播放/评估全挂 → 已 ``pip install -e source/mos_one --no-deps`。
  **以后再改包名/挪目录，必须在 env_isaaclab 重装**（详见 README「闭链 USD 注意事项」）。
- `third_party/rl_sar` 子模块此前为空目录 → 已 `git submodule update --init` 恢复。
- 逐项验证可跑：Isaac 任务注册（`list_envs.py`）✓、`zero_agent.py` 50 步 ✓、
  HIM `him/train.py` 2-iter 训练+存档 ✓、`play_mujoco.py`（新旧 ckpt 格式均可，见下）✓、
  `standup_torque.py` 出图 ✓、`deploy/common/*.py --selftest`（.venv）✓。
- 尚不能在本机验证的只剩**真机硬件链路**（`deploy/real/rl_deploy.py`、
  `motor_control` 网页工具）——需要 4 路 USB-RS485 + GO-M8010-6 实机。

✅ **四足真机已能站起来**（从趴姿安全启动 → 撑起到站姿，见 [站立标定/方向验证网页工具]）。

⚠️ **主要瓶颈：力矩 / 电流不足**
- 站立维持或动态动作时，电机输出力矩（电流）达不到所需值，表现为：撑起吃力 / 站姿下垂 / 抗扰差。
- 整机质量 **≈ 11.70 kg**（USD 标称：base 4.52 kg + 单腿 ≈ 1.80 kg ×4），站立时单腿需承担约 1/4 整机重量的静态保持力矩。
- 需要确认是「电机/减速比选型不够」还是「驱动器电流上限设置过低 / 供电不足」。

🔧 已有调试工具：电机调试网页（前馈力矩、扭矩监控、角度跟踪）、`scripts/rsl_rl/play.py`（力矩统计 CSV / 曲线 / TensorBoard 实时曲线）。

---

## 🛤️ 真机跑起来·关键路径（主线｜更新于 2026-06-11）

> **唯一目标：让四足在现实中真正走起来。** 按依赖顺序排 4 个里程碑,每个有明确
> 「通过标准」,不达标不进下一步。不在这条主线上的工作(地形课程、eval 打磨、
> mjlab、HIM 收敛)一律放到「跑起来之后」。
>
> **前提认知(代码审计已确认)**:现有所有 checkpoint **结构上不可部署**——
> obs 含线速度真值(真机没有)、固定单指令(对 `--cmd_vx` 零响应)、零域随机化。
> 所以第一步不是部署,而是**重训一版可部署策略**;旧 checkpoint 不要再花部署功夫。

### 里程碑 1:硬件先决条件(可与训练并行)

- [ ] **力矩/电流余量**(现状:站立都吃力,不解决一切免谈;详见 §力矩专项)
  - [ ] 按「先软后硬」排查:驱动器电流上限是否设低 → 电池大负载下是否掉压限流
    → Kt/减速比换算核对(选型分析已排除减速比,大概率前两个)
  - ✅ 通过标准:站立维持各关节实测力矩 ≤ 额定 ~60%,侧推能恢复
- [ ] **IMU 接入**(`rl_deploy.py` 现为 `imu_source="stub"`,ang_vel=0、
  gravity=[0,0,-1] —— **没有 IMU 策略就是盲的,物理上不可能平衡**,最高优先硬件项)
  - [ ] 在 `_build_obs` 加真实 IMU 分支(serial/udp)
  - ✅ 通过标准:吊起手动晃机身,ang_vel 与重力投影的方向/单位/坐标系逐项核对正确

### 里程碑 2:重训一版「可部署」策略(GPU,1–2 周)

一次重训合并以下改动(各自骨架都已落地,从没合在一起训过):

- [ ] **actor 盲化**:obs 去掉 `root_lin_vel_b` 真值;指令 vx/vy/wz 进 obs(指令条件化)
- [ ] **路线**:先用「盲 actor + 非对称 critic」最小改动版,**不要先上 HIM**
  (HIM 从未收敛过,新算法调试与真机调试两个变量不要叠加;HIM 留作走稳后的升级)
- [ ] **域随机化放量**:`--domain_rand --obs_noise_std 0.02`(冒烟已过)
- [~] **力矩三件套**(对应真机「小腿满载、蹲低硬走」):☑️ 配置已落地(2026-06-12,
  16-env 冒烟过、reward 中生效):`reward_scales["torque"]=-2e-4` + `effort_limit_sim`
  12→16;同批落地 `armature=0.01`(反射转子惯量,取自 menagerie go2 同款执行器)与
  `foot_contact_height_threshold` 0.07→0.15(foot_slip 此前从未生效)。
  剩余:抬高站姿目标(base 0.256→0.32,先看力矩惩罚重训后是否自然改善)+ **重训验证**。
- [ ] **可部署增益(2026-07-04 实测确认为硬阻塞)**:训练 PD 25/0.5 vs 真机实际下发
  ≈320/3.2(关节侧等效),二者必须统一——要么训练改用可部署增益,要么先做单关节
  chirp 标定确认真机能稳定实现的最软增益再定训练值(见 §运控缺口 2)
- [ ] **步态整形奖励**:加 feet air time(目标 0.2~0.3 s)、`action_rate` -0.005→-0.02、
  `joint_vel` -0.0001→-0.001(症状→旋钮对照表见 `doc/步态评价指标.md` §六)
- [ ] **obs 合约三处同步对账**:训练 env / `policy_export.py` 的 `OBS_DIM` /
  `rl_deploy.py` 的 `_build_obs`,逐字段(顺序、单位、scale)对齐
- ✅ 通过标准:`eval.py` 平地 0 跌倒;对不同 `--cmd_vx` 有明确响应;
  `near_limit_frac` 显著下降;推搡/摩擦/±质量扰动下存活率不崩

### 里程碑 3:MuJoCo 受限观测闸门(上真机前最后一道门)

- [~] 把 `play_mujoco.py` 的「假 sim2real」(直接读 `data.qvel` 喂真值)改成
  **只用真机拿得到的量**跑策略
  - ☑️ 受限观测通路已落地(2026-07-04):`--lin-vel-source zero --imu-source stub`
    + `--kp/--kv` 部署增益 + `[gait]` 步态报告
  - [ ] 剩余:直接加载导出的 `policy.pt`(现仍走 checkpoint 重建);接估计器后
    加 `--lin-vel-source estimator` 分支
- ⚠️ 已知限制(2026-07-04):训练增益 kp=25 下 MuJoCo 软闭链会翻四连杆折叠分支,
  **无法在训练增益下验证**;此闸门必须用部署增益跑,或先修闭链建模(见 §运控缺口)
- ✅ 通过标准:受限观测 + 部署增益下在 MuJoCo 走稳。**这步过不了真机一定过不了;
  过了,真机问题就收敛到硬件标定**

### 里程碑 4:渐进上机(每步可回退,严格按序)

1. [ ] **`sim_sign` 逐关节实机验证**(yaml 全是默认值 1,从未验证——AI 写的配置里
   最典型的「看着对、没验过」;用 robot_web 单腿/方向验证工具)
2. [ ] 吊线 + `--no_rl`:站姿下 `q_sim ≈ 0`、IMU 读数正确
3. [ ] 吊线 + RL 低增益:空中做出行走动作、不抽搐
4. [ ] 落地有人扶 + 零速指令站立 → 小速度(0.3 m/s)直线 → 逐步放开

### ⚠️ 给「大量 AI 实现代码」的验证原则

接口合约必须**逐项物理验证**,不能用信任代替:关节顺序、方向符号、单位(rad/rot)、
action_scale、obs 排布——sim2real 死掉的案例 90% 死在这些约定上,不在算法上。
仓库已备好验证工具:`--no_rl`、robot_web 单腿/方向验证、各模块 `--selftest`。

### 📦 非主线(跑起来之后再做)

- [ ] MuJoCo FK 对齐(§E 收尾):装 mujoco + XML 加 foot site,FK vs `mj_forward`
  对比;标定 `L_shank` 与闭链传动映射
- [ ] `play_mujoco.py` 直接加载 `policy_export.py` 产出的 `policy.pt`/`.onnx`,
  做导出产物 sim-to-sim 回归(新旧 ckpt 格式兼容、结构自动推导已于 2026-06-11 完成)
- [ ] 训练侧 actuator 延迟/带宽(EventTerm 或 env 内缓冲),配合 `eval.py
  --action_delay` 形成闭环
- [ ] 地形课程启用、HIM 收敛模型、mjlab 第二后端

---

## 🎯 后续工作（力矩/电流不足专项）

> 目标：定位力矩/电流缺口的根因，先让站立稳定且有余量，再谈行走。

- [~] **量化需求 vs 实际**
  - [ ] 用 `play.py` 力矩统计 + 电机网页扭矩监控，记录站立/支撑相各关节**实测力矩与电流**
  - [x] 用整机质量(11.70 kg)和几何，估算各关节**理论静态保持力矩**（2026-06-09，
    `deploy/common/dynamics.py` + `doc/dynamics_gear_ratio_analysis.md`）：静立 4 腿
    τ_knee≈2.2 / trot 2 腿≈4.5 / 动态蹬地 τ_hip≈12 / 深蹲(0.256)≈14 N·m。
  - [ ] 确认仿真里 `play_torque_stats.csv` 的 |max| / rms 是否超过电机额定/峰值力矩
- [ ] **驱动链路排查（先软后硬）**
  - [ ] 检查驱动器**电流上限 / 力矩上限**参数是否被设低，逐步上调到电机额定
  - [ ] 检查**供电电压/电流能力**（电池/电源是否在大负载下掉压限流）
  - [ ] 核对**减速比 / 力矩常数 Kt** 标定值，确认下发力矩→实际力矩换算正确
- [ ] **控制侧补偿**
  - [ ] 站立加**重力前馈**（gravity compensation），减少纯 PD 的力矩负担
  - [ ] 复核站立目标姿态，避免大力臂的高耗能站姿
- [x] **选型评估**（2026-06-09，`doc/dynamics_gear_ratio_analysis.md`）：
  - 结论：**减速比 6.33 已接近最优**（余量平衡最优 N\*=6.16；可行带 [3.56,10.66]；
    力矩余量 1.78× ≈ 转速余量 1.68×）。**力矩不足的根因不是减速比**，而是
    `effort_limit_sim=12` 卡在需求线零余量 + 真机驱动器电流上限/母线掉压。
  - 行动项：(1) 仿真 effort_limit 放到 **16–18 N·m**（≤电机峰值 23.7）；
    (2) 真机核查驱动器电流上限与大负载下母线电压；(3) 若主打中低速/抗扰可上调到 N≈8。
  - [x] 在仿真里对齐真实电机的 actuator 力矩-速度曲线（T-N 曲线已建模并按 N 反射到关节）。
- [ ] **闭环验证**：上调后重测站立维持 + 抗扰（侧推），确认有力矩余量再推进行走

### 📊 评估发现（2026-06-07，`eval.py` / `model_750.pt` 平地 1.0 m/s）

> 工具：`scripts/rsl_rl/{eval.py, eval_report.py, eval_plot.py}`，结果见 `logs/.../eval/`。
> 结论：策略**站立稳、速度跟踪近乎完美（0% 跌倒、vx 0.99/1.0）**，但**靠蹲低 + 小腿电机近乎满载**换来——这正是真机"力矩/电流不足"在仿真侧的同源表现。

- [~] **加力矩惩罚项**（根因）：当前 `reward_scales` 无任何 torque penalty，PPO 没有省力矩动机 → 学出贴上限硬走。
  - 仿真实测：`fl/rl/rr_shank` 有 **83~84%** 时间 ≥90% 上限（12 N·m），`rl_thigh` 66%，整体近上限占比 40%，CoT 高达 **4.5**、功率 515 W。
  - ☑️ 已落地（2026-06-08）：`custom_rewards.py` 增加 `sum(τ²)` 项（取 `applied_torque`
    的 12 受控关节，nan_to_num+clamp），`reward_scales["torque"]` 默认 **0.0（opt-in，
    不改变现有训练）**。
  - [ ] **待办**：把权重设为 −2e-4 起步重训，用 `eval_plot.py` 对比 `near_limit_frac`/
    CoT 后微调（需 Isaac/GPU）。
- [ ] **抬高站姿**：base 高度只有 **0.256**（目标 0.32，低 20%），蹲低直接抬高膝/踝力矩需求 → 调大 `base_height` 权重或核对目标可达性。
- [ ] **修接触阈值**：`foot_contact_height_threshold=0.07` 太低，shank body 中心始终高于阈值 → **`foot_slip` 奖励训练时大概率从未生效**，步态/打滑指标也全退化。抬到 ~0.15（评估时可用 `eval.py --foot_contact_height 0.15` 验证）。
- [~] **速度-力矩扫描**（2026-06-09 跑了，**结论：当前策略做不了这条扫描**）：
  用 `eval.py --cmd_vx {0.3,0.6,1.0,1.3}`（model_750）实测——4 个指令下**实际 vx 恒为 0.991、
  力矩/CoT/功率/饱和关节全部逐位相同**（near_limit 40.4%、CoT 4.49、功率 515W、`fl/rl/rr_shank`
  恒 ≥80% 饱和）。只有 `vx_mae` 随指令变（0.3→0.70 / 1.0→0.04 / 1.3→0.31），证明 `cmd_vx`
  确实进了「跟踪误差参考」但**完全没进 policy 行为**——即 §A「指令不进 observation、固定单指令
  训练」的直接后果：**该策略只有单一速度，无速度可控性**。
  - → 要刻画「力矩随速度恶化」定真机速度上限，必须先把**指令喂进 obs 重训指令条件化策略**
    （或每个速度各训一个）。在那之前这条扫描无意义。
  - 🔧 配套交付：旧 checkpoint 是 `actor/critic_state_dict` 格式（老 rsl_rl 训练），env_isaaclab
    新 rsl_rl 的 `runner.load` 只认 `model_state_dict` → `tools/ckpt/convert_checkpoint_to_rsl_rl.py`
    做纯键名重映射转换（已验证转换后 eval 指标与原始逐位一致），解锁旧模型在当前环境 eval/play。

---

## 🧩 代码审计缺口（更新于 2026-06-08）

> 基于对 `source/`、`him/`、`deploy/`、`motor_control/` 的实际代码核查（非路线图层面）。
> 区别于上面的力矩专项：这里是**结构性缺口**——不补，策略「能在仿真里跑」但「上不了真机」。
> 优先级总原则：先结硬件根因（力矩专项）→ 训练侧补 sim2real 必需项 → 定线（stock vs HIM）→ 导出+校验 → 真机控制器。

### A. 仿真训练侧（最致命：训练设定本身不为真机服务）

- [~] **域随机化 / 扰动注入 = 0（最高优先级）**
  - 全仓 `grep` 不到任何 `EventTermCfg / randomize_rigid_body / push / apply_external_force`，连骨架都没有。
  - 质量、质心、摩擦、关节 friction/damping、Kp/Kd 全是单一标称值；观测零噪声（`_get_observations` 直接 `cat` 真值）；无 push 外推、无 actuator 延迟/带宽。
  - ☑️ 已落地骨架（2026-06-08）：`env_cfg.py` 新增 `EventCfg`（startup 随机摩擦/base±质量/
    腿±20%质量；reset 随机 Kp/Kd±20%/关节零位±0.05rad；interval 周期推搡 ±0.5 m/s）；
    `_get_observations` 加观测高斯噪声（`obs_noise_std`）；`train.py` 加 `--domain_rand`
    / `--obs_noise_std` 开关（**默认关闭，不污染 eval 的干净测量**）。
  - [x] **冒烟验证通过（2026-06-08，env_isaaclab + RTX5090）**：`--num_envs 16
    --max_iterations 5 --domain_rand --obs_noise_std 0.01` 跑通，`TRAINING_COMPLETED`。
    EventManager 成功加载全部 6 term（startup: physics_material/add_base_mass/scale_leg_mass；
    reset: actuator_gains/reset_joint_bias；interval: push_robot）——即所有 body/joint 正则
    （`base`、`.*(thigh|shank).*` 匹配 `*_shank_link_a`、`.*`）都命中真实 USD，**无
    SceneEntityCfg 报错**；各 mdp 事件函数签名与本机 Isaac Lab 2.3.2 **全部兼容**（无签名错）。
    reward -0.52→0.24 正常上升，obs 噪声生效。下一步可放量正式训练对比鲁棒性。
  - [ ] **未覆盖项**：actuator 延迟/带宽、控制延迟（`eval.py --action_delay` 已有评估侧，
    训练侧仍缺），可后续作为 EventTerm 或 env 内缓冲实现。
- [ ] **stock PPO 观测含特权信息 `root_lin_vel_b`（结构性不可部署）**
  - `observation_space=45` 前 3 维是机身线速度真值，真机**没有传感器直接给**（需 IMU+里程做状态估计）。
  - `state_space=0` → critic 也对称，无法用特权信息训练估计器。
  - → stock PPO 这条线结构上无法部署，必须二选一（见 §B 定线）。
- [ ] **无观测历史 / 时序建模**（stock）：单帧 MLP 对真机噪声很脆，缺历史去滤波/估速。HIM 有，stock 没有。
- [ ] **地形泛化基本未启用**：`terrain_curriculum_enabled=False`，默认 `plane`；rough/stairs 地形 cfg 写了但未纳入课程。对应 Phase 4 实质未做。

### B. 路线定线：stock PPO vs HIM（需先决策）

- [ ] **决策：放弃 stock PPO 转 HIM，还是给 stock 加非对称 critic + 速度估计器？**
  - `him/`（盲 actor + 6 步历史 270 维 + HIM 速度估计器 + 特权 critic 51 维）是 sim2real 正解，但**未见收敛模型/验证产物**。
  - [ ] 把 HIM 跑出第一个收敛模型；在「受限观测」MuJoCo 下验证估速精度（不再喂 `qvel` 真值）。

### C. Sim2Real 中间层（几乎空白）

- [x] **policy 导出管线**（`deploy/real/policy_export.py`）：从 rsl_rl checkpoint 重建
  `[256,256,128]`-ELU actor（丢弃 std，部署确定性均值），导出 **TorchScript + ONNX** 双格式。
  - ☑️ TorchScript（2026-06-08 前已有）：`jit.trace`+`freeze`，scripted vs eager `max|Δ|=0`。
  - ☑️ ONNX（2026-06-08 新增）：`torch.onnx.export`（opset17，batch 动态轴）+ `onnx.checker`
    结构校验 + **onnxruntime vs torch 数值一致性门**（256 随机样本播种，`max|Δ|=1.14e-05`
    < 1e-4 容差；onnxruntime 缺失则降级为仅结构校验并提示）。产物 `deploy/real/policy/policy.{pt,onnx}`。
  - 注：`play_mujoco.py` 仍走 `torch.load`+硬编码重建（旧路径），真机 `rl_deploy.py` 已用
    `jit.load`。后续可让 `play_mujoco.py` 也复用导出产物，彻底消除「结构一变静默错」。
- [ ] **MuJoCo 部署是「假」sim2real**：从 `data.qvel` 直接读机身速度喂 obs（继续用特权量），只验证了网络数值复现，没验证「真机拿不到的量怎么办」。
  - → 出一版「真机同款受限观测 + 估计器」的 MuJoCo 闭环。
- [ ] **缺真机 obs 构建 / 单位对齐文档**：IMU 系→机体系旋转；关节方向/零位（注意编码器掉电丢整圈的隐患）；关节顺序映射（`joint_map.default.json` 未与 obs 的 12 维顺序对账）。

### D. 实机部署侧

- [x] **`deploy/real/rl_deploy.py`（2026-06-08 完成）**：50 Hz 确定性控制进程，
  连接 robot_web 已有的 motor_ctrl servo 接口，实现完整的 policy → 真机链路。
  - 关节状态从 servo FB 行异步读取（motor_ctrl 已重编，FB 默认 20ms = 50 Hz）
  - 45 维 obs 按训练合约组建；坐标转换（旋子↔sim）由 `JointMeta` 封装
  - 动作 clip + action_scale 与训练 env 对齐；prev_action 正确维护
  - 安全：`--max_pitch_roll` 姿态超限急停（imu stub 阶段不生效）；Ctrl-C 干净退出
  - `--no_rl` 模式可在不跑 policy 情况下验证 obs 读取正确性
- [ ] **IMU 接入**（当前 `imu_source="stub"`）：
  - ang_vel=0、gravity_vec=[0,0,-1]，policy 降级运行（建议先吊线/低增益验证）
  - → 有真实 IMU 后在 `rl_deploy.py` 的 `_build_obs` 里扩展 imu_source 分支
- [ ] **线速度估计**（`lin_vel_source="zero"`）：obs[0:3]=0 → policy 降级，行走性能受限
  - → 最终方案：重训时去掉 lin_vel 或接入 HIM 估计器
- [ ] **首次上机流程**：先吊线 + `--no_rl` 验证 obs 读数（q_sim ≈ 0 在站姿），再切 RL 低速测试

### E. 运动学（FK/IK）缺口（更新于 2026-06-08）

> 全仓 `grep` 不到任何 `forward_kinematic / inverse_kinematic / jacobian / IK`：
> **本工程此前完全没有运动学层**。纯 RL 端到端（policy 直接出 12 个关节目标）
> 确实可以不用 FK/IK，但缺它会卡住三件事，且这正是「机器人运控」岗位的核心考点：
>
> 1. **Phase 2 传统控制 baseline 做不了**——手写 gait 需要在“足端轨迹空间”规划
>    （抬腿/前摆/落地是足端的笛卡尔轨迹），再用 **IK** 转成关节目标。没有 IK
>    只能在关节空间硬凑正弦，既不直观也无法对齐真机步幅/步高。
> 2. **足端接触/打滑判定不准**——当前 `eval.py` 和 reward 用 *shank body 高度*
>    近似触地（见 §力矩专项「修接触阈值」）。有 **FK** 就能算真实足端点高度/速度，
>    把 `foot_slip`、`duty_factor`、触地判据建立在足端而非连杆中心上。
> 3. **里程计 / 状态估计缺一块**——足端 FK + 接触相位是腿式里程计（leg odometry）
>    估机身线速度的标准做法，正好对应 §C「线速度估计 lin_vel_source=zero」的降级问题。
>
> ⚠️ **闭链特殊性**：本机器人膝关节由平行四连杆驱动（MuJoCo 里 actuator 实际驱动
> `*_shank_link`，真实 `*_shank` 经 `equality/connect` 闭环跟随）。所以「电机轴角 →
> 等效膝关节角」存在一个**连杆传动关系**，FK/IK 工作在“等效 3-DOF 串联腿”
> （hip 外摆 + thigh + 等效 knee）抽象上，电机↔等效膝角的映射需单独标定。

- [x] **足端正运动学 FK**（2026-06-08，`deploy/common/kinematics.py`）：
  `(q_ab, q_hip, q_knee) → 足端 (x,y,z)`（hip/base 系），numpy 向量化覆盖 4 条腿。
  几何取自 `deploy/mujoco/assets/mos2026_2.xml`：L_thigh = 0.180 m；hip 外摆轴
  FL/FR=−x、RL/RR=+x；俯仰轴 左腿=−y、右腿=+y。
- [x] **足端逆运动学 IK**（同上）：`足端 (x,y,z) → (q_ab, q_hip, q_knee)`，解析解
  （外摆 + 矢状面 2 连杆），含可达性 clamp 与 knee_sign 分支选择。
- [x] **自洽验证**（`kinematics.py --selftest`）：足端 `FK(IK(p))` 往返 **1e-16**、
  关节角往返 **3e-15**；零位 FK 与 XML 累计偏移精确一致（err=0）。
- [ ] **对齐 MuJoCo（待装 mujoco）**：随机关节角下 FK 足端 vs `mj_forward` 的足端
  body `xpos` 对比，确认与权威模型数值一致（当前 `.venv` 未装 mujoco，
  无 foot site，需先在 XML 加 foot site 再比）。
- [ ] **L_shank / 足端点标定**：XML 无 foot site，shank→足端长度暂按站立几何估计
  （L1+L2 ≈ 0.34 ⇒ L_shank ≈ 0.16 m），需用 foot site 或真机实测标定。
- [ ] **闭链传动标定**：标定「shank 电机轴角 ↔ 等效膝关节角」映射 + 约定角→sim/motor
  的仿射映射（sign·q+offset），使 IK 解能直接下发到 `rl_deploy.py` / robot_web。

---

## 🦿 运控技术栈缺口（2026-07-05 全仓分析）

> 视角:**运控算法工程师的完整技术栈**。区别于「代码审计缺口」(合约/结构问题),
> 这里是**层级性空缺**:本仓库「RL 训练栈很厚,经典运控栈很薄」——奖励设计/评估/
> sim2sim/实验管理已相当专业,但 RL 之下应该托底的三层(状态估计、执行器建模、
> 传统控制 baseline)基本为空,且互相放大:没估计器 → obs 有缺口;没执行器模型 →
> 增益失配;没 baseline → 出问题分不清硬件还是策略。
> 能力地图对照:`doc/运控算法工程师_能力地图` 第 10 章(状态估计)/第 11 章(部署
> 实时)标红区与此完全一致;地图本身还缺一章「MPC/WBC」(见缺口 3)。

### 缺口 1:状态估计层(最大空白,直接阻塞主线)

> 现状:`imu_source="stub"`、`lin_vel_source="zero"`,仓库里没有任何一个估计器。

- [ ] **IMU 驱动 + 姿态融合**(互补滤波或 Mahony/EKF,~200 行;即里程碑 1 的 IMU 项,
  但要的不只是"读到数",是融合出干净的 ang_vel + 重力投影)
- [ ] **腿式里程计**:足端 FK + 接触相位 → 机身线速度。这是 lin_vel 缺口的**经典解**,
  目前仓库只押注 HIM 一条路,两条路应该都有
  - 可复用件:`deploy/common/kinematics.py` FK 已验证到 1e-16,就差接上
  - ✅ 通过标准:MuJoCo 里 `--lin-vel-source zero` + 估计器 vs 真值,平地误差 <0.1 m/s
- [ ] **接触估计**(无足底传感器 → 力矩残差/广义动量观测器判触地;FB 已回传每关节 τ)
- 建议顺序:互补滤波 → 腿式里程计 → 在 MuJoCo 受限观测闸门里对照真值验证精度

### 缺口 2:执行器建模与系统辨识(「增益失配」的正解在这层)

> 训练用理想 PD,真机是 6.33 减速 + 转子侧 PD + 摩擦 + 齿隙 + 带宽限制,
> 目前两边只靠一个 ×6.33² 换算公式对齐,零实测标定。

- [ ] **单关节 PD 频响标定(最值得先做,半天)**:用 `motor_ctrl servo` 发 chirp/阶跃,
  实测跟踪带宽与等效关节刚度,验证 `kp×6.33²` 换算是否成立
  - → 直接决定里程碑 2 重训该用什么增益,不做这个实验增益取值就是拍脑袋
- [ ] **执行器模型进训练**:一阶延迟 + 力矩带宽 + 摩擦;或用真机 chirp 数据训
  actuator net(Hwangbo 2019 做法)
- [ ] **闭链传动标定**(§E 已列,归属这层):shank 电机轴角 ↔ 等效膝角实测映射
- [ ] **整机 sysid**:称重核对 USD 标称 11.70 kg、质心/惯量粗校

### 缺口 3:传统控制 baseline(运控工程师的「对照组」缺失)

> `gait.py` trot 轨迹 + IK 都自测通过但从未闭环;真机站立是开环插值,无主动平衡。
> 没有 baseline:出问题无法二分「硬件 vs 策略」、RL 失败无回退模式。

- [~] **最小闭环:IMU + 姿态 PD 主动站立平衡**(按机身倾角微调四足高度)
  - 价值极高:同时验证 IMU 驱动、50 Hz 实时环、坐标系约定、安全急停——
    是 RL 部署前的完美脚手架
  - ☑️ MuJoCo 版已落地(2026-07-05,`deploy/mujoco/stand_balance.py`):投影重力
    PD → 每腿足端高度修正 → 闭链数值雅可比 → thigh/曲柄目标。控制律、抗抖
    三件套、增益标定实验、推搡数据、工程发现(曲柄雅可比奇异 / kp_att>0.3
    起振边界)全部整理在 **[`doc/站立平衡控制器.md`](doc/站立平衡控制器.md)**。
  - [ ] 真机版(M2):IMU 到货后把同一控制律搬进 rl_deploy 底层(把仿真投影重力
    换成姿态融合输出即可)
  - ✅ 通过标准:真机站立侧推 5°,1 s 内回平
- [ ] **慢速 trot baseline**:把 `gait.py` 轨迹经闭链映射闭环跑起来(Phase 2 收尾)
- [ ] **知识层**:能力地图补「MPC(convex MPC/MIT Cheetah 系)、WBC、ZMP/capture
  point」一章——仓库可以没有 MPC,运控工程师的知识栈里不能没有

### 缺口 4:实时系统与数据记录(上机就会暴露)

- [ ] **黑匣子(优先,几十行)**:`rl_deploy.py` 每帧落盘 obs/action/FB 全量时序
  (npz/csv)——现在真机跑 RL 不留任何数据,摔一次等于白摔
- [ ] **实时性度量**:控制环 jitter 统计、FB 丢帧率统计(50 Hz 靠 sleep 定频,
  只有超时打印,没有数据证明链路余量)
- [ ] **安全层补口**:
  - [ ] 无线急停(现在急停靠键盘空格,机器人走起来人追不上)
  - [ ] 关节软限位(位置指令 clamp 到安全范围再下发)
  - [ ] 跌倒检测 → 阻尼保护模式(检测到摔倒切 kp=0 + 小 kw,别硬顶)
  - [ ] 上电自检序列(逐关节小幅动作 + FB 核对,代替人工逐项检查)
- [ ] (远期)Python 4 子进程文本管道 → `third_party/rl_sar`(C++)迁移,
  子模块已接从未对接

### 缺口 5:命令通道(从遥控器到策略的整条链空白)

- [ ] 手柄/键盘 → 命令速度 (vx, vy, wz) → obs 的部署侧通路(里程碑 2 指令条件化
  重训时**一起打通**,否则新策略到手仍只能直行)
- [ ] `rl_deploy.py` 与 `play_mujoco.py` 各加命令输入接口(MuJoCo 侧可先用键盘)

### 缺口 6:临门一脚(已有 90%,差收尾)

- [ ] **HIM 第一个收敛模型**(框架移植完整,只跑过 2-iter 冒烟)
- [ ] **AMP 落地**(`doc/AMP_集成方案.md` 方案已写)
- [ ] **回归管线串珠**:训练 → 导出 → MuJoCo 受限观测 → 步态报告,每环都有了,
  串成一条命令做 CI 式冒烟——本周发现的 torque_limits 漂移、增益 bug 本质都是
  缺这道闸

### 🎯 如果只挑三件事(都不占 GPU,与重训并行)

1. **IMU 驱动 + 姿态融合 + 主动站立平衡**(缺口 1+3 交集,一切闭环的地基)
2. **单关节 chirp 标定**(缺口 2,半天,直接决定重训增益取值)
3. **rl_deploy 黑匣子**(缺口 4,几十行,上机前必须有)

---

## 🧱 Phase 1：机械建模与仿真环境

📄 [sub/Structural_Design.md](sub/Structural_Design.md) ｜ [sub/urdf.md](sub/urdf.md)

- [x] 结构设计：Body / Legs / Joints / Sensors / Battery
- [ ] 重心、刚性、承载能力评估
- [x] SolidWorks → URDF / USD 导出
- [x] 关节类型 / link 层级 / 惯量参数 检查
- [x] 导入 Isaac Lab / MuJoCo，加载无报错
- [x] 重力、关节限制、无穿模/抖动 验证

---

## 🤖 Phase 2：基础控制（非 RL，先能动）

> 关键：先用传统控制让它「站起来 / 走一步」，作为 RL 的 baseline。
> 依赖 §E 的 FK/IK（足端轨迹 → 关节目标）。

- [ ] **腿部运动学库**（§E）：FK/IK 解析解 + 自洽验证，4 腿向量化。
  ☑️ 已落地 `deploy/common/kinematics.py`（含 `--selftest`），见下方进度。
- [ ] 关节 PD 控制器：`τ = kp·(q*−q) + kd·(dq*−dq)`，每关节稳定无震荡
- [x] **足端轨迹 gait**（2026-06-08，`deploy/common/gait.py`）：对角 trot，足端笛卡尔
  空间规划——支撑相贴地直线 + 摆动相摆线（cycloid）抬腿，经 IK 转关节目标。
  `--selftest` 全绿（足端往返 1e-16、膝角峰值 1.40<1.57 限位、对角相位正确）；
  `--demo` 出 `outputs/gait_demo/` 关节曲线 + 足端轨迹图 + CSV。
  - [ ] **待办**：把约定角经仿射映射下发到 MuJoCo/真机，实际走起来（需 sim/硬件）。
- [ ] 状态观测打通：joint pos/vel、base orientation/vel
- [ ] （可选）足端 FK + 接触相位做腿式里程计，给 §C 的线速度估计兜底

---

## 🧠 Phase 3：强化学习（平地步态）

📄 [sub/Isaac_Lab_env_create.md](sub/Isaac_Lab_env_create.md)

- [ ] 定义 RL 环境
  - obs: `[joint_pos, joint_vel, base_vel]`
  - action: `joint_target / torque / residual`
- [ ] 奖励函数：前进速度、姿态稳定（roll/pitch）、能耗 penalty
- [ ] PPO + GPU 并行训练
- [ ] 在平地收敛，学会前进 + 不摔倒

---

## 🏔️ Phase 4：地形泛化（Curriculum Learning）

- [ ] 难度递增：平地 → 不平地 → 楼梯 / 小障碍   curriculum 
- [ ] 地形观测：height map、foot contact
- [ ] 稳定性奖励增强：slip penalty、contact timing
- [ ] 同一 policy 跨地形保持稳定
> 参考论文：*Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response*
---

## ⚡ Phase 5：高级动态训练（Hybrid Internal Model 风格）

> 引入 simulated robot response，作为 internal model 反馈到 policy。

- [ ] 高阶步态：跳跃、快速转向、急停
- [ ] 奖励扩展：动态稳定性、敏捷性、能耗最优
- [ ] Sim2Real 预备：domain randomization（mass / friction / sensor noise）
- [ ] 加入控制延迟、actuator 动力学偏差

---

## 🔁 Phase 6：Sim2Real 与真实硬件

- [ ] 硬件实物：电机 / 编码器 / IMU / 控制器
- [ ] 实时控制：MCU 或工控机，1 kHz 控制循环
- [ ] 软件部署：policy 推理 + ROS 2 接口
- [ ] 上机测试：低速 → 加速 → 多地形，配急停保护

---

## 🔧 Phase 7：调试与优化

- [ ] 监控 reward 收敛 / 前进能力 / 摔倒次数
- [ ] 高阶任务：楼梯、跳跃、急转
- [ ] 反推调优：PD 基线、奖励项、动作空间
- [ ] 仿真 inertia 与实机标定对齐

---

## 🚀 最小可行版本（MVP）

```text
1. 导入仿真（Phase 1）
2. PD 让它能动（Phase 2）
3. RL 在平地学会向前走（Phase 3）
```

完成这三步就有一个能跑的雏形，其它阶段都是在这之上叠加。

---

## ⚠️ 常见坑

- ❌ 一上来做 Sim2Real → 必崩
- ❌ 没有 PD baseline → RL 学不会
- ❌ reward 太复杂 → 不收敛
- ❌ 模型 inertia 错 → 一切错



先让四足稳定的站起来，确保力矩足够，站起来的时候，可以通过控制单个关节，让机器人做动作，现在站起来的时候，单个电机控制无反馈