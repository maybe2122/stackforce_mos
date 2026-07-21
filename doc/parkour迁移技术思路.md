# Parkour 算法迁移到 Mos-One 技术思路

> 2026-07-20。源:`~/code/rl/parkour/parkour`(Robot Parkour Learning,legged_gym/Isaac Gym + 魔改 rsl_rl 1.0.2);
> 目标:`~/code/rl/Mos-One`(mos2026_2 闭链四足,Isaac Lab 2.3.2 DirectRLEnv)。
> 本文基于对两侧代码的实际核查,不是泛泛的方法论。

---

## 1. 迁移的到底是什么

parkour 项目可拆成四个可独立迁移的部件,价值和成本各不相同:

| 部件 | 在 parkour 仓库里的位置 | 内容 | 迁移价值 |
|---|---|---|---|
| **A. 障碍地形 + 课程** | `legged_gym/utils/terrain/barrier_track.py` | BarrierTrack:jump / leap / hurdle / down / tilted_ramp / stairsup / stairsdown / discrete_rect / slope / wave 十种技能障碍,按 track 串联,难度随课程升级 | ★★★ 核心资产,Mos-One 现有地形只有 rough/楼梯 |
| **B. 感知观测** | `legged_robot.py` 高度采样;蒸馏阶段为深度图 | 231 点(21×11)scandots 高度图,`clip(base_z - 0.5 - terrain_h, ±1) × 5.0` | ★★★ 越障必需,盲策略过不了 jump/leap |
| **C. 网络与算法** | `rsl_rl/modules/encoder_actor_critic.py`、`algorithms/estimator.py` | `EncoderStateAcRecurrent`(MLP 编码 scandots→32 维 + GRU actor)+ `EstimatorPPO`(监督学习一个 lin_vel 估计器,推理时用估计值顶替观测里的真值,`replace_state_prob=1.0`) | ★★ 思想迁移,代码不搬(见 §4) |
| **D. 三阶段流水线** | rough → field(oracle)→ distill(深度相机 DAgger) | 走路策略 → 障碍 oracle 策略(特权观测)→ 蒸馏到真机可得观测 | ★★ 框架照搬,第三阶段受硬件制约(见 §7) |

奖励项(跟踪、能耗、dof 误差、越障 engaging 奖励)和域随机化属于随部件顺带迁移的内容,Mos-One 的 `custom_rewards.py` / `EventCfg` 已有等价框架,做映射即可。

---

## 2. 路线选择:算法搬过来,而不是机器人搬过去

**结论:在 Mos-One 的 Isaac Lab 工程内复刻 parkour 流水线。不要把 mos2026_2 塞进 legged_gym/Isaac Gym。**

理由:

1. **闭链是一票否决项。** Isaac Gym Preview 的 URDF 导入不支持闭运动链;mos2026_2 的四连杆腿依赖 USD 闭链 + PhysX loop joint projection(`_patch_projected_loop_joints`),这套只在 Isaac Lab/Sim 里跑得通。README「闭链 USD 注意事项」也明确不能走 URDF Converter。
2. Isaac Gym 已停止维护,parkour 仓库锁死 Python 3.8 + 旧 torch;Mos-One 的整条 sim2sim(`deploy/mujoco`)→ 真机(`deploy/real` + `motor_control`)管线都在 Isaac Lab 侧,搬机器人过去等于放弃整条部署链。
3. Mos-One 已有的 HIM 路线(`him/`)证明了"在现有 DirectRLEnv 上只换 obs 布局 + policy/algo"的移植模式可行,parkour 迁移沿用同一模式,风险最小。

备选:`mjlab_tasks/`(MuJoCo-warp)如果日后成为主训练路线,本方案的地形/观测/算法设计同样适用,只是 TerrainGenerator 换成 MJX 场景生成。

---

## 3. 闭链四连杆腿 vs go1 串联腿:算法层面差异清单

parkour 原版的所有算法假设都建立在 go1/go2 串联腿(hip–thigh–calf 三关节直连,12 关节 = 12 自由度,URDF 树状结构)之上。mos2026_2 的四连杆闭链腿(USD 闭链 + PhysX loop joint projection:closure joint 以 `excludeFromArticulation=True` 排除出简化坐标 articulation,由求解器当额外约束处理并开投影防锚点漂移)使下面这些假设逐条失效。迁移时每一条都要有对应处理,不处理的后果写在各条末尾。

**① 关节空间 ≠ 自由度空间。**
go1:12 个关节就是全部自由度,obs 的 dof_pos/dof_vel 与动作一一对应、语义直白。
mos2026_2:关节总数远大于 12(每条腿有 `*_shank_link_a/b`、`*_shank_motor_gear`、`*_shank` 等 4 个 shank 相关 body 及被动关节),只有 12 个被驱动(`actuated_joint_names`),被动关节坐标由闭链约束隐式决定。obs/action 只取 actuated 12 维——维度与 go1 相同,但 actuated shank 关节角**不是**足端小腿摆角,中间隔着一层随构型变化的四连杆传动。→ 影响:parkour 中所有以"关节角≈腿构型"为前提的项(dof_pos 惩罚、default 姿态奖励、越障时的姿态 shaping)语义都偏了,权重不能照抄,要在 mos 上重调;连杆振动/间隙等被动关节动态对策略是不可观测的隐动态,只能靠域随机化和历史观测兜。

**② 默认姿态与 reset 随机化受约束。**
go1:`default_joint_angles` 在 config 里手写,legged_gym 的 `_reset_dofs` 还会对关节角乘 0.5~1.5 随机缩放。
mos2026_2:闭链 USD 依赖 authored 被动关节坐标,default 姿态必须从 USD 初始状态捕获(`_capture_usd_default_joint_state`),不能手写;随意随机化关节初始角会违反闭链约束,PhysX 强行装配时产生巨大约束力(炸 NaN 或翻进错误装配分支)。→ 影响:parkour 的 dof 初始随机化必须裁剪成"只随机 actuated 关节的小幅偏移 + 让被动关节由约束求解落位",或干脆只随机 base 状态;`init_at_random_ep_len` 可保留。

**③ 力矩/增益语义差一个构型相关的传动比。**
go1:直驱语义,标量 `action_scale=0.5`、kp=40/kd=1 全关节统一,电机力矩≈关节力矩。
mos2026_2:齿轮减速 6.33 + 四连杆瞬时传动比(姿态的函数),关节侧/转子侧增益差 6.33²≈40 倍(deploy README 已记录一次把关节侧 kp=25 当转子侧下发的危险 bug);`action_scale` 是 per-joint 元组(0.5 hip / 0.8145 thigh/shank)。→ 影响:力矩限幅(`effort_limit_sim=16`)、力矩/能耗惩罚、动作幅度都是"关节侧"量,与电机真实能力之间的映射随姿态变化——跳跃这种大构型变化动作下,同一电机在不同腿型时可用关节力矩不同,力矩余量核算(Phase 0)必须按最不利构型做,不能只看站立姿态。

**④ 控制接口:显式力矩 PD vs 隐式执行器。**
go1(parkour):每个物理步显式算 `τ = kp(0.5a + q_default − q) − kd·q̇` 再 `set_dof_actuation_force_tensor` 下发,力矩裁剪自己做。
mos2026_2:Isaac Lab `ImplicitActuatorCfg`(stiffness=25/damping=0.5),PD 在 PhysX 求解器内部隐式积分——对闭链这是必要的(显式力矩 + 大冲击更容易让约束发散),但也意味着 parkour 里"改 `_compute_torques` 实现动作延迟、电机模型、力矩噪声"这类技巧要改走 Isaac Lab 的 actuator/EventCfg 路径,不能直接搬代码。

**⑤ 约束求解稳定性与装配分支。**
go1:串联树,PhysX 简化坐标求解,无约束漂移、无多解装配问题;jump/leap 落地冲击在原版训练里是常规操作。
mos2026_2:loop joint 是求解器额外约束,大冲击下会漂移(projection 只能缓解)、偶发 NaN(env 已内置 `nan_to_num` + 强制 reset),且四连杆存在奇异点和折叠装配分支——MuJoCo 测试中已实测穿越奇异点翻进折叠分支,真机靠物理限位挡、仿真里没有。→ 影响:parkour 高冲击技能(jump/leap/down)风险被放大,必须:solver 迭代数调高、actuated 关节加软限位奖励把构型约束在奇异点安全侧、落地冲击惩罚常开、NaN reset 率纳入训练监控(它同时是"策略在利用仿真 bug"的报警器)。

**⑥ 接触与足端定义。**
go1:URDF 有独立 foot link,`contact_forces` 直接可读,parkour 的 feet_air_time 奖励、大腿/小腿碰撞终止、`max_contact_force` 异常终止全部基于接触力张量。
mos2026_2:"脚"是 `*_shank_link_a` body(需正则精确匹配,否则会把电机齿轮/被动连杆算进去),当前 env 用高度阈值 0.15 m 估计接触而非接触力。→ 影响:parkour 依赖接触力的奖励/终止项要么在 Isaac Lab 侧给闭链 body 配 ContactSensor 并验证读数可靠,要么全部改写成高度/运动学判据;越障任务里"身体撞到障碍就终止"是核心训练信号,这一条不解决,Phase 2 无法开工。

**⑦ 仿真吞吐与并行规模。**
go1:每 env 13 个 body 的串联树,原版训练开数千 env。
mos2026_2:每 env body 数多(仅 shank 相关就 16 个)+ 闭链约束求解,单 env 成本显著更高,现有配置 128~几百 env 规模。→ 影响:parkour 的课程和 PPO 超参(batch 由 num_envs × steps 决定)按数千 env 设计,直接照搬会 batch 不足;要么压 env 数换更多 steps_per_env,要么接受更长墙钟训练时间,超参需重调而非复制。

**⑧ 域随机化覆盖面不同。**
go1:关节属性(friction/armature/damping)随机化覆盖全部 12 个关节即覆盖全部动力学。
mos2026_2:EventCfg 的关节随机化只作用于 articulation 内的关节,closure joint 不在 articulation 里,其约束柔度/间隙不可随机化;四连杆的杆长/间隙误差(真机与 CAD 的差异)没有现成随机化入口。→ 影响:sim2real gap 中"闭链机构本身的误差"只能靠观测噪声、扰动注入和电机侧随机化间接覆盖,期望管理上要接受这部分 gap 比 go1 大。

**⑨ 跨引擎一致性(sim2sim 闸门)。**
go1:MuJoCo 与 PhysX 对串联树的求解高度一致,sim2sim 主要验证观测合约。
mos2026_2:同一闭链在 PhysX 是 excluded joint + projection,在 MuJoCo 是 equality constraint,两种建模的刚度/阻尼/漂移特性不同——sim2sim 差异本身就比串联腿大。→ 影响:MuJoCo 闸门(Phase 4)的判据要区分"策略不行"和"引擎建模差异",建议先用走路策略标定两引擎的行为差异基线,再评越障策略。

**⑩ 资产与工作空间。**
go1:URDF 管线,BarrierTrack 的几何参数(跳高 0.2~0.5 m、跨距、爬行高度)按 go1 腿长/工作空间标定。
mos2026_2:URDF Converter 不可用,资产改动走 USD 管线;四连杆腿的足端工作空间、等效膝角范围与 go1 不同(折叠极限还受装配分支约束)。→ 影响:所有障碍尺寸必须按 mos 的可达工作空间重标定(先用 `deploy/common` 的 FK 扫一遍足端包络),照抄 go1 数值会让课程起点就不可达。

一句话总结:**维度上看两边都是"12 维动作 + 本体观测",接口似乎兼容;但闭链让"关节角=构型、力矩=电机能力、接触力可直接读、初始化可随机、引擎间行为一致"这五个 parkour 的隐含假设全部失效**——迁移工作量的大头不在网络和算法,而在这份清单。

---

## 4. 与 parkour 原实现的三个刻意偏离

写在前面,避免照抄踩坑:

1. **不搬 parkour 的 rsl_rl fork。** 它基于 rsl_rl 1.0.2,魔改了 `obs_segments`、`EstimatorAlgoMixin`、`TPPO`(蒸馏)、GRU 尺寸 hack,与 Isaac Lab 配套的 rsl_rl 2.x API 完全不兼容。正确做法是按 `him/` 的模式:自定义 `ActorCritic` 子类 + 自定义 PPO 子类挂进 rsl_rl 2.x runner,estimator 监督损失加在 `update()` 里(HIM 的 estimator + swap loss 已经是这个结构,可直接扩展)。
2. **lin_vel 估计问题已被 HIM 解决,不必重做 parkour 的 estimator。** parkour 的 EstimatorPPO 和 HIM 干的是同一件事(从本体感受历史估计线速度)。HIM 的 6 帧历史 + 对比学习方案更新,且已在 mos2026_2 上跑通。**建议 parkour 策略直接以 HIM 观测布局为底座**(盲 45×6 + 估计速度),在其上加高度图分支。
3. **蒸馏阶段按硬件现实裁剪。** 原版第三阶段蒸馏到 Go1 头部深度相机;mos2026_2 目前只有 IMU + 电机反馈(`deploy/real/README.md`,lin_vel 靠 stub),没有深度相机/雷达。见 §7 的两条出路。

---

## 5. 分阶段实施方案

### Phase 0 — 前置阻塞清障(不做完,后面全白训)

真机 README 已记录两个硬阻塞,parkour 动作(跳、跨)比走路更暴力,只会放大它们:

- **增益失配**:策略按 kp=25 训练,部署增益 kp≈320,MuJoCo 实测接管 0.12 s 摔倒。→ 先按 todo.md 主线用可部署增益重训走路策略并过 MuJoCo 闸门,parkour 训练从一开始就用同一套增益。
- **力矩余量**:`effort_limit_sim=16` 是按走路调的。用 `传统步态/文档/dynamics_gear_ratio_analysis.md` 的方法核算跳跃峰值力矩(经验上是站立静载的 3~5 倍);如果硬件给不出,BarrierTrack 的 jump/leap 高度上限要相应压低,或者砍掉这两个技能只保留 hurdle/stairs/slope 类。

**验收**:可部署增益下的走路策略通过 `deploy/mujoco/play_mujoco.py` 受限观测测试。

### Phase 1 — 高度图观测 + 感知型 rough 策略

1. 在 DirectRLEnv 中加 Isaac Lab `RayCasterCfg`(`GridPatternCfg`,建议先 11×11≈0.8m×0.8m 前视网格,比 parkour 的 21×11 小,够用且省算力),挂在 base link 上。
2. obs 布局:`policy = HIM盲观测(45×6) + height_scan(121)`,critic 在现有 66 维特权观测上追加同一 height_scan 与真值 lin_vel。高度图数值处理照抄 parkour:`clip(base_z - offset - h, ±1) × 5.0`,并加测量噪声(域随机化)。
3. 网络:height_scan 过一个 2 层 MLP 编码成 32 维再与本体感受拼接(对应 `EncoderStateAcRecurrent` 的 encoder 分支;GRU 可先不加,HIM 的帧堆叠已提供时序信息)。
4. 在现有 `CURRICULUM_TERRAIN_CFG` 上训练,把楼梯/网格难度上限调高(0.12→0.20 m 台阶),验证高度图分支确实被利用(对比盲策略在同地形的课程升级速度)。

**验收**:感知策略在 0.15 m+ 楼梯地形的通过率显著高于 HIM 盲策略;`eval_suite.sh` 步态指标不劣化。

### Phase 2 — BarrierTrack 地形移植 + oracle 越障策略

1. **地形移植是本次迁移最大的一块纯工程活。** `barrier_track.py`(~2000 行)生成的是 heightfield + trimesh 混合 track;Isaac Lab 侧实现为一组自定义 `SubTerrainBaseCfg`(每种障碍一个 `mesh_*` 生成函数),复用 TerrainGenerator 的行=难度课程机制(Mos-One 已在用)。首批技能建议:`hurdle、down、stairsup、stairsdown、slope、discrete_rect`(闭链腿力矩风险最低的子集),jump/leap 视 Phase 0 力矩核算结果决定。
2. **oracle 特权观测**:parkour 的 field 阶段给策略喂 engaging block 信息(当前障碍类型 one-hot + 距离/几何参数)。在 SubTerrain 生成时把每段障碍的类型和几何写入 per-env 查询表,作为 critic/oracle 特权观测。
3. **奖励迁移**:从 `go2_field_config.py` 映射到 `custom_rewards.py`——前进速度跟踪(track 方向)、越障成功、碰撞惩罚、能耗/力矩惩罚(闭链腿建议把现有 opt-in 的 sum(τ²) 常开)。
4. 从 Phase 1 checkpoint 热启动(parkour 原版同样是 rough → field 续训)。

**验收**:oracle 策略在 6 技能 BarrierTrack 上课程升到最高难度档,单技能通过率 >80%(参考原版 play 的统计口径)。

### Phase 3 — 部署形态收敛(蒸馏或裁剪)

按硬件二选一,见 §7 详述:

- **3a(现有硬件,盲部署)**:把 oracle 策略蒸馏到"无高度图、只有 HIM 盲观测"的学生策略(rsl_rl 2.x 自带 Distillation runner,或手写 DAgger:学生跑、教师标注动作)。预期只能保住 stairs/slope/discrete 类"摸得出来"的地形,hurdle/jump 会显著缩水——这是感知缺失的物理上限,不是算法问题。
- **3b(加装深度相机后)**:复刻原版 distill:仿真渲染深度图(Isaac Lab `TiledCamera`)+ 延迟/噪声模型,学生网络 = CNN 深度编码器 + 盲观测,多进程采集可参考原版 `collect.py` 的 DAgger 结构。

**验收**:学生策略观测中不含任何仿真特权量,布局与 `deploy/real/rl_deploy.py` 的 45 维合约兼容(3a)或扩展合约有明确的传感器来源(3b)。

### Phase 4 — sim2sim 闸门 + 真机

1. `deploy/mujoco/` 新增 parkour 场景 XML(台阶/坡/沟的 MuJoCo 版,和 Phase 2 的 SubTerrain 几何一一对应),`play_mujoco.py` 加载学生策略跑受限观测测试——沿用项目已有的「MuJoCo 闸门」标准。
2. 真机渐进:平地 → 单个 5 cm 台阶 → 坡道,每级过了再上难度。`rl_deploy.py` 的 45 维 obs 合约、急停(姿态 0.5 rad)、力矩监护(FB τ×6.33 超限告警)全部沿用。

---

## 6. 工程改动清单(按仓库路径)

| 位置 | 改动 |
|---|---|
| `source/mos_one/.../mos2026_2_closed_usd/` | env 加 RayCaster 高度扫描 + obs 拼接;env_cfg 加 `PARKOUR_TERRAIN_CFG`(新 SubTerrain 集)+ 障碍元信息表 |
| `source/mos_one/.../custom_rewards.py` | 加越障/track 方向速度/碰撞奖励项 |
| 新增 `parkour/`(与 `him/` 平级) | `parkour_env_cfg.py`、encoder+actor 网络、蒸馏 runner、adapter —— 照 `him/` 的目录模式 |
| `source/mos_one/.../terrains/`(新) | BarrierTrack 各障碍的 mesh 生成函数(从 `barrier_track.py` 翻译) |
| `scripts/rsl_rl/train.py` | 加 `--task parkour_oracle / parkour_distill` 入口 |
| `deploy/mujoco/assets/` | parkour 验证场景 XML;`play_mujoco.py` 支持新 obs 布局 |
| 参考(只读,不改) | `~/code/rl/parkour/parkour/legged_gym/.../barrier_track.py`、`go2_field_config.py`、`rsl_rl/rsl_rl/{algorithms/estimator.py, modules/encoder_actor_critic.py}`、`legged_gym/scripts/collect.py` |

---

## 7. 关键风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| **闭链腿跳跃力矩不足** | effort_limit 16 N·m 按走路调;跳跃落地冲击可能直接超限或穿过四连杆奇异点(MuJoCo 已观察到折叠分支翻转) | Phase 0 先核算;技能集从低冲击子集起步;落地冲击惩罚 + 关节限位软约束进奖励;真机靠物理限位兜底但仿真里必须加 |
| **真机无外感知** | scandots 只在仿真存在,真机 IMU+电机反馈摸不出前方 30 cm 的沟 | 接受 3a 的能力上限,或立项加深度相机(3b);中间态:仅部署 stairs/slope 技能 |
| **闭链仿真稳定性** | env 里已有 NaN 防护(`nan_to_num` + 强制 reset);越障的大冲击会更频繁触发 | 保留防护;PhysX solver 迭代数在越障任务上调高;NaN reset 率纳入训练监控指标 |
| **rsl_rl 版本鸿沟** | fork 1.0.2 与 2.x runner 接口完全不同 | §4 已定:只迁思想,按 HIM 模式重写,估 2~3 天/模块 |
| **增益失配未解决就开训** | 训出的 oracle 全部作废 | Phase 0 设为硬前置,不并行 |
| **地形移植工作量低估** | barrier_track 含大量 goal/engaging 逻辑与地形生成耦合 | 只译几何生成,goal 逻辑用 Isaac Lab command/curriculum 机制重写;逐技能移植逐技能验证,不一次性全搬 |

---

## 8. 建议里程碑

1. **M0(阻塞清障)**:可部署增益重训走路 + MuJoCo 闸门通过;跳跃力矩核算结论落 `doc/`。
2. **M1(感知底座)**:height_scan 进 obs,感知策略在加难课程地形上超过盲基线。
3. **M2(oracle)**:6 技能 BarrierTrack 课程训满,play 视频 + 通过率报告。
4. **M3(学生策略)**:蒸馏完成,观测合约部署可行,MuJoCo parkour 场景通过。
5. **M4(真机)**:台阶/坡道渐进实测。

每个里程碑产出对应 `doc/experiments/` 台账条目,沿用现有 CHANGELOG/todo 三层记录习惯。
