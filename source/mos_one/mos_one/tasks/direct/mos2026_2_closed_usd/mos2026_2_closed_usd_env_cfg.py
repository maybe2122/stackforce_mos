from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass


ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    # 每个子地形为 8x8 m。3x3 网格覆盖 24x24 m，在 env_spacing=4 的情况下
    # 足以容纳 128 个环境实例。
    size=(16.0, 16.0),
    border_width=10.0,
    num_rows=6,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # 轻微凹凸地面：1-4 cm 随机高度变化。增大凸起幅度可提高地形难度；
        # 作为新策略的初始训练地形效果较好。
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.04),
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)


# 课程学习地形：行 = 难度等级（等级 0 最简单）。
# `curriculum=True` 使子地形沿行方向从难度 0 到 1 生成，
# 因此每个环境从最平坦的变体开始。环境的 `_reset_idx` 会根据
# 策略在每个回合中行走的距离来升级/降级环境所在行。
CURRICULUM_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=6,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # 纯平面 - 与难度无关。
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        # 凹凸地面：噪声随难度从 0 缩放至 8 cm。
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.0, 0.08),
            noise_step=0.005,
            border_width=0.25,
        ),
        # 上楼梯：台阶高度随难度从 0 缩放至 12 cm。
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.12),
            step_width=0.32,
            platform_width=2.0,
            border_width=1.0,
        ),
        # 下楼梯：同样的缩放但方向相反。
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.12),
            step_width=0.32,
            platform_width=2.0,
            border_width=1.0,
        ),
        # 离散网格障碍：单元格高度随难度从 0 缩放至 6 cm。
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.0, 0.06),
            platform_width=2.0,
        ),
    },
)


ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "mos2026_2_closed_usd" / "usd"
USD_PATH = ASSET_DIR / "mos2026_2.usd"


@configclass
class EventCfg:
    """域随机化 / 扰动注入（对应 todo.md §A「域随机化=0」最高优先级缺口）。

    sim2real 的核心：训练时随机化物理参数 + 注入扰动，逼策略学出鲁棒控制，否则
    单一标称值训出的策略上真机基本必崩。本配置默认**不挂载**到 env（见 cfg.events
    默认 None），由 `train.py --domain_rand` 显式开启；eval.py 保持关闭以保证
    「干净测量」。

    ⚠️ 需在 Isaac 机器上 smoke test 验证（本仓库 .venv 无 torch/Isaac，无法运行）：
      1. body_names="base" / 正则需对齐**真实 USD 的 body/joint 名**（可能与 MuJoCo
         XML 的 base/FL/fl_thigh… 不同）；先用 env._robot.body_names 打印核对。
      2. 各 mdp 事件函数签名以本机 Isaac Lab 2.3.2 为准；如有差异按报错微调。
    建议先 `--num_envs 16 --max_iterations 20 --domain_rand` 冒烟，再放量。
    """

    # --- startup：一次性随机化物理参数 ---
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 1.0),  # ±kg，模拟负载/电池差异
            "operation": "add",
        },
    )
    scale_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*(thigh|shank).*"),
            "mass_distribution_params": (0.8, 1.2),  # ±20% 腿质量
            "operation": "scale",
        },
    )

    # --- reset：每回合随机化执行器增益 + 关节零位偏置 ---
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),  # ±20% Kp
            "damping_distribution_params": (0.8, 1.2),    # ±20% Kd
            "operation": "scale",
        },
    )
    reset_joint_bias = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),  # ~±3°，模拟标定误差/编码器零点漂移
            "velocity_range": (0.0, 0.0),
        },
    )

    # --- interval：周期性推搡（抗扰）---
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class Mos20262ClosedUsdEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    # 域随机化事件。默认 None=关闭（不改变现有训练/eval）。train.py --domain_rand
    # 会把它设为 EventCfg() 开启；DirectRLEnv 在 cfg.events 非空时自动建 EventManager。
    events: EventCfg | None = None
    # 观测高斯噪声标准差（加在 45 维 obs 上，模拟传感器噪声）。默认 0=关闭。
    # 与 EventCfg 配套，由 train.py --obs_noise_std 设置；eval.py 用 --obs_noise 单独控制。
    obs_noise_std: float = 0.0
    decimation = 4
    # 每个关节相对于默认姿态的偏转系数（弧度）。
    # joint_target = default + action_scale * clamp(action, ±action_clip)。
    # 当 action_clip=1.5 时：
    #   髀关节：       0.5    -> ±0.75 rad   ≈ ±43°
    #   大腿/小腿：    0.8145 -> ±1.2217 rad ≈ ±70°
    # 元组长度和顺序必须与下方的 `actuated_joint_names` 匹配。
    action_scale = (
        0.5, 0.5, 0.5, 0.5,                      # fl_hip, fr_hip, rl_hip, rr_hip
        0.8145, 0.8145, 0.8145, 0.8145,          # fl_thigh, fr_thigh, rl_thigh, rr_thigh
        0.8145, 0.8145, 0.8145, 0.8145,          # *_shank_link*
    )
    # "position"：动作为关节角度增量，叠加到默认姿态上，
    #              通过 set_joint_position_target 发送（闭环 PD 控制）。
    # "effort"：  动作为力矩，经 action_scale 缩放后通过
    #              set_joint_effort_target 施加（开环力矩控制）。
    action_control_mode = "position"
    # 策略动作向量维度。必须等于 len(actuated_joint_names)
    # — 每个驱动关节一个标量（4 髀 + 4 大腿 + 4 小腿 = 12）。
    action_space = 12
    # 策略观测向量维度。在 env._get_observations 中按此顺序拼接：
    #   root_lin_vel_b (3) + root_ang_vel_b (3) + projected_gravity_b (3)
    # + (joint_pos - default_joint_pos) (12) + joint_vel (12) + last_actions (12)
    # = 45。
    # 注意：obs **不含**速度指令（commanded_lin_vel_xy / commanded_ang_vel_z）。
    # 当前指令是写死的常量（见下方 commanded_* 字段），只在 reward 里用来算
    # 跟踪误差；因此该策略只能跟踪这个固定前进指令，无法在线变速/转向。
    # 若要做 command-conditioned 策略：把 3 维 command 拼进 obs（→48）、reward
    # 改用动态 command、并在 _reset_idx 里随机采样指令，同时同步更新此值。
    # 每当 env.py 的 obs 拼接发生变化时都要更新此值。
    observation_space = 45
    # 特权“仅评论家”观测维度（非对称 actor-critic）。critic 只在训练时用，
    # 可以看仿真里才有的真值信号，价值估计更准 → PPO 优势估计方差更小。
    # 在 env._get_observations 中按此顺序拼接（"critic" 键）：
    #   干净版 policy obs（无噪声，45）
    # + base 相对地形高度 (1)
    # + 足端相对地形高度 (4) + 足端接触状态 (4)
    # + 受控关节实际下发力矩 applied_torque (12)
    # = 66。设为 0 可退回对称 critic（旧 checkpoint 评估时用）。
    # 每当 env.py 的 critic 拼接发生变化时都要更新此值。
    state_space = 66
    # 非无头模式运行时的 GUI 相机位姿（也用于 --livestream）。`eye`
    # 为相机位置，`lookat` 为目标点，均以米为单位在世界坐标系中；
    # `resolution` 为渲染窗口尺寸。
    viewer = ViewerCfg(
        eye=(3.0, -4.0, 2.0),
        lookat=(0.0, 0.0, 0.45),
        origin_type="world",
        resolution=(1280, 720),
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
        # 粗糙高度场地形 + 闭链关节会产生远超 PhysX 默认值的宽相位碰撞对。
        # 注意：之前这些上限是针对 32 GB GPU 调的，其中
        # gpu_found_lost_aggregate_pairs_capacity=2**29 单独就要 2**29*8 = 4 GiB
        # 连续显存，在 11 GB 的 2080 Ti 上无法分配，会导致 PhysScene 创建失败
        # （PxgCudaDeviceMemoryAllocator failed to allocate 4294967296 bytes），
        # 进而 simulation_view 为 None 报 create_articulation_view 崩溃。
        # 以下值面向 ~11 GB GPU 收紧；若换回大显存卡可调高。
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**24,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**20,
        ),
    )

    # 默认使用平坦平面。在训练/播放脚本中使用 `--terrain rough`
    # 可切换到 ROUGH_TERRAIN_CFG（或在代码中直接覆盖 `terrain`）。
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=None,
        collision_group=-1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.32)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # 快速克隆路径：闭链 USD 物理只解析一次，然后复刻到其余 N-1 个 env 子树。
    # 启动时间近似与 num_envs 无关，单 env 显存占用大幅下降。
    # 之前设为 False 是因为上游 USD 里嵌了 PhysicsScene + GroundPlane，
    # 会让 replicate 路径报错；现在两者都已经被剥离
    # （见 tools/strip_embedded_ground.py；扫描 stage 已确认没有 PhysicsScene
    # 也没有内嵌灯光/相机），所以可以放心启用快路径。
    # 如果以后换上游 USD 而新 USD 又带了 scene prim，把这里改回 False 即可，
    # 否则启动时会看到 PhysX "clone failed" / 重复 PhysicsScene 报错。
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=20.0,
                max_angular_velocity=30.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
                stabilization_threshold=0.0001,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
                stabilization_threshold=0.0001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),
            rot=(1, 0, 0, 0),
            joint_pos={
                # 后髋从 ±0.06 改成 ±0.15：训完观察到后两脚靠太近，原因是
                # 默认值本身就把后腿往内收（小幅度），加上 gait_symmetry 把
                # hip 锁死在默认值附近，policy 没机会把后腿撑开。
                # 这里把后髋绝对值放大到 0.15（≈ 8.6°），跟前髋一起在站姿里
                # 提供明显的外撇，让后脚有合理的横向间距。
                # 注意：rear hip 沿用原来的负号——如果发现这个改动反而让后脚
                # 更靠近（说明 USD 里 rear hip 轴的方向跟假设相反），把这两行
                # 的符号翻成正的就行。
                "fl_hip": 0.15, "fr_hip": 0.15, "rl_hip": -0.15, "rr_hip": -0.15,
                "fl_thigh": 0.0, "fr_thigh": 0.0, "rl_thigh": 0.0, "rr_thigh": 0.0,
                # FL 腿的 "shank_link_a" 在 USD 中命名为 `fl_shank_link`（无 _a 后缀）；
                # 其余三条腿使用 `*_shank_link_a` 命名约定。
                "fl_shank_link": 0.0, "fr_shank_link_a": 0.0, "rl_shank_link_a": 0.0, "rr_shank_link_a": 0.0,
            },
            joint_vel={},
        ),
        actuators={
            "main_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "fl_hip", "fr_hip", "rl_hip", "rr_hip",
                    "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
                    "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a",
                ],
                stiffness=25.0,
                damping=0.5,
                # 反射转子惯量（armature）：电机转子惯量经减速比平方放大后叠加在
                # 关节上（J_rotor × 6.33² ≈ 40×）。不设这项仿真的腿比真腿"轻"，
                # 策略学得过于敏捷（快速收摆腿零代价、偏好蹦跳步态），上真机跟不上。
                # 0.01 kg·m² 取自 DeepMind mujoco_menagerie 官方 unitree_go2 模型
                # （同款 GO-M8010-6 执行器，ctrlrange ±23.7 可证），大量 Go2
                # sim2real 工作验证过；更准的值待真机系统辨识后替换。
                armature=0.01,
                # GO-M8010-6 电机：最大扭矩 23.7 N·m，最大转速 30 rad/s（24V 供电）。
                # 2026-06-12 起 effort 从 12 上调到 16（doc/dynamics_gear_ratio_analysis.md
                # 行动项）：12 N·m 恰好卡在深蹲/动态蹬地的需求线上零余量，策略被迫
                # 「贴上限硬走」；16 N·m（≤峰值 23.7）配合 reward_scales["torque"]
                # 惩罚，让策略有余量但用力矩要付费。
                # 真实电机峰值（23.7 N·m / 30 rad/s）不建议在仿真里直接贴满，
                # 不然策略会学到一直贴峰值的动作，真机上跑不出来。
                effort_limit_sim=16.0,
                velocity_limit_sim=15.0,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )

    actuated_joint_names = [
        "fl_hip", "fr_hip", "rl_hip", "rr_hip",
        "fl_thigh", "fr_thigh", "rl_thigh", "rr_thigh",
        "fl_shank_link", "fr_shank_link_a", "rl_shank_link_a", "rr_shank_link_a",
    ]
    # 用来解析四个"脚"对应 body 的名字正则。
    # 这个闭链 USD 每条腿其实有 4 个 shank 相关 body
    # (`*_shank_link_a`, `*_shank_link_b`, `*_shank_motor_gear`, `*_shank`)，
    # 如果直接写 `.*shank.*` 会匹配到 16 个 body，把电机齿轮和被动平行连杆
    # 也算成"脚"，foot_slip 奖励会算到错误的物体上。
    # 这里只要每条腿"被驱动的主小腿"——也就是 actuated shank 关节的子 body。
    # FL 腿命名比较特殊（`fl_shank_link` 而不是 `fl_shank_link_a`），
    # 其他三条腿都遵循 `*_shank_link_a` 约定。用 ^...$ 精确匹配避免误中子串。
    foot_body_names_expr = [
        "^fl_shank_link_a$",
        "^fr_shank_link_a$",
        "^rl_shank_link_a$",
        "^rr_shank_link_a$",
    ]
    # 脚 body 相对地形原点的高度低于此阈值时，视为"接触地面"，
    # 才会参与 foot_slip 惩罚的统计。
    # 2026-06-12 从 0.07 抬到 0.15：eval 实测 duty_factor/步态指标全部退化、
    # foot_slip 训练时大概率从未生效——shank body 中心实际高度高于 0.07
    # （见 todo §评估发现）。0.15 覆盖支撑相的真实 body 高度；如步态指标
    # 出现"摆动相也被算成接触"，可用 eval.py --foot_contact_height 扫描微调。
    foot_contact_height_threshold = 0.15
    # 对角线 trot 步态对称奖励用到的两条对角线分组。
    # 旧的"左侧 vs 右侧"分组（fl+rl vs fr+rr）有个致命问题：bound 和 pronk
    # 这两种"跳着走"的步态左右两边能量是相等的，奖励给 0 惩罚 → policy 学到
    # 一跳一跳的步态。现在改成对角对：
    #   diag1 = FL + RR（前左 + 后右）
    #   diag2 = FR + RL（前右 + 后左）
    # 两个列表必须严格按"同关节类型 / 同身体位置"对齐（详见 reward 公式注释）：
    #   index 0,1 -> 各腿的 thigh；index 2,3 -> 各腿的 shank
    # 每个 index 上的 diag1[i] 和 diag2[i] 配成一对"同类型、对角线上对侧"的关节。
    # trot 时这两个关节反相位（一前一后），偏差相加≈0；bound/pronk 时同相位
    # （一起前/一起后），偏差相加≠0 → 触发惩罚。详见 env._compute_gait_symmetry。
    #
    # hip 关节故意不放进来：hip 管的是横向外摆，trot 里基本不参与相位，
    # 但因为左右 hip 轴是镜像的，"对称外撇"（合理站姿）刚好满足"两侧偏差同号"，
    # 会被这个公式当成 bound 来扣分——结果就是把 hip 锁死在默认值，后腿撑不开。
    # 所以这里只对真正驱动 trot 相位的 thigh / shank 用对角相位奖励，
    # hip 的对称由 init_state 默认值 + 物理动力学自然约束。
    diag1_leg_joint_names = [
        "fl_thigh", "rr_thigh",
        "fl_shank_link", "rr_shank_link_a",
    ]
    diag2_leg_joint_names = [
        "fr_thigh", "rl_thigh",
        "fr_shank_link_a", "rl_shank_link_a",
    ]
    # 闭环约束关节（每条腿用来"闭合"四连杆平行机构的那根 PhysX 关节）。
    # 这 4 个关节在 USD 里标了 `physics:excludeFromArticulation=True`，
    # 也就是说 PhysX 把它们当作独立约束求解，不放进缩减坐标的 articulation 树。
    # 如果不开 projection，它们的 anchor 会逐步漂移、solver 误差累积，
    # 最终爆掉——这就是 2026-05-15 跑训练时看到的 1e+28 奖励尖峰的根因。
    # env.py 里的 `_patch_projected_loop_joints` 会给下面每个名字开
    # physxJoint:enableProjection。
    #
    # 如果以后换 USD，把这个列表留空并把 `auto_detect_loop_joints` 设为 True，
    # env._setup_scene 启动时会把所有被 exclude 的关节名打印出来，照贴即可。
    projected_loop_joint_names = [
        "fl_close_loop",
        "fr_close_loop",
        "rl_close_loop",
        "rr_close_loop",
    ]
    auto_detect_loop_joints = True
    auto_collision_from_visuals = True
    # 原 USD 自带的 /mos2026_2/GroundPlane 已经通过
    # tools/strip_embedded_ground.py 永久从 USD 文件中删除，
    # 启动时不需要再扫一遍 stage 软关闭。
    # 如果以后换上游 USD 而新 USD 又带 demo ground plane，改回 True 即可。
    strip_embedded_ground_prims = False
    # 设为 True 时，_reset_idx 会根据每个回合走出的距离把走得远的 env
    # 升到更难的地形行、走得太近的 env 降级。
    # 需要 terrain_type="generator" 且地形 cfg 设 curriculum=True
    # （比如 CURRICULUM_TERRAIN_CFG）才生效。
    terrain_curriculum_enabled = False
    base_height_target = 0.32
    action_clip = 1.5
    visual_disable_resets = False
    commanded_lin_vel_xy = (1.0, 0.0)
    commanded_ang_vel_z = 0.0
    # --- 命令速度课程 ---
    # 训练早期低速命令更容易学出迈步（借鉴 robot_lab 的 command_levels 课程思路）：
    # 命令从 start_scale 倍随 env 步数线性升到 1.0 倍，command_curriculum_steps
    # 是升满所需的 env 步数（common_step_counter 口径：每次 env.step +1，
    # 与并行环境数无关；默认 30000 ≈ 默认训练量 5000 iter × 24 steps 的 1/4）。
    # 0 = 关闭。eval.py / play.py 会显式设 0，保证按全速命令评估回放。
    # 注意 resume 训练时计数器从 0 重新开始，课程会重新爬坡一次。
    command_curriculum_steps = 30000
    command_curriculum_start_scale = 0.4
    # feet_air_time 奖励的期望腾空时长（秒）：落地时结算 (腾空时长 - 该值)。
    feet_air_time_threshold = 0.4
    show_velocity_arrows = True
    # 从 0.15 抬到 0.22：避免机器人"趴下"后（base 高度 0.10-0.18 m 之间）
    # 还在持续薅 alive / orientation 奖励。低于这个高度就判终止。
    fall_height_threshold = 0.22
    # 投影重力 z 分量的 cos(angle) 阈值，用来判定"翻倒"。
    # 0.85 大约对应离竖直方向 32° 倾角；越接近 1.0 越早终止。
    fall_cos_threshold = 0.85
    # 这套 reward 权重是为了打破"原地不动"局部最优而调出来的：
    #   - 去掉 `alive`：之前 0.5/step 的白送奖励压过了训练初期微弱的速度跟踪奖励。
    #   - `track_ang_vel_z` 从 1.0 降到 0.5：原来不转身的机器人 exp(0) 就拿满分。
    #   - `flat_orientation` 是 L2 惩罚项（在 _get_rewards 里有 clamp 上限），
    #     翻倒代价高但不会被 exp 饱和掉。
    #   - `base_height` 在 _get_rewards 里 clamp(., 0, 4)；这里的权重决定 cap
    #     范围内的强度。之前是 -10 + 没有 clamp，叠加闭链 PhysX 爆掉时
    #     单步能炸到 1e+28。改成 -3 后最坏情况是 4 * 3 = 12（再乘 step_dt），
    #     既保留惩罚作用又不会主导整个 reward 预算。
    #   - `lin_vel_z` 之前 -0.5、lin_vel_z² 上限 100 → 单步最多 -50，
    #     完全盖过了 +4 的跟踪奖励，这是之前 reward 卡在 500-1100 的主因。
    #     现在抬到 -0.5（上限 100 → 单步最多 -50），用来配合下面新的对角对称
    #     奖励一起压制"跳着走"。正常 trot 时 z 速度 ≈ 0，lin_vel_z² ≪ 1 几乎
    #     没代价；跳跃步态瞬时 |v_z| ≈ 1 m/s 时单步 -0.5，刚好抵消 track 奖励
    #     的收益。如果发现训练初期被这一项压得不敢动，再调到 -0.2 看看。
    #   - `track_lin_vel_xy` 从 4.0 提到 6.0，让"往前走"明显比"站住"赚得多。
    #   - joint_vel / action_rate / ang_vel_xy 的惩罚都保持很小，
    #     这样 PPO 在探索时还能自由地抖动腿。
    reward_scales = {
        "alive": 0.0,
        "upright": 0.0,
        "flat_orientation": -2.5,
        "base_height": -3.0,
        "lin_vel_z": -0.5,
        "ang_vel_xy": -0.02,
        "joint_vel": -0.0001,
        "action_rate": -0.005,
        "track_lin_vel_xy": 6.0,
        "track_ang_vel_z": 0.5,
        # 足端打滑惩罚：脚 body 水平速度的平方，乘以一个"是否着地"的接触掩码后求和。
        # 没有这一项时 policy 容易学出"滑步"的步态——脚虽然踩着地，但实际上
        # 在身体下方持续滑动，从视觉上就是严重打滑。
        # -0.1 这个权重保证打滑代价能盖过 track_lin_vel_xy 拿到的收益
        # （≈ 6 * exp(-err)）：一只脚以 1 m/s 滑动时 vel² = 1，对应单步单脚 -0.1。
        "foot_slip": -0.1,
        # 对角 trot 步态对称奖励：惩罚"对角两腿同相位"的程度
        # （详见 env._compute_gait_symmetry 与 diag1/diag2 配置注释）。
        # trot 时 paired_sum ≈ 0 拿满分；bound/pronk 这种"跳着走"会被狠扣。
        # 2026-07-05 从 -0.5 降到 -0.25：正向引导交给下面新增的
        # feet_gait / feet_air_time，这里只保留兜底惩罚，避免惩罚主导
        # 让策略不敢探索。
        "gait_symmetry": -0.25,
        # 同端两脚同步抬落地的惩罚（bound/pronk 的直接物理特征）。
        # gait_symmetry 是关节空间静态姿态对称；anti_bound 是脚的 z 速度
        # 动态同步检测——前左+前右脚一起向上/向下时触发，后对同理。
        # 单脚 |v_z| 上限 5 m/s，pair 求和后单步最坏 (2*5)² *2 = 200，
        # clamp 到 100；正常 trot 时这一项 ≪ 1 几乎没代价。
        # 2026-07-05 从 -1.0 降到 -0.5：理由同 gait_symmetry——好步态由
        # 正向项奖励，坏步态惩罚只做兜底。
        "anti_bound": -0.5,
        # --- 正向步态引导（移植自 robot_lab / IsaacLab velocity 任务）---
        # feet_air_time：落地瞬间结算 (腾空时长 - feet_air_time_threshold)，
        # 鼓励迈出有明显腾空期的步子；拖地滑步拿 0 分甚至负分。
        # 与惩罚项相反，它告诉策略"该怎么走"，给探索指方向。
        "feet_air_time": 0.25,
        # feet_gait：对角脚 (FL,RR)/(FR,RL) 接触状态一致且前左/前右反相时
        # 每步 +1（binary），直接正向奖励 trot 模式；robot_lab Go2 同款思路
        # （其 feet_gait 权重 0.5，track 3.0；这里 track 6.0，等比放大到 1.0）。
        "feet_gait": 1.0,
        # 力矩惩罚 sum(τ²)（custom_rewards.py 计算）。2026-06-12 起默认 -2e-4 开启
        # （此前 0.0 关闭）：评估显示策略「贴力矩上限硬走」（shank 83~84% 时间 ≥90%
        # 上限）、CoT 4.5——是真机力矩/电流不足在仿真侧的同源表现；蹦跳步态的高峰值
        # 力矩在没有这项时也是"免费"的。重训后用 eval_plot.py 对比 near_limit_frac /
        # CoT 微调；过大会让策略不敢发力、走不动。
        "torque": -2.0e-4,
        "custom_reward": 0.0,
    }
    # 速度跟踪奖励里 exp() 的带宽。带宽越大，部分达成（比如指令 1.0 m/s
    # 实际只走 0.4 m/s）也能拿到有意义的梯度，引导策略向目标速度靠拢；
    # 带宽太小会像窄高斯一样在 0 附近直接饱和到 0。
    tracking_sigma = 1.0
