from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, UsdPhysics

from .custom_rewards import compute_custom_reward_terms
from .mos2026_2_closed_usd_env_cfg import Mos20262ClosedUsdEnvCfg


class Mos20262ClosedUsdEnv(DirectRLEnv):
    cfg: Mos20262ClosedUsdEnvCfg

    def __init__(self, cfg: Mos20262ClosedUsdEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actuated_joint_ids, self._actuated_joint_names = self._robot.find_joints(
            self.cfg.actuated_joint_names, preserve_order=True
        )
        if len(self._actuated_joint_ids) != gym.spaces.flatdim(self.single_action_space):
            raise RuntimeError(
                "Closed-chain USD actuator mismatch: "
                f"configured action_space={gym.spaces.flatdim(self.single_action_space)}, "
                f"matched_joints={len(self._actuated_joint_ids)}, "
                f"matched_names={self._actuated_joint_names}"
            )
        self._capture_usd_default_joint_state()
        self._resolve_foot_bodies()
        self._resolve_symmetry_joint_groups()
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        action_scale_cfg = self.cfg.action_scale
        if isinstance(action_scale_cfg, (int, float)):
            self._action_scale = float(action_scale_cfg)
        else:
            scale = torch.as_tensor(action_scale_cfg, dtype=torch.float, device=self.device)
            if scale.numel() != len(self._actuated_joint_ids):
                raise RuntimeError(
                    f"action_scale length {scale.numel()} does not match "
                    f"actuated_joint count {len(self._actuated_joint_ids)}"
                )
            self._action_scale = scale
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.cfg.reward_scales.keys()
        }
        # feet_air_time 奖励的腾空计时器（秒），每脚一列；固定 4 列与
        # _foot_heights_and_contact 的输出对齐（脚未解析时全零、奖励为 0）。
        self._feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._cmd_vel_arrow = None
        self._actual_vel_arrow = None

    def _resolve_foot_bodies(self):
        # 启动时把"脚"对应 body 的索引解析好，后面 foot_slip 奖励就能直接
        # 用 self._robot.data.body_pos_w / body_lin_vel_w 索引拿数据，省开销。
        # 这个闭链 USD 里没有专门的 foot 链接，所以用每条腿被驱动的小腿 body
        # (`*_shank_link_a` 等) 当脚使——它本来就是每条腿最低的刚体。
        expr = getattr(self.cfg, "foot_body_names_expr", None)
        if not expr:
            self._foot_body_ids: list[int] = []
            self._foot_body_names: list[str] = []
            return
        try:
            ids, names = self._robot.find_bodies(expr, preserve_order=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve foot bodies with patterns {expr}: {exc}"
            ) from exc
        if not ids:
            raise RuntimeError(
                f"foot_body_names_expr={expr} matched no bodies. Available bodies: "
                f"{self._robot.body_names}"
            )
        self._foot_body_ids = list(ids)
        self._foot_body_names = list(names)
        print(
            f"[MosOne] Resolved {len(ids)} foot bodies for slip penalty: {names}",
            flush=True,
        )

    def _resolve_symmetry_joint_groups(self):
        # 解析对角 trot 步态对称奖励要用的关节索引。两个列表必须等长，
        # 且按"同关节类型 / 同身体位置"一一对应（详见 cfg 注释）：
        # 同一 index 上的 diag1[i] 和 diag2[i] 在 trot 步态里应该反相位，
        # 在 bound/pronk 步态里同相位 —— 这正是 `_compute_gait_symmetry`
        # 用 `diag1_dev + diag2_dev` 做相位检测的前提。
        diag1_names = getattr(self.cfg, "diag1_leg_joint_names", None) or []
        diag2_names = getattr(self.cfg, "diag2_leg_joint_names", None) or []
        if not diag1_names or not diag2_names:
            self._sym_diag1_ids: list[int] = []
            self._sym_diag2_ids: list[int] = []
            return
        if len(diag1_names) != len(diag2_names):
            raise RuntimeError(
                "diag1_leg_joint_names and diag2_leg_joint_names must have "
                f"equal length (got {len(diag1_names)} vs {len(diag2_names)})"
            )
        diag1_ids, _ = self._robot.find_joints(diag1_names, preserve_order=True)
        diag2_ids, _ = self._robot.find_joints(diag2_names, preserve_order=True)
        if len(diag1_ids) != len(diag1_names) or len(diag2_ids) != len(diag2_names):
            raise RuntimeError(
                "Failed to resolve all symmetry joints; "
                f"diag1 {diag1_names} -> {diag1_ids}; diag2 {diag2_names} -> {diag2_ids}"
            )
        self._sym_diag1_ids = list(diag1_ids)
        self._sym_diag2_ids = list(diag2_ids)

    def _capture_usd_default_joint_state(self):
        # Closed-chain USD assets often rely on authored passive joint coordinates.
        # Isaac Lab's config defaults every joint to zero unless we explicitly preserve
        # the parsed PhysX state before the first reset.
        self._robot.update(0.0)
        joint_pos = self._robot.data.joint_pos.clone()
        # PhysX may not have populated joint state yet; replace NaN/Inf with the
        # existing default so we don't seed every reset with garbage.
        invalid = ~torch.isfinite(joint_pos)
        if invalid.any():
            joint_pos = torch.where(invalid, self._robot.data.default_joint_pos, joint_pos)
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel)
        self._robot.data.default_joint_pos[:] = joint_pos
        self._robot.data.default_joint_vel[:] = joint_vel

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._strip_embedded_ground_prims()
        self._report_loop_joint_candidates()
        self._patch_projected_loop_joints()
        self._patch_missing_rigid_body_collisions()
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2400.0, color=(0.78, 0.82, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _report_loop_joint_candidates(self):
        # When projected_loop_joint_names is empty, scan the stage for joints
        # marked `physics:excludeFromArticulation=True` — those are exactly the
        # closure joints PhysX solves outside the reduced-coordinate tree, and
        # therefore the ones that need projection to stay numerically stable.
        # Prints them at startup so the user can paste straight into cfg.
        if getattr(self.cfg, "projected_loop_joint_names", None):
            return
        if not getattr(self.cfg, "auto_detect_loop_joints", False):
            return
        stage = get_current_stage()
        excluded: list[str] = []
        all_joint_names: list[str] = []
        for prim in stage.TraverseAll():
            if "Joint" not in str(prim.GetTypeName()):
                continue
            path = str(prim.GetPath())
            if "/Robot/" not in path and not path.endswith("/Robot"):
                continue
            all_joint_names.append(prim.GetName())
            attr = prim.GetAttribute("physics:excludeFromArticulation")
            if attr and attr.IsValid() and attr.Get() is True:
                excluded.append(prim.GetName())
        if not all_joint_names:
            return
        if excluded:
            unique = sorted(set(excluded))
            print(
                "[MosOne] projected_loop_joint_names is empty — detected "
                f"{len(unique)} joints with excludeFromArticulation=True (these "
                "are the closure joints that need projection):",
                flush=True,
            )
            print(f"  {unique}", flush=True)
            print(
                "[MosOne] Copy that list into Mos20262ClosedUsdEnvCfg."
                "projected_loop_joint_names and restart to enable projection.",
                flush=True,
            )
        else:
            print(
                "[MosOne] projected_loop_joint_names is empty but no joints "
                "with excludeFromArticulation=True were found under the robot prim. "
                f"All joints seen ({len(set(all_joint_names))} unique): "
                f"{sorted(set(all_joint_names))}",
                flush=True,
            )

    def _patch_projected_loop_joints(self):
        # Some closed-chain USD files model the closure joint as a regular
        # PhysX joint excluded from the reduced-coordinate articulation. Enable
        # projection on those closure joints so their anchors stay coincident.
        loop_joint_names = set(getattr(self.cfg, "projected_loop_joint_names", []))
        if not loop_joint_names:
            return
        stage = get_current_stage()
        patched = []
        for prim in stage.TraverseAll():
            if prim.GetName() not in loop_joint_names:
                continue
            prim.CreateAttribute("physxJoint:enableProjection", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("physxJoint:projectionLinearTolerance", Sdf.ValueTypeNames.Float).Set(0.002)
            prim.CreateAttribute("physxJoint:projectionAngularTolerance", Sdf.ValueTypeNames.Float).Set(0.05)
            patched.append(str(prim.GetPath()))
        patched_names = {path.rsplit("/", 1)[-1] for path in patched}
        missing = sorted(loop_joint_names - patched_names)
        if missing:
            raise RuntimeError(f"Missing projected closed-chain loop joints: {missing}; patched={patched}")

    def _strip_embedded_ground_prims(self):
        # Some shared USD examples include a demo GroundPlane inside the robot
        # asset. When Isaac Lab spawns the asset as an Articulation, that plane
        # moves with the robot root and can hide the real terrain or collide at
        # the robot spawn height. Remove those demo-only ground prims at runtime.
        if not getattr(self.cfg, "strip_embedded_ground_prims", False):
            return
        stage = get_current_stage()
        to_remove = []
        for prim in stage.TraverseAll():
            path = str(prim.GetPath())
            if not path.startswith("/World/envs/") or "/Robot/" not in path:
                continue
            name = prim.GetName().lower()
            if name in {"groundplane", "ground_plane"} or (name.startswith("ground") and prim.GetTypeName() == "Plane"):
                to_remove.append(prim.GetPath())
        for path in to_remove:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.SetActive(False)
        if to_remove:
            print(f"[MosOne] Deactivated embedded demo ground prims: {[str(path) for path in to_remove]}", flush=True)

    def _patch_missing_rigid_body_collisions(self):
        # Some USD examples contain rigid-body visual meshes but only partial
        # collision coverage. For visible validation and training, missing leg
        # colliders make the robot pass through the ground. Convert visual Mesh
        # descendants of collider-less rigid bodies into convex-hull colliders.
        if not getattr(self.cfg, "auto_collision_from_visuals", False):
            return
        stage = get_current_stage()
        patched = []
        for prim in stage.TraverseAll():
            schemas = {str(item) for item in prim.GetAppliedSchemas()}
            if "PhysicsRigidBodyAPI" not in schemas:
                continue
            descendants = [item for item in stage.TraverseAll() if item.GetPath().HasPrefix(prim.GetPath()) and item != prim]
            has_collision = any("PhysicsCollisionAPI" in {str(schema) for schema in item.GetAppliedSchemas()} for item in descendants)
            if has_collision:
                continue
            for item in descendants:
                if item.GetTypeName() != "Mesh":
                    continue
                UsdPhysics.CollisionAPI.Apply(item)
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(item)
                mesh_collision.CreateApproximationAttr().Set("convexHull")
                patched.append(str(item.GetPath()))
        if patched:
            print(f"[MosOne] Added convex-hull collision to {len(patched)} visual meshes without colliders.", flush=True)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions.clone(), -self.cfg.action_clip, self.cfg.action_clip)
        if self.cfg.action_control_mode == "effort":
            self._processed_actions = self._action_scale * self._actions
        else:
            default_pos = self._robot.data.default_joint_pos[:, self._actuated_joint_ids]
            self._processed_actions = self._action_scale * self._actions + default_pos

    def _apply_action(self):
        if self.cfg.action_control_mode == "effort":
            self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._actuated_joint_ids)
        else:
            self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._actuated_joint_ids)

    def _ensure_velocity_arrows(self):
        if self._cmd_vel_arrow is not None:
            return
        cmd_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        cmd_cfg.prim_path = "/Visuals/Mos20262/velocity_command"
        cmd_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self._cmd_vel_arrow = VisualizationMarkers(cmd_cfg)
        act_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
        act_cfg.prim_path = "/Visuals/Mos20262/velocity_actual"
        act_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self._actual_vel_arrow = VisualizationMarkers(act_cfg)

    def _xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, marker: VisualizationMarkers):
        # `xy_velocity` is in the robot base frame. Return (scale, world-frame quat).
        default_scale = marker.cfg.markers["arrow"].scale
        scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0 + 1e-3
        heading = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading)
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = quat_mul(base_quat_w, arrow_quat)
        return scale, arrow_quat

    def _update_velocity_arrows(self):
        self._ensure_velocity_arrows()
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.35
        cmd_xy = (
            torch.tensor(self.cfg.commanded_lin_vel_xy, device=self.device).expand(self.num_envs, 2)
            * self._current_command_scale()
        )
        cmd_scale, cmd_quat = self._xy_velocity_to_arrow(cmd_xy, self._cmd_vel_arrow)
        actual_xy = self._robot.data.root_lin_vel_b[:, :2]
        act_scale, act_quat = self._xy_velocity_to_arrow(actual_xy, self._actual_vel_arrow)
        # Offset the actual-velocity arrow slightly so the two don't visually overlap.
        act_pos = base_pos_w.clone()
        act_pos[:, 2] += 0.08
        self._cmd_vel_arrow.visualize(base_pos_w, cmd_quat, cmd_scale)
        self._actual_vel_arrow.visualize(act_pos, act_quat, act_scale)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, self._actuated_joint_ids]
                - self._robot.data.default_joint_pos[:, self._actuated_joint_ids],
                self._robot.data.joint_vel[:, self._actuated_joint_ids],
                self._actions,
            ],
            dim=-1,
        )
        # Closed-chain physics can occasionally produce NaN/Inf in PhysX outputs.
        # Replace with 0 so rsl_rl's check_nan doesn't abort; the matching envs
        # are flagged for reset in `_get_dones`.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        # 观测高斯噪声（sim2real）：只加在 policy obs 上，critic 始终拿干净观测
        # （critic 只在训练时用，不部署，噪声只会拖慢价值学习）。
        # 默认 cfg.obs_noise_std=0 即关闭，由 train.py --obs_noise_std 开启。
        policy_obs = obs
        noise_std = float(getattr(self.cfg, "obs_noise_std", 0.0))
        if noise_std > 0.0:
            policy_obs = obs + noise_std * torch.randn_like(obs)
        observations = {"policy": policy_obs}
        # 非对称 critic 的特权观测：干净 policy obs + 仿真真值信号。
        # 拼接顺序/维度必须与 cfg.state_space 的注释一致（当前 66）。
        if int(getattr(self.cfg, "state_space", 0) or 0) > 0:
            root_height = self._robot.data.root_pos_w[:, 2:3] - self._terrain.env_origins[:, 2:3]
            foot_height, foot_contact = self._foot_heights_and_contact()
            tau = self._robot.data.applied_torque[:, self._actuated_joint_ids]
            critic_obs = torch.cat(
                [obs, root_height, foot_height, foot_contact.float(), tau],
                dim=-1,
            )
            observations["critic"] = torch.nan_to_num(critic_obs, nan=0.0, posinf=0.0, neginf=0.0)
        self._previous_actions = self._actions.clone()
        if getattr(self.cfg, "show_velocity_arrows", True) and self.sim.has_gui():
            self._update_velocity_arrows()
        return observations

    def _foot_heights_and_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回 (foot_height, contact)，各 (N, 4)，顺序 [FL, FR, RL, RR]。
        # 接触用"相对地形原点高度低于阈值"近似（此闭链 USD 无接触传感器），
        # 与 _compute_foot_slip 的判定口径一致。脚未解析时返回全零/全 False。
        if not getattr(self, "_foot_body_ids", None) or len(self._foot_body_ids) < 4:
            height = torch.zeros(self.num_envs, 4, device=self.device)
            return height, torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        foot_pos_w = torch.nan_to_num(
            self._robot.data.body_pos_w[:, self._foot_body_ids, :], nan=0.0, posinf=0.0, neginf=0.0
        )
        foot_height = foot_pos_w[..., 2] - self._terrain.env_origins[:, 2:3]
        threshold = float(getattr(self.cfg, "foot_contact_height_threshold", 0.06))
        return foot_height, foot_height < threshold

    def _current_command_scale(self) -> float:
        # 命令速度课程：从 start_scale 倍命令随 env 步数线性升到 1.0 倍。
        # common_step_counter 每次 env.step +1（与并行环境数无关）。
        # cfg.command_curriculum_steps <= 0 时关闭；eval/play 显式设 0。
        steps = int(getattr(self.cfg, "command_curriculum_steps", 0) or 0)
        if steps <= 0:
            return 1.0
        start = float(getattr(self.cfg, "command_curriculum_start_scale", 0.4))
        progress = min(1.0, float(self.common_step_counter) / float(steps))
        return start + (1.0 - start) * progress

    def _compute_feet_air_time(self) -> torch.Tensor:
        # 正向步态奖励（移植自 robot_lab / IsaacLab feet_air_time）：脚落地的
        # 那一步结算 (腾空时长 - cfg.feet_air_time_threshold)，鼓励迈出有明显
        # 腾空期的步子；拖地滑步拿不到分，腾空过久的跳跃步态由 lin_vel_z /
        # anti_bound 负责压制。
        # 注意：本函数带状态（_feet_air_time 计时器），每个控制步只能调用一次。
        _, contact = self._foot_heights_and_contact()
        threshold = float(getattr(self.cfg, "feet_air_time_threshold", 0.4))
        first_contact = contact & (self._feet_air_time > 0.0)
        reward = torch.sum((self._feet_air_time - threshold) * first_contact.float(), dim=1)
        self._feet_air_time += self.step_dt
        self._feet_air_time *= (~contact).float()
        # 正常步态单步只有 1-2 只脚落地、|单脚结算| < 1；clamp 挡 PhysX 抽风。
        return torch.clamp(reward, -1.0, 1.0)

    def _compute_feet_gait(self) -> torch.Tensor:
        # 正向 trot 奖励（robot_lab feet_gait 的接触状态简化版）：对角脚
        # (FL,RR) 与 (FR,RL) 接触状态各自一致、且前左/前右反相时每步 +1。
        # 与 gait_symmetry / anti_bound 互补：惩罚项堵坏步态，这项奖励好步态。
        if not getattr(self, "_foot_body_ids", None) or len(self._foot_body_ids) < 4:
            return torch.zeros(self.num_envs, device=self.device)
        _, contact = self._foot_heights_and_contact()
        fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]
        diag_sync = (fl == rr) & (fr == rl)
        antiphase = fl != fr
        return (diag_sync & antiphase).float()

    def _compute_gait_symmetry(self) -> torch.Tensor:
        # 对角 trot 对称：把同一关节类型在两条对角线上对侧的两个关节配成一对，
        # 比较它们的偏差之和。**只对 thigh / shank 生效**，hip 故意排除——
        # 详见 cfg 里 diag1/diag2 注释（hip 镜像轴 + 这个公式 = 会把"合理外撇"
        # 错当成 bound 扣分，导致后腿被锁死贴默认值）。
        #
        # 配对规则（顺序由 cfg.diag1/diag2_leg_joint_names 决定）：
        #   pair[0] = (fl_thigh,         fr_thigh)         — 前腿两侧 thigh
        #   pair[1] = (rr_thigh,         rl_thigh)         — 后腿两侧 thigh
        #   pair[2] = (fl_shank_link,    fr_shank_link_a)  — 前腿两侧 shank
        #   pair[3] = (rr_shank_link_a,  rl_shank_link_a)  — 后腿两侧 shank
        # 假设左右关节轴在 USD 里是镜像的（从 fl_thigh/fr_thigh 默认值同为 0
        # 看不出来，但 hip 那对 0.06/0.06 同号已经间接验证了这套机器人是镜像
        # 约定）。镜像 + trot 反相位 ⇒ pair 内 diag1[i] ≈ -diag2[i]，相加 ≈ 0；
        # bound（前后对）/pronk（全同步）⇒ pair 内同号，相加 = 2×dev，
        # 平方求和给出明确的非零惩罚——这正是旧版"左右幅度差"奖励抓不到的
        # 失败模式（一跳一跳的步态满足 |left|² = |right|²，旧奖励给 0 惩罚）。
        if not getattr(self, "_sym_diag1_ids", None) or not getattr(self, "_sym_diag2_ids", None):
            return torch.zeros(self.num_envs, device=self.device)
        diag1_ids = self._sym_diag1_ids
        diag2_ids = self._sym_diag2_ids
        joint_pos = self._robot.data.joint_pos
        default_pos = self._robot.data.default_joint_pos
        joint_pos = torch.nan_to_num(joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
        diag1_dev = joint_pos[:, diag1_ids] - default_pos[:, diag1_ids]
        diag2_dev = joint_pos[:, diag2_ids] - default_pos[:, diag2_ids]
        paired_sum = diag1_dev + diag2_dev
        phase_violation = torch.sum(torch.square(paired_sum), dim=1)
        # clamp 用来挡 PhysX 抽风：4 个 pair × (2×1.5)² ≈ 36 是动作裁剪边界下
        # 的理论上限，正常 trot 时这一项 ≪ 1；25 取接近上限作为爆炸保护。
        return torch.clamp(phase_violation, 0.0, 25.0)

    def _compute_anti_bound(self) -> torch.Tensor:
        # 惩罚"同端两条腿同步抬落地"的步态（bound/pronk 的物理特征）。
        # gait_symmetry 是关节空间的静态姿态对称；这一项是脚的动态速度同步——
        # bound 时前左+前右脚一起向上、一起向下，v_z 同号；trot 时一只腿
        # 抬另一只腿落，v_z 互相抵消。两个项互补，一起锁死跳跃步态。
        #
        # foot_body_ids 顺序由 cfg.foot_body_names_expr 决定，目前是
        # [FL, FR, RL, RR]——如果以后改这个列表的顺序，下面的 pair 切片
        # 也要同步改。
        if not getattr(self, "_foot_body_ids", None) or len(self._foot_body_ids) < 4:
            return torch.zeros(self.num_envs, device=self.device)
        foot_ids = self._foot_body_ids
        foot_vel_z = self._robot.data.body_lin_vel_w[:, foot_ids, 2]  # (N, 4)
        foot_vel_z = torch.nan_to_num(foot_vel_z, nan=0.0, posinf=0.0, neginf=0.0)
        # 单脚 |v_z| 正常步态下 < 1.5 m/s。clamp 单脚速度避免某脚 PhysX 抽风
        # 把整批 reward 拉爆，再做 pair 求和。
        foot_vel_z = torch.clamp(foot_vel_z, -5.0, 5.0)
        front_pair_sync = torch.square(foot_vel_z[:, 0] + foot_vel_z[:, 1])  # FL + FR
        rear_pair_sync = torch.square(foot_vel_z[:, 2] + foot_vel_z[:, 3])   # RL + RR
        return torch.clamp(front_pair_sync + rear_pair_sync, 0.0, 100.0)

    def _compute_foot_slip(self) -> torch.Tensor:
        # 足端打滑惩罚：脚 body 着地时（用相对地形原点的 z 高度低于阈值近似），
        # 它在世界坐标系下的 xy 速度应该 ≈ 0。
        # 返回每个 env 的标量 = Σ_over_feet(||v_xy||² * contact_mask)。
        if not getattr(self, "_foot_body_ids", None):
            return torch.zeros(self.num_envs, device=self.device)
        foot_ids = self._foot_body_ids
        foot_pos_w = self._robot.data.body_pos_w[:, foot_ids, :]
        foot_lin_vel_w = self._robot.data.body_lin_vel_w[:, foot_ids, :]
        foot_pos_w = torch.nan_to_num(foot_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        foot_lin_vel_w = torch.nan_to_num(foot_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        terrain_z = self._terrain.env_origins[:, 2:3]  # (N, 1)
        foot_height = foot_pos_w[..., 2] - terrain_z  # (N, num_feet)
        threshold = float(getattr(self.cfg, "foot_contact_height_threshold", 0.06))
        contact_mask = (foot_height < threshold).float()  # (N, num_feet)
        foot_vel_xy_sq = torch.sum(torch.square(foot_lin_vel_w[..., :2]), dim=-1)  # (N, num_feet)
        # 求和前先 clamp，避免某个 env 单脚 PhysX 抽风产生一个巨大瞬时速度
        # 把整批 reward 拉爆；25 m²/s² 对应 |v| ≤ 5 m/s，正常步态远小于这个值。
        foot_vel_xy_sq = torch.clamp(foot_vel_xy_sq, 0.0, 25.0)
        return torch.sum(foot_vel_xy_sq * contact_mask, dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        scales = self.cfg.reward_scales
        # Detect envs whose PhysX state went non-finite (closed-chain blow-up).
        # These envs will be reset by _get_dones this step; zero their reward
        # so PPO doesn't see garbage. Matches the checks in _get_dones.
        root_pos_w = self._robot.data.root_pos_w
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        joint_pos_full = self._robot.data.joint_pos
        joint_vel_full = self._robot.data.joint_vel
        invalid_state = (
            ~torch.isfinite(root_pos_w).all(dim=-1)
            | ~torch.isfinite(root_lin_vel_b).all(dim=-1)
            | ~torch.isfinite(root_ang_vel_b).all(dim=-1)
            | ~torch.isfinite(joint_pos_full).all(dim=-1)
            | ~torch.isfinite(joint_vel_full).all(dim=-1)
        )

        # Sanitize raw signals before any squaring. nan_to_num catches NaN/Inf;
        # the clamps below catch the (more common) "large finite garbage" case
        # where PhysX returns e.g. root_z = 1e13 after a constraint failure.
        root_pos_w = torch.nan_to_num(root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        root_lin_vel_b = torch.nan_to_num(root_lin_vel_b, nan=0.0, posinf=0.0, neginf=0.0)
        root_ang_vel_b = torch.nan_to_num(root_ang_vel_b, nan=0.0, posinf=0.0, neginf=0.0)
        joint_vel_act = torch.nan_to_num(
            joint_vel_full[:, self._actuated_joint_ids], nan=0.0, posinf=0.0, neginf=0.0
        )

        root_height = root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        # Clamp the height error so a teleported root can't produce a giant
        # squared penalty. 2 m of height error is already off-the-charts wrong.
        base_height_error = torch.clamp(torch.square(root_height - self.cfg.base_height_target), 0.0, 4.0)
        # projected_gravity is unit-length by construction; clamp as belt-and-braces.
        upright_error = torch.clamp(
            torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1), 0.0, 2.0
        )
        lin_vel_z = torch.clamp(torch.square(root_lin_vel_b[:, 2]), 0.0, 100.0)
        ang_vel_xy = torch.clamp(torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1), 0.0, 100.0)
        joint_vel = torch.clamp(torch.sum(torch.square(joint_vel_act), dim=1), 0.0, 1.0e4)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        foot_slip = self._compute_foot_slip()
        gait_symmetry = self._compute_gait_symmetry()
        anti_bound = self._compute_anti_bound()
        feet_air_time = self._compute_feet_air_time()
        feet_gait = self._compute_feet_gait()
        # Constant forward locomotion command. Without a tracking term the
        # policy converges to the "stand still" optimum allowed by the other
        # shaping terms, so add explicit xy/yaw tracking rewards.
        # 命令速度课程：训练早期按比例缩小命令（见 _current_command_scale）。
        cmd_scale = self._current_command_scale()
        cmd_xy = torch.tensor(self.cfg.commanded_lin_vel_xy, device=self.device).expand(self.num_envs, 2) * cmd_scale
        lin_vel_xy_err = torch.sum(torch.square(cmd_xy - root_lin_vel_b[:, :2]), dim=1)
        ang_vel_yaw_err = torch.square(self.cfg.commanded_ang_vel_z * cmd_scale - root_ang_vel_b[:, 2])
        tracking_sigma = float(getattr(self.cfg, "tracking_sigma", 0.25))
        rewards = {
            "alive": torch.ones(self.num_envs, device=self.device) * scales.get("alive", 0.0),
            "upright": torch.exp(-upright_error / 0.25) * scales.get("upright", 0.0),
            # L2 penalty version of the orientation term — no exp saturation, so
            # the more the robot tips, the more it pays. This is what actually
            # punishes the "lying down" pose, since `upright` plateaus near 0.
            "flat_orientation": upright_error * scales.get("flat_orientation", 0.0),
            "base_height": base_height_error * scales.get("base_height", 0.0),
            "lin_vel_z": lin_vel_z * scales.get("lin_vel_z", 0.0),
            "ang_vel_xy": ang_vel_xy * scales.get("ang_vel_xy", 0.0),
            "joint_vel": joint_vel * scales.get("joint_vel", 0.0),
            "action_rate": action_rate * scales.get("action_rate", 0.0),
            "foot_slip": foot_slip * scales.get("foot_slip", 0.0),
            "gait_symmetry": gait_symmetry * scales.get("gait_symmetry", 0.0),
            "anti_bound": anti_bound * scales.get("anti_bound", 0.0),
            "feet_air_time": feet_air_time * scales.get("feet_air_time", 0.0),
            "feet_gait": feet_gait * scales.get("feet_gait", 0.0),
            "track_lin_vel_xy": torch.exp(-lin_vel_xy_err / tracking_sigma) * scales.get("track_lin_vel_xy", 0.0),
            "track_ang_vel_z": torch.exp(-ang_vel_yaw_err / tracking_sigma) * scales.get("track_ang_vel_z", 0.0),
        }
        for key, value in compute_custom_reward_terms(self).items():
            rewards[key] = rewards.get(key, torch.zeros_like(value)) + value * scales.get(key, 0.0)
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0) * self.step_dt
        # Wipe rewards for envs whose state was non-finite this step. _get_dones
        # will recycle them; PPO shouldn't see whatever number the math produced.
        reward = torch.where(invalid_state, torch.zeros_like(reward), reward)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value * self.step_dt
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if getattr(self.cfg, "visual_disable_resets", False):
            no_reset = torch.zeros_like(time_out)
            return no_reset, no_reset
        root_pos = self._robot.data.root_pos_w
        root_height = root_pos[:, 2] - self._terrain.env_origins[:, 2]
        died = root_height < self.cfg.fall_height_threshold
        # Orientation-based fall detection: when the robot is upright,
        # projected_gravity_b ≈ (0, 0, -1). After tipping past the configured
        # tilt threshold the z-component rises toward 0 — terminate so a
        # toppled robot doesn't continue accruing rewards on its side.
        gravity_z = self._robot.data.projected_gravity_b[:, 2]
        tipped_over = gravity_z > -self.cfg.fall_cos_threshold
        died = died | tipped_over
        # NaN/Inf state means the closed-chain sim blew up — a *simulator* failure,
        # not a policy failure. Report it as truncation (time_out) instead of
        # termination: the env still gets recycled, but PPO bootstraps the value
        # instead of treating it as a fall the policy should be punished for.
        # (The step reward is already zeroed in _get_rewards for these envs.)
        invalid_state = (
            ~torch.isfinite(root_pos).all(dim=-1)
            | ~torch.isfinite(self._robot.data.root_lin_vel_b).all(dim=-1)
            | ~torch.isfinite(self._robot.data.joint_pos).all(dim=-1)
            | ~torch.isfinite(self._robot.data.joint_vel).all(dim=-1)
        )
        died = died & ~invalid_state
        time_out = time_out | invalid_state
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._update_terrain_curriculum(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        # Promote envs that walked at least half the sub-terrain size during
        # the episode; demote envs that covered less than half the distance
        # the commanded velocity would have produced. Mirrors the
        # `terrain_levels_vel` curriculum from Isaac Lab's locomotion tasks.
        if not getattr(self.cfg, "terrain_curriculum_enabled", False):
            return
        if getattr(self._terrain, "terrain_origins", None) is None:
            return
        distance_walked = torch.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1
        )
        terrain_size = self.cfg.terrain.terrain_generator.size[0]
        # 期望距离按课程缩放后的实际命令算，否则低速课程期会被全员误判降级。
        cmd_xy = (
            torch.tensor(self.cfg.commanded_lin_vel_xy, device=self.device, dtype=distance_walked.dtype)
            * self._current_command_scale()
        )
        expected_distance = torch.norm(cmd_xy) * self.max_episode_length * self.step_dt
        move_up = distance_walked > terrain_size / 2
        move_down = (distance_walked < expected_distance * 0.5) & ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)
