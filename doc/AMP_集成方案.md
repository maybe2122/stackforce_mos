# AMP（Adversarial Motion Priors）集成方案 — Mos-One / HIM 训练栈

> 目标：用一个**判别器学到的 style reward** 替代当前手工步态塑形项
> （`gait_symmetry`、`anti_bound`，可选 `foot_slip`），获得更自然的 trot 步态，
> 同时**不改动 policy 网络结构、不影响部署链路（rl_sar / MuJoCo sim2sim / ONNX）**。
>
> AMP 与 HIM 正交：HIM 管状态估计（用历史估 base 线速度 + latent），AMP 管 reward 风格。
> 最终形态 = HIMActorCritic 不变 + HIMPPO 上挂一个 AMP 判别器。
>
> 本文档约定：actuated 关节数 `J = 12`（见 `actuated_joint_names`）。

---

## 0. 总览：改哪些文件

| 文件 | 操作 | 作用 |
|---|---|---|
| `him/him_env.py` | 改 | 新增 `_amp_obs()`，在 `_get_observations` 返回 dict 加 `"amp"` 键；`_reset_idx` 快照 `_pre_reset_amp` |
| `him/adapter.py` | 改 | step/reset 透传 amp 观测的**转移对** `(amp_obs, next_amp_obs)` 进 `extras` |
| `him/him_rl/modules/amp_discriminator.py` | **新建** | 判别器网络 + LSGAN 损失 + 梯度惩罚 + style reward |
| `him/him_rl/datasets/amp_loader.py` | **新建** | 加载参考运动 `.npy`、构造转移对、采样 expert minibatch |
| `him/him_rl/storage/him_rollout_storage.py` | 改 | rollout 里多存 `amp_observations` / `next_amp_observations` / 一个 amp 采样器 |
| `him/him_rl/algorithms/him_ppo.py` | 改 | 持有判别器 + 优化器；`update()` 增加判别器训练；新增 `predict_style_reward()` |
| `him/him_rl/runners/him_on_policy_runner.py` | 改 | rollout 内把 style reward 融进 env reward；TB 记 `disc_*`；save/load 带判别器 |
| `him/him_cfg.py` | 改 | 新增 `"amp"` 配置块 |
| `him/train.py` | 改 | 加 `--amp` / `--motion_files` 等开关，传给 cfg |
| `tools/asset/record_amp_motion.py` | **新建** | 从已训 HIM checkpoint rollout 录参考动作（Phase 0 自举） |

> 关键不变量：`him/play.py`、`deploy/`、ONNX 导出、`third_party/rl_sar` **一行都不用改**——
> 判别器只在训练用，`act_inference` 路径不动。

---

## 1. AMP 观测定义（最重要的设计决策）

AMP 观测 ≠ policy 观测。它必须：
- **干净无噪声**（用原始 robot data，不走 `him_actor_noise` 那条加噪路径）；
- **指令无关**（不含 `cmd`）——判别器只判断"动作像不像"，不判断"是否听指令"；
- **能完整描述姿态 + 运动相位**。

### 1.1 核心布局（`AMP_OBS_DIM = 33`）

```
amp_obs = [
    joint_pos_rel(12),   # joint_pos[jids] - default_joint_pos[jids]
    joint_vel(12),       # joint_vel[jids]
    base_lin_vel_b(3),   # root_lin_vel_b   ← 步态周期/速度的关键
    base_ang_vel_b(3),   # root_ang_vel_b
    projected_gravity(3) # 让风格对 roll/pitch 敏感（防歪着走）
]                        # 共 33
```

判别器输入是**转移对** `concat(amp_obs_t, amp_obs_{t+1})`，维度 `2 * 33 = 66`。
转移对编码了"速度/加速度"信息，这是 AMP 能学出节律的根本。

### 1.2 可选增强（先跑通核心再加）
- 足端相对 base 的位置 `4 feet × 3 = 12`（用 `foot_body_names_expr` 解析的 4 个 body，
  在 base frame 下）→ 显著提升落脚点自然度，但要算坐标变换。
- base 高度 `1`（`root_pos_w[:,2] - 地形高度`）。

加这些只需同步改：`_amp_obs()`、参考数据生成、`cfg["amp"]["obs_dim"]`。

### 1.3 `him_env.py` 实现

```python
# him_env.py，与 _him_critic_obs 同款（都用干净 robot data）
def _amp_obs(self) -> torch.Tensor:
    jids = self._actuated_joint_ids
    jpos_rel = (self._robot.data.joint_pos[:, jids]
                - self._robot.data.default_joint_pos[:, jids])
    obs = torch.cat([
        jpos_rel,                               # [0:12]
        self._robot.data.joint_vel[:, jids],    # [12:24]
        self._robot.data.root_lin_vel_b,        # [24:27]
        self._robot.data.root_ang_vel_b,        # [27:30]
        self._robot.data.projected_gravity_b,   # [30:33]
    ], dim=-1)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

def _get_observations(self) -> dict:
    policy = self._him_actor_obs()
    critic = self._him_critic_obs()
    amp = self._amp_obs()                       # ← 新增
    self._previous_actions = self._actions.clone()
    if getattr(self.cfg, "show_velocity_arrows", True) and self.sim.has_gui():
        self._update_velocity_arrows()
    return {"policy": policy, "critic": critic, "amp": amp}   # ← 加 "amp"

def _reset_idx(self, env_ids):
    try:
        self._pre_reset_critic = self._him_critic_obs()
        self._pre_reset_amp = self._amp_obs()   # ← 快照终局 amp，供 adapter 修正终止转移
    except Exception:
        self._pre_reset_critic = None
        self._pre_reset_amp = None
    super()._reset_idx(env_ids)
```

---

## 2. adapter.py：透传 amp 转移对

DirectRLEnv 在 `step()` 内会先 reset 终止的 env 再算返回的 obs，所以 `step` 返回的
`obs_dict["amp"]` 对**已终止的 env 是 post-reset 帧**。AMP 要的是 step **当步**的转移
`(s_t, s_{t+1})`：
- `s_t`   = 上一步存下的 amp（pre-step）；
- `s_{t+1}` = 本步 amp，但终止 env 要用 `_pre_reset_amp` 还原成"真正的终局帧"
  （和 `_pre_reset_critic` 同机制）。

终止转移本身在判别器更新时按 `done` mask 掉（见 §5），所以这里只要保证非终止 env 的
转移对连续即可。

```python
# adapter.py __init__ 末尾
self.amp_obs = torch.zeros(self.num_envs, AMP_OBS_DIM, device=self.device)

# reset()
def reset(self, env_ids=None):
    obs_dict, _ = self.env.reset()
    current = obs_dict["policy"]
    self.obs_buf = current.repeat(1, HISTORY_LENGTH)
    self.critic_buf = obs_dict["critic"]
    self.amp_obs = obs_dict["amp"].clone()           # ← s_0
    return self.obs_buf, self.critic_buf

# step()：在算 extras 时加 amp 转移对
def step(self, actions):
    obs_dict, rew, terminated, truncated, info = self.env.step(actions)
    dones = terminated | truncated
    done_ids = dones.nonzero(as_tuple=False).flatten()

    current = obs_dict["policy"]
    self.critic_buf = obs_dict["critic"]
    self._push_history(current, done_ids)

    # --- AMP 转移对 ---
    amp_obs_t = self.amp_obs                          # pre-step s_t
    next_amp = obs_dict["amp"].clone()                # post-step s_{t+1}
    pre_amp = getattr(self.env, "_pre_reset_amp", None)
    if pre_amp is not None and done_ids.numel() > 0:
        next_amp[done_ids] = pre_amp[done_ids]        # 还原终局帧
    self.extras["amp_obs"] = amp_obs_t
    self.extras["next_amp_obs"] = next_amp
    self.amp_obs = obs_dict["amp"].clone()            # 下一步的 s_t（含 post-reset）

    # ...（其余 termination_privileged_obs / time_outs 逻辑不变）...
    return (self.obs_buf, self.critic_buf, rew, dones.to(torch.long),
            self.extras, done_ids, termination_privileged_obs)
```

> `AMP_OBS_DIM` 建议放进 `him_env_cfg.py` 当常量（和 `NUM_ONE_STEP_OBS` 并列），
> adapter 与各处统一 import。

---

## 3. 参考运动数据集

### 3.1 格式
每个 clip 一个 `.npy`，形状 `(T, AMP_OBS_DIM)`，按**控制频率**（env 的 policy dt，
注意是 decimation 后的步频，不是物理 dt）逐帧存 amp 观测。多 clip 用列表传入。

### 3.2 数据来源（按成本从低到高）
1. **自举（推荐起步）**：用 `tools/asset/record_amp_motion.py` 加载当前最好的 HIM
   checkpoint，在 env 里 rollout，把 `_amp_obs()` 逐帧 dump。先把整条 AMP 管线跑通，
   再换更好的参考源。上限受种子策略限制，但零额外资产。
2. **轨迹优化 / 现有 trot 控制器**生成关节轨迹 → 用 FK 转成 amp 观测。
3. **四足 mocap retarget**（Unitree A1/Go 等）→ 映射到 12 关节 `actuated_joint_names`。
   闭链/命名特殊，成本最高，留到效果验证后再上。

### 3.3 `amp_loader.py` 骨架

```python
class AMPLoader:
    def __init__(self, motion_files, obs_dim, device,
                 motion_dt, sim_dt, normalizer=None):
        # 加载所有 clip，按 sim_dt/motion_dt 重采样到控制频率，
        # 预构造转移对 (s_t, s_{t+1})，拼成大 buffer：
        #   self.expert_obs      (M, obs_dim)
        #   self.expert_next_obs (M, obs_dim)
        ...
    def sample(self, batch_size):
        idx = torch.randint(0, self.expert_obs.shape[0], (batch_size,),
                            device=self.device)
        return self.expert_obs[idx], self.expert_next_obs[idx]
```

构造转移对时**不要跨 clip 边界**（每个 clip 内部取 `s[:-1], s[1:]`）。

---

## 4. 判别器 `amp_discriminator.py`

```python
import torch, torch.nn as nn
from him_rl.modules.actor_critic import get_activation

class AMPDiscriminator(nn.Module):
    def __init__(self, obs_dim, hidden_dims=(1024, 512),
                 activation='relu', device='cuda:0',
                 reward_scale=2.0, grad_penalty_coef=10.0):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.reward_scale = reward_scale
        self.grad_penalty_coef = grad_penalty_coef
        act = get_activation(activation)
        layers, inp = [], obs_dim * 2            # 转移对 (s, s')
        for h in hidden_dims:
            layers += [nn.Linear(inp, h), act]; inp = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(inp, 1)            # 输出 logit（LSGAN 不过 sigmoid）
        self.to(device)

    def forward(self, obs, next_obs):
        return self.head(self.trunk(torch.cat([obs, next_obs], dim=-1)))

    # LSGAN：expert→+1，policy→-1
    def loss(self, expert_obs, expert_next, policy_obs, policy_next):
        d_exp = self.forward(expert_obs, expert_next)
        d_pol = self.forward(policy_obs, policy_next)
        loss_exp = 0.5 * (d_exp - 1).pow(2).mean()
        loss_pol = 0.5 * (d_pol + 1).pow(2).mean()
        gp = self.gradient_penalty(expert_obs, expert_next)
        with torch.no_grad():
            acc_exp = (d_exp > 0).float().mean()
            acc_pol = (d_pol < 0).float().mean()
        return loss_exp + loss_pol + self.grad_penalty_coef * gp, \
               dict(d_exp=d_exp.mean().item(), d_pol=d_pol.mean().item(),
                    acc=(0.5*(acc_exp+acc_pol)).item(), gp=gp.item())

    def gradient_penalty(self, obs, next_obs):
        obs = obs.clone().requires_grad_(True)
        next_obs = next_obs.clone().requires_grad_(True)
        d = self.forward(obs, next_obs)
        grad = torch.autograd.grad(d.sum(), [obs, next_obs],
                                   create_graph=True)[0]
        return grad.pow(2).sum(-1).mean()

    @torch.no_grad()
    def predict_reward(self, obs, next_obs):
        d = self.forward(obs, next_obs)
        # AMP style reward：D 越接近 expert(=1) 越高，下界 0
        r = torch.clamp(1.0 - 0.25 * (d - 1.0).pow(2), min=0.0)
        return self.reward_scale * r.squeeze(-1)
```

> 判别器输入需做归一化：复用 `him_actor_critic.py` 里的 `Normalization`/`RunningMeanStd`
> （对 amp_obs 维护一个 running mean/std，expert 与 policy 用**同一个** normalizer），
> 在 `forward` 前套 `self.normalizer(x)`。归一化器状态随 checkpoint 存。

---

## 5. storage 改动

`HIMRolloutStorage`：
- `Transition` 加 `amp_observations` / `next_amp_observations`；
- `__init__` 加两个 buffer `(num_transitions_per_env, num_envs, amp_dim)`；
- `add_transitions` 里 copy；
- 新增一个 **AMP 采样器**（判别器训练用，扁平化、不需 GAE 的轨迹结构）：

```python
def amp_mini_batch_generator(self, num_mini_batches, num_epochs):
    amp_obs = self.amp_observations.flatten(0, 1)
    amp_next = self.next_amp_observations.flatten(0, 1)
    not_done = (1 - self.dones.flatten(0, 1).squeeze(-1)).bool()  # 屏蔽终止转移
    amp_obs, amp_next = amp_obs[not_done], amp_next[not_done]
    n = amp_obs.shape[0]; mb = n // num_mini_batches
    for _ in range(num_epochs):
        idx = torch.randperm(n, device=self.device)
        for i in range(num_mini_batches):
            b = idx[i*mb:(i+1)*mb]
            yield amp_obs[b], amp_next[b]
```

---

## 6. HIMPPO 改动

```python
# __init__ 增参：discriminator, amp_loader, amp_cfg
self.discriminator = discriminator
self.amp_loader = amp_loader
self.amp_optimizer = optim.Adam(self.discriminator.parameters(),
                                lr=amp_cfg["disc_lr"])
self.amp_num_epochs = amp_cfg["num_learning_epochs"]
self.amp_num_mini_batches = amp_cfg["num_mini_batches"]

# Transition 透传（act 之后、process_env_step 里把 amp 存进 transition）
# —— 由 runner 在 step 后写 self.alg.transition.amp_observations / next_amp_observations

@torch.no_grad()
def predict_style_reward(self, amp_obs, next_amp_obs):
    return self.discriminator.predict_reward(amp_obs, next_amp_obs)
```

判别器训练并入 `update()` 末尾（policy minibatch 与 expert 配对）：

```python
mean_disc_loss = mean_disc_acc = 0.0
amp_gen = self.storage.amp_mini_batch_generator(
    self.amp_num_mini_batches, self.amp_num_epochs)
for policy_obs, policy_next in amp_gen:
    expert_obs, expert_next = self.amp_loader.sample(policy_obs.shape[0])
    d_loss, info = self.discriminator.loss(
        expert_obs, expert_next, policy_obs, policy_next)
    self.amp_optimizer.zero_grad(); d_loss.backward(); self.amp_optimizer.step()
    mean_disc_loss += d_loss.item(); mean_disc_acc += info["acc"]
# 归一化后随其它 loss 一起 return（runner 写 TB）
```

> 注意 PPO policy 更新逻辑（surrogate / value / HIM estimator）**完全不动**。
> 判别器是独立优化器、独立 minibatch，不与 actor-critic 共享梯度。

---

## 7. runner 改动：在 rollout 内融合 style reward

style reward 必须在 `process_env_step` **之前**注入（它会做 timeout bootstrap，
要基于最终 reward）。判别器用的是**上一轮**训练好的 D（reward 略滞后一个 iteration，
AMP 标准做法，可接受）。

```python
# learn() rollout 循环内：
actions = self.alg.act(obs, critic_obs)
obs, privileged_obs, rewards, dones, infos, term_ids, term_priv = self.env.step(actions)
# ... 设备搬运、next_critic_obs 拼接（不变）...

# --- AMP：取转移对、算 style reward、融合 ---
amp_obs      = infos["amp_obs"].to(self.device)
next_amp_obs = infos["next_amp_obs"].to(self.device)
style_rew = self.alg.predict_style_reward(amp_obs, next_amp_obs)
task_w  = self.amp_cfg["task_reward_coef"]    # 例 1.0
style_w = self.amp_cfg["style_reward_coef"]   # 例 0.5
rewards = task_w * rewards + style_w * style_rew

# 把 amp 转移对挂到 transition，供 storage 落盘
self.alg.transition.amp_observations = amp_obs
self.alg.transition.next_amp_observations = next_amp_obs

self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
```

TB / SwanLab 新增标量：`AMP/disc_loss`、`AMP/disc_acc`、`AMP/style_reward_mean`、
`AMP/d_expert`、`AMP/d_policy`、`AMP/grad_penalty`。SwanLab 已镜像 SummaryWriter，
自动可见，无需额外改 `train.py` 的 SwanLab 段。

`save()/load()` 增加 `discriminator_state_dict` 与 `amp_optimizer_state_dict`
（resume 必需；部署不需要）。

---

## 8. 配置块（him_cfg.py）

```python
"amp": {
    "enabled": False,                  # train.py --amp 打开
    "obs_dim": 33,
    "motion_files": [],                # train.py --motion_files 注入
    "motion_dt": 0.02,                 # 参考数据帧间隔（与 env 控制 dt 对齐）
    "disc_hidden_dims": [1024, 512],
    "disc_activation": "relu",
    "disc_lr": 1.0e-4,                 # 关键：≪ policy lr，防判别器跑太快
    "num_learning_epochs": 1,          # 判别器 epoch（少，避免过拟合）
    "num_mini_batches": 4,
    "reward_scale": 2.0,
    "grad_penalty_coef": 10.0,         # 稳定性核心，别关
    "task_reward_coef": 1.0,
    "style_reward_coef": 0.5,
},
```

---

## 9. reward 退役计划（收益兑现）

AMP 的价值在于**替代手工步态塑形**。分两阶段验证：
1. **并存期**：AMP 开启，`gait_symmetry / anti_bound` 权重保持，确认判别器精度稳定
   在 ~0.6–0.8、`style_reward_mean` 有梯度、步态不退化。
2. **退役期**：把 `gait_symmetry=0`、`anti_bound=0`（保留 `foot_slip` 这类物理正则
   可选），重训，用 `eval_plot.py` 对比 duty_factor / 步态相位 / CoT / near_limit_frac。
   若 AMP 步态 ≥ 手工版，即兑现"少调参 + 更自然"。

`track_lin_vel_xy / track_ang_vel_z` 等**任务奖励永远保留**——AMP 只管风格，不管听不听指令。

---

## 10. 实现顺序（里程碑）

- **M0** `record_amp_motion.py` 录自举参考 → 出 `motion.npy`。
- **M1** env/adapter 产出 amp 转移对；storage 落盘；先不接判别器，确认形状/连续性
  （打印 `amp_obs` 均值方差、终止 mask 比例）。
- **M2** 判别器 + loader + HIMPPO.update 接通；TB 看 `disc_acc` 收敛、`grad_penalty` 不爆。
- **M3** style reward 融进 reward；`style_reward_coef` 从 0 → 0.5 渐进；步态目检。
- **M4** 退役手工步态项重训，`eval_plot.py` 对比。
- **M5** `him/play.py` + MuJoCo sim2sim 跑一遍确认部署一致（应零改动）。

---

## 11. 风险与排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `disc_acc` 迅速→1，style_reward→0 | 判别器太强 | 降 `disc_lr` / 减 `num_learning_epochs` / 升 `grad_penalty_coef` |
| loss 爆 / NaN | 无梯度惩罚或 amp_obs 未归一化 | 确认 `grad_penalty_coef>0` + normalizer 生效 |
| 步态没变化 | style_reward_coef 太小 / 参考数据太差 | 升权重；换更好参考源 |
| 蹦跳步态仍在 | 参考里就有 / 闭链被动关节进了 amp_obs | 检查参考；amp_obs 只用 actuated 关节（已排除 `projected_loop_joint_names`） |
| 终止处步态突变 | 终止转移没 mask | 确认 `amp_mini_batch_generator` 的 `not_done` 与 `_pre_reset_amp` |

---

## 12. 不做 AMP 的替代（轻量级）

若只想要"更自然 trot + 少调参"，成本低一个量级：**周期相位步态奖励**
（Margolis & Agrawal）——给每条腿一个相位时钟，奖励"该摆动时离地/该支撑时着地"，
一次替代 `gait_symmetry + anti_bound + foot_slip`，**无需任何参考数据**，
只改 `custom_rewards.py` + env，不碰 `him_rl`。建议先试它，不够再上 AMP。
