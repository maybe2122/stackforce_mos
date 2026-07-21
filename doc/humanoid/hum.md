这是目前**人形机器人强化学习（RL）运动控制最核心的问题之一**。实际上，**仅依靠奖励函数（Reward Engineering）很难训练出自然、美观、接近人类的步态**。过去几年（尤其是 2023–2026），研究趋势已经从"调奖励"逐渐转向**引入先验（Prior）**。

如果按照效果和目前学术界的发展路线来看，可以把技术路线分为几个层次。

---

# 第一阶段：纯 RL + Reward Engineering（你现在的阶段）

这是 Isaac Gym、Isaac Lab、Unitree、RSL-RL 最经典的做法。

奖励通常包括：

```
Tracking reward
Foot clearance
Foot slip
Feet air time
Energy penalty
Action smoothness
Torque penalty
Joint limit penalty
Orientation penalty
Base height
```

但是，这种方法存在天然的问题：

**奖励只能约束"做什么"，很难约束"怎么做"。**

例如：

目标：

```
速度 = 1.0m/s
```

机器人可能会：

* 歪着身子走
* 两腿外八字
* 上身左右摇摆
* 手臂完全不动
* 小碎步
* 高抬腿

因为这些都没有违反奖励。

所以你会发现：

> **Reward 可以保证能走，但很难保证走得漂亮。**

---

# 第二阶段：增加更多姿态约束（推荐先尝试）

很多实验室都会增加一些专门针对人体姿态的奖励，而不是继续调 tracking reward。

例如：

## （1）保持身体直立

```
R = exp(-||roll,pitch||)
```

限制：

```
base roll
base pitch
```

不是只限制 torso，而是限制整个 COM。

---

## （2）髋部对称

很多人形机器人会出现：

```
左腿：
正常

右腿：
一直外摆
```

加入

```
left hip ≈ - right hip
```

或者

```
left knee ≈ right knee
```

的对称奖励。

---

## （3）关节默认姿态（Posture Prior）

例如：

```
Hip
Shoulder
Elbow
```

不要偏离默认站姿。

IsaacLab里面经常有：

```
joint_deviation_l1
```

就是干这个。

---

## （4）头部保持水平

限制：

```
Neck

Head

Torso
```

Roll Pitch。

这样不会出现：

```
歪着脑袋
```

---

## （5）COM位于支撑多边形

很多机器人走路像企鹅。

其实是：

COM左右摇摆太大。

增加

```
CoM lateral deviation penalty
```

效果会明显改善。

---

# 第三阶段：加入步态先验（目前主流）

这是近几年变化最大的地方。

不再让 RL 从零学习。

而是告诉它：

```
人就是这么走路的。
```

例如：

Human Motion Dataset

↓

Motion Prior

↓

RL

例如：

```
AMASS

CMU Mocap

Human3.6M
```

机器人不是瞎探索。

而是：

```
参考人类动作
```

再学会保持平衡。

这一步效果通常远大于继续调 Reward。

---

# 第四阶段：AMP（目前最经典）

这是目前最推荐的路线。

AMP = Adversarial Motion Priors

思想非常简单：

```
RL

+

GAN
```

训练一个判别器：

```
真人动作

VS

机器人动作
```

如果机器人动作：

```
不像人
```

判别器给负奖励。

如果：

```
像人
```

判别器给高奖励。

于是最终奖励变成：

```
Task Reward

+

Motion Prior Reward
```

不是：

```
Task Reward
```

论文：

* **AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control**（2021）

这是后面很多工作的基础。

---

# 第五阶段：Mimic Learning（2024–2026 最热门）

目前 Google DeepMind、NVIDIA、Figure AI 等大量工作都采用：

```
Motion Tracking

+

RL
```

即：

```
给机器人一个参考动作

↓

RL 去跟踪
```

奖励：

```
Joint Position

Joint Velocity

End Effector

COM

Orientation
```

全部来自参考动作。

这样机器人自然就会：

```
越来越像人
```

---

# 第六阶段：Residual RL

目前工业界越来越喜欢。

例如：

```
Reference Controller

+

Residual RL
```

控制器：

```
正常走路
```

RL：

```
只负责修正
```

所以：

```
Policy

=

Reference

+

Residual
```

训练非常稳定。

---

# 第七阶段：Phase-based Walking

很多机器人已经不用：

```
cmd→policy
```

而是：

```
Phase

↓

Policy
```

例如：

```
phase

sinφ

cosφ
```

Policy知道：

```
现在左脚应该摆动

现在右脚应该落地
```

而不是自己猜。

自然很多。

---

# 第八阶段：加入上半身控制

很多RL机器人：

```
腿很好

手完全不会摆
```

于是：

```
加 Arm Swing Reward
```

或者：

```
Shoulder follows gait phase
```

整个动作自然很多。

---

# 近几年最值得阅读的论文

如果你的目标是**训练出自然的人形步态**，下面这些论文非常值得按顺序阅读：

| 论文                                                                           | 核心思想            | 推荐指数  |
| ---------------------------------------------------------------------------- | --------------- | ----- |
| AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control  | 用对抗判别器学习自然动作先验  | ⭐⭐⭐⭐⭐ |
| Learning Humanoid Locomotion over Challenging Terrain                        | 地形行走与自然步态结合     | ⭐⭐⭐⭐⭐ |
| Humanoid Parkour Learning                                                    | 高动态人形运动，结合模仿与RL | ⭐⭐⭐⭐⭐ |
| Robot Parkour Learning                                                       | 高难度全身控制与泛化      | ⭐⭐⭐⭐☆ |
| PHC: Universal Humanoid Control through Physics-Based Reinforcement Learning | 大规模动作模仿与物理控制    | ⭐⭐⭐⭐⭐ |
| MaskedMimic                                                                  | 稀疏目标条件下的人形动作模仿  | ⭐⭐⭐⭐⭐ |

---

# 结合你目前的情况（纯 RL、人形行走）

基于我们之前的交流，我知道你目前主要在 **Isaac Lab / MuJoCo / RSL-RL** 环境中做人形机器人运控，并且已经在调奖励函数。

如果希望在**不引入模仿学习数据**的前提下，优先提升步态自然性，我建议按下面的顺序投入精力：

1. **先检查观测与动作设计**：确保策略能观察到足够的姿态信息（如相位、关节历史、接触状态），动作输出平滑（动作延迟、低通滤波、Action Rate Penalty）。
2. **优化仿真真实性**：域随机化、执行器模型、控制频率、PD 参数、摩擦参数、质量惯量等，这些往往比再增加一个奖励项更有效。
3. **加入姿态与对称性奖励**：如 pelvis/torso 保持竖直、左右腿关节对称、COM 横向摆动约束、手臂自然摆动等。
4. **引入步态相位（Gait Phase）**：即使没有动作数据，也可以用 `sin(phase)`、`cos(phase)` 引导左右脚交替，通常会让步态更规律。
5. **如果允许引入先验，优先尝试 AMP 或 Mimic Learning**：这是目前学术界和工业界获得自然步态最有效的技术路线。

---

## 我建议你的学习路线

如果目标是 **2026 年的人形机器人运动控制**，我会按下面的顺序学习和实践：

1. **把纯 RL 做到优秀**（奖励、观测、动作、课程学习、仿真参数）。
2. **深入研究 Phase-based Gait**（步态相位、步态生成器）。
3. **学习 AMP**（动作先验）。
4. **学习 Mimic Learning / Motion Tracking**（动作模仿）。
5. **最后学习 Foundation Policy / 大规模预训练策略**（2025–2026 的最新趋势）。

这条路线基本覆盖了目前 NVIDIA、DeepMind、Figure AI、Agility Robotics 等团队在人形运动控制上的主流技术方向。 再细化这端总结