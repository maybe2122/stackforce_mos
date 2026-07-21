下面整理了一份**人形机器人（Humanoid）运动控制经典论文路线图**，按照技术发展的时间顺序进行分类，并附上 **arXiv/OpenReview** 链接（如果论文没有 arXiv，则给出官方论文链接）。

---

# 人形机器人运动控制里程碑论文

| 时间        | 论文                                                                                       | 核心贡献                                      | 推荐指数  | 链接                                                                                                     |
| --------- | ---------------------------------------------------------------------------------------- | ----------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------ |
| 2003      | **Biped Walking Pattern Generation by using Preview Control of Zero-Moment Point**       | 提出 Preview Control + ZMP，现代双足步行基础         | ⭐⭐⭐⭐⭐ | [https://ieeexplore.ieee.org/document/1215093](https://ieeexplore.ieee.org/document/1215093) （无 arXiv） |
| 2004–2012 | Whole-Body Control（Sentis 等）                                                             | 提出 Whole Body Control (WBC) 框架，多接触全身动力学控制 | ⭐⭐⭐⭐⭐ | 无 arXiv，主要发表于 ICRA/IROS                                                                                |
| 2010–2018 | Trajectory Optimization / Direct Collocation                                             | 将机器人运动规划转化为全局优化问题，是现代优化控制基础               | ⭐⭐⭐⭐  | 多篇综述，无统一 arXiv                                                                                         |
| 2018      | DeepMimic                                                                                | 强化学习+动作模仿，开启 RL 人形控制时代                    | ⭐⭐⭐⭐⭐ | [https://arxiv.org/abs/1804.02717](https://arxiv.org/abs/1804.02717) ([arXiv][1])                      |
| 2021      | AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control              | GAN 学习动作先验，无需复杂 Reward                    | ⭐⭐⭐⭐⭐ | [https://arxiv.org/abs/2104.02180](https://arxiv.org/abs/2104.02180) ([arXiv.gg][2])                   |
| 2021      | Rapid Motor Adaptation for Legged Robots                                                 | 在线估计环境参数，实现 Sim2Real 自适应                  | ⭐⭐⭐⭐⭐ | [https://arxiv.org/abs/2107.04034](https://arxiv.org/abs/2107.04034)                                   |
| 2023      | Robot Parkour Learning                                                                   | 首个统一视觉策略学习跑、跳、爬、钻等多技能（四足）                 | ⭐⭐⭐⭐⭐ | [https://arxiv.org/abs/2309.05665](https://arxiv.org/abs/2309.05665) ([arXiv][3])                      |
| 2024      | Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer | 人形机器人 RL 开源训练框架                           | ⭐⭐⭐⭐  | [https://arxiv.org/abs/2404.05695](https://arxiv.org/abs/2404.05695) ([Cool Papers][4])                |
| 2024      | Expressive Whole-Body Control for Humanoid Robots                                        | 利用大规模人体动作数据训练全身控制策略                       | ⭐⭐⭐⭐⭐ | [https://arxiv.org/abs/2402.16796](https://arxiv.org/abs/2402.16796) ([Hugging Face][5])               |
| 2024      | Humanoid Parkour Learning                                                                | 首个无需动作先验的人形机器人视觉跑酷策略                      | ⭐⭐⭐⭐⭐ | [https://arxiv.org/abs/2406.10759](https://arxiv.org/abs/2406.10759) ([arXiv][6])                      |

---

# 技术发展路线

## 第一阶段（2003–2012）：模型控制时代

代表论文：

* Kajita 2003（Preview Control）
* Whole Body Control（Sentis）

主要思想：

```
机器人动力学模型
        ↓
   COM 规划
        ↓
      ZMP
        ↓
Inverse Dynamics
        ↓
     Joint Torque
```

关键词：

* LIPM（Linear Inverted Pendulum Model）
* ZMP
* Whole Body Control
* QP Optimization

特点：

* 完全基于模型
* 稳定性强
* 动作较保守
* 工业机器人广泛采用

---

## 第二阶段（2012–2018）：优化控制时代

代表工作：

* Trajectory Optimization
* Direct Collocation
* MPC

主要思想：

```
整个动作

↓

一次优化

↓

得到所有关节轨迹
```

特点：

* 能生成复杂运动
* 可处理跳跃、跑步
* 计算量较大

Boston Dynamics Atlas 的许多高动态动作都建立在这类优化控制思想之上。

---

## 第三阶段（2018–2021）：强化学习时代

代表论文：

### DeepMimic

首次证明：

```
Motion Capture
        ↓
Reinforcement Learning
        ↓
Backflip
Cartwheel
Martial Arts
```

影响：

几乎所有后续 Motion Imitation 工作都引用了它。([arXiv][1])

---

### AMP

解决的问题：

以前需要设计大量 Reward：

```
Pose Reward

Foot Reward

Velocity Reward

...
```

AMP：

```
Motion Dataset

↓

Discriminator

↓

Style Reward
```

机器人自动学习动作风格。([arXiv.gg][2])

---

### RMA

提出：

```
Observation

↓

Adaptation Module

↓

Latent Dynamics

↓

Policy
```

机器人无需知道：

* 摩擦系数
* 质量
* 地形

运行时自动估计，实现更稳健的 Sim-to-Real。

---

## 第四阶段（2023–2024）：多技能视觉控制

代表论文：

### Robot Parkour Learning

首次实现：

一个 Policy 学会：

* Run
* Jump
* Crawl
* Climb
* Squeeze

而不是多个 Skill Policy。([arXiv][3])

---

### Humanoid Parkour Learning

进一步扩展到人形机器人：

```
Depth Camera

↓

Vision Encoder

↓

Whole Body Policy

↓

Joint Action
```

机器人可以：

* 跳高台
* 跳沟
* 跑步
* 上坡
* 下坡
* 越障

全部由一个端到端视觉策略完成。([arXiv][6])

---

### ExBody

目标：

不是简单行走，而是生成**自然的人类风格动作**。

例如：

* 挥手
* 跳舞
* 握手
* 不同步态风格

训练方式：

```
Human Motion Dataset

↓

RL

↓

Whole-body Policy

↓

Real Robot
```

([Hugging Face][5])

---

# 推荐阅读顺序（适合科研入门）

如果希望系统进入人形机器人运动控制方向，建议按下面顺序阅读：

### 第一阶段：经典控制理论

1. Kajita (2003) —— Preview Control + ZMP
2. Whole Body Control（Sentis）
3. Trajectory Optimization / MPC

> 建立机器人动力学与优化控制基础。

---

### 第二阶段：强化学习基础

4. DeepMimic（2018）
5. AMP（2021）
6. Rapid Motor Adaptation（2021）

> 理解现代 RL 如何学习稳定、自然的运动。

---

### 第三阶段：视觉与多技能控制

7. Robot Parkour Learning（2023）
8. Humanoid-Gym（2024）
9. ExBody（2024）
10. Humanoid Parkour Learning（2024）

> 掌握当前端到端视觉控制、多技能学习和 Sim-to-Real 的主流方向。

---

## 如果只能读 10 篇

这是我最推荐的阅读清单：

1. Preview Control of ZMP（2003）
2. Whole Body Control（Sentis）
3. Trajectory Optimization / Direct Collocation
4. DeepMimic（2018）
5. AMP（2021）
6. Rapid Motor Adaptation（2021）
7. Robot Parkour Learning（2023）
8. Humanoid-Gym（2024）
9. Expressive Whole-Body Control（ExBody，2024）
10. Humanoid Parkour Learning（2024）

这条路线完整覆盖了**模型控制 → 优化控制 → 强化学习 → 模仿学习 → 视觉感知 → 多技能统一策略**的发展脉络，也是目前人形机器人运动控制研究最重要的知识体系。

[1]: https://arxiv.org/abs/1804.02717?utm_source=chatgpt.com "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills"
[2]: https://arxiv.gg/abs/2104.02180?utm_source=chatgpt.com "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control - arXiv.gg"
[3]: https://arxiv.org/abs/2309.05665?utm_source=chatgpt.com "Robot Parkour Learning"
[4]: https://papers.cool/arxiv/2404.05695?utm_source=chatgpt.com "Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer | Cool Papers - Immersive Paper Discovery"
[5]: https://huggingface.co/papers/2402.16796?utm_source=chatgpt.com "Paper page - Expressive Whole-Body Control for Humanoid Robots"
[6]: https://arxiv.org/abs/2406.10759?utm_source=chatgpt.com "Humanoid Parkour Learning"
