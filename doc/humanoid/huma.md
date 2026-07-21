如果按照**人形机器人运动控制（Physics-based Humanoid Control）**的发展脉络，近几年最经典的一条路线可以整理为：

> **DeepMimic → AMP → UHC → PHC → PULSE → 闭环数据生成（2026）**

下面给出每个阶段最值得读的论文（按推荐顺序）。

---

# ① DeepMimic（现代 Motion Imitation 的起点）

**DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills**

* SIGGRAPH 2018
* 作者：Xue Bin Peng 等
* arXiv：[https://arxiv.org/abs/1804.02717](https://arxiv.org/abs/1804.02717)

贡献：

* 第一篇真正意义上的 Physics-based Motion Imitation
* 使用 PPO + Reference Motion
* 奠定了后面所有工作的基础

创新：

```
Reference Motion
      │
      ▼
Reward Tracking
      │
      ▼
PPO
```

几乎所有 AMP、PHC 都继承了它的 imitation reward。

---

# ② AMP（2021）

**Adversarial Motion Priors for Stylized Physics-Based Character Control**

* SIGGRAPH 2021
* 作者：Xue Bin Peng 等
* arXiv：[https://arxiv.org/abs/2104.02180](https://arxiv.org/abs/2104.02180)

贡献：

首次把 GAN 引入强化学习运动控制。

核心思想：

```
Reference Motion

↓

Discriminator

↓

Style Reward

↓

PPO
```

特点：

* 不再逐帧tracking
* 学习 Motion Prior
* 动作更加自然

这是目前机器人 imitation RL 最经典论文之一。

---

# ③ UHC（2022）

**Universal Humanoid Control**

* SIGGRAPH Asia 2022
* arXiv：[https://arxiv.org/abs/2205.01906](https://arxiv.org/abs/2205.01906)

贡献：

首次实现

> 一个 Policy 模仿上万段 AMASS 动作。

以前：

```
一个动作
↓

一个Policy
```

UHC：

```
10000 Motion

↓

One Policy
```

提出：

Motion-conditioned Policy

---

# ④ PHC（2023）

**Perpetual Humanoid Control for Real-time Simulated Avatars**

* ICCV 2023
* arXiv：[https://arxiv.org/abs/2309.06489](https://arxiv.org/abs/2309.06489)
* GitHub：[PHC Repository](https://github.com/ZhengyiLuo/PHC?utm_source=chatgpt.com)

贡献：

解决了 UHC 最大问题：

> 一摔倒就 Reset。

PHC：

```
Fall

↓

Stand up

↓

Continue
```

提出：

* PMCP
* Progressive Primitive
* Get-up
* Recovery

真正实现

Perpetual Control。([GitHub][1])

---

# ⑤ PULSE（2024）

**Universal Humanoid Motion Representations for Physics-Based Control**

* ICLR 2024 Spotlight
* arXiv：[https://arxiv.org/abs/2310.04582](https://arxiv.org/abs/2310.04582)
* GitHub：[PULSE Repository](https://github.com/ZhengyiLuo/PULSE?utm_source=chatgpt.com)

贡献：

学习统一 Motion Latent Space。

以前：

```
Task

↓

Policy
```

现在：

```
Motion

↓

Latent Space

↓

Task Policy
```

论文提出：

Physics-based Universal motion Latent SpacE

简称：

PULSE。([GitHub][2])

---

# ⑥ PHC+（2024）

虽然没有单独论文，

但 PHC GitHub 发布了

PHC+

包括：

* 100% AMASS success
* 更好的 recovery
* PULSE 默认 backbone

官方仓库已有说明。([GitHub][1])

---

# ⑦ Iterative Closed-Loop Motion Synthesis（2026）

**Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control**

* arXiv 2026
* [https://arxiv.org/abs/2602.21599](https://arxiv.org/abs/2602.21599)

贡献：

提出

```
Policy

↓

Generate Motion

↓

Train Again

↓

Generate Better Motion
```

实现

Data ↔ Policy

共同进化。

这是目前 PHC/PULSE 最新的发展方向之一。([arXiv Troller][3])

---

# ⑧ Spherical Latent Motion Prior（2026）

**Spherical Latent Motion Prior for Physics-Based Simulated Humanoid Control**

* arXiv 2026
* [https://arxiv.org/abs/2603.01294](https://arxiv.org/abs/2603.01294)

贡献：

认为：

AMP

和

PULSE(VAE)

都存在 latent distribution 的问题，

提出

Sphere Latent

使 Motion Prior 更稳定。([ResearchGate][4])

---

# 整体技术演化图

```text
DeepMimic (2018)
        │
        ▼
 Motion Tracking
        │
        ▼
AMP (2021)
        │
        ▼
 Adversarial Motion Prior
        │
        ▼
UHC (2022)
        │
        ▼
 Universal Motion Tracking
        │
        ▼
PHC (2023)
        │
        ▼
 Recovery + Perpetual Control
        │
        ▼
PULSE (2024)
        │
        ▼
 Universal Motion Latent Space
        │
        ▼
PHC+ (2024)
        │
        ▼
Closed-loop Motion Generation (2026)
        │
        ▼
Spherical Motion Prior (2026)
```

## 如果你的研究方向是**人形机器人强化学习运动控制**，建议按照下面的阅读顺序：

1. **DeepMimic (2018)** —— 理解参考动作跟踪（Motion Tracking）的基础。
2. **AMP (2021)** —— 学习对抗运动先验（Adversarial Motion Prior），理解为什么 GAN 能提升动作自然性。
3. **UHC (2022)** —— 学习统一策略如何覆盖海量动作。
4. **PHC (2023)** —— 学习持续控制（Perpetual Control）、跌倒恢复和 PMCP。
5. **PULSE (2024)** —— 学习通用动作潜空间（Universal Motion Representation），为下游任务复用运动表示。([GitHub][2])
6. **2026 两篇工作** —— 了解当前最前沿的自动数据生成与新型 Motion Prior。([ResearchGate][4])

这一套论文基本覆盖了 **2018–2026 年 Physics-based Humanoid Control** 最重要的技术演进，也是目前许多机器人实验室（如 NVIDIA、CMU、Berkeley 等）相关工作的核心知识体系。

[1]: https://github.com/ZhengyiLuo/PHC?utm_source=chatgpt.com "GitHub - ZhengyiLuo/PHC: Official Implementation of the ICCV 2023 paper: Perpetual Humanoid Control for Real-time Simulated Avatars · GitHub"
[2]: https://github.com/ZhengyiLuo/PULSE?utm_source=chatgpt.com "GitHub - ZhengyiLuo/PULSE: Official Implementation of the ICLR 2024 spotlight paper: Universal Humanoid Motion Representations for Physics-Based Control · GitHub"
[3]: https://arxiv-troller.com/paper/3098541/?utm_source=chatgpt.com "Iterative Closed-Loop Motion Synthesis for Scalin… - arXiv"
[4]: https://www.researchgate.net/publication/401468959_Spherical_Latent_Motion_Prior_for_Physics-Based_Simulated_Humanoid_Control?utm_source=chatgpt.com "(PDF) Spherical Latent Motion Prior for Physics-Based Simulated Humanoid Control"
