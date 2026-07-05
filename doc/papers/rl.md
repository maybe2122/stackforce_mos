# 强化学习经典论文阅读路线（面向机器人运动控制）

> **目标读者**：做**机器人运动控制**（连续动作空间）的人。
> 这类任务的动作是连续的关节力矩 / 速度，因此主线落在 **DDPG → TD3 → SAC → TQC/REDQ**（off-policy 连续控制）与 **TRPO → PPO**（on-policy）两条线上。
> 下面是 **Actor-Critic 领域最经典、影响最大的算法**，正文**按论文发表时间排序**；正式开始前先补一节**前置基础**。

---

## 📚 论文速查表（按时间排序）

| 时间 | 算法 | 论文 | 链接 |
| :--: | :-- | :-- | :-- |
| 2013.12 | DQN | Playing Atari with Deep Reinforcement Learning | [arXiv:1312.5602](https://arxiv.org/abs/1312.5602) ¹ |
| 2015.02 | TRPO | Trust Region Policy Optimization | [arXiv:1502.05472](https://arxiv.org/abs/1502.05472) ³ |
| 2015.09 | DDPG | Continuous Control with Deep Reinforcement Learning | [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) |
| 2015.09 | Double DQN | Deep Reinforcement Learning with Double Q-learning | [arXiv:1509.06461](https://arxiv.org/abs/1509.06461) |
| 2016.01 | AlphaGo | Mastering the game of Go with deep neural networks and tree search | [Nature](https://www.nature.com/articles/nature16961) ² |
| 2016.02 | A3C | Asynchronous Methods for Deep Reinforcement Learning | [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) |
| 2017.07 | PPO | Proximal Policy Optimization Algorithms | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| 2017.12 | AlphaZero | Mastering Chess and Shogi by Self-Play with a General RL Algorithm | [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) |
| 2018.01 | SAC | Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor | [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) |
| 2018.02 | TD3 | Addressing Function Approximation Error in Actor-Critic Methods | [arXiv:1802.09477](https://arxiv.org/abs/1802.09477) |
| 2020.05 | TQC | Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics | [arXiv:2005.04269](https://arxiv.org/abs/2005.04269) |
| 2021.01 | REDQ | Randomized Ensembled Double Q-Learning: Learning Fast Without a Model | [arXiv:2101.05982](https://arxiv.org/abs/2101.05982) |

> ¹ 此处 arXiv 链接为 DQN 的前身 NeurIPS 2013 工作站版《Playing Atari with Deep Reinforcement Learning》；其 Nature 正式版《Human-level control through deep reinforcement learning》（[Nature 2015](https://www.nature.com/articles/nature14236)）未在 arXiv 发布，两者思想一致。
> ² AlphaGo 仅发表于 Nature，无 arXiv 版本。
> ³ TRPO 是 PPO 的直接前身，详见下方「前置基础 §0.5」。

---

## 0. 前置基础（读论文前必须先懂的概念）

下面这些不是某一篇论文，而是贯穿所有算法的**通用语言**。机器人运动控制基本只关心**连续动作**，所以这里以连续控制为落点来讲。

### 0.1 MDP：强化学习的问题框架

把控制问题写成 **马尔可夫决策过程（MDP）** $\langle S, A, P, r, \gamma\rangle$：

- $S$ 状态（如关节角、角速度、躯干姿态）
- $A$ 动作（机器人里是**连续**的关节力矩 / 目标位置）
- $P(s'\mid s,a)$ 环境动力学（仿真器 / 真实机器人）
- $r(s,a)$ 奖励（如前进速度、保持平衡、能耗惩罚）
- $\gamma\in[0,1)$ 折扣因子

目标：找一个策略 $\pi$ 最大化期望累积回报

$$
J(\pi)=\mathbb{E}_{\pi}\Big[\textstyle\sum_{t=0}^{\infty}\gamma^{t} r(s_t,a_t)\Big]
$$

### 0.2 价值函数与 Bellman 方程

- **状态价值** $V^\pi(s)$：从 $s$ 出发按 $\pi$ 走的期望回报
- **动作价值** $Q^\pi(s,a)$：在 $s$ 先执行 $a$、之后按 $\pi$ 走的期望回报
- **优势** $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$：动作 $a$ 比平均好多少（PPO 用它）

满足 **Bellman 方程**（所有值方法的根）：

$$
Q^\pi(s,a)=r(s,a)+\gamma\,\mathbb{E}_{s',a'}\big[Q^\pi(s',a')\big]
$$

### 0.3 两大流派

| 流派 | 学什么 | 代表 | 连续动作？ |
| :-- | :-- | :-- | :-- |
| **值方法** | 学 $Q$，再 $\arg\max_a Q$ 选动作 | Q-learning → DQN → Double DQN | ❌ 连续动作下 $\arg\max$ 无法穷举 |
| **策略梯度** | 直接学策略 $\pi_\theta(a\mid s)$ | REINFORCE → TRPO → PPO | ✅ 天然支持 |

> 这正是机器人运动控制几乎不用纯 DQN 的原因：连续动作空间里对 $Q$ 求 $\arg\max$ 不可行。
> 解决办法有两条——要么像 **DDPG** 用一个 Actor 网络去逼近这个 $\arg\max$，要么走**策略梯度（PPO/SAC）**。

#### 策略梯度定理

策略梯度方法的理论基石，告诉你怎么对策略参数求梯度：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a\mid s)\,A^\pi(s,a)\big]
$$

- **REINFORCE**：最朴素实现，方差极大、样本效率低。
- 工程上几乎不直接用，但 PPO/SAC 的目标函数都从这里推导而来。

### 0.4 Actor-Critic：本文档主线的由来

把两大流派**合二为一**：

- **Actor**（策略 $\pi_\theta$）：负责输出动作 —— 来自策略梯度
- **Critic**（价值 $Q_\phi$ 或 $V_\phi$）：评估动作好坏，给 Actor 提供低方差的学习信号 —— 来自值方法

> 后面 **DDPG / TD3 / SAC / PPO** 全部是 Actor-Critic 结构，区别只在于 Critic 怎么估、Actor 怎么更新。

### 0.5 TRPO：PPO 的直接前身

- **论文**：《Trust Region Policy Optimization》 · [arXiv:1502.05472](https://arxiv.org/abs/1502.05472)（2015.02，Schulman 等）
- **核心**：用 **KL 散度信赖域**约束每次策略更新幅度，保证单调改进，避免一步走崩。
- **问题**：要算二阶信息（Fisher 矩阵 / 共轭梯度），实现复杂、计算重。
- **PPO 的意义**：用一阶的 **clip** 近似 TRPO 的信赖域，效果相近但简单得多——所以读完 TRPO 思想再看 PPO 会非常顺。

> **机器人运动控制的最小前置**：理解 0.1–0.2（MDP + Bellman）和 0.4（Actor-Critic）即可上手 DDPG/TD3/SAC；
> 想吃透 PPO 再补 0.3 的策略梯度定理与 0.5 的 TRPO。

---

## 2013.12 — DQN

- **论文**：《Playing Atari with Deep Reinforcement Learning》（2013 前身版）
- **链接**：[arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
- **Nature 正式版**：《Human-level control through deep reinforcement learning》（[Nature 2015](https://www.nature.com/articles/nature14236)，未上 arXiv）
- **作者**：Volodymyr Mnih 等
- **定位**：深度强化学习起点

**贡献**：第一次让深度神经网络成功玩 Atari 游戏。

**核心**：直接用神经网络拟合 $Q(s,a)$。

**提出两大技巧**（现在几乎所有 RL 算法都在用）：

- Experience Replay（经验回放）
- Target Network（目标网络）

---

## 2015.09 — DDPG

- **论文**：《Continuous Control with Deep Reinforcement Learning》
- **链接**：[arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- **定位**：连续动作空间的开山之作，TD3/SAC 祖先

**贡献**：把 DQN 思想扩展到**连续动作**，第一次成功实现下面这套结构：

```text
Actor + Critic + Replay Buffer
```

很多后续算法（**TD3**、**SAC**）都直接继承自 DDPG。

---

## 2015.09 — Double DQN

- **论文**：《Deep Reinforcement Learning with Double Q-learning》
- **链接**：[arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
- **定位**：双 Q 思想来源

**问题**：DQN 中的 $\max_a Q(s,a)$ 会**高估**动作价值。

**改进**——把「选动作」和「评估动作」拆给两个网络：

$$
y = r + \gamma\, Q_{\theta^-}\!\left(s',\ \arg\max_a Q_\theta(s',a)\right)
$$

即：

- 用**在线网络** $Q_\theta$ **选动作**：$a^* = \arg\max_a Q_\theta(s',a)$
- 用**目标网络** $Q_{\theta^-}$ **评估**该动作的价值

对比原始 DQN 的目标 $y = r + \gamma\,\max_a Q_{\theta^-}(s',a)$——后者「选」和「评」用同一个网络，导致系统性高估。

> 这是后来 TD3 双 Critic 思想的祖先。

---

## 2016.01 — AlphaGo

- **论文**：《Mastering the game of Go with deep neural networks and tree search》
- **链接**：[Nature](https://www.nature.com/articles/nature16961)
- **定位**：搜索 + RL 的标志性突破

**核心**：

```text
Policy Network + Value Network + MCTS
```

---

## 2016.02 — A3C

- **论文**：《Asynchronous Methods for Deep Reinforcement Learning》
- **链接**：[arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
- **定位**：让 Actor-Critic 普及

**核心**：用多个并行 worker 异步采样与更新，打破样本相关性，无需 Replay Buffer 即可稳定训练。

---

## 2017.07 — PPO

- **论文**：《Proximal Policy Optimization Algorithms》
- **链接**：[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **作者**：John Schulman 等
- **定位**：工业界和学术界最流行的算法之一

**核心**——限制策略更新幅度。定义概率比：

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
$$

通过 clip 防止训练崩掉：

$$
\text{clip}\big(r_t,\ 1-\epsilon,\ 1+\epsilon\big)
$$

**特点**：稳定、容易调参，OpenAI、DeepMind 大量使用。

很多大模型 RLHF 都走这条路线：

```text
PPO  →  RLHF  →  ChatGPT
```

---

## 2017.12 — AlphaZero

- **论文**：《Mastering Chess and Shogi by Self-Play with a General RL Algorithm》
- **链接**：[arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
- **定位**：搜索 + RL 巅峰

**训练过程**：

```text
自我对弈  →  MCTS  →  Policy Learning
```

> 成为现代「搜索 + RL」结合的经典范式。

---

## 2018.01 — SAC

- **论文**：《Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor》
- **链接**：[arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- **定位**：连续控制霸主

> 很多人认为：**SAC = TD3 的升级版**

**核心思想**——最大熵框架，同时最大化奖励与熵：

$$
\mathbb{E}[R] + \alpha\, H(\pi)
$$

不仅要高奖励，还要高熵 $H(\pi)$，即**鼓励探索**。

| | 策略类型 | 形式 |
| :-- | :-- | :-- |
| TD3 | 确定性策略 | $a = \pi(s)$ |
| SAC | 随机策略 | $a \sim \pi(a\mid s)$ |

**优点**：

- 更稳定
- 样本效率高
- 超参数不敏感

> 现在机器人控制领域最常见的 baseline 之一。

---

## 2018.02 — TD3

- **论文**：《Addressing Function Approximation Error in Actor-Critic Methods》
- **链接**：[arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
- **定位**：解决 Actor-Critic 高估
- 你刚看的这篇。

**三大核心改进**（解决 DDPG 不稳定问题）：

- Twin Critic（双 Critic）
- Delayed Update（延迟更新）
- Target Smoothing（目标平滑）

---

## 2020.05 — TQC

- **论文**：《Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics》
- **链接**：[arXiv:2005.04269](https://arxiv.org/abs/2005.04269)
- **定位**：SAC 增强版

**背景**：TD3 把高估解决了，但大家又发现开始**低估**。

```text
高估  →  解决了  →  开始低估
```

**思想**：把高估的部分 quantile 直接砍掉。效果非常强，机器人控制中经常超过 TD3。

---

## 2021.01 — REDQ

- **论文**：《Randomized Ensembled Double Q-Learning: Learning Fast Without a Model》
- **链接**：[arXiv:2101.05982](https://arxiv.org/abs/2101.05982)
- **定位**：现代高样本效率

**思想**：不是两个 Critic，而是很多个。

```text
10 个 Q 网络  →  随机抽样  →  取最小
```

> 大幅提升样本效率。

---

## 🏆 按「历史地位」排序的 Top 10

| 排名 | 算法 | 地位 |
| :--: | :-- | :-- |
| 1 | DQN | 深度 RL 起点 |
| 2 | PPO | 最广泛使用 |
| 3 | SAC | 连续控制霸主 |
| 4 | TD3 | 解决 Actor-Critic 高估 |
| 5 | DDPG | TD3/SAC 祖先 |
| 6 | Double DQN | 双 Q 思想来源 |
| 7 | AlphaZero | 搜索 + RL 巅峰 |
| 8 | A3C | Actor-Critic 普及 |
| 9 | REDQ | 现代高样本效率 |
| 10 | TQC | SAC 增强版 |

---

## 🧭 推荐阅读顺序

> 注意：阅读顺序≠发表时间。为理解思想演进，推荐按下面的**逻辑主线**阅读
> （尤其面向机器人控制和大模型 RL）。

```text
DQN
 ↓
Double DQN
 ↓
DDPG
 ↓
TD3
 ↓
SAC
 ↓
TRPO
 ↓
PPO
 ↓
AlphaZero
 ↓
REDQ / TQC
```

这样你会清楚看到一条主线：

```text
DQN        ↓ 解决离散动作
DDPG       ↓ 扩展到连续动作
TD3        ↓ 解决 Q 高估
SAC        ↓ 引入最大熵框架
REDQ/TQC   ↓ 进一步改进 Critic 估计
```

> 读完 **DDPG → TD3 → SAC** 这三篇，你基本就掌握了现代连续控制强化学习最核心的思想。
