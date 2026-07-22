# 传统步态 baseline（无 RL · 闭链运动学 + trot + 位置伺服）

mos2026_2 的**经典运控栈**：不训练策略，纯运动学 + 步态 + PD/位置伺服，在 MuJoCo 里
把四足走起来。代码、文档、图/数据都收在本文件夹里，自成一体。
速度命令 → 步态生成 → 足端轨迹 → 闭链逆运动学 → 关节目标 → MuJoCo位置控制 → 机器人运动
```
传统步态/
  代码/            纯 numpy 解算器 + MuJoCo 行走主循环
    closed_chain_kin.py   闭链腿解析 FK + 数值 IK + Jacobian（与 MuJoCo 交叉验证 1e-8）
    gait.py               trot 足端轨迹 + 全向指令（vx/vy/wz）
    walk_gait.py          MuJoCo 行走主循环 + 六工况验收
    speed_map.py          机身速度 → 电机转速可行性
    dynamics.py           静力递推 + 减速比选型
    kinematics.py         早期串联近似（已退役，dynamics 仍依赖）
  文档/            控制栈说明、闭链运动学详解、动力学选型、步态评价、电机规格…
  图与数据/        gait_demo / speed_map / dynamics 的出图与 CSV
```

## 快速开始（都从仓库根目录跑）

```bash
PY=/home/maybe/code/rl/env_isaaclab/bin/python   # torch+mujoco 的 venv

# 行走（默认 vx≈0.4 m/s，达成率 ~88%）
$PY 传统步态/代码/walk_gait.py --headless -T 10
$PY 传统步态/代码/walk_gait.py --headless --vx 0.6        # 指定前进速度
$PY 传统步态/代码/walk_gait.py --headless --vx 0 --wz 0.5  # 原地转
MUJOCO_GL=glx $PY 传统步态/代码/walk_gait.py --viewer      # 开窗看
$PY 传统步态/代码/walk_gait.py --sweep -T 10               # 六工况验收

# 自测
$PY 传统步态/代码/closed_chain_kin.py --selftest
$PY 传统步态/代码/gait.py --selftest
$PY 传统步态/代码/speed_map.py --selftest

# 出图（写进 传统步态/图与数据/）
$PY 传统步态/代码/gait.py --demo
$PY 传统步态/代码/dynamics.py --plot
```

## 从这里开始读

- **[文档/闭链运动学与MuJoCo行走.md](文档/闭链运动学与MuJoCo行走.md)** —— 底层实现详解：
  从 USD 拿长度、闭链 FK/IK 推导、步态、控制环、调参实录、测试步骤（编号分步）。
- [文档/control_stack.md](文档/control_stack.md) —— 控制栈总览。
- [文档/dynamics_gear_ratio_analysis.md](文档/dynamics_gear_ratio_analysis.md) —— 动力学与减速比选型。
- **[文档/真机部署-01-角度映射标定与吊空验证.md](文档/真机部署-01-角度映射标定与吊空验证.md)** ——
  往真机搬的第一步方案（吊空、验证复用已有 `motor_bus` 映射，不下地）。

## 说明

- 代码之间靠**同目录 bare import**（`from gait import …`），所以整组必须放在一起。
- 场景/资产仍在 `deploy/mujoco/assets/`（未移动），`walk_gait.py` 按仓库根定位它。
- 外部引用方 `tools/isaac/speed_viz_isaac.py` 的 sys.path 已指向 `传统步态/代码/`。
- 原 `deploy/common/` 仅留一个指向本处的 README 存根。
