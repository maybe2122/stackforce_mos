# deploy/common — 已迁移

原本这里的运动学 / 步态 / 动力学模块（`closed_chain_kin.py`、`gait.py`、
`kinematics.py`、`speed_map.py`、`dynamics.py`）已于 2026-07-21 整体迁到仓库根目录的
**`传统步态/`** bundle，与文档、图/数据放在一起：

```
传统步态/
  代码/        ← 上述模块 + walk_gait.py（MuJoCo 行走主循环）
  文档/        ← 控制栈说明、闭链运动学详解、动力学选型、步态评价…
  图与数据/    ← gait_demo / speed_map / dynamics 的出图与 CSV
```

运行入口与说明见 **`传统步态/文档/闭链运动学与MuJoCo行走.md`** 与
**`传统步态/文档/control_stack.md`**。

> 这些模块之间靠同目录 bare import（`from gait import …`），所以整组一起搬、
> 仍能独立运行。外部引用方（`tools/isaac/speed_viz_isaac.py`）的 sys.path 已同步更新。
