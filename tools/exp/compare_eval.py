#!/usr/bin/env python3
"""对比多个 eval.json 的指标，输出到终端并写一份 Markdown 报告。

eval.json 由 scripts/rsl_rl/eval.py 生成，结构为 {condition:{...}, metrics:{...}}。
本脚本把每个 run 当成一列，指标当成行，做并排对比；并对“越大越好/越小越好”
的指标标出各行最优值（终端加 * / ★，Markdown 加 **粗体**）。

用法:
    # 直接给若干 eval.json（或包含 eval.json 的目录，或 glob）
    python tools/exp/compare_eval.py runA/eval/eval.json runB/eval/eval.json
    python tools/exp/compare_eval.py logs/rsl_rl/mos2026_2_closed_usd/*/eval

    # 自定义每列标签（与文件顺序一一对应）
    python tools/exp/compare_eval.py a.json b.json --labels baseline dr

    # 指定输出 md 路径（默认写到第一个文件同级的 compare_eval.md）
    python tools/exp/compare_eval.py *.json -o /tmp/cmp.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

# 指标方向：+1 越大越好，-1 越小越好，0 中性（不标最优）。
# 未列出的标量指标默认按中性处理。
METRIC_DIR: dict[str, int] = {
    "success_rate": +1,
    "mean_survival_s": +1,
    "mean_return": +1,
    "fall_rate": -1,
    "vx_mae": -1, "vx_rmse": -1,
    "vy_mae": -1, "vy_rmse": -1,
    "yaw_mae": -1, "yaw_rmse": -1,
    "mean_upright_err": -1,
    "mean_power_w": -1,
    "mean_cot": -1,
    "mean_foot_slip": -1,
    "torque_abs_mean_all": -1,
    "torque_abs_max_all": -1,
    "torque_near_limit_frac_all": -1,
    "diag_inphase_rate": +1,
    "lateral_inphase_rate": +1,
}

# condition 里通常用来区分 run 的字段（会单独列一张“条件对比”表）。
CONDITION_KEYS = [
    "tag", "task", "checkpoint", "terrain", "num_envs", "num_steps",
    "cmd_vx", "cmd_vy", "cmd_wz", "friction", "mass_scale",
    "obs_noise", "action_delay", "push_interval_s", "push_vel",
]


def resolve_inputs(inputs: list[str]) -> list[Path]:
    """把参数（文件 / 目录 / glob）解析成 eval.json 路径列表，保持顺序、去重。"""
    files: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path):
        p = p.resolve()
        if p not in seen and p.is_file():
            seen.add(p)
            files.append(p)

    for raw in inputs:
        matches = glob.glob(raw)
        if not matches and not os.path.exists(raw):
            print(f"[warn] 没有匹配到: {raw}")
            continue
        for m in (matches or [raw]):
            p = Path(m)
            if p.is_dir():
                cand = p / "eval.json"
                if cand.is_file():
                    add(cand)
                else:  # 目录下递归找 eval.json
                    for f in sorted(p.rglob("eval.json")):
                        add(f)
            else:
                add(p)
    return files


def auto_label(path: Path) -> str:
    """给每列起个可读标签：优先用 run 目录时间戳（eval 的上上级），退化到父目录名。"""
    parts = path.parts
    if "eval" in parts:
        i = parts.index("eval")
        if i > 0:
            return parts[i - 1]  # .../<run>/eval/eval.json -> <run>
    return path.parent.name


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        av = abs(v)
        if av < 1e-3 or av >= 1e5:
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(v)


def best_index(vals: list, direction: int) -> int | None:
    """返回该行最优值所在列下标；无可比数值或中性指标返回 None。"""
    if direction == 0:
        return None
    nums = [(i, v) for i, v in enumerate(vals)
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) < 2:
        return None
    key = max if direction > 0 else min
    bi, bv = key(nums, key=lambda t: t[1])
    # 若所有值相等则不标最优
    if all(v == bv for _, v in nums):
        return None
    return bi


def collect_metric_keys(datas: list[dict]) -> list[str]:
    """并集，保留首个文件的出现顺序，跳过 list 类型（per_joint/per_foot 另论）。"""
    order: list[str] = []
    seen: set[str] = set()
    for d in datas:
        for k, v in d.get("metrics", {}).items():
            if isinstance(v, list):
                continue
            if k not in seen:
                seen.add(k)
                order.append(k)
    return order


def build_rows(datas, keys, section):
    """返回 [(metric, [vals...], best_idx), ...]。"""
    rows = []
    for k in keys:
        vals = [d.get(section, {}).get(k) for d in datas]
        if all(v is None for v in vals):
            continue
        bi = best_index(vals, METRIC_DIR.get(k, 0)) if section == "metrics" else None
        rows.append((k, vals, bi))
    return rows


def render_terminal(labels, metric_rows, cond_rows) -> str:
    out: list[str] = []

    def table(title, rows, mark_best):
        headers = ["metric"] + labels
        cells = [headers]
        for name, vals, bi in rows:
            row = [name]
            for i, v in enumerate(vals):
                s = fmt(v)
                if mark_best and bi is not None and i == bi:
                    s = "★" + s
                row.append(s)
            cells.append(row)
        widths = [max(len(r[c]) for r in cells) for c in range(len(headers))]
        out.append(f"\n{title}")
        for ri, row in enumerate(cells):
            line = "  ".join(cell.ljust(widths[c]) for c, cell in enumerate(row))
            out.append("  " + line)
            if ri == 0:
                out.append("  " + "  ".join("-" * widths[c] for c in range(len(headers))))

    if cond_rows:
        table("== 条件对比 (condition) ==", cond_rows, mark_best=False)
    table("== 指标对比 (metrics)  ★=该行最优 ==", metric_rows, mark_best=True)
    return "\n".join(out)


def render_markdown(labels, metric_rows, cond_rows, files) -> str:
    out: list[str] = ["# eval.json 对比\n"]
    out.append("| # | 列标签 | 文件 |")
    out.append("|---|---|---|")
    for i, (lb, f) in enumerate(zip(labels, files), 1):
        out.append(f"| {i} | `{lb}` | `{f}` |")
    out.append("")

    def table(title, rows, mark_best):
        out.append(f"## {title}\n")
        out.append("| metric | " + " | ".join(labels) + " |")
        out.append("|" + "---|" * (len(labels) + 1))
        for name, vals, bi in rows:
            cells = []
            for i, v in enumerate(vals):
                s = fmt(v)
                if mark_best and bi is not None and i == bi:
                    s = f"**{s}**"
                cells.append(s)
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")

    if cond_rows:
        table("条件对比 (condition)", cond_rows, mark_best=False)
    table("指标对比 (metrics)　**加粗** = 该行最优", metric_rows, mark_best=True)
    out.append("> 方向约定：`success_rate/mean_survival_s/mean_return/*_inphase_rate` 越大越好；")
    out.append("> `fall_rate/*_mae/*_rmse/mean_cot/mean_power_w/torque_*/upright_err/foot_slip` 越小越好；其余为中性不标注。")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="对比多个 eval.json 的指标")
    ap.add_argument("inputs", nargs="+", help="eval.json 文件 / 目录 / glob")
    ap.add_argument("--labels", nargs="*", default=None, help="每列标签（与输入顺序对应）")
    ap.add_argument("-o", "--output", default=None, help="Markdown 输出路径")
    ap.add_argument("--only-diff-cond", action="store_true",
                    help="condition 表只显示各列存在差异的字段")
    args = ap.parse_args()

    files = resolve_inputs(args.inputs)
    if not files:
        raise SystemExit("没有找到任何 eval.json")
    if len(files) < 2:
        print(f"[warn] 只找到 1 个文件，无对比对象：{files[0]}")

    datas = [load(f) for f in files]

    if args.labels:
        if len(args.labels) != len(files):
            raise SystemExit(f"--labels 数量({len(args.labels)})与文件数({len(files)})不一致")
        labels = args.labels
    else:
        labels = [auto_label(f) for f in files]
        # 去重：重复标签追加序号
        counts: dict[str, int] = {}
        for i, lb in enumerate(labels):
            if labels.count(lb) > 1:
                counts[lb] = counts.get(lb, 0) + 1
                labels[i] = f"{lb}#{counts[lb]}"

    metric_keys = collect_metric_keys(datas)
    metric_rows = build_rows(datas, metric_keys, "metrics")

    cond_rows = build_rows(datas, CONDITION_KEYS, "condition")
    if args.only_diff_cond:
        cond_rows = [r for r in cond_rows if len(set(map(str, r[1]))) > 1]

    print(render_terminal(labels, metric_rows, cond_rows))

    out_path = Path(args.output) if args.output else files[0].parent / "compare_eval.md"
    out_path.write_text(render_markdown(labels, metric_rows, cond_rows,
                                        [str(f) for f in files]), encoding="utf-8")
    print(f"\n[ok] Markdown 已写入: {out_path}")


if __name__ == "__main__":
    main()
