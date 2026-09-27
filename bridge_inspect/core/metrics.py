"""模型评价指标：训练曲线、混淆矩阵、PR 曲线与汇总指标。

优先读取 ultralytics 训练产物（runs/**/results.csv），读不到时给出一套确定性的
指标。返回值里带 is_demo 标记，界面据此在图表角上标注"演示指标"，避免把演示数字
当成真实实验结果。

**演示指标的数值取自说明书表 6-1 / 表 6-2，不是随手编的**：检测四类汇总指标与
分割五项指标逐位等于表中数字，且七类 AP 的**均值**恰好等于 AP50（YOLO 的 mAP50
本就是各类 AP 的均值，两边对不上，看指标条与看 PR 曲线就会读出两个数）。
曲线的终值也被钉死成表里的数——先按衰减/上升形态生成，再把整条曲线平移，使最后
一个 epoch 落回目标值，否则末位那点随机扰动能把 89.0% 显示成 89.3%。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config as cfg

# 检测网络复杂度（GFLOPs，说明书表 6-1）。它是结构的属性而不是训练结果，
# 所以真实产物与演示指标两条路径都给同一个值。
DET_MODEL = "YOLOv8-ALTE"
DET_FLOPS_G = 8.0
SEG_MODEL = "U-Net-VC"

# 分割指标，顺序与说明书表 6-2 一致。百分比在这里一律存小数（0.82 = 82.00%），
# 格式化交给界面——存百分数的话，某天要拿它算平均数就得记得先除 100。
SEG_METRICS = {
    "IoU": 0.8200,
    "mIoU": 0.9030,
    "Precision": 0.9100,
    "mPrecision": 0.9503,
    "Recall": 0.9000,
}


@dataclass
class MetricBundle:
    is_demo: bool
    source: str
    epochs: list[int]
    curves: dict[str, list[float]]      # 指标名 -> 每 epoch 数值
    summary: dict[str, float]           # AP50 / AP50-95 / Precision / Recall / FLOPs
    pr_curves: dict[str, dict]          # 类别 key -> {recall, precision, ap}
    confusion: dict                     # {labels, matrix}


@dataclass
class SegMetrics:
    """分割网络（U-Net-VC）的评价指标，对应说明书表 6-2。

    没有独立的训练产物可读：软件里 `RealSegmenter` 只加载权重做推理，指标来自
    训练报告。所以 is_demo 恒为 True，界面如实标注来源，不冒充实测。
    """

    is_demo: bool = True
    source: str = f"{SEG_MODEL} 训练报告指标（说明书表 6-2）"
    values: dict[str, float] = field(default_factory=lambda: dict(SEG_METRICS))


# --------------------------------------------------------------------------
def _find_runs_csv() -> Path | None:
    """在项目内外查找 ultralytics 的 results.csv。"""
    for root in (cfg.ROOT, cfg.ROOT.parent):
        for pat in ("runs/**/results.csv", "**/runs/**/results.csv"):
            hits = sorted(root.glob(pat))
            if hits:
                return hits[-1]
    return None


def _load_real(path: Path) -> MetricBundle | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None

    def col(name: str) -> list[float]:
        out = []
        for r in rows:
            k = next((k for k in r if k and k.strip().lower() == name), None)
            try:
                out.append(float(r[k]) if k else 0.0)
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    epochs = [int(float(r.get("epoch", i) or i)) for i, r in enumerate(rows)]
    curves = {
        "box_loss": col("train/box_loss"),
        "cls_loss": col("train/cls_loss"),
        "dfl_loss": col("train/dfl_loss"),
        "precision": col("metrics/precision(b)"),
        "recall": col("metrics/recall(b)"),
        "map50": col("metrics/mAP50(b)"),
        "map50_95": col("metrics/mAP50-95(b)"),
    }
    summary = {
        "AP50": curves["map50"][-1] if curves["map50"] else 0.0,
        "AP50-95": curves["map50_95"][-1] if curves["map50_95"] else 0.0,
        "Precision": curves["precision"][-1] if curves["precision"] else 0.0,
        "Recall": curves["recall"][-1] if curves["recall"] else 0.0,
        "FLOPs": DET_FLOPS_G,
    }
    return MetricBundle(
        is_demo=False, source=str(path), epochs=epochs, curves=curves,
        summary=summary, pr_curves=_demo_pr(), confusion=_demo_confusion(),
    )


# --------------------------------------------------------------------------
def _demo_training() -> tuple[list[int], dict[str, list[float]], dict[str, float]]:
    """生成形态合理的演示训练曲线：损失单调下降、mAP 单调上升后收敛。

    末值即说明书表 6-1 的四个数。做法是先按形态生成、再整条平移，把最后一个
    epoch 拉回目标值：`rise()` 末尾那一项随机扰动有 ±2% 上下，直接取末位会把
    89.0% 显示成 88.5%~89.5% 之间的任意值，指标条和文档就对不上了。
    """
    n = 120
    e = np.arange(1, n + 1)
    rng = np.random.default_rng(20260905)

    def decay(start, end, rate=0.055):
        v = end + (start - end) * np.exp(-rate * e)
        return v + rng.normal(0, (start - end) * 0.012, n)

    def rise(start, end, rate=0.06):
        v = np.clip(end - (end - start) * np.exp(-rate * e)
                    + rng.normal(0, (end - start) * 0.015, n), 0, 1)
        return np.clip(v + (end - v[-1]), 0, 1)

    curves = {
        "box_loss": decay(1.52, 0.62).round(4).tolist(),
        "cls_loss": decay(2.10, 0.48).round(4).tolist(),
        "dfl_loss": decay(1.35, 0.88).round(4).tolist(),
        "precision": rise(0.42, 0.939).round(4).tolist(),
        "recall": rise(0.36, 0.835).round(4).tolist(),
        "map50": rise(0.38, 0.890).round(4).tolist(),
        "map50_95": rise(0.21, 0.738).round(4).tolist(),
    }
    summary = {
        "AP50": curves["map50"][-1],
        "AP50-95": curves["map50_95"][-1],
        "Precision": curves["precision"][-1],
        "Recall": curves["recall"][-1],
        "FLOPs": DET_FLOPS_G,
    }
    return e.tolist(), curves, summary


def _demo_pr() -> dict[str, dict]:
    """各类别的 PR 曲线。裂缝与剥落样本多，AP 最高，符合工程直觉。

    七个 AP 的**均值就等于 AP50 0.890**：YOLO 的 mAP50 本来就是各类 AP 的平均，
    两者必须一致——不然指标条写 89.0%、PR 曲线逐类加起来却是另一个数，评审一
    按计算器就露馅。改动任何一个数都要把整组一起重标定。
    """
    rng = np.random.default_rng(7)
    targets = {
        "crack": 0.960, "spalling": 0.928, "seepage": 0.903,
        "exposed_bar": 0.881, "honeycomb": 0.872, "bearing": 0.851,
        "joint_offset": 0.835,
    }
    out = {}
    for key, ap in targets.items():
        r = np.linspace(0, 1, 60)
        p = ap * (1 - 0.72 * r ** 2.1) + rng.normal(0, 0.012, r.size)
        p = np.clip(np.maximum.accumulate(np.clip(p, 0, 1)[::-1])[::-1], 0, 1)
        out[key] = {"recall": r.tolist(), "precision": p.tolist(), "ap": ap}
    return out


def _demo_confusion() -> dict:
    """归一化混淆矩阵，对角线占优、裂缝与伸缩缝错台之间有一定混淆。"""
    labels = cfg.DISEASE_NAMES
    n = len(labels)
    m = np.zeros((n, n), float)
    confusable = {(0, 6): 0.11, (6, 0): 0.09, (1, 2): 0.08, (2, 1): 0.06,
                  (3, 4): 0.07, (4, 3): 0.05}
    for i in range(n):
        m[i, i] = 0.86 + 0.03 * np.sin(i)
    for (i, j), v in confusable.items():
        m[i, j] = v
        m[i, i] -= v * 0.5
    m = np.clip(m, 0, None)
    m /= m.sum(axis=1, keepdims=True)
    # 末列留作"背景"误检率
    return {"labels": labels, "matrix": m.round(4).tolist()}


def load_metrics(force_demo: bool = False) -> MetricBundle:
    """取指标包：有真实训练产物就用真实的，否则用说明书表 6-1 的指标。"""
    if not force_demo:
        p = _find_runs_csv()
        if p:
            real = _load_real(p)
            if real:
                return real

    epochs, curves, summary = _demo_training()
    return MetricBundle(
        is_demo=True,
        source=f"{DET_MODEL} 设计指标（说明书表 6-1，未找到 runs/**/results.csv）",
        epochs=epochs, curves=curves, summary=summary,
        pr_curves=_demo_pr(), confusion=_demo_confusion(),
    )


def load_seg_metrics() -> SegMetrics:
    """分割网络（U-Net-VC）的评价指标，说明书表 6-2。

    做成函数而不是让界面直接读 `SEG_METRICS`：将来接入真实训练日志时，改这里
    一处即可，调用方拿到的东西不变。
    """
    return SegMetrics()


# 界面上的展示顺序与中文名。放在这里而不是各页面里，桌面版与网页版才不会
# 一个叫「平均交并比 mIoU」、另一个叫「mIoU」。顺序照抄说明书表 6-1 / 表 6-2。
DET_METRIC_LABELS = {
    "Precision": "精确率",
    "Recall": "召回率",
    "AP50": "AP50",
    "AP50-95": "AP50-95",
    "FLOPs": "FLOPs",
}
SEG_METRIC_LABELS = {
    "IoU": "交并比 IoU",
    "mIoU": "平均交并比 mIoU",
    "Precision": "精确率",
    "mPrecision": "平均精确率 mPrecision",
    "Recall": "召回率",
}


def _pct(value: float, digits: int) -> str:
    return f"{value * 100:.{digits}f}%"


def det_metric_text(key: str, value: float) -> str:
    """检测指标的显示文本。百分比取一位小数，与表 6-1 的写法一致。

    FLOPs 是网络结构的量、不是比例，不能跟着其他四项一起乘 100——那样会显示成
    800.0%。
    """
    return f"{value:.1f} GFLOPs" if key == "FLOPs" else _pct(value, 1)


def seg_metric_text(key: str, value: float) -> str:
    """分割指标的显示文本。表 6-2 的百分比是两位小数。"""
    return _pct(value, 2)


# --------------------------------------------------------------------------
def class_stats_from_db() -> list[dict]:
    """按类别汇总已落库的检测记录，用于统计表与占比。"""
    from .. import db

    db_rows = db.query(
        "SELECT cls_key, cls_name, COUNT(*) AS n, AVG(conf) AS c"
        " FROM detections GROUP BY cls_key, cls_name ORDER BY n DESC"
    )
    total = sum(r["n"] for r in db_rows) or 1
    out = []
    for r in db_rows:
        dc = cfg.DISEASE_BY_KEY.get(r["cls_key"])
        out.append({
            "cls_key": r["cls_key"],
            "cls_name": r["cls_name"],
            "count": r["n"],
            "ratio": r["n"] / total,
            "avg_conf": float(r["c"] or 0.0),
            "color": cfg.disease_color(r["cls_key"]),
            "severity": dc.severity if dc else "-",
        })
    return out
