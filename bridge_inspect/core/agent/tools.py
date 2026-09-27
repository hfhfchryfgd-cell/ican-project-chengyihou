"""工具注册表——智能体与既有算法之间唯一的接口层。

## 一条硬规矩：工具里没有大模型

每个工具封装的都是 `core/` 里已经跑通的确定性函数（检测、分割、GSD 换算、
趋势聚合、条文检索）。大模型只出现在两个地方：**规划**（把自然语言拆成工具调用序列）
和**措辞**（把已经算出来的结论写成通顺的研判文字）。数值一律由工具产出。

这条分界的实际后果：同一句指令跑两次，检测框坐标、面积、宽度、复飞距离完全一致。
评委可以任选一条结论回溯到工具调用记录，再回溯到数据库记录。如果让模型"参与"计算，
第一遍和第二遍的数字就会不一样——演示视频与现场答辩对不上，是最难解释的事故。

## 为什么每个工具都要返回 summary

轨迹时间线上会顺序出现十几个工具调用。若每步都展开完整载荷，界面会被坐标数组淹没；
若只显示工具名，又看不出发生了什么。折中是：工具自己产出一句**中文结论**（如
「D02 裂缝：面积 12.4 cm²，最大宽度 1.50 mm，置信度 0.96」），时间线显示这句，
点开才看结构化数据。

## 数据库安全

`detect_image` / `segment_defect` 默认**不写库**，只在影像尚无记录时补写。智能体在
演示库上会被反复运行，若每次都插入检测行，跑三遍之后同一个检测框就会出现三份，
趋势统计随之翻倍——而 schema 没有外键级联，清理起来要手工比对。
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ... import config as cfg
from ... import db
from .. import backend, imgutil, ingest, report, segment
from .. import llm
from . import reflight, specs

# 后端实例缓存。make_detector() 每次调用都会重新判定权重、构造后端并记日志，
# 而智能体一次任务里可能连续调它十几次。
_CACHE: dict[str, Any] = {}


def detector(settings: dict | None = None):
    if "det" not in _CACHE:
        _CACHE["det"] = backend.make_detector(settings)
    return _CACHE["det"]


def segmenter(settings: dict | None = None):
    if "seg" not in _CACHE:
        _CACHE["seg"] = segment.make_segmenter(settings)
    return _CACHE["seg"]


def clear_cache() -> None:
    """设置页改了模型或阈值后调用，让下次工具调用重建后端。"""
    _CACHE.clear()


# --------------------------------------------------------------------------
# 结果与规格
# --------------------------------------------------------------------------
@dataclass
class ToolResult:
    ok: bool
    name: str
    summary: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""
    ms: int = 0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "name": self.name, "summary": self.summary,
                "data": self.data, "error": self.error, "ms": self.ms}


@dataclass
class ToolSpec:
    name: str
    cn: str                       # 中文名，轨迹时间线上显示这个
    desc: str                     # 给大模型看的用途说明
    params: dict                  # JSON Schema
    func: Callable[..., ToolResult]
    agent: str                    # 归属专家智能体（events.AGENTS 的 key）
    readonly: bool = True         # 是否只读；写操作在界面上要有区分

    def schema(self) -> dict:
        """OpenAI / 百炼通用的 function calling 描述。"""
        return {"type": "function", "function": {
            "name": self.name, "description": self.desc, "parameters": self.params}}


def _obj(**props: dict) -> dict:
    """把若干属性拼成 JSON Schema 的 object。没有 required 是因为工具都有默认值，
    缺参数时按"全库"处理比直接报错更符合巡检场景的口语习惯。"""
    return {"type": "object", "properties": props}


_STR = {"type": "string"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}


def _ok(name: str, summary: str, **data) -> ToolResult:
    return ToolResult(True, name, summary, data)


def _err(name: str, msg: str) -> ToolResult:
    return ToolResult(False, name, "", {}, msg)


# --------------------------------------------------------------------------
# 工具实现
# --------------------------------------------------------------------------
def t_db_overview() -> ToolResult:
    s = db.stat_summary()
    imgs = db.list_images(limit=100000)
    batches = sorted({r["batch"] for r in imgs if r["batch"]})
    # 类别计数按检测框、不按量化记录：像素级分割只对裂缝做，按量化记录数出来的
    # 「类别分布」只有一类，模型据此判断数据边界时会以为库里就没有别的病害。
    cls_cnt: dict[str, int] = {}
    for r in db.query("SELECT cls_name, COUNT(*) AS n FROM detections"
                      " GROUP BY cls_name"):
        cls_cnt[r["cls_name"]] = r["n"]
    top = sorted(cls_cnt.items(), key=lambda kv: -kv[1])
    summary = (f"库内 {s['images']} 张影像、{s['points']} 个点位、"
               f"{s['segments']} 条量化记录，覆盖 {len(batches)} 个批次")
    return _ok("db_overview", summary,
               images=s["images"], points=s["points"], segments=s["segments"],
               detections=s.get("detections", 0), batches=batches,
               by_class=[{"cls_name": k, "n": v} for k, v in top])


def t_list_points(component: str = "", batch: str = "") -> ToolResult:
    have = set(db.distinct_points())
    pts = [p for p in cfg.ALL_POINTS if p in have] if have else list(cfg.ALL_POINTS)
    if component:
        pts = [p for p in pts if p[:1].upper() == component.upper()[:1]]
    if batch:
        bset = {r["point_id"] for r in db.list_images(batch=batch, limit=100000)}
        pts = [p for p in pts if p in bset]
    by_comp: dict[str, list[str]] = {}
    for p in pts:
        by_comp.setdefault(cfg.point_component(p), []).append(p)
    summary = (f"{len(pts)} 个点位于 {len(by_comp)} 类构件："
               + "；".join(f"{k}({'/'.join(v)})" for k, v in by_comp.items()))
    return _ok("list_points", summary, points=pts,
               by_component={k: v for k, v in by_comp.items()})


def t_import_images(paths: list[str] | None = None, source_dir: str = "",
                    batch: str = "") -> ToolResult:
    """把本次飞行的影像读入库——「图像导入」是智能体流水线的第一步。

    不重写导入逻辑：解析文件名、取航点元数据、写 images 表这一整套都在
    `core/ingest.ingest()`，桌面「智能体工作台 → 导入影像」与网页上传端点走的是同一段代码。
    这里只负责"从哪取文件"，否则同一张图在三条入口下会落成三种记录。

    未指定 paths 与 source_dir 时扫 `cfg.IMG_DIR`（SD 卡暂存目录）。重复导入是
    幂等的——已在库的按原路径跳过，所以这个默认动作在任何时候都可以安全执行，
    这也正是它能被编排进"生成报告"这类任务的原因。
    """
    given: list[str] = [str(p) for p in (paths or []) if str(p).strip()]
    scanned = ""
    if not given:
        d = Path(source_dir) if source_dir else cfg.IMG_DIR
        if not d.is_dir():
            return _err("import_images", f"目录不存在：{d}")
        # 只收支持的影像扩展名。同目录下还有 *.gt.json 这类边车文件，
        # 一并交下去会被记成"未能导入"，把一次干净的导入报成有失败项。
        given = [str(p) for p in sorted(d.iterdir())
                 if p.is_file() and imgutil.is_supported(str(p))]
        scanned = str(d)
    if not given:
        return _ok("import_images", "没有待导入的影像（目录内无支持的影像格式）",
                   added=0, skipped=0, failed=0, rows=[], source_dir=scanned)

    res = ingest.ingest(given, default_batch=batch.strip().upper() or None)
    d = res.to_dict()
    # 新增了哪几张、落在哪个点位/批次，是后续检测与趋势要用的上下文；
    # 未入库的那些只留计数与原因，逐条列出来会把时间线撑成一屏文件名。
    added = [{"filename": r.filename, "point_id": r.point_id, "batch": r.batch,
              "width": r.width, "height": r.height, "overlap": round(r.overlap, 3)}
             for r in res.added]
    return _ok("import_images",
               f"{res.summary()}"
               + (f"｜来源 {Path(scanned).name}" if scanned else ""),
               added=len(res.added), skipped=len(res.skipped),
               failed=len(res.failed), images=added,
               failed_rows=[{"filename": r.filename, "status": r.status,
                             "message": r.message} for r in res.failed],
               source_dir=scanned)


def t_inspect_waypoint(point_id: str, batch: str = "") -> ToolResult:
    if not point_id:
        return _err("inspect_waypoint", "未指定点位编号")
    pid = point_id.strip().upper()
    b = batch.strip().upper() if batch else (cfg.BATCHES[-1].id if cfg.BATCHES else "")
    row = db.query_one(
        "SELECT * FROM images WHERE point_id=? AND batch=? ORDER BY id LIMIT 1",
        (pid, b)) if b else None
    if row:
        d = dict(row)
        rec = {k: d.get(k) for k in (
            "waypoint", "point_id", "batch", "rtk_lon", "rtk_lat", "rtk_alt",
            "rtk_status", "gimbal_yaw", "gimbal_pitch", "gimbal_roll",
            "shoot_distance", "overlap", "filename")}
    else:
        # 库里没有就按演示数据生成一份，用于"下一个批次该怎么飞"的规划场景
        rec = dict(_waypoint_meta(pid, b))
    gsd = cfg.gsd_mm_per_px(float(rec.get("shoot_distance") or 0))
    tag = "库内记录" if row else "规划值（该点位本批次尚无实拍影像）"
    summary = (f"{pid}（{cfg.point_component(pid)}）{b} 批次：航点 "
               f"{rec.get('waypoint') or '—'}，拍摄距离 "
               f"{(rec.get('shoot_distance') or 0):.2f} m，GSD {gsd:.3f} mm/px，"
               f"重合率 {(rec.get('overlap') or 0):.0%}｜{tag}")
    rec["gsd_mm_per_px"] = round(gsd, 4)
    rec["component"] = cfg.point_component(pid)
    return _ok("inspect_waypoint", summary, waypoint=rec, source="db" if row else "规划")


def _waypoint_meta(point_id: str, batch_id: str) -> dict:
    from .. import demo_data
    return demo_data.waypoint_meta(point_id, batch_id)


def t_query_timeseries(point_id: str = "", cls_key: str = "") -> ToolResult:
    rows = db.timeseries_overview()
    pid = point_id.strip().upper() if point_id else ""
    if pid:
        rows = [r for r in rows if r["point_id"] == pid]
    if cls_key:
        rows = [r for r in rows if r["cls_key"] == cls_key]
    out = [{"point_id": r["point_id"], "component": cfg.point_component(r["point_id"]),
            "cls_key": r["cls_key"], "cls_name": r["cls_name"], "batch": r["batch"],
            "n": r["n"], "area_cm2": round(float(r["area"]), 2),
            "max_width_mm": round(float(r["width"] or 0), 2),
            # 趋势页的 Y 轴可切到「裂缝长度」，它直接读这份行的 length_mm。
            # 少这一个键，切过去三条线全是 NaN——图上什么都不画，也不报错。
            "length_mm": round(float(r["length"] or 0), 2)} for r in rows]
    if not out:
        return _ok("query_timeseries", "未查询到匹配的量化记录", rows=[])
    span = f"{pid}" if pid else f"{len({r['point_id'] for r in out})} 个点位"
    summary = (f"{span} 命中 {len(out)} 条量化记录，涉及 "
               f"{len({r['cls_name'] for r in out})} 类病害、"
               f"{len({r['batch'] for r in out})} 个批次")
    return _ok("query_timeseries", summary, rows=out)


def t_analyze_trend(point_id: str, cls_key: str = "") -> ToolResult:
    if not point_id:
        return _err("analyze_trend", "未指定点位编号")
    pid = point_id.strip().upper()
    if not cls_key:
        row = db.query_one(
            "SELECT cls_key FROM segments s JOIN images i ON i.id=s.image_id"
            " WHERE i.point_id=? GROUP BY s.cls_key"
            " ORDER BY MAX(s.area_cm2) DESC LIMIT 1", (pid,))
        cls_key = row["cls_key"] if row else ""
    if not cls_key:
        return _err("analyze_trend", f"{pid} 没有可用于趋势比对的量化记录")
    t = llm._trend_of(pid, cls_key)
    if not t:
        return _err("analyze_trend",
                    f"{pid} 的 {cls_key} 不足两个批次，无法构成趋势序列")
    dc = cfg.DISEASE_BY_KEY.get(cls_key)
    grade, why = llm._grade(t["growth"], t["width"],
                            dc.severity if dc else "中", cls_key)
    summary = (f"{pid}（{t['component']}）{t['cls_name']}：面积由 "
               f"{t['area0']:.1f} 发展至 {t['area1']:.1f} cm²，累计增幅 "
               f"{t['growth'] * 100:+.1f}%，最新宽度 {t['width']:.2f} mm → 建议{grade}")
    return _ok("analyze_trend", summary, trend=t, grade=grade, reason=why,
               citation=specs.for_disease(cls_key, pid[:1], top_k=2)[0].citation)


def t_find_anomalies(threshold: float = 0.30) -> ToolResult:
    anom = llm.detect_anomalies(growth_threshold=float(threshold))
    if not anom:
        return _ok("find_anomalies",
                   f"未发现增幅超过 {threshold:.0%} 的点位", anomalies=[])
    heads = "、".join(f"{a['point']}{a['cls_name']}{a['growth'] * 100:+.0f}%"
                     for a in anom[:5])
    more = f" 等 {len(anom)} 处" if len(anom) > 5 else ""
    return _ok("find_anomalies",
               f"筛出 {len(anom)} 处异常发展：{heads}{more}", anomalies=anom)


def t_detect_image(image_id: int = 0, point_id: str = "", batch: str = "",
                   refresh: bool = False, output_dir: str = "") -> ToolResult:
    """对影像跑检测。默认优先用库内已有结果，避免重复写入。"""
    row = _pick_image(image_id, point_id, batch)
    if row is None:
        return _err("detect_image", "未找到匹配的影像记录")
    iid = row["id"]
    exist = db.query("SELECT * FROM detections WHERE image_id=?", (iid,))
    if exist and not refresh:
        dets = [dict(d) for d in exist]
        src = "库内已有结果"
    else:
        bgr = imgutil.imread(row["path"])
        if bgr is None:
            return _err("detect_image", f"影像无法解码：{row['filename']}")
        found = detector().predict(bgr)
        dets = [dataclasses.asdict(d) for d in found]
        src = "本次实时推理"
        if not exist:
            # 只在影像尚无记录时补写。智能体在演示库上会被反复运行，
            # 每次都插入会让同一个检测框出现多份，趋势统计随之翻倍。
            db.add_detections([{**d, "image_id": iid,
                                "point_id": row["point_id"]} for d in dets])
    by_cls: dict[str, int] = {}
    for d in dets:
        by_cls[d["cls_name"]] = by_cls.get(d["cls_name"], 0) + 1
    # 图像类产物按总控传入的任务目录写入；未传时仍写入通用 agent 目录，方便单独调用。
    art_dir = Path(output_dir) if output_dir else cfg.OUT_DIR / "agent"
    art_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{row['batch']}_{row['point_id']}_{iid}_detect_{int(time.time() * 1000)}"
    ann_path = art_dir / f"{stem}.jpg"
    bgr = imgutil.imread(row["path"])
    annotated = ""
    if bgr is not None:
        visual = [backend.Detection(
            d["cls_key"], d["cls_name"], int(d.get("cls_id") or
            getattr(cfg.DISEASE_BY_KEY.get(d["cls_key"]), "id", 0)),
            float(d["conf"]), int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]),
            str(d.get("source") or src), int(d.get("gt_idx") or 0)) for d in dets]
        if imgutil.imwrite(ann_path, imgutil.draw_detections(bgr, visual)):
            annotated = str(ann_path)
    if dets:
        top = max(dets, key=lambda d: d["conf"])
        summary = (f"{row['filename']}（{row['point_id']}/{row['batch']}）检出 "
                   f"{len(dets)} 处，最高置信度 {top['conf']:.2f}"
                   f"（{top['cls_name']}）｜" + "、".join(
                       f"{k}×{v}" for k, v in by_cls.items()) + f"｜{src}")
    else:
        summary = f"{row['filename']} 未检出病害"
    return _ok("detect_image", summary, image=row["filename"], image_id=iid,
               point_id=row["point_id"], batch=row["batch"], source=src,
               detections=dets, by_class=by_cls, annotated=annotated,
               csv="")


def det_of_row(d) -> backend.Detection:
    """把 `detections` 表的一行还原成 Detection 对象。

    **不能按字段名直接取。** `Detection` 有 10 个字段，`detections` 表只有 8 个——
    `cls_id` 与 `source` 没有对应列（检测框存库时这两项本就是导出量：类别 id 可由
    `cls_key` 反查，来源只在当次推理的上下文里有意义）。早先这里写的是
    `d["cls_id"]`，只要走到"重新分割"这条分支就抛 `IndexError: No item with that key`；
    因为演示库里每个影像都已有分割记录，默认路径走的是"库内已有结果"，这个错误一直
    没露头——直到调用方传了 `cls_key` 强制重算。sqlite3.Row 支持 `keys()`，按列名
    是否存在决定取值，越界的那两项用 `cls_key` 反查与默认串补上。
    """
    have = set(d.keys())
    ck = d["cls_key"]
    dc = cfg.DISEASE_BY_KEY.get(ck)
    return backend.Detection(
        ck, d["cls_name"],
        d["cls_id"] if "cls_id" in have else (dc.id if dc else 0),
        d["conf"], d["x1"], d["y1"], d["x2"], d["y2"],
        d["source"] if "source" in have else "库内记录",
        d["gt_idx"] if "gt_idx" in have else 0,
    )


def _pick_image(image_id: int = 0, point_id: str = "", batch: str = ""):
    if image_id:
        return db.query_one("SELECT * FROM images WHERE id=?", (int(image_id),))
    if point_id:
        pid = point_id.strip().upper()
        if batch:
            r = db.query_one("SELECT * FROM images WHERE point_id=? AND batch=?"
                             " ORDER BY id LIMIT 1", (pid, batch.strip().upper()))
            if r:
                return r
        return db.query_one("SELECT * FROM images WHERE point_id=?"
                            " ORDER BY id DESC LIMIT 1", (pid,))
    if batch:
        return db.query_one("SELECT * FROM images WHERE batch=?"
                            " ORDER BY id LIMIT 1", (batch.strip().upper(),))
    return None


def t_segment_defect(image_id: int = 0, point_id: str = "", batch: str = "",
                     cls_key: str = "", output_dir: str = "") -> ToolResult:
    row = _pick_image(image_id, point_id, batch)
    if row is None:
        return _err("segment_defect", "未找到匹配的影像记录")
    iid = row["id"]
    exist = db.query("SELECT * FROM segments WHERE image_id=?", (iid,))
    masks = []
    if exist and not cls_key:
        segs = [dict(s) for s in exist]
        src = "库内已有结果"
    else:
        bgr = imgutil.imread(row["path"])
        if bgr is None:
            return _err("segment_defect", f"影像无法解码：{row['filename']}")
        dets = db.query("SELECT * FROM detections WHERE image_id=?", (iid,))
        if not dets:
            found = detector().predict(bgr)
            if not exist:
                db.add_detections([{**dataclasses.asdict(d), "image_id": iid,
                                    "point_id": row["point_id"]} for d in found])
            dets = db.query("SELECT * FROM detections WHERE image_id=?", (iid,))
        dist = float(row["shoot_distance"] or 0)
        segs, src = [], "本次实时分割"
        seg = segmenter()
        for d in dets:
            if cls_key and d["cls_key"] != cls_key:
                continue
            det = det_of_row(d)
            # mask 是整幅 ndarray，绝不能进结果——事件层要 json.dumps 它。
            # 这里只取标量字段，掩膜要看的话另走落盘路径。
            r = seg.segment(bgr, det, dist)
            if r is None:
                continue
            # 量化字段走 segment.segment_fields 这一份（四处落库共用），
            # 再补一个只在返回值里出现、不落库的 gsd_mm_per_px。
            rec = dict(segment.segment_fields(r), gsd_mm_per_px=r.gsd_mm_per_px)
            segs.append(rec)
            masks.append(r.mask)
            if not exist:
                # point_id 与 batch 一并写入：segments 表靠它们做点位聚合，
                # 缺了这两列记录在趋势查询里等于不存在。
                db.add_segment(image_id=iid, point_id=row["point_id"],
                               batch=row["batch"], **rec)
    for s in segs:
        # segments 表没有 gsd_mm_per_px 列——它是导出量不是持久化字段。库内结果
        # 缺这一项时按拍摄距离现算，公式见 config.gsd_mm_per_px。
        g = float(s.get("gsd_mm_per_px") or 0.0)
        if g <= 0:
            g = cfg.gsd_mm_per_px(float(row["shoot_distance"] or 0))
            s["gsd_mm_per_px"] = round(g, 4)
        s["uncertainty_mm"] = round(2 * g, 3)
    art_dir = Path(output_dir) if output_dir else cfg.OUT_DIR / "agent"
    art_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{row['batch']}_{row['point_id']}_{iid}_segment_{int(time.time() * 1000)}"
    mask_path = art_dir / f"{stem}_mask.png"
    if masks:
        import numpy as np
        merged = np.maximum.reduce(masks)
        imgutil.imwrite(mask_path, merged)
    else:
        # 旧记录没有持久化整幅掩膜，不能伪造一张“掩膜图”。仍导出量化表，并在
        # 返回中明确 mask 为空；新导入影像会在上面的实时分割分支产出真实掩膜。
        mask_path = None
    if segs:
        w = max(segs, key=lambda s: s["max_width_mm"])
        gsd = float(w["gsd_mm_per_px"])
        summary = (f"{row['filename']}（{row['point_id']}）量化 {len(segs)} 处，"
                   f"最大者 {w['cls_name']}：面积 {w['area_cm2']:.2f} cm²、"
                   f"宽度 {w['max_width_mm']:.2f} mm，GSD {gsd:.3f} mm/px，"
                   f"量测不确定度 ±{2 * gsd:.3f} mm｜{src}")
    else:
        summary = f"{row['filename']} 无可量化的分割结果"
    return _ok("segment_defect", summary, image=row["filename"], image_id=iid,
               distance_m=float(row["shoot_distance"] or 0), source=src,
               segments=segs, mask=str(mask_path) if mask_path else "",
               csv="")


def t_grade_defect(cls_key: str, width_mm: float = 0.0, growth: float = 0.0,
                   component: str = "") -> ToolResult:
    dc = cfg.DISEASE_BY_KEY.get(cls_key)
    if dc is None:
        return _err("grade_defect", f"未知病害类别：{cls_key}")
    grade, why = llm._grade(float(growth), float(width_mm), dc.severity, cls_key)
    clauses = specs.for_disease(cls_key, component, top_k=3)
    cite = clauses[0].citation if clauses else ""
    limit = dc.width_limit_mm
    lim_txt = f"限值 {limit:g} mm" if limit is not None else "无宽度限值"
    summary = (f"{dc.name}（{lim_txt}，实测 {width_mm:.2f} mm，增幅 "
               f"{growth * 100:+.1f}%）→ 处置等级「{grade}」｜依据 {cite}")
    return _ok("grade_defect", summary, cls_key=cls_key, cls_name=dc.name,
               grade=grade, reason=why, limit_mm=limit, severity=dc.severity,
               clauses=[c.to_dict() for c in clauses])


def t_search_spec(cls_key: str = "", component: str = "", query: str = "",
                  top_k: int = 3) -> ToolResult:
    words = [w for w in (query or "").replace("，", " ").replace(",", " ").split() if w]
    clauses = specs.search([cls_key] if cls_key else [],
                           [component] if component else [],
                           top_k=int(top_k), extra_kw=words)
    summary = "；".join(f"{c.citation} {c.title}" for c in clauses)
    return _ok("search_spec", f"检索到 {len(clauses)} 条：{summary}",
               clauses=[c.to_dict() for c in clauses],
               disclaimer=specs.DISCLAIMER)


def t_plan_reflight(points: list[str] | None = None,
                    batches: list[str] | None = None,
                    register: bool = True) -> ToolResult:
    issues, wps = reflight.plan(points, batches)
    if not issues:
        return _ok("plan_reflight", "采集质量审计未发现问题，无需复飞",
                   issues=[], waypoints=[])
    cites = reflight.citations_for(issues)
    csv_path = json_path = None
    if wps:
        csv_path, json_path = reflight.write_mission(wps)
        if register:
            tid = db.create_task(
                name=f"自主复飞任务（{len(wps)} 个航点）", kind="复飞任务",
                batch=wps[0].batch, point_scope=",".join(w.point_id for w in wps),
                image_count=len(wps), note="由智能体质量审计自动生成")
            db.add_output(tid, "复飞航点CSV", str(csv_path))
            db.add_output(tid, "复飞航点JSON", str(json_path))
    hi = sum(1 for i in issues if i.level == "高")
    summary = (f"审计 {len(issues)} 项质量问题（高 {hi} 项），"
               f"解算 {len(wps)} 个复飞航点"
               + (f"，已导出 {Path(csv_path).name}" if csv_path else ""))
    return _ok("plan_reflight", summary, issues=[i.to_dict() for i in issues],
               waypoints=[w.to_dict() for w in wps], citations=cites,
               csv=str(csv_path) if csv_path else "",
               json=str(json_path) if json_path else "")


def t_make_report(llm_text: str = "", register: bool = True,
                  trace: dict | None = None) -> ToolResult:
    """装配报告。

    `trace` 由编排器注入（见 Orchestrator._tool），**不在 ToolSpec 的参数 schema 里**：
    模型看不到它，也就不会去填一个它无从知道的字段。两份报告都要能逐章对照，所以
    没有轨迹时第七章照样存在，内容降级为静态流水线环节表——不编造耗时数字。

    章节名直接取 report.CHAPTERS，不再回读一遍 Markdown 去正则捞 `## `：那等于把
    同一份报告生成两次，两边的章节还可能因取数时刻不同而对不上。
    """
    md, html, no = report.build_and_save(cfg.load_settings(), llm_text, trace)
    if register:
        tid = db.create_task(name="巡检报告编制", kind="报告", note="由智能体自动生成")
        db.add_output(tid, "巡检报告Markdown", str(md))
        db.add_output(tid, "巡检报告HTML", str(html))
    chapters = list(report.CHAPTERS)
    summary = (f"报告 {no} 已生成，{len(chapters)} 个章节："
               + "、".join(chapters[:6]) + " 等")
    return _ok("make_report", summary, report_no=no, markdown=str(md),
               html=str(html), chapters=chapters)


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------
TOOLS: dict[str, ToolSpec] = {}


def _reg(spec: ToolSpec) -> ToolSpec:
    TOOLS[spec.name] = spec
    return spec


_reg(ToolSpec("import_images", "图像导入", "把本次飞行的影像读入库：解析文件名里的"
              "点位与批次、读取影像尺寸、按点位补齐航点元数据。已在库的按原路径跳过，"
              "重复导入不会产生重复记录。不指定路径时扫描 SD 卡暂存目录。",
              _obj(paths={"type": "array", "items": _STR,
                          "description": "影像文件路径列表，留空则扫描目录"},
                   source_dir={**_STR, "description": "待扫描的目录，留空为 SD 卡暂存目录"},
                   batch={**_STR, "description": "文件名里认不出批次时的默认批次"}),
              t_import_images, "mission", readonly=False))
_reg(ToolSpec("db_overview", "数据总览", "查看数据库整体规模：影像数、点位数、"
              "量化记录数、覆盖批次与病害类别分布。任何分析任务开始前先调它了解数据边界。",
              _obj(), t_db_overview, "mission"))
_reg(ToolSpec("list_points", "点位清单", "列出巡检点位及其所属构件（桥墩/梁底/支座/"
              "梁侧腹板/桥台/伸缩缝），可按构件字母或批次过滤。",
              _obj(component={**_STR, "description": "构件首字母 A~F"},
                   batch={**_STR, "description": "批次号，如 P1/P2/P3"}),
              t_list_points, "mission"))
_reg(ToolSpec("inspect_waypoint", "航点档案", "读取某个点位在某批次的航点元数据："
              "RTK 经纬高、云台三轴姿态、拍摄距离、图像重合率，并算出该距离下的 GSD。",
              _obj(point_id={**_STR, "description": "点位编号，如 C03"},
                   batch={**_STR, "description": "批次号，默认最新批次"}),
              t_inspect_waypoint, "mission"))
_reg(ToolSpec("query_timeseries", "时序查询", "查询病害量化时序记录"
              "（面积 cm²、最大宽度 mm、裂缝长度 mm），可按点位与病害类别过滤。",
              _obj(point_id={**_STR, "description": "点位编号，留空为全桥"},
                   cls_key={**_STR, "description": "病害类别英文键，如 crack"}),
              t_query_timeseries, "trend"))
_reg(ToolSpec("analyze_trend", "趋势计算", "对某点位某病害做跨批次趋势比对，"
              "算出面积增幅与处置等级。未指定类别时自动取该点位的主病害。",
              _obj(point_id={**_STR, "description": "点位编号，如 D02"},
                   cls_key={**_STR, "description": "病害类别，留空自动判定"}),
              t_analyze_trend, "trend"))
_reg(ToolSpec("find_anomalies", "异常筛查", "全库扫描，筛出面积增幅超过阈值的"
              "点位×病害组合，按增幅降序。适合回答「哪些地方发展最快」。",
              _obj(threshold={**_NUM, "description": "增幅阈值，默认 0.30 即 30%"}),
              t_find_anomalies, "trend"))
_reg(ToolSpec("detect_image", "病害检测", "对指定影像执行七类病害检测，返回检测框"
              "坐标、类别与置信度。可用 image_id 或 点位+批次 指定影像。",
              _obj(image_id={**_INT, "description": "影像 ID"},
                   point_id={**_STR, "description": "点位编号"},
                   batch={**_STR, "description": "批次号"},
                   refresh={**_BOOL, "description": "忽略库内结果强制重跑"}),
              t_detect_image, "vision", readonly=False))
_reg(ToolSpec("segment_defect", "分割量化", "对检测到的病害做像素级分割，"
              "按 GSD 换算出实际面积与最大宽度，并给出量测不确定度 ±2·GSD。",
              _obj(image_id={**_INT, "description": "影像 ID"},
                   point_id={**_STR, "description": "点位编号"},
                   batch={**_STR, "description": "批次号"},
                   cls_key={**_STR, "description": "只量化该类别"}),
              t_segment_defect, "quant", readonly=False))
_reg(ToolSpec("grade_defect", "按条文定级", "依据现行养护规范条文给出"
              "观察/维修/加固三档处置建议，并附条文出处。两条判据彼此独立："
              "实测值超限、或发展速度过快，命中任一条即升级。",
              _obj(cls_key={**_STR, "description": "病害类别英文键，如 crack"},
                   width_mm={**_NUM, "description": "实测最大宽度 mm"},
                   growth={**_NUM, "description": "面积累计增幅，如 0.65"},
                   component={**_STR, "description": "构件首字母 A~F"}),
              t_grade_defect, "trend"))
_reg(ToolSpec("search_spec", "规范检索", "在桥梁养护规范条文库中检索相关条文，"
              "返回条文号、要点与限值。用于为结论提供规范依据。",
              _obj(cls_key={**_STR, "description": "病害类别"},
                   component={**_STR, "description": "构件首字母"},
                   query={**_STR, "description": "关键词，空格分隔"},
                   top_k={**_INT, "description": "返回条数，默认 3"}),
              t_search_spec, "trend"))
_reg(ToolSpec("plan_reflight", "复飞决策", "审计影像采集质量并反解复飞航点。"
              "五条判据：分辨率不足以支撑量测精度、同视场重合率不足、目标被画面边界裁切、"
              "疑似漏检、置信度过低。解算结果含新拍摄距离、变焦倍率、预期 GSD 与像素宽度，"
              "并导出可下发飞控的航点文件。",
              _obj(points={"type": "array", "items": _STR,
                           "description": "限定点位，留空为全库"},
                   batches={"type": "array", "items": _STR,
                            "description": "限定批次，留空为全部"}),
              t_plan_reflight, "refly", readonly=False))
_reg(ToolSpec("make_report", "报告编制", "装配完整的巡检分析报告（八章：检测概况与依据、"
              "检测方法与 AI 模型、检测结果统计、长周期病害发展趋势、复飞建议、智能研判"
              "与养护建议、智能体执行摘要、结论与说明），输出 Markdown 与白底 A4 版式的"
              "HTML 两份，返回报告编号。第七章的执行摘要由系统注入，无需模型提供。",
              _obj(llm_text={**_STR, "description": "智能研判章节的文字，留空则用规则文本"}),
              t_make_report, "report", readonly=False))


def call(name: str, /, **kwargs) -> ToolResult:
    """按名调用工具。异常一律收成 ToolResult，不让它掀翻整条智能体流水线。"""
    spec = TOOLS.get(name)
    if spec is None:
        return _err(name, f"未注册的工具：{name}")
    t0 = time.perf_counter()
    try:
        res = spec.func(**kwargs)
    except TypeError as exc:                      # 参数名对不上
        ms = int((time.perf_counter() - t0) * 1000)
        return ToolResult(False, name, "", {}, f"参数错误：{exc}", ms)
    except Exception as exc:                      # noqa: BLE001 —— 工具边界必须兜住一切
        ms = int((time.perf_counter() - t0) * 1000)
        return ToolResult(False, name, "", {}, f"{type(exc).__name__}: {exc}", ms)
    res.ms = int((time.perf_counter() - t0) * 1000)
    return res


def openai_tools(names: list[str] | None = None) -> list[dict]:
    """给大模型看的工具描述清单。不传 names 就是全部。"""
    picked = [TOOLS[n] for n in names if n in TOOLS] if names else list(TOOLS.values())
    return [s.schema() for s in picked]


def catalog() -> list[dict]:
    """给界面看的工具目录（含归属智能体与中文名）。"""
    return [{"name": s.name, "cn": s.cn, "agent": s.agent, "desc": s.desc,
             "readonly": s.readonly} for s in TOOLS.values()]
