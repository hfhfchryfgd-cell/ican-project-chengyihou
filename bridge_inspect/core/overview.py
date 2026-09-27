"""态势总览的聚合统计。

原先这几个函数长在 `web/api.py` 里。桌面版要做同一块态势总览，若各算各的，两边
迟早会因为某个 `round()` 的位数、某条 SQL 的过滤条件不同而对不上数——同一套数据
在两处显示成两个值，是最难查也最伤信任的一类问题。因此提到 core：网页端与桌面端
都调这里的同一份实现。

本模块只读写 `db` 与 `config`，不依赖任何界面框架。
"""

from __future__ import annotations

from bridge_inspect import config as cfg
from bridge_inspect import db


def _dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


def _f(v, nd: int = 2) -> float:
    """把可能为 None 的数值列安全地转成 float。SQLite 的 REAL 列允许 NULL。"""
    try:
        return round(float(v or 0.0), nd)
    except (TypeError, ValueError):
        return 0.0


def anomaly_rows(threshold: float = 0.30) -> list[dict]:
    """发展异常点位。总览页是首屏，任何一条统计算不出来都不该让整页打不开。"""
    from bridge_inspect.core import llm
    try:
        return list(llm.detect_anomalies(growth_threshold=float(threshold)) or [])
    except Exception:                                  # noqa: BLE001
        return []


def overview() -> dict:
    """一次取齐大屏需要的全部聚合量。

    做成单个调用而不是七八个：大屏要同时渲染遥测条、病害分布、批次时间轴，
    拆开会让首屏发出七八个并发请求，而它们查的是同一张表——合并后只查一遍，
    也避免各块数字来自不同时刻的快照而对不上。
    """
    s = db.stat_summary()
    imgs = _dicts(db.list_images(limit=100000))
    segs = _dicts(db.query(
        "SELECT point_id, cls_key, cls_name, batch, area_cm2, max_width_mm"
        " FROM segments WHERE area_cm2>0"))
    # 类别分布按**检测框**统计，面积与宽度仍按 segments。像素级分割只对裂缝做
    # （见 core/segment 的门控），若类别分布也走 segments，剥落、露筋那几类会整类
    # 从总览上消失——界面上看不出这是"没检出"还是"没量化"，读的人只会以为漏检。
    dets = _dicts(db.query("SELECT point_id, cls_key FROM detections"))

    # —— 批次统计 ——
    batches: list[dict] = []
    for b in cfg.BATCHES:
        bi = [r for r in imgs if r["batch"] == b.id]
        bs = [r for r in segs if r["batch"] == b.id]
        batches.append({
            "id": b.id, "name": b.name, "date": b.date, "note": b.note,
            "images": len(bi), "segments": len(bs),
            "points": len({r["point_id"] for r in bi if r["point_id"]}),
            "area_cm2": round(sum(_f(r["area_cm2"], 3) for r in bs), 2),
            "avg_overlap": round(
                sum(_f(r["overlap"], 4) for r in bi) / len(bi), 4) if bi else 0.0,
        })

    # —— 病害类别分布 ——
    # n / points 取自 detections（该类别检出多少处、覆盖多少点位），
    # area_cm2 / max_width_mm 取自 segments（只有裂缝有像素级量测）。
    classes: list[dict] = []
    for d in cfg.DISEASES:
        drows = [r for r in dets if r["cls_key"] == d.key]
        srows = [r for r in segs if r["cls_key"] == d.key]
        classes.append({
            "key": d.key, "name": d.name, "color": d.color,
            "severity": d.severity, "suggestion": d.suggestion,
            "limit_mm": d.width_limit_mm,
            "n": len(drows),
            "points": len({r["point_id"] for r in drows if r["point_id"]}),
            "area_cm2": round(sum(_f(r["area_cm2"], 3) for r in srows), 2),
            "max_width_mm": round(max((_f(r["max_width_mm"], 3) for r in srows),
                                      default=0.0), 3),
            "n_seg": len(srows),        # 其中做了像素级量测的条数
        })

    # —— 构件分布 ——
    # 按**实拍影像**归构件，不按量化记录：只做裂缝分割之后，按 segments 数会把
    # 支座、桥台、伸缩缝这几个没有裂缝的构件整类算成 0 个点位。
    comp_points: dict[str, list[str]] = {}
    for p in sorted({r["point_id"] for r in imgs if r["point_id"]}):
        comp_points.setdefault(p[:1], []).append(p)
    components = [{
        "letter": k, "name": v,
        "n": len(comp_points.get(k, [])),
        "points": comp_points.get(k, []),
    } for k, v in cfg.COMPONENTS.items()]

    return {
        "stats": s,
        "batches": batches,
        "classes": classes,
        "components": components,
        "anomalies": anomaly_rows(),
        "tasks": _dicts(db.list_tasks(limit=8)),
        "outputs": _dicts(db.list_outputs(limit=8)),
        "with_images": sorted({r["point_id"] for r in imgs if r["point_id"]}),
        "all_points": cfg.ALL_POINTS,
    }


def waypoints(batch: str = "") -> dict:
    """航点空间布局，供 RTK 航线俯视图。

    48 个点位 × 3 个批次共 144 条。只有 8 个点位有实拍影像，其余是**规划航点**——
    界面必须能区分这两者，否则评委看到 48 个点会以为都飞过了。因此每条记录带
    `flown` 标记，前端用实心/空心区分。
    """
    have: dict[tuple[str, str], dict] = {}
    for r in db.list_images(limit=100000):
        if r["point_id"] and r["batch"]:
            have.setdefault((r["point_id"], r["batch"]), dict(r))

    from bridge_inspect.core import demo_data

    out: list[dict] = []
    for pid in cfg.ALL_POINTS:
        for b in cfg.BATCHES:
            if batch and b.id != batch:
                continue
            row = have.get((pid, b.id))
            if row:
                src = "db"
                rec = {k: row.get(k) for k in (
                    "waypoint", "rtk_lon", "rtk_lat", "rtk_alt", "rtk_status",
                    "gimbal_yaw", "gimbal_pitch", "gimbal_roll",
                    "shoot_distance", "overlap")}
                rec["filename"] = row.get("filename", "")
                rec["image_id"] = row.get("id")
            else:
                src = "plan"
                rec = demo_data.waypoint_meta(pid, b.id)
                rec["image_id"] = None
                rec["filename"] = ""
            dist = _f(rec.get("shoot_distance"), 3)
            out.append({
                "point_id": pid, "component": cfg.point_component(pid),
                "batch": b.id, "flown": bool(row), "source": src,
                "lon": _f(rec.get("rtk_lon"), 7),
                "lat": _f(rec.get("rtk_lat"), 7),
                "alt": _f(rec.get("rtk_alt"), 2),
                "rtk_status": rec.get("rtk_status") or "固定解",
                "gimbal_yaw": _f(rec.get("gimbal_yaw"), 2),
                "gimbal_pitch": _f(rec.get("gimbal_pitch"), 2),
                "gimbal_roll": _f(rec.get("gimbal_roll"), 2),
                "shoot_distance": dist,
                "overlap": _f(rec.get("overlap"), 4),
                "gsd_mm_per_px": round(cfg.gsd_mm_per_px(dist), 4),
                "waypoint": rec.get("waypoint") or f"WP-{pid}-{b.id}",
                "image_id": rec.get("image_id"),
                "filename": rec.get("filename") or "",
            })
    lons = [w["lon"] for w in out if w["lon"]]
    lats = [w["lat"] for w in out if w["lat"]]
    return {
        "waypoints": out,
        "bounds": {
            "lon_min": min(lons) if lons else 0.0, "lon_max": max(lons) if lons else 0.0,
            "lat_min": min(lats) if lats else 0.0, "lat_max": max(lats) if lats else 0.0,
        },
        "batches": [{"id": b.id, "name": b.name, "date": b.date} for b in cfg.BATCHES],
        # 桥本身的尺寸一并给出：只按航点外接矩形画，画出的是"航点覆盖到的那一段"
        # （本桥 70 m），看着像一条粗线；有了桥长桥宽，才能画出 120 m × 12 m
        # 的桥面轮廓，墩位与航迹才落在它们真正该在的地方。
        "bridge": {
            "len_m": cfg.BRIDGE_LEN_M,
            "width_m": cfg.BRIDGE_WIDTH_M,
            "pier_spacing_m": cfg.PIER_SPACING_M,
            "origin_lat": cfg.ORIGIN_LAT,
            "origin_lon": cfg.ORIGIN_LON,
        },
    }
