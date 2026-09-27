"""自主复飞决策闭环：采集质量审计 + 航点反解。

这是本项目区别于普通"检测 + 出报告"方案的核心。普通方案止步于"这条裂缝 1.5 mm，
超过 0.30 mm 限值，建议加固"；本模块往前再走一步——**先判断这个 1.5 mm 值本身
可不可信，再算出下一趟该怎么飞才能把它量准**，并输出可直接下发飞控的航点文件。

## 判据 Q5：从 GSD 公式反推"这个数还准不准"

宽度由 `cv2.distanceTransform` 取最大内切圆直径得到（见 segment.py 的 quantify）。
轮廓抖动 1 个像素，直径就差 2 个像素，换算成实际尺寸即 `2·gsd`。工程上通行的口径是
**测量不确定度应小于判据限值的 1/3**，于是：

    2·gsd ≤ limit / 3
    gsd   = pixel_um · D / focal          （config.gsd_mm_per_px）
    →  D ≤ limit · focal / (6 · pixel_um)

    裂缝 crack（限值 0.30 mm）:  D ≤ 0.30 × 24 / (6 × 2.4) = 0.50 m
    错台 joint_offset（5.0 mm）: D ≤ 5.00 × 24 / (6 × 2.4) = 8.33 m

演示库里的真实数据：D02 在 2.5 m 距离下 gsd = 0.250 mm/px，量得裂缝宽 1.50 mm，
但它在画面上只有 **6 个像素**宽；`2·gsd = 0.50 mm`，已经**超过限值 0.30 mm 本身**。
也就是说这条裂缝究竟超没超限，在统计上根本不可信。而 C03 的伸缩缝错台有 78 个
像素宽，`2·gsd/限值` 仅 6%，同样拍摄距离下它的结论是可信的。

这条判据的好处是**可以被评委当场用计算器复算**，而且它触发与否取决于物理参数，
不是"重合率低于阈值"那种拍脑袋规则。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ... import config as cfg
from ... import db
from . import specs

# 各构件的安全作业距离下限（m）。低于此值旋翼下洗气流会扰动构件表面、
# 且视觉避障余量不足。按构件形态给：梁底空间最窄，支座周边遮挡最多。
SAFE_DISTANCE: dict[str, float] = {
    "A": 1.0,   # 桥墩，竖向立面，作业面开阔
    "B": 0.8,   # 梁底，空间狭窄
    "C": 0.7,   # 支座，遮挡最多但目标小，必须贴近
    "D": 0.8,   # 梁侧腹板
    "E": 1.2,   # 桥台
    "F": 1.5,   # 伸缩缝，横贯全幅，需要足够视场
}

ZOOM_MAX = 5.0          # 云台相机长焦/广角端等效焦距比上限
# 期望病害宽度在画面上的像素数下限。注意这只是**可检出**的门槛（人眼/算法能看见），
# 不保证量得准——量得准的门槛由 Q5 判据反算，通常更严，真正驱动解算的是后者。
TARGET_PX = 12
OVERLAP_HARD = cfg.OVERLAP_OK      # 0.85 红牌：必飞（说明书表 5-1、5-2 的合格线）
OVERLAP_SOFT = cfg.OVERLAP_WARN    # 0.92 黄牌：建议飞
EDGE_MARGIN = 0.02      # 检测框到画面边界的归一化边距下限
LOW_CONF = 0.55         # 最高置信度低于此值判为不可靠
BOX_FRAME_RATIO = 0.90  # 目标框宽度不应超过画面宽度的比例

# 同一点位命中多条问题时的处置优先级（小者先）。排序规则是「先看这条结论可不可信，
# 再看这条数据能不能比」：
#   Q5 分辨率不足 —— 当前量测值本身就不可信，后续一切判断都建在沙子上，最优先
#   Q3 目标贴边   —— 当前量测值被低估，也是"这次就错了"
#   Q4 疑似漏检   —— 当前根本没测到，是缺失而非错误
#   Q1 重合率     —— 影响的是历次可比性，但仍在合格线之上时属于隐患不是错误
#   Q2 置信度     —— 交给人工复核即可，不改变几何
# 注意 Q1 的红牌（< 0.85）是"高"，会由 level 先把它排到前面；这里排的是黄牌。
CODE_RANK = {"Q5": 0, "Q3": 1, "Q4": 2, "Q1": 3, "Q2": 4}


_BATCH_ORDER = {b.id: i for i, b in enumerate(cfg.BATCHES)}


def _severity(iss: "Issue") -> tuple[float, int]:
    """同一条问题出现多次时比谁更该处理（越大越优先）。

    严重程度相同时**取较晚的批次**：复飞是为了把"现在"这条病害量准，几何参数
    当然要按最新一期的尺寸算。B02 的裂缝 P1 时 0.60 mm、P3 已长到 0.96 mm，
    按 P1 反算出的 12 px 目标拿到 P3 去拍，量出来仍是不合格的。
    """
    d = iss.data
    if iss.code == "Q5":
        metric = float(d.get("ratio") or 0.0)
    elif iss.code == "Q1":
        metric = 1.0 - float(d.get("overlap") or 0.0)
    elif iss.code == "Q3":
        metric = 1.0 - float(d.get("margin") or 0.0)
    elif iss.code == "Q2":
        metric = 1.0 - float(d.get("top_conf") or 0.0)
    else:
        metric = 0.0
    return (metric, _BATCH_ORDER.get(iss.batch, -1))


@dataclass
class Issue:
    """一条采集质量问题。"""
    code: str
    point_id: str
    batch: str
    cls_key: str = ""
    cls_name: str = ""
    level: str = "中"
    title: str = ""
    detail: str = ""
    image_id: int = 0
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReflyWaypoint:
    """一个复飞航点。列名与 images 表字段对齐，可直接被导入流程消费。"""
    waypoint: str
    point_id: str
    batch: str
    rtk_lon: float
    rtk_lat: float
    rtk_alt: float
    rtk_status: str
    gimbal_yaw: float
    gimbal_pitch: float
    gimbal_roll: float
    shoot_distance: float
    zoom: float
    focal_eff_mm: float
    expected_gsd_mm_per_px: float
    expected_px_width: float
    expected_overlap: float
    target_class: str
    cls_name: str
    trigger_code: str
    trigger_desc: str
    priority: str
    source_image: str
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# 质量审计
# --------------------------------------------------------------------------
def _margin_ratio(det_x1: int, det_x2: int, img_w: int) -> float:
    """检测框到左右画面边界的最小归一化边距。越小越可能被裁切。"""
    if img_w <= 0:
        return 1.0
    return min(det_x1, img_w - det_x2) / img_w


def audit_quality(points: list[str] | None = None,
                  batches: list[str] | None = None) -> list[Issue]:
    """审计影像采集质量，返回需复飞的问题清单。

    五条判据彼此独立，同一点位可能同时命中多条（例如 A01 既分辨率不足、
    又略微贴边）。多条命中时复飞解算只取优先级最高的一条作为主因。
    """
    pts = {p for p in (points or []) if p}
    bts = {b for b in (batches or []) if b}

    # 同一个 (判据, 点位, 批次, 类别) 只留最严重的一条。一张影像里往往有多道裂缝、
    # 多个检出框，逐条累积会让 A01 的问题在清单里出现三次、B02 出现六次（实测过）。
    # 清单是给人看并据此决策的，重复项只会稀释真正该关注的那一条。
    acc: dict[tuple, Issue] = {}

    def _push(iss: Issue) -> None:
        k = (iss.code, iss.point_id, iss.batch, iss.cls_key)
        cur = acc.get(k)
        if cur is None or _severity(iss) > _severity(cur):
            acc[k] = iss

    sql = ("SELECT i.* FROM images i WHERE i.point_id<>''")
    rows = [r for r in db.query(sql)
            if (not pts or r["point_id"] in pts) and (not bts or r["batch"] in bts)]

    # 历史类别集合：用于 Q4 疑似漏检（本批次某类别"消失"了）
    hist: dict[str, set[str]] = {}
    for s in db.query("SELECT DISTINCT point_id, cls_key, batch FROM segments"):
        hist.setdefault(s["point_id"], set()).add(s["cls_key"])

    for img in rows:
        pid, bid = img["point_id"], img["batch"]
        segs = db.query("SELECT * FROM segments WHERE image_id=?", (img["id"],))
        dets = db.query("SELECT * FROM detections WHERE image_id=?", (img["id"],))

        # Q1 重合率不足
        ov = float(img["overlap"] or 0.0)
        if ov < OVERLAP_HARD:
            _push(Issue(
                "Q1", pid, bid, level="高", image_id=img["id"],
                title=f"重合率 {ov:.0%} 低于合格线 {OVERLAP_HARD:.0%}",
                detail="同视场复拍的重合率不足，该点位历次影像不构成可比序列，"
                       "本批次数据无法用于趋势判断。",
                data={"overlap": ov, "threshold": OVERLAP_HARD}))
        elif ov < OVERLAP_SOFT:
            _push(Issue(
                "Q1", pid, bid, level="中", image_id=img["id"],
                # 这一档只保留一位小数：0.916 与建议值 0.92 若都按整数个百分点显示，
                # 就会写出"重合率 92% 未达建议值 92%"这种自相矛盾的句子。
                title=f"重合率 {ov:.1%} 未达建议值 {OVERLAP_SOFT:.0%}"
                      f"（合格线 {OVERLAP_HARD:.0%}）",
                # 这一档是"合格但不理想"：数据仍可用于趋势判断，只是视场有轻微偏移。
                # 措辞上不能说"低于合格线"——上面那档才是。
                detail="重合率在合格线之上、未达建议值，历次影像仍可比，"
                       "但复拍视场存在轻微偏移，建议下一批次校正站位。",
                data={"overlap": ov, "threshold": OVERLAP_SOFT}))

        # Q2 置信度偏低
        if dets:
            top = max(float(d["conf"]) for d in dets)
            if top < LOW_CONF:
                _push(Issue(
                    "Q2", pid, bid, cls_key=dets[0]["cls_key"],
                    cls_name=dets[0]["cls_name"], level="中", image_id=img["id"],
                    title=f"最高置信度仅 {top:.2f}",
                    detail="检出置信度偏低，通常由逆光或欠曝导致对比度不足引起；"
                           "建议复飞时提高曝光补偿并人工复核。",
                    data={"top_conf": top, "threshold": LOW_CONF}))

        # Q3 疑似裁切（框贴边）
        if dets and img["width"]:
            worst = min(dets, key=lambda d: _margin_ratio(d["x1"], d["x2"], img["width"]))
            m = _margin_ratio(worst["x1"], worst["x2"], img["width"])
            if m < EDGE_MARGIN:
                _push(Issue(
                    "Q3", pid, bid, cls_key=worst["cls_key"],
                    cls_name=worst["cls_name"], level="中", image_id=img["id"],
                    title=f"{worst['cls_name']}贴合画面边界（边距 {m:.1%}）",
                    detail="检测框触及画面边缘，病害两端可能被裁切，"
                           "面积与周长会被低估，需拉远视场重拍。",
                    data={"margin": m, "threshold": EDGE_MARGIN,
                          "box_px": int(worst["x2"]) - int(worst["x1"]),
                          "img_w": int(img["width"])}))

        # Q4 疑似漏检：该点位历史上出现过的类别，本批次一张都没检出
        have = {s["cls_key"] for s in segs}
        missing = (hist.get(pid, set()) - have) if segs else set()
        if segs and missing:
            for ck in sorted(missing):
                dc = cfg.DISEASE_BY_KEY.get(ck)
                _push(Issue(
                    "Q4", pid, bid, cls_key=ck,
                    cls_name=dc.name if dc else ck, level="高", image_id=img["id"],
                    title=f"{dc.name if dc else ck} 本批次未检出",
                    detail="该点位历史批次存在此类别病害，同视场复拍本不应消失，"
                           "疑似漏检或被遮挡，需复飞确认。",
                    data={"missing_cls": ck}))

        # Q5 分辨率不足 —— 本方案的核心判据，详见模块开头推导
        for s in segs:
            dc = cfg.DISEASE_BY_KEY.get(s["cls_key"])
            limit = dc.width_limit_mm if dc else None
            if limit is None:
                continue                       # 面状病害无宽度限值，不适用
            dist = float(img["shoot_distance"] or 0.0)
            gsd = cfg.gsd_mm_per_px(dist)
            if gsd <= 0:
                continue
            unc = 2 * gsd                      # 宽度量测的不确定度（mm）
            ratio = unc / limit                # 与限值之比；判据是 ≤ 1/3
            if ratio <= 1.0 / 3.0:
                continue
            w = float(s["max_width_mm"] or 0.0)
            px = w / gsd
            _push(Issue(
                "Q5", pid, bid, cls_key=s["cls_key"], cls_name=s["cls_name"],
                level="高", image_id=img["id"],
                title=(f"{s['cls_name']}量测不确定度 {unc:.2f} mm，"
                       f"达限值 {limit:g} mm 的 {ratio:.0%}"),
                detail=(f"拍摄距离 {dist:.2f} m 下 GSD 为 {gsd:.3f} mm/px，"
                        f"该{s['cls_name']}在画面上仅 {px:.1f} 像素宽。轮廓抖动 1 像素"
                        f"即引起 {unc:.2f} mm 的宽度偏差，已超过规范限值 {limit:g} mm 的"
                        f"三分之一，超限与否在统计上不可判定。"),
                data={"distance": dist, "gsd": gsd, "width_mm": w, "px": px,
                      "uncertainty_mm": unc, "limit_mm": limit, "ratio": ratio}))

    order = {"高": 0, "中": 1, "低": 2}
    return sorted(acc.values(),
                  key=lambda i: (order.get(i.level, 3), CODE_RANK.get(i.code, 9),
                                 i.point_id, i.batch))


# --------------------------------------------------------------------------
# 航点反解
# --------------------------------------------------------------------------
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点间距离，米。用于判断复飞站位与基准站位的偏差。"""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def required_px(width_mm: float, limit_mm: float) -> int:
    """满足 Q5 判据所需的病害像素宽度。

    不确定度 2·gsd = 2·(w/px)，要求它 ≤ limit/3：

        2w/px ≤ limit/3   →   px ≥ 6w/limit

    裂缝类（限值小）算出来远大于 TARGET_PX：A01 的 1.08 mm 裂缝要 22 px，
    D02 的 1.50 mm 要 30 px。**这个数才是驱动拍摄距离的量**——若只按固定 12 px
    解算，量出来的宽度看着够粗，实际不确定度仍是限值的两倍，等于白飞一趟。
    """
    if width_mm <= 0 or limit_mm <= 0:
        return TARGET_PX
    return max(TARGET_PX, int(math.ceil(6.0 * width_mm / limit_mm)))


def solve_distance(width_mm: float, prefix: str,
                   target_px: int = TARGET_PX) -> dict:
    """反解拍摄距离与变焦倍率，使病害宽度达到 target_px 个像素。

    先试"单纯靠近"：把 gsd 压到 width/target_px 所需的距离。
    若该距离跌破构件的安全作业下限，则退回安全距离、用变焦补足分辨率——
    这条分支是必要的，因为梁底与支座周围根本飞不进去，
    而相机变焦不会改变安全裕度。
    """
    d_safe = SAFE_DISTANCE.get(prefix, 1.0)
    gsd_target = width_mm / max(1, target_px) if width_mm > 0 else 0.0
    if gsd_target <= 0:
        return {"distance": d_safe, "zoom": 1.0, "focal_eff": cfg.CAM_FOCAL_MM,
                "gsd": cfg.gsd_mm_per_px(d_safe), "zoom_limited": False,
                "target_px": target_px}

    d_by_dist = gsd_target * cfg.CAM_FOCAL_MM / cfg.CAM_PIXEL_UM
    if d_by_dist >= d_safe:
        return {"distance": round(d_by_dist, 2), "zoom": 1.0,
                "focal_eff": cfg.CAM_FOCAL_MM,
                "gsd": round(cfg.gsd_mm_per_px(d_by_dist), 4),
                "zoom_limited": False, "target_px": target_px}

    focal_req = cfg.CAM_PIXEL_UM * d_safe / gsd_target
    zoom = min(focal_req / cfg.CAM_FOCAL_MM, ZOOM_MAX)
    gsd = cfg.CAM_PIXEL_UM * d_safe / (cfg.CAM_FOCAL_MM * zoom)
    return {"distance": round(d_safe, 2), "zoom": round(zoom, 2),
            "focal_eff": round(cfg.CAM_FOCAL_MM * zoom, 1),
            "gsd": round(gsd, 4), "zoom_limited": zoom >= ZOOM_MAX,
            "target_px": target_px}


def _dominant_class(point_id: str, batch: str) -> str:
    """该点位该批次里面积最大的病害类别。

    Q1（重合率）与 Q2（置信度）是影像级问题，载荷里没有类别；但它们同样要落到
    复飞航点上，而航点必须写清"这趟去拍什么"。取主病害类别即可——复飞本就是把
    整个点位重拍一遍。
    """
    row = db.query_one(
        "SELECT s.cls_key AS k FROM segments s"
        " JOIN images i ON i.id = s.image_id"
        " WHERE i.point_id=? AND i.batch=?"
        " GROUP BY s.cls_key ORDER BY MAX(s.area_cm2) DESC LIMIT 1",
        (point_id, batch))
    return row["k"] if row and row["k"] else ""


def _max_width(point_id: str, batch: str, cls_key: str) -> float:
    """该点位该批次该类别的最大实测宽度。Q1/Q3 的问题载荷里没有宽度，
    补上它复飞表才能给出"复飞后目标会有多少像素"这个可核对的预期值。"""
    if not cls_key:
        return 0.0
    row = db.query_one(
        "SELECT MAX(s.max_width_mm) AS w FROM segments s"
        " JOIN images i ON i.id = s.image_id"
        " WHERE i.point_id=? AND i.batch=? AND s.cls_key=?",
        (point_id, batch, cls_key))
    return float(row["w"] or 0.0) if row and row["w"] is not None else 0.0


def _baseline_station(point_id: str) -> dict | None:
    """该点位 P1 基准批次的站位与云台姿态——同视场复拍的参照。

    以 P1 为基准而不是上一批次：批次间站位本身可能已经漂移，
    拿漂移过的站位当参照会把误差继续传下去。
    """
    first = cfg.BATCHES[0].id if cfg.BATCHES else "P1"
    return db.query_one(
        "SELECT rtk_lon, rtk_lat, rtk_alt, gimbal_yaw, gimbal_pitch, gimbal_roll,"
        " shoot_distance FROM images WHERE point_id=? AND batch=?"
        " ORDER BY id LIMIT 1", (point_id, first))


def solve_waypoint(issue: Issue, target_px: int = TARGET_PX) -> ReflyWaypoint | None:
    """按一条质量问题解算复飞航点。"""
    img = db.query_one("SELECT * FROM images WHERE id=?", (issue.image_id,))
    if not img:
        return None
    pid = issue.point_id
    prefix = pid[:1].upper()
    d0 = float(img["shoot_distance"] or 0.0)
    base = _baseline_station(pid)

    dist, zoom, focal_eff = d0, 1.0, cfg.CAM_FOCAL_MM
    note = ""

    if issue.code == "Q5":
        w = float(issue.data.get("width_mm") or 0.0)
        lim = float(issue.data.get("limit_mm") or 0.0)
        # 驱动距离的是判据反算出的像素宽度。传进来的 target_px 若仍是默认下限，
        # 说明调用方没有特别要求，就按 Q5 判据反算——12 px 只够"看得见"，
        # 裂缝要 20~30 px 才量得准。
        tgt = target_px if target_px != TARGET_PX else required_px(w, lim)
        sol = solve_distance(w, prefix, tgt)
        dist, zoom, focal_eff = sol["distance"], sol["zoom"], sol["focal_eff"]
        after = (2 * sol["gsd"] / lim) if lim > 0 else 0.0
        note = (f"目标像素宽 {tgt} px，不确定度自 {issue.data['ratio']:.0%} "
                f"降至 {after:.0%}")
        if sol["zoom_limited"]:
            note += "；变焦已到上限，实际不确定度仍高于 1/3 判据"

    elif issue.code == "Q3":
        # 拉远使目标框宽不超过画面宽度的 90%
        box_px = float(issue.data.get("box_px") or 0.0)
        img_w = float(issue.data.get("img_w") or 0.0)
        if box_px > 0 and img_w > 0:
            dist = d0 * box_px / (BOX_FRAME_RATIO * img_w)
            dist = max(dist, SAFE_DISTANCE.get(prefix, 1.0))
            note = f"视场由 {d0:.2f} m 拉远至 {dist:.2f} m，使目标完整入画"
        else:
            dist = d0 * 1.15
            note = "视场适度拉远，确保目标两端完整入画"

    elif issue.code == "Q1":
        # 回到基准站位；重合率问题本质是站位漂移
        dist = float(base["shoot_distance"]) if base else d0
        if base:
            note = "站位回退至 P1 基准航点，恢复同视场复拍条件"
        else:
            note = "缺基准站位记录，按标称距离重拍"

    elif issue.code == "Q2":
        note = "几何参数不变，曝光补偿 +0.7 EV 后重拍，并标记人工复核"

    elif issue.code == "Q4":
        note = "按历史批次的站位与云台姿态补拍，确认病害是否真实消失"

    dist = round(max(dist, 0.2), 2)
    gsd = cfg.gsd_mm_per_px(dist) / max(zoom, 1e-6)
    # Q1/Q2 是影像级问题，自身不带类别；落到航点上要写明这趟去拍什么
    cls_key = issue.cls_key or _dominant_class(pid, issue.batch)
    cls_name = issue.cls_name or (cfg.DISEASE_BY_KEY[cls_key].name
                                  if cls_key in cfg.DISEASE_BY_KEY else "")
    w_mm = float(issue.data.get("width_mm") or 0.0)
    if w_mm <= 0:
        w_mm = _max_width(pid, issue.batch, cls_key)
    px = (w_mm / gsd) if (gsd > 0 and w_mm > 0) else 0.0

    # 站位优先沿用基准，保证同视场；基准缺失时退回当前站位
    lon = float(base["rtk_lon"]) if base else float(img["rtk_lon"] or 0.0)
    lat = float(base["rtk_lat"]) if base else float(img["rtk_lat"] or 0.0)
    alt = float(base["rtk_alt"]) if base else float(img["rtk_alt"] or 0.0)
    yaw = float(base["gimbal_yaw"]) if base else float(img["gimbal_yaw"] or 0.0)
    pitch = float(base["gimbal_pitch"]) if base else float(img["gimbal_pitch"] or 0.0)
    roll = float(base["gimbal_roll"]) if base else float(img["gimbal_roll"] or 0.0)

    return ReflyWaypoint(
        waypoint=f"WP-{pid}-{issue.batch}-REF",
        point_id=pid, batch=issue.batch,
        rtk_lon=round(lon, 7), rtk_lat=round(lat, 7), rtk_alt=round(alt, 2),
        rtk_status="固定解",
        gimbal_yaw=round(yaw, 2), gimbal_pitch=round(pitch, 2),
        gimbal_roll=round(roll, 2),
        shoot_distance=dist, zoom=round(zoom, 2), focal_eff_mm=focal_eff,
        expected_gsd_mm_per_px=round(gsd, 4),
        expected_px_width=round(px, 1),
        # 复飞要的是同视场可比，目标重合率取合格线之上
        expected_overlap=0.92,
        target_class=cls_key, cls_name=cls_name,
        trigger_code=issue.code, trigger_desc=issue.title,
        priority=issue.level, source_image=img["filename"], note=note,
    )


# --------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------
def plan(points: list[str] | None = None, batches: list[str] | None = None,
         target_px: int = TARGET_PX
         ) -> tuple[list[Issue], list[ReflyWaypoint]]:
    """审计 → 每个点位取优先级最高的一条问题 → 解算航点。

    同一点位命中多条时不重复安排航点：一趟飞过去本来就能同时解决，
    分两条反而会给出互相矛盾的拍摄距离。
    """
    issues = audit_quality(points, batches)
    best: dict[str, Issue] = {}
    order = {"高": 0, "中": 1, "低": 2}
    for i in issues:
        cur = best.get(i.point_id)
        # 同级别同判据时取较晚批次，与 _severity 的口径一致
        key = (order.get(i.level, 9), CODE_RANK.get(i.code, 9),
               -_BATCH_ORDER.get(i.batch, -1))
        cur_key = (order.get(cur.level, 9), CODE_RANK.get(cur.code, 9),
                   -_BATCH_ORDER.get(cur.batch, -1)) if cur else None
        if cur is None or key < cur_key:
            best[i.point_id] = i

    wps: list[ReflyWaypoint] = []
    for pid in sorted(best):
        w = solve_waypoint(best[pid], target_px)
        if w:
            wps.append(w)
    return issues, wps


FIELDS = list(ReflyWaypoint.__dataclass_fields__.keys())


def write_mission(waypoints: list[ReflyWaypoint],
                  out_dir: Path | None = None,
                  stamp: str | None = None) -> tuple[Path, Path]:
    """写出复飞航点 CSV 与 JSON，返回两个路径。

    CSV 列名与 images 表字段一一对应，可直接被导入流程消费；
    JSON 供网页端与后续飞控对接使用。
    """
    from ... import config as _cfg
    d = Path(out_dir) if out_dir else (_cfg.OUT_DIR / "refly")
    d.mkdir(parents=True, exist_ok=True)
    st = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = waypoints[0].batch if waypoints else "P1"
    csv_path = d / f"ReflyMission_{batch}_{st}.csv"
    json_path = d / f"ReflyMission_{batch}_{st}.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for wp in waypoints:
            w.writerow(wp.to_dict())

    json_path.write_text(json.dumps({
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch": batch,
        "count": len(waypoints),
        "waypoints": [wp.to_dict() for wp in waypoints],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


# 各判据对应的条文检索关键词。取的是**为什么这条判据成立**，不是病害本身怎么处置——
# 复飞决策要引用的是作业条件类条文（分辨率、重合率、记录基准），
# 病害处置条文留给研判智能体在用 `_grade` 定级时引用。
_CODE_KW: dict[str, list[str]] = {
    "Q5": ["分辨率", "量测", "精度"],     # 第20条：分辨率不低于量测精度的 1/3
    "Q1": ["重合率", "复拍", "同视场"],   # 第19条：复拍重叠率不低于 85%
    "Q3": ["记录", "基准", "比对"],       # 目标被裁切 → 记录不完整、无法按同一基准比对
    "Q4": ["记录", "基准", "比对"],       # 漏检同理
    "Q2": ["记录"],
}


def citations_for(issues: list[Issue]) -> list[dict]:
    """给审计结果挂规范依据。第 20 条正是 Q5 判据的出处。

    检索**不限定构件**：构件过滤是为了找"某类构件该查什么"，而这里要找的是
    "这条质量判据凭什么成立"，与病害长在哪个构件上无关。带上构件反而会把条文
    误杀——演示库里 C03 是一条伸缩缝错台，而伸缩缝条文限定构件 F，
    加上构件条件后 JT 系列整体被淘汰，该点位一条依据都引不出来。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for i in issues:
        cls = i.cls_key or ""
        top = specs.search([cls] if cls else [], [],
                           top_k=2, extra_kw=_CODE_KW.get(i.code, []),
                           strict_component=False)
        for c in top:
            if c.cid not in seen:
                seen.add(c.cid)
                out.append(c.to_dict())
    return out
