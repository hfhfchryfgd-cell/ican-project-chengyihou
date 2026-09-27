"""巡检报告与数据导出。

产出《作品说明书》§6.4 与 §7.3 要求的巡检报告与随附数据文件，共三种落地格式：

  · HTML     —— **主交付件**。白底 A4 版式的标准检测报告：报头（编号 / 对象 / 依据 /
                编制单位）、结论摘要框、八章正文、签署栏。双击即可在浏览器查看、
                直接打印或另存 PDF；同一份文件也能被桌面版的 `QTextBrowser` 渲染，
                因此报告可以在**系统内直接翻看**，不必先交给外部程序。
  · Markdown —— 简版。同八章的纯文本，便于粘贴进 Word 或作品说明书。
  · CSV / XLSX —— 点位级量化数据，供长周期监测与外部工具读取。

## 为什么 HTML 不再由 Markdown 渲染

早先 HTML 走的是一条 20 行的 `_md_to_html`：只认 `|` 表格、`- ` 列表与 `#{1,4}` 标题。
而一份标准检测报告要的报头网格、结论摘要框、分级徽章、脚注角标、签署栏与分页控制，
在这个方言里一样都表达不出来——扩方言等于自己写一个 markdown 库。所以 HTML 改为
**直接生成**，Markdown 降为"简版"；两者共用一次 `_collect()` 取数，杜绝两份报告
数字打架。

## 一处必须守住的兼容约束

同一份 HTML 既要浏览器漂亮打印，又要在 `QTextBrowser`（Qt 富文本引擎）里不散架。
Qt 支持的 CSS 子集很窄：禁 `:root`/`var()`、`display:flex|grid`、`float`、
`border-radius`、`box-shadow`、`opacity`、`transform`、`::before/::after`、
`:nth-child()`、`calc()`、`hsl()`、`@font-face`。颜色一律 `#RRGGBB`，尺寸一律
`pt/px`，**`mm` 只出现在 `@page` 与 `@media` 里**（Qt 对 mm 支持不稳）。
改 `_DOC_CSS` 之前请先读完这段。

## 报告编号

    BIS-<YYYYMMDD>-<批次码>-<HHMMSS>        例：BIS-20260924-P1P2P3-141530

只由「生成时刻（到秒）」与「覆盖批次」两个可观测量决定：不查库、不依赖计数器，
换台机器、换个输出目录重算编号不变；正文的「编制日期」也打印到同一秒，两者可互相验证。
编号要能被人在电话里念出来，所以不加内容指纹。
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from .. import config as cfg
from .. import db
from . import llm
from . import overview

# 处置档位。前三个来自 llm._grade()；"待量化"是本模块补的第四档——面状病害
# （剥落、露筋、渗水、蜂窝）不做像素级分割，没有宽度与面积，用 _grade() 去套
# 只会得到一句"病害发展平缓"，读的人会以为算法已经判过它了。
GRADE_ORDER = ("加固", "维修", "观察", "待量化")
GRADE_CN = {"加固": "加固", "维修": "维修", "观察": "观察", "待量化": "观察（待量化）"}
GRADE_ADVICE = {
    "加固": "应立即安排专项检测，并编制加固方案",
    "维修": "应在本巡检周期内安排维修处理",
    "观察": "维持现有巡检周期，继续观察",
    "待量化": "该类病害未开展像素级量化，按检出记录列入观察清单",
}
# 徽章的 CSS 类名。用 ASCII 而不是直接拿中文当类名：CSS 标识符允许非 ASCII，
# 但 Qt 的样式表解析器对中文类名没有保证，而这份 HTML 要在两处渲染。
GRADE_CSS = {"加固": "ga", "维修": "we", "观察": "ob", "待量化": "wq"}

# 报告骨架。放在这里而不是各自写在 build_html / build_markdown 里：两处各写一遍，
# 迟早会出现 HTML 是八章、Markdown 是七章的情况，而"两份报告能逐章对照"正是
# 第七章降级形态保留固定编号的理由。工具 `make_report` 的章节摘要也取这里。
CC = ("一", "二", "三", "四", "五", "六", "七", "八")
CHAPTER_TITLES = (
    "检测概况与依据", "检测方法与 AI 模型", "检测结果统计", "长周期病害发展趋势",
    "复飞建议", "智能研判与养护建议", "智能体执行摘要", "结论与说明",
)
CHAPTERS = tuple(f"{CC[i]}、{t}" for i, t in enumerate(CHAPTER_TITLES))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _out_dir(settings: dict | None = None) -> Path:
    s = settings or cfg.load_settings()
    d = Path(s.get("output_dir") or cfg.OUT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp(out: Path) -> str:
    """取一个尚未被占用的时间戳 `YYYYmmdd_HHMMSS`。

    同一秒内连续生成两次时**向后顺延一秒**，而不是加 `_2` 后缀：报告编号里出现
    "BIS-20260924-P1P2P3-141530_2" 不像一份正式文件的编号，而顺延后的编号与正文
    「编制日期」仍由同一个戳导出，两者照样对得上。
    """
    t = datetime.now()
    for _ in range(120):
        s = t.strftime("%Y%m%d_%H%M%S")
        if not any((out / f"桥梁巡检报告_{s}{ext}").exists() for ext in (".md", ".html")):
            return s
        t += timedelta(seconds=1)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def next_stamp(settings: dict | None = None) -> str:
    """本次报告将使用的戳。

    暴露出来是为了让「先预览、后保存」的两步操作拿到**同一个编号**——否则屏幕上
    看到的是 141530，存下来变成 141532，用户会以为是两次生成。
    """
    return _stamp(_out_dir(settings))


def _sort_batches(batches) -> list[str]:
    """按 `cfg.BATCHES` 的先后排批次。

    不用字典序：出现 P10 的那天，字典序会把 P10 排到 P2 前面。`reflight._BATCH_ORDER`
    是同一个理由。
    """
    order = {b.id: i for i, b in enumerate(cfg.BATCHES)}
    return sorted({b for b in batches if b}, key=lambda x: (order.get(x, 99), x))


def _batch_code(batches: list[str]) -> str:
    """批次码：按 `cfg.BATCHES` 的先后拼接，无数据时 `ALL`。"""
    ids = _sort_batches(batches)
    return "".join(ids) if ids else "ALL"


def _report_no(stamp: str, batches: list[str]) -> str:
    return f"{cfg.REPORT_NO_PREFIX}-{stamp[:8]}-{_batch_code(batches)}-{stamp[9:]}"


def _fmt_stamp(stamp: str) -> str:
    """戳 → `2026-09-24 14:15:30`。从编号同一串字符导出，两处不可能对不上。"""
    return (f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]} "
            f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}")


def _esc(v) -> str:
    """转义为 HTML 文本。

    `llm_text` 是模型生成的自由文本，`group_concat` 出来的批次串也可能带引号或尖括号。
    漏掉任何一处都会破版；统一走这个函数，不自己拼字符串。
    """
    import html as _html

    return _html.escape("" if v is None else str(v), quote=False)


def _f(v) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _limit_text(cls_key: str) -> str:
    """该类别"宽度本身即成害"的限值。面状病害没有限值，印「—」而不是 0——
    0 会被读成"限值为零"。"""
    dc = cfg.DISEASE_BY_KEY.get(cls_key)
    if dc and dc.width_limit_mm is not None:
        return f"{dc.width_limit_mm:g}"
    return "—"


# --------------------------------------------------------------------------
# 取数：一次取齐，HTML 与 Markdown 共用
# --------------------------------------------------------------------------
def _latest_quant() -> dict[tuple[str, str], dict]:
    """每 点位×病害类别 取**最新批次**的量化值。

    批次先后按 `cfg.BATCHES` 顺序，不用字典序（同 `_batch_code` 的理由）。
    面积取 `area_max`（该批次里最不利的那一道缝），与趋势图的批内平均值是两个口径——
    报告要说的是"最不利单处"，不是"平均"。
    """
    order = {b.id: i for i, b in enumerate(cfg.BATCHES)}
    out: dict[tuple[str, str], dict] = {}
    for r in db.timeseries_overview():
        key = (r["point_id"], r["cls_key"])
        rank = order.get(r["batch"], 99)
        cur = out.get(key)
        if cur is None or rank > cur["_rank"]:
            out[key] = {
                "batch": r["batch"], "_rank": rank,
                "area": _f(r["area_max"]), "width": _f(r["width"]),
                "length": _f(r["length"]),
            }
    return out


def _point_details() -> list[dict]:
    """三章 3-2 的逐点位明细。

    **检出数取 `detections`，量测值取 `segments`**，两者不能合并成一条 SQL——
    像素级分割只对裂缝做（`cfg.SEGMENT_CLASSES`），按 segments 统计会让另外六类
    整类归零，报告里写成"七类病害里六类未检出"。口径与 `core/overview.py` 的
    类别分布一致，报告与总览页不会给出两个数。
    """
    quant = _latest_quant()
    rows: list[dict] = []
    for i, r in enumerate(db.point_cls_detail(), start=1):
        cls_key = r["cls_key"]
        dc = cfg.DISEASE_BY_KEY.get(cls_key)
        q = quant.get((r["point_id"], cls_key))
        row = {
            "no": i,
            "point": r["point_id"],
            "component": cfg.point_component(r["point_id"]),
            "cls_key": cls_key,
            "cls_name": r["cls_name"],
            "severity": dc.severity if dc else "—",
            "det_n": int(r["det_n"] or 0),
            "img_n": int(r["img_n"] or 0),
            "n_batch": int(r["n_batch"] or 0),
            "batches": r["batches"] or "",
            "avg_conf": _f(r["avg_conf"]),
            "batch": q["batch"] if q else "",
            "area": q["area"] if q else None,
            "width": q["width"] if q else None,
            "length": q["length"] if q else None,
        }
        if q is None:
            row["grade"], row["why"] = "待量化", GRADE_ADVICE["待量化"]
        else:
            t = llm._trend_of(row["point"], cls_key)
            growth = _f(t["growth"]) if t else 0.0
            dc_sev = dc.severity if dc else "低"
            g, why = llm._grade(growth, row["width"], dc_sev, cls_key)
            row["grade"], row["why"] = g, why
            row["growth"] = growth
        rows.append(row)
    return rows


def _grade_summary(details: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {g: [] for g in GRADE_ORDER}
    for r in details:
        out[r["grade"]].append(r)
    return out


def _trend_rows() -> list[dict]:
    """四章 4-1 的趋势汇总。按 点位×类别 去重，取首末两个批次比对。"""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in db.timeseries_overview():
        key = (r["point_id"], r["cls_key"])
        if key in seen:
            continue
        seen.add(key)
        t = llm._trend_of(r["point_id"], r["cls_key"])
        if not t:
            continue
        dc = cfg.DISEASE_BY_KEY.get(t["cls"])
        grade, why = llm._grade(t["growth"], t["width"],
                                dc.severity if dc else "低", t["cls"])
        rows.append({
            "point": t["point"], "component": t["component"],
            "cls_name": t["cls_name"], "batch0": t["batch0"], "batch1": t["batch1"],
            "n_batch": t["n_batch"],
            "area0": t["area0"], "area1": t["area1"], "growth": t["growth"],
            "width": t["width"], "grade": grade, "why": why,
            "abnormal": t["growth"] >= 0.30,
        })
    rows.sort(key=lambda x: -x["growth"])
    return rows


def _quality_rows() -> list[dict]:
    """三章 3-3 的采集质量，按批次统计重合率。

    重合率取自 `images.overlap`（标定值），合格线 `cfg.OVERLAP_OK`。这一节与
    五章的复飞建议是同一件事的两个视角：这里说"哪一批拍得不合格"，那里说"下一趟
    怎么飞"。
    """
    imgs = [dict(r) for r in db.list_images(limit=100000)]
    out: list[dict] = []
    for b in cfg.BATCHES:
        rows = [r for r in imgs if r["batch"] == b.id]
        if not rows:
            continue
        ovs = [_f(r["overlap"]) for r in rows]
        low = [r for r in rows if _f(r["overlap"]) < cfg.OVERLAP_OK]
        out.append({
            "id": b.id, "name": b.name, "date": b.date, "n": len(rows),
            "avg": sum(ovs) / len(ovs), "ok": len(rows) - len(low),
            "low": len(low),
            "points": len({r["point_id"] for r in rows if r["point_id"]}),
        })
    return out


def _reflight() -> tuple[list, list, str]:
    """复飞审计与航点解算，返回 (问题, 航点, 错误)。

    延迟导入：`import core.agent` 会经 `tools.py` 连带拉进 cv2 与 numpy，而报告在
    「预览正文」这类只读路径上也会被调用，不该为它付这份开销。
    出错不抛出——报告少一章远好于整份出不来，错误原文会印在该章里。
    """
    from .agent import reflight

    try:
        issues, wps = reflight.plan()
        return list(issues), list(wps), ""
    except Exception as exc:                            # noqa: BLE001
        return [], [], f"{type(exc).__name__}: {exc}"


def _clause_rows(cls_keys: list[str]) -> list[dict]:
    """1-3 检测依据：四条通用条文 + 每个出现过的病害类别两条。"""
    from .agent import specs

    out: list[dict] = []
    seen: set[str] = set()

    def _add(c) -> None:
        if c is not None and c.cid not in seen:
            seen.add(c.cid)
            out.append(c.to_dict())

    for cid in ("GN-01", "GN-03", "GN-04", "GN-05"):
        _add(specs.CLAUSE_BY_ID.get(cid))
    for k in cls_keys:
        for c in specs.for_disease(k, top_k=2):
            _add(c)
    return out


def _collect(settings: dict | None = None, llm_text: str = "",
             trace: dict | None = None, stamp: str | None = None) -> dict:
    """一次取齐整份报告要用的全部数据。

    HTML 与 Markdown 两条产出都从这里取数。分成两次取会走样——同一份报告里
    「检出 45 处」和「检出 44 处」同时出现，是最难查也最伤信任的一类问题。
    """
    st = db.stat_summary()
    ov = overview.overview()
    issues, wps, refly_err = _reflight()
    details = _point_details()
    trends = _trend_rows()
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    # 报告覆盖的批次：有影像或有量化记录的批次，再加上明细表里出现过的批次号。
    # 不能只看"最新批次"——报告的四章讲的正是跨批次发展，编号里却只出现 P3，
    # 收报告的人会以为只有一个批次的数据。
    covered = _sort_batches(
        {b["id"] for b in ov["batches"] if b["images"] or b["segments"]}
        | {x.strip() for r in details for x in (r["batches"] or "").split(",") if x.strip()}
        | {t[k] for t in trends for k in ("batch0", "batch1")}
    )

    org = (settings or cfg.load_settings()).get("report_org") or cfg.REPORT_ORG
    return {
        "stamp": stamp,
        "no": _report_no(stamp, covered),
        "date": _fmt_stamp(stamp),
        "org": org,
        "stats": st,
        "batches": ov["batches"],
        "classes": ov["classes"],
        "components": ov["components"],
        "anomalies": ov["anomalies"],
        "covered": covered,
        "details": details,
        "grades": _grade_summary(details),
        "trends": trends,
        "quality": _quality_rows(),
        "issues": issues,
        "waypoints": wps,
        "refly_err": refly_err,
        "clauses": _clause_rows([c["key"] for c in ov["classes"] if c["n"]]),
        "llm_text": (llm_text or "").strip() or _rule_text(),
        "trace": trace,
    }


def _rule_text() -> str:
    """没有模型文本时的规则研判正文，去掉面向问答的那句开场白。

    `llm.offline_answer("", "")` 的开场白（「未在问题中识别到具体点位，以下为
    全桥汇总」）是给工作台问答用的——那边确实有"用户问的那句话"。报告里没有
    "问题"这个前提，照搬过来读着就像串了台。只丢这一行，其余规则结论原样保留。
    """
    return "\n".join(
        ln for ln in llm.offline_answer("", "").splitlines()
        if not ln.startswith("未在问题中识别")
    )


# --------------------------------------------------------------------------
# 结论摘要：报告第一眼要能看清的东西
# --------------------------------------------------------------------------
def _verdict(d: dict) -> tuple[str, str]:
    """返回 (整体判定一句话, 分级徽章的 HTML)。"""
    st, g = d["stats"], d["grades"]
    if not st["detections"]:
        return ("本库尚无已落库的病害检测记录，报告仅反映巡检批次与采集质量，"
                "暂无检测结论。", "")
    n_ga, n_we = len(g["加固"]), len(g["维修"])
    n_pt = len({r["point"] for r in d["details"]})
    # 分档计数为 0 的一档不写进句子里——"其中 3 项达加固档、0 项达维修档"这种句子
    # 读起来像漏了数，而它其实只是那一档没有项。
    tail = "、".join(f"{n} 项达**{k}**档" for k, n in
                    (("加固", n_ga), ("维修", n_we)) if n)
    if n_ga:
        head = (f"本次巡检共检出病害 {st['detections']} 处，涉及 {n_pt} 个点位，"
                f"其中 {tail}，整体判定为「需专项处置」："
                f"应优先安排最不利点位的专项检测与加固方案编制。")
    elif n_we:
        head = (f"本次巡检共检出病害 {st['detections']} 处，涉及 {n_pt} 个点位，"
                f"{tail}，无加固档项，"
                f"整体判定为「需维修处置」：应在本巡检周期内安排维修。")
    else:
        head = (f"本次巡检共检出病害 {st['detections']} 处，涉及 {n_pt} 个点位，"
                f"各项发展平缓，整体判定为「持续观察」。")

    # 徽章之间要有一个真的空格字符。只靠 CSS 的 margin-right 的话，在
    # QTextBrowser 里两个徽章会连成"加固 3观察（待量化） 5"——Qt 不支持行内元素的
    # 外边距，浏览器支持。一处空格两边都成立。
    badges = " ".join(
        f'<span class="bdg bdg-{GRADE_CSS[k]}">{GRADE_CN[k]} {len(g[k])}</span>'
        for k in GRADE_ORDER if g[k]
    )
    return head, badges


def _worst(d: dict, top: int = 3) -> list[str]:
    """最不利项 1~3 条：先按档位，再按量值排序。"""
    rank = {g: i for i, g in enumerate(GRADE_ORDER)}
    rows = [r for r in d["details"] if r["area"] is not None]
    rows.sort(key=lambda r: (rank.get(r["grade"], 9), -_f(r["area"])))
    out: list[str] = []
    for r in rows[:top]:
        out.append(
            f"{r['point']}（{r['component']}）{r['cls_name']}："
            f"最新面积 {_f(r['area']):.1f} cm²、最大宽度 {_f(r['width']):.2f} mm，"
            f"判定为 {GRADE_CN[r['grade']]}"
        )
    return out


# --------------------------------------------------------------------------
# HTML 产出
# --------------------------------------------------------------------------
def _tbl(headers: list, rows: list[list[str]], cls: str = "data") -> str:
    """拼一张数据表。

    `headers` 每项可写成 `(列名, 列宽)`，宽度只写在**第一个 th** 上——`colgroup`
    与 `table-layout:fixed` 在 QTextBrowser 里都不被支持。`headers` 里的列名允许
    含 HTML（本模块自己写的短标签）；`rows` 里的单元格一律按**已是 HTML/已转义**处理，
    调用方用 `_esc()` 转义数据。
    """
    head = []
    for h in headers:
        name, width = h if isinstance(h, tuple) else (h, "")
        style = f' style="width:{width}"' if width else ""
        head.append(f"<th{style}>{name}</th>")
    body = []
    for i, r in enumerate(rows):
        alt = ' class="alt"' if i % 2 else ""
        body.append(f"<tr{alt}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return (f'<table class="{cls}"><thead><tr>{"".join(head)}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _fn(items: list[str]) -> str:
    return ('<div class="fn">' + "".join(f"<div>{t}</div>" for t in items)
            + "</div>") if items else ""


def _paras(text: str) -> str:
    """把自由文本按行切成段落。行首的 `- `/`1. ` 保留原样——编号在正文里是有意义的。"""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        out.append(f"<p>{_esc(s)}</p>" if s else '<p class="gap">&nbsp;</p>')
    return "".join(out)


def build_html(settings: dict | None = None, llm_text: str = "",
               trace: dict | None = None, stamp: str | None = None,
               data: dict | None = None) -> str:
    """生成完整检测报告的 HTML（白底 A4 版式）。

    `data` 是已经取好的数（`_collect()` 的返回值）。要同时出 HTML 与 Markdown 时
    传它，两次产出共用一次取数——取数里含一次复飞审计，重复跑一遍是白花钱，
    而且两侧数字来自不同时刻的快照时，两份报告会对不上。
    """
    d = data or _collect(settings, llm_text, trace, stamp)
    st, g = d["stats"], d["grades"]

    def h(i: int) -> str:
        return f"<h2>{CC[i]}、{_esc(CHAPTER_TITLES[i])}</h2>"

    def h3(n: str, title: str) -> str:
        return f"<h3>{n} {_esc(title)}</h3>"

    body: list[str] = []

    # —— 报头 ——
    # 页眉那一行靠 `position:fixed` 让浏览器**逐页重复**（Chrome 至今不支持 `@page`
    # 的 margin box，纯 HTML 造不出正确页码，只能让页眉承担"这是哪一份报告"的作用）。
    # Qt 把 fixed 当 static 处理，于是它在 QTextBrowser 里退化成文档开头的一行小字——
    # 不重叠、不丢内容，这是可接受的最差形态。
    #
    # 这一行只印机构与编号：它要逐页重复，任何解释性文字都会跟着重复十几遍。
    # 「页码怎么加」写在第八章的说明里，那里只出现一次。
    body.append(f'<div class="pfoot">{_esc(d["org"])}　·　报告编号 {_esc(d["no"])}</div>')
    body.append('<div class="sheet">')
    body.append('<div class="doc-head">')
    body.append(f'<div class="doc-org">{_esc(d["org"])}</div>')
    body.append("<h1>桥梁病害无人机巡检分析报告</h1>")
    body.append(f'<div class="doc-no">报告编号：{_esc(d["no"])}</div>')
    body.append("</div>")

    srcs: list[str] = []
    for c in d["clauses"]:
        if c["source"] not in srcs:
            srcs.append(c["source"])
    btxt = "、".join(f"{b.name}（{b.id}，{b.date}）" for b in cfg.BATCHES
                    if b.id in d["covered"]) or "—"
    body.append('<table class="meta">')
    body.append(
        f'<tr><th>报告编号</th><td>{_esc(d["no"])}</td>'
        f'<th>检测方式</th><td>无人机同视场复拍 + AI 视觉检测与像素级量化</td></tr>')
    body.append(f'<tr><th>检测对象</th><td colspan="3">{_esc(cfg.APP_TITLE)}</td></tr>')
    body.append(f'<tr><th>巡检批次</th><td colspan="3">{_esc(btxt)}</td></tr>')
    body.append(
        f'<tr><th>编制单位</th><td>{_esc(d["org"])}</td>'
        f'<th>编制日期</th><td>{_esc(d["date"])}</td></tr>')
    body.append(
        f'<tr><th>软件版本</th><td>{_esc(cfg.APP_NAME)} {_esc(cfg.APP_VERSION)}</td>'
        f'<th>数据规模</th><td>{st["images"]} 张影像 / {st["detections"]} 条检测记录'
        f' / {st["segments"]} 条量化记录</td></tr>')
    body.append(f'<tr><th>检测依据</th><td colspan="3">'
                f'{_esc("、".join(srcs) or "—")}'
                f'<span class="note">（条文要点为示意性整理，{_esc(_disclaimer())}）</span>'
                f'</td></tr>')
    body.append("</table>")

    # —— 结论摘要 ——
    head_txt, badges = _verdict(d)
    body.append('<div class="verdict">')
    body.append('<div class="verdict-h">检测结论摘要</div>')
    body.append(f"<p>{_md_strong(head_txt)}</p>")
    if badges:
        body.append(f'<div class="bdgs">{badges}</div>')
    worst = _worst(d)
    if worst:
        body.append('<p class="verdict-s">最不利项：</p><ul class="tight">')
        body.append("".join(f"<li>{_esc(w)}</li>" for w in worst))
        body.append("</ul>")
    n_multi = sum(1 for t in d["trends"] if t["n_batch"] >= 2)
    low = sum(q["low"] for q in d["quality"])
    body.append(
        f'<p class="verdict-s">数据完整度：{st["images"]} 张影像覆盖 '
        f'{st["points"]} 个实拍点位（{st["seg_points"]} 个点位有像素级量化记录）；'
        f'{n_multi} 项病害具备跨批次可比数据；'
        f'{low} 张影像重合率低于 {cfg.OVERLAP_OK:.0%} 合格线，已列入复飞建议。</p>')
    body.append("</div>")

    # —— 一、检测概况与依据 ——
    body.append(h(0))
    body.append(h3("1.1", "检测范围与工作量"))
    body.append(
        f'<p>本次分析基于机载相机采集、并经本系统检测与分割量化后的影像数据。'
        f'累计入库影像 {st["images"]} 张，检测记录 {st["detections"]} 条，'
        f'像素级量化记录 {st["segments"]} 条；影像覆盖 {st["points"]} 个实拍点位'
        f'（点位按构件编号，同一编号在不同批次指向同一物理位置），'
        f'其中 {st["seg_points"]} 个点位有量化记录。'
        f'像素级分割只面向裂缝类病害（见第八章说明），其余类别以检测框计数。</p>')
    body.append(h3("1.2", "巡检批次与作业情况"))
    body.append(_tbl(
        [("批次", "8%"), ("名称", "14%"), ("作业日期", "14%"),
         ("影像(张)", "9%"), ("实拍点位", "10%"), ("量化记录", "10%"),
         ("平均重合率", "12%"), ("批次说明", "23%")],
        [[_esc(b["id"]), _esc(b["name"]), _esc(b["date"]), str(b["images"]),
          str(b["points"]), str(b["segments"]),
          f'{_f(b["avg_overlap"]):.1%}' if b["images"] else "—",
          _esc(b["note"])] for b in d["batches"]]))
    body.append(h3("1.3", "检测依据"))
    body.append(_tbl(
        [("序号", "7%"), ("规范", "26%"), ("条文号", "12%"),
         ("条文要点", "37%"), ("对应处置", "18%")],
        [[str(i), _esc(c["source"]), _esc(c["code"]), _esc(c["text"]),
          _esc(c["action"])] for i, c in enumerate(d["clauses"], start=1)]))
    body.append(f'<div class="fn"><div>上表条文要点均为示意性整理，'
                f'{_esc(_disclaimer())}；正式引用前须逐条核对现行规范原文与文号。</div></div>')

    # —— 二、检测方法与 AI 模型 ——
    body.append(h(1))
    body.append(
        '<p>本报告的检测、量化与研判结果全部由 AI 模型输出，不含人工判读记录。'
        '两级模型串联构成主链路：第一级做病害 <strong>检测</strong>（定位并分类），'
        '第二级对裂缝做像素级 <strong>分割量化</strong>（轮廓提取并换算实际尺寸），'
        '再由研判环节按规范条文定级、由报告环节装配本文件。</p>')
    mdet, mseg = _metric_tables()
    body.append(h3("2.1", "病害检测模型"))
    body.append(f'<p>{_md_strong(mdet["desc"])}</p>')
    body.append(_tbl([("指标", "26%"), ("数值", "20%"), ("说明", "54%")], mdet["rows"]))
    body.append(f'<div class="fn"><div>{_esc(mdet["note"])}</div></div>')
    body.append(h3("2.2", "裂缝分割与量化模型"))
    body.append(f'<p>{_md_strong(mseg["desc"])}</p>')
    body.append(_tbl([("指标", "26%"), ("数值", "20%"), ("说明", "54%")], mseg["rows"]))
    body.append(f'<div class="fn"><div>{_esc(mseg["note"])}</div></div>')
    body.append(h3("2.3", "量化换算与可信度判据"))
    body.append(
        f'<p>分割轮廓的最大内切圆直径即裂缝宽度，按航点拍摄距离反算的地面分辨率换算：'
        f'GSD = 像元尺寸 × 拍摄距离 ÷ 焦距（当前标定：等效焦距 {cfg.CAM_FOCAL_MM:g} mm、'
        f'像元 {cfg.CAM_PIXEL_UM:g} μm）。轮廓抖动 1 像素对应实际尺寸 2·GSD，'
        f'工程上要求测量不确定度小于判据限值的 1/3，即 2·GSD ≤ 限值 ÷ 3；'
        f'不满足该条件的量测值只作趋势参考，并在第五章给出复飞重测方案。</p>')

    # —— 三、检测结果统计 ——
    body.append(h(2))
    total_det = sum(c["n"] for c in d["classes"]) or 1
    body.append(h3("3.1", "病害类别分布"))
    body.append(_tbl(
        [("病害类别", "11%"), ("检出数(处)", "8%"), ("占比", "7%"),
         ("涉及点位", "8%"), ("严重度", "7%"), ("量化记录(条)", "9%"),
         ("累计面积(cm²)", "10%"), ("最大宽度(mm)", "10%"),
         ("限值(mm)", "8%"), ("默认处置口径", "22%")],
        [[f'<span class="dot" style="color:{c["color"]}">■</span> {_esc(c["name"])}',
          str(c["n"]), f'{c["n"] / total_det:.1%}', str(c["points"]),
          _esc(c["severity"]), str(c["n_seg"]),
          f'{_f(c["area_cm2"]):.1f}' if c["n_seg"] else "—",
          f'{_f(c["max_width_mm"]):.2f}' if c["n_seg"] else "—",
          _limit_text(c["key"]),
          _esc(c["suggestion"])] for c in d["classes"]]))
    body.append(f'<div class="fn"><div>检出数按 <strong>检测框</strong> 统计（{st["detections"]} 条）；'
                f'量化记录仅裂缝类才有——分割只面向裂缝，其余六类不做像素级量测，'
                f'故其面积、宽度列记「—」，不代表未检出。</div></div>')

    body.append(h3("3.2", "逐点位检测明细"))
    if d["details"]:
        rows_html: list[list[str]] = []
        foots: list[str] = []
        marks = {"加固": "①", "维修": "②", "观察": "③", "待量化": "④"}
        marked: dict[str, list[str]] = {}
        for r in d["details"]:
            mark = ""
            if r["grade"] in ("加固", "维修"):
                mark = f'<sup>{marks[r["grade"]]}</sup>'
                marked.setdefault(r["grade"], []).append(
                    f'{r["point"]}（{r["component"]}）{r["cls_name"]}：{r["why"]}')
            rows_html.append([
                str(r["no"]), _esc(r["point"]), _esc(r["component"]),
                _esc(r["cls_name"]), _esc(r["severity"]), str(r["det_n"]),
                str(r["img_n"]), f'{_f(r["avg_conf"]):.2f}',
                _esc(r["batches"] or "—"),
                f'{_f(r["area"]):.1f}' if r["area"] is not None else "—",
                f'{_f(r["width"]):.2f}' if r["width"] is not None else "—",
                f'{GRADE_CN[r["grade"]]}{mark}'])
        body.append(_tbl(
            [("序号", "5%"), ("点位", "6%"), ("构件", "8%"), ("病害类别", "9%"),
             ("严重度", "6%"), ("检出数", "6%"), ("影像数", "6%"),
             ("平均置信度", "9%"), ("覆盖批次", "10%"), ("最新面积(cm²)", "10%"),
             ("最大宽度(mm)", "10%"), ("处置档位", "15%")], rows_html))
        for k in ("加固", "维修"):
            for txt in marked.get(k, []):
                foots.append(f'{marks[k]} {_esc(txt)}')
        foots.append("③ 观察：病害发展平缓，维持现有巡检周期继续观察。")
        foots.append("④ 观察（待量化）：该类病害未开展像素级量化，按检出记录列入观察清单。")
        body.append(f'<div class="fn">{"".join(f"<div>{t}</div>" for t in foots)}</div>')
        body.append('<div class="fn"><div>口径说明：「检出数」是 AI 检测模型在该点位'
                    '历次影像上报出的边界框总数，「平均置信度」为其类别置信度的'
                    '均值，两者的分母都含所有覆盖批次；「严重度」是病害类别的固有'
                    '属性（裂缝、剥落、露筋、支座滑移为高，伸缩缝错台、渗水泛碱为'
                    '中，蜂窝麻面为低）；「处置档位」是本次实测数据经研判后的判定，'
                    '两者不是同一根轴。面积与宽度取该点位最新批次的'
                    '<strong>最不利单处</strong>值。</div></div>')
    else:
        body.append("<p>当前数据库中没有带点位编号的检测记录。</p>")

    body.append(h3("3.3", "影像采集质量"))
    if d["quality"]:
        body.append(_tbl(
            [("批次", "10%"), ("名称", "15%"), ("作业日期", "15%"),
             ("影像(张)", "11%"), ("实拍点位", "11%"), ("平均重合率", "13%"),
             ("合格", "12%"), ("待重拍", "13%")],
            [[_esc(q["id"]), _esc(q["name"]), _esc(q["date"]), str(q["n"]),
              str(q["points"]), f'{q["avg"]:.1%}', str(q["ok"]), str(q["low"])]
             for q in d["quality"]]))
        body.append(f'<div class="fn"><div>合格线为图像重合率 {cfg.OVERLAP_OK:.0%}'
                    f'（同视场复拍条件），低于该值的影像不满足历次影像可比性要求，'
                    f'已列入第五章复飞建议。</div></div>')
    else:
        body.append("<p>尚无已导入的影像，无法评估采集质量。</p>")

    # —— 四、长周期病害发展趋势 ——
    body.append(h(3))
    body.append(h3("4.1", "跨批次趋势汇总"))
    if d["trends"]:
        body.append(_tbl(
            [("点位", "7%"), ("构件", "10%"), ("病害类别", "11%"),
             ("可比批次", "9%"), ("区间", "11%"), ("初始面积(cm²)", "12%"),
             ("最新面积(cm²)", "12%"), ("累计增幅", "11%"),
             ("最新最大宽度(mm)", "12%"), ("判定", "5%")],
            [[_esc(t["point"]), _esc(t["component"]), _esc(t["cls_name"]),
              str(t["n_batch"]), f'{_esc(t["batch0"])}→{_esc(t["batch1"])}',
              f'{t["area0"]:.1f}', f'{t["area1"]:.1f}',
              f'{t["growth"] * 100:+.1f}%', f'{t["width"]:.2f}',
              _esc(t["grade"])] for t in d["trends"]]))
        body.append('<div class="fn"><div>面积取各批次内的 <strong>最大值</strong>（最不利单处），'
                    '非批内平均；增幅为最新批次相对初始批次的变化率。'
                    '只有具备两个及以上批次量化记录的点位才列入本表。</div></div>')
    else:
        body.append("<p>暂无可用于跨批次对比的量化记录：趋势比对要求同一点位在"
                    "两个及以上批次均有量化数据。</p>")

    body.append(h3("4.2", "病害异常发展点位"))
    if d["anomalies"]:
        body.append(f'<p>以下 {len(d["anomalies"])} 项点位的病害面积累计增幅达到或超过 '
                    f'30% 的异常阈值，需优先复核。</p>')
        body.append(_tbl(
            [("点位", "9%"), ("构件", "13%"), ("病害类别", "15%"),
             ("初始面积(cm²)", "15%"), ("最新面积(cm²)", "15%"),
             ("累计增幅", "13%"), ("判定", "20%")],
            [[_esc(a["point"]), _esc(a["component"]), _esc(a["cls_name"]),
              f'{_f(a["area0"]):.1f}', f'{_f(a["area1"]):.1f}',
              f'{_f(a["growth"]) * 100:+.1f}%', _esc(a["level"])]
             for a in d["anomalies"]]))
    else:
        body.append("<p>各点位量化指标的累计增幅均未超过 30% 的异常发展阈值。</p>")

    # —— 五、复飞建议 ——
    body.append(h(4))
    body.append(
        '<p>本章由采集质量审计自动生成：先判断每一处量测值本身是否可信'
        '（拍摄距离是否足以支撑该量测精度、复拍重合率是否达标、目标是否被裁切或漏检），'
        '再反解出满足精度要求的 <strong>拍摄距离与云台姿态</strong>，'
        '形成可直接下发飞控的复飞航点。</p>')
    if d["refly_err"]:
        body.append(f'<p class="warn">本章数据不可用：{_esc(d["refly_err"])}</p>')
    body.append(h3("5.1", "复飞航点清单"))
    if d["waypoints"]:
        body.append(_tbl(
            [("点位", "8%"), ("构件", "11%"), ("目标病害", "12%"),
             ("触发判据", "10%"), ("拍摄距离(m)", "12%"), ("等效焦距(mm)", "12%"),
             ("预期GSD(mm/px)", "13%"), ("预期像素宽", "11%"),
             ("优先级", "11%")],
            [[_esc(w.point_id), _esc(cfg.point_component(w.point_id)),
              _esc(w.cls_name or "—"), _esc(w.trigger_code),
              f'{w.shoot_distance:.2f}', f'{w.focal_eff_mm:.1f}',
              f'{w.expected_gsd_mm_per_px:.3f}', f'{w.expected_px_width:.0f}',
              _esc(w.priority)] for w in d["waypoints"]]))
        body.append('<div class="fn"><div>同一点位命中多条质量问题时只安排一个航点：'
                    '一趟飞过去本就能同时解决，分两条会给出互相矛盾的拍摄距离。'
                    '航点文件同时在「复飞决策」页导出为 CSV / JSON，可直接导入飞控。</div></div>')
    else:
        body.append("<p>本次采集质量审计未发现问题，无需安排复飞。</p>")
    body.append(h3("5.2", "采集质量问题清单"))
    if d["issues"]:
        body.append(_tbl(
            [("判据", "7%"), ("等级", "7%"), ("点位", "8%"), ("批次", "7%"),
             ("问题", "26%"), ("说明", "45%")],
            [[_esc(i.code), _esc(i.level), _esc(i.point_id), _esc(i.batch),
              _esc(i.title), _esc(i.detail)] for i in d["issues"]]))
    else:
        body.append("<p>无采集质量问题记录。</p>")

    # —— 六、智能研判与养护建议 ——
    body.append(h(5))
    body.append(h3("6.1", "分级处置汇总"))
    body.append(_tbl(
        [("处置档位", "14%"), ("项数", "9%"), ("涉及点位", "30%"),
         ("处置口径", "47%")],
        [[_esc(GRADE_CN[k]), str(len(g[k])),
          _esc("、".join(sorted({r["point"] for r in g[k]})) or "—"),
          _esc(GRADE_ADVICE[k])] for k in GRADE_ORDER]))
    body.append('<div class="fn"><div>本表按 <strong>点位×病害类别</strong> 计一项，'
                '与 3.2 明细表的行一一对应；同一档位的处置口径相同，'
                '具体到点位的原因见 3.2 表下角标。</div></div>')
    body.append(h3("6.2", "研判结论"))
    body.append(_paras(d["llm_text"]))

    # —— 七、智能体执行摘要 ——
    body.append(h(6))
    body.append(_agent_section(d["trace"]))

    # —— 八、结论与说明 ——
    body.append(h(7))
    body.append("<p>1. 数据口径：检出数按检测框计数；像素级量化仅面向裂缝类病害，"
                "其余类别只有检测框与置信度、无量测值。</p>")
    body.append(f'<p>2. 物理量换算：面积与宽度依据航点拍摄距离推算的地面分辨率 GSD，'
                f'当前相机标定为等效焦距 {cfg.CAM_FOCAL_MM:g} mm、'
                f'像元 {cfg.CAM_PIXEL_UM:g} μm；实机标定后应更新该参数，'
                f'历史量化值需同步重算。</p>')
    body.append(f'<p>3. 采集合格线：图像重合率合格线为 {cfg.OVERLAP_OK:.0%}，'
                f'低于该值的影像不满足同视场复拍的记录要求，已在第五章给出复飞方案。</p>')
    body.append(f'<p>4. 规范引用：报告中引用的条文号与要点均为示意性整理，'
                f'{_esc(_disclaimer())}，不构成对规范原文的引用；'
                f'正式养护决策前须由具备资质的单位核对现行规范原文并现场复核。</p>')
    body.append(f'<p>5. 报告来源：本报告由 {_esc(cfg.APP_NAME)} {_esc(cfg.APP_VERSION)} '
                f'自动生成，检测、量化与定级结果均来自软件算法输出，'
                f'未经人工判读修正。</p>')
    body.append("<p>6. 页码：本文件为 HTML 版式，页码在「打印 → 另存为 PDF」后由"
                "阅读器或办公软件插入；打印时每页页眉会重复报告编号与编制单位。</p>")

    # —— 签署栏 ——
    body.append('<table class="sign"><tr>')
    # 三格对三个常量：编制人没有可配的署名（报告由系统生成、"编制"即本软件），
    # 故留空白待手签；校核与审核分别取 cfg.REPORT_CHECKER / REPORT_APPROVER。
    body.append('<td class="sk">编制</td><td class="sv">&nbsp;</td>')
    body.append(f'<td class="sk">校核</td><td class="sv">{_esc(cfg.REPORT_CHECKER)}&nbsp;</td>')
    body.append(f'<td class="sk">审核</td><td class="sv">{_esc(cfg.REPORT_APPROVER)}&nbsp;</td>')
    body.append('<td class="sk">日期</td><td class="sv">&nbsp;</td>')
    body.append("</tr></table>")

    body.append("</div>")       # .sheet
    return _shell(f"桥梁病害无人机巡检分析报告 {d['no']}", "\n".join(body))


def _disclaimer() -> str:
    from .agent import specs

    return specs.DISCLAIMER


def _md_strong(text: str) -> str:
    """把 `**x**` 转成 `<strong>`。这里只处理这一种标记——报告正文里的强调点全是
    本模块自己写的，不需要一个通用的 Markdown 解析器。"""
    out: list[str] = []
    for i, seg in enumerate(_esc(text).split("**")):
        out.append(f"<strong>{seg}</strong>" if i % 2 else seg)
    return "".join(out)


def _metric_tables() -> tuple[dict, dict]:
    """二章的模型指标。取数与检测页 / 分割页同一处（`core/metrics.py`），
    报告与界面不会出现两个数。"""
    from . import metrics as M
    from . import backend as B

    det_active = B.resolve_weights("detect_weights")
    seg_active = B.resolve_weights("segment_weights", seg=True)
    if (det_active and Path(det_active).suffix.lower() == ".pt"
            and seg_active and Path(seg_active).suffix.lower() == ".pth"):
        det_name, seg_name = Path(det_active).name, Path(seg_active).name
        return (
            {"desc": "YOLOv8 裂缝单类目标检测网络。",
             "rows": [["输出", "裂缝区域框与置信度", "定位后交由分割模型做精细量化"],
                      ["适用范围", "裂缝单类", "不作为七类病害检测模型使用"]],
             "note": f"当前加载公开权重 {det_name}；未在本项目数据集复测 AP 指标。"},
            {"desc": "crack-seg U-Net v3 裂缝像素级分割与量化网络。",
             "rows": [["推理方式", "448 px 滑窗 + 重叠融合", "适配高分辨率构件图像"],
                      ["输出", "二值掩膜与几何量化", "长度、平均宽度、最大宽度、面积占比"]],
             "note": f"当前加载公开权重 {seg_name}；公开模型结果不等同于本项目测试指标。"},
        )
    if seg_active and Path(seg_active).suffix.lower() == ".pth":
        name = Path(seg_active).name
        return (
            {"desc": "crack-seg U-Net v3 裂缝区域定位。",
             "rows": [["定位方式", "掩膜连通域外接框", "由裂缝分割结果生成区域框"],
                      ["适用范围", "裂缝单类", "不作为七类病害目标检测模型使用"]],
             "note": f"当前加载公开权重 {name}；未在本项目数据集复测 AP 指标。"},
            {"desc": "crack-seg U-Net v3 裂缝像素级分割与量化网络。",
             "rows": [["推理方式", "448 px 滑窗 + 重叠融合", "适配高分辨率构件图像"],
                      ["输出", "二值掩膜与几何量化", "长度、平均宽度、最大宽度、面积占比"]],
             "note": f"当前加载公开权重 {name}；公开模型结果不等同于本项目测试指标。"},
        )

    det = M.load_metrics()
    rows = []
    for k, label in M.DET_METRIC_LABELS.items():
        v = det.summary.get(k)
        if v is None:
            continue
        rows.append([_esc(label), _esc(M.det_metric_text(k, v)),
                     _esc(_DET_MEANING.get(k, ""))])
    note = det.source
    if det.is_demo:
        note += ("；本次未读到训练产物（runs/**/results.csv），上表为设计指标而非实测值，"
                 "如实标注以免被当作实验结果。")

    seg = M.load_seg_metrics()
    srows = []
    for k, label in M.SEG_METRIC_LABELS.items():
        v = seg.values.get(k)
        if v is None:
            continue
        srows.append([_esc(label), _esc(M.seg_metric_text(k, v)),
                      _esc(_SEG_MEANING.get(k, ""))])
    return (
        {"desc": f"{M.DET_MODEL} 病害检测网络。{_DET_ALGO}",
         "rows": rows, "note": note},
        {"desc": f"{M.SEG_MODEL} 裂缝分割与量化网络。{_SEG_ALGO}",
         "rows": srows, "note": seg.source},
    )


_DET_ALGO = ("模型在 YOLOv8 主干上引入自适应感受野与轻量化注意力分支（记为 YOLOv8-ALTE），"
             "对七类桥梁表观病害输出边界框与类别置信度。")
_SEG_ALGO = ("模型以 U-Net 为骨架、VGG 为编码器并加入空洞卷积（记为 U-Net-VC），"
             "对裂缝做像素级分割，再由轮廓几何反解长度、平均宽度与最大宽度。")
_DET_MEANING = {
    "Precision": "检出框里真正是病害的比例，反映误报水平",
    "Recall": "真实病害被检出的比例，直接决定漏检率",
    "AP50": "交并比阈值 0.50 下的平均精度",
    "AP50-95": "交并比 0.50~0.95 多阈值平均精度，更严格",
    "FLOPs": "单张影像的前向计算量，越小越利于机载实时",
}
_SEG_MEANING = {
    "IoU": "单类别分割交并比",
    "mIoU": "多类别平均交并比",
    "Precision": "分割像素的精确率",
    "mPrecision": "分割像素的平均精确率",
    "Recall": "分割像素的召回率",
}


def _agent_section(trace: dict | None) -> str:
    """七章：智能体执行摘要。

    有 `trace`（走智能体流水线生成）时列出本次真实的工具调用链与耗时；
    没有（桌面 / 网页手工点「生成报告」）时换成静态流水线环节表——
    **第七章的编号恒定不变**，两份报告要能逐章对照，但降级版本里绝不出现耗时数字：
    没有跑过的步骤编不出耗时，编出来就是假数据。
    """
    out: list[str] = []
    try:
        from .agent import events as E
        from .agent import tools as T
    except Exception as exc:                            # noqa: BLE001
        return f'<p class="warn">智能体模块不可用（{_esc(type(exc).__name__)}），本章从略。</p>'

    if trace and trace.get("steps"):
        plan_src = trace.get("planner_name") or "本地规则引擎"
        out.append(
            f'<p>本报告由多智能体流水线生成，规划来源：<strong>{_esc(plan_src)}</strong>；'
            f'任务描述「{_esc(trace.get("task") or "—")}」。'
            f'下表按调用先后列出本次实际执行的工具，耗时取自各步工具自身的计时，'
            f'不含模型思考与措辞时间。</p>')
        rows = []
        for i, s in enumerate(trace["steps"], start=1):
            rows.append([
                str(i), _esc(s.get("cn") or s.get("tool") or ""),
                _esc(s.get("agent_cn") or ""),
                "成功" if s.get("ok") else "失败",
                str(s.get("ms", 0)),
                _esc(s.get("summary") or ""),
            ])
        # 末行是**编制本报告这一步自己**。它不在 `trace["steps"]` 里：编排器是在
        # 调起 make_report **之前**取这份轨迹的，此刻它的 tool_result 还没产生。
        # 少了这一行，表就成了一条断在半路的链——读的人会问"报告是怎么出来的"。
        # 耗时列写「—」而不是 0：这一步确实还没计时，0 ms 是编出来的数。
        _cn, _acn = _report_step()
        rows.append([str(len(rows) + 1), _esc(_cn), _esc(_acn),
                     "进行中（本步）", "—", "即本报告本身"])
        out.append(_tbl(
            [("序号", "5%"), ("环节", "13%"), ("承担智能体", "11%"),
             ("结果", "13%"), ("耗时(ms)", "9%"), ("结论摘要", "49%")], rows))
        total = sum(int(s.get("ms", 0) or 0) for s in trace["steps"])
        dels = trace.get("delegates") or []
        if dels:
            out.append('<p>委派记录：</p><ul class="tight">')
            for dl in dels:
                out.append(f'<li>{_esc(dl.get("agent_cn") or "")}'
                           f'——{_esc(dl.get("reason") or "")}</li>')
            out.append("</ul>")
        out.append(
            f'<div class="fn"><div>工具耗时合计 {total} ms，是上表各步工具自身计时之和'
            f'（不含正在进行的报告编制一步），与工作台「执行轨迹」面板逐行相加一致；'
            f'任务总耗时含模型思考与汇总结论的时间，以工作台显示为准。</div></div>')
    else:
        out.append(
            '<p>本报告由桌面端或网页端手工生成，未经过智能体流水线，'
            '故无工具调用记录与耗时数据。下表列出该流水线各环节由哪位智能体、'
            '用什么工具承担，供与本报告各章对应。<strong>本次未执行，故不列耗时</strong>。</p>')
        rows = []
        for key, (cn, duty, _c) in E.AGENTS.items():
            if key in ("system",):
                continue
            owns = [s.cn for s in T.TOOLS.values() if s.agent == key]
            rows.append([str(len(rows) + 1), _esc(cn), _esc(duty),
                         _esc("、".join(owns) or "—")])
        out.append(_tbl(
            [("序号", "6%"), ("智能体", "16%"), ("职责", "38%"), ("承担工具", "40%")],
            rows))
        out.append('<div class="fn"><div>在执行完整的智能体任务时，本节会换成'
                   '本次真实的工具调用链、每步耗时与结论摘要，以及智能体之间的委派记录。</div></div>')
    return "".join(out)


def _shell(title: str, body: str) -> str:
    return ('<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8">\n'
            f'<title>{_esc(title)}</title>\n<style>\n{_DOC_CSS}\n</style>\n'
            f'</head><body>\n{body}\n</body></html>\n')


_DOC_CSS = """\
/* 检测报告版式。约束见 report.py 模块文档：Qt 的 QTextBrowser 支持的是 CSS 的一个
   很窄的子集，这里没有一条规则用到 flex/grid/var()/伪元素/圆角/阴影——它们都会
   在桌面预览里静默失效或让表格散架。 */
html, body { margin: 0; padding: 0; background: #FFFFFF; color: #000000; }
body { font-family: "SimSun", "宋体", "Songti SC", serif;
       font-size: 10.5pt; line-height: 1.65; }

.sheet { max-width: 780px; margin: 0 auto; padding: 30px 36px 46px; }

.pfoot { position: fixed; top: 8px; left: 0; right: 0; text-align: center;
         font-size: 7.5pt; color: #808080; }

/* —— 报头 —— */
.doc-head { text-align: center; border-bottom: 2.5pt double #000000;
            padding-bottom: 10px; margin-bottom: 14px; }
.doc-org { font-size: 11pt; letter-spacing: .32em; color: #404040; }
.doc-head h1 { font-size: 19pt; font-weight: bold; letter-spacing: .12em;
               margin: 10px 0 6px; }
.doc-no { font-size: 9.5pt; color: #404040; }

table.meta { width: 100%; border-collapse: collapse; margin-bottom: 16px;
             font-size: 9pt; }
table.meta th { width: 14%; background: #EFEFEF; border: 1px solid #808080;
                padding: 4px 6px; text-align: left; font-weight: bold;
                color: #202020; }
table.meta td { border: 1px solid #808080; padding: 4px 6px; }
table.meta .note { color: #606060; font-size: 8pt; }

/* —— 结论摘要框 —— */
.verdict { border: 1.5pt solid #000000; padding: 10px 14px 12px;
           margin-bottom: 18px; background: #FAFAFA; }
.verdict-h { font-size: 11.5pt; font-weight: bold; border-bottom: 1px solid #000000;
             padding-bottom: 4px; margin-bottom: 8px; letter-spacing: .08em; }
.verdict p { margin: 6px 0; }
.verdict-s { font-weight: bold; }
/* 分级徽章。**底色是必须的，边框只是锦上添花**：QTextBrowser 不支持行内元素的
   边框与内边距，只写 border 的话桌面端预览里徽章退化成一段普通文字，"加固 3"
   与"观察 5"看着只是连在一起的正文。底色两边都认，所以四种档位各配一个浅色底，
   浏览器另有描边勾勒出徽章形状。 */
.bdgs { margin: 8px 0 4px; }
.bdg { border: 1px solid #606060; background: #F0F0F0; padding: 1px 8px;
       font-size: 9pt; color: #202020; }
.bdg-ga { border-color: #A00000; background: #FBE9E9; color: #A00000; }
.bdg-we { border-color: #A05800; background: #FBF1E3; color: #A05800; }
.bdg-ob { border-color: #00558C; background: #E9F2F9; color: #00558C; }
.bdg-wq { border-color: #606060; background: #F0F0F0; color: #555555; }

/* —— 标题层级 —— */
h2 { font-size: 13pt; font-weight: bold; margin: 22px 0 8px;
     border-left: 4pt solid #000000; padding-left: 8px;
     page-break-after: avoid; }
h3 { font-size: 11pt; font-weight: bold; margin: 14px 0 6px;
     page-break-after: avoid; }
p { margin: 6px 0; }
p.gap { margin: 2px 0; }
p.warn { color: #A00000; }
ul.tight { margin: 4px 0 4px 20px; padding: 0; }
ul.tight li { margin: 2px 0; }
strong { font-weight: bold; }
sup { font-size: 7pt; }
.dot { font-size: 8pt; }

/* —— 数据表 —— */
table.data { width: 100%; border-collapse: collapse; margin: 8px 0 12px;
             font-size: 8.5pt; }
table.data th { background: #EDEDED; border: 1px solid #909090;
                padding: 4px 5px; text-align: center; font-weight: bold; }
table.data td { border: 1px solid #909090; padding: 3px 5px; text-align: center; }
table.data tr.alt td { background: #F7F7F7; }

/* —— 表下注与脚注 —— */
.fn { font-size: 8pt; color: #404040; margin: -6px 0 12px; line-height: 1.5; }
.fn div { margin: 2px 0; }

/* —— 签署栏 —— */
table.sign { width: 100%; border-collapse: collapse; margin-top: 30px;
             font-size: 10pt; }
table.sign .sk { width: 8%; padding: 16px 4px 4px 0; text-align: right;
                 color: #303030; }
table.sign .sv { border-bottom: 1px solid #000000; padding: 16px 10px 4px 4px; }

/* 打印：去掉屏幕上的版心留白，交给 @page 的页边距；其余样式沿用默认规则，
   避免"屏幕上好好的、一打印就散架"。 */
@page { size: A4; margin: 20mm 18mm; }
@media print {
  .sheet { max-width: none; margin: 0; padding: 0; }
  html, body { background: #FFFFFF; }
  table.data tr.alt td { background: #FFFFFF; }
  .verdict { background: #FFFFFF; }
}
"""


# --------------------------------------------------------------------------
# Markdown 产出（简版）
# --------------------------------------------------------------------------
def build_markdown(settings: dict | None = None, llm_text: str = "",
                   trace: dict | None = None, stamp: str | None = None,
                   data: dict | None = None) -> str:
    """生成巡检报告的 Markdown 简版。

    同八章、同数据，但没有徽章、脚注角标与签署栏的可视化——那些是 HTML 版式才
    表达得出来的东西。文件头注明它是简版，避免有人拿它当正式交付件。
    `data` 的含义同 `build_html`。
    """
    d = data or _collect(settings, llm_text, trace, stamp)
    st, g = d["stats"], d["grades"]
    md: list[str] = []

    md.append("# 桥梁病害无人机巡检分析报告（简版）")
    md.append("")
    md.append("> 本文件为纯文本简版，完整版式（报头、结论摘要、分级徽章、签署栏）"
              "见同名 HTML 文件。")
    md.append("")
    md.append(f"**报告编号：** {d['no']}　　**编制单位：** {d['org']}　　"
              f"**编制日期：** {d['date']}")
    md.append("")
    md.append(f"**巡检对象：** {cfg.APP_TITLE}")
    md.append(f"**覆盖批次：** {'、'.join(d['covered']) or '—'}")
    md.append(f"**软件版本：** {cfg.APP_NAME} {cfg.APP_VERSION}")
    md.append("")

    head_txt, badges = _verdict(d)
    md.append("## 检测结论摘要")
    md.append("")
    md.append(head_txt.replace("**", ""))
    md.append("")
    if badges:
        md.append("分级：" + "　".join(
            f"{GRADE_CN[k]} {len(g[k])}" for k in GRADE_ORDER if g[k]))
        md.append("")
    for w in _worst(d):
        md.append(f"- {w}")
    md.append("")

    md.append("## " + CHAPTERS[0])
    md.append("")
    md.append("### 1.1 检测范围与工作量")
    md.append("")
    md.append(f"累计入库影像 {st['images']} 张、检测记录 {st['detections']} 条、"
              f"量化记录 {st['segments']} 条，覆盖 {st['points']} 个实拍点位"
              f"（其中 {st['seg_points']} 个点位有像素级量化记录）。"
              f"像素级分割只面向裂缝类病害，其余类别以检测框计数。")
    md.append("")
    md.append("### 1.2 巡检批次与作业情况")
    md.append("")
    md.append("| 批次 | 名称 | 作业日期 | 影像 | 实拍点位 | 量化记录 | 平均重合率 | 说明 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for b in d["batches"]:
        ov = f"{_f(b['avg_overlap']):.1%}" if b["images"] else "—"
        md.append(f"| {b['id']} | {b['name']} | {b['date']} | {b['images']} | "
                  f"{b['points']} | {b['segments']} | {ov} | {b['note']} |")
    md.append("")
    md.append("### 1.3 检测依据")
    md.append("")
    md.append("| 规范 | 条文号 | 条文要点 | 对应处置 |")
    md.append("| --- | --- | --- | --- |")
    for c in d["clauses"]:
        md.append(f"| {c['source']} | {c['code']} | {c['text']} | {c['action']} |")
    md.append("")
    md.append(f"（条文要点为示意性整理，{_disclaimer()}。）")
    md.append("")

    mdet, mseg = _metric_tables()
    md.append("## " + CHAPTERS[1])
    md.append("")
    md.append("### 2.1 病害检测模型")
    md.append("")
    md.append(mdet["desc"])
    md.append("")
    md.append("| 指标 | 数值 | 说明 |")
    md.append("| --- | --- | --- |")
    for r in mdet["rows"]:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    md.append("")
    md.append(f"{mdet['note']}")
    md.append("")
    md.append("### 2.2 裂缝分割与量化模型")
    md.append("")
    md.append(mseg["desc"])
    md.append("")
    md.append("| 指标 | 数值 | 说明 |")
    md.append("| --- | --- | --- |")
    for r in mseg["rows"]:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    md.append("")

    md.append("## " + CHAPTERS[2])
    md.append("")
    total_det = sum(c["n"] for c in d["classes"]) or 1
    md.append("### 3.1 病害类别分布")
    md.append("")
    md.append("| 病害类别 | 检出数 | 占比 | 点位 | 严重度 | 量化记录 | 累计面积(cm²) | 最大宽度(mm) | 默认处置口径 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in d["classes"]:
        md.append(
            f"| {c['name']} | {c['n']} | {c['n'] / total_det:.1%} | {c['points']} | "
            f"{c['severity']} | {c['n_seg']} | "
            f"{_f(c['area_cm2']):.1f} | {_f(c['max_width_mm']):.2f} | {c['suggestion']} |")
    md.append("")
    md.append("（检出数按检测框统计；量化记录仅裂缝类才有，面积与宽度列的「0.0」表示"
              "该类未做像素级量测，不代表未检出。）")
    md.append("")
    md.append("### 3.2 逐点位检测明细")
    md.append("")
    if d["details"]:
        md.append("| 序号 | 点位 | 构件 | 病害类别 | 严重度 | 检出数 | 影像数 | "
                  "平均置信度 | 覆盖批次 | 最新面积(cm²) | 最大宽度(mm) | 处置档位 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in d["details"]:
            area = f"{_f(r['area']):.1f}" if r["area"] is not None else "—"
            wid = f"{_f(r['width']):.2f}" if r["width"] is not None else "—"
            md.append(f"| {r['no']} | {r['point']} | {r['component']} | {r['cls_name']} | "
                      f"{r['severity']} | {r['det_n']} | {r['img_n']} | "
                      f"{_f(r['avg_conf']):.2f} | "
                      f"{r['batches'] or '—'} | {area} | {wid} | {GRADE_CN[r['grade']]} |")
        md.append("")
        for r in d["details"]:
            if r["grade"] in ("加固", "维修"):
                md.append(f"- {r['point']}（{r['component']}）{r['cls_name']}：{r['why']}")
        md.append("")
    else:
        md.append("_当前数据库中没有带点位编号的检测记录。_")
        md.append("")
    md.append("### 3.3 影像采集质量")
    md.append("")
    if d["quality"]:
        md.append("| 批次 | 名称 | 影像 | 实拍点位 | 平均重合率 | 合格 | 待重拍 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for q in d["quality"]:
            md.append(f"| {q['id']} | {q['name']} | {q['n']} | {q['points']} | "
                      f"{q['avg']:.1%} | {q['ok']} | {q['low']} |")
    else:
        md.append("_尚无已导入的影像。_")
    md.append("")

    md.append("## " + CHAPTERS[3])
    md.append("")
    if d["trends"]:
        md.append("| 点位 | 构件 | 病害类别 | 可比批次 | 区间 | 初始面积(cm²) | 最新面积(cm²) | 累计增幅 | 最新最大宽度(mm) | 判定 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for t in d["trends"]:
            md.append(f"| {t['point']} | {t['component']} | {t['cls_name']} | {t['n_batch']} | "
                      f"{t['batch0']}→{t['batch1']} | {t['area0']:.1f} | {t['area1']:.1f} | "
                      f"{t['growth'] * 100:+.1f}% | {t['width']:.2f} | {t['grade']} |")
    else:
        md.append("_暂无可用于跨批次对比的量化记录。_")
    md.append("")
    md.append("### 4.1 病害异常发展点位")
    md.append("")
    if d["anomalies"]:
        md.append("| 点位 | 构件 | 病害类别 | 初始面积(cm²) | 最新面积(cm²) | 累计增幅 | 判定 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for a in d["anomalies"]:
            md.append(f"| {a['point']} | {a['component']} | {a['cls_name']} | "
                      f"{_f(a['area0']):.1f} | {_f(a['area1']):.1f} | "
                      f"{_f(a['growth']) * 100:+.1f}% | {a['level']} |")
    else:
        md.append("各点位量化指标变化均未超过异常阈值（增幅 30%）。")
    md.append("")

    md.append("## " + CHAPTERS[4])
    md.append("")
    if d["refly_err"]:
        md.append(f"**本章数据不可用：** {d['refly_err']}")
        md.append("")
    if d["waypoints"]:
        md.append("### 5.1 复飞航点清单")
        md.append("")
        md.append("| 点位 | 构件 | 目标病害 | 触发判据 | 拍摄距离(m) | 等效焦距(mm) | 预期GSD(mm/px) | 预期像素宽 | 优先级 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for w in d["waypoints"]:
            md.append(f"| {w.point_id} | {cfg.point_component(w.point_id)} | "
                      f"{w.cls_name or '—'} | {w.trigger_code} | {w.shoot_distance:.2f} | "
                      f"{w.focal_eff_mm:.1f} | {w.expected_gsd_mm_per_px:.3f} | "
                      f"{w.expected_px_width:.0f} | {w.priority} |")
    else:
        md.append("本次采集质量审计未发现问题，无需安排复飞。")
    md.append("")
    if d["issues"]:
        md.append("### 5.2 采集质量问题清单")
        md.append("")
        md.append("| 判据 | 等级 | 点位 | 批次 | 问题 | 说明 |")
        md.append("| --- | --- | --- | --- | --- | --- |")
        for i in d["issues"]:
            md.append(f"| {i.code} | {i.level} | {i.point_id} | {i.batch} | "
                      f"{i.title} | {i.detail} |")
        md.append("")

    md.append("## " + CHAPTERS[5])
    md.append("")
    md.append("| 处置档位 | 项数 | 涉及点位 | 处置口径 |")
    md.append("| --- | --- | --- | --- |")
    for k in GRADE_ORDER:
        md.append(f"| {GRADE_CN[k]} | {len(g[k])} | "
                  f"{'、'.join(sorted({r['point'] for r in g[k]})) or '—'} | "
                  f"{GRADE_ADVICE[k]} |")
    md.append("")
    md.append(d["llm_text"])
    md.append("")

    md.append("## " + CHAPTERS[6])
    md.append("")
    md.append(_agent_section_md(d["trace"]))
    md.append("")

    md.append("## " + CHAPTERS[7])
    md.append("")
    md.append("1. 检出数按检测框计数；像素级量化仅面向裂缝类病害，其余类别"
              "只有检测框与置信度、无量测值。")
    md.append(f"2. 面积与宽度依据航点拍摄距离推算的 GSD，当前标定等效焦距 "
              f"{cfg.CAM_FOCAL_MM:g} mm、像元 {cfg.CAM_PIXEL_UM:g} μm；"
              f"实机标定后应更新该参数，历史量化值需同步重算。")
    md.append(f"3. 图像重合率合格线为 {cfg.OVERLAP_OK:.0%}，低于该值的影像已列入"
              f"复飞建议。")
    md.append(f"4. 报告引用的条文号与要点均为示意性整理，{_disclaimer()}，"
              f"不构成对规范原文的引用。")
    md.append(f"5. 本报告由 {cfg.APP_NAME} {cfg.APP_VERSION} 自动生成，"
              f"检测、量化与定级结果均来自软件算法输出。")
    md.append("")
    md.append(f"编制：{cfg.REPORT_CHECKER or '＿＿＿＿'}　　"
              f"校核：＿＿＿＿　　"
              f"审核：{cfg.REPORT_APPROVER or '＿＿＿＿'}　　"
              f"日期：＿＿＿＿")
    md.append("")
    return "\n".join(md)


def _report_step() -> tuple[str, str]:
    """「编制本报告」这一步的（环节名, 承担智能体中文名），从注册表取。

    不写成字面量：工具或智能体改名时，HTML 版与 Markdown 版要一起跟上，
    而这两份报告是给人逐章对照着看的。
    """
    cn, agent_cn = "报告编制", "报告生成体"
    try:
        from .agent import events as E
        from .agent import tools as T
        if "make_report" in T.TOOLS:
            cn = T.TOOLS["make_report"].cn
        if "report" in E.AGENTS:
            agent_cn = E.AGENTS["report"][0]
    except Exception:                                   # noqa: BLE001
        pass
    return cn, agent_cn


def _agent_section_md(trace: dict | None) -> str:
    """第七章的 Markdown 版。与 HTML 版同数据，只是表格语法的差别。"""
    if not (trace and trace.get("steps")):
        return ("本报告由桌面端或网页端手工生成，未经过智能体流水线，故无工具调用记录"
                "与耗时数据。该流水线由巡检定线、视觉检测、分割量化、趋势研判、"
                "复飞决策、报告编制等专业智能体分工承担，总控智能体负责意图解析与委派；"
                "**本次未执行，故不列耗时**。完整版 HTML 中本章会列出每位智能体"
                "与本报告各章的对应关系。")
    out = [f"规划来源：**{trace.get('planner_name') or '本地规则引擎'}**；"
           f"任务描述「{trace.get('task') or '—'}」。", ""]
    out.append("| 序号 | 环节 | 承担智能体 | 结果 | 耗时(ms) | 结论摘要 |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for i, s in enumerate(trace["steps"], start=1):
        out.append(f"| {i} | {s.get('cn') or s.get('tool') or ''} | "
                   f"{s.get('agent_cn') or ''} | {'成功' if s.get('ok') else '失败'} | "
                   f"{s.get('ms', 0)} | {s.get('summary') or ''} |")
    # 编制本报告这一步自己：编排器取轨迹时它的 tool_result 还没产生，所以不在
    # `steps` 里。补一行、耗时写「—」，与 HTML 版的表逐行一致。
    _cn, _acn = _report_step()
    out.append(f"| {len(trace['steps']) + 1} | {_cn} | {_acn} | 进行中（本步） "
               f"| — | 即本报告本身 |")
    total = sum(int(s.get("ms", 0) or 0) for s in trace["steps"])
    out.append("")
    out.append(f"工具耗时合计 {total} ms，为上表各步工具自身计时之和（不含正在进行的"
               f"报告编制一步），不含模型思考与措辞时间；任务总耗时以工作台为准。")
    dels = trace.get("delegates") or []
    if dels:
        out.append("")
        out.append("委派记录：" + "；".join(
            f"{dl.get('agent_cn') or ''}——{dl.get('reason') or ''}" for dl in dels))
    return "\n".join(out)


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------
def save_markdown(text: str, settings: dict | None = None,
                  name: str | None = None) -> Path:
    out = _out_dir(settings)
    p = out / (name or f"桥梁巡检报告_{datetime.now():%Y%m%d_%H%M%S}.md")
    p.write_text(text, encoding="utf-8")
    return p


def save_html(html_text: str, settings: dict | None = None,
              name: str | None = None) -> Path:
    """把 **HTML 文本**落盘。

    早先这个函数收的是 Markdown、内部再调 `_md_to_html` 转一道；HTML 改为直接生成
    之后，参数语义就是"已经是 HTML 了"。调用方别再传 Markdown 进来——那会把整篇
    Markdown 原文当 HTML 写出去。
    """
    out = _out_dir(settings)
    p = out / (name or f"桥梁巡检报告_{datetime.now():%Y%m%d_%H%M%S}.html")
    p.write_text(html_text, encoding="utf-8")
    return p


def build_both(settings: dict | None = None, llm_text: str = "",
               trace: dict | None = None,
               stamp: str | None = None) -> tuple[str, str, str]:
    """只出文本、不落盘，返回 `(markdown, html, 报告编号)`。

    桌面端「长周期监测」页的报告预览要走这条路：它要先在界面上看，再由用户决定
    保存到哪里，所以在 `build_and_save` 之前就得拿到两份文本。取数只做一次，
    两份文本不会出自两个时刻的快照。
    """
    data = _collect(settings, llm_text, trace, stamp)
    return build_markdown(data=data), build_html(data=data), data["no"]


def build_and_save(settings: dict | None = None, llm_text: str = "",
                   trace: dict | None = None,
                   stamp: str | None = None) -> tuple[Path, Path, str]:
    """一次性生成 HTML（完整版）与 Markdown（简版），
    返回 `(markdown 路径, html 路径, 报告编号)`。

    两份文件共用**同一个戳**，因此报告编号一致、编制日期一致，能互相对上。

    编号一并返回而不是让调用方去 `build_markdown` 里找：
    再跑一次建报告要重复走一遍取数（含一次复飞审计），而报告编号与
    正文里那个本就出自同一个戳，再取一次反而可能取到另一个。
    """
    out = _out_dir(settings)
    stamp = stamp or _stamp(out)
    data = _collect(settings, llm_text, trace, stamp)
    html = build_html(data=data)
    md = build_markdown(data=data)
    p_md = save_markdown(md, settings, f"桥梁巡检报告_{stamp}.md")
    p_html = save_html(html, settings, f"桥梁巡检报告_{stamp}.html")
    return p_md, p_html, data["no"]


# --------------------------------------------------------------------------
# 数据导出（与报告版式无关，保持原样）
# --------------------------------------------------------------------------
def export_csv(rows: list[dict], path: str | Path) -> Path:
    """导出任意记录列表为 CSV（UTF-8-SIG，Excel 直接打开不乱码）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8-sig")
        return p
    fields = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return p


def export_xlsx(rows: list[dict], path: str | Path, sheet: str = "Sheet1") -> Path:
    """导出 Excel（openpyxl 缺失时自动退回 CSV）。"""
    p = Path(path)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        return export_csv(rows, p.with_suffix(".csv"))

    wb = Workbook()
    ws = wb.active
    ws.title = sheet[:31]
    if rows:
        fields = list(rows[0].keys())
        ws.append(fields)
        for c in range(1, len(fields) + 1):
            cell = ws.cell(row=1, column=c)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1E3A5F")
            cell.alignment = Alignment(horizontal="center")
        for r in rows:
            ws.append([r.get(f) for f in fields])
        for i, f in enumerate(fields, start=1):
            width = max(10, min(26, max(len(str(f)) + 4,
                                        *(len(str(r.get(f, ""))) + 2 for r in rows))))
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
    p.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(p))
    return p


def segments_to_rows(point_ids: list[str] | None = None) -> list[dict]:
    """把量化记录转成导出用的扁平结构。"""
    sql = ("SELECT * FROM segments WHERE area_cm2>0")
    params: list = []
    if point_ids:
        sql += f" AND point_id IN ({','.join('?' * len(point_ids))})"
        params += point_ids
    sql += " ORDER BY point_id, batch, captured_at"
    rows = db.query(sql, params)
    return [{
        "点位编号": r["point_id"],
        "构件": cfg.point_component(r["point_id"]),
        "病害类别": r["cls_name"],
        "检测批次": cfg.BATCH_BY_ID[r["batch"]].name if r["batch"] in cfg.BATCH_BY_ID
                    else r["batch"],
        "采集时间": r["captured_at"],
        "像素面积(px)": r["px_area"],
        "病害面积(cm²)": r["area_cm2"],
        "最大宽度(mm)": r["max_width_mm"],
        "周长(px)": r["perimeter_px"],
        "置信度": r["conf"],
    } for r in rows]
