"""影像入库：从文件名解析点位与批次，读图、取航点元数据、写 images 表。

桌面版智能体工作台的「导入影像」与网页版的上传端点都走这里。放在 `core/` 而不是某个页面里，
有两个原因：

1. **两边必须同规则**。同一张 `P1_A01_WP-A01-P1.jpg` 在桌面版和网页版导入，落到库里的
   点位、批次、航点元数据必须是同一组值——否则同一张图在两边的趋势曲线里会站在
   不同的点位上。
2. **网页版后端不能碰 PyQt**。`core/` 是唯一被两个前端共用的层，入库逻辑放这儿，
   `web/api.py` 才能直接调；放进桌面页面会把 PyQt6 拖进服务端。

本模块不依赖 PyQt，也不做任何界面动作——它只返回一份结构化结果，由调用方决定怎么显示。
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .. import config as cfg
from .. import db
from . import demo_data, imgutil

# 点位编号：一个字母 + 两位数字（A01 / C21）。与 `config.ALL_POINTS` 同一套命名。
POINT_RE = re.compile(r"([A-Fa-f]\d{2})")
# 批次：P1 / P2 / P3。`\b` 在 `_` 两侧不成立（下划线算单词字符），所以
# `P1_A01_WP-A01-P1` 里命中的是行首那个 P1 而不是结尾的——这与既有行为一致，
# 不要改成宽松匹配：文件名尾部也带批次号时，宽松匹配会取到错的那个。
BATCH_RE = re.compile(r"\b(P[1-9])\b", re.I)

# 单张状态
ADDED = "added"
EXISTS = "exists"
UNSUPPORTED = "unsupported"
UNDECODABLE = "undecodable"


@dataclass
class IngestRow:
    """一张影像的入库结果。"""
    path: str
    filename: str
    status: str
    point_id: str = ""
    batch: str = ""
    width: int = 0
    height: int = 0
    size_kb: float = 0.0
    overlap: float = 0.0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path, "filename": self.filename, "status": self.status,
            "point_id": self.point_id, "batch": self.batch,
            "width": self.width, "height": self.height,
            "size_kb": self.size_kb, "overlap": self.overlap,
            "message": self.message,
        }


@dataclass
class IngestResult:
    rows: list[IngestRow] = field(default_factory=list)

    @property
    def added(self) -> list[IngestRow]:
        return [r for r in self.rows if r.status == ADDED]

    @property
    def skipped(self) -> list[IngestRow]:
        return [r for r in self.rows if r.status == EXISTS]

    @property
    def failed(self) -> list[IngestRow]:
        return [r for r in self.rows
                if r.status in (UNSUPPORTED, UNDECODABLE)]

    @property
    def first_added_id(self) -> int | None:
        """本次第一张新入库影像的 id，供调用方直接载入到当前工作区。"""
        for r in self.added:
            row = db.query_one("SELECT id FROM images WHERE path=?", (r.path,))
            if row:
                return int(row["id"])
        return None

    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"新增 {len(self.added)} 张")
        if self.skipped:
            parts.append(f"已在库 {len(self.skipped)} 张")
        if self.failed:
            parts.append(f"未能导入 {len(self.failed)} 张")
        return "，".join(parts) or "没有可导入的影像"

    def to_dict(self) -> dict:
        return {"rows": [r.to_dict() for r in self.rows],
                "added": len(self.added), "skipped": len(self.skipped),
                "failed": len(self.failed), "summary": self.summary()}


def parse_name(stem: str) -> tuple[str, str]:
    """从文件名（不含扩展名）解析出（点位编号, 批次号）。解析不到就返回空串。"""
    m = POINT_RE.search(stem)
    point_id = m.group(1).upper() if m else ""
    mb = BATCH_RE.search(stem)
    batch = mb.group(1).upper() if mb else ""
    return point_id, batch


def ingest(paths, default_batch: str | None = None) -> IngestResult:
    """把一批影像读入库。已经入库过的按原路径跳过，不重复插行。

    传进来的路径可以是任意格式；不支持的扩展名与无法解码的文件都会被记成一行结果，
    而不是抛异常中断整批——一次导入几百张，不该因为其中一张坏了就全军覆没。
    """
    default_batch = default_batch or cfg.BATCHES[0].id
    out = IngestResult()
    idx = 0
    for raw in paths or []:
        p = Path(raw)
        if not imgutil.is_supported(str(p)):
            out.rows.append(IngestRow(str(p), p.name, UNSUPPORTED,
                                      message="不是支持的影像格式"))
            continue
        idx += 1
        bgr = imgutil.imread(p)
        if bgr is None:
            out.rows.append(IngestRow(str(p), p.name, UNDECODABLE,
                                      message="无法解码"))
            continue

        point_id, batch = parse_name(p.stem)
        if not point_id:
            point_id = f"X{idx:02d}"
        if batch not in cfg.BATCH_BY_ID:
            batch = default_batch

        meta = demo_data.waypoint_meta(point_id, batch)
        h, w = bgr.shape[:2]
        size_kb = round(p.stat().st_size / 1024, 1)
        row = IngestRow(str(p), p.name, ADDED, point_id=point_id, batch=batch,
                        width=w, height=h, size_kb=size_kb,
                        overlap=float(meta.get("overlap") or 0.0))

        if db.image_by_path(str(p)):
            row.status = EXISTS
            row.message = "已在库中，跳过重复入库"
            out.rows.append(row)
            continue

        db.add_image(task_id=None, path=str(p), filename=p.name,
                     width=w, height=h, size_kb=size_kb,
                     source="SD卡导入", **meta)
        out.rows.append(row)
    return out


def ingest_one(path: str) -> tuple[int | None, str, str]:
    """把**一张还没入库**的影像收进系统。

    返回 `(image_id, 入库后的路径, 说明文案)`；收不进来时 image_id 为 None，说明文案
    里是原因。

    给「目标检测 / 分割量化」页的「打开影像」用：用户从桌面或图片文件夹随手挑一张照片
    时，那张图原先不在 images 表里，检测结果就没处可落——他会在分割量化收到一句
    "该影像还没有检测记录"，而矛盾的是一分钟前明明检测过。调用方拿到这里的结果后要
    改用返回的**新路径**：文件已经复制进 IMG_DIR 了。

    复制而不是只登记原路径：网页版的产物与标注图端点都要求路径落在 DATA_DIR 内
    （`web/api.py` 的 `_safe_file`），只登记外部路径的话，同一个点位在桌面端能用、
    在网页端 403，两副面孔。顺带也保住了"影像都在 data/images/"这条不变量，拷走
    data/ 就是一套完整数据。

    同名文件已存在时换名保存（`_1` / `_2`），**绝不覆盖**——覆盖会悄悄改掉库里另一张
    影像的内容，而它的路径一个字都没变，是那种事后无从追查的坏。
    """
    src = Path(path)
    if not imgutil.is_supported(str(src)):
        return None, str(src), "不是支持的影像格式"

    dest = cfg.IMG_DIR / src.name
    try:
        same = dest.exists() and src.resolve() == dest.resolve()
    except OSError:                       # 网络盘/权限问题，按"不同"处理走复制
        same = False
    if not same:
        cfg.IMG_DIR.mkdir(parents=True, exist_ok=True)
        n = 1
        while dest.exists():
            dest = cfg.IMG_DIR / f"{src.stem}_{n}{src.suffix}"
            n += 1
        try:
            shutil.copy2(src, dest)
        except OSError as e:
            return None, str(src), f"复制到 data/images/ 失败：{e}"

    res = ingest([str(dest)])
    row = res.rows[0] if res.rows else None
    if row is None or row.status in (UNSUPPORTED, UNDECODABLE):
        return None, str(dest), (row.message if row else "入库失败")
    got = db.image_by_path(str(dest))
    if got is None:                       # 理论上不会：ingest 成功必然有行
        return None, str(dest), "入库后查不到记录"
    # 点位与批次是解析出来的，直接告诉用户给他挂到哪儿了——`X01` 这种自动编号不写出来，
    # 他会在点位列表里看到个陌生的编号而不知道是自己刚打开的那张。
    note = (f"已入库：点位 {got['point_id'] or '—'} · 批次 {got['batch'] or '—'}"
            if row.status == ADDED else f"该影像已在库中（{got['point_id']}）")
    return int(got["id"]), str(dest), note
