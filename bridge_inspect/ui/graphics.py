"""矢量插图渲染器（桌面后端）。

几何存在 `core/figures.json` 里，网页端由 `web/static/js/figures.js` 渲染成 SVG，
这里用 QPainter 画同一份数据。**一份几何、两套后端**：扫描策略图与闭环框图在
演示视频、应用方案 PDF、桌面软件里必须是同一张图，改一次三处同时生效。

坐标系与 SVG 一致：viewBox 统一 100 × 58，y 轴向下；`core/figures.json` 的
`_primitives` 字段是这套图元的正式说明，改图元要同步改那里和 JS 那一侧。
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor, QFont, QFontMetricsF, QPainter, QPen, QPixmap, QPolygonF,
)
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .. import theme

C = theme.C

SPEC_PATH = Path(__file__).resolve().parent.parent / "core" / "figures.json"

VB_W, VB_H = 100.0, 58.0          # viewBox 尺寸，与 figures.json 的 _coords 一致
ARROW_SIZE = 2.1

# 族清单从 theme 取，不在这里另存一份：插图与界面必须是同一套中文字体。
_FAMILIES = theme.FONT_FAMILIES
_MONO_FAMILIES = theme.MONO_FAMILIES


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_pack() -> dict:
    """读取 figures.json。读不到就返回空包，界面显示占位而不是崩掉。"""
    try:
        data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tokens": {}, "glyphs": {}, "figures": {}}
    glyphs = {k: v for k, v in (data.get("glyphs") or {}).items() if isinstance(v, list)}
    return {
        "tokens": data.get("tokens") or {},
        "glyphs": glyphs,
        "figures": data.get("figures") or {},
    }


def figure_keys(pack: dict | None = None) -> list[str]:
    return list((pack or load_pack())["figures"].keys())


def token_color(name: str | None, tok: dict) -> QColor | None:
    """令牌名 → QColor。`none` 与未知名都返回 None（表示不描边/不填充）。

    取色优先走 figures.json 自带的 tokens，而不是 theme.C：那份 JSON 是两套
    渲染器共同的源头，从这里取才能保证网页端和桌面端是同一个颜色。
    """
    if not name or name == "none":
        return None
    val = tok.get(name) or C.get(name)
    if not val or val == "none":
        return None
    col = QColor(val)
    return col if col.isValid() else None


# --------------------------------------------------------------------------
# 图元
# --------------------------------------------------------------------------
def _arrow_head(painter: QPainter, x: float, y: float, ang: float,
                color: QColor, size: float = ARROW_SIZE) -> None:
    """在 (x, y) 处按 ang 方向画一个实心箭头，与 JS 的 arrowHead 同参数。"""
    a1, a2 = ang + math.pi * 0.82, ang - math.pi * 0.82
    tri = QPolygonF([
        QPointF(x, y),
        QPointF(x + size * math.cos(a1), y + size * math.sin(a1)),
        QPointF(x + size * math.cos(a2), y + size * math.sin(a2)),
    ])
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawPolygon(tri)


def _pen(color: QColor | None, width: float, dash: str | None = None,
         cap_round: bool = True) -> QPen:
    if color is None:
        return QPen(Qt.PenStyle.NoPen)
    pen = QPen(color)
    pen.setWidthF(max(width, 0.01))
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if cap_round:
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    if dash:
        try:
            pattern = [float(v) for v in str(dash).split()]
        except ValueError:
            pattern = []
        if pattern:
            pen.setDashPattern(pattern)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    return pen


def _draw_text(painter: QPainter, p: dict, tok: dict) -> None:
    """文字单独在**设备坐标**里画。

    画布为了让 viewBox 铺满控件，整体缩放了 s 倍。文字若沿用同一变换，字号会被
    连带放大 s 倍；而 QFont 只收整数像素尺寸，反算回 viewBox 又会把 3.2 这类字号
    抹成 3（宽度 380 时相当于 12 px 变 11.4 px）。所以这里把变换还原、直接按
    round(size × s) 出字——舍入到整像素本来就是光栅化必做的一步，没有额外损失。
    """
    color = token_color(p.get("color"), tok) or QColor(C["text"])
    font = QFont()
    font.setFamilies(_MONO_FAMILIES if p.get("mono") else _FAMILIES)
    font.setWeight(_weight(p.get("w")))

    origin = painter.transform().map(QPointF(p.get("x", 0), p.get("y", 0)))
    scale = painter.transform().m11() or 1.0
    font.setPixelSize(max(1, round((p.get("size") or 3) * scale)))

    painter.save()
    painter.resetTransform()
    painter.setFont(font)
    painter.setPen(color)
    text = str(p.get("v", ""))
    metrics = QFontMetricsF(font)
    anchor = p.get("anchor") or "start"
    x = origin.x()
    if anchor == "middle":
        x -= metrics.horizontalAdvance(text) / 2.0
    elif anchor == "end":
        x -= metrics.horizontalAdvance(text)
    # Qt 的 drawText 以基线定位，SVG 的 y 也是基线，两者口径一致
    painter.drawText(QPointF(x, origin.y()), text)
    painter.restore()


def _weight(w) -> QFont.Weight:
    if not w:
        return QFont.Weight.Normal
    try:
        n = int(w)
    except (TypeError, ValueError):
        return QFont.Weight.DemiBold if str(w).lower() == "bold" else QFont.Weight.Normal
    if n >= 700:
        return QFont.Weight.Bold
    if n >= 600:
        return QFont.Weight.DemiBold
    if n >= 500:
        return QFont.Weight.Medium
    return QFont.Weight.Normal


def _draw_dim(painter: QPainter, p: dict, tok: dict) -> None:
    """带端线与文字底板的尺寸标注，与 JS 的 dim 同一套画法。"""
    color = token_color(p.get("color"), tok) or QColor(C["text"])
    x1, y1 = p.get("x1", 0), p.get("y1", 0)
    x2, y2 = p.get("x2", 0), p.get("y2", 0)
    ang = math.atan2(y2 - y1, x2 - x1)
    px, py = math.cos(ang + math.pi / 2) * 1.2, math.sin(ang + math.pi / 2) * 1.2

    pen = _pen(color, 0.7)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    painter.drawLine(QPointF(x1 - px, y1 - py), QPointF(x1 + px, y1 + py))
    painter.drawLine(QPointF(x2 - px, y2 - py), QPointF(x2 + px, y2 + py))

    # 底板宽度按实际字宽量。JS 那边用"字符数 × 1.55"估，对中文标签偏窄；
    # 底板只是压在线上的一块底衬，两端各自量准不会造成图形本身不一致。
    label = str(p.get("label", ""))
    font = QFont()
    font.setFamilies(_FAMILIES)
    scale = painter.transform().m11() or 1.0
    font.setPixelSize(max(1, round(2.8 * scale)))
    lw = QFontMetricsF(font).horizontalAdvance(label) / max(scale, 1e-6) + 1.6
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    plate = token_color("panel", tok) or QColor(C["card"])
    painter.setPen(Qt.PenStyle.NoPen)
    plate.setAlphaF(0.92)
    painter.setBrush(plate)
    painter.drawRoundedRect(QRectF(mx - lw / 2, my - 2.3, lw, 4.4), 1, 1)

    _draw_text(painter, {"x": mx, "y": my + 1.1, "v": label, "size": 2.8,
                         "anchor": "middle", "color": p.get("color")}, tok)


def draw_prim(painter: QPainter, p: dict, tok: dict, glyphs: dict) -> None:
    """画一个图元。图元表见 figures.json 的 `_primitives`。"""
    kind = p.get("t")
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(Qt.PenStyle.NoPen)

    if kind == "line":
        color = token_color(p.get("k"), tok)
        w = p.get("w", 1)
        if p.get("a"):
            # 先画线再叠箭头：箭头是实心三角，反过来画会被线头压住尖端
            painter.setPen(_pen(color, w, p.get("d")))
            painter.drawLine(QPointF(p["x1"], p["y1"]), QPointF(p["x2"], p["y2"]))
            if color is not None:
                _arrow_head(painter, p["x2"], p["y2"],
                            math.atan2(p["y2"] - p["y1"], p["x2"] - p["x1"]), color)
        else:
            painter.setPen(_pen(color, w, p.get("d")))
            painter.drawLine(QPointF(p["x1"], p["y1"]), QPointF(p["x2"], p["y2"]))

    elif kind == "rect":
        fill = token_color(p.get("f"), tok)
        pen = _pen(token_color(p.get("k"), tok), p.get("w", 1))
        painter.setPen(pen)
        painter.setBrush(fill if fill is not None else Qt.BrushStyle.NoBrush)
        rect = QRectF(p["x"], p["y"], p["w"], p["h"])
        rx = p.get("rx")
        if rx:
            painter.drawRoundedRect(rect, rx, rx)
        else:
            painter.drawRect(rect)

    elif kind == "circle":
        fill = token_color(p.get("f"), tok)
        painter.setPen(_pen(token_color(p.get("k"), tok), p.get("dw", 1), p.get("d")))
        painter.setBrush(fill if fill is not None else Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(p["cx"], p["cy"]), p["r"], p["r"])

    elif kind == "poly":
        pts = QPolygonF([QPointF(x, y) for x, y in p.get("p", [])])
        if not pts.isEmpty():
            fill = token_color(p.get("f"), tok)
            painter.setPen(_pen(token_color(p.get("k"), tok), p.get("w", 1)))
            painter.setBrush(fill if fill is not None else Qt.BrushStyle.NoBrush)
            if p.get("close"):
                painter.drawPolygon(pts)
            else:
                painter.drawPolyline(pts)

    elif kind == "arc":
        # 角度口径与 JS 相同：0° 指向 3 点钟、顺时针为正（y 轴向下）。
        # Qt 的正角是逆时针，所以要取负；见 _qt_arc_span。
        color = token_color(p.get("k"), tok)
        cx, cy, r = p["cx"], p["cy"], p["r"]
        a0, a1 = p.get("a0", 0), p.get("a1", 0)
        painter.setPen(_pen(color, p.get("w", 1)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r),
                        int(-a0 * 16), int(-(a1 - a0) * 16))
        if p.get("a") and color is not None:
            a1r = math.radians(a1)
            tip = (cx + r * math.cos(a1r), cy + r * math.sin(a1r))
            _arrow_head(painter, tip[0], tip[1],
                        a1r + (math.pi / 2 if a1 > a0 else -math.pi / 2), color)

    elif kind == "text":
        _draw_text(painter, p, tok)

    elif kind == "icon":
        glyph = glyphs.get(p.get("name"))
        if not glyph:
            return
        k = (p.get("size", 2)) / 10.0
        painter.save()
        painter.translate(p["x"], p["y"])
        painter.scale(k, k)
        for q in glyph:
            draw_prim(painter, {**q, "k": q.get("k") or "accent"}, tok, glyphs)
        painter.restore()

    elif kind == "dim":
        _draw_dim(painter, p, tok)


def draw_figure(painter: QPainter, spec: dict, pack: dict | None = None,
                size: tuple[float, float] | None = None) -> None:
    """把一张插图铺满 size 给定的区域，等比、不裁切。

    单独拆出来是因为除了控件重绘，导出 PDF/PNG 时也要用同一套绘制——
    报告里的插图和软件里看到的必须是同一张。
    """
    pack = pack or load_pack()
    w, h = size if size else (float(painter.device().width()),
                              float(painter.device().height()))
    scale = min(w / VB_W, h / VB_H)
    painter.save()
    painter.translate((w - VB_W * scale) / 2.0, (h - VB_H * scale) / 2.0)
    painter.scale(scale, scale)
    for item in spec.get("items", []):
        draw_prim(painter, item, pack["tokens"], pack["glyphs"])
    painter.restore()


def render_pixmap(spec: dict, width: int = 760, pack: dict | None = None,
                  ratio: float = 2.0) -> QPixmap:
    """离屏渲染成位图，导出报告插图时用。ratio 是超采样倍数。

    **不要设 `QT_QPA_PLATFORM=offscreen` 来跑这段。** 该平台下 Qt 查不到任何
    系统字族（实测可用字族 0 个，而默认 windows 平台有 345 个），汉字会全部
    退化成等宽方框：宽度恰好等于 pixelSize，图形照画不误、也不报任何错，只有
    字是空的。要写无窗口的渲染自检，用默认平台——不 show() 窗口一样能往
    QPixmap 上画。
    """
    pack = pack or load_pack()
    height = int(round(width * VB_H / VB_W))
    pm = QPixmap(int(width * ratio), int(height * ratio))
    pm.setDevicePixelRatio(ratio)
    pm.fill(QColor(C["card"]))
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    draw_figure(painter, spec, pack, (width, height))
    painter.end()
    return pm


# --------------------------------------------------------------------------
# 控件
# --------------------------------------------------------------------------
class FigureCanvas(QWidget):
    """按 100:58 等比绘制一张插图，宽度撑满、高度自动推出。"""

    def __init__(self, spec: dict | None = None, pack: dict | None = None,
                 width: int = 380, parent: QWidget | None = None):
        super().__init__(parent)
        self.pack = pack or load_pack()
        self.spec = spec or {}
        self._width = width
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self.height_for_width(width))

    @staticmethod
    def height_for_width(width: int) -> int:
        return max(60, int(round(width * VB_H / VB_W)))

    def set_spec(self, spec: dict) -> None:
        self.spec = spec or {}
        self.update()

    def set_pack(self, pack: dict) -> None:
        self.pack = pack
        self.update()

    def resizeEvent(self, ev):                      # noqa: N802  Qt 命名
        super().resizeEvent(ev)
        want = self.height_for_width(self.width())
        if want != self.minimumHeight():
            self.setMinimumHeight(want)

    def paintEvent(self, ev):                       # noqa: N802  Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        if not self.spec.get("items"):
            painter.setPen(QColor(C["text_muted"]))
            font = QFont()
            font.setFamilies(_FAMILIES)
            font.setPixelSize(12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "尚无插图数据")
            return
        draw_figure(painter, self.spec, self.pack,
                    (self.width(), self.height()))


class FigureCard(QFrame):
    """插图卡：图形 + 标题 + 副题。对应网页端的 figureCard()。"""

    def __init__(self, spec: dict, pack: dict | None = None, width: int = 380,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.pack = pack or load_pack()
        box = QVBoxLayout(self)
        box.setContentsMargins(10, 10, 10, 10)
        box.setSpacing(6)

        self.canvas = FigureCanvas(spec, self.pack, width)
        box.addWidget(self.canvas)

        title = QLabel(str(spec.get("title", "")))
        title.setObjectName("CardTitle")
        box.addWidget(title)
        sub = QLabel(str(spec.get("sub", "")))
        sub.setObjectName("Hint")
        sub.setWordWrap(True)
        box.addWidget(sub)

    def resizeEvent(self, ev):                      # noqa: N802  Qt 命名
        super().resizeEvent(ev)
        # 画布跟着卡片宽度走，但不要让高度把卡片撑得过长
        want = self.canvas.height_for_width(max(self.width() - 20, 120))
        self.canvas.setMinimumHeight(want)
