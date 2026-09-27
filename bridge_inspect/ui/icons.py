"""语义图标渲染器（桌面后端）。

几何存在 `core/icons.json` 里，网页端由 `web/gen_icons.py` 生成
`web/static/js/icons.js`（SVG），这里用 QPainter 画同一份数据。**一份几何、两套
后端**——导航栏在桌面版与网页版上是同一套线稿图标，不会一边是几何字符一边是图形。

为什么不用字符图标：`◆ ▦ ▣ ◉ ◈ ◔ ↻ ▤ ▥ ⚙` 依赖字体里恰好有这些字形。缺字形时
Qt 会静默回退成方框，换一台机器字宽还不一样，导航栏就对不齐了。线稿图标不依赖
字体，能任意缩放，也画得出"无人机""云台""RTK"这些几何字符根本表达不了的语义。

坐标：24 × 24 的 viewBox，y 轴向下，与 `icons.json` 的 `_coords` 一致。线宽也是
这套坐标下的值（1.6），随缩放一起放大缩小，和 SVG 的 `stroke-width` 行为相同。
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

SPEC_PATH = Path(__file__).resolve().parent.parent / "core" / "icons.json"

VB = 24.0                      # viewBox 边长，与 icons.json 的 _coords 一致
DEFAULT_SW = 1.6               # 默认线宽（viewBox 单位）

# SVG 路径语法的子集：M L H V C A Z，大小写分别为绝对与相对。
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_CMD = re.compile(r"[MmLlHhVvCcAaZz]")
_ARGC = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "A": 7, "Z": 0}


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load() -> dict:
    """读取 icons.json。读不到就返回空表，界面照常起来、只是不画图标。"""
    try:
        data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    icons = data.get("icons") or {}
    if not isinstance(icons, dict):
        return {}
    icons["__aliases__"] = data.get("aliases") or {}
    return icons


def names() -> list[str]:
    return [k for k in load() if k != "__aliases__"]


def _resolve(name: str) -> str:
    """把页面 key 换算成图标名。没有专属图标的页面走 icons.json 的 aliases。"""
    return (load().get("__aliases__") or {}).get(name, name)


def _shapes(name: str) -> list[dict]:
    return (load().get(_resolve(name)) or {}).get("shapes") or []


def about(name: str) -> str:
    """图标画的是什么。用来当工具提示，也让两边的注释出自同一句话。"""
    return (load().get(_resolve(name)) or {}).get("about") or ""


# --------------------------------------------------------------------------
# 路径解析
# --------------------------------------------------------------------------
def _tokens(d: str) -> list:
    """把 `d` 拆成 [命令, 数字…, 命令, 数字…] 的扁平序列。

    SVG 的数字允许紧挨着写（`a2 2 0 0 1 2-2` 里 `2-2` 是 **两个** 数 2 和 -2），
    所以不能按空格切——必须按数字文法逐个吃。
    """
    out, i, n = [], 0, len(d)
    while i < n:
        c = d[i]
        if c in " ,\t\r\n":
            i += 1
            continue
        m = _CMD.match(d, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        m = _NUM.match(d, i)
        if not m:
            # 不认识的字符直接跳过而不是抛错：抛错会走到 paintEvent 里，
            # 未捕获异常在 PyQt 里是 qFatal，整个进程退出且没有回溯。
            i += 1
            continue
        out.append(float(m.group(0)))
        i = m.end()
    return out


def _arc_to(path: QPainterPath, x1: float, y1: float, rx: float, ry: float,
            phi_deg: float, large: bool, sweep: bool, x2: float,
            y2: float) -> None:
    """按 SVG 规范的「端点参数 → 圆心参数」换算后画一段圆弧。

    圆（rx == ry 且不旋转）走精确的 `arcTo`；椭圆或带旋转的弧在本项目里一个都没有，
    真遇上了退化成一串折线——宁可差一点，也不要为了不存在的需求写一大段换算。
    """
    if x1 == x2 and y1 == y2:
        return
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        path.lineTo(x2, y2)
        return

    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cp * dx + sp * dy
    y1p = -sp * dx + cp * dy

    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1.0:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s

    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    k = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large == sweep:
        k = -k
    cxp = k * rx * y1p / ry
    cyp = -k * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0

    def ang(ux: float, uy: float, vx: float, vy: float) -> float:
        dot = ux * vx + uy * vy
        nrm = math.hypot(ux, uy) * math.hypot(vx, vy)
        if nrm == 0:
            return 0.0
        a = math.acos(max(-1.0, min(1.0, dot / nrm)))
        return -a if ux * vy - uy * vx < 0 else a

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    t0 = ang(1.0, 0.0, ux, uy)
    dt = ang(ux, uy, vx, vy)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi

    if abs(rx - ry) > 1e-9 or abs(phi_deg) > 1e-9:
        # 兜底：按折线逼近。弧长越长分段越多，24 段对本项目的尺寸足够。
        for i in range(1, 25):
            t = t0 + dt * i / 24.0
            ex = cx + rx * math.cos(t) * cp - ry * math.sin(t) * sp
            ey = cy + rx * math.cos(t) * sp + ry * math.sin(t) * cp
            path.lineTo(ex, ey)
        return

    # Qt 的正角是逆时针，SVG 的是从 +x 转向 +y（在 y 向下的屏幕上是顺时针），
    # 两者方向相反，所以起始角与扫过角都要取负。
    rect = QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry)
    path.arcTo(rect, -math.degrees(t0), -math.degrees(dt))


def _build(d: str) -> QPainterPath:
    """把一段 `d` 变成 QPainterPath。"""
    path = QPainterPath()
    toks = _tokens(d)
    i, n = 0, len(toks)
    cmd = None
    x = y = sx = sy = 0.0          # 当前点与子路径起点
    while i < n:
        if isinstance(toks[i], str):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                path.closeSubpath()
                x, y = sx, sy
                continue
        if cmd is None:
            break
        up = cmd.upper()
        rel = cmd.islower()
        need = _ARGC.get(up)
        if need is None or i + need > n:
            break
        a = [float(v) for v in toks[i:i + need]]
        i += need

        if up == "M":
            x, y = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            path.moveTo(x, y)
            sx, sy = x, y
            cmd = "l" if rel else "L"      # 后续隐式坐标对是 lineto
        elif up == "L":
            x, y = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            path.lineTo(x, y)
        elif up == "H":
            x = x + a[0] if rel else a[0]
            path.lineTo(x, y)
        elif up == "V":
            y = y + a[0] if rel else a[0]
            path.lineTo(x, y)
        elif up == "C":
            p1 = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            p2 = (x + a[2], y + a[3]) if rel else (a[2], a[3])
            p3 = (x + a[4], y + a[5]) if rel else (a[4], a[5])
            path.cubicTo(QPointF(*p1), QPointF(*p2), QPointF(*p3))
            x, y = p3
        elif up == "A":
            ex, ey = (x + a[5], y + a[6]) if rel else (a[5], a[6])
            _arc_to(path, x, y, a[0], a[1], a[2], a[3] != 0, a[4] != 0, ex, ey)
            x, y = ex, ey
    return path


@lru_cache(maxsize=64)
def path_of(name: str) -> QPainterPath:
    """一个图标里所有 path 形状的并集（在 24 网格坐标系里）。

    只用于测量与自检——真正绘制走 `paint()`，那里还要处理圆、矩形、虚线与填充。
    """
    out = QPainterPath()
    for s in _shapes(name):
        if s.get("t") == "path":
            out.addPath(_build(s.get("d") or ""))
        elif s.get("t") == "circle":
            out.addEllipse(QPointF(s["cx"], s["cy"]), s["r"], s["r"])
        elif s.get("t") == "rect":
            r = QRectF(s["x"], s["y"], s["w"], s["h"])
            if s.get("rx"):
                out.addRoundedRect(r, s["rx"], s["rx"])
            else:
                out.addRect(r)
    return out


# --------------------------------------------------------------------------
# 绘制
# --------------------------------------------------------------------------
def paint(p: QPainter, name: str, rect: QRectF, color: QColor,
          sw: float = DEFAULT_SW, opacity: float = 1.0) -> None:
    """把一个图标等比铺在 rect 里。

    等比而不是拉伸：图标是方的，铺进非方形区域必须按短边缩放并居中，
    否则"齿轮"会被压成椭圆。
    """
    shapes = _shapes(name)
    if not shapes or rect.width() <= 1 or rect.height() <= 1:
        return
    s = min(rect.width(), rect.height()) / VB
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.translate(rect.center().x() - VB * s / 2.0, rect.center().y() - VB * s / 2.0)
    p.scale(s, s)

    for sh in shapes:
        p.save()
        p.setOpacity(opacity * float(sh.get("o", 1.0)))
        stroke = None if sh.get("k") == "none" else color
        pen = QPen(stroke if stroke is not None else Qt.GlobalColor.transparent,
                   sw)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if sh.get("dash"):
            # Qt 的虚线间隔以**线宽为单位**，SVG 的 stroke-dasharray 是用户单位。
            # 不除掉线宽，"3 2.4" 会按 3×1.6 个用户单位去画，虚线拉长一半——
            # 与网页版一比就是两圈节奏不同的点线，而且看不出是哪里错了。
            pen.setDashPattern([float(v) / sw for v in sh["dash"].split()])
        p.setPen(pen if stroke is not None else Qt.PenStyle.NoPen)
        p.setBrush(color if sh.get("f") else Qt.BrushStyle.NoBrush)

        t = sh.get("t")
        if t == "path":
            p.drawPath(_build(sh.get("d") or ""))
        elif t == "circle":
            p.drawEllipse(QPointF(sh["cx"], sh["cy"]), sh["r"], sh["r"])
        elif t == "rect":
            r = QRectF(sh["x"], sh["y"], sh["w"], sh["h"])
            if sh.get("rx"):
                p.drawRoundedRect(r, sh["rx"], sh["rx"])
            else:
                p.drawRect(r)
        p.restore()
    p.restore()


def pixmap(name: str, size: int, color: QColor, sw: float = DEFAULT_SW,
           ratio: float = 2.0) -> QPixmap:
    """离屏渲染成位图。ratio 是超采样倍数——小图标在 1× 下边缘会发毛。"""
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    paint(p, name, QRectF(0, 0, size, size), color, sw)
    p.end()
    return pm


def qicon(name: str, size: int, color: QColor, sw: float = DEFAULT_SW) -> QIcon:
    return QIcon(pixmap(name, size, color, sw))
