"""驾驶舱装饰层：网格底纹 + 无人机／桥梁线稿。

这一整个模块**不承载任何信息**，删掉它界面功能一点不少。加它的理由只有一条：
这是一套无人机巡检系统，屏幕上该有仪表盘的样子。三条自律与网页版
`web/static/cockpit.css` 的「驾驶舱精密感」一节完全一致：

1. **不抢内容。** 网格线 alpha 只有 9~16（约 4~6%），线稿 alpha 10~120 且只出现在
   卡片之间的空隙里。装饰一旦和正文抢注意力就该删掉。
2. **不依赖字体。** 全部用 QPainter 画线与椭圆，不用任何几何字符充当图形——
   字体里没有对应字形时会被回退成方框，而且换台机器宽度还不一样。
3. **不动布局。** 只在已有控件的 `paintEvent` 里画，不新增占位的控件、不改边距。

为什么不用图片：项目承诺零外部依赖，且位图在 4K 屏上会糊。这里的场景几何与网页版
`web/static/js/hud.js` **逐点对应**（同一套 520×250 场景坐标），两边看起来才是一套东西。
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from .. import theme

# 场景坐标系。与 web/static/js/hud.js 的 viewBox 一致，改一边就要同步另一边。
SCENE_W, SCENE_H = 520.0, 250.0
STRIP_W, STRIP_H = 152.0, 74.0


def _pen(p: QPainter, color: QColor, width: float = 1.3,
         dash: list[float] | None = None) -> None:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if dash:
        pen.setDashPattern(dash)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _lines(p: QPainter, color: QColor, width: float, *segs: tuple[float, ...],
           dash: list[float] | None = None) -> None:
    """按 (x1,y1,x2,y2) 四元组批量画线。省得每个点都写一次 drawLine。"""
    _pen(p, color, width, dash)
    for x1, y1, x2, y2 in segs:
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


# --------------------------------------------------------------------------
# 底纹
# --------------------------------------------------------------------------
def paint_grid(p: QPainter, rect: QRectF, fine: QColor, coarse: QColor,
               step: float = 28.0, every: int = 5) -> None:
    """坐标纸底纹：每格 28 px，每 5 格一根粗线。"""
    if rect.width() < step or rect.height() < step:
        return
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    for i in range(int(rect.width() // step) + 2):
        x = round(rect.left() + i * step) + 0.5     # +0.5 才是 1px 实线，否则糊成 2px
        if x > rect.right():
            break
        _pen(p, coarse if i % every == 0 else fine, 1.0)
        p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
    for j in range(int(rect.height() // step) + 2):
        y = round(rect.top() + j * step) + 0.5
        if y > rect.bottom():
            break
        _pen(p, coarse if j % every == 0 else fine, 1.0)
        p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
    p.restore()


def paint_ruler(p: QPainter, rect: QRectF, tick: QColor, major: QColor,
                step: float = 16.0, every: int = 8) -> None:
    """刻度尺：贴着 rect 下沿向上出齿，每 8 齿一根长齿。"""
    if rect.width() < step * 2:
        return
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    base = rect.bottom() - 0.5
    for i in range(int(rect.width() // step) + 1):
        x = round(rect.left() + i * step) + 0.5
        if x > rect.right():
            break
        big = (i % every == 0)
        _pen(p, major if big else tick, 1.0)
        p.drawLine(QPointF(x, base), QPointF(x, base - (9.0 if big else 5.0)))
    p.restore()


# --------------------------------------------------------------------------
# 线稿
# --------------------------------------------------------------------------
def paint_scene(p: QPainter, rect: QRectF, color: QColor) -> None:
    """整幅「无人机沿桥巡检」线稿，锚在 rect 的右下角。

    布局在控件之外、尺寸不足时直接不画——半截桥比没有桥更难看。
    """
    if rect.width() < 200 or rect.height() < 110:
        return
    s = min(rect.width() / SCENE_W, rect.height() / SCENE_H)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.translate(rect.right() - SCENE_W * s, rect.bottom() - SCENE_H * s)
    p.scale(s, s)

    # —— 桥：桥面两道横线 + 桁架斜撑 + 桥墩伸进水里 ——
    _lines(p, color, 2.2, (18, 148, 502, 148), (18, 196, 502, 196))
    _lines(p, color, 1.3,
           (52, 148, 52, 196), (158, 148, 158, 196), (264, 148, 264, 196),
           (370, 148, 370, 196), (476, 148, 476, 196),
           (52, 196, 106, 148), (158, 196, 212, 148), (264, 196, 318, 148),
           (370, 196, 424, 148), (476, 196, 502, 165),
           (52, 196, 52, 236), (158, 196, 158, 242), (264, 196, 264, 248),
           (370, 196, 370, 242), (476, 196, 476, 236))

    water = QColor(color)
    water.setAlpha(int(color.alpha() * 0.6))
    _lines(p, water, 1.3,
           (10, 236, 58, 236), (118, 242, 158, 242), (224, 248, 264, 248),
           (330, 242, 370, 242), (436, 236, 484, 236), dash=[6, 7])

    # —— 航迹：与 SVG 的 C…S… 同一条三次贝塞尔 ——
    _pen(p, color, 1.3, dash=[5, 7])
    path = _path((42, 62), (112, 20), (190, 96), (268, 58))
    path.cubicTo(QPointF(346, 20), QPointF(400, 78), QPointF(448, 40))
    p.drawPath(path)

    _pen(p, color, 1.3)
    p.setBrush(color)
    for cx, cy in ((42, 62), (170, 62), (268, 58), (386, 66)):
        p.drawEllipse(QPointF(cx, cy), 3.2, 3.2)
    p.setBrush(Qt.BrushStyle.NoBrush)

    # —— 云台视场：从无人机拉出的锥形，落在桥面上 ——
    fov = QColor(color)
    fov.setAlpha(int(color.alpha() * 0.55))
    _lines(p, fov, 1.3, (448, 52, 344, 148), (448, 52, 500, 140), dash=[3, 5])

    # —— 四旋翼 ——
    p.save()
    p.translate(448, 40)
    _pen(p, color, 1.3)
    p.drawRoundedRect(QRectF(-11, -4.5, 22, 9), 3, 3)
    _lines(p, color, 1.3,
           (-11, -2.5, -21, -8), (11, -2.5, 21, -8),
           (-11, 2.5, -21, 8), (11, 2.5, 21, 8))
    for sx in (-24, 24):
        for sy in (-11, 11):
            p.drawEllipse(QPointF(sx, sy), 8, 2.2)
    _lines(p, color, 1.3, (0, 5.5, 0, 16.5))
    p.drawPolygon(QPolygonF([QPointF(-4, 22), QPointF(4, 22), QPointF(0, 28)]))
    p.restore()

    # —— 四角准星：告诉人这是一张"图"，不是随手画的 ——
    _lines(p, color, 1.6,
           (18, 14, 18, 8), (18, 8, 26, 8),
           (502, 14, 502, 8), (502, 8, 494, 8),
           (18, 244, 18, 250), (18, 250, 26, 250),
           (502, 244, 502, 250), (502, 250, 494, 250))
    p.restore()


def paint_strip(p: QPainter, rect: QRectF, color: QColor) -> None:
    """侧边栏底部的窄条：桥 + 一架悬停的无人机。"""
    if rect.width() < 60 or rect.height() < 30:
        return
    s = min(rect.width() / STRIP_W, rect.height() / STRIP_H)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.translate(rect.center().x() - STRIP_W * s / 2,
                rect.bottom() - STRIP_H * s)
    p.scale(s, s)

    _lines(p, color, 1.8, (6, 46, 146, 46), (6, 62, 146, 62))
    _lines(p, color, 1.2,
           (24, 46, 24, 62), (60, 46, 60, 62), (96, 46, 96, 62), (132, 46, 132, 62),
           (24, 62, 42, 46), (60, 62, 78, 46), (96, 62, 114, 46), (132, 62, 146, 62))

    faint = QColor(color)
    faint.setAlpha(int(color.alpha() * 0.6))
    _lines(p, faint, 1.2,
           (10, 68, 16, 68), (34, 68, 40, 68), (70, 68, 76, 68),
           (106, 68, 112, 68), (142, 68, 146, 68), dash=[4, 5])

    _pen(p, color, 1.2, dash=[4, 5])
    path = _path((22, 18), (50, 6), (76, 24), (104, 12))
    p.drawPath(path)

    _pen(p, color, 1.2)
    p.setBrush(color)
    p.drawEllipse(QPointF(22, 18), 2.2, 2.2)
    p.drawEllipse(QPointF(104, 12), 2.2, 2.2)
    p.setBrush(Qt.BrushStyle.NoBrush)

    p.save()
    p.translate(120, 10)
    p.drawRoundedRect(QRectF(-7, -3, 14, 6), 2, 2)
    _lines(p, color, 1.2,
           (-7, -1.6, -13, -5), (7, -1.6, 13, -5),
           (-7, 1.6, -13, 5), (7, 1.6, 13, 5))
    for sx in (-15, 15):
        for sy in (-6.6, 6.6):
            p.drawEllipse(QPointF(sx, sy), 5, 1.5)
    p.restore()
    p.restore()


def _path(p0: tuple[float, float], c1: tuple[float, float],
          c2: tuple[float, float], p1: tuple[float, float]) -> QPainterPath:
    """从 (起点, 控制点1, 控制点2, 终点) 造一条三次贝塞尔。"""
    path = QPainterPath(QPointF(*p0))
    path.cubicTo(QPointF(*c1), QPointF(*c2), QPointF(*p1))
    return path


# --------------------------------------------------------------------------
# 控件
# --------------------------------------------------------------------------
class HudRoot(QWidget):
    """主内容区底板：底纹 + 右下角线稿。

    全局 QSS 里 `QWidget { background: transparent }`，各页面的卡片之间的空隙因此
    是透的——这一层画的东西正好从缝里透出来，就是仪表盘底纹该有的样子。
    底色由本控件自己填：`paintEvent` 被重写且不调 `super()`，样式表那层不会画。
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("HudRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def paintEvent(self, _ev) -> None:              # noqa: N802 —— Qt 接口名
        p = QPainter(self)
        rect = QRectF(self.rect())
        p.fillRect(self.rect(), QColor(theme.C["app"]))
        # alpha 是量出来的，不是猜的：底色 #0B1220 的蓝通道是 32，下面这几个值把
        # 网格线压到 50 / 67，线稿到 68——刚好"看得出是一张坐标纸"，又远低于任何
        # 一档正文的对比度（正文 dim 色已经在 148 上下）。
        # 试过 9/16/10：在 1080p 上量出来只差 4~7 个色阶，肉眼和截图里都等于没画。
        paint_grid(p, rect, QColor(34, 211, 238, 20), QColor(34, 211, 238, 40))
        art = QColor(theme.C["accent"])
        art.setAlpha(42)
        paint_scene(p, rect.adjusted(24, 24, -24, -24), art)


class StripArt(QWidget):
    """侧边栏底部的窄条线稿。纯装饰，不接收鼠标事件。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, _ev) -> None:              # noqa: N802 —— Qt 接口名
        p = QPainter(self)
        c = QColor(theme.C["accent"])
        c.setAlpha(115)
        paint_strip(p, QRectF(self.rect()), c)
