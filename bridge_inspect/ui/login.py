"""登录 / 注册窗口。

账号真实入库（users 表，口令用 PBKDF2 加盐哈希存储），登录成功后把用户信息交给
主窗口，侧边栏底部的用户卡片显示的即是该账号。

左栏是一张无人机沿航线巡检铁路桥梁的插图 + 四条短标签；插图与网页版登录页
**是同一份文件**（`core/login_hero.svg`），见 `HeroArt` 的注释。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QRadialGradient
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QStyle, QStyleOption, QStylePainter,
    QVBoxLayout, QWidget,
)

from .. import config as cfg
from .. import db, theme
from . import hud, icons
from .widgets import IconBadge, Toast

C = theme.C

# 窗口尺寸。440 + 540 = 980 是硬对应：左栏由 HeroPanel 自己 setFixedWidth，
# 改窗口尺寸就必须一起改那个数，否则右栏会宽窄不均而不会报错。
WIN_W, WIN_H = 980, 600
HERO_W = 440
# 投影仪 / 虚拟机 / 高缩放屏上放不下 600（含标题栏要 632px 屏高）时退回旧尺寸，
# 插图会等比缩小、两处 addStretch 保证布局不塌。
WIN_SMALL = (880, 520)

# 四条短标签，与网页版 `web/static/login.html` 的 `.hero-points` **逐条对应**
# （顺序、文案都一样，改一边要连另一边一起改）。
# 图标取自 core/icons.json：网页版这里原本是一列青色圆点，四个圆点谁也说不清
# 哪点是哪点；crack / segment / trend / reflight 正好是这四句话的语义。
POINTS = [
    ("crack", "七类病害识别"),
    ("segment", "像素级尺寸换算"),
    ("trend", "跨批次趋势比对"),
    ("reflight", "采集质量审计与自主复飞"),
]


class HeroArt(QWidget):
    """登录页左栏的航线示意图。

    图形来自 `cfg.HERO_SVG`——**与网页版内联的那张是同一个文件**。网页端由
    `web/gen_hero.py` 注入 `login.html`，这里由 QSvgRenderer 直接读盘。

    为什么是 SVG + QtSvg，而不是像 `graphics.py` 那样加进 `figures.json`：
    这张画的体积感全部来自 linearGradient / radialGradient（桥面、桥墩、扫描锥、
    无人机辉光），而 figures.json 的图元集里没有渐变，改过去整张图会塌成平面线框，
    并且与网页版不再是同一张画——那正是"一份几何"要防的漂移。QtSvg 随 PyQt6 一起
    装，不新增依赖；`icons.py` / `graphics.py` 之所以手写解析，是因为它们要从 JSON
    里读**逐图元可覆写**的数据格式，一张静态插图没有这个需求。
    """

    def __init__(self, width: int = 376, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._svg: QSvgRenderer | None = None
        self._vb = QSize(560, 300)               # 与 SVG 的 viewBox 同值
        self._load(cfg.HERO_SVG)
        self.setFixedHeight(self.height_for_width(width))

    def _load(self, path) -> None:
        """读一次盘。

        **不挂模块级 lru_cache**：`icons.load()` / `graphics.load_pack()` 缓存的是纯
        数据 dict，而 QSvgRenderer 是 QObject，模块级缓存会活过 QApplication 的析构，
        退出时可能踩到已销毁的 Qt 运行时。挂在实例上、以 self 为父，生命周期就跟着
        窗口走；登录窗口每次登录只建一个，开销可以忽略。

        任何一步失败都只是让 `self._svg` 保持 None，由 `paintEvent` 走降级——
        插图读不到不该拦住人登录。
        """
        f = Path(path)
        if not f.exists():            # 先判存在：QSvgRenderer 对不存在的文件会往
            return                    # stderr 吐一行警告，读盘前挡掉
        r = QSvgRenderer(str(f), self)
        if not r.isValid():
            return
        self._svg = r
        if not r.defaultSize().isEmpty():
            self._vb = r.defaultSize()

    def height_for_width(self, width: int) -> int:
        """宽度撑满、高度按 viewBox 推出。与 graphics.FigureCanvas 同形。"""
        w = self._vb.width() or 1
        return max(90, int(round(width * self._vb.height() / float(w))))

    def resizeEvent(self, ev) -> None:              # noqa: N802  Qt 接口名
        super().resizeEvent(ev)
        want = self.height_for_width(self.width())
        if want != self.height():                   # 守卫：否则 setFixedHeight 递归
            self.setFixedHeight(want)

    def paintEvent(self, _ev) -> None:              # noqa: N802  Qt 接口名
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._svg is None:
            self._fallback(p)
            p.end()
            return
        # `render(painter, rect)` 是把图形**拉满**给定矩形：直接传 self.rect()，
        # 控件一旦不是 560:300，桥就被拉扁。先按 viewBox 算等比缩放再居中。
        vb = self._vb
        s = min(self.width() / float(vb.width()),
                self.height() / float(vb.height()))
        w, h = vb.width() * s, vb.height() * s
        self._svg.render(p, QRectF((self.width() - w) / 2.0,
                                   (self.height() - h) / 2.0, w, h))
        p.end()

    def _fallback(self, p: QPainter) -> None:
        """SVG 读不到时退回仓库里那张线稿。

        登录页是用户见到的第一屏，这里不该出现"加载失败""尚无插图数据"这类字样。
        `hud.paint_scene` 画的正是无人机 + 桥梁，语义相同、只是没有渐变，看起来
        像有意为之的线稿而不是坏掉的图；它在 rect 小于 200×110 时自己不画，
        这种沉默正是这里要的降级行为。
        """
        c = QColor(C["accent"])
        c.setAlpha(70)
        hud.paint_scene(p, QRectF(self.rect()), c)


class HeroPanel(QFrame):
    """登录页左栏：品牌渐变底 + 两团辉光。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("HeroPanel")     # 底色由 build_qss() 出
        self.setFixedWidth(HERO_W)          # 必须与 WIN_W 一起改，见文件头的注释

    def paintEvent(self, _ev) -> None:      # noqa: N802  Qt 接口名
        """样式表底 + 两团径向辉光。

        `PE_Widget` 那一笔不能省：重写 paintEvent 后 QWidget 的默认实现不再执行，
        而 `#HeroPanel` 的渐变正是由它画的——省掉整栏就变透明，与 PageHeader 是
        同一处坑。

        两团辉光只能在这里画：一条 QSS 规则的 background 只收一层画刷，而网页版那面
        底是「两团 radial-gradient + 一层 linear-gradient」叠出来的。alpha 值照抄
        网页（26 / 31），画序也照抄——先蓝后青，与 CSS 里第一层在最上一致。

        （这里以前是一条**没有选择器的**样式表，写在控件上。Qt 会把它应用到该控件
        及其整棵子树，于是每个标签都在自己身上重画了一遍渐变、还带一条 1px 右边框，
        只因为那两个色近乎相同才没露馅；而且祖先样式表的优先级高于应用级样式表，
        它能悄悄压过 build_qss() 里的 #Hint。改用 objectName 就是为了根治这个。）
        """
        opt = QStyleOption()
        opt.initFrom(self)
        p = QStylePainter(self)
        p.drawPrimitive(QStyle.PrimitiveElement.PE_Widget, opt)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        w, h = self.width(), self.height()
        for cx, cy, r, col, a in (
                (0.88 * w, 0.92 * h, 0.60 * w, C["accent_2"], 31),   # 右下：蓝
                (0.12 * w, 0.08 * h, 0.62 * w, C["accent"],   26)):  # 左上：青
            g = QRadialGradient(QPointF(cx, cy), r)
            c0, c1 = QColor(col), QColor(col)
            c0.setAlpha(a)
            c1.setAlpha(0)
            g.setColorAt(0.0, c0)
            g.setColorAt(1.0, c1)
            p.setBrush(g)
            p.drawRect(0, 0, w, h)


class _PointIcon(QWidget):
    """短标签前面那枚小线稿。

    用 QPainter 画而不是 `QLabel.setPixmap`：位图快照在高分屏上会糊，
    与 NavButton 里那段理由是同一段。这一枚也替代了原来的 `◆` 几何字符——
    字体里没有对应字形时它会掉进回退、不同机器宽度还不一致。
    """

    def __init__(self, name: str, size: int = 16, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, _ev) -> None:              # noqa: N802  Qt 接口名
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # 线宽 1.8：16/24 的缩放会把默认的 1.6 落到约 1.07px，偏细发灰
        icons.paint(p, self.name, QRectF(self.rect()), QColor(C["accent"]), 1.8)
        p.end()


class LoginWindow(QDialog):
    """登录窗口。登录成功后 accepted 并暴露 self.user。"""

    loggedIn = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("AppRoot")
        self.setWindowTitle(f"{cfg.APP_NAME} · 登录")
        self.setFixedSize(*self._fit_size())
        self.user: dict | None = None

        s = cfg.load_settings()
        self.remember = QCheckBox("记住账号")
        self.remember.setChecked(bool(s.get("last_username")))
        if s.get("last_username"):                  # 记住过就把账号填回去，只剩口令要打
            self._last_user = str(s["last_username"])
        else:
            self._last_user = ""

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._left(), 0)
        root.addWidget(self._right(), 1)

        self.toast = Toast(self)

    @staticmethod
    def _fit_size() -> tuple[int, int]:
        """放得下就用大尺寸，放不下退回旧的 880×520。

        `setFixedSize` 定的是**客户区**，600 加上约 32px 标题栏要 632px 屏高。
        本机可用高 912px 没问题，但答辩现场可能接投影、也可能是高缩放屏，
        真放不下时窗口会被裁掉一截，且没有任何报错。
        """
        scr = QApplication.primaryScreen()
        if scr is not None:
            avail = scr.availableGeometry().height()
            if avail and avail < WIN_H + 60:
                return WIN_SMALL
        return WIN_W, WIN_H

    # -- 左：品牌与插图 --------------------------------------------------
    def _left(self) -> QWidget:
        w = HeroPanel()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(34, 34, 30, 30)
        lay.setSpacing(0)

        lay.addWidget(IconBadge("bridge", 46), 0, Qt.AlignmentFlag.AlignLeft)
        lay.addSpacing(16)

        t = QLabel(cfg.APP_NAME)
        t.setObjectName("HeroTitle")
        lay.addWidget(t)
        lay.addSpacing(8)

        sub = QLabel(cfg.APP_TITLE)
        sub.setObjectName("HeroSub")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        # 两处等分伸缩把插图竖直居中，对应网页版 .login-hero 的 space-between
        # 加 .hero-art 的 align-self: center。只给一处 stretch 会在插图上下
        # 各留一个大小不等的洞，看着像布局错了而不是留白。
        lay.addStretch(1)
        self.art = HeroArt(HERO_W - 64)
        lay.addWidget(self.art)
        lay.addStretch(1)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(9)
        for i, (icon, text) in enumerate(POINTS):
            grid.addWidget(self._point(icon, text), i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        lay.addSpacing(16)

        # 只留版本号。这里原来把 cfg.APP_TITLE 又印了一遍——那句 21 字的副标题
        # 在这一栏里出现了两次（上面一次、这里一次）。
        foot = QLabel(f"{cfg.APP_NAME_EN} · {cfg.APP_VERSION}")
        foot.setObjectName("HeroFoot")
        lay.addWidget(foot)
        return w

    @staticmethod
    def _point(icon: str, text: str) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7)
        lay.addWidget(_PointIcon(icon))
        lab = QLabel(text)
        lab.setObjectName("HeroPoint")
        lay.addWidget(lab)
        lay.addStretch(1)
        return w

    # -- 右：表单 --------------------------------------------------------
    def _right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(80, 44, 80, 40)
        lay.setSpacing(0)

        # 首尾两道等分伸缩把表单竖直居中，对应网页版 `.login-side` 的
        # `align-items: center`。只留末尾一道的话，所有富余高度都堆在按钮下面
        # ——980×600 下那是 137px 的一整片空白，看着像页面没写完。
        lay.addStretch(1)

        self.tab_login = QPushButton("登录")
        self.tab_reg = QPushButton("注册")
        tabs = QHBoxLayout()
        tabs.setSpacing(6)
        for b in (self.tab_login, self.tab_reg):
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setMinimumHeight(34)
            tabs.addWidget(b)
        self.tab_login.setChecked(True)
        self.tab_login.clicked.connect(lambda: self._switch(0))
        self.tab_reg.clicked.connect(lambda: self._switch(1))
        lay.addLayout(tabs)
        lay.addSpacing(28)

        self.title = QLabel("欢迎回来")
        self.title.setObjectName("FormTitle")
        lay.addWidget(self.title)
        self.subtitle = QLabel("请使用巡检账号登录系统")
        self.subtitle.setObjectName("FormSub")
        lay.addWidget(self.subtitle)
        lay.addSpacing(26)

        self.in_user = QLineEdit(self._last_user)
        self.in_user.setPlaceholderText("账号")
        self.in_user.setMinimumHeight(40)
        self.in_pwd = QLineEdit()
        self.in_pwd.setPlaceholderText("密码")
        self.in_pwd.setEchoMode(QLineEdit.EchoMode.Password)
        self.in_pwd.setMinimumHeight(40)
        self.in_pwd2 = QLineEdit()
        self.in_pwd2.setPlaceholderText("确认密码")
        self.in_pwd2.setEchoMode(QLineEdit.EchoMode.Password)
        self.in_pwd2.setMinimumHeight(40)
        self.in_pwd2.hide()
        lay.addWidget(self.in_user)
        lay.addSpacing(12)
        lay.addWidget(self.in_pwd)
        # 确认密码框上面那道 12px 空白必须**跟它一起收掉**，所以装成一个 QWidget
        # 而不是 addSpacing()：后者装的 QSpacerItem 不会跟着 in_pwd2 一起 hide()，
        # 登录模式下密码框底下就挂着一个解释不通的空隙（早先正是这么写的）。
        self.gap_pwd2 = QWidget()
        self.gap_pwd2.setFixedHeight(12)
        self.gap_pwd2.hide()
        lay.addWidget(self.gap_pwd2)
        lay.addWidget(self.in_pwd2)
        lay.addSpacing(12)
        self.in_pwd.returnPressed.connect(self._submit)
        self.in_pwd2.returnPressed.connect(self._submit)

        opt = QHBoxLayout()
        opt.addWidget(self.remember)
        opt.addStretch(1)
        hint = QLabel("内置账号 admin / 123456")
        hint.setObjectName("HeroFoot")
        opt.addWidget(hint)
        lay.addLayout(opt)
        lay.addSpacing(18)

        self.btn = QPushButton("登 录")
        self.btn.setProperty("accent", "primary")
        self.btn.setMinimumHeight(42)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.clicked.connect(self._submit)
        lay.addWidget(self.btn)
        lay.addSpacing(12)

        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        lay.addWidget(self.msg)
        lay.addSpacing(20)

        lay.addWidget(self._status())
        lay.addStretch(1)

        self._switch(0)
        return w

    @staticmethod
    def _status() -> QWidget:
        """表单下方那行运行状态，对应网页版登录页的 `#mode`。

        网页版这一行是登录前 fetch /api/health 拿到的；桌面版直接问本地。
        **只报「已配置 / 未配置」，不回显 Key 原文**——与 web/api.py 的
        health()/settings() 是同一条规矩：界面上要能看出有没有配 Key，
        但 Key 一个字符都不该出现。

        这里**不 import core.agent**：那个包会经 tools.py → backend.py 拉进
        cv2 / numpy。main.py 是先建登录窗、后建主窗口，引它等于把整个检测后端
        的导入挪到登录路径上——启动变慢，且 cv2 缺失时会连登录都进不去。
        """
        online = cfg.llm_ready()
        src = (cfg.llm_provider_label() + "（在线）" if online
               else "本地规则引擎（未配置 API Key）")
        try:
            st = db.stat_summary()
            n_img = int(st.get("images") or 0)
        except Exception:                           # noqa: BLE001
            n_img = 0                               # 库还没建好也不该拦着人登录

        lab = QLabel(f"规划来源：{src}　影像 {n_img} 张")
        lab.setObjectName("HeroFoot")
        lab.setWordWrap(True)
        return lab

    # -- 逻辑 ------------------------------------------------------------
    def _switch(self, mode: int) -> None:
        self.mode = mode
        self.tab_login.setChecked(mode == 0)
        self.tab_reg.setChecked(mode == 1)
        self.tab_login.setProperty("accent", "primary" if mode == 0 else "")
        self.tab_reg.setProperty("accent", "primary" if mode == 1 else "")
        for b in (self.tab_login, self.tab_reg):
            b.style().unpolish(b)
            b.style().polish(b)
        self.title.setText("欢迎回来" if mode == 0 else "创建巡检账号")
        self.subtitle.setText("请使用巡检账号登录系统" if mode == 0
                              else "账号与口令将写入本地用户库")
        self.btn.setText("登 录" if mode == 0 else "注 册")
        self.in_pwd2.setVisible(mode == 1)
        self.gap_pwd2.setVisible(mode == 1)
        self.msg.setText("")

    def _submit(self) -> None:
        u = self.in_user.text().strip()
        p = self.in_pwd.text()
        if not u or not p:
            self.msg.setText("请填写账号与密码")
            return
        if self.mode == 0:
            row = db.verify_user(u, p)
            if not row:
                self.msg.setText("账号或密码不正确")
                return
            s = cfg.load_settings()
            s["last_username"] = u if self.remember.isChecked() else ""
            cfg.save_settings(s)
            self.user = dict(row)
            self.loggedIn.emit(self.user)
            self.accept()
            return

        if len(p) < 6:
            self.msg.setText("密码至少 6 位")
            return
        if p != self.in_pwd2.text():
            self.msg.setText("两次输入的密码不一致")
            return
        ok, msg = db.create_user(u, p, role="巡检员")
        if not ok:
            self.msg.setText(msg)
            return
        self.msg.setStyleSheet(f"font-size: 12.5px; color: {C['ok']};")
        self.msg.setText(msg + "，请登录")
        self.in_pwd.clear()
        self.in_pwd2.clear()
        QTimer.singleShot(900, lambda: self._switch(0))


def run_login() -> dict | None:
    """便捷入口：弹出登录窗口，返回登录用户或 None。"""
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    # 与 main.py 一样要设默认字体：漏了这一步，本入口路径下所有裸 QFont() 都会
    # 落到 Qt 默认族，同一个界面里出现两种中文字体。
    theme.apply_app_font(app)
    app.setStyleSheet(theme.build_qss())
    db.ensure_default_user()
    w = LoginWindow()
    w.show()
    app.exec()
    return w.user
