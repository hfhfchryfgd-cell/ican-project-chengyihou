"""主窗口：左侧导航 + 六页堆栈 + 底部状态栏。

页面用 QStackedWidget 常驻，切换时只调 on_show() 刷新数据，不重建控件——重建会让
各页的图表与选中状态丢失，演示时来回切换会闪。
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QStatusBar, QWidget,
)

from .. import config as cfg
from .context import AppContext
from .hud import HudRoot
from .pages import AgentPage, DetectPage, HistoryPage, SegmentPage, SettingsPage, TrendPage
from .widgets import Sidebar, Toast


class MainWindow(QMainWindow):
    def __init__(self, user: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = AppContext(user, self)
        self.logged_out = False
        self._syncing = False

        self.setObjectName("AppRoot")
        self.setWindowTitle(f"{cfg.APP_NAME} · {cfg.APP_TITLE}")
        self.resize(1440, 902)
        self.setMinimumSize(1180, 760)

        # 底板用 HudRoot：它在卡片之间的空隙里画网格与线稿。全局 QSS 里
        # `QWidget { background: transparent }`，页面本身是透的，装饰才透得出来。
        central = HudRoot()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar(user)
        self.sidebar.navChanged.connect(self.goto)
        self.sidebar.logoutClicked.connect(self.logout)
        root.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        # 顺序必须与 Sidebar.ITEMS 一致：侧边栏按序号发导航事件，
        # 两处顺序一旦错位，点"目标检测"会跳到别的页。
        self.pages = [AgentPage(self.ctx), DetectPage(self.ctx), SegmentPage(self.ctx),
                      HistoryPage(self.ctx), TrendPage(self.ctx), SettingsPage(self.ctx)]
        for p in self.pages:
            self.stack.addWidget(p)

        self._build_status()
        self.ctx.notified.connect(self.toast.show_message)
        self.ctx.navigate.connect(self.goto_key)
        self.goto(0)


    # ------------------------------------------------------------------
    def _build_status(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.lbl_task = QLabel("就绪")
        self.lbl_backend = QLabel("")
        self.lbl_db = QLabel("")
        for w in (self.lbl_task, self.lbl_backend):
            w.setObjectName("Hint")
        self.lbl_db.setObjectName("Hint")

        sb.addWidget(self.lbl_task, 1)
        sb.addPermanentWidget(self.lbl_backend)
        sb.addPermanentWidget(self.lbl_db)
        self.toast = Toast(self)
        self._refresh_status()

    def _refresh_status(self) -> None:
        from .. import theme

        name, demo = self.ctx.backend_label()
        self.lbl_backend.setText(f"检测后端：{name}")
        self.lbl_backend.setStyleSheet(
            f"color: {theme.C['warn'] if demo else theme.C['ok']};")
        self.lbl_db.setText(f"数据库：{cfg.DB_PATH.name}")

    # ------------------------------------------------------------------
    def goto(self, index: int) -> None:
        if not (0 <= index < len(self.pages)):
            return
        self.stack.setCurrentIndex(index)
        page = self.pages[index]
        # 从代码跳转（如点击表格行联动）时同步侧边栏高亮，避免"在长周期监测页
        # 而菜单还亮着目标检测"这种不一致
        if not self._syncing:
            self._syncing = True
            btn = self.sidebar.group.button(index)
            if btn and not btn.isChecked():
                btn.setChecked(True)
            self._syncing = False
        if hasattr(page, "on_show"):
            page.on_show()
        # 状态栏不再复述标题与副标题——PageHeader 就在正上方，同一句话在屏幕
        # 上出现两遍只会占地方。这里留给"正在检测…"这类**过程**信息，
        # 空闲时就是"就绪"。

    def goto_key(self, key: str) -> None:
        """按页面 key 切页。页面之间互相跳转时用这个，而不是记序号——
        序号会随导航顺序调整而失效，key 不会。"""
        for i, p in enumerate(self.pages):
            if getattr(p, "key", "") == key:
                self.goto(i)
                return

    def logout(self) -> None:
        """退出登录：关掉主窗口，由入口重新拉起登录页。"""
        self.logged_out = True
        self.close()
