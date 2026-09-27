"""桥梁智巡 —— 面向铁路桥梁的同视场无人机病害巡检分析系统。

启动方式：
    py main.py            （或双击 run.bat）

流程：登录 / 注册 → 主窗口（图片导入 → 目标检测 → 分割量化 → 长周期监测 →
检测历史 → 系统管理）。退出登录后回到登录页，可切换账号。

关于算法后端：
    当前没有桥梁病害权重与实拍数据集，检测与分割使用内置演示后端，界面角标与
    状态栏会明确标注。把训练好的 .pt/.onnx 放进 models/ 或到「系统管理」里指定
    路径，重载后整条链路自动切换到真实推理，界面代码无需改动。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge_inspect import config as cfg          # noqa: E402
from bridge_inspect import db, theme              # noqa: E402


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName(cfg.APP_NAME)
    app.setApplicationVersion(cfg.APP_VERSION)

    # 字体与样式都从 theme 取：族清单与字号在那边定义一次，本文件不再自带副本。
    theme.apply_app_font(app)
    app.setStyleSheet(theme.build_qss())

    db.connect()
    db.ensure_default_user()

    # 登录 → 主窗口 →（退出登录则回到登录页）
    while True:
        from bridge_inspect.ui.login import LoginWindow

        login = LoginWindow()
        if not login.exec():
            return 0

        from bridge_inspect.ui.main_window import MainWindow

        win = MainWindow(login.user)
        win.show()
        app.exec()
        if not getattr(win, "logged_out", False):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
