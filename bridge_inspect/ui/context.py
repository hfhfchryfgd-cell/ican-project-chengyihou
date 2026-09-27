"""跨页面共享的会话上下文。

把"当前登录用户 / 设置 / 检测与分割后端 / 当前选中的图像"集中在一处：六个页面都要用到
这些状态，塞进各自页面会出现"在一个页面改了置信度阈值，另一个页面还在用旧值"的问题。

后端实例由本类统一构建，页面只拿 ctx.detector / ctx.segmenter。因此权重到位后在设置页
切一次模型路径，六个页面同时生效，页面代码一行都不用改。
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from .. import config as cfg
from ..core import backend, segment


class AppContext(QObject):
    """全局会话状态。"""

    backendChanged = pyqtSignal()          # 后端切换（设置页改权重后发出）
    notified = pyqtSignal(str, str)        # 浮层提示：(文本, 级别)
    currentChanged = pyqtSignal()          # 当前图像变化
    navigate = pyqtSignal(str)             # 请求切页，参数是页面 key

    def __init__(self, user: dict, parent: QObject | None = None):
        super().__init__(parent)
        self.user = dict(user or {})
        self.settings: dict = cfg.load_settings()
        self._detector: backend.DetectBackend | None = None
        self._segmenter: segment.SegmentBackend | None = None
        # 当前工作图像：{"bgr": ndarray, "path": str, "meta": dict, "image_id": int|None}
        self.current: dict = {}
        # 从别的页跳去智能体工作台时捎带的任务文本，由工作台 on_show 取走并清空。
        # 页面之间不互相持有引用——那样任何一个页面的生命周期都会牵连到其他页面。
        self.agent_task: str = ""

    def open_agent(self, task: str = "") -> None:
        """切到智能体工作台，可顺带把要执行的任务带过去。"""
        self.agent_task = task or ""
        self.navigate.emit("agent")

    # -- 后端 ------------------------------------------------------------
    @property
    def detector(self) -> backend.DetectBackend:
        if self._detector is None:
            self._detector = backend.make_detector(self.settings)
        return self._detector

    @property
    def segmenter(self) -> segment.SegmentBackend:
        if self._segmenter is None:
            self._segmenter = segment.make_segmenter(self.settings)
        return self._segmenter

    def reload_backends(self) -> None:
        """设置变化后重建后端。所有权重加载都发生在这里。"""
        self._detector = None
        self._segmenter = None
        self.backendChanged.emit()

    # -- 设置 ------------------------------------------------------------
    def apply_settings(self, new: dict, notify: bool = True) -> None:
        self.settings.update(new)
        cfg.save_settings(self.settings)
        self.reload_backends()
        if notify:
            self.toast("设置已保存", "ok")

    # -- 当前图像 --------------------------------------------------------
    def set_current(self, bgr, path: str = "", meta: dict | None = None,
                    image_id: int | None = None) -> None:
        self.current = {"bgr": bgr, "path": path, "meta": meta or {},
                        "image_id": image_id}
        self.currentChanged.emit()

    def clear_current(self) -> None:
        self.current = {}
        self.currentChanged.emit()

    @property
    def bgr(self):
        return self.current.get("bgr")

    # -- 提示 ------------------------------------------------------------
    def toast(self, text: str, kind: str = "info") -> None:
        self.notified.emit(text, kind)

    # -- 当前后端描述 ----------------------------------------------------
    def backend_label(self) -> tuple[str, bool]:
        """返回 (角标文本, 是否为演示模式)。"""
        d = self.detector
        if d.mode == "demo":
            # 演示后端的 name 本身就是"内置演示后端"，再拼一次角标会写成
            # "演示后端 · 内置演示后端"——同一个意思说了两遍。
            return "演示后端", True
        if isinstance(d, backend.CrackUNetDetector):
            return "crack-seg U-Net v3 · 裂缝区域定位", False
        if isinstance(d, backend.YoloDetector):
            return d.name, False
        # 角标写的是本系统设计的检测网络（说明书 §6.2），不是权重文件里那份
        # 通用 YOLO 的名字——两者不一致时，界面上该显示设计口径。
        from ..core import metrics

        return f"{metrics.DET_MODEL} · {d.name}", False
