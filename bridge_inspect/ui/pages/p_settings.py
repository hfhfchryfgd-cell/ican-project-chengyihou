"""⑥ 系统管理。

对应《作品说明书》§6.5「系统管理」：账号信息、模型配置、存储路径、大模型 API 四块，右侧
常驻系统状态。

模型配置是这一页最要紧的地方——它是"当前为什么是演示后端"和"以后怎么换成真模型"的
唯一入口：把训练好的权重放进 models/ 或在这里指定路径，重载后六个页面同时切到真实推理。
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from ... import config as cfg
from ... import db, theme
from ...core import backend, llm
from ..context import AppContext
from ..widgets import (
    Badge, Hint, KeyValue, Panel, PageHeader, ToolBar, hline,
)

C = theme.C


class SettingsPage(QWidget):
    key = "settings"
    title = "系统管理"
    subtitle = "账号、模型权重、存储路径与大模型接口配置"

    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self._build()
        self.load_values()
        self.refresh_status()
        self.ctx.backendChanged.connect(self.refresh_status)

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = PageHeader(self.key, self.title, self.subtitle)
        bar = ToolBar()
        bar.add("保存设置", self.save, accent="primary")
        bar.add("恢复默认", self.restore_default)
        bar.add("重载模型后端", self.reload_backend, accent="cyan")
        head.add_action(bar)
        root.addWidget(head)

        body = QWidget()
        lay = QHBoxLayout(body)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)
        root.addWidget(body, 1)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(12)
        lv.addWidget(self._account_panel())
        lv.addWidget(self._model_panel())
        lv.addStretch(1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(12)
        rv.addWidget(self._storage_panel())
        rv.addWidget(self._llm_panel())
        rv.addWidget(self._status_panel())
        rv.addStretch(1)

        lay.addWidget(left, 3)
        lay.addWidget(right, 2)

    # -- 账号 ------------------------------------------------------------
    def _account_panel(self) -> Panel:
        p = Panel("账号信息")
        self.kv_user = KeyValue(columns=2)
        p.body().addWidget(self.kv_user)
        p.body().addWidget(hline())

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.addWidget(QLabel("原密码"), 0, 0)
        self.in_old = QLineEdit()
        self.in_old.setEchoMode(QLineEdit.EchoMode.Password)
        grid.addWidget(self.in_old, 0, 1)
        grid.addWidget(QLabel("新密码"), 1, 0)
        self.in_new = QLineEdit()
        self.in_new.setEchoMode(QLineEdit.EchoMode.Password)
        grid.addWidget(self.in_new, 1, 1)
        grid.addWidget(QLabel("确认新密码"), 2, 0)
        self.in_new2 = QLineEdit()
        self.in_new2.setEchoMode(QLineEdit.EchoMode.Password)
        grid.addWidget(self.in_new2, 2, 1)
        btn = QPushButton("修改密码")
        btn.clicked.connect(self.change_pwd)
        grid.addWidget(btn, 2, 2)
        grid.setColumnStretch(1, 1)
        p.body().addLayout(grid)
        return p

    def change_pwd(self) -> None:
        old, new, new2 = self.in_old.text(), self.in_new.text(), self.in_new2.text()
        if not old or not new:
            self.toast("请填写原密码与新密码", "warn")
            return
        if new != new2:
            self.toast("两次输入的新密码不一致", "warn")
            return
        ok, msg = db.change_password(self.ctx.user.get("username", ""), old, new)
        self.toast(msg, "ok" if ok else "error")
        if ok:
            for w in (self.in_old, self.in_new, self.in_new2):
                w.clear()

    # -- 模型 ------------------------------------------------------------
    def _model_panel(self) -> Panel:
        p = Panel("模型配置")
        p.body().addWidget(Hint(
            "models/ 下存在权重时系统自动切换为真实推理；此处可手动指定路径。"
            "未加载权重时使用内置演示后端。"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        r = 0

        grid.addWidget(QLabel("检测权重"), r, 0)
        self.in_det = QLineEdit()
        self.in_det.setPlaceholderText("*.pt / *.onnx，留空则用演示后端")
        grid.addWidget(self.in_det, r, 1)
        b = QPushButton("浏览")
        b.clicked.connect(lambda: self._pick(self.in_det, "选择检测权重"))
        grid.addWidget(b, r, 2)
        r += 1

        grid.addWidget(QLabel("分割权重"), r, 0)
        self.in_seg = QLineEdit()
        self.in_seg.setPlaceholderText("*-seg.pt，留空则用演示分割")
        grid.addWidget(self.in_seg, r, 1)
        b = QPushButton("浏览")
        b.clicked.connect(lambda: self._pick(self.in_seg, "选择分割权重"))
        grid.addWidget(b, r, 2)
        r += 1

        grid.addWidget(QLabel("置信度阈值"), r, 0)
        self.sl_conf = QSlider(Qt.Orientation.Horizontal)
        self.sl_conf.setRange(5, 95)
        self.lbl_conf = QLabel("0.35")
        self.lbl_conf.setObjectName("Mono")
        self.sl_conf.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"{v / 100:.2f}"))
        grid.addWidget(self.sl_conf, r, 1)
        grid.addWidget(self.lbl_conf, r, 2)
        r += 1

        grid.addWidget(QLabel("IoU 阈值"), r, 0)
        self.sl_iou = QSlider(Qt.Orientation.Horizontal)
        self.sl_iou.setRange(10, 90)
        self.lbl_iou = QLabel("0.45")
        self.lbl_iou.setObjectName("Mono")
        self.sl_iou.valueChanged.connect(
            lambda v: self.lbl_iou.setText(f"{v / 100:.2f}"))
        grid.addWidget(self.sl_iou, r, 1)
        grid.addWidget(self.lbl_iou, r, 2)
        r += 1

        grid.addWidget(QLabel("推理设备"), r, 0)
        self.cb_dev = QComboBox()
        self.cb_dev.addItems(["auto", "cpu", "cuda:0"])
        grid.addWidget(self.cb_dev, r, 1)
        r += 1

        grid.addWidget(QLabel("推理尺寸"), r, 0)
        self.sp_size = QDoubleSpinBox()
        self.sp_size.setRange(320, 1920)
        self.sp_size.setSingleStep(32)
        self.sp_size.setDecimals(0)
        grid.addWidget(self.sp_size, r, 1)
        r += 1

        grid.setColumnStretch(1, 1)
        p.body().addLayout(grid)
        return p

    def _storage_panel(self) -> Panel:
        p = Panel("存储路径")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.addWidget(QLabel("输出目录"), 0, 0)
        self.in_out = QLineEdit()
        grid.addWidget(self.in_out, 0, 1)
        b = QPushButton("浏览")
        b.clicked.connect(self._pick_out)
        grid.addWidget(b, 0, 2)
        grid.setColumnStretch(1, 1)
        p.body().addLayout(grid)
        p.body().addWidget(Hint(
            f"影像库：{cfg.IMG_DIR}\n定位数据库：{cfg.DB_PATH}\n"
            f"权重目录：{cfg.MODELS_DIR}"))
        return p

    def _llm_panel(self) -> Panel:
        p = Panel("大模型接口")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        grid.addWidget(QLabel("服务商"), 0, 0)
        self.cb_prov = QComboBox()
        for provider in ("openai", "deepseek", "dashscope"):
            self.cb_prov.addItem(cfg.LLM_PROVIDER_PRESETS[provider]["label"], provider)
        self.cb_prov.currentIndexChanged.connect(self._apply_provider_preset)
        grid.addWidget(self.cb_prov, 0, 1)
        grid.addWidget(QLabel("Base URL"), 1, 0)
        self.in_url = QLineEdit()
        self.in_url.setPlaceholderText(cfg.LLM_PROVIDER_PRESETS["openai"]["base_url"])
        grid.addWidget(self.in_url, 1, 1)
        grid.addWidget(QLabel("API Key"), 2, 0)
        self.in_key = QLineEdit()
        self.in_key.setEchoMode(QLineEdit.EchoMode.Password)
        grid.addWidget(self.in_key, 2, 1)
        grid.addWidget(QLabel("模型名"), 3, 0)
        self.in_model = QLineEdit()
        self.in_model.setPlaceholderText("gpt-4o-mini / deepseek-chat / qwen-plus")
        grid.addWidget(self.in_model, 3, 1)
        row = QHBoxLayout()
        test = QPushButton("测试连接")
        test.clicked.connect(self.test_llm)
        row.addWidget(test)
        row.addStretch(1)
        grid.addLayout(row, 4, 1)
        grid.setColumnStretch(1, 1)
        p.body().addLayout(grid)
        # 原先这里是 48 个字的设计自辩（"不会把规则结果冒充成大模型输出"）。
        # 那是对评委说的话，不是对使用者说的话；使用者只需要知道填了会怎样。
        p.body().addWidget(Hint(
            "选择 DeepSeek 会自动填入 https://api.deepseek.com 和 deepseek-chat；"
            "填入自己的 API Key 后点“测试连接”。留空则由本地规则引擎接管。"))
        return p

    def _status_panel(self) -> Panel:
        p = Panel("系统状态")
        self.kv_status = KeyValue(columns=1)
        p.body().addWidget(self.kv_status)
        p.body().addWidget(hline())
        self.badge = Badge("演示后端")
        p.body().addWidget(self.badge)
        return p

    # -- 取值 / 存值 ------------------------------------------------------
    def load_values(self) -> None:
        s = self.ctx.settings
        u = self.ctx.user
        self.kv_user.clear()
        self.kv_user.add("账号", u.get("username", "—"))
        self.kv_user.add("角色", u.get("role", "—"))
        self.kv_user.add("显示名称", u.get("display_name") or u.get("username", "—"))

        self.in_det.setText(s.get("detect_weights", ""))
        self.in_seg.setText(s.get("segment_weights", ""))
        self.sl_conf.setValue(int(float(s.get("conf_thres", 0.35)) * 100))
        self.sl_iou.setValue(int(float(s.get("iou_thres", 0.45)) * 100))
        self.sp_size.setValue(float(s.get("img_size", 960)))
        i = self.cb_dev.findText(s.get("device", "auto"))
        self.cb_dev.setCurrentIndex(i if i >= 0 else 0)
        self.in_out.setText(s.get("output_dir", str(cfg.OUT_DIR)))
        provider = cfg.normalize_llm_provider(s.get("llm_provider"))
        i = self.cb_prov.findData(provider)
        self.cb_prov.setCurrentIndex(i if i >= 0 else 0)
        self.in_url.setText(s.get("llm_base_url", ""))
        self.in_key.setText(s.get("llm_api_key", ""))
        self.in_model.setText(s.get("llm_model", ""))

    def _collect(self) -> dict:
        return {
            "detect_weights": self.in_det.text().strip(),
            "segment_weights": self.in_seg.text().strip(),
            "conf_thres": self.sl_conf.value() / 100,
            "iou_thres": self.sl_iou.value() / 100,
            "device": self.cb_dev.currentText(),
            "img_size": int(self.sp_size.value()),
            "output_dir": self.in_out.text().strip() or str(cfg.OUT_DIR),
            "llm_provider": str(self.cb_prov.currentData() or "openai"),
            "llm_base_url": self.in_url.text().strip(),
            "llm_api_key": self.in_key.text().strip(),
            "llm_model": self.in_model.text().strip(),
            "llm_enabled": bool(self.in_key.text().strip()),
        }

    def _apply_provider_preset(self) -> None:
        """切换服务商时只替换空值或旧预设，避免覆盖用户的自定义网关地址。"""
        provider = str(self.cb_prov.currentData() or "openai")
        preset = cfg.LLM_PROVIDER_PRESETS[provider]
        known_urls = {p["base_url"] for p in cfg.LLM_PROVIDER_PRESETS.values()}
        known_models = {p["model"] for p in cfg.LLM_PROVIDER_PRESETS.values()}
        if not self.in_url.text().strip() or self.in_url.text().strip() in known_urls:
            self.in_url.setText(preset["base_url"])
        if not self.in_model.text().strip() or self.in_model.text().strip() in known_models:
            self.in_model.setText(preset["model"])
        self.in_url.setPlaceholderText(preset["base_url"])

    def save(self) -> None:
        self.ctx.apply_settings(self._collect())
        self.refresh_status()

    def restore_default(self) -> None:
        self.ctx.settings.update(cfg.DEFAULT_SETTINGS)
        cfg.save_settings(self.ctx.settings)
        self.load_values()
        self.ctx.reload_backends()
        self.refresh_status()
        self.toast("已恢复默认设置", "ok")

    def reload_backend(self) -> None:
        self.save()
        self.ctx.reload_backends()
        name, demo = self.ctx.backend_label()
        self.toast(f"后端已重载：{name}", "warn" if demo else "ok")

    def test_llm(self) -> None:
        s = dict(self.ctx.settings)
        s.update(self._collect())
        ok, msg = llm.test_connection(s)
        self.toast(msg, "ok" if ok else "error")

    # -- 状态 ------------------------------------------------------------
    def refresh_status(self) -> None:
        self.kv_status.clear()
        self.kv_status.add("Python", sys.version.split()[0])
        try:
            import PyQt6.QtCore as qc

            self.kv_status.add("PyQt6", qc.PYQT_VERSION_STR)
        except Exception:                       # pragma: no cover
            self.kv_status.add("PyQt6", "—")
        try:
            import cv2

            self.kv_status.add("OpenCV", cv2.__version__)
        except Exception:                       # pragma: no cover
            self.kv_status.add("OpenCV", "—")
        try:
            import torch

            gpu = "可用" if torch.cuda.is_available() else "不可用（CPU 推理）"
            self.kv_status.add("torch", f"{torch.__version__}　CUDA {gpu}",
                               C["ok"] if torch.cuda.is_available() else C["text_dim"])
        except Exception:                       # pragma: no cover
            self.kv_status.add("torch", "未安装")

        d, sg = self.ctx.detector, self.ctx.segmenter
        self.kv_status.add("检测后端", d.name, C["warn"] if d.mode == "demo" else C["ok"])
        self.kv_status.add("分割后端", sg.name, C["warn"] if sg.mode == "demo" else C["ok"])
        found = backend.discover_weights()
        self.kv_status.add("models/ 权重", f"{len(found)} 个"
                           + ("（" + "、".join(p.name for p in found[:3]) + "）"
                              if found else "　目录为空"))
        st = db.stat_summary()
        self.kv_status.add("影像 / 检测 / 量化",
                           f"{st['images']} / {st['detections']} / {st['segments']}")
        self.kv_status.add("巡检点位", f"{st['points']} 个")
        self.badge.set_state(*self.ctx.backend_label())

    # -- 工具 ------------------------------------------------------------
    def _pick(self, edit: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, title, str(cfg.MODELS_DIR), "模型权重 (*.pt *.pth *.onnx)")
        if path:
            edit.setText(path)

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录",
                                             self.in_out.text() or str(cfg.OUT_DIR))
        if d:
            self.in_out.setText(d)

    def toast(self, text: str, kind: str = "info") -> None:
        self.ctx.toast(text, kind)

    def on_show(self) -> None:
        self.refresh_status()
