"""⑦ 智能体工作台。

用自然语言下达一次巡检任务，总控智能体解析意图、委派给专业智能体，每一步的工具
调用与判定依据实时打在执行轨迹上，末尾汇总结论并落盘产物。

这一页与网页版 `/agent` 是同一个内核的两条通路：内核 `core/agent` 只产出事件流
（生成器），这里用 QThread 逐个取、网页版写进 SSE。算法与编排一份实现，界面各写各的。

线程边界要守住：`Orchestrator` 在工作线程里跑，控件只在主线程碰。跨线程一律走
`AgentWorker.event` 信号（Qt 默认队列连接会自动切回主线程），不在回调里直接改界面。
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QListWidget,
    QListWidgetItem, QPlainTextEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QProgressBar, QVBoxLayout, QWidget,
)

from ... import config as cfg
from ... import db
from ... import theme
from ...core.agent import AGENTS, Orchestrator
from ...core import ingest
from ..context import AppContext
from ..widgets import Badge, Hint, PageHeader, Panel, StatCard, ToolBar

C = theme.C

# 事件类型的中文名与配色。与 events.KINDS 一一对应，网页版取同一套语义。
KIND_CN: dict[str, tuple[str, str]] = {
    "run_start":   ("开始", "text_dim"),
    "thought":     ("思考", "info"),
    "delegate":    ("委派", "accent"),
    "tool_call":   ("调用", "purple"),
    "tool_result": ("返回", "ok"),
    "decision":    ("结论", "warn"),
    "artifact":    ("产物", "accent"),
    "warning":     ("告警", "warn"),
    "error":       ("错误", "danger"),
    "run_end":     ("结束", "text_dim"),
}

EXAMPLES = [
    "检查 C 区支座三个批次的情况并生成报告",
    "看看全桥有没有发展异常的病害",
    "A 区桥墩的裂缝这三次飞得怎么样",
    "有哪些点位需要复飞",
    "把这次新拍的片子导入，跑一遍检测再出报告",
]


class AgentWorker(QThread):
    """在工作线程里跑一次智能体任务，逐条把事件抛回主线程。"""

    event = pyqtSignal(object)
    done = pyqtSignal(object)          # 结束时抛 (answer, artifacts, elapsed_ms, degraded)

    def __init__(self, task: str, settings: dict, image_ids: list[int] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.task = task
        self.settings = dict(settings or {})
        self.image_ids = list(image_ids or [])
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:                                  # noqa: D102  QThread
        orc = Orchestrator(settings=self.settings, image_ids=self.image_ids,
                           on_event=lambda ev: self.event.emit(ev))
        try:
            for _ in orc.stream(self.task):
                if self._cancelled:
                    break
        except Exception as exc:                            # noqa: BLE001
            # 内核已经尽量自己兜异常；这里再兜一层，免得线程里抛出后
            # 界面停在"运行中"不动，看起来像卡死
            self.done.emit((f"任务执行失败：{type(exc).__name__}: {exc}", [], 0, False))
            return
        self.done.emit((orc.answer, list(orc.artifacts),
                        getattr(orc, "elapsed_ms", 0),
                        getattr(orc, "degraded", False)))


class AgentGraph(QWidget):
    """智能体拓扑：总控居中，六位专家环列，正在工作的那个亮起来。

    画成拓扑而不是列表，是因为要一眼看出"谁在替谁干活"——比赛文件明确要求
    展示智能体之间的协同，列表看不出协同关系。
    """

    NODE_R = 15.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(216)
        self.active = ""            # 正在工作的智能体 key
        self.seen: set[str] = set()  # 本次任务出现过的
        self.done: set[str] = set()

    def reset(self) -> None:
        self.active = ""
        self.seen.clear()
        self.done.clear()
        self.update()

    def mark(self, agent: str) -> None:
        if agent in AGENTS:
            self.seen.add(agent)
            if self.active and self.active != agent:
                self.done.add(self.active)
            self.active = agent
            self.update()

    def _ring(self) -> list[str]:
        """除总控与 system 外的专家，按花名册顺序排成一圈。"""
        return [k for k in AGENTS if k not in ("system", "orchestrator")]

    def paintEvent(self, ev) -> None:                       # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0 + 5
        ring = self._ring()
        rad = min(w / 2.0 - self.NODE_R - 8, h / 2.0 - self.NODE_R - 4)
        rad = max(rad, 44.0)

        positions: dict[str, QPointF] = {}
        for i, key in enumerate(ring):
            ang = -90.0 + i * 360.0 / len(ring)
            positions[key] = QPointF(cx + rad * math.cos(math.radians(ang)),
                                     cy + rad * math.sin(math.radians(ang)))

        hub = QPointF(cx, cy)

        # 连线：已参与本次任务的连线实、其余淡
        for key, pt in positions.items():
            lit = key in self.seen
            col = QColor(AGENTS[key][2] if lit else C["border"])
            col.setAlpha(210 if lit else 110)
            pen = QPen(col)
            pen.setWidthF(1.8 if lit else 1.0)
            p.setPen(pen)
            p.drawLine(hub, pt)

        self._node(p, hub, "总控", "#22D3EE", self.active == "orchestrator",
                   self.active, hub_text="总控")
        for key, pt in positions.items():
            self._node(p, pt, AGENTS[key][0], AGENTS[key][2], self.active == key,
                       self.active, ring_member=True, key=key)
        p.end()

    def _node(self, p: QPainter, pt: QPointF, label: str, color: str, lit: bool,
              active: str, hub_text: str = "", ring_member: bool = False,
              key: str = "") -> None:
        r = self.NODE_R if ring_member else self.NODE_R + 3
        col = QColor(color)
        seen = key in self.seen if ring_member else True

        if lit:
            halo = QColor(col)
            halo.setAlpha(58)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(halo)
            p.drawEllipse(pt, r + 6.5, r + 6.5)

        fill = QColor(col)
        fill.setAlpha(228 if lit else (52 if seen else 26))
        p.setBrush(fill)
        pen = QPen(col if (lit or seen) else QColor(C["border"]))
        pen.setWidthF(2.2 if lit else 1.2)
        p.setPen(pen)
        p.drawEllipse(pt, r, r)

        # 节点名压在圆里，两字竖排会挤，改画在圆的下方
        p.setPen(QColor(C["text"] if (lit or seen) else C["text_muted"]))
        f = QFont()
        f.setFamilies(theme.FONT_FAMILIES)
        f.setPixelSize(10)
        f.setWeight(QFont.Weight.DemiBold if lit else QFont.Weight.Normal)
        p.setFont(f)
        tr = QRectF(pt.x() - 42, pt.y() + r + 1, 84, 14)
        p.drawText(tr, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, label)


class AgentPage(QWidget):
    key = "agent"
    title = "任务工作台"
    subtitle = "用自然语言下达巡检任务，编排与执行过程实时可见"

    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.worker: AgentWorker | None = None
        self.n_steps = 0
        self.n_tools = 0
        self.task_images: list[dict] = []
        self._stage = 0
        self._detect_done = 0
        self._segment_done = 0
        self._delegates: dict[int, QTreeWidgetItem] = {}
        self.current_task_dir: Path | None = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = PageHeader(self.key, self.title, self.subtitle)
        self.badge = Badge("待命", demo=True)
        bar = ToolBar()
        bar.add("导出轨迹", self.export_trace)
        bar.add("清空", self.clear)
        bar.addWidget(self.badge)
        head.add_action(bar)
        root.addWidget(head)

        body = QHBoxLayout()
        body.setContentsMargins(16, 12, 16, 12)
        body.setSpacing(12)
        root.addLayout(body, 1)

        body.addWidget(self._col_task(), 0)
        body.addWidget(self._col_trace(), 1)
        body.addWidget(self._col_side(), 0)

    # -- 左栏：下达任务与本次统计 ---------------------------------------
    def _col_task(self) -> QWidget:
        col = QWidget()
        col.setFixedWidth(350)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # 任务影像不只是一个文件选择器：它明确告诉总控“本次该处理哪一批输入”。
        # 导入后即落入系统影像库，执行时只把这些 image_id 传给智能体内核。
        assets = Panel("本次巡检任务")
        task_row = QHBoxLayout()
        self.btn_add_images = QPushButton("导入影像")
        self.btn_add_images.setProperty("accent", "primary")
        self.btn_add_images.clicked.connect(self.add_task_images)
        self.btn_clear_images = QPushButton("清空")
        self.btn_clear_images.setProperty("accent", "ghost")
        self.btn_clear_images.clicked.connect(self.clear_task_images)
        task_row.addWidget(self.btn_add_images, 1)
        task_row.addWidget(self.btn_clear_images)
        assets.body().addLayout(task_row)
        self.task_files = QListWidget()
        self.task_files.setFixedHeight(82)
        self.task_files.setToolTip("本次任务关联的影像；检测与量化只处理这些文件")
        assets.body().addWidget(self.task_files)
        self.task_scope = Hint("尚未关联影像。可先导入影像，或直接对历史数据下达任务。")
        assets.body().addWidget(self.task_scope)
        lay.addWidget(assets)

        box = Panel("下达任务")
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("例如：对本次任务进行自动巡检并生成报告")
        self.input.setFixedHeight(74)
        box.body().addWidget(self.input)

        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(10)
        self.btn_run = QPushButton("执行")
        self.btn_run.setProperty("accent", "primary")
        self.btn_run.setMinimumWidth(132)
        self.btn_run.setMinimumHeight(38)
        self.btn_run.clicked.connect(self.execute)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setMinimumWidth(132)
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        row.addWidget(self.btn_run, 1)
        row.addWidget(self.btn_stop, 1)
        box.body().addLayout(row)

        box.body().addWidget(Hint("示例指令"))
        for t in EXAMPLES:
            b = QPushButton(t)
            b.setProperty("accent", "ghost")
            b.setStyleSheet("text-align: left; padding: 5px 9px;")
            b.setToolTip("点击填入输入框")
            b.clicked.connect(lambda _=False, s=t: (self.input.setPlainText(s),
                                                    self.execute()))
            box.body().addWidget(b)
        lay.addWidget(box)

        lay.addWidget(self._stats_panel(), 1)
        return col

    # -- 中栏：执行轨迹 --------------------------------------------------
    def _col_trace(self) -> QWidget:
        box = Panel("执行轨迹")
        self.trace_hint = Hint("")
        box.add_head_widget(self.trace_hint)

        # 事件树给技术复核用；进度条给现场人员快速判断“现在做到哪一步”。
        self.progress = QProgressBar()
        self.progress.setRange(0, 6)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(12)
        self.progress_label = Hint("待命：等待创建巡检任务")
        box.body().addWidget(self.progress)
        box.body().addWidget(self.progress_label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["步", "智能体", "类型", "内容", "耗时"])
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setUniformRowHeights(True)
        head = self.tree.header()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        head.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(0, 38)
        self.tree.setColumnWidth(1, 88)
        self.tree.setColumnWidth(2, 46)
        self.tree.setColumnWidth(4, 58)
        box.body().addWidget(self.tree, 1)
        return box

    # -- 右栏：拓扑与产物 -------------------------------------------------
    def _col_side(self) -> QWidget:
        col = QWidget()
        col.setFixedWidth(300)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        box = Panel("智能体协同")
        self.graph = AgentGraph()
        box.body().addWidget(self.graph)
        lay.addWidget(box)

        box3 = Panel("本次任务产物")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_open_task_dir = QPushButton("打开任务文件夹")
        self.btn_open_task_dir.setProperty("accent", "ghost")
        self.btn_open_task_dir.clicked.connect(self.open_task_dir)
        self.btn_export_task_dir = QPushButton("一键导出任务包")
        self.btn_export_task_dir.setProperty("accent", "primary")
        self.btn_export_task_dir.clicked.connect(self.export_task_dir)
        self.btn_open_task_dir.setEnabled(False)
        self.btn_export_task_dir.setEnabled(False)
        row.addWidget(self.btn_open_task_dir)
        row.addWidget(self.btn_export_task_dir)
        box3.body().addLayout(row)
        self.outputs = QListWidget()
        self.outputs.setMinimumHeight(250)
        self.outputs.itemDoubleClicked.connect(self._open_output)
        box3.body().addWidget(self.outputs, 1)
        box3.body().addWidget(Hint("全部产物会统一归档到本次任务文件夹；双击可打开单个文件。"))
        lay.addWidget(box3, 1)
        return col

    def _stats_panel(self) -> Panel:
        """放在左栏原结论位置，让任务输入与本轮执行数据上下对应。"""
        box = Panel("本次统计")
        grid = QHBoxLayout()
        grid.setSpacing(8)
        self.c_steps = StatCard("执行步数", "—", "步", accent=True)
        self.c_tools = StatCard("工具调用", "—", "次")
        grid.addWidget(self.c_steps)
        grid.addWidget(self.c_tools)
        box.body().addLayout(grid)
        grid2 = QHBoxLayout()
        grid2.setSpacing(8)
        self.c_agents = StatCard("参与智能体", "—", "位")
        self.c_ms = StatCard("总耗时", "—", "ms")
        grid2.addWidget(self.c_agents)
        grid2.addWidget(self.c_ms)
        box.body().addLayout(grid2)
        box.body().addWidget(Hint("统计随真实执行轨迹实时更新。"))
        return box

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    def execute(self) -> None:
        task = self.input.toPlainText().strip()
        if not task:
            self.ctx.toast("请先输入任务", "warn")
            return
        if self.worker is not None and self.worker.isRunning():
            return

        self._reset_run()
        self.badge.set_state("运行中", demo=False)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = AgentWorker(task, self.ctx.settings,
                                  [int(r["id"]) for r in self.task_images], self)
        self.worker.event.connect(self.on_event)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.badge.set_state("已中止", demo=True)
            self.ctx.toast("已请求停止", "warn")

    def _reset_run(self) -> None:
        self.tree.clear()
        self.outputs.clear()
        self.graph.reset()
        self._delegates.clear()
        self.current_task_dir = None
        self.btn_open_task_dir.setEnabled(False)
        self.btn_export_task_dir.setEnabled(False)
        self.n_steps = self.n_tools = 0
        self.c_steps.set_value("—")
        self.c_tools.set_value("—")
        self.c_agents.set_value("—")
        self.c_ms.set_value("—")
        self.trace_hint.setText("")
        self._stage = self._detect_done = self._segment_done = 0
        self.progress.setRange(0, 6)
        self.progress.setValue(0)
        self.progress_label.setText("任务已启动：正在解析指令与巡检范围")

    # -- 本次巡检任务：导入与关联影像 ---------------------------------
    def add_task_images(self) -> None:
        from ..widgets import pick_images

        paths = pick_images(self, multiple=True)
        if not paths:
            return
        result = ingest.ingest(paths)
        added = exists = 0
        for row in result.rows:
            if row.status not in (ingest.ADDED, ingest.EXISTS):
                continue
            image = db.image_by_path(row.path)
            if image is None:
                continue
            rec = dict(image)
            if any(int(x["id"]) == int(rec["id"]) for x in self.task_images):
                continue
            self.task_images.append(rec)
            if row.status == ingest.ADDED:
                added += 1
            else:
                exists += 1
        self._refresh_task_images()
        if self.task_images:
            self.input.setPlainText("对本次任务进行自动巡检并生成报告")
        self.ctx.toast(f"已关联 {added + exists} 张影像（新入库 {added} 张）",
                       "ok" if added + exists else "warn")

    def clear_task_images(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.ctx.toast("任务运行中，不能修改关联影像", "warn")
            return
        self.task_images.clear()
        self._refresh_task_images()

    def _refresh_task_images(self) -> None:
        self.task_files.clear()
        points: set[str] = set()
        batches: set[str] = set()
        for r in self.task_images:
            point = r.get("point_id") or "未识别点位"
            batch = r.get("batch") or "未识别批次"
            points.add(point)
            batches.add(batch)
            item = QListWidgetItem(f"{r.get('filename', '未命名')}  ·  {point}/{batch}")
            item.setToolTip(str(r.get("path") or ""))
            self.task_files.addItem(item)
        if self.task_images:
            self.task_scope.setText(
                f"已关联 {len(self.task_images)} 张影像｜"
                f"{len(points)} 个点位｜批次 {'、'.join(sorted(batches))}")
        else:
            self.task_scope.setText("尚未关联影像。可先导入影像，或直接对历史数据下达任务。")

    def _set_progress(self, stage: int, label: str) -> None:
        self._stage = max(self._stage, stage)
        self.progress.setValue(self._stage)
        self.progress_label.setText(f"阶段 {self._stage}/6：{label}")

    # -- 事件落到界面 ----------------------------------------------------
    def on_event(self, ev) -> None:
        """每条事件一行。主线程执行（信号是队列连接）。"""
        self.n_steps += 1
        if ev.kind == "tool_call":
            self.n_tools += 1
        if ev.agent:
            self.graph.mark(ev.agent)

        # 只由真实工具调用/返回推进，绝不做与执行脱钩的定时动画。
        tool = (ev.data or {}).get("tool", "")
        if ev.kind == "artifact" and (ev.data or {}).get("kind") == "任务归档目录":
            path = Path((ev.data or {}).get("path") or "")
            if path.is_dir():
                self.current_task_dir = path
                self.btn_open_task_dir.setEnabled(True)
                self.btn_export_task_dir.setEnabled(True)
        if tool == "import_images" or (ev.kind == "decision" and "影像已入库" in ev.title):
            self._set_progress(1, "影像已入库，任务范围已锁定")
        elif tool == "detect_image":
            if ev.kind == "tool_result":
                self._detect_done += 1
            total = len(self.task_images) or "多张"
            self._set_progress(2, f"病害检测 {self._detect_done}/{total} 张")
        elif tool == "segment_defect":
            if ev.kind == "tool_result":
                self._segment_done += 1
            total = len(self.task_images) or "多张"
            self._set_progress(3, f"裂缝分割量化 {self._segment_done}/{total} 张")
        elif tool in ("query_timeseries", "analyze_trend", "find_anomalies", "grade_defect"):
            self._set_progress(4, "正在比对历史记录并进行趋势研判")
        elif tool == "plan_reflight":
            self._set_progress(5, "正在审计采集质量并生成复飞建议")
        elif tool == "make_report":
            self._set_progress(6, "正在装配报告与任务产物")

        cn, ckey = KIND_CN.get(ev.kind, (ev.kind, "text_dim"))
        item = QTreeWidgetItem([
            str(ev.seq), ev.agent_cn, cn,
            ev.title + (f"　{ev.detail}" if ev.detail and ev.kind != "decision" else ""),
            f"{ev.duration_ms} ms" if ev.duration_ms else "",
        ])
        item.setForeground(1, QColor(ev.color))
        item.setForeground(2, QColor(C.get(ckey, C["text_dim"])))
        item.setToolTip(3, ev.detail or ev.title)

        # 两级委派树：专家的事件挂在它所属的 delegate 节点下
        parent = self._delegates.get(ev.parent) if ev.parent else None
        if ev.kind == "delegate" and parent is None:
            self.tree.addTopLevelItem(item)
            self._delegates[ev.seq] = item
            item.setExpanded(True)
            f = item.font(2)
            f.setBold(True)
            item.setFont(2, f)
        elif parent is not None:
            parent.addChild(item)
            parent.setExpanded(True)
        else:
            self.tree.addTopLevelItem(item)

        self.tree.scrollToItem(item)

        self.c_steps.set_value(str(self.n_steps))
        self.c_tools.set_value(str(self.n_tools))
        self.c_agents.set_value(str(len(self.graph.seen)))
        if ev.duration_ms:
            self.c_ms.set_value(str(ev.duration_ms))
        self.trace_hint.setText(f"共 {self.n_steps} 步")

    def on_done(self, payload) -> None:
        answer, artifacts, ms, degraded = payload
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.c_ms.set_value(str(ms) if ms else "—")
        self.badge.set_state("降级运行" if degraded else "已完成",
                             demo=bool(degraded))
        if self._stage:
            # 对“只查趋势”这类小任务，不能硬画成跑满六个阶段；完成时把进度条
            # 收束为本轮实际启用的阶段数，既完整也不暗示未执行的检测/报告已完成。
            self.progress.setRange(0, self._stage)
            self.progress.setValue(self._stage)
            self.progress_label.setText(
                f"任务完成：已执行 {self._stage} 个分析阶段，结果与产物已汇总")

        for a in artifacts or []:
            path = a.get("path", "")
            it = QListWidgetItem(f"{a.get('kind', '产物')}　{Path(path).name}")
            it.setToolTip(path)
            it.setData(Qt.ItemDataRole.UserRole, path)
            self.outputs.addItem(it)

        if degraded:
            self.ctx.toast("在线规划未完成，已降级到本地规则引擎", "warn")
        else:
            self.ctx.toast(f"任务完成，用时 {ms} ms", "ok")

    # -- 产物 ------------------------------------------------------------
    def _open_output(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole) or ""
        if not path or not Path(path).exists():
            self.ctx.toast("产物文件不存在", "warn")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)                          # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except OSError as exc:
            self.ctx.toast(f"打不开：{exc}", "danger")

    def open_task_dir(self) -> None:
        path = self.current_task_dir
        if path is None or not path.is_dir():
            self.ctx.toast("本次任务归档文件夹不存在", "warn")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))                    # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            self.ctx.toast(f"打不开任务文件夹：{exc}", "danger")

    def export_task_dir(self) -> None:
        """将当前任务归档目录压缩成一个 ZIP，便于一次性交付或上传。"""
        source = self.current_task_dir
        if source is None or not source.is_dir():
            self.ctx.toast("请先完成一次带归档的任务", "warn")
            return
        suggested = str(source.parent / f"{source.name}.zip")
        target, _ = QFileDialog.getSaveFileName(self, "一键导出本次任务包", suggested,
                                                 "ZIP 压缩包 (*.zip)")
        if not target:
            return
        target_path = Path(target)
        if target_path.suffix.lower() != ".zip":
            target_path = target_path.with_suffix(".zip")
        try:
            if target_path.parent.resolve() == source.resolve():
                self.ctx.toast("导出位置不能位于任务文件夹内部", "warn")
                return
            archive = shutil.make_archive(str(target_path.with_suffix("")), "zip",
                                          root_dir=str(source))
            self.ctx.toast(f"已导出任务包：{Path(archive).name}", "ok")
        except OSError as exc:
            self.ctx.toast(f"导出任务包失败：{exc}", "danger")

    def export_trace(self) -> None:
        if not self.tree.topLevelItemCount():
            self.ctx.toast("还没有可导出的轨迹", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出执行轨迹", str(cfg.OUT_DIR / "智能体执行轨迹.csv"),
            "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            import csv
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["步", "智能体", "类型", "内容", "耗时"])

                def walk(item, depth=0):
                    w.writerow(["  " * depth + item.text(0), item.text(1),
                                item.text(2), item.text(3), item.text(4)])
                    for i in range(item.childCount()):
                        walk(item.child(i), depth + 1)

                for i in range(self.tree.topLevelItemCount()):
                    walk(self.tree.topLevelItem(i))
            self.ctx.toast(f"已导出到 {Path(path).name}", "ok")
        except OSError as exc:
            self.ctx.toast(f"导出失败：{exc}", "danger")

    def clear(self) -> None:
        self._reset_run()
        self.input.clear()
        self.badge.set_state("待命", demo=True)

    def on_show(self) -> None:
        # 别的页面（如长周期监测）点"完整轨迹"跳过来时会把任务文本捎在 ctx 上。
        # 取走即清空，否则下次从侧边栏切回来会莫名其妙又填一次。
        pending = getattr(self.ctx, "agent_task", "")
        if pending:
            self.ctx.agent_task = ""
            self.input.setPlainText(pending)
            self.input.setFocus()
