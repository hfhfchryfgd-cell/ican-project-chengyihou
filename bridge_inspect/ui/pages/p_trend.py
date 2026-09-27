"""历史检测任务对比与 AI 智能分析。"""

from __future__ import annotations

import html
import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from ... import config as cfg
from ... import db, theme
from ...core import llm, report
from ..context import AppContext
from ..widgets import Badge, ChartView, Hint, HtmlView, Panel, PageHeader, ToolBar, fill_table, make_table, open_html_dialog

C = theme.C


class ComparisonWorker(QThread):
    """在线 API 调用必须离开界面线程；只回传结果，不直接改控件。"""

    progress = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(self, question: str, context: str, settings: dict, parent=None):
        super().__init__(parent)
        self.question = question
        self.context = context
        self.settings = dict(settings or {})

    def run(self) -> None:  # noqa: D102
        self.progress.emit("[3/4] 已提交两次任务的对齐数据，等待 AI 生成研判。")
        result = llm.chat(self.question, self.settings, context=self.context, timeout=60)
        self.done.emit(result)


class TrendPage(QWidget):
    key = "trend"
    title = "长周期监测"
    subtitle = "选择两次真实检测任务进行对比，由已接入的 AI 生成可追溯分析报告"

    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.tasks: list[dict] = []
        self.rows: list[dict] = []
        self.analysis_text = ""
        self.analysis_html = ""
        self.comparison_dir: Path | None = None
        self.artifacts: dict[str, Path] = {}
        self._worker: ComparisonWorker | None = None
        self._build()
        self.refresh()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = PageHeader(self.key, self.title, self.subtitle)
        bar = ToolBar()
        bar.add("AI智能分析", self.start_analysis, accent="primary")
        bar.add("生成报告", self.build_report, accent="cyan")
        bar.add("刷新历史任务", self.refresh)
        head.add_action(bar)
        root.addWidget(head)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)
        root.addWidget(body, 1)

        choose = Panel("选择历史检测任务")
        choose.setFixedHeight(112)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel("基准任务"))
        self.cb_base = QComboBox()
        self.cb_base.setMinimumWidth(300)
        self.cb_base.currentIndexChanged.connect(self._selection_changed)
        row.addWidget(self.cb_base, 1)
        row.addWidget(QLabel("对比任务"))
        self.cb_target = QComboBox()
        self.cb_target.setMinimumWidth(300)
        self.cb_target.currentIndexChanged.connect(self._selection_changed)
        row.addWidget(self.cb_target, 1)
        btn = QPushButton("加载并对比")
        btn.setProperty("accent", "ghost")
        btn.clicked.connect(self.prepare_comparison)
        row.addWidget(btn)
        choose.body().addLayout(row)
        self.lbl_selected = Hint("加载两次实际检测任务后，可提交 AI 进行智能分析。")
        choose.body().addWidget(self.lbl_selected)
        lay.addWidget(choose)

        upper = QHBoxLayout()
        upper.setSpacing(12)
        visual = Panel("对比数据可视化")
        self.chart = ChartView(height=238, toolbar_hint="按点位 × 病害类别比较两次任务的量化面积")
        visual.body().addWidget(self.chart, 1)
        upper.addWidget(visual, 3)

        trace = Panel("分析过程（可追溯）")
        self.txt_trace = QPlainTextEdit()
        self.txt_trace.setReadOnly(True)
        self.txt_trace.setPlainText("等待选择两次历史检测任务。\n\n这里展示数据读取、对齐、AI 调用与产物生成过程；不展示或伪造模型的内部推理。")
        trace.body().addWidget(self.txt_trace, 1)
        upper.addWidget(trace, 2)
        lay.addLayout(upper, 1)

        lower = QHBoxLayout()
        lower.setSpacing(12)
        data = Panel("对比数据预览")
        self.tbl = make_table(
            ["点位", "病害类别", "基准检出", "对比检出", "基准面积(cm²)", "对比面积(cm²)",
             "面积变化", "基准最大宽度(mm)", "对比最大宽度(mm)", "变化说明"],
            stretch_col=9,
        )
        data.body().addWidget(self.tbl, 1)
        data.body().addWidget(Hint("只展示选中两次任务中实际写入的检测与量化数据；无量化值会明确显示为“—”。"))
        lower.addWidget(data, 3)

        result = Panel("AI 分析结果预览")
        badge_row = QHBoxLayout()
        self.badge = Badge("未分析")
        badge_row.addWidget(self.badge)
        badge_row.addStretch(1)
        self.btn_open_folder = QPushButton("打开分析资料文件夹")
        self.btn_open_folder.setProperty("accent", "ghost")
        self.btn_open_folder.clicked.connect(self.open_comparison_folder)
        self.btn_open_folder.setEnabled(False)
        badge_row.addWidget(self.btn_open_folder)
        result.body().addLayout(badge_row)
        self.preview = HtmlView()
        self.preview.setHtml("<p>选择任务后，点击“AI智能分析”生成详细对比结论。</p>")
        result.body().addWidget(self.preview, 1)
        self.lbl_artifacts = Hint("分析完成后将生成对比 CSV、HTML 报告和可视化图片。")
        result.body().addWidget(self.lbl_artifacts)
        lower.addWidget(result, 2)
        lay.addLayout(lower, 1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        previous_base = self.cb_base.currentData()
        previous_target = self.cb_target.currentData()
        self.tasks = [dict(row) for row in db.list_detection_tasks()]
        for combo, previous in ((self.cb_base, previous_base), (self.cb_target, previous_target)):
            combo.blockSignals(True)
            combo.clear()
            for task in self.tasks:
                label = (f"#{task['id']}  {task['name']}  ·  "
                         f"{(task.get('detected_at') or '')[:16]}  ·  "
                         f"{task.get('detected_image_count') or 0} 张 / "
                         f"{task.get('detection_count') or 0} 处")
                combo.addItem(label, task["id"])
            if previous is not None:
                i = combo.findData(previous)
                if i >= 0:
                    combo.setCurrentIndex(i)
            combo.blockSignals(False)

        if len(self.tasks) >= 2:
            if self.cb_base.currentIndex() < 0:
                self.cb_base.setCurrentIndex(1)
            if self.cb_target.currentIndex() < 0:
                self.cb_target.setCurrentIndex(0)
            # 两个下拉框首次填充时都会默认选中第 0 项；默认给出一组不同任务，
            # 用户仍可自行改选，改成同一项时会明确提示不能比较。
            if self.cb_base.currentData() == self.cb_target.currentData():
                self.cb_base.blockSignals(True)
                self.cb_base.setCurrentIndex(1)
                self.cb_base.blockSignals(False)
            self.prepare_comparison()
        else:
            self.rows = []
            self.tbl.setRowCount(0)
            self.chart.empty_hint("至少需要两次实际检测任务才能进行对比")
            self.lbl_selected.setText("当前实际检测任务不足两次，请先完成两次批量检测或批量分割任务。")

    def _selection_changed(self, _index: int) -> None:
        if self.cb_base.count() >= 2 and self.cb_target.count() >= 2:
            self.prepare_comparison()

    def _task(self, combo: QComboBox) -> dict | None:
        task_id = combo.currentData()
        return next((task for task in self.tasks if task["id"] == task_id), None)

    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate(task_id: int) -> dict[tuple[str, str], dict]:
        groups: dict[tuple[str, str], dict] = {}
        for det in db.detections_of_task(task_id):
            key = (det["point_id"] or "未标注点位", det["cls_key"])
            group = groups.setdefault(key, {
                "point": key[0], "cls_key": key[1], "cls_name": det["cls_name"],
                "detections": 0, "area": 0.0, "length": 0.0, "max_width": 0.0,
                "segments": 0,
            })
            group["detections"] += 1
        for seg in db.segments_of_task(task_id):
            key = (seg["point_id"] or "未标注点位", seg["cls_key"])
            group = groups.setdefault(key, {
                "point": key[0], "cls_key": key[1], "cls_name": seg["cls_name"],
                "detections": 0, "area": 0.0, "length": 0.0, "max_width": 0.0,
                "segments": 0,
            })
            group["area"] += float(seg["area_cm2"] or 0)
            group["length"] += float(seg["length_mm"] or 0)
            group["max_width"] = max(group["max_width"], float(seg["max_width_mm"] or 0))
            group["segments"] += 1
        return groups

    def prepare_comparison(self) -> bool:
        base, target = self._task(self.cb_base), self._task(self.cb_target)
        if base is None or target is None:
            return False
        if base["id"] == target["id"]:
            self.rows = []
            self.tbl.setRowCount(0)
            self.chart.empty_hint("请选择两次不同的历史检测任务")
            self.lbl_selected.setText("基准任务和对比任务不能相同。")
            return False

        base_data, target_data = self._aggregate(base["id"]), self._aggregate(target["id"])
        rows: list[dict] = []
        for key in sorted(set(base_data) | set(target_data)):
            before = base_data.get(key, {})
            after = target_data.get(key, {})
            old_area, new_area = float(before.get("area", 0)), float(after.get("area", 0))
            old_count, new_count = int(before.get("detections", 0)), int(after.get("detections", 0))
            if key not in base_data:
                note = "对比任务新增"
            elif key not in target_data:
                note = "对比任务未检出"
            elif new_area > old_area + 1e-6:
                note = "量化面积增加"
            elif new_area < old_area - 1e-6:
                note = "量化面积减少"
            elif new_count != old_count:
                note = "检出数量变化"
            else:
                note = "数据基本一致"
            rows.append({
                "point": key[0], "cls_key": key[1],
                "cls_name": after.get("cls_name") or before.get("cls_name") or key[1],
                "base_count": old_count, "target_count": new_count,
                "base_area": old_area, "target_area": new_area,
                "area_delta": new_area - old_area,
                "base_width": float(before.get("max_width", 0)),
                "target_width": float(after.get("max_width", 0)),
                "base_length": float(before.get("length", 0)),
                "target_length": float(after.get("length", 0)),
                "note": note,
            })
        self.rows = rows
        self._fill_table()
        self._draw_chart()
        self.analysis_text = ""
        self.analysis_html = ""
        self.preview.setHtml("<p>数据已对齐。点击“AI智能分析”生成详细对比报告。</p>")
        self.badge.set_state("数据已就绪", demo=True)
        self.lbl_selected.setText(
            f"基准：{base['name']}（{base.get('detected_at') or '—'}）  →  "
            f"对比：{target['name']}（{target.get('detected_at') or '—'}）"
        )
        self.txt_trace.setPlainText(
            f"[1/4] 已读取基准任务 #{base['id']}：{base.get('detected_image_count') or 0} 张影像、"
            f"{base.get('detection_count') or 0} 处检出。\n"
            f"[2/4] 已读取对比任务 #{target['id']}：{target.get('detected_image_count') or 0} 张影像、"
            f"{target.get('detection_count') or 0} 处检出。\n"
            f"[2/4] 已按点位 × 病害类别对齐 {len(rows)} 组真实记录，等待提交 AI。"
        )
        return True

    def _fill_table(self) -> None:
        fill_table(self.tbl, [
            (row["point"], row["cls_name"], row["base_count"], row["target_count"],
             self._number(row["base_area"]), self._number(row["target_area"]),
             self._signed(row["area_delta"]), self._number(row["base_width"]),
             self._number(row["target_width"]), row["note"])
            for row in self.rows
        ], color_of=lambda row: C["danger"] if row[9] in ("对比任务新增", "量化面积增加") else None)

    def _draw_chart(self) -> None:
        usable = [row for row in self.rows if row["base_area"] or row["target_area"]]
        if not usable:
            self.chart.empty_hint("所选任务没有可比较的裂缝量化面积；下方仍保留实际检测记录")
            return
        usable.sort(key=lambda row: abs(row["area_delta"]), reverse=True)
        usable = usable[:6]
        base, target = self._task(self.cb_base), self._task(self.cb_target)
        labels = [f"{row['point']}\n{row['cls_name']}" for row in usable]
        self.chart.plot_lines(labels, {
            f"基准 #{base['id']}": [row["base_area"] for row in usable],
            f"对比 #{target['id']}": [row["target_area"] for row in usable],
        }, xlabel="点位 / 病害类别", ylabel="量化面积 (cm²)")

    @staticmethod
    def _number(value: float) -> str:
        return "—" if abs(float(value)) < 1e-9 else f"{float(value):.2f}"

    @staticmethod
    def _signed(value: float) -> str:
        return "—" if abs(float(value)) < 1e-9 else f"{float(value):+.2f}"

    # ------------------------------------------------------------------
    def _context(self, base: dict, target: dict) -> str:
        lines = [
            "本次仅比较以下两次真实检测任务，所有数值均来自数据库，未提供的数据不得补造。",
            f"基准任务：#{base['id']}，{base['name']}，检测时间 {base.get('detected_at') or '—'}。",
            f"对比任务：#{target['id']}，{target['name']}，检测时间 {target.get('detected_at') or '—'}。",
            "字段：点位,病害类别,基准检出数,对比检出数,基准面积cm2,对比面积cm2,面积变化cm2,基准最大宽度mm,对比最大宽度mm,变化说明",
        ]
        lines.extend(
            f"{row['point']},{row['cls_name']},{row['base_count']},{row['target_count']},"
            f"{row['base_area']:.4f},{row['target_area']:.4f},{row['area_delta']:.4f},"
            f"{row['base_width']:.4f},{row['target_width']:.4f},{row['note']}"
            for row in self.rows
        )
        return "\n".join(lines)

    def start_analysis(self) -> None:
        if not self.prepare_comparison():
            self.toast("请先选择两次不同的实际检测任务", "warn")
            return
        if not self.rows:
            self.toast("两次任务没有可对齐的检测或量化记录", "warn")
            return
        if self._worker is not None and self._worker.isRunning():
            self.toast("上一轮 AI 智能分析仍在进行中", "warn")
            return
        base, target = self._task(self.cb_base), self._task(self.cb_target)
        question = (
            "请依据给出的两次真实检测任务数据，生成一份详细的桥梁病害对比分析。"
            "按“核心结论、逐项变化、风险判断、养护建议、数据局限”五部分作答；"
            "必须说明新增/消失记录和缺少量化值，禁止编造未提供的数值。"
        )
        self.txt_trace.appendPlainText("[3/4] 正在提交对齐后的检测与量化数据至已配置 AI。")
        self.badge.set_state("AI 智能分析中", demo=False)
        self.preview.setHtml("<p>正在等待 AI 智能分析结果…</p>")
        self.cb_base.setEnabled(False)
        self.cb_target.setEnabled(False)
        self._worker = ComparisonWorker(question, self._context(base, target), self.ctx.settings, self)
        self._worker.progress.connect(self.txt_trace.appendPlainText)
        self._worker.done.connect(self._analysis_done)
        self._worker.start()

    def _analysis_done(self, result: llm.LLMResult) -> None:
        self.cb_base.setEnabled(True)
        self.cb_target.setEnabled(True)
        base, target = self._task(self.cb_base), self._task(self.cb_target)
        if base is None or target is None:
            return
        if result.offline:
            self.analysis_text = self._local_analysis(base, target)
            self.badge.set_state("离线数据分析（AI 未连接）", demo=True)
            self.txt_trace.appendPlainText(f"[3/4] AI 未连接：{result.error or '未配置'}。已按真实数据差异生成离线结果。")
        else:
            self.analysis_text = result.text or "API 未返回可用分析文本。"
            self.badge.set_state(f"AI 智能分析完成 · {result.model}", demo=False)
            self.txt_trace.appendPlainText(f"[3/4] AI 已返回分析结果，模型：{result.model or '已配置模型'}。")
        self._write_analysis_artifacts(base, target)
        self.txt_trace.appendPlainText("[4/4] 已生成对比 CSV、HTML 报告与可视化图片，可在资料文件夹中查看。")
        self.toast("AI 智能分析完成" if not result.offline else "AI 未连接，已生成离线数据对比", "ok" if not result.offline else "warn")

    def _local_analysis(self, base: dict, target: dict) -> str:
        changed = [row for row in self.rows if row["note"] != "数据基本一致"]
        lines = [
            "【离线数据对比结果】",
            f"本次比较基准任务 #{base['id']} 与对比任务 #{target['id']}，共对齐 {len(self.rows)} 组点位×病害记录。",
            f"其中 {len(changed)} 组存在检出或量化变化；AI 未配置或未连通时，本结论仅按实际数据差异生成。",
        ]
        for row in sorted(changed, key=lambda item: abs(item["area_delta"]), reverse=True)[:8]:
            lines.append(
                f"- {row['point']} {row['cls_name']}：检出 {row['base_count']}→{row['target_count']}，"
                f"面积 {self._number(row['base_area'])}→{self._number(row['target_area'])} cm²，"
                f"变化 {self._signed(row['area_delta'])} cm²（{row['note']}）。"
            )
        lines.append("建议：以本表的新增记录、面积增加项和最大宽度变化项为复核重点；无量化值的类别仅作检出数量比较。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _write_analysis_artifacts(self, base: dict, target: dict) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = cfg.OUT_DIR / "历史任务对比" / f"任务对比_{base['id']}_{target['id']}_{stamp}"
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = report.export_csv([{
            "点位": row["point"], "病害类别": row["cls_name"],
            "基准检出数": row["base_count"], "对比检出数": row["target_count"],
            "基准面积(cm²)": round(row["base_area"], 4), "对比面积(cm²)": round(row["target_area"], 4),
            "面积变化(cm²)": round(row["area_delta"], 4),
            "基准最大宽度(mm)": round(row["base_width"], 4), "对比最大宽度(mm)": round(row["target_width"], 4),
            "变化说明": row["note"],
        } for row in self.rows], folder / "任务对比数据.csv")
        image_path: Path | None = folder / "任务对比可视化.png"
        if self.chart.fig is not None:
            self.chart.fig.savefig(image_path, dpi=160, bbox_inches="tight")
        else:
            image_path = None
        self.analysis_html = self._comparison_html(base, target, image_path)
        html_path = folder / "历史任务对比分析报告.html"
        html_path.write_text(self.analysis_html, encoding="utf-8")
        self.preview.setHtml(self.analysis_html)
        self.comparison_dir = folder
        self.artifacts = {"csv": Path(csv_path), "html": html_path}
        if image_path:
            self.artifacts["image"] = image_path
        for kind, path in (("历史任务对比数据CSV", csv_path), ("历史任务对比分析报告", html_path)):
            db.add_output(None, kind, str(path))
        if image_path:
            db.add_output(None, "历史任务对比可视化", str(image_path))
        self.btn_open_folder.setEnabled(True)
        names = "、".join(path.name for path in self.artifacts.values())
        self.lbl_artifacts.setText(f"已生成：{names}；目录：{folder}")

    def _comparison_html(self, base: dict, target: dict, image_path: Path | None) -> str:
        trs = "".join(
            "<tr>"
            f"<td>{html.escape(str(row['point']))}</td><td>{html.escape(str(row['cls_name']))}</td>"
            f"<td>{row['base_count']} → {row['target_count']}</td>"
            f"<td>{self._number(row['base_area'])} → {self._number(row['target_area'])}</td>"
            f"<td>{self._signed(row['area_delta'])}</td><td>{html.escape(row['note'])}</td>"
            "</tr>"
            for row in self.rows
        )
        origin = "AI 智能分析" if not self.badge.text().startswith("离线") else "离线数据分析（AI 未连接）"
        img = (f"<p>可视化图片：{html.escape(image_path.name)}</p>" if image_path else "")
        return f"""
        <html><body style='font-family:Microsoft YaHei,Arial; color:#172033; padding:12px;'>
        <h2>历史检测任务对比分析报告</h2>
        <p><b>基准任务：</b>#{base['id']} {html.escape(base['name'])}<br>
        <b>对比任务：</b>#{target['id']} {html.escape(target['name'])}<br>
        <b>分析来源：</b>{origin}</p>
        <h3>智能分析结论</h3>
        <div style='white-space:pre-wrap; line-height:1.65'>{html.escape(self.analysis_text)}</div>
        <h3>数据对比明细</h3>
        <table border='1' cellspacing='0' cellpadding='5' style='border-collapse:collapse; width:100%; font-size:12px;'>
        <tr><th>点位</th><th>病害类别</th><th>检出数</th><th>面积(cm²)</th><th>面积变化</th><th>说明</th></tr>
        {trs}</table>{img}</body></html>"""

    # ------------------------------------------------------------------
    def build_report(self) -> None:
        """保留原有正式巡检报告入口，并将当前对比分析写入研判章节。"""
        try:
            md_path, html_path, report_no = report.build_and_save(
                self.ctx.settings, self.analysis_text)
            html_text = html_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.toast(f"报告生成失败：{exc}", "error")
            return
        db.add_output(None, "巡检报告Markdown", str(md_path))
        db.add_output(None, "巡检报告HTML", str(html_path))
        open_html_dialog(self, html_text, f"巡检报告 {report_no}", html_path)
        self.toast(f"巡检报告 {report_no} 已生成", "ok")

    def open_comparison_folder(self) -> None:
        if self.comparison_dir is None or not self.comparison_dir.exists():
            self.toast("尚未生成对比资料文件夹", "warn")
            return
        os.startfile(str(self.comparison_dir))  # noqa: S606 -- Windows 桌面端打开用户刚生成的资料

    def toast(self, text: str, kind: str = "info") -> None:
        self.ctx.toast(text, kind)

    def on_show(self) -> None:
        self.refresh()
