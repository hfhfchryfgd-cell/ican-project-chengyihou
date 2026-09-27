"""⑤ 检测历史。

对应《作品说明书》§6.5「检测历史」：查看、筛选历史检测任务，右侧给出任务摘要与产物文件列表。
任务来自批量检测与报告生成，产物（标注图片 / 掩膜 / CSV / 报告）落在 outputs 表里，
因此"这个任务当时导出了哪些文件"是可追溯的。

报告生成不单列页面（新文档 §6.1、§6.5 点名的八个功能视图里没有它）：智能体工作台
按任务拆解调起，这里提供同一个 `core/report.py` 的手工入口，两边的产物用同一种 kind
落库，产物列表也就把它们一起筛得出来。「全部产物」子标签跨任务列出，含 task_id 为空
的临时产物——只看"选中任务的产物"时，这些文件在界面上是找不到的。
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import (
    QDateEdit, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QWidget,
)

from ... import config as cfg
from ... import db, theme
from ...core.report import build_and_save, export_csv
from ..context import AppContext
from ..widgets import (
    Hint, KeyValue, Panel, PageHeader, StatCard, SubTabs, ToolBar, fill_table,
    make_table, open_html_dialog,
)

C = theme.C


class HistoryPage(QWidget):
    key = "history"
    title = "检测历史"
    subtitle = "仅显示实际完成检测的任务；可按检测时间筛选并导出资料文件夹"

    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.tasks: list = []
        self.outs: list = []          # 选中任务的产物
        self.all_outs: list = []       # 全部产物（含未归属任务的）
        self._all_tab = -1             # 「全部产物」页签下标，建页签时定下
        self._build()
        self.refresh()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = PageHeader(self.key, self.title, self.subtitle)
        bar = ToolBar()
        bar.add("刷新", self.refresh, accent="primary")
        bar.add("生成报告", self.make_report,
                tooltip="按当前库内量化记录生成 Markdown + HTML 巡检报告，并立即预览")
        bar.add("预览报告", self.preview_output,
                tooltip="在系统内以白底 A4 版式查看选中的 HTML 报告"
                        "（可打印/另存 PDF）")
        bar.add("重新加载", self.reload_current)
        bar.add("导出历史资料文件夹", self.export_history_folder)
        bar.add("打开产物文件", self.open_output,
                tooltip="交给系统默认程序打开：打印、另存 PDF、用浏览器看都走这条路")
        bar.add("删除任务", self.delete_task, accent="danger")
        head.add_action(bar)
        root.addWidget(head)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)
        root.addWidget(body, 1)

        # 筛选
        filt = Panel("筛选")
        filt.setMinimumHeight(88)
        row = QHBoxLayout()
        row.setContentsMargins(2, 3, 2, 3)
        row.setSpacing(12)
        keyword_label = QLabel("关键词")
        keyword_label.setMinimumWidth(46)
        row.addWidget(keyword_label)
        self.in_kw = QLineEdit()
        self.in_kw.setPlaceholderText("任务名称")
        self.in_kw.setMinimumWidth(220)
        self.in_kw.setFixedHeight(38)
        self.in_kw.returnPressed.connect(self.refresh)
        row.addWidget(self.in_kw)
        row.addSpacing(12)
        date_label = QLabel("检测时间")
        date_label.setMinimumWidth(62)
        row.addWidget(date_label)
        self.date_from = QDateEdit(QDate(2000, 1, 1))
        self.date_from.setCalendarPopup(True)
        self.date_from.setDisplayFormat("yyyy-MM-dd")
        self.date_from.setMinimumWidth(138)
        self.date_from.setFixedHeight(38)
        self.date_from.dateChanged.connect(lambda _: self.refresh())
        row.addWidget(self.date_from)
        to_label = QLabel("至")
        to_label.setMinimumWidth(16)
        row.addWidget(to_label)
        self.date_to = QDateEdit(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.setDisplayFormat("yyyy-MM-dd")
        self.date_to.setMinimumWidth(138)
        self.date_to.setFixedHeight(38)
        self.date_to.dateChanged.connect(lambda _: self.refresh())
        row.addWidget(self.date_to)
        btn_all_dates = QPushButton("全部时间")
        btn_all_dates.setProperty("accent", "ghost")
        btn_all_dates.setMinimumWidth(92)
        btn_all_dates.setMinimumHeight(38)
        btn_all_dates.clicked.connect(self.reset_date_filter)
        row.addWidget(btn_all_dates)
        row.addStretch(1)
        filt.body().addLayout(row)
        lay.addWidget(filt)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.c_tasks = StatCard("实际检测次数", "0", "次", accent=True)
        self.c_imgs = StatCard("实际处理影像", "0", "张")
        self.c_recs = StatCard("检出病害", "0", "处")
        self.c_files = StatCard("关联产物", "0", "个")
        for c in (self.c_tasks, self.c_imgs, self.c_recs, self.c_files):
            cards.addWidget(c)
        lay.addLayout(cards)

        left = Panel("任务列表")
        self.tbl = make_table(
            ["任务名称", "类型", "实际影像数", "检出数", "检测时间"],
            stretch_col=0)
        self.tbl.itemSelectionChanged.connect(self._on_pick)
        left.body().addWidget(self.tbl, 1)

        right = Panel("任务详情")
        self.tabs = SubTabs()
        self.tabs.addTab(self._tab_summary(), "任务摘要")
        self.tabs.addTab(self._tab_files(), "产物文件")
        self._all_tab = self.tabs.addTab(self._tab_all_outputs(), "全部产物")
        right.body().addWidget(self.tabs, 1)

        sp = QHBoxLayout()
        sp.setSpacing(12)
        sp.addWidget(left, 3)
        sp.addWidget(right, 2)
        lay.addLayout(sp, 1)

    def _tab_summary(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 8, 0, 0)
        self.kv = KeyValue(columns=1)
        lay.addWidget(self.kv)
        lay.addStretch(1)
        self.kv.clear()
        self.kv.add("提示", "选择左侧任务查看摘要")
        return w

    def _tab_files(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)
        self.tbl_out = make_table(["类型", "文件名", "生成时间"], stretch_col=1)
        self.tbl_out.itemSelectionChanged.connect(self.open_output)
        lay.addWidget(self.tbl_out, 1)
        lay.addWidget(Hint("选中一行即可用系统默认程序打开该文件"))
        return w

    def _tab_all_outputs(self) -> QWidget:
        """全部产物：跨任务，含 task_id 为空的行。

        智能体临时出的报告可能没归到任何任务下，这类文件只在"选中任务的产物"
        里翻是找不到的。
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)
        self.tbl_all = make_table(
            ["类型", "文件名", "所属任务", "生成时间"], stretch_col=1)
        self.tbl_all.itemSelectionChanged.connect(self.open_output)
        lay.addWidget(self.tbl_all, 1)
        lay.addWidget(Hint("跨任务列出全部产物，「所属任务」为空的即未归属任务的临时产物"))
        return w

    # -- 数据 ------------------------------------------------------------
    def refresh(self) -> None:
        self.tasks = [dict(r) for r in db.list_detection_tasks(
            keyword=self.in_kw.text().strip(),
            date_from=self.date_from.date().toString("yyyy-MM-dd"),
            date_to=self.date_to.date().toString("yyyy-MM-dd"),
        )]
        fill_table(self.tbl, [
            (t["name"], t["kind"], t["detected_image_count"],
             t["detection_count"], (t["detected_at"] or "")[:16])
            for t in self.tasks
        ])
        self._draw_all_outputs()
        self.c_tasks.set_value(len(self.tasks), "次")
        self.c_imgs.set_value(sum(int(t["detected_image_count"] or 0) for t in self.tasks), "张")
        self.c_recs.set_value(sum(int(t["detection_count"] or 0) for t in self.tasks), "处")
        self.c_files.set_value(sum(int(t["output_count"] or 0) for t in self.tasks), "个")
        # 默认选中第一条，右侧摘要不至于空着
        if self.tbl.rowCount() and self.tbl.currentRow() < 0:
            self.tbl.selectRow(0)

    def reset_date_filter(self) -> None:
        self.date_from.blockSignals(True)
        self.date_to.blockSignals(True)
        self.date_from.setDate(QDate(2000, 1, 1))
        self.date_to.setDate(QDate.currentDate())
        self.date_from.blockSignals(False)
        self.date_to.blockSignals(False)
        self.refresh()

    def _draw_all_outputs(self) -> None:
        """灌「全部产物」表。任务名在表格里显示比显示 id 好认，故做一次映射。"""
        self.all_outs = [dict(r) for r in db.list_outputs(limit=300)]
        name_of = {r["id"]: r["name"] for r in db.list_tasks(limit=1000)}
        fill_table(self.tbl_all, [
            (o["kind"], Path(o["path"]).name,
             name_of.get(o["task_id"], "未归属任务" if not o["task_id"] else
                         f"任务 #{o['task_id']}（已删除）"),
             (o["created_at"] or "")[:16])
            for o in self.all_outs
        ])

    def make_report(self) -> None:
        """手工生成一份巡检报告。

        与智能体工作台的报告生成同源——都走 `core/report.py`，产物以同一种
        kind 落进 outputs 表，类型筛选因此能把两种来源一起筛出来。
        """
        try:
            md, html, no = build_and_save(cfg.load_settings())
        except Exception as e:                    # noqa: BLE001 —— 生成失败不该拖垮界面
            self.toast(f"报告生成失败：{e}", "error")
            return
        tid = db.create_task(name="巡检报告编制", kind="报告",
                             note="由检测历史页手工生成")
        db.add_output(tid, "巡检报告Markdown", str(md))
        db.add_output(tid, "巡检报告HTML", str(html))
        # 报告是产物，不是检测任务；生成后在「全部产物」中查看即可。
        self.in_kw.clear()
        self.refresh()
        self.tabs.setCurrentIndex(self._all_tab)   # 新报告在最上面
        self.toast(f"报告 {no} 已生成：{Path(md).name}", "ok")
        self.preview_html(Path(html), no)

    # ------------------------------------------------------------------
    def preview_output(self) -> None:
        """预览当前子标签里选中的那条产物，仅对 HTML 报告有意义。"""
        if self.tabs.currentIndex() == self._all_tab:
            tbl, rows = self.tbl_all, self.all_outs
        else:
            tbl, rows = self.tbl_out, self.outs
        r = tbl.currentRow()
        if not (0 <= r < len(rows)):
            self.toast("请先在产物列表里选中一条", "warn")
            return
        self.preview_html(Path(rows[r]["path"]))

    def preview_html(self, path: Path, report_no: str = "") -> None:
        """把一份落盘的 HTML 报告读进模态预览窗。

        Markdown 产物不走这里：`QTextBrowser` 只认 HTML，把 .md 丢进去会原样
        显示井号与竖线。选中 .md 时给出明确提示，而不是弹一个看不懂的窗口。
        """
        if not path.exists():
            self.toast("产物文件已被移动或删除", "error")
            return
        if path.suffix.lower() not in (".html", ".htm"):
            self.toast(f"{path.name} 不是 HTML，无法在系统内预览——"
                       "请用「打开产物文件」交给外部程序", "warn")
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            self.toast(f"报告读取失败：{e}", "error")
            return
        # 没有编号就退回文件名：预览窗标题是给人确认"我现在看的是哪一份"的位置，
        # 留空比印一个文件名更糟。
        open_html_dialog(self, text, f"巡检报告 {report_no or path.stem}", path)

    def reload_current(self) -> None:
        item = self.ctx.current.get("path")
        if not item:
            self.toast("当前没有已载入的影像", "warn")
            return
        from ...core import imgutil

        bgr = imgutil.imread(item)
        if bgr is None:
            self.toast("影像文件已不在原路径", "error")
            return
        row = db.query_one("SELECT * FROM images WHERE path=?", (item,))
        self.ctx.set_current(bgr, item, dict(row) if row else {},
                             row["id"] if row else None)
        self.toast("已重新载入当前影像", "ok")

    def _on_pick(self) -> None:
        r = self.tbl.currentRow()
        if not (0 <= r < len(self.tasks)):
            return
        t = self.tasks[r]
        task_id = t["id"]
        self.outs = [dict(o) for o in db.outputs_of_task(task_id)]
        outs = self.outs
        segs = db.segments_of_task(task_id)
        self.kv.clear()
        self.kv.add("任务名称", t["name"])
        self.kv.add("任务类型", t["kind"])
        self.kv.add("检测时间", t.get("detected_at") or "—")
        self.kv.add("实际处理影像", f"{t.get('detected_image_count') or 0} 张")
        self.kv.add("检出病害", f"{t.get('detection_count') or 0} 处")
        self.kv.add("任务创建人", t["owner"] or "—")
        self.kv.add("任务创建时间", t["created_at"] or "—")
        self.kv.add("量化记录", f"{len(segs)} 条",
                    C["accent"] if segs else None)
        if segs:
            area = sum(float(s["area_cm2"] or 0) for s in segs)
            self.kv.add("病害总面积", f"{area:.2f} cm²")
            self.kv.add("涉及类别", "、".join(sorted({s["cls_name"] for s in segs})))
        self.kv.add("关联产物", f"{len(outs)} 个")

        fill_table(self.tbl_out, [
            (o["kind"], Path(o["path"]).name, (o["created_at"] or "")[:16])
            for o in outs
        ])

    def delete_task(self) -> None:
        r = self.tbl.currentRow()
        if not (0 <= r < len(self.tasks)):
            self.toast("请先选中一个任务", "warn")
            return
        t = self.tasks[r]
        db.delete_task(t["id"])
        self.refresh()
        self.kv.clear()
        self.kv.add("提示", "选择左侧任务查看摘要")
        self.tbl_out.setRowCount(0)
        self.outs = []
        self.toast(f"已删除任务「{t['name']}」及其关联记录", "ok")

    def export_history_folder(self) -> None:
        if not self.tasks:
            self.toast("当前筛选条件下没有实际检测任务可导出", "warn")
            return
        target = QFileDialog.getExistingDirectory(self, "选择历史资料导出位置", str(cfg.OUT_DIR))
        if not target:
            return
        folder = Path(target) / f"检测历史资料_{datetime.now():%Y%m%d_%H%M%S}"
        suffix = 1
        while folder.exists():
            folder = Path(target) / f"检测历史资料_{datetime.now():%Y%m%d_%H%M%S}_{suffix}"
            suffix += 1
        folder.mkdir(parents=True)

        export_csv([{
            "任务名称": t["name"], "任务类型": t["kind"],
            "检测时间": t.get("detected_at") or "",
            "实际处理影像数": t.get("detected_image_count") or 0,
            "检出病害数": t.get("detection_count") or 0,
            "量化记录数": t.get("segment_count") or 0,
            "关联产物数": t.get("output_count") or 0,
            "创建人": t["owner"] or "", "任务创建时间": t["created_at"] or "",
        } for t in self.tasks], folder / "检测任务清单.csv")

        task_ids = [t["id"] for t in self.tasks]
        export_csv([{
            "任务名称": r["task_name"], "任务类型": r["task_kind"],
            "检测时间": r["created_at"], "影像文件": r["filename"] or "",
            "影像路径": r["image_path"] or "", "点位": r["point_id"] or "",
            "批次": r["image_batch"] or "", "病害类别": r["cls_name"],
            "置信度": r["conf"], "X1": r["x1"], "Y1": r["y1"],
            "X2": r["x2"], "Y2": r["y2"],
        } for r in db.detection_rows_of_tasks(task_ids)], folder / "检测明细.csv")

        quant_rows = []
        for task in self.tasks:
            for seg in db.segments_of_task(task["id"]):
                quant_rows.append({
                    "任务名称": task["name"], "影像ID": seg["image_id"],
                    "点位": seg["point_id"], "病害类别": seg["cls_name"],
                    "像素面积": seg["px_area"], "面积(cm²)": seg["area_cm2"],
                    "裂缝长度(mm)": seg["length_mm"], "平均宽度(mm)": seg["avg_width_mm"],
                    "最大宽度(mm)": seg["max_width_mm"], "面积占比": seg["area_ratio"],
                    "裂缝数": seg["crack_count"], "置信度": seg["conf"],
                })
        export_csv(quant_rows, folder / "分割量化明细.csv")

        artifact_dir = folder / "关联产物"
        copied = 0
        for task in self.tasks:
            for output in db.outputs_of_task(task["id"]):
                source = Path(output["path"])
                if not source.is_file():
                    continue
                artifact_dir.mkdir(exist_ok=True)
                shutil.copy2(source, artifact_dir / f"任务{task['id']}_{output['id']}_{source.name}")
                copied += 1
        db.add_output(None, "历史检测资料文件夹", str(folder))
        self._draw_all_outputs()
        self.toast(f"已导出 {len(self.tasks)} 次实际检测、{copied} 个关联文件：{folder}", "ok")

    def open_output(self) -> None:
        """打开当前子标签里选中的那条产物。

        两个子标签各有一张表，按当前页签取对应的那份数据——不能写死取
        `self.tbl_out`，否则在「全部产物」里选中的是第 r 行，打开的可能
        是另一个任务的同名文件。
        """
        if self.tabs.currentIndex() == self._all_tab:
            tbl, rows = self.tbl_all, self.all_outs
        else:
            tbl, rows = self.tbl_out, self.outs
        r = tbl.currentRow()
        if not (0 <= r < len(rows)):
            return
        p = Path(rows[r]["path"])
        if not p.exists():
            self.toast("产物文件已被移动或删除", "error")
            return
        os.startfile(str(p))          # noqa: S606 —— 交给系统默认程序打开

    def toast(self, text: str, kind: str = "info") -> None:
        self.ctx.toast(text, kind)

    def on_show(self) -> None:
        self.refresh()
