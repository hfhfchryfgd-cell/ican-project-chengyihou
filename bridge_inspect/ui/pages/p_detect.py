"""目标检测：单张或批量导入影像后，展示真实推理结果。"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QProgressDialog, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from ... import config as cfg
from ... import db, theme
from ...core import imgutil, ingest
from ...core.report import export_csv
from ...core.segment import segment_fields
from ..context import AppContext
from ..widgets import (
    Badge, Hint, ImageView, PageHeader, Panel, StatCard, ToolBar, ask_ingest,
    fill_table, make_table, pick_images,
)

C = theme.C


class DetectPage(QWidget):
    key = "detect"
    title = "目标检测"
    subtitle = ""

    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.dets: list = []
        self.last_elapsed = 0.0
        self.active_record: dict = {}
        self.batch_images: list[dict] = []
        self.batch_results: dict[int, list] = {}
        self.batch_index = -1
        self.result_rows: list[tuple[dict, object]] = []
        self._build()
        self.ctx.currentChanged.connect(self._on_current_changed)
        self.ctx.backendChanged.connect(self._sync_badge)

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = PageHeader(self.key, self.title, self.subtitle)
        bar = ToolBar()
        bar.add("打开图片", self.open_image, accent="primary")
        bar.add("单张检测", self.detect_single, accent="cyan")
        bar.add("批量导入并检测", self.detect_batch)
        bar.add("导出标注图片", self.export_image)
        bar.add("导出 CSV", self.export_csv_file)
        bar.add("清除", self.clear_all, accent="danger")
        head.add_action(bar)
        self.badge = Badge("演示后端")
        head.add_action(self.badge)
        root.addWidget(head)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)
        root.addWidget(body, 1)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.c_n = StatCard("本次检出", "0", "处", accent=True)
        self.c_cls = StatCard("涉及类别", "0", "类")
        self.c_conf = StatCard("平均置信度", "—", "")
        self.c_ms = StatCard("推理耗时", "—", "ms")
        for card in (self.c_n, self.c_cls, self.c_conf, self.c_ms):
            cards.addWidget(card)
        lay.addLayout(cards)

        left = Panel("检测预览")
        ctl = QHBoxLayout()
        ctl.setSpacing(8)
        ctl.addWidget(QLabel("置信度阈值"))
        self.sl_conf = QSlider(Qt.Orientation.Horizontal)
        self.sl_conf.setRange(5, 95)
        self.sl_conf.setValue(int(self.ctx.settings.get("conf_thres", 0.35) * 100))
        self.sl_conf.setFixedWidth(130)
        self.lbl_conf = QLabel(f"{self.sl_conf.value() / 100:.2f}")
        self.lbl_conf.setObjectName("Mono")
        self.sl_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"{v / 100:.2f}"))
        self.sl_conf.sliderReleased.connect(self._on_conf_changed)
        ctl.addWidget(self.sl_conf)
        ctl.addWidget(self.lbl_conf)
        ctl.addStretch(1)
        self.btn_prev = QPushButton("‹ 上一张")
        self.btn_prev.setProperty("accent", "ghost")
        self.btn_prev.clicked.connect(lambda: self._show_batch(self.batch_index - 1))
        self.lbl_batch = QLabel("单张预览")
        self.lbl_batch.setObjectName("Hint")
        self.btn_next = QPushButton("下一张 ›")
        self.btn_next.setProperty("accent", "ghost")
        self.btn_next.clicked.connect(lambda: self._show_batch(self.batch_index + 1))
        ctl.addWidget(self.btn_prev)
        ctl.addWidget(self.lbl_batch)
        ctl.addWidget(self.btn_next)
        left.head.addWidget(self._row(ctl))
        self.view = ImageView("尚未载入影像")
        left.body().addWidget(self.view, 1)
        self.lbl_hint = Hint("可打开单张图片，或点击“批量导入并检测”选择多张图片。")
        left.body().addWidget(self.lbl_hint)

        right = Panel("检测结果")
        self.tbl_det = make_table(
            ["影像文件", "点位", "#", "病害类别", "置信度", "X1", "Y1", "X2", "Y2", "来源"],
            stretch_col=0,
        )
        self.tbl_det.itemSelectionChanged.connect(self._on_pick_det)
        right.body().addWidget(self.tbl_det, 1)
        right.body().addWidget(Hint("表格仅列出当前单张或当前批量任务的真实检测结果；点击某行可切换到对应影像。"))

        sp = QHBoxLayout()
        sp.setSpacing(12)
        sp.addWidget(left, 3)
        sp.addWidget(right, 2)
        lay.addLayout(sp, 1)

        self._set_preview_controls()
        self._sync_badge()

    @staticmethod
    def _row(lay) -> QWidget:
        w = QWidget()
        lay.setContentsMargins(0, 0, 0, 0)
        w.setLayout(lay)
        return w

    def _sync_badge(self) -> None:
        text, demo = self.ctx.backend_label()
        self.badge.set_state(text, demo)

    # -- 单张影像 -------------------------------------------------------
    def open_image(self) -> None:
        picked = pick_images(self, multiple=False)
        if not picked:
            return
        path = picked[0]
        bgr = imgutil.imread(path)
        if bgr is None:
            self.toast("影像无法解码", "error")
            return
        row = db.query_one("SELECT * FROM images WHERE path=?", (path,))
        if row is None:
            if not ask_ingest(self, Path(path).name):
                self.ctx.set_current(bgr, path, {}, None)
                self.toast("仅查看：这张影像未入库，检测结果不会保存", "warn")
                return
            iid, path, note = ingest.ingest_one(path)
            if iid is None:
                self.toast(f"入库失败：{note}", "error")
                return
            row = db.query_one("SELECT * FROM images WHERE id=?", (iid,))
            bgr = imgutil.imread(path)
            self.toast(f"{note}　已载入", "ok")
        else:
            self.toast(f"已载入 {Path(path).name}", "ok")
        self.ctx.set_current(bgr, path, dict(row) if row else {}, row["id"] if row else None)

    def _on_current_changed(self) -> None:
        if self.ctx.bgr is None:
            return
        self.batch_images = []
        self.batch_results = {}
        self.batch_index = -1
        self.active_record = self._current_record()
        self.dets = []
        self.result_rows = []
        self.view.set_image(self.ctx.bgr, Path(self.active_record.get("path") or "").name)
        self.tbl_det.setRowCount(0)
        self._set_cards([], 0.0)
        self._set_preview_controls()
        self.lbl_hint.setText("已载入影像，点击“单张检测”开始识别。")

    def _current_record(self) -> dict:
        meta = self.ctx.current.get("meta") or {}
        path = str(self.ctx.current.get("path") or "")
        return {
            "id": self.ctx.current.get("image_id"), "path": path,
            "filename": Path(path).name if path else "当前影像",
            "point_id": meta.get("point_id") or "", "batch": meta.get("batch") or "",
            "shoot_distance": meta.get("shoot_distance") or 2.0,
        }

    def detect_single(self, silent: bool = False) -> None:
        bgr = self.ctx.bgr
        if bgr is None:
            self.toast("请先打开一张影像", "warn")
            return
        t0 = time.perf_counter()
        dets = self.ctx.detector.predict(bgr)
        self.last_elapsed = (time.perf_counter() - t0) * 1000
        self.active_record = self._current_record()
        self.dets = dets
        why = self._store(dets)
        self.result_rows = [(self.active_record, d) for d in dets]
        self._show_preview(self.active_record, dets)
        self._fill_result_table()
        self._set_cards(self.result_rows, self.last_elapsed)
        self.lbl_hint.setText(f"后端：{self.ctx.detector.describe()}")
        if not silent:
            msg = (f"检测完成，识别出 {len(dets)} 处病害（{self.last_elapsed:.0f} ms）"
                   + (f"；{why}" if why else ""))
            self.toast(msg, "ok" if not why else "warn")

    # -- 批量导入、检测与预览 -------------------------------------------
    def detect_batch(self) -> None:
        paths = pick_images(self, multiple=True)
        if not paths:
            return
        imported = ingest.ingest(paths)
        selected: list[dict] = []
        for item in imported.rows:
            if item.status not in (ingest.ADDED, ingest.EXISTS):
                continue
            row = db.image_by_path(item.path)
            if row is not None:
                selected.append(dict(row))
        selected = list({int(row["id"]): row for row in selected}.values())
        if not selected:
            self.toast("没有可检测的影像", "warn")
            return
        self._run_batch(selected, imported.summary())

    def _run_batch(self, images: list[dict], import_summary: str) -> None:
        dlg = QProgressDialog("批量检测中…", "取消", 0, len(images), self)
        dlg.setWindowTitle("批量导入并检测")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumWidth(420)
        detector, segmenter = self.ctx.detector, self.ctx.segmenter
        task_id = db.create_task(
            name=f"批量检测 {datetime.now():%m-%d %H:%M}", kind="批量检测",
            point_scope="本次导入影像", image_count=len(images),
            owner=self.ctx.user.get("username", ""),
        )
        results: dict[int, list] = {}
        done_images: list[dict] = []
        t0 = time.perf_counter()
        for i, image in enumerate(images):
            if dlg.wasCanceled():
                break
            dlg.setValue(i)
            dlg.setLabelText(f"正在检测：{image['filename']}")
            bgr = imgutil.imread(image["path"])
            if bgr is None:
                continue
            dets = detector.predict(bgr)
            self._store(dets, task_id=task_id, point_id=image.get("point_id"),
                        image_id=image["id"])
            db.execute("DELETE FROM segments WHERE image_id=?", (image["id"],))
            for det in dets:
                result = segmenter.segment(bgr, det, image.get("shoot_distance") or 2.0)
                if result and result.px_area > 0:
                    db.add_segment(
                        image_id=image["id"], task_id=task_id,
                        point_id=image.get("point_id") or "", batch=image.get("batch") or "",
                        captured_at=image.get("captured_at") or "", **segment_fields(result),
                    )
            results[int(image["id"])] = dets
            done_images.append(image)
        dlg.setValue(len(images))
        self.last_elapsed = (time.perf_counter() - t0) * 1000
        self.batch_images = done_images
        self.batch_results = results
        self.batch_index = 0 if done_images else -1
        self.result_rows = [
            (image, det) for image in done_images for det in results.get(int(image["id"]), [])
        ]
        self._fill_result_table()
        self._set_cards(self.result_rows, self.last_elapsed)
        if done_images:
            self._show_batch(0)
        else:
            self.view.clear_image()
            self._set_preview_controls()
        self.lbl_hint.setText(
            f"{import_summary}；已完成 {len(done_images)} 张影像的真实模型推理。"
        )
        self.toast(
            f"批量检测完成：{len(done_images)} 张影像、{len(self.result_rows)} 处病害，"
            f"耗时 {self.last_elapsed / 1000:.1f} s",
            "ok",
        )

    def _show_batch(self, index: int) -> None:
        if not (0 <= index < len(self.batch_images)):
            return
        self.batch_index = index
        record = self.batch_images[index]
        self.active_record = record
        self.dets = self.batch_results.get(int(record["id"]), [])
        self._show_preview(record, self.dets)
        self._set_preview_controls()

    def _show_preview(self, record: dict, dets: list) -> None:
        bgr = imgutil.imread(record.get("path") or "")
        if bgr is None:
            self.view.clear_image()
            return
        image = imgutil.draw_detections(bgr, dets)
        prefix = (f"第 {self.batch_index + 1}/{len(self.batch_images)} 张 · "
                  if self.batch_images else "")
        self.view.set_image(image, prefix + str(record.get("filename") or ""))

    def _set_preview_controls(self) -> None:
        total = len(self.batch_images)
        self.btn_prev.setEnabled(total > 1 and self.batch_index > 0)
        self.btn_next.setEnabled(total > 1 and self.batch_index < total - 1)
        self.lbl_batch.setText(
            f"第 {self.batch_index + 1}/{total} 张" if total else "单张预览"
        )

    # -- 数据存储与结果表 ------------------------------------------------
    NO_IMAGE = "这张影像还没入库，结果不会保存；请先到“智能体工作台”点击“导入影像”"
    NO_DETS = "没有检出目标，已清除该影像旧检测记录"

    def _store(self, dets, task_id: int | None = None, point_id: str | None = None,
               image_id: int | None = None) -> str:
        image_id = image_id if image_id is not None else self.ctx.current.get("image_id")
        point_id = point_id if point_id is not None else (self.ctx.current.get("meta") or {}).get("point_id")
        if image_id is None:
            return self.NO_IMAGE
        db.execute("DELETE FROM detections WHERE image_id=?", (image_id,))
        if not dets:
            return self.NO_DETS
        db.add_detections([{
            "image_id": image_id, "task_id": task_id, "point_id": point_id or "",
            "cls_key": det.cls_key, "cls_name": det.cls_name, "conf": det.conf,
            "x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2,
            "gt_idx": det.gt_idx,
        } for det in dets])
        return ""

    def _fill_result_table(self) -> None:
        rows = [
            (record.get("filename") or "", record.get("point_id") or "—", index + 1,
             det.cls_name, f"{det.conf:.3f}", det.x1, det.y1, det.x2, det.y2,
             self._source_text(det))
            for index, (record, det) in enumerate(self.result_rows)
        ]
        fill_table(
            self.tbl_det, rows,
            color_of=lambda row: cfg.disease_color(next(
                det.cls_key for record, det in self.result_rows
                if record.get("filename") == row[0] and det.cls_name == row[3]
            )),
        )

    @staticmethod
    def _source_text(det) -> str:
        if det.source == "model":
            return "模型推理"
        if det.source == "demo-gt":
            return "演示标定"
        return det.source or "检测结果"

    def _set_cards(self, entries: list[tuple[dict, object]], elapsed_ms: float) -> None:
        confs = [det.conf for _, det in entries]
        self.c_n.set_value(len(entries), "处")
        self.c_cls.set_value(len({det.cls_key for _, det in entries}), "类")
        self.c_conf.set_value(f"{sum(confs) / len(confs):.3f}" if confs else "—")
        self.c_ms.set_value(f"{elapsed_ms:.0f}" if elapsed_ms else "—", "ms")

    def _on_pick_det(self) -> None:
        row = self.tbl_det.currentRow()
        if not (0 <= row < len(self.result_rows)):
            return
        record, det = self.result_rows[row]
        if self.batch_images:
            for index, image in enumerate(self.batch_images):
                if int(image["id"]) == int(record["id"]):
                    self._show_batch(index)
                    break
        self.toast(
            f"{record.get('filename', '')} · {det.cls_name}　置信度 {det.conf:.3f}　"
            f"外接框 {det.x2 - det.x1}×{det.y2 - det.y1} px",
            "info",
        )

    def _on_conf_changed(self) -> None:
        self.ctx.settings["conf_thres"] = self.sl_conf.value() / 100
        cfg.save_settings(self.ctx.settings)
        self.ctx.reload_backends()
        if self.ctx.bgr is not None and not self.batch_images:
            self.detect_single(silent=True)

    # -- 导出 ------------------------------------------------------------
    def export_image(self) -> None:
        if not self.active_record or not self.dets:
            self.toast("请先完成一次检测", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前标注图片", str(cfg.OUT_DIR / "detected.jpg"),
            "JPEG 图片 (*.jpg);;PNG 图片 (*.png)",
        )
        if not path:
            return
        bgr = imgutil.imread(self.active_record.get("path") or "")
        if bgr is None:
            self.toast("当前影像不存在", "error")
            return
        imgutil.imwrite(path, imgutil.draw_detections(bgr, self.dets))
        db.add_output(self.active_record.get("id"), "标注图片", path)
        self.toast(f"已导出：{Path(path).name}", "ok")

    def export_csv_file(self) -> None:
        if not self.result_rows:
            self.toast("请先完成一次检测", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出检测结果", str(cfg.OUT_DIR / "检测结果.csv"), "CSV 文件 (*.csv)",
        )
        if not path:
            return
        rows = [{
            "影像文件": record.get("filename") or "", "点位编号": record.get("point_id") or "",
            "病害类别": det.cls_name, "置信度": det.conf,
            "X1": det.x1, "Y1": det.y1, "X2": det.x2, "Y2": det.y2,
            "来源": self._source_text(det),
        } for record, det in self.result_rows]
        export_csv(rows, path)
        db.add_output(self.active_record.get("id"), "检测CSV", path)
        self.toast(f"已导出 {len(rows)} 条检测记录：{Path(path).name}", "ok")

    def clear_all(self) -> None:
        self.dets = []
        self.active_record = {}
        self.batch_images = []
        self.batch_results = {}
        self.batch_index = -1
        self.result_rows = []
        self.view.clear_image()
        self.tbl_det.setRowCount(0)
        self._set_cards([], 0.0)
        self._set_preview_controls()
        self.ctx.clear_current()
        self.lbl_hint.setText("可打开单张图片，或点击“批量导入并检测”选择多张图片。")
        self.toast("已清除当前检测结果", "ok")

    def toast(self, text: str, kind: str = "info") -> None:
        self.ctx.toast(text, kind)

    def on_show(self) -> None:
        self._sync_badge()
