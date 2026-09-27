"""裂缝分割量化：单张或批量任务的掩膜预览与真实量化数据。"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QProgressDialog, QPushButton, QVBoxLayout,
    QWidget,
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


class SegmentPage(QWidget):
    key = "segment"
    title = "分割量化"
    subtitle = ""

    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.dets: list = []
        self.results: list = []
        self.active_record: dict = {}
        self.batch_images: list[dict] = []
        self.batch_results: dict[int, list] = {}
        self.batch_dets: dict[int, list] = {}
        self.batch_index = -1
        self.result_rows: list[tuple[dict, object]] = []
        self.mode = "overlay"
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
        bar.add("开始分割", self.run_segment, accent="cyan")
        bar.add("批量导入并分割", self.run_batch)
        bar.add("导出当前掩膜", self.export_mask)
        bar.add("导出数据", self.export_data)
        bar.add("清除", self.clear_all, accent="danger")
        head.add_action(bar)
        self.badge = Badge("演示分割")
        head.add_action(self.badge)
        root.addWidget(head)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)
        root.addWidget(body, 1)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.c_n = StatCard("量化区域", "0", "处", accent=True)
        self.c_area = StatCard("病害总面积", "—", "cm²")
        self.c_len = StatCard("裂缝总长", "—", "mm")
        self.c_width = StatCard("最大宽度", "—", "mm")
        self.c_gsd = StatCard("地面分辨率 GSD", "—", "mm/px")
        self.cards = (self.c_n, self.c_area, self.c_len, self.c_width, self.c_gsd)
        for card in self.cards:
            cards.addWidget(card)
        lay.addLayout(cards)

        left = Panel("影像 / 掩膜")
        ctl = QHBoxLayout()
        ctl.setSpacing(6)
        self.mode_btns = {}
        for key, name in (("src", "原图"), ("mask", "掩膜"), ("overlay", "叠加")):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setChecked(key == self.mode)
            button.setMinimumWidth(56)
            button.clicked.connect(lambda _, k=key: self.set_mode(k))
            self.mode_btns[key] = button
            ctl.addWidget(button)
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
        self.lbl_hint = Hint("可打开单张图片，或点击“批量导入并分割”选择多张图片。")
        left.body().addWidget(self.lbl_hint)

        right = Panel("数据预览")
        self.tbl = make_table(
            ["影像文件", "点位", "病害类别", "像素面积", "面积(cm²)", "裂缝长度(mm)",
             "平均宽度(mm)", "最大宽度(mm)", "面积占比", "裂缝数", "置信度"],
            stretch_col=0,
        )
        self.tbl.itemSelectionChanged.connect(self._on_pick_row)
        right.body().addWidget(self.tbl, 1)
        right.body().addWidget(Hint("表格仅显示当前单张或当前批量任务的实际分割量化数据；点击行可切换到对应影像。"))

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
        segmenter = self.ctx.segmenter
        self.badge.set_state("演示分割" if segmenter.mode == "demo" else segmenter.name,
                             segmenter.mode == "demo")

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
                self.toast("仅查看：这张影像未入库，分割结果不会保存", "warn")
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
        self.batch_dets = {}
        self.batch_index = -1
        self.active_record = self._current_record()
        self.dets = self._load_detections(self.active_record.get("id"))
        self.results = []
        self.result_rows = []
        self.view.set_image(self.ctx.bgr, self.active_record.get("filename") or "")
        self.tbl.setRowCount(0)
        self._set_cards([])
        self._set_preview_controls()
        self.lbl_hint.setText(
            "该影像尚未入库，结果无法保存" if self.active_record.get("id") is None
            else f"已载入影像，数据库中已有 {len(self.dets)} 条检测记录。"
        )

    def _current_record(self) -> dict:
        meta = self.ctx.current.get("meta") or {}
        path = str(self.ctx.current.get("path") or "")
        return {
            "id": self.ctx.current.get("image_id"), "path": path,
            "filename": Path(path).name if path else "当前影像",
            "point_id": meta.get("point_id") or "", "batch": meta.get("batch") or "",
            "shoot_distance": meta.get("shoot_distance") or 2.0,
            "captured_at": meta.get("captured_at") or "",
        }

    def _load_detections(self, image_id) -> list:
        if not image_id:
            return []
        from ...core.backend import Detection

        return [Detection(
            cls_key=row["cls_key"], cls_name=row["cls_name"],
            cls_id=cfg.DISEASE_BY_KEY.get(row["cls_key"]).id
            if cfg.DISEASE_BY_KEY.get(row["cls_key"]) else 0,
            conf=row["conf"], x1=row["x1"], y1=row["y1"], x2=row["x2"], y2=row["y2"],
            source="stored", gt_idx=row["gt_idx"] or 0,
        ) for row in db.detections_of_image(image_id)]

    def run_segment(self) -> None:
        bgr = self.ctx.bgr
        if bgr is None:
            self.toast("请先打开一张影像", "warn")
            return
        if not self.dets:
            self.toast(
                "该影像还没有检测记录，请先到“目标检测”执行检测" if self.active_record.get("id")
                else "这张影像还没有入库，请先到“智能体工作台”点击“导入影像”",
                "warn",
            )
            return
        self.results = self._segment_image(bgr, self.dets, self.active_record, task_id=None)
        self.result_rows = [(self.active_record, result) for result in self.results]
        self._render_active()
        self._fill_table()
        self._set_cards(self.result_rows)
        self.lbl_hint.setText(f"分割后端：{self.ctx.segmenter.describe()}")
        self.toast(f"分割完成，量化出 {len(self.results)} 处病害区域", "ok")

    # -- 批量导入、检测、分割 -------------------------------------------
    def run_batch(self) -> None:
        paths = pick_images(self, multiple=True)
        if not paths:
            return
        imported = ingest.ingest(paths)
        images: list[dict] = []
        for item in imported.rows:
            if item.status not in (ingest.ADDED, ingest.EXISTS):
                continue
            row = db.image_by_path(item.path)
            if row is not None:
                images.append(dict(row))
        images = list({int(image["id"]): image for image in images}.values())
        if not images:
            self.toast("没有可分割的影像", "warn")
            return
        self._run_batch(images, imported.summary())

    def _run_batch(self, images: list[dict], import_summary: str) -> None:
        dlg = QProgressDialog("批量分割中…", "取消", 0, len(images), self)
        dlg.setWindowTitle("批量导入并分割")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumWidth(420)
        detector = self.ctx.detector
        task_id = db.create_task(
            name=f"批量分割 {datetime.now():%m-%d %H:%M}", kind="批量分割",
            point_scope="本次导入影像", image_count=len(images),
            owner=self.ctx.user.get("username", ""),
        )
        results: dict[int, list] = {}
        dets_by_image: dict[int, list] = {}
        done: list[dict] = []
        t0 = time.perf_counter()
        for i, image in enumerate(images):
            if dlg.wasCanceled():
                break
            dlg.setValue(i)
            dlg.setLabelText(f"正在分割：{image['filename']}")
            bgr = imgutil.imread(image["path"])
            if bgr is None:
                continue
            dets = detector.predict(bgr)
            self._store_detections(dets, image, task_id)
            dets_by_image[int(image["id"])] = dets
            results[int(image["id"])] = self._segment_image(bgr, dets, image, task_id)
            done.append(image)
        dlg.setValue(len(images))
        self.batch_images = done
        self.batch_results = results
        self.batch_dets = dets_by_image
        self.batch_index = 0 if done else -1
        self.result_rows = [
            (image, result) for image in done for result in results.get(int(image["id"]), [])
        ]
        self._fill_table()
        self._set_cards(self.result_rows)
        if done:
            self._show_batch(0)
        else:
            self.view.clear_image()
            self._set_preview_controls()
        elapsed = time.perf_counter() - t0
        self.lbl_hint.setText(
            f"{import_summary}；已完成 {len(done)} 张影像的检测与裂缝分割量化。"
        )
        self.toast(
            f"批量分割完成：{len(done)} 张影像、{len(self.result_rows)} 条量化记录，耗时 {elapsed:.1f} s",
            "ok",
        )

    def _store_detections(self, dets: list, image: dict, task_id: int) -> None:
        db.execute("DELETE FROM detections WHERE image_id=?", (image["id"],))
        if not dets:
            return
        db.add_detections([{
            "image_id": image["id"], "task_id": task_id,
            "point_id": image.get("point_id") or "", "cls_key": det.cls_key,
            "cls_name": det.cls_name, "conf": det.conf,
            "x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2,
            "gt_idx": det.gt_idx,
        } for det in dets])

    def _segment_image(self, bgr, dets: list, record: dict, task_id: int | None) -> list:
        db.execute("DELETE FROM segments WHERE image_id=?", (record.get("id"),))
        results = []
        for det in dets:
            result = self.ctx.segmenter.segment(bgr, det, record.get("shoot_distance") or 2.0)
            if not result or result.px_area <= 0:
                continue
            results.append(result)
            if record.get("id"):
                db.add_segment(
                    image_id=record["id"], task_id=task_id,
                    point_id=record.get("point_id") or "", batch=record.get("batch") or "",
                    captured_at=record.get("captured_at") or "", **segment_fields(result),
                )
        return results

    # -- 预览与数据表 ----------------------------------------------------
    def set_mode(self, key: str) -> None:
        self.mode = key
        for name, button in self.mode_btns.items():
            button.setChecked(name == key)
        self._render_active()

    def _show_batch(self, index: int) -> None:
        if not (0 <= index < len(self.batch_images)):
            return
        self.batch_index = index
        self.active_record = self.batch_images[index]
        self.dets = self.batch_dets.get(int(self.active_record["id"]), [])
        self.results = self.batch_results.get(int(self.active_record["id"]), [])
        self._render_active()
        self._set_preview_controls()

    def _set_preview_controls(self) -> None:
        total = len(self.batch_images)
        self.btn_prev.setEnabled(total > 1 and self.batch_index > 0)
        self.btn_next.setEnabled(total > 1 and self.batch_index < total - 1)
        self.lbl_batch.setText(f"第 {self.batch_index + 1}/{total} 张" if total else "单张预览")

    def _render_active(self) -> None:
        if not self.active_record:
            return
        bgr = imgutil.imread(self.active_record.get("path") or "")
        if bgr is None:
            self.view.clear_image()
            return
        prefix = (f"第 {self.batch_index + 1}/{len(self.batch_images)} 张 · "
                  if self.batch_images else "")
        self.view.set_image(self._compose(bgr, self.results),
                            prefix + str(self.active_record.get("filename") or ""))

    def _compose(self, bgr, results: list):
        if self.mode == "src" or not results:
            return bgr
        color = np.zeros_like(bgr)
        any_mask = np.zeros(bgr.shape[:2], np.uint8)
        for result in results:
            blue, green, red = imgutil.hex_to_bgr(cfg.disease_color(result.cls_key))
            color[result.mask > 0] = (blue, green, red)
            any_mask[result.mask > 0] = 255
        if self.mode == "mask":
            canvas = np.full_like(bgr, 12)
            canvas[any_mask > 0] = color[any_mask > 0]
            return canvas
        import cv2

        output = bgr.copy()
        index = any_mask > 0
        output[index] = cv2.addWeighted(bgr, 0.45, color, 0.55, 0)[index]
        contours, _ = cv2.findContours(any_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 255, 255), 1)
        return output

    def _fill_table(self) -> None:
        rows = [
            (record.get("filename") or "", record.get("point_id") or "—", result.cls_name,
             result.px_area, f"{result.area_cm2:.2f}", f"{result.length_mm:.1f}",
             f"{result.avg_width_mm:.2f}", f"{result.max_width_mm:.2f}",
             f"{result.area_ratio * 100:.2f}%", result.crack_count, f"{result.conf:.3f}")
            for record, result in self.result_rows
        ]
        fill_table(
            self.tbl, rows,
            color_of=lambda row: cfg.disease_color(next(
                result.cls_key for record, result in self.result_rows
                if record.get("filename") == row[0] and result.cls_name == row[2]
            )),
        )

    def _set_cards(self, entries: list[tuple[dict, object]]) -> None:
        results = [result for _, result in entries]
        self.c_n.set_value(len(results), "处")
        self.c_area.set_value(f"{sum(result.area_cm2 for result in results):.2f}" if results else "—", "cm²")
        self.c_len.set_value(f"{sum(result.length_mm for result in results):.1f}" if results else "—", "mm")
        self.c_width.set_value(f"{max((result.max_width_mm for result in results), default=0):.2f}" if results else "—", "mm")
        self.c_gsd.set_value(f"{results[0].gsd_mm_per_px:.3f}" if results else "—", "mm/px")

    def _on_pick_row(self) -> None:
        row = self.tbl.currentRow()
        if not (0 <= row < len(self.result_rows)):
            return
        record, result = self.result_rows[row]
        if self.batch_images:
            for index, image in enumerate(self.batch_images):
                if int(image["id"]) == int(record["id"]):
                    self._show_batch(index)
                    break
        self.toast(
            f"{record.get('filename', '')} · {result.cls_name}　面积 {result.area_cm2:.2f} cm²　"
            f"最大宽度 {result.max_width_mm:.2f} mm",
            "info",
        )

    # -- 导出 ------------------------------------------------------------
    def export_mask(self) -> None:
        if not self.active_record or not self.results:
            self.toast("请先完成一次分割", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前掩膜", str(cfg.OUT_DIR / "mask.png"), "PNG 图片 (*.png)",
        )
        if not path:
            return
        bgr = imgutil.imread(self.active_record.get("path") or "")
        if bgr is None:
            self.toast("当前影像不存在", "error")
            return
        original = self.mode
        self.mode = "mask"
        image = self._compose(bgr, self.results)
        self.mode = original
        imgutil.imwrite(path, image)
        db.add_output(self.active_record.get("id"), "分割掩膜", path)
        self.toast(f"已导出掩膜：{Path(path).name}", "ok")

    def export_data(self) -> None:
        if not self.result_rows:
            self.toast("请先完成一次分割", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出量化数据", str(cfg.OUT_DIR / "量化数据.csv"), "CSV 文件 (*.csv)",
        )
        if not path:
            return
        rows = [{
            "影像文件": record.get("filename") or "", "点位编号": record.get("point_id") or "",
            "病害类别": result.cls_name, "像素面积(px)": result.px_area,
            "病害面积(cm²)": round(result.area_cm2, 4),
            "裂缝长度(mm)": round(result.length_mm, 3),
            "平均宽度(mm)": round(result.avg_width_mm, 4),
            "最大宽度(mm)": round(result.max_width_mm, 4),
            "面积占比": round(result.area_ratio, 6), "裂缝条数": result.crack_count,
            "地面分辨率(mm/px)": round(result.gsd_mm_per_px, 5), "置信度": result.conf,
        } for record, result in self.result_rows]
        export_csv(rows, path)
        db.add_output(self.active_record.get("id"), "量化CSV", path)
        self.toast(f"已导出 {len(rows)} 条量化记录：{Path(path).name}", "ok")

    def clear_all(self) -> None:
        self.dets = []
        self.results = []
        self.active_record = {}
        self.batch_images = []
        self.batch_results = {}
        self.batch_dets = {}
        self.batch_index = -1
        self.result_rows = []
        self.tbl.setRowCount(0)
        self.view.clear_image()
        self._set_cards([])
        self._set_preview_controls()
        self.ctx.clear_current()
        self.lbl_hint.setText("可打开单张图片，或点击“批量导入并分割”选择多张图片。")
        self.toast("已清除当前量化结果", "ok")

    def toast(self, text: str, kind: str = "info") -> None:
        self.ctx.toast(text, kind)

    def on_show(self) -> None:
        self._sync_badge()
