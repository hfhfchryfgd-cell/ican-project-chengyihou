"""分割与量化后端。

本模块把分割掩膜换算成工程口径的数值（说明书 §6.2）：裂缝长度由掩膜的面积与周长
反解，最大宽度取距离变换的最大内切圆直径，另给平均宽度、面积占比与裂缝数量。
两个主指标都与裂缝在画面里的倾角无关——同一道缝换个拍摄方向量出来是同一个数。
像素→物理量的换算依赖拍摄距离推出的 GSD，与说明书里航点规划时的拍摄距离一脉相承：
距离变了，同一道裂缝的测量值才仍然可比。

分割**只面向裂缝类**（`config.SEGMENT_CLASSES`）：剥落、露筋、蜂窝麻面等区域性病害
只在检测阶段输出边界框与置信度，不做像素级面积分割。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import config as cfg
from .backend import Detection

ROI_MARGIN = 0.15   # 检测框宽高各外扩 15%，确保整条裂缝被完整包住（说明书 §6.2）


def length_from_area_perimeter(area_px: float, perim_px: float) -> float:
    """由掩膜的面积与周长反解裂缝长度（像素）。

    不数细化骨架的像素。中轴骨架在非 0°/45°/90° 的倾角下会退化成带大量冗余像素的
    锯齿链——实测 30° 的直裂缝，238 个骨架点里有 236 个是三度点、微小三角环上百个，
    数像素或数链码都会把长度算到真值的一倍以上；先剪毛刺再去阶梯，规则越修越复杂，
    仍压不住。改用细长形状的闭式关系：把裂缝看成长 L、宽 w 的细长条，则

        面积 A = L·w       周长 P = 2(L + w)

    消去 w 得 2L² − P·L + 2A = 0，取大根 **L = (P + √(P² − 16A)) / 4**。
    实测 0°~90° 各倾角下与真值相差 ±8% 以内（0°/45°/90° 优于 1%），
    比数骨架像素稳得多，且与倾角基本无关。

    判别式为负说明形状并不细长（近圆、近方），此时退化成 L = P/4，仍是有界的
    合理值——本函数只作用于裂缝类掩膜，走到这一支说明掩膜本身不像裂缝。
    """
    if area_px <= 0:
        return 0.0
    disc = perim_px * perim_px - 16.0 * area_px
    return (perim_px + float(np.sqrt(disc))) / 4.0 if disc > 0 else perim_px / 4.0


def segmentable(cls_key: str) -> bool:
    """该类病害是否做像素级分割。口径见 `config.SEGMENT_CLASSES`。"""
    return cls_key in cfg.SEGMENT_CLASSES


def expand_roi(xyxy: tuple[float, float, float, float],
               w: int, h: int, margin: float = ROI_MARGIN) -> tuple[int, int, int, int]:
    """把检测框按宽高各外扩 margin 倍再裁进画面，返回整数 ROI。"""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - bw * margin, x2 + bw * margin
    y1, y2 = y1 - bh * margin, y2 + bh * margin
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    return x1, y1, max(x2, x1 + 4), max(y2, y1 + 4)


@dataclass
class SegmentResult:
    """单个病害实例的分割与量化结果。"""
    cls_key: str
    cls_name: str
    mask: np.ndarray               # 与输入图同尺寸的二值掩码
    px_area: int
    area_cm2: float
    perimeter_px: float
    max_width_mm: float
    conf: float
    gsd_mm_per_px: float
    length_mm: float = 0.0         # 裂缝长度（面积-周长反解）
    avg_width_mm: float = 0.0      # 平均宽度 = 面积 ÷ 长度
    area_ratio: float = 0.0        # 掩膜像素占整幅画面的比例
    crack_count: int = 0           # 掩膜内的连通裂缝条数

    @property
    def max_width_px(self) -> float:
        return self.max_width_mm / self.gsd_mm_per_px if self.gsd_mm_per_px else 0.0


def segment_fields(res: SegmentResult) -> dict:
    """把一次分割结果摊成 `db.add_segment` 的字段（掩膜除外）。

    落库的地方有四处（演示数据播种、桌面批量检测、桌面手工分割、网页批量检测），
    全都调这一份。各写各的话，将来再加一个量化指标就要改四处——漏掉的那一处该列
    恒为 0，查询不报错、界面也不报错，只有趋势图上少一条线，看不出是漏了还是本来
    就没有。`gsd_mm_per_px` 不在其中：它不是持久化字段，由库内结果按拍摄距离现算。
    """
    return {
        "cls_key": res.cls_key, "cls_name": res.cls_name,
        "px_area": res.px_area, "area_cm2": res.area_cm2,
        "perimeter_px": res.perimeter_px, "max_width_mm": res.max_width_mm,
        "length_mm": res.length_mm, "avg_width_mm": res.avg_width_mm,
        "area_ratio": res.area_ratio, "crack_count": res.crack_count,
        "conf": res.conf,
    }


def quantify(mask: np.ndarray, distance_m: float, conf: float,
             cls_key: str) -> SegmentResult:
    """把掩码换算成工程量。distance_m 为该图对应航点的拍摄距离。

    长度用面积-周长反解（见 `length_from_area_perimeter`），最大宽度用距离变换的
    最大内切圆直径——这两个口径都不依赖裂缝的倾角，同一道缝换个拍摄方向量出来
    是同一个数。平均宽度再由「面积 ÷ 长度」回推，与长度自洽：长度×平均宽度==面积。
    """
    dc = cfg.DISEASE_BY_KEY.get(cls_key, cfg.DISEASES[0])
    gsd = cfg.gsd_mm_per_px(distance_m) if distance_m > 0 else 0.0

    m = (mask > 0).astype(np.uint8)
    px_area = int(m.sum())
    mh, mw = m.shape[:2]

    contours, _ = cv2.findContours(m * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours)) if contours else 0.0

    max_w_px = length_px = 0.0
    n_crack = 0
    if px_area > 0:
        # 最大宽度：距离变换的最大内切圆半径×2，比外接矩形宽度更贴近工程定义
        max_w_px = float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max()) * 2.0
        # 长度：由面积与周长反解（口径见 length_from_area_perimeter），与倾角无关
        length_px = length_from_area_perimeter(px_area, perimeter)
        n_crack = max(0, cv2.connectedComponents(m, 8)[0] - 1)

    px2mm = gsd
    return SegmentResult(
        cls_key=cls_key,
        cls_name=dc.name,
        mask=m * 255,
        px_area=px_area,
        area_cm2=round(px_area * (px2mm ** 2) / 100.0, 2) if px2mm else 0.0,
        perimeter_px=round(perimeter, 1),
        max_width_mm=round(max_w_px * px2mm, 2) if px2mm else 0.0,
        conf=conf,
        gsd_mm_per_px=round(gsd, 4),
        length_mm=round(length_px * px2mm, 2) if px2mm else 0.0,
        # 平均宽度按「面积 ÷ 长度」回推，与长度口径自洽：裂缝是细长条，
        # 长度×平均宽度应当还原出面积。
        avg_width_mm=(round(px_area * (px2mm ** 2) / (length_px * px2mm), 2)
                      if px2mm and length_px > 0 else 0.0),
        area_ratio=round(px_area / float(mw * mh), 6) if mw and mh else 0.0,
        crack_count=n_crack,
    )


# --------------------------------------------------------------------------
class SegmentBackend(ABC):
    """分割后端基类。"""

    name = "未知分割后端"
    mode = "demo"
    detail = ""

    @abstractmethod
    def segment(self, bgr: np.ndarray, det: Detection,
                distance_m: float) -> SegmentResult | None:
        """对给定的检测框做分割并量化。"""

    def describe(self) -> str:
        return f"{self.name} · {'演示模式' if self.mode == 'demo' else '模型推理'}{self.detail}"


class DemoSegmenter(SegmentBackend):
    """演示分割后端。

    在检测框内用自适应阈值 + 形态学提取暗色连通域作为裂缝区域，再按长宽比与面积做
    一次筛选，避免把整块阴影当成裂缝。只处理裂缝类。同样是确定性的。
    """

    name = "内置演示分割"
    mode = "demo"
    detail = "（未加载权重，结果为阈值分割，非模型推理）"

    def segment(self, bgr: np.ndarray, det: Detection,
                distance_m: float) -> SegmentResult | None:
        if bgr is None or bgr.size == 0:
            return None

        # 只做裂缝类。剥落、露筋、蜂窝麻面这些区域性病害的外接尺度是几十毫米的
        # 斑块尺度，换不成有工程含义的"裂缝长度/宽度"，硬做只会给出一堆噪声数字。
        if not segmentable(det.cls_key):
            return None

        # 演示影像：直接取真值掩膜。量化面积必须与画面上真实长出来的病害一致，
        # 否则长周期趋势里的"面积增长"只是阈值抖动的产物，讲不通。
        if det.gt_idx > 0:
            gt = self._gt_mask(bgr, det.gt_idx)
            if gt is not None:
                return quantify(gt, distance_m, det.conf, det.cls_key)

        h, w = bgr.shape[:2]
        # 检测框外扩 15%：裂缝两端常常探出框外，贴着框裁会把长度截短一截。
        x1, y1, x2, y2 = expand_roi(det.xyxy, w, h)
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)

        block = max(15, (min(roi.shape[:2]) // 2) * 2 + 1)
        binimg = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, block, 9)

        # 裂缝细长，用矩形小核做闭运算即可把断线连起来；圆核会把细缝填平。
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, k, iterations=2)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(binimg, 8)
        if n <= 1:
            return None

        # 选连通域：裂缝取最细长的（长宽比 × √面积），避免把阴影块当成裂缝
        best_idx, best_score = -1, -1.0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            score = max(bw, bh) / max(1, min(bw, bh)) * np.sqrt(area)
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx < 0:
            return None

        full = np.zeros((h, w), np.uint8)
        full[y1:y2, x1:x2] = np.where(labels == best_idx, 255, 0).astype(np.uint8)

        return quantify(full, distance_m, det.conf, det.cls_key)

    @staticmethod
    def _gt_mask(bgr: np.ndarray, label: int) -> np.ndarray | None:
        """取出演示影像真值掩膜中指定标号的连通区域，返回整幅尺寸的 0/255 掩膜。"""
        from . import demo_data

        gt = demo_data.gt_lookup(bgr)
        if not gt:
            return None
        m = gt["mask"]
        if label > int(m.max()):
            return None
        return np.where(m == label, 255, 0).astype(np.uint8)


class RealSegmenter(SegmentBackend):
    """真实分割后端（加载分割权重）。

    说明书 §6.2 的分割模型是 **U-Net-VC**（U-Net 主干 + VGG16 前 13 层编码器 +
    解码端 CBAM 注意力）。权重就位后启用：模型输出 masks，本类只负责把掩码裁剪回
    原图并交给 quantify() 做量化，因此下游的数值口径与演示后端完全一致，历史数据
    可以无缝衔接。
    """

    mode = "model"

    def __init__(self, weights: str, device: str = "auto", img_size: int = 960):
        self.weights = weights
        self.device = device
        self.img_size = img_size
        self.name = Path(weights).stem
        self.detail = f"（Ultralytics-Seg · {Path(weights).name}）"
        from ultralytics import YOLO

        self._model = YOLO(weights)

    def _device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def segment(self, bgr: np.ndarray, det: Detection,
                distance_m: float) -> SegmentResult | None:
        # 与演示后端同一道门：非裂缝类不做像素级分割。
        if not segmentable(det.cls_key):
            return None
        res = self._model.predict(bgr, imgsz=self.img_size, verbose=False,
                                  device=self._device())[0]
        if res.masks is None or res.boxes is None:
            return None

        from .backend import normalize_class

        names = getattr(res, "names", {}) or {}
        best, best_iou = None, 0.0
        for i, b in enumerate(res.boxes):
            bx1, by1, bx2, by2 = (float(v) for v in b.xyxy[0].tolist())
            iou = _iou(det.xyxy, (bx1, by1, bx2, by2))
            if iou > best_iou:
                best_iou, best = iou, i
        if best is None or best_iou < 0.1:
            return None

        m = res.masks.data[best].cpu().numpy()
        h, w = bgr.shape[:2]
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (m > 0.5).astype(np.uint8) * 255

        raw = str(names.get(int(res.boxes[best].cls[0]), det.cls_key))
        key = normalize_class(raw)
        conf = float(res.boxes[best].conf[0])
        return quantify(mask, distance_m, conf, key)


class CrackUNetSegmenter(SegmentBackend):
    """Adapter for the published crack-seg U-Net `.pth` checkpoint."""

    mode = "model"

    def __init__(self, weights: str, device: str = "auto", **_kw):
        self.weights = weights
        self.name = "crack-seg U-Net v3"
        self.detail = f"（公开裂缝分割权重 · {Path(weights).name}）"
        from .crack_unet import get_runtime
        self._runtime = get_runtime(weights, device)

    def segment(self, bgr: np.ndarray, det: Detection,
                distance_m: float) -> SegmentResult | None:
        if not segmentable(det.cls_key):
            return None
        mask, prob = self._runtime.predict(bgr)
        # The U-Net predicts a whole-image crack mask, while YOLO may return
        # multiple candidate regions.  Limit this instance to its expanded
        # detection ROI so each box contributes one independent measurement
        # instead of duplicating the same full-image mask in the CSV.
        h, w = mask.shape[:2]
        x1, y1, x2, y2 = expand_roi(det.xyxy, w, h)
        instance = np.zeros_like(mask)
        instance[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        if not np.any(instance):
            return None
        confidence = float(prob[instance > 0].mean())
        return quantify(instance, distance_m, round(min(confidence, 0.99), 2), "crack")


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------
def make_segmenter(settings: dict | None = None) -> SegmentBackend:
    """与 make_detector 同构的分割后端工厂。"""
    s = settings or cfg.load_settings()
    from .backend import resolve_weights

    path = resolve_weights("segment_weights", seg=True)
    if path:
        try:
            if Path(path).suffix.lower() == ".pth":
                return CrackUNetSegmenter(path, device=s.get("device", "auto"))
            return RealSegmenter(path, device=s.get("device", "auto"),
                                 img_size=int(s.get("img_size", 960)))
        except Exception as exc:
            demo = DemoSegmenter()
            demo.detail = f"（权重加载失败：{exc}，已回退演示分割）"
            return demo
    return DemoSegmenter()
