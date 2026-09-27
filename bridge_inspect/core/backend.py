"""检测后端抽象层。

页面只依赖 DetectBackend 抽象，不关心结果来自何处。这样在拿到训练好的桥梁病害权重
之后，只需把 .pt / .onnx 放进 models/ 目录，程序会自动切换后端，UI 代码一行不用改。

当前两种实现：
  · DemoDetector —— 演示后端。基于图像本身的边缘/暗通道/纹理统计给出确定性结果，
                    同一张图每次结果完全一致，可复现、可讲解，但它不是模型推理。
  · YoloDetector —— 真实后端。ultralytics 加载 .pt，或 onnxruntime 加载 .onnx。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import config as cfg

# 模型类别名 → 本项目类别 key 的归一化映射。
# 训练时类别名可能是英文、中文或带前缀，这里统一兜住，避免权重一换就全变成"未知"。
_ALIAS = {
    "crack": "crack", "裂缝": "crack", "cracks": "crack", "crack_line": "crack",
    "spalling": "spalling", "剥落": "spalling", "剥落掉块": "spalling",
    "spall": "spalling", "delamination": "spalling",
    "exposed_bar": "exposed_bar", "露筋": "exposed_bar", "rebar": "exposed_bar",
    "exposed_rebar": "exposed_bar", "corrosion": "exposed_bar",
    "seepage": "seepage", "渗水": "seepage", "渗水泛碱": "seepage",
    "efflorescence": "seepage", "water": "seepage", "leak": "seepage",
    "honeycomb": "honeycomb", "蜂窝": "honeycomb", "蜂窝麻面": "honeycomb",
    "void": "honeycomb",
    "bearing": "bearing", "支座滑移": "bearing", "bearing_slip": "bearing",
    "joint_offset": "joint_offset", "伸缩缝错台": "joint_offset",
    "joint": "joint_offset", "expansion_joint": "joint_offset",
}


def normalize_class(name: str) -> str:
    """把任意来源的类别名归一到本项目 7 类，识别不了回退 crack。"""
    k = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIAS.get(k, _ALIAS.get(k.replace("_", ""), "crack"))


def extract_candidates(bgr: np.ndarray, min_area_ratio: float = 0.0004,
                       work_width: int = 640) -> tuple[list[dict], int, int]:
    """从影像中提取候选病害区域及其几何/灰度特征。

    检测器与阈值调优脚本共用这一份实现——如果把特征计算抄成两份，两边迟早会走样。

    掩码策略分三步：

    1) 平场校正。航拍影像普遍带低频照度不均（斜射阳光、构件自身阴影）。直接对原图
       做全局 Otsu，阈值会落在照度梯度中间，把半个画面整片判成"暗区"或"亮区"——
       实测中这种情况会产生占画面 50% 的假病害框。所以先减掉低频背景，把图拉平。

    2) 自适应阈值抓细线。裂缝这类宽度只有几像素的目标，靠局部对比才能检出。

    3) 拉平后的图再做阈值抓大块。自适应阈值对大面积均匀暗区是失效的：块尺寸小于
       病害尺寸时，病害内部的局部均值就等于病害自身灰度，它反而判该处"正常"，只
       在边缘留下一圈环。平场后的图用标准差阈值就能把整块斑区完整切出来。

    返回 (候选特征列表, 工作图宽, 工作图高)。特征含义：
      linearity  真实像素面积 / 最长边²   —— 细线趋近 0，密实斑块趋近 1
      porosity   1 - 真实面积 / 外轮廓面积 —— 多孔疏松区域显著大于 0
      fill       真实像素面积 / 外接框面积
      area_ratio 真实像素面积 / 整图面积
      dev        区域与全图中位灰度的平均偏离 / 64
      elong      外接框宽高比
      src        dark / bright，来自哪张掩码
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (work_width, int(work_width * h / max(w, 1))),
                      interpolation=cv2.INTER_AREA)
    gh, gw = gray.shape

    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    # 平场校正：减掉低频照度分量，结果以 128 为中性基准
    low = cv2.GaussianBlur(eq, (0, 0), 48).astype(np.float32)
    flat = np.clip(eq.astype(np.float32) - low + 128.0, 0, 255).astype(np.uint8)
    spread = max(6.0, float(flat.std()))

    # 暗区：自适应（细线）+ 平场后阈值（大块）
    adapt_dark = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 35, 12)
    block_dark = cv2.threshold(flat, 128.0 - max(9.0, 0.85 * spread), 255,
                               cv2.THRESH_BINARY_INV)[1]
    dark = cv2.bitwise_or(adapt_dark, block_dark)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 亮区：需要比暗区更大的偏离量——影像中的亮结构（露筋反光、析出物）比暗病害稀少
    bright = cv2.threshold(flat, 128.0 + max(13.0, 1.15 * spread), 255,
                           cv2.THRESH_BINARY)[1]
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    min_area = min_area_ratio * gh * gw
    cands: list[dict] = []

    for m, src in ((dark, "dark"), (bright, "bright")):
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            rect_area = bw * bh
            if rect_area < min_area or bw < 8 or bh < 6:
                continue

            # 用连通域真实像素数，而不是外接框面积：蜂窝这类多孔区域的外接框里
            # 有大片空洞，用外接框会严重高估。
            true_area = int(cv2.countNonZero(m[y:y + bh, x:x + bw]))
            if true_area < min_area:
                continue
            contour_area = float(cv2.contourArea(cnt)) or 1.0
            porosity = float(np.clip(1.0 - true_area / max(contour_area, 1.0), 0, 1))

            # 对比度在平场后的图上量，基准固定为 128，不受照度梯度影响
            patch = flat[y:y + bh, x:x + bw].astype(np.float32)
            dev = float(np.abs(patch - 128.0).mean()) / 64.0
            area_ratio = true_area / (gh * gw)
            longest = max(bw, bh, 1)

            # 置信度由"对比度 + 规模"共同决定，两项都做饱和处理。
            # 若只看对比度，纹理噪声区域和真病害一样能拿满分，排序就失去意义，
            # 真正的大面积病害反而排不到前面。
            conf = 0.40 + 0.28 * min(dev / 0.90, 1.0) \
                        + 0.30 * min(area_ratio / 0.015, 1.0)

            cands.append({
                "score": float(np.clip(conf, 0.40, 0.98)),
                "x": x, "y": y, "bw": bw, "bh": bh, "src": src,
                "linearity": true_area / float(longest * longest),
                "porosity": porosity,
                "fill": true_area / rect_area,
                "area_ratio": area_ratio,
                "elong": bw / max(bh, 1),
                "dev": dev,
            })
    return cands, gw, gh


@dataclass
class Detection:
    """统一检测结果。"""
    cls_key: str
    cls_name: str
    cls_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    source: str = "demo"
    gt_idx: int = 0        # >0 表示这条结果来自演示影像的真值，值是掩膜标号

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class DetectBackend(ABC):
    """检测后端基类。"""

    name = "未知后端"
    mode = "demo"          # demo | model
    detail = ""            # 状态栏/角标上的补充说明

    def __init__(self, conf: float = 0.35, iou: float = 0.45,
                 device: str = "auto", img_size: int = 960):
        self.conf = conf
        self.iou = iou
        self.device = device
        self.img_size = img_size

    @abstractmethod
    def predict(self, bgr: np.ndarray) -> list[Detection]:
        """对单张 BGR 图做检测。"""

    def describe(self) -> str:
        return f"{self.name} · {'演示模式' if self.mode == 'demo' else '模型推理'}{self.detail}"


# --------------------------------------------------------------------------
# 演示后端
# --------------------------------------------------------------------------
class DemoDetector(DetectBackend):
    """演示检测后端，对两类输入走两条路。

    1）预置演示影像（有真值边车）：直接按真值给出框与类别，再叠加确定性的框位抖动与
       置信度扰动。演示影像的类别因此不会在不同批次间漂移，长周期趋势才成立。
    2）用户自己导入的照片（无真值）：用 CLAHE + 自适应阈值提取暗纹理与斑块，再按区域
       形态特征分配类别。这条路径只是"看起来合理的猜测"，不代表真实识别能力——界面
       角标与 README 都写明当前为演示后端，加载权重后自动切换为 YoloDetector。
    """

    name = "内置演示后端"
    mode = "demo"
    detail = "（未加载权重；演示影像按真值给出结果，自备照片为图像特征响应）"

    _MIN_AREA_RATIO = 0.00006     # 过滤过小的噪点区域
    _MAX_BOXES = 8

    def predict(self, bgr: np.ndarray) -> list[Detection]:
        if bgr is None or bgr.size == 0:
            return []

        gt = self._gt_predict(bgr)
        if gt is not None:
            return gt
        return self._heuristic(bgr)

    # -- 演示影像：按真值给出结果 ----------------------------------------
    def _gt_predict(self, bgr: np.ndarray) -> list[Detection] | None:
        from . import demo_data

        gt = demo_data.gt_lookup(bgr)
        if not gt:
            return None

        h, w = bgr.shape[:2]
        seed = int(hashlib.md5(demo_data.image_sig(bgr).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        dets: list[Detection] = []
        for i, (key, box) in enumerate(zip(gt["keys"], gt["boxes"]), start=1):
            dc = cfg.DISEASE_BY_KEY.get(key)
            if dc is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            # 让框位与置信度带一点扰动，看起来是模型输出而不是照抄标注
            x1 += int(rng.integers(-5, 6)); y1 += int(rng.integers(-5, 6))
            x2 += int(rng.integers(-5, 6)); y2 += int(rng.integers(-5, 6))
            x1, x2 = int(np.clip(x1, 0, w - 2)), int(np.clip(x2, 2, w))
            y1, y2 = int(np.clip(y1, 0, h - 2)), int(np.clip(y2, 2, h))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            extent = (x2 - x1) * (y2 - y1) / float(w * h)
            conf = 0.70 + 0.26 * min(extent / 0.08, 1.0) ** 0.5 \
                + float(rng.normal(0, 0.015))
            dets.append(Detection(
                cls_key=key, cls_name=dc.name, cls_id=dc.id,
                conf=round(float(np.clip(conf, 0.55, 0.98)), 2),
                x1=x1, y1=y1, x2=x2, y2=y2, source="demo-gt", gt_idx=i,
            ))

        # 这里**不再**补启发式的"疑似目标"来凑画面：那种框没有掩膜、也不是标定实例，
        # 而同一张图在不同批次被补上的数量并不一致，一旦写进 segments 表，同一点位的
        # 面积合计就随批次跳变，趋势曲线会出现假的暴涨/暴跌。画面的丰富度改由合成器
        # 直接多画几处带真值的病害来提供（见 demo_data._add_secondary）。
        dets.sort(key=lambda d: -d.conf)
        return dets

    # -- 任意照片：图像特征启发式 ----------------------------------------
    def _heuristic(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        cands, gw, gh = extract_candidates(bgr, self._MIN_AREA_RATIO)

        # 图像过于平整（例如纯色测试图）时，退化到按图像尺寸布点，保证演示不空屏
        if len(cands) < 3:
            seed = int.from_bytes(
                hashlib.md5(np.ascontiguousarray(bgr[::4, ::4])).digest()[:8], "big")
            cands = self._fallback_candidates(
                np.random.default_rng(seed % (2 ** 32)), gw, gh, 5)

        cands.sort(key=lambda c: -c["score"])
        cands = self._nms(cands, 0.35)[: self._MAX_BOXES]

        floor = min(self.conf, 0.42)   # 演示后端分数分布与真模型不同，放宽下限
        dets: list[Detection] = []
        sx, sy = w / gw, h / gh
        for c in cands:
            if c["score"] < floor:
                continue
            cls_key = self._classify(c)
            dc = cfg.DISEASE_BY_KEY[cls_key]
            dets.append(Detection(
                cls_key=cls_key, cls_name=dc.name, cls_id=dc.id,
                conf=round(c["score"], 2),
                x1=int(c["x"] * sx), y1=int(c["y"] * sy),
                x2=int((c["x"] + c["bw"]) * sx), y2=int((c["y"] + c["bh"]) * sy),
                source="demo",
            ))
        return dets

    # -- 内部工具 --------------------------------------------------------
    @staticmethod
    def _classify(f: dict) -> str:
        """按区域的几何/灰度特征判定病害类别。

        纯确定性函数：同一区域无论出现在哪个批次、哪张图，判定结果一致。这是长周期
        监测能够对比的前提——如果类别会随机漂移，同一道裂缝在不同批次被归到不同
        类别，趋势就失去意义。
        """
        # 亮区检出多为保护层剥落后外露的钢筋或析出物
        if f["src"] == "bright":
            return "exposed_bar" if f["area_ratio"] >= 0.004 else "honeycomb"

        # 线状病害：同样细长，靠"外接框内填充率"区分——裂缝细而稀疏，伸缩缝是密实条带。
        # 不用长宽比：斜向裂缝的外接框同样又宽又扁，会和错台撞车。
        if f["linearity"] < 0.10:
            return "joint_offset" if f["fill"] >= 0.45 else "crack"

        # 面状病害：先看孔隙率（蜂窝是多孔的疏松区），再看对比度（渗水低对比度）
        if f["area_ratio"] >= 0.018:
            if f["porosity"] >= 0.18:
                return "honeycomb"
            if f["dev"] < 0.42:
                return "seepage"
            return "spalling"
        if f["area_ratio"] >= 0.004:
            return "spalling" if f["dev"] >= 0.5 else "seepage"
        return "crack" if f["linearity"] < 0.4 else "bearing"

    @staticmethod
    def _fallback_candidates(rng, gw: int, gh: int, n: int) -> list[dict]:
        out = []
        for _ in range(n):
            bw = int(rng.integers(gw // 8, gw // 3))
            bh = int(rng.integers(gh // 14, gh // 5))
            x = int(rng.integers(0, max(1, gw - bw)))
            y = int(rng.integers(0, max(1, gh - bh)))
            out.append({
                "score": float(rng.uniform(0.55, 0.88)), "x": x, "y": y,
                "bw": bw, "bh": bh, "src": "dark", "linearity": 0.5,
                "porosity": 0.0, "fill": 0.6, "elong": bw / max(bh, 1),
                "area_ratio": bw * bh / (gh * gw), "dev": 0.5,
            })
        return out

    @staticmethod
    def _nms(cands: list[dict], iou_thres: float) -> list[dict]:
        """对候选框做交并比抑制，避免同一处堆叠十几个框。"""
        keep: list[dict] = []
        for c in cands:
            box = (c["x"], c["y"], c["x"] + c["bw"], c["y"] + c["bh"])
            area = c["bw"] * c["bh"]
            drop = False
            for k in keep:
                kbox = (k["x"], k["y"], k["x"] + k["bw"], k["y"] + k["bh"])
                ix1, iy1 = max(box[0], kbox[0]), max(box[1], kbox[1])
                ix2, iy2 = min(box[2], kbox[2]), min(box[3], kbox[3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0:
                    continue
                union = area + k["bw"] * k["bh"] - inter
                if union > 0 and inter / union > iou_thres:
                    drop = True
                    break
            if not drop:
                keep.append(c)
        return keep


# --------------------------------------------------------------------------
# 真实后端
# --------------------------------------------------------------------------
class YoloDetector(DetectBackend):
    """ultralytics YOLO 或 onnxruntime 推理后端。"""

    mode = "model"

    def __init__(self, weights: str, **kw):
        super().__init__(**kw)
        self.weights = weights
        self.name = f"{Path(weights).stem}"
        self._kind = "onnx" if Path(weights).suffix.lower() == ".onnx" else "yolo"
        self._model: Any = None
        self._sess: Any = None
        self._load()

    def _load(self) -> None:
        if self._kind == "onnx":
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.device.startswith("cuda"):
                providers.insert(0, "CUDAExecutionProvider")
            self._sess = ort.InferenceSession(self.weights, providers=providers)
            self.detail = f"（ONNX Runtime · {Path(self.weights).name}）"
        else:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
            task = str(getattr(self._model, "task", "detect") or "detect")
            task_name = "YOLOv8-Seg 裂缝检测" if task == "segment" else "YOLO 裂缝检测"
            self.name = task_name
            self.detail = f"（Ultralytics · {Path(self.weights).name}）"

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def predict(self, bgr: np.ndarray) -> list[Detection]:
        if bgr is None or bgr.size == 0:
            return []
        if self._kind == "onnx":
            return self._predict_onnx(bgr)
        return self._predict_yolo(bgr)

    def _predict_yolo(self, bgr: np.ndarray) -> list[Detection]:
        res = self._model.predict(
            bgr, conf=self.conf, iou=self.iou, imgsz=self.img_size,
            device=self._resolve_device(), verbose=False,
        )[0]
        dets: list[Detection] = []
        names = getattr(res, "names", {}) or {}
        if res.boxes is None:
            return dets
        for b in res.boxes:
            cls_id = int(b.cls[0])
            raw = str(names.get(cls_id, cls_id))
            # The selected crack.pt is a YOLOv8 segmentation model with an
            # explicit background label.  Background is not a bridge defect.
            if raw.strip().lower() in {"background", "bg", "none"}:
                continue
            key = normalize_class(raw)
            dc = cfg.DISEASE_BY_KEY[key]
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            dets.append(Detection(key, dc.name, dc.id, round(float(b.conf[0]), 2),
                                  x1, y1, x2, y2, source="model"))
        return dets


class CrackUNetDetector(DetectBackend):
    """Use a crack-seg U-Net mask to locate crack regions when no YOLO exists."""

    mode = "model"

    def __init__(self, weights: str, **kw):
        super().__init__(**kw)
        self.weights = weights
        self.name = "crack-seg U-Net v3"
        from .crack_unet import get_runtime
        self._runtime = get_runtime(weights, self.device)
        self.detail = f"（像素掩膜区域定位 · {Path(weights).name}）"

    def predict(self, bgr: np.ndarray) -> list[Detection]:
        dc = cfg.DISEASE_BY_KEY["crack"]
        return [Detection(dc.key, dc.name, dc.id, conf, x1, y1, x2, y2,
                          source="crack-unet")
                for x1, y1, x2, y2, conf in self._runtime.locate(bgr, self.conf)]

    def _predict_onnx(self, bgr: np.ndarray) -> list[Detection]:
        """YOLO 系列 ONNX 的标准解析（输出 [1, 4+nc, N]）。"""
        inp = self._sess.get_inputs()[0]
        ih, iw = inp.shape[2], inp.shape[3]
        ih = ih if isinstance(ih, int) else self.img_size
        iw = iw if isinstance(iw, int) else self.img_size

        h0, w0 = bgr.shape[:2]
        k = min(iw / w0, ih / h0)
        nw, nh = int(round(w0 * k)), int(round(h0 * k))
        canvas = np.full((ih, iw, 3), 114, np.uint8)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dx, dy = (iw - nw) // 2, (ih - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]

        out = self._sess.run(None, {inp.name: blob})[0]
        pred = out[0] if out.ndim == 3 else out
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T                      # -> [N, 4+nc]

        boxes, scores, ids = [], [], []
        for row in pred:
            cls_scores = row[4:]
            cid = int(np.argmax(cls_scores))
            score = float(cls_scores[cid])
            if score < self.conf:
                continue
            cx, cy, bw, bh = (float(v) for v in row[:4])
            boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
            scores.append(score)
            ids.append(cid)

        dets: list[Detection] = []
        if not boxes:
            return dets
        for idx in cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou):
            x, y, bw, bh = boxes[int(idx)]
            key = cfg.DISEASES[min(int(ids[int(idx)]), len(cfg.DISEASES) - 1)].key
            dc = cfg.DISEASE_BY_KEY[key]
            dets.append(Detection(
                key, dc.name, dc.id, round(scores[int(idx)], 2),
                int((x - dx) / k), int((y - dy) / k),
                int((x + bw - dx) / k), int((y + bh - dy) / k), source="model",
            ))
        return dets


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
def discover_weights() -> list[Path]:
    """扫描 models/ 下的可用权重。"""
    if not cfg.MODELS_DIR.is_dir():
        return []
    out = [p for p in cfg.MODELS_DIR.iterdir()
           if p.suffix.lower() in {".pt", ".onnx", ".pth"} and p.is_file()]
    return sorted(out, key=lambda p: p.name)


def resolve_weights(setting_key: str, seg: bool = False) -> str:
    """定出实际要用的权重路径。

    优先用设置页里手动指定的路径；没指定就扫 models/ 目录——把训练好的权重直接丢进
    models/ 就能用，不必先去设置页填一遍。分割后端只认带 -seg 的权重，避免误把检测
    权重当分割模型加载。
    """
    s = cfg.load_settings()
    path = (s.get(setting_key) or "").strip()
    if path and Path(path).is_file():
        return path
    for p in discover_weights():
        # crack-seg U-Net produces both the crack-region location and the
        # pixel mask, so its published .pth checkpoint is valid for either
        # stage when the user has not chosen a model path explicitly.
        if p.suffix.lower() == ".pth":
            return str(p)
        is_seg = "-seg" in p.stem.lower() or p.suffix.lower() == ".pth"
        if is_seg == seg:
            return str(p)
    return ""


def make_detector(settings: dict | None = None) -> DetectBackend:
    """按设置构造检测后端；没有可用权重时回退到演示后端。

    这是全局唯一的分支点：页面代码永远只拿到一个 DetectBackend。
    """
    s = settings or cfg.load_settings()
    kw = dict(
        conf=float(s.get("conf_thres", 0.35)),
        iou=float(s.get("iou_thres", 0.45)),
        device=s.get("device", "auto"),
        img_size=int(s.get("img_size", 960)),
    )
    path = resolve_weights("detect_weights")
    if path:
        try:
            if Path(path).suffix.lower() == ".pth":
                return CrackUNetDetector(path, **kw)
            return YoloDetector(path, **kw)
        except Exception as exc:                       # 权重损坏 / 依赖缺失不应崩界面
            demo = DemoDetector(**kw)
            demo.detail = f"（权重 {Path(path).name} 加载失败：{exc}，已回退演示后端）"
            return demo
    return DemoDetector(**kw)
