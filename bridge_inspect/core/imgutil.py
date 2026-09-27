"""图像工具：中文路径读写、Qt 转换、带中文标签的检测框绘制。

Windows 下 cv2.imread 遇到中文路径会静默返回 None，本项目所有图片路径都可能是
中文（桥梁名、点位名），因此统一走 imdecode/imencode 通道。
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np

# 中文字体候选，按优先级取第一个存在的。
# 宋体排首位是为了与界面统一（这是全仓库唯一"按存在性探测"的字体逻辑，兜底链
# 要覆盖住）。标注框文字是**画在实心色块上**的，没有压在照片纹理上的半透明底，
# 所以宋体笔画细也不影响可读性——这一条是选它而非雅黑的前提。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
)


@functools.lru_cache(maxsize=16)
def cjk_font(size: int):
    """取一个可绘制中文的 PIL 字体，失败返回 PIL 默认字体。"""
    from PIL import ImageFont

    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def imread(path: str | Path) -> np.ndarray | None:
    """读取图片（支持中文路径），返回 BGR。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def imwrite(path: str | Path, bgr: np.ndarray) -> bool:
    """写出图片（支持中文路径）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def imread_gray(path: str | Path) -> np.ndarray | None:
    img = imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# 尺寸与坐标
# --------------------------------------------------------------------------
def fit_size(w: int, h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """等比缩放到不超过 (max_w, max_h)。"""
    if w <= 0 or h <= 0 or max_w <= 0 or max_h <= 0:
        return max(w, 1), max(h, 1)
    k = min(max_w / w, max_h / h)
    return max(int(w * k), 1), max(int(h * k), 1)


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    return (max(0, min(int(x1), w - 1)), max(0, min(int(y1), h - 1)),
            max(0, min(int(x2), w - 1)), max(0, min(int(y2), h - 1)))


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# --------------------------------------------------------------------------
# Qt 转换
# --------------------------------------------------------------------------
def bgr_to_qimage(bgr: np.ndarray):
    """BGR ndarray -> QImage（数据已拷贝，可脱离原数组生命周期）。"""
    from PyQt6.QtGui import QImage

    if bgr is None:
        return QImage()
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def bgr_to_qpixmap(bgr: np.ndarray):
    from PyQt6.QtGui import QPixmap

    return QPixmap.fromImage(bgr_to_qimage(bgr))


# --------------------------------------------------------------------------
# 标注绘制
# --------------------------------------------------------------------------
def draw_detections(
    bgr: np.ndarray,
    dets: list,
    thickness: int = 2,
    show_label: bool = True,
) -> np.ndarray:
    """在图上绘制检测框与中文标签。

    dets 为 Detection 列表（见 core.backend），每项含 cls_name / conf / xyxy / cls_key。
    """
    from PIL import Image, ImageDraw

    if bgr is None:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    if not dets:
        return out

    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    from ..config import disease_color

    box_w = max(2, thickness + 1)
    for d in dets:
        x1, y1, x2, y2 = clamp_box(d.x1, d.y1, d.x2, d.y2, w, h)
        color = disease_color(getattr(d, "cls_key", ""))
        rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

        # 用 PIL 画框，保证线条柔和不糊；粗线用双线模拟
        for i in range(box_w):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=rgb)

        if not show_label:
            continue
        text = f"{d.cls_name} {d.conf:.2f}"
        font = cjk_font(max(13, int(min(w, h) * 0.022)))
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 4
        ly = y1 - th - 2 * pad
        if ly < 0:
            ly = min(y2 + 1, h - th - 2 * pad)
        lx = min(x1, w - tw - 2 * pad)

        draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=rgb)
        # 亮色底用深字，暗色底用白字，保证对比度
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        fg = (12, 20, 32) if lum > 150 else (255, 255, 255)
        draw.text((lx + pad - tb[0], ly + pad - tb[1]), text, font=font, fill=fg)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_mask_overlay(bgr: np.ndarray, mask: np.ndarray, color_hex: str = "#22D3EE",
                      alpha: float = 0.45) -> np.ndarray:
    """把二值掩码以半透明色叠加到原图，并勾出轮廓。"""
    if bgr is None or mask is None:
        return bgr
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    out = bgr.copy()
    m = (mask > 0)
    layer = np.zeros_like(out)
    layer[:] = hex_to_bgr(color_hex)
    out[m] = (out[m] * (1 - alpha) + layer[m] * alpha).astype(np.uint8)

    contours, _ = cv2.findContours((m.astype(np.uint8)) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, hex_to_bgr(color_hex), 2)
    return out


def make_thumbnail(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """生成缩略图（保持比例，补深色底）。"""
    if bgr is None:
        return np.full((size[1], size[0], 3), 22, np.uint8)
    th, tw = size[1], size[0]
    h, w = bgr.shape[:2]
    k = min(tw / w, th / h)
    nw, nh = max(1, int(w * k)), max(1, int(h * k))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), 22, np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def image_info(path: str | Path) -> tuple[int, int, float]:
    """返回 (宽, 高, 大小KB)，读不到返回 (0,0,0)。"""
    p = Path(path)
    try:
        kb = p.stat().st_size / 1024.0
    except OSError:
        kb = 0.0
    img = imread(p)
    if img is None:
        return 0, 0, kb
    h, w = img.shape[:2]
    return w, h, kb


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def dialog_filter() -> str:
    """文件对话框的过滤器串，如「影像文件 (*.jpg *.jpeg *.png ...)」。

    SUPPORTED_EXT 存的后缀**自带点**（".jpg"），拼通配符时不能再补一个点，否则
    得到的是 "*..jpg"——这个模式匹配不到任何文件，用户会看到对话框里空空如也，
    以为自己的图片有问题，实际上翻遍整个磁盘也是一样的结果。

    三处打开图片的入口（导入、检测、分割）以前各自拼过一遍这个串，所以放进这里
    统一构造，改一次就不会再走样。
    """
    return "影像文件 (%s)" % " ".join(f"*{e}" for e in sorted(SUPPORTED_EXT))


def is_supported(path: str | Path) -> bool:
    """该路径的后缀是否在支持列表内。

    一律用它判断，不要自己写 str(p).rsplit(".", 1)[-1] in SUPPORTED_EXT：那样切出来
    的 "jpg" 不带点，和集合里的 ".jpg" 对不上，`in` 恒为假——结果是把**任何**图片
    都判成不支持的格式，用户拖进来的照片会一声不响地被丢掉。
    """
    return Path(path).suffix.lower() in SUPPORTED_EXT


def list_images_in(folder: str | Path) -> list[Path]:
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(x for x in p.iterdir()
                  if x.is_file() and x.suffix.lower() in SUPPORTED_EXT)
