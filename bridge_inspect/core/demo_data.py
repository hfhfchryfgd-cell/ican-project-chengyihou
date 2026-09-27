"""演示数据生成与播种。

项目当前阶段没有实拍影像、没有病害数据集、硬件也未到货，但软件要能当场演示完整业务
闭环。本模块合成一批"航拍桥梁"影像并播种历史批次数据，使长周期监测一打开就有可讲解
的趋势曲线。

重要：合成影像带有确定的损伤等级（病害几何随批次单调增长），入库的检测/分割记录是把
这些影像真实送进 DemoDetector / DemoSegmenter 跑出来的，不是写死的数字。因此界面展示
的数值与算法行为一致，换成真实权重与实拍图后整条链路不变。

每张合成图里只有一种"主病害"，其形态特征与该类别在真实影像中的外观对应（裂缝是细线、
剥落是密实暗斑、渗水是低对比度大渍、蜂窝是带孔洞的疏松区、伸缩缝是宽条带），这样
确定性分类器的判定结果才会在不同批次之间保持稳定——这是长周期趋势能够成立的前提。
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .. import config as cfg
from .. import db
from . import imgutil
from .backend import Detection, DemoDetector
from .segment import DemoSegmenter, segment_fields

IMG_W, IMG_H = 1280, 900

# 演示点位：(点位编号, 标称拍摄距离 m, 主病害类别, 该点位自身的劣化速率)
DEMO_POINTS: list[tuple[str, float, str, float]] = [
    ("A01", 1.8, "crack",        0.85),
    ("A03", 2.0, "spalling",     0.30),
    ("B02", 1.5, "crack",        1.00),
    ("B05", 1.6, "seepage",      0.55),
    ("C01", 1.2, "exposed_bar",  0.75),
    ("C03", 1.4, "joint_offset", 0.12),
    ("D02", 2.5, "crack",        0.92),
    ("D05", 2.4, "honeycomb",    0.42),
]

# 三个批次的损伤等级（P1 基准 → P3 汛期后），病害几何按此单调增长
DAMAGE_BY_BATCH = {"P1": 0.35, "P2": 0.62, "P3": 1.00}


def _damage_of(point_id: str, batch_id: str) -> float:
    """点位在该批次的病害发展程度。

    各点位按自身的劣化速率沿批次推进，而不是全桥同涨同落：墩柱上一个已经发展充分的
    老裂缝可能几期都不动，梁底新出现的一处却在快速扩展——真桥就是这个样子。
    若所有点位共用一条增长曲线，每个点算出来的累计增幅都超过 60%，报告会把全桥
    点位一律判成"加固"，观察/维修/加固三档分级就失去了区分能力。

    速率是逐点常数，批次顺序仍是 P1<P2<P3，所以每个点位的面积在各批次保持单调，
    长周期曲线该升的照样升——只是升降快慢因点而异。
    """
    rate = next((r for pid, _, _, r in DEMO_POINTS if pid == point_id), 0.6)
    # rate=1 → 完全跟随批次曲线；rate=0 → 各批次几何不变，即一处稳定的老病害
    return 1.0 - rate * (1.0 - DAMAGE_BY_BATCH[batch_id])


# 病害的"暗化/亮化"幅度在各批次保持恒定——只让几何长大。
# 如果幅度也随批次变，分类器依赖的对比度特征会漂移，同一病害可能在中期被判成别的类。
DARK_SPALLING = 58.0
DARK_SEEPAGE = 20.0
DARK_HONEYCOMB = 52.0
DARK_CRACK = 46.0
DARK_JOINT = 62.0

# 伸缩缝两端距画面边缘的固定留白（见 _draw_joint）
_JOINT_INSET = 24


def _seed(*parts) -> int:
    """稳定种子。不能用内置 hash()——字符串哈希每个进程都会变，会导致每次运行
    生成的演示影像都不一样。"""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.md5(key).digest()[:4], "big")


# --------------------------------------------------------------------------
# 底图与结构特征
# --------------------------------------------------------------------------
def _concrete_base(rng: np.random.Generator, tone: int = 150) -> np.ndarray:
    """混凝土底色：低频照度不均 + 细颗粒噪声。"""
    base = np.full((IMG_H, IMG_W), float(tone), np.float32)
    small = rng.normal(0, 13, (IMG_H // 64 + 2, IMG_W // 64 + 2)).astype(np.float32)
    base += cv2.resize(small, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
    base += rng.normal(0, 5.0, (IMG_H, IMG_W)).astype(np.float32)
    return np.clip(base, 0, 255)


def _add_structure(img: np.ndarray, rng: np.random.Generator, prefix: str) -> None:
    """按构件类型铺设模板缝、阴影等结构痕迹，让画面像真实构件。"""
    if prefix == "A":                      # 桥墩：竖向构件，上下有阴影
        for _ in range(int(rng.integers(2, 5))):
            x = int(rng.integers(60, IMG_W - 60))
            cv2.line(img, (x, 0), (x + int(rng.integers(-14, 14)), IMG_H),
                     float(rng.uniform(0, 10)), int(rng.integers(1, 3)))
        img[:70, :] -= 24
        img[-55:, :] -= 30
    elif prefix == "B":                    # 梁底：两侧暗
        for _ in range(int(rng.integers(2, 5))):
            y = int(rng.integers(80, IMG_H - 80))
            cv2.line(img, (0, y), (IMG_W, y + int(rng.integers(-12, 12))),
                     float(rng.uniform(0, 10)), int(rng.integers(1, 3)))
        img[:, :85] -= 38
        img[:, -85:] -= 38
    elif prefix == "C":                    # 支座：钢板 + 橡胶垫分层
        mid = IMG_H // 2
        img[mid - 130:mid - 30] += 16
        img[mid - 30:mid + 30] -= 26
        img[mid + 30:mid + 140] += 9
        cv2.line(img, (0, mid - 30), (IMG_W, mid - 28), -34, 3)
    elif prefix == "D":                    # 梁侧腹板
        for _ in range(int(rng.integers(2, 5))):
            y = int(rng.integers(90, IMG_H - 90))
            cv2.line(img, (0, y), (IMG_W, y + int(rng.integers(-10, 10))),
                     float(rng.uniform(0, 10)), int(rng.integers(1, 3)))
        img[:100, :] -= 22
    else:
        for _ in range(int(rng.integers(2, 5))):
            x = int(rng.integers(60, IMG_W - 60))
            cv2.line(img, (x, 0), (x, IMG_H), float(rng.uniform(0, 10)), 2)

    # 拉杆孔
    for _ in range(int(rng.integers(2, 5))):
        cx, cy = int(rng.integers(50, IMG_W - 50)), int(rng.integers(50, IMG_H - 50))
        cv2.circle(img, (cx, cy), int(rng.integers(5, 8)),
                   float(rng.uniform(0, 14)), -1)


# --------------------------------------------------------------------------
# 主病害绘制
# --------------------------------------------------------------------------
# 裂缝的名义长度（像素）：dmg=0 时最短、dmg=1 时最长
_CRACK_LEN_MIN, _CRACK_LEN_SPAN = 200.0, 620.0
_CRACK_LEN_MAX = _CRACK_LEN_MIN + _CRACK_LEN_SPAN
_CRACK_STEP_PX = 24.0          # 折线每步的长度
_CRACK_EDGE_PX = 6             # 对齐后留给画幅边缘的余量


def _crack_walk(rng: np.random.Generator, x: float, y: float,
                base_ang: float, steps: int) -> list[tuple[float, float]]:
    """从 (x, y) 出发，按"基准角 + 有界游走"走 steps 步，返回折线顶点。

    有界游走：偏离基准角的角度被限制在 ±0.5 rad 内，不会像纯随机游走那样走上几十步
    就漂成任意方向。折线本身不裁边——裁边由调用方按整条裂缝统一平移解决。
    """
    dev = 0.0
    pts = [(x, y)]
    for _ in range(steps):
        dev = float(np.clip(dev + rng.normal(0, 0.22), -0.5, 0.5))
        ang = base_ang + dev
        x += float(np.cos(ang) * _CRACK_STEP_PX)
        y += float(np.sin(ang) * _CRACK_STEP_PX)
        pts.append((x, y))
    return pts


def _draw_crack(img: np.ndarray, rng: np.random.Generator, dmg: float,
                mask: np.ndarray | None = None, label: int = 1) -> None:
    """细长折线裂缝。

    走向用"基准角 + 小幅有界游走"，而不是纯随机游走。纯游走走几十步后会漂成任意
    方向，裂缝的外接框可能变成又宽又扁的一条，形态上就和伸缩缝分不开了。

    位置按**最长**的那条裂缝（dmg=1）一次性定下来，再按批次截短。此处必须这样绕一下：
    早先起点是在画面里随机撒的、基准方向又偏向下（≈π/2），而裂缝最长 820 px、画幅只有
    900 px 高，第 3 次飞行必然越出底边；越界后坐标被 clip，裂缝只能贴着底边滑行，
    长度不再随损伤度增长，长周期曲线会掉头向下——正好毁掉本系统要展示的那个结论。
    """
    base_ang = np.pi / 2 + float(rng.uniform(-0.55, 0.55))
    x0 = float(rng.integers(160, IMG_W - 320))
    y0 = float(rng.integers(160, IMG_H - 320))

    # 一、先按最长裂缝走一遍，只为拿到它的外接框。这一步不含 dmg，各批次完全相同，
    #     所以下面反推出来的平移量与留白也相同——同一处病灶在各次飞行里位置不变。
    pts_max = _crack_walk(rng, x0, y0, base_ang,
                          max(6, int(_CRACK_LEN_MAX / _CRACK_STEP_PX)))
    xs = [p[0] for p in pts_max]
    ys = [p[1] for p in pts_max]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    # 二、把外接框摆到画幅中间，再在剩下的空白里按点位随机挪一挪，位置不至于千篇一律
    slack_x = max(0.0, (IMG_W - bw) / 2.0 - _CRACK_EDGE_PX)
    slack_y = max(0.0, (IMG_H - bh) / 2.0 - _CRACK_EDGE_PX)
    dx = (IMG_W - bw) / 2.0 - min(xs) + float(rng.uniform(-slack_x, slack_x))
    dy = (IMG_H - bh) / 2.0 - min(ys) + float(rng.uniform(-slack_y, slack_y))

    length = _CRACK_LEN_MIN + _CRACK_LEN_SPAN * dmg
    width = max(2, int(round(1.5 + 2.5 * dmg)))
    steps = max(6, int(length / _CRACK_STEP_PX))
    # 批次自己的折线就是最长那条的前缀：游走是纯前缀式的，前 steps 步逐位相同
    pts = [(px + dx, py + dy) for px, py in pts_max[:steps + 1]]

    # 线段先全部收集，再同时画到影像和真值掩膜上，保证两者严格一致
    segs: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for i in range(len(pts) - 1):
        w = max(1, int(width * (1 - 0.7 * i / len(pts))))
        segs.append(((int(pts[i][0]), int(pts[i][1])),
                     (int(pts[i + 1][0]), int(pts[i + 1][1])), w))

    # 分叉（同样限制在基准方向附近）
    for _ in range(int(rng.integers(1, 3 + int(dmg * 3)))):
        i = int(rng.integers(1, max(2, len(pts) - 2)))
        bx, by = pts[i]
        ba = base_ang + float(rng.choice([-1, 1])) * float(rng.uniform(0.5, 0.9))
        for _ in range(int(rng.integers(3, 7))):
            nx = float(np.clip(bx + np.cos(ba) * 20, 3, IMG_W - 4))
            ny = float(np.clip(by + np.sin(ba) * 20, 3, IMG_H - 4))
            segs.append(((int(bx), int(by)), (int(nx), int(ny)), 1))
            bx, by = nx, ny
            ba += float(rng.normal(0, 0.25))

    for p0, p1, w in segs:
        cv2.line(img, p0, p1, -DARK_CRACK, w)
        if mask is not None:
            cv2.line(mask, p0, p1, int(label), w)


def _draw_joint(img: np.ndarray, rng: np.random.Generator, dmg: float,
                mask: np.ndarray | None = None, label: int = 1) -> None:
    """伸缩缝错台：一条实心的近水平暗带，中部有台阶错位。

    必须画成实心：若带内部留出一条较亮的缝，阈值化后暗带会被切成两条细线，外轮廓
    一封就把中间的亮缝算成"孔洞"，形态特征和蜂窝麻面撞车。
    """
    y = int(rng.integers(IMG_H // 3, IMG_H * 2 // 3))
    half = int(26 + 12 * dmg)
    # 水平范围固定：伸缩缝本就横贯整个桥面，出框反而是错的。更要紧的是面积＝
    # 宽度×2×half，若宽度也随机，它 ±4% 的抖动会盖过 half 随批次的增长，该点位
    # 的面积就不再单调了——长周期曲线会冒出一个下行拐点。
    x0, x1 = _JOINT_INSET, IMG_W - _JOINT_INSET
    step_x = int(rng.integers(IMG_W // 3, IMG_W * 2 // 3))
    step = int(10 + 16 * dmg)

    rects = [(x0, y - half, step_x, y + half),
             (step_x, y - half + step, x1, y + half + step)]
    for xa, ya, xb, yb in rects:
        cv2.rectangle(img, (xa, ya), (xb, yb), -DARK_JOINT, -1)
        if mask is not None:
            cv2.rectangle(mask, (xa, ya), (xb, yb), int(label), -1)


def _blob_poly(rng: np.random.Generator, cx: int, cy: int, r: int) -> np.ndarray:
    """生成一个不规则闭合多边形，用于模拟剥落/渗水的边界。"""
    pts = []
    for k in range(16):
        a = 2 * np.pi * k / 16
        rr = r * float(rng.uniform(0.68, 1.22))
        pts.append([int(cx + np.cos(a) * rr), int(cy + np.sin(a) * rr)])
    return np.array(pts, np.int32)


def _draw_spalling(img: np.ndarray, rng: np.random.Generator, dmg: float,
                   mask: np.ndarray | None = None, label: int = 1) -> None:
    """剥落掉块：密实的不规则暗斑，内部有粗糙颗粒。"""
    cx = int(rng.integers(260, IMG_W - 260))
    cy = int(rng.integers(220, IMG_H - 220))
    r = int(58 + 72 * dmg)
    poly = _blob_poly(rng, cx, cy, r)
    region = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(region, [poly], 255)
    noise = cv2.GaussianBlur(rng.normal(0, 15, region.shape).astype(np.float32),
                             (0, 0), 2)
    img[region > 0] += noise[region > 0] - DARK_SPALLING
    if mask is not None:
        cv2.fillPoly(mask, [poly], int(label))


def _draw_seepage(img: np.ndarray, rng: np.random.Generator, dmg: float,
                  mask: np.ndarray | None = None, label: int = 1) -> None:
    """渗水泛碱：大面积、低对比度的柔和湿渍，边缘弥散。"""
    cx = int(rng.integers(300, IMG_W - 300))
    cy = int(rng.integers(260, IMG_H - 260))
    r = int(95 + 125 * dmg)
    poly = _blob_poly(rng, cx, cy, r)
    region = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(region, [poly], 255)
    soft = cv2.GaussianBlur(region.astype(np.float32) / 255.0, (0, 0), r * 0.28)
    img -= soft * DARK_SEEPAGE
    if mask is not None:
        cv2.fillPoly(mask, [poly], int(label))


def _draw_honeycomb(img: np.ndarray, rng: np.random.Generator, dmg: float,
                    mask: np.ndarray | None = None, label: int = 1) -> None:
    """蜂窝麻面：暗斑内部布满未密实的孔洞，形成疏松的多孔区域。"""
    cx = int(rng.integers(300, IMG_W - 300))
    cy = int(rng.integers(250, IMG_H - 250))
    r = int(75 + 90 * dmg)
    poly = _blob_poly(rng, cx, cy, r)
    region = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(region, [poly], 255)
    img[region > 0] -= DARK_HONEYCOMB
    if mask is not None:
        cv2.fillPoly(mask, [poly], int(label))

    # 在斑块内挖出大量小亮孔，形成真实的孔洞感
    n = int(28 + 70 * dmg)
    for _ in range(n):
        a = float(rng.uniform(0, 2 * np.pi))
        d = float(rng.uniform(0, r * 0.86))
        hx = int(cx + np.cos(a) * d)
        hy = int(cy + np.sin(a) * d)
        hr = int(rng.integers(4, 11))
        if region[max(0, hy - hr - 2):hy + hr + 3,
                  max(0, hx - hr - 2):hx + hr + 3].any():
            cv2.circle(img, (hx, hy), hr, float(rng.uniform(38, 58)), -1)


def _draw_exposed_bar(img: np.ndarray, rng: np.random.Generator, dmg: float,
                      mask: np.ndarray | None = None, label: int = 1) -> None:
    """露筋：混凝土保护层剥落后外露的钢筋，呈明暗相间的条带。"""
    cx = int(rng.integers(300, IMG_W - 300))
    cy = int(rng.integers(240, IMG_H - 240))
    r = int(60 + 70 * dmg)

    # 外露区域的暗底
    poly = _blob_poly(rng, cx, cy, r)
    region = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(region, [poly], 255)
    img[region > 0] -= 46
    if mask is not None:
        cv2.fillPoly(mask, [poly], int(label))

    # 内部外露的钢筋：多条平行亮带
    angle = float(rng.uniform(0, np.pi))
    dx, dy = np.cos(angle), np.sin(angle)
    n_bars = int(2 + 2 * dmg)
    for k in range(n_bars):
        off = (k - (n_bars - 1) / 2) * 24
        px, py = cx - dy * off, cy + dx * off
        x0 = int(px - dx * r * 1.1)
        y0 = int(py - dy * r * 1.1)
        x1 = int(px + dx * r * 1.1)
        y1 = int(py + dy * r * 1.1)
        cv2.line(img, (x0, y0), (x1, y1), float(rng.uniform(42, 62)), 7)


_DRAWERS = {
    "crack": _draw_crack,
    "joint_offset": _draw_joint,
    "spalling": _draw_spalling,
    "seepage": _draw_seepage,
    "honeycomb": _draw_honeycomb,
    "exposed_bar": _draw_exposed_bar,
}


# --------------------------------------------------------------------------
# 次要病害处数上限，按类别给。伸缩缝错台的主病害是一条近乎横贯画面的暗带，画面里
# 腾不出不重叠的位置——硬塞一处的化它总有一部分被主病害吃掉，而"被吃掉多少"随批次
# 变化，该点位的面积合计就不再单调。这类病害一处就是它真实的形态。
_EXTRA_MAX = {"joint_offset": 0}


def _extra_count(point_id: str, cls_key: str) -> int:
    """该点位影像里次要病害的处数。

    只由点位决定，不含批次——若处数随批次变，同一个病灶在各批次里"算了几处"就不同，
    面积合计失去可比性，而可比性正是长周期监测成立的前提。
    """
    cap = _EXTRA_MAX.get(cls_key, 2)
    return 0 if cap == 0 else int(_seed("sub_n", point_id) % (cap + 1))


def _add_secondary(img: np.ndarray, gt_mask: np.ndarray, point_id: str,
                   cls_key: str, dmg: float) -> None:
    """叠 0~2 处同类轻微病害，让画面不止一个目标。

    这些是**真实标定的**病害实例（有掩膜、进真值清单），不是装饰性噪声：检测、分割、
    长周期三条链路都会如实统计它们。位置同样只由点位决定，保证各批次"同视场"。
    """
    drawer = _DRAWERS.get(cls_key, _draw_crack)
    for k in range(1, _extra_count(point_id, cls_key) + 1):
        srng = np.random.default_rng(_seed("sub", point_id, k))
        scratch = img.copy()
        smask = np.zeros((IMG_H, IMG_W), np.uint8)
        drawer(scratch, srng, dmg * 0.5, smask, 1)
        # 只取主病害未占用的像素：主病害始终完整，次要病害让路
        free = (smask > 0) & (gt_mask == 0)
        if int(np.count_nonzero(free)) < 24:
            continue
        img[free] = scratch[free]
        gt_mask[free] = k + 1


def synthesize(point_id: str, batch_id: str, cls_key: str
               ) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """合成一张"航拍"桥梁影像：底图 + 构件结构 + 一种主病害。

    同时返回该图的真值：病害实例清单与逐像素标签掩膜。掩膜标号 i 对应 instances[i-1]，
    供 DemoDetector / DemoSegmenter 在演示影像上给出与画面内容一致的判定。
    """
    # 两个随机源分工不同：
    #   rng —— 底图照度、颗粒噪声、模板缝等拍摄条件，逐批次重掷，像真的复飞；
    #   geo —— 病害的**形态与位置**，只由点位决定，各批次完全一致。
    # 后者是关键：同视场复拍本来就该拍的是同一处病灶、同一个位置，它的轮廓不会每飞
    # 一次就重新长一遍。若几何也逐批次重掷，斑块轮廓 ±25% 的顶点抖动会盖过各批次
    # 之间那点真实增长，长周期曲线就会冒出下行拐点——而这正是本系统的核心结论。
    rng = np.random.default_rng(_seed("img", point_id, batch_id))
    geo = np.random.default_rng(_seed("geo", point_id))
    dmg = _damage_of(point_id, batch_id)
    prefix = point_id[:1]

    tone = {"A": 148, "B": 124, "C": 158, "D": 140}.get(prefix, 145)
    img = _concrete_base(rng, tone)
    _add_structure(img, rng, prefix)

    gt_mask = np.zeros((IMG_H, IMG_W), np.uint8)
    # 先铺次要病害（标号 2、3），再画主病害（标号 1）。主病害后画，掩膜上会直接
    # 覆写重叠处，所以它的面积只由自身几何决定——这是面积随批次单调增长的前提。
    _add_secondary(img, gt_mask, point_id, cls_key, dmg)
    _DRAWERS.get(cls_key, _draw_crack)(img, geo, dmg, gt_mask, 1)

    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tint = np.array([int(rng.integers(2, 8)), int(rng.integers(1, 4)), 0], np.int16)
    bgr = np.clip(bgr.astype(np.int16) + tint, 0, 255).astype(np.uint8)

    instances: list[dict] = []
    n_lab = int(gt_mask.max())
    for i in range(1, n_lab + 1):
        ys, xs = np.nonzero(gt_mask == i)
        if xs.size < 24:
            continue
        instances.append({
            "cls_key": cls_key,
            "box": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            "px": int(xs.size),
        })
    return bgr, instances, gt_mask


def synth_image(point_id: str, batch_id: str, cls_key: str) -> np.ndarray:
    """只要影像本身（真值单独走 save_gt）。"""
    return synthesize(point_id, batch_id, cls_key)[0]


# --------------------------------------------------------------------------
# 演示影像真值登记
#
# 演示后端在预置影像上必须给出与画面内容一致的判定，否则长周期趋势会因为类别漂移而
# 失去意义。做法是把合成时已知的真值按"图像内容指纹"存成边车文件，推理时反查。
# 用户自己导入的照片没有边车，走启发式检测路径。
# --------------------------------------------------------------------------
GT_SUFFIX = ".gt.npz"
_GT_INDEX: dict[str, str] | None = None


def image_sig(bgr: np.ndarray) -> str:
    """图像内容指纹：灰度缩到 64×64 再取 md5，对 JPEG 压缩不敏感。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


def _gt_path(image_path) -> object:
    from pathlib import Path

    return Path(image_path).with_suffix(GT_SUFFIX)


def save_gt(image_path, instances: list[dict], gt_mask: np.ndarray) -> None:
    """把真值写到影像旁边的边车文件。"""
    if not instances:
        return
    keys = np.array([it["cls_key"] for it in instances])
    boxes = np.array([it["box"] for it in instances], np.int32)
    np.savez_compressed(
        _gt_path(image_path),
        sig=np.array(image_sig(imgutil.imread(image_path))),
        keys=keys, boxes=boxes, mask=gt_mask,
    )


def _build_gt_index() -> dict[str, str]:
    global _GT_INDEX
    if _GT_INDEX is not None:
        return _GT_INDEX
    idx: dict[str, str] = {}
    for p in cfg.IMG_DIR.glob("*" + GT_SUFFIX):
        try:
            with np.load(p) as z:
                idx[str(z["sig"])] = str(p)
        except (OSError, ValueError, KeyError):
            continue
    _GT_INDEX = idx
    return idx


def drop_gt_cache() -> None:
    """重新生成演示数据后清缓存。"""
    global _GT_INDEX
    _GT_INDEX = None


def gt_lookup(bgr: np.ndarray) -> dict | None:
    """按内容指纹查真值；不是演示影像则返回 None。"""
    p = _build_gt_index().get(image_sig(bgr))
    if not p:
        return None
    try:
        with np.load(p) as z:
            return {"keys": [str(k) for k in z["keys"]],
                    "boxes": z["boxes"], "mask": z["mask"]}
    except (OSError, ValueError, KeyError):
        return None


# --------------------------------------------------------------------------
# 航点元数据
# --------------------------------------------------------------------------
def _nominal_distance(point_id: str) -> float:
    """该点位的标称拍摄距离：演示点位取设定值，其余按构件类型取默认值。"""
    for pid, dist, *_ in DEMO_POINTS:
        if pid == point_id:
            return dist
    # 梁底／支座要贴得更近才拍得清，墩柱与腹板可以退远一点
    return {"A": 1.8, "B": 1.6, "C": 1.3, "D": 2.4}.get(point_id[:1], 2.0)


def waypoint_meta(point_id: str, batch_id: str,
                  distance: float | None = None, order: int = 1) -> dict:
    """为一张影像生成航点元数据（导入外部影像时按点位与批次补齐标称值）。"""
    return _waypoint_meta(point_id, batch_id,
                          distance if distance else _nominal_distance(point_id),
                          order)


def _waypoint_meta(point_id: str, batch_id: str, distance: float,
                   order: int) -> dict:
    """生成该图对应的航点 / RTK / 云台元数据（口径与作品说明书的航点规划一致）。"""
    rng = np.random.default_rng(_seed("wp", point_id, batch_id))
    prefix = point_id[:1]

    # RTK 由桥位几何推出来，而不是按"第几个点"排布：站位与横向偏移见
    # `config.WAYPOINT_PLAN_M`。这样俯视图上的点位分布与真实桥面一致——
    # 墩位上的点排在墩位，跨中的点排在跨中，支座分列桥轴两侧。
    station_m, offset_m = cfg.waypoint_plan(point_id)
    base_lon = cfg.ORIGIN_LON + station_m / cfg.m_per_deg_lon(cfg.ORIGIN_LAT)
    base_lat = cfg.ORIGIN_LAT + offset_m / cfg.M_PER_DEG_LAT
    # 云台姿态：梁底仰拍、支座侧仰拍，墩柱与腹板正对（0°）
    pitch = {"A": 0.0, "B": -32.0, "C": -45.0, "D": 0.0}.get(prefix, 0.0)
    yaw = {"A": 0.0, "B": 0.0, "C": 90.0, "D": 0.0}.get(prefix, 0.0)

    return {
        "point_id": point_id,
        "waypoint": f"WP-{point_id}-{batch_id}",
        "batch": batch_id,
        "rtk_lon": round(base_lon + float(rng.normal(0, 2e-7)), 7),
        "rtk_lat": round(base_lat + float(rng.normal(0, 2e-7)), 7),
        "rtk_alt": round(24.0 + float(rng.normal(0, 0.12)), 2),
        "rtk_status": "固定解" if rng.random() > 0.12 else "浮点解",
        "gimbal_yaw": round(yaw + float(rng.normal(0, 1.2)), 2),
        "gimbal_pitch": round(pitch + float(rng.normal(0, 1.0)), 2),
        "gimbal_roll": round(float(rng.normal(0, 0.8)), 2),
        # 拍摄距离按航点规划给定，**不随批次抖动**。它是标定面积的基准：GSD 与距离
        # 成正比，面积与 GSD² 成正比，这里若加 ±0.05 m 的抖动，面积就会带上 ±7% 的
        # 随机起伏，足以把一处真实增长盖成下行。复飞同步的意义正是让每次回到同一
        # 个站位，所以这个值在各批次必须一模一样——抖动只保留在 RTK、云台与重合率上。
        "shoot_distance": round(distance, 2),
        # 同视场复拍重合率：多数优良，少量偏低以演示异常标记
        "overlap": round(float(np.clip(rng.normal(0.91, 0.05), 0.62, 0.99)), 3),
        "captured_at": (f"{cfg.BATCH_BY_ID[batch_id].date} "
                        f"{8 + order:02d}:{int(rng.integers(0, 60)):02d}:00"),
    }


# --------------------------------------------------------------------------
# 播种
# --------------------------------------------------------------------------
_DEMO_IMG = "waypoint LIKE 'WP-%'"


def purge_demo_rows() -> None:
    """清空演示影像及其派生的检测/分割记录与边车真值文件。

    必须显式删 detections 与 segments 再删 images：库里没有建外键级联（这些表刻意
    保持轻量），只删 images 会把子表留成孤儿行。而长周期监测是**直接查 segments**、
    不 JOIN images 的，于是重建一次演示数据，上一代的记录还在，同一个点位就同时有
    两套面积——趋势线取批次最大值时会挑中旧的那套，曲线冒出一次无从解释的回落。
    """
    for t in ("detections", "segments"):
        # 先扫孤儿行：早先版本只删了 images，这些子表行的父影像已经不在库里，按
        # image_id 再怎么匹配都找不回来，只能按"指向不存在的影像"识别。留着它们，
        # 长周期监测会把上一代的数据一并统计进去。
        db.execute(f"DELETE FROM {t} WHERE image_id IS NULL"
                   f" OR image_id NOT IN (SELECT id FROM images)")
        db.execute(f"DELETE FROM {t} WHERE image_id IN"
                   f" (SELECT id FROM images WHERE {_DEMO_IMG})")
    db.execute(f"DELETE FROM images WHERE {_DEMO_IMG}")
    db.execute("DELETE FROM tasks WHERE name LIKE '%全桥巡检' AND owner='admin'")
    for p in cfg.IMG_DIR.glob("*" + GT_SUFFIX):
        try:
            p.unlink()
        except OSError:
            pass
    drop_gt_cache()


def ensure_demo_dataset(force: bool = False, progress=None) -> dict:
    """生成演示影像、跑通检测与分割、把结果落库。已播种过则直接返回统计。"""
    already = db.query_one(
        "SELECT COUNT(*) AS n FROM images WHERE waypoint LIKE 'WP-%'")
    if already and already["n"] > 0 and not force:
        return {"skipped": True, "images": already["n"]}

    if force:
        purge_demo_rows()

    detector = DemoDetector()
    segmenter = DemoSegmenter()
    stats = {"images": 0, "detections": 0, "segments": 0}

    total = len(DEMO_POINTS) * len(cfg.BATCHES)
    step = 0
    for batch in cfg.BATCHES:
        # 长周期样例只是用于展示跨期曲线，不伪装成用户执行过的检测任务。
        # 因此它的检测/量化记录不挂 task_id，也不会进入「检测历史」计数。
        task_id = None

        for order, (point_id, distance, cls_key, _rate) in enumerate(DEMO_POINTS, 1):
            step += 1
            if progress:
                progress(step, total, f"{point_id} · {batch.name}")

            # 传"本次飞行的第几个点"而不是全局计数器：拍摄时刻由它推出来，用全局
            # 序号会让第三个批次的影像落到 25:13 这种不存在的钟点上。
            meta = _waypoint_meta(point_id, batch.id, distance, order)
            bgr, instances, gt_mask = synthesize(point_id, batch.id, cls_key)

            fname = f"{batch.id}_{point_id}_{meta['waypoint']}.jpg"
            path = cfg.IMG_DIR / fname
            if force or not path.exists():
                drop_gt_cache()
                imgutil.imwrite(path, bgr)
                save_gt(path, instances, gt_mask)
                # 后面统一用"从磁盘读回来的"影像跑算法：JPEG 有损压缩会改变像素，
                # 用内存原图跑出来的框和真值指纹对不上，反查就会落空。
                bgr = imgutil.imread(path)

            existing = db.image_by_path(str(path))
            if existing and not force:
                continue
            if existing:
                # 同一路径已有记录：连子表一起换掉，不能只删 images 留孤儿行
                db.execute("DELETE FROM detections WHERE image_id=?", (existing["id"],))
                db.execute("DELETE FROM segments WHERE image_id=?", (existing["id"],))
                db.execute("DELETE FROM images WHERE id=?", (existing["id"],))

            h, w = bgr.shape[:2]
            image_id = db.add_image(
                task_id=task_id, path=str(path), filename=fname, width=w, height=h,
                size_kb=round(path.stat().st_size / 1024, 1) if path.exists() else 0,
                source="SD卡导入", **meta,
            )
            stats["images"] += 1

            # 真实跑一遍检测 —— 库里存的框就是算法算出来的
            dets: list[Detection] = detector.predict(bgr)
            rows = [{
                "image_id": image_id, "task_id": task_id, "point_id": point_id,
                "cls_key": d.cls_key, "cls_name": d.cls_name, "conf": d.conf,
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                "gt_idx": d.gt_idx,
            } for d in dets]
            db.add_detections(rows)
            stats["detections"] += len(rows)

            # 对每个检测框做分割量化，写入 segments —— 长周期监测的数据源。
            # 演示影像上检测器给的都是真值实例，这里的过滤只是防御：万一将来检测端
            # 又冒出没有掩膜的框，也不会把它按启发式阈值量出一个带随机性的面积、
            # 混进趋势里盖过真实增长。
            has_gt = gt_lookup(bgr) is not None
            for d in dets:
                if has_gt and d.gt_idx <= 0:
                    continue
                res = segmenter.segment(bgr, d, meta["shoot_distance"])
                if res is None or res.px_area <= 0:
                    continue
                db.add_segment(
                    image_id=image_id, task_id=task_id, point_id=point_id,
                    batch=batch.id, captured_at=meta["captured_at"],
                    **segment_fields(res),
                )
                stats["segments"] += 1

    stats["skipped"] = False
    return stats


def regenerate_all(progress=None) -> dict:
    """清空演示数据重新生成（系统管理页的"重建演示数据"用）。"""
    return ensure_demo_dataset(force=True, progress=progress)
