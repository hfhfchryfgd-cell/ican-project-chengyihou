"""全局配置：路径、病害类别体系、构件—点位体系、巡检批次、默认参数与设置持久化。

本模块是整个软件的常量中心。任何页面、算法后端都从这里取类别定义、点位规则和默认
参数，避免同一套口径在多处重复定义后走样。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"          # 航拍原图 / 演示图
OUT_DIR = DATA_DIR / "outputs"         # 检测标注图、掩码、CSV、报告
MODELS_DIR = ROOT / "models"           # 放入 .pt / .pth / .onnx 即自动启用真实后端
DB_PATH = DATA_DIR / "bridge_inspect.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
CORE_DIR = ROOT / "bridge_inspect" / "core"
HERO_SVG = CORE_DIR / "login_hero.svg"  # 登录页插图，与网页版共用同一份

for _d in (DATA_DIR, IMG_DIR, OUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

APP_NAME = "桥梁智巡"
APP_NAME_EN = "BRIDGE INSPECTION SUITE"
APP_TITLE = "面向铁路桥梁的同视场无人机病害巡检分析系统"
APP_VERSION = "v1.0"

# --------------------------------------------------------------------------
# 报告报头
# --------------------------------------------------------------------------
# 报告编号形如 BIS-20260924-P1P2P3-141530：前缀 + 生成日期 + 覆盖批次 + 生成时刻。
# 只用"生成时刻"与"覆盖批次"两个可观测量，不查库、不依赖计数器——换台机器、
# 换个输出目录重算，编号不变；正文的"编制日期"也打印到秒，两者可互相验证。
REPORT_NO_PREFIX = "BIS"                          # 取 APP_NAME_EN 的首字母缩写
REPORT_ORG = "桥梁智巡巡检数据分析中心"             # 编制单位；settings["report_org"] 可覆盖
# 校核/审核人。留空则签署栏印空白格待手签——报告是给人签字的，编一个名字印上去
# 比留空更糟。
REPORT_CHECKER = ""
REPORT_APPROVER = ""

# --------------------------------------------------------------------------
# 病害类别体系
# 对齐《公路桥涵养护规范》口径，覆盖作品说明书中提到的裂缝、剥落、支座滑移、伸缩缝错台等。
# colour 同时用于检测框、图例、统计表与趋势图，保证全局视觉一致。
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiseaseClass:
    id: int
    key: str
    name: str
    color: str
    severity: str          # 严重度分级，用于右侧统计表与研判建议
    suggestion: str        # 该类别默认养护建议口径
    # 该类别"宽度本身即成害"的限值(mm)，None 表示宽度不参与定级。
    # 只对线状病害成立：裂缝按规范 0.30 mm，伸缩缝错台按错台量 5 mm。剥落、渗水、
    # 蜂窝、露筋是面状病害，算法给出的 max_width_mm 是斑块的外接尺度（几十毫米），
    # 拿裂缝的限值去套它们，会让一块静止的麻面仅因为"块大"就被判成加固。
    width_limit_mm: float | None = None


DISEASES: tuple[DiseaseClass, ...] = (
    DiseaseClass(0, "crack",        "裂缝",       "#F87171", "高", "观测裂缝宽度与长度发展，超限时灌注封闭", 0.30),
    DiseaseClass(1, "spalling",     "剥落掉块",   "#FBBF24", "高", "凿除松散层后修补，防止钢筋进一步外露"),
    DiseaseClass(2, "exposed_bar",  "露筋",       "#A78BFA", "高", "除锈阻锈处理后高强砂浆修复保护层"),
    DiseaseClass(3, "seepage",      "渗水泛碱",   "#38BDF8", "中", "排查防水层与排水通道，清理析出物"),
    DiseaseClass(4, "honeycomb",    "蜂窝麻面",   "#34D399", "低", "表面封闭处理，观察是否伴随渗水"),
    DiseaseClass(5, "bearing",      "支座滑移",   "#FB923C", "高", "复测支座位移量，必要时顶升复位"),
    DiseaseClass(6, "joint_offset", "伸缩缝错台", "#F472B6", "中", "清理缝内杂物，复核错台量并调整", 5.0),
)

DISEASE_BY_KEY = {d.key: d for d in DISEASES}
DISEASE_BY_ID = {d.id: d for d in DISEASES}
DISEASE_NAMES = [d.name for d in DISEASES]
COLOR_BY_KEY = {d.key: d.color for d in DISEASES}


def disease_color(key: str) -> str:
    """按类别 key 取展示色，未知类别回退到中性灰。"""
    return COLOR_BY_KEY.get(key, "#8FA3C0")


# --------------------------------------------------------------------------
# 构件—点位体系
# 点位编号是长周期监测的索引键：同一编号在不同批次必须指向同一物理位置。
# --------------------------------------------------------------------------
COMPONENTS: dict[str, str] = {
    "A": "桥墩",
    "B": "梁底",
    "C": "支座",
    "D": "梁侧腹板",
    "E": "桥台",
    "F": "伸缩缝",
}

POINT_PREFIX_ORDER = ("A", "B", "C", "D", "E", "F")


def make_point_id(prefix: str, index: int) -> str:
    """生成点位编号，如 ('A', 3) -> 'A03'。"""
    return f"{prefix}{index:02d}"


def point_component(point_id: str) -> str:
    """由点位编号反查构件名称，如 'B07' -> '梁底'。"""
    return COMPONENTS.get(point_id[:1].upper(), "其他构件") if point_id else "未指定"


ALL_POINTS: list[str] = [
    make_point_id(p, i) for p in POINT_PREFIX_ORDER for i in range(1, 9)
]

# --------------------------------------------------------------------------
# 桥位几何
# 这几个数不是给界面凑图形的：俯视图、复飞航点表、报告里的墩号引用同一套坐标，
# 改一处三处一起变。此前航点只按"第几个点"排布，经度铺开 33 m、纬度却因构件
# 分行铺开 155 m，画出来是一条竖线而不是一座桥。
# --------------------------------------------------------------------------
BRIDGE_LEN_M = 120.0        # 桥长：6 跨 × 20 m
BRIDGE_WIDTH_M = 12.0       # 桥面宽
PIER_SPACING_M = 20.0       # 墩距

# 起算点（首跨桥台处的 RTK 坐标）。整座桥在这个点上沿经线方向展开：
# 经度增加 = 沿桥向，纬度增加 = 横桥向。选经度作桥轴是因为在北纬 32°
# 经线方向更"值钱"——同样的米数只占 0.85 倍的经度差，点位之间不易挤在一起。
ORIGIN_LAT = 32.0512400
ORIGIN_LON = 118.7426100

M_PER_DEG_LAT = 110540.0


def m_per_deg_lon(lat: float) -> float:
    """该纬度上 1° 经度对应的米数。

    俯视图必须做这个改正：北纬 32° 处 1° 经度只有纬度的 0.85 倍长，不改正
    会把桥横向拉宽 18%，墩距看着就不是等距的了。
    """
    return 111320.0 * math.cos(math.radians(lat))


# 航点平面布置：(沿桥向站位 m, 横桥向偏移 m)。
# 0 m 站位 = 起点桥台，0 m 偏移 = 桥轴，正偏移 = 上游侧。
# 墩位在 20/40/60/80/100 m，桥台在 0/120 m，跨中在 10/30/…/110 m。
WAYPOINT_PLAN_M: dict[str, tuple[float, float]] = {
    # 桥墩：墩顶正对桥轴
    "A01": (20.0, 0.0), "A03": (60.0, 0.0),
    # 梁底：跨中仰拍，测点在梁底中部
    "B02": (30.0, 0.0), "B05": (90.0, 0.0),
    # 支座：分列桥轴两侧的两条支座线（12 m 宽桥面的支座中心距约 7.2 m）
    "C01": (20.0, 3.6), "C03": (60.0, -3.6),
    # 梁侧腹板：箱梁外侧面，横向落在桥面边缘
    "D02": (30.0, 6.0), "D05": (90.0, 6.0),
}

# 各构件的横向偏移（米）。逐点登册的那 8 个点用上表，其余 40 个规划航点按
# 构件落到各自的横向上——48 个点若全排在桥轴上，俯视图上看不出构件之分。
COMPONENT_OFFSET_M: dict[str, float] = {
    "A": 0.0,    # 桥墩：墩顶正对桥轴
    "B": 0.0,    # 梁底：桥轴正下方
    "C": 3.6,    # 支座：支座线
    "D": 6.0,    # 梁侧腹板：桥面边缘
    "E": 0.0,    # 桥台：桥轴
    "F": 0.0,    # 伸缩缝：横贯全宽，取桥轴
}


# 各构件沿桥向的分布区间（起, 止），米。桥墩类构件落在两端桥台之间，
# 梁底与腹板的测点落在跨中一侧，桥台只在两端——用区间均分而不是
# "序号 × 墩距"，否则 8 个编号里有 3 个会叠在终点桥台上。
COMPONENT_SPAN_M: dict[str, tuple[float, float]] = {
    "A": (PIER_SPACING_M, BRIDGE_LEN_M - PIER_SPACING_M),
    "B": (PIER_SPACING_M / 2, BRIDGE_LEN_M - PIER_SPACING_M / 2),
    "C": (PIER_SPACING_M, BRIDGE_LEN_M - PIER_SPACING_M),
    "D": (PIER_SPACING_M / 2, BRIDGE_LEN_M - PIER_SPACING_M / 2),
    "E": (0.0, 0.0),                    # 两端交替，见下
    "F": (PIER_SPACING_M, BRIDGE_LEN_M - PIER_SPACING_M),
}


def waypoint_plan(point_id: str) -> tuple[float, float]:
    """点位在桥面上的平面位置 (沿桥向站位 m, 横桥向偏移 m)。

    没逐点登册的规划航点按**构件类型**定站位与横向：桥墩、支座、伸缩缝排在
    两端桥台之间，梁底与腹板排在跨中一侧，桥台只在两端。这样 `ALL_POINTS`
    里的 48 个编号都落在桥上真正该在的位置，且互不重叠。
    """
    if point_id in WAYPOINT_PLAN_M:
        return WAYPOINT_PLAN_M[point_id]
    prefix = point_id[:1].upper()
    try:
        idx = max(1, min(8, int(point_id[1:])))
    except ValueError:
        idx = 1
    if prefix == "F":
        # 伸缩缝横贯全宽，不是顺桥向排开的：8 个测点落在前 4 道墩位、桥轴两侧。
        # 横向取 ±1.8 m——支座线在 ±3.6 m、腹板在 ±6.0 m，三段各占一条带，
        # 俯视图上正好读成桥面横断面的一条条纵向测线。
        # 若也跟着"沿桥向均分"，8 个测点会与桥墩的逐个重合，一个圈顶掉另一个圈。
        return (PIER_SPACING_M * ((idx - 1) % 4 + 1), 1.8 if idx <= 4 else -1.8)
    lo, hi = COMPONENT_SPAN_M.get(prefix, (0.0, BRIDGE_LEN_M))
    if hi <= lo:
        # 桥台只在两端：前 4 个点位落在起点桥台、后 4 个落在终点桥台。
        # 此前是"奇数去起点、偶数去终点"，8 个点的里程成了 0、120、0、120…，
        # 俯视图上连成一条纵贯全桥的折返线——那正是"俯视图不像桥"最扎眼的一处。
        return (0.0 if idx <= 4 else BRIDGE_LEN_M,
                1.8 if idx % 2 else -1.8)
    station = lo + (idx - 1) / 7.0 * (hi - lo)
    return (station, COMPONENT_OFFSET_M.get(prefix, 0.0))

# --------------------------------------------------------------------------
# 巡检批次
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Batch:
    id: str
    name: str
    date: str
    note: str


BATCHES: tuple[Batch, ...] = (
    Batch("P1", "第 1 次飞行", "2026-03-12", "初始基准期，建立全桥基准影像库"),
    Batch("P2", "第 2 次飞行", "2026-06-18", "汛期前复检，重点复核支座与梁底"),
    Batch("P3", "第 3 次飞行", "2026-09-05", "汛期后复检，关注渗水与裂缝发展"),
)

BATCH_BY_ID = {b.id: b for b in BATCHES}
BATCH_NAMES = [b.name for b in BATCHES]

# --------------------------------------------------------------------------
# 业务阈值与标定参数
# --------------------------------------------------------------------------
OVERLAP_OK = 0.85          # 图像重合率合格线，低于此值判为需重拍（说明书表 5-1、5-2）
OVERLAP_WARN = 0.92        # 高于此值视为同视场复拍质量优良

# 地面分辨率标定：GSD(mm/px) = 像元尺寸(um) × 拍摄距离(m) / 焦距(mm)
# 量纲上 μm→mm 要除 1000、m→mm 要乘 1000，两下正好抵消，所以式子只剩这三项。
# 默认取等效焦距 24 mm、像元 2.4 um 的机载相机**设计标定值**；实机标定后应更新，
# 历史量化值需同步重算（面积是 GSD 的平方项，改一点动一片）。
CAM_FOCAL_MM = 24.0
CAM_PIXEL_UM = 2.4
CAM_SENSOR_MP = 20.0

# 做像素级分割的病害类别。说明书 §6.2/§3.3：图像分割只面向裂缝类，剥落、露筋、
# 蜂窝麻面等区域性病害只在检测阶段输出边界框与置信度，不换算面积。
SEGMENT_CLASSES = ("crack",)


def gsd_mm_per_px(distance_m: float, focal_mm: float = CAM_FOCAL_MM,
                  pixel_um: float = CAM_PIXEL_UM, theta_deg: float = 0.0) -> float:
    """由拍摄距离反算地面分辨率 GSD（mm/像素）。

    theta_deg 为斜拍角：相机光轴不垂直于目标表面时，沿斜视方向的实际分辨率按
    GSD_θ = GSD / cosθ 放大（说明书 §6.2）。默认 0 表示正拍，结果与早先逐位相同。
    """
    if distance_m <= 0 or focal_mm <= 0:
        return 0.0
    base = pixel_um * distance_m / focal_mm
    if theta_deg:
        c = math.cos(math.radians(max(-89.0, min(89.0, theta_deg))))
        if c > 1e-6:
            base /= c
    return base


DEFAULT_SETTINGS: dict = {
    "detect_weights": "",          # 桥梁病害检测权重路径（.pt / .onnx）
    "segment_weights": "",         # 分割权重路径
    "conf_thres": 0.35,
    "iou_thres": 0.45,
    "device": "auto",              # auto / cpu / cuda:0
    "img_size": 960,
    "output_dir": str(OUT_DIR),
    "llm_provider": "openai",
    "llm_base_url": "https://api.openai.com/v1",
    "llm_api_key": "",
    "llm_model": "gpt-4o-mini",
    "llm_enabled": False,
    "last_username": "",
    "last_image_dir": "",        # 上次打开图片选择框时所在的目录
    "report_org": "",            # 编制单位；留空则用 REPORT_ORG
}

# 所有在线规划服务都走 OpenAI Chat Completions 兼容协议；这里的预设只负责
# 填充正确地址、模型名和界面文案，不绑定任何厂商 SDK。
LLM_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "label": "OpenAI 兼容接口",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "deepseek": {
        "label": "DeepSeek（OpenAI 兼容）",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "label": "阿里云百炼 DashScope",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def normalize_llm_provider(value: str | None) -> str:
    """兼容旧版中文配置，并收敛到稳定的服务商标识。"""
    raw = (value or "").strip().lower()
    if raw in LLM_PROVIDER_PRESETS:
        return raw
    if "deepseek" in raw or "深度求索" in raw:
        return "deepseek"
    if "dashscope" in raw or "通义" in raw or "百炼" in raw:
        return "dashscope"
    return "openai"


def llm_provider_label(settings: dict | None = None) -> str:
    s = settings if settings is not None else load_settings()
    provider = normalize_llm_provider(s.get("llm_provider"))
    return LLM_PROVIDER_PRESETS[provider]["label"]


def load_settings() -> dict:
    """读取设置，缺项用默认值补齐（升级后老配置文件不会缺键）。"""
    cfg = dict(DEFAULT_SETTINGS)
    try:
        if SETTINGS_PATH.exists():
            cfg.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


def llm_ready(settings: dict | None = None) -> bool:
    """设置里的配置是否足以走大模型规划。

    放在 config 而不是 `core/agent/planner.py`：登录页要在主窗口之前显示「当前规划
    来源」，而 `import core.agent` 会经 tools.py → backend.py 连带拉进 cv2 与 numpy，
    把一个重量级依赖挪进登录路径（`main.py` 是先建登录窗、后建主窗口的）。本模块只依赖
    标准库，登录页可以放心引。

    判据与 `LLMPlanner.available()` 是同一个，后者改成调用这里，两处不会走样。
    """
    s = settings if settings is not None else load_settings()
    return bool(s.get("llm_enabled") and s.get("llm_api_key")
                and (s.get("llm_base_url") or "").strip())


def save_settings(cfg: dict) -> None:
    merged = dict(DEFAULT_SETTINGS)
    merged.update(cfg or {})
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)


