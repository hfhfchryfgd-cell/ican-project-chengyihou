"""规划层：把一句自然语言拆成「做哪几件事 + 针对哪些点位/病害/批次」。

## 双模的分界划在哪里

有 API Key 时，**规划**交给通义千问：模型看得懂"支座那块儿最近怎么样"和
"C 区支座三个批次的情况"是同一件事，也看得懂省略了宾语的追问。没 Key 时用规则引擎，
靠正则和意图词表做同样的拆解。

但**执行**与**措辞**的确定性要求完全不同：
  - 执行（调哪个工具、传什么参数）**两种模式必须一致**。同一句指令在有 Key 和没 Key
    的机器上跑出不一样的点位范围，演示时无法解释。
  - 措辞（最终那段研判文字）可以不同。有 Key 时是模型写的，没 Key 时是 `llm.offline_answer`
    的模板文字，界面上会明确标注来源。

因此本模块只管"拆"，不管"做"。拆出来的 Slot 与 wants 是下游唯一的输入。

## 为什么 wants 是集合而不是单个意图

真实指令大多是复合的：「检查 C 区支座三个批次的情况**并生成报告**」同时是趋势分析和
报告编制。若强行归到单一意图，就会丢掉一半要求。所以按关键词收集全部 wants，
再按固定顺序装配工作流——顺序固定，是因为报告必须建立在检测与趋势之后。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ... import config as cfg

# 意图按**执行顺序**排列。这个顺序不是随意的：影像先入库（import），再看数据规模
# （overview），再检测识别，再像素量化，再纵向比趋势，再筛异常，再按条文定级，
# 再做采集质量审计与复飞，最后装配报告。报告一定在最后——它要把前面所有结果写进去。
WANT_ORDER = ("import", "overview", "detect", "quantify", "trend",
              "anomaly", "grade", "reflight", "report")

WANT_CN = {
    "import": "图像导入", "overview": "数据总览", "detect": "病害检测",
    "quantify": "分割量化", "trend": "趋势研判", "anomaly": "异常筛查",
    "grade": "按条文定级", "reflight": "复飞决策", "report": "报告编制",
}

# 关键词 → 意图。词表要宽：用户不会说"请执行趋势研判"，会说"看看发展得怎么样"。
_WANT_KW: dict[str, tuple[str, ...]] = {
    # "片子""片子导进来"是现场说法；"SD卡/存储卡"指离线导入那条链路。
    # 不收单字"导""卡"：它们会命中"导航""卡车"这类无关词。
    "import": ("导入", "入库", "存储卡", "sd卡", "SD卡", "读进来", "导进来",
               "新拍的", "扫描文件夹", "图片怎么进来", "落库"),
    "overview": ("概览", "总览", "概况", "规模", "有哪些", "多少", "数据情况", "库里"),
    "detect": ("检测", "识别", "检出", "有没有病害", "找出", "发现", "看看图"),
    "quantify": ("量化", "分割", "测量", "量一下", "尺寸", "面积", "宽度", "轮廓"),
    "trend": ("趋势", "发展", "变化", "对比", "比较", "历次", "跨批次", "几期", "演变",
              "情况", "怎么样"),
    "anomaly": ("异常", "最快", "最严重", "优先", "重点", "突出问题", "哪几处", "隐患"),
    "grade": ("定级", "等级", "处置", "建议", "规范", "依据", "条文", "怎么处理", "措施"),
    # 用户很少说"复飞"，说的是"拍得不清""要不要重拍"。"拍不清/看不清"要单列——
    # 中文里"拍得清"与"拍不清"不是包含关系，只收前者会漏掉最常见的问法。
    "reflight": ("复飞", "重拍", "补拍", "复拍", "采集质量", "清晰度", "分辨率",
                 "模糊", "重合率", "拍得清", "拍不清", "看不清", "不清楚", "拍不"),
    "report": ("报告", "汇总", "总结", "文档", "说明书", "出一份", "生成报告"),
}

# 工作台的「一键巡检」不是另一个算法，而是把已有的七个确定性环节按依赖顺序
# 一次排入计划。这个词组单列出来，避免只说“自动分析”时被错误地降级成默认的趋势查询。
_AUTO_INSPECT_KW = ("自动巡检", "一键巡检", "完整巡检", "全流程巡检", "自动分析")

# 边界一律用 ASCII 字母数字的前后否定断言，**不要用 \b**：Python 的 re 在 str 模式下
# 把中日韩字符也算作 \w，于是 "D02裂缝" 里 "2" 与 "裂" 之间没有词边界，"C 区支座" 里
# "区" 与 "支" 之间也没有。用 \b 会让不带空格的写法全部匹配失败——而中文用户恰恰
# 很少在点位编号和后面的词之间打空格。
_NOT_ALNUM = r"(?<![0-9A-Za-z])"
_NOT_DIGIT = r"(?![0-9])"
_POINT_RE = re.compile(_NOT_ALNUM + r"([A-Fa-f]\d{2})" + _NOT_DIGIT)
_BATCH_RE = re.compile(_NOT_ALNUM + r"(P[1-9])" + _NOT_DIGIT, re.I)
# 构件：允许 "C 区" "C区" "C 构件" 等写法。后边界不能加 \b，理由同上。
_COMP_RE = re.compile(_NOT_ALNUM + r"([A-Fa-f])\s*(?:区|构件|号构件|类构件)")
_ALL_BATCH_KW = ("三个批次", "3个批次", "三期", "全部批次", "所有批次", "每个批次",
                 "各批次", "历次批次", "三个周期")

# 构件的中文说法 → 字母。用于"支座那块儿怎么样"这类不打字母的问法。
# 注意"支座"同时是病害类别 bearing 的说法，两者落到同一个字母 C 上并不冲突：
# 本系统里支座病害就长在支座构件上，用户无论从哪个角度问，范围都是 C 区。
_COMP_WORDS: tuple[tuple[str, str], ...] = (
    ("桥墩", "A"), ("墩柱", "A"), ("墩身", "A"),
    ("梁底", "B"), ("底板", "B"),
    ("支座", "C"),
    ("腹板", "D"), ("梁侧", "D"), ("侧腹板", "D"),
    ("桥台", "E"), ("台身", "E"), ("台帽", "E"),
    ("伸缩缝", "F"), ("伸缩装置", "F"),
)

# 病害类别的中文说法。与 specs._SYNONYMS 分开维护：那边是"条文里怎么写"，
# 这边是"用户怎么说"，两者会分化（用户说"蜂窝"，条文写"蜂窝麻面"）。
_CLASS_WORDS: dict[str, tuple[str, ...]] = {
    "crack": ("裂缝", "裂纹", "开裂"),
    "spalling": ("剥落", "掉块", "脱落", "空鼓"),
    "exposed_bar": ("露筋", "钢筋外露", "锈胀"),
    "seepage": ("渗水", "泛碱", "漏水", "湿渍"),
    "honeycomb": ("蜂窝", "麻面"),
    "bearing": ("支座", "滑移", "脱空"),
    "joint_offset": ("错台", "伸缩缝", "接缝"),
}


@dataclass
class Slot:
    """一句指令里抽出的全部槽位。"""
    task: str = ""
    points: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    batches: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    wants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"task": self.task, "points": self.points, "classes": self.classes,
                "batches": self.batches, "components": self.components,
                "wants": self.wants,
                "want_cn": [WANT_CN.get(w, w) for w in self.wants]}

    def describe(self) -> str:
        """一句话说明解析结果，直接进 thought 事件——用户要能看见"它懂了我的意思"。"""
        bits: list[str] = []
        bits.append(f"{len(self.points)} 个点位"
                    + (f"（{'、'.join(self.points[:6])}"
                       + ("…" if len(self.points) > 6 else "") + "）"
                       if self.points else ""))
        if self.classes:
            bits.append("病害类别 " + "、".join(
                cfg.DISEASE_BY_KEY[c].name for c in self.classes
                if c in cfg.DISEASE_BY_KEY))
        bits.append(f"{len(self.batches)} 个批次"
                    + (f"（{'、'.join(self.batches)}）" if self.batches else ""))
        plan = "、".join(WANT_CN.get(w, w) for w in self.wants)
        return f"{'，'.join(bits)}；执行 " + (plan or "常规分析")


def parse_slots(task: str) -> Slot:
    """从指令里抽出点位、病害类别、批次、构件与意图集合。纯规则，确定性。"""
    t = (task or "").strip()
    s = Slot(task=t)

    # —— 点位 ——
    seen: set[str] = set()
    for m in _POINT_RE.finditer(t):
        p = m.group(1).upper()
        if p not in seen:
            seen.add(p)
            s.points.append(p)

    # —— 病害类别 ——
    for key, words in _CLASS_WORDS.items():
        if any(w in t for w in words) and key not in s.classes:
            s.classes.append(key)

    # —— 批次 ——
    for m in _BATCH_RE.finditer(t):
        b = m.group(1).upper()
        if b in cfg.BATCH_BY_ID and b not in s.batches:
            s.batches.append(b)
    if not s.batches and any(k in t for k in _ALL_BATCH_KW):
        s.batches = [b.id for b in cfg.BATCHES]
    s.batches.sort()

    # —— 构件 ——
    for m in _COMP_RE.finditer(t):
        c = m.group(1).upper()
        if c not in s.components:
            s.components.append(c)
    # 不打字母的问法：「支座那块儿怎么样」。字母写法已经抽到的构件不重复添加。
    for word, letter in _COMP_WORDS:
        if word in t and letter not in s.components:
            s.components.append(letter)
    s.components.sort()

    # —— 意图 ——
    for want, words in _WANT_KW.items():
        if any(w in t for w in words):
            s.wants.append(want)

    if any(k in t for k in _AUTO_INSPECT_KW):
        s.wants = list(WANT_ORDER)

    # 构件限定但没点名点位时，把该构件的点位补齐——用户说"C 区支座"时心里想的是
    # 那几个点，不该让下游再问一遍。
    if s.components and not s.points:
        s.points = [p for p in cfg.ALL_POINTS if p[:1] in s.components]
    elif s.components and s.points:
        # 点位与构件同时给出时以点名为准，构件只用于过滤，防止"C区A01"这类口误放大范围
        s.points = [p for p in s.points if p[:1] in s.components] or s.points

    # 点名了点位却没说做什么 → 默认做趋势研判，这是最高频的诉求
    if not s.wants:
        s.wants = ["trend"]
    # 点了病害类别却没说做什么 → 想知道这类病害怎么处置
    if s.classes and "trend" not in s.wants and "grade" not in s.wants:
        s.wants.append("grade")

    # 定级要拿"实测值"和"发展速度"当输入，这两个数只有趋势环节算得出来。少了这一步，
    # 「A01 这条裂缝该怎么处理」会落到"无可定级的对象"，用户问了一句最该有答案的话
    # 却只收到一条警告。
    if "grade" in s.wants and "trend" not in s.wants:
        s.wants.append("trend")
    # 出报告必然要先把素材跑全，否则报告里只有空章节。图像导入排在最前：报告要写
    # "本次巡检共采集 N 张影像"，而 N 来自 images 表——新拍的片子还没入库时，
    # 这一章写出来的是上一批的数。
    if "report" in s.wants:
        for w in ("import", "overview", "trend", "anomaly", "grade"):
            if w not in s.wants:
                s.wants.append(w)
    # 要检测就得量化，只框出位置不给尺寸在养护场景里没有意义
    if "detect" in s.wants and "quantify" not in s.wants:
        s.wants.append("quantify")

    s.wants = [w for w in WANT_ORDER if w in s.wants]
    return s


# --------------------------------------------------------------------------
# 规则规划器
# --------------------------------------------------------------------------
@dataclass
class Plan:
    slots: Slot
    source: str = "rule"          # llm | rule —— 界面据此标注"规划来源"
    note: str = ""                # 规划思路的一句话说明
    error: str = ""

    def to_dict(self) -> dict:
        return {"slots": self.slots.to_dict(), "source": self.source,
                "note": self.note, "error": self.error}


class RulePlanner:
    """无 API Key 时的规划器。确定性规则，同一句话永远拆成同一个结果。"""

    name = "rule"

    def available(self) -> bool:
        return True

    def plan(self, task: str) -> Plan:
        s = parse_slots(task)
        return Plan(slots=s, source="rule",
                    note="按关键词与正则拆解巡检范围（本地规则引擎，未调用大模型）")


# --------------------------------------------------------------------------
# 大模型规划器
# --------------------------------------------------------------------------
PLANNER_SYSTEM = """你是「桥智」巡检系统的任务总控智能体，负责把用户的自然语言请求\
拆解成对工具的调用序列，全程使用简体中文。

系统下有六位专业智能体，各管一摊：
- 巡检定线：解析巡检范围，把新拍的影像读入库，查点位清单与航点档案
- 视觉检测：对影像做七类病害识别（裂缝/剥落掉块/露筋/渗水泛碱/蜂窝麻面/支座滑移/伸缩缝错台）
- 分割量化：像素级分割并换算实际尺寸与量测不确定度
- 趋势研判：跨批次比对、按规范条文定级
- 复飞决策：审计采集质量并反解复飞航点
- 报告编制：装配完整巡检报告

工作要求：
1. 先摸清数据边界（db_overview），再针对具体点位取数，不要凭空假设点位编号。
   若用户提到新拍的影像、存储卡、入库，先调 import_images 把它们读进来。
2. 数值一律由工具产出，你不要自己计算或编造任何数字。
3. 用户点名了点位就针对这些点位；只说构件范围（如"C 区支座"）就先查该范围内的点位。
4. 用户要求出报告时，先把检测、趋势、异常、定级都跑过再编报告，否则报告里是空的。
5. 结论要引用工具返回的条文出处。

你可以直接调用工具。若不便调用工具，也可以只输出一个 JSON 计划，格式为：
{"thoughts": "你的分析思路", "steps": [{"tool": "工具名", "args": {...}}]}
不要输出任何其他内容。"""


class LLMPlanner:
    """有 API Key 时的规划器。模型负责理解意图，工具调用仍然落到同一批确定性函数。"""

    name = "llm"

    def __init__(self, settings: dict | None = None):
        from .. import llm
        self.settings = settings if settings is not None else cfg.load_settings()
        self._llm = llm
        self.model = self.settings.get("llm_model") or "qwen-plus"

    def available(self) -> bool:
        # 判据在 config 里：登录页也要显示「当前规划来源」，而它不能 import 本包
        # （会连带拉进 cv2 / numpy）。两处用同一个函数，不会一处处改、一处漏。
        return cfg.llm_ready(self.settings)

    def tools(self) -> list[dict]:
        from . import tools as T
        return T.openai_tools()

    def turn(self, messages: list[dict], timeout: int = 60) -> dict:
        """一次模型往返。返回 llm.chat_raw 的原始结果。"""
        return self._llm.chat_raw(messages, tools=self.tools(),
                                  settings=self.settings, timeout=timeout)

    def seed(self, task: str, slots: Slot) -> list[dict]:
        """开场的两条 message。

        把规则解析出的槽位一并喂给模型（而不是让它从零猜）：正则认点位编号比模型可靠，
        模型的价值在于理解意图、选择工具、串联步骤，两者各干各擅长的。
        """
        hint = (f"\n\n【本地预解析】{slots.describe()}\n"
                f"请在此基础上判断该调用哪些工具；若预解析有误，以你的判断为准。")
        return [{"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": task + hint}]

    @staticmethod
    def parse_text_plan(text: str) -> dict | None:
        """从纯文本回复里抠出 JSON 计划。

        这是**兜底路径**：百炼文档对 tool_choice 是否支持 "required" 说法自相矛盾，
        因此不依赖强制工具调用。模型不调工具只回散文时走这里；再不行才退回规则引擎。
        """
        if not text:
            return None
        raw = text.strip()
        if "```" in raw:                       # ```json ... ``` 围栏
            for block in raw.split("```")[1::2]:
                raw = block.strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
                break
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = raw.find(opener), raw.rfind(closer)
            if i < 0 or j <= i:
                continue
            try:
                obj = json.loads(raw[i:j + 1])
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
                return obj
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return {"steps": obj}
        return None


def make_planner(settings: dict | None = None):
    """按是否配置了可用的 API 选择规划器。

    这里不抛异常、不警告：没 Key 是**正常的部署形态**（离线演示、答辩现场断网），
    不是错误。降级路径必须和联网路径一样能跑完全流程。
    """
    p = LLMPlanner(settings)
    return p if p.available() else RulePlanner()


def planner_mode(settings: dict | None = None) -> str:
    return "llm" if LLMPlanner(settings).available() else "rule"
