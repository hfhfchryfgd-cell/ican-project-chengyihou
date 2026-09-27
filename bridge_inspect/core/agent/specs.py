"""桥梁养护规范条文库（本地轻量检索）。

## 关于准确性的声明——请先读这一段

**本库中的条文号与正文均为示意性整理，用于演示"智能体按规范条文定级并标注依据"
这一机制，不等于规范原文。** 正式参赛材料与任何实际养护决策前，必须逐条核对
现行规范原文与文号（《铁路桥隧建筑物修理规则》铁运〔2010〕38 号，后续如有修订
以现行版本为准；《公路桥梁技术状况评定标准》JTG/T H21）。

界面与报告中引用条文时，卡片上常驻「示意文本，以规范原文为准」角标，报告末尾
的说明章节也复述这句。这不是免责套话——把示意性文字当条文原文引用给养护单位，
是会造成实际后果的。

## 为什么用关键词匹配而不是向量库

七类病害 × 六类构件的组合是**封闭且很小**的（几十条），类别本身就是最强的检索键。
上向量库要引入 sentence-transformers 或调 embedding 接口，既加依赖又加一次网络
往返，而召回质量在这个规模上不会更好。类别命中 + 构件命中 + 关键词同义词展开，
三档加权打分足够，而且**打分过程可以被评委逐条复算**。

## 与复飞闭环的咬合点

第 20 条（用于裂缝宽度量测的影像，地面分辨率不应低于所需量测精度的 1/3）正是
`reflight.py` 里 Q5 判据的规范出处。研判智能体引用它 → 采集质量审计发现不满足 →
复飞智能体解算新航点。这条链是设计上咬得最紧的一处，讲的时候要连起来讲。
"""

from __future__ import annotations

from dataclasses import dataclass

# 引用时统一挂的角标。放在这里而不是各调用点，保证不会漏。
DISCLAIMER = "示意文本，以规范原文为准"

RAIL = "《铁路桥隧建筑物修理规则》"
HIGHWAY = "《公路桥梁技术状况评定标准》JTG/T H21"
PRACTICE = "无人机桥梁巡检作业通行做法"


@dataclass(frozen=True)
class Clause:
    cid: str                              # 内部编号
    source: str                           # 规范名
    code: str                             # 条文号
    title: str
    text: str                             # 条文要点（示意性摘录）
    cls_keys: tuple[str, ...] = ()        # 适用病害类别；空 = 通用
    components: tuple[str, ...] = ()      # 适用构件前缀；空 = 不限
    limit_mm: float | None = None
    action: str = "观察"                  # 加固 / 维修 / 观察
    keywords: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        return f"{self.source}{self.code}"

    def to_dict(self) -> dict:
        return {
            "cid": self.cid, "source": self.source, "code": self.code,
            "title": self.title, "text": self.text, "citation": self.citation,
            "limit_mm": self.limit_mm, "action": self.action,
            "disclaimer": DISCLAIMER,
        }


CLAUSES: tuple[Clause, ...] = (
    # ── 裂缝 ────────────────────────────────────────────────────────────
    Clause("CR-01", RAIL, "第3.2.1条", "裂缝超限处置",
           "钢筋混凝土构件裂缝宽度超过限值时，应查明原因，采用灌注、封闭或加固处理。",
           ("crack",), limit_mm=0.30, action="加固"),
    Clause("CR-02", RAIL, "第3.2.2条", "裂缝接近限值",
           "裂缝宽度已达到限值八成以上或持续发展者，应缩短检查周期并作专项观测。",
           ("crack",), limit_mm=0.24, action="维修", keywords=("发展", "扩展")),
    Clause("CR-03", RAIL, "第3.2.5条", "裂缝观测记录",
           "裂缝观测应记录长度、宽度、走向与所在部位，并附有比例尺的影像资料。",
           ("crack",), action="观察"),
    Clause("CR-04", PRACTICE, "第20条", "量测影像的分辨率要求",
           "用于裂缝宽度量测的影像，其地面分辨率不应低于所需量测精度的 1/3，"
           "否则量测结果不具备判定超限与否的统计意义。",
           ("crack",), action="维修", keywords=("分辨率", "GSD", "精度", "量测")),
    # ── 剥落掉块 ────────────────────────────────────────────────────────
    Clause("SP-01", RAIL, "第3.3.2条", "剥落修补",
           "混凝土剥落掉块应及时凿除松散层、清理至密实基面后修补，防止钢筋进一步外露。",
           ("spalling",), action="维修"),
    Clause("SP-02", RAIL, "第3.3.4条", "剥落面积较大",
           "剥落面积大于 0.5 m² 或深度超过保护层厚度时，应编制专项修补方案。",
           ("spalling",), action="加固", keywords=("面积", "深度")),
    # ── 露筋 ────────────────────────────────────────────────────────────
    Clause("EB-01", RAIL, "第3.4.1条", "露筋除锈阻锈",
           "钢筋外露、锈蚀应除锈并涂阻锈剂后，用高强砂浆或聚合物砂浆恢复保护层。",
           ("exposed_bar",), action="维修"),
    Clause("EB-02", RAIL, "第3.4.3条", "主筋截面损失",
           "主筋外露且截面损失率大于 5% 时，应进行承载力验算并加固。",
           ("exposed_bar",), action="加固", keywords=("截面", "损失", "承载力")),
    # ── 渗水泛碱 ────────────────────────────────────────────────────────
    Clause("SW-01", RAIL, "第3.5.1条", "渗水先治水",
           "渗水、泛碱应排查防水层破损与排水通道堵塞，先治水后治面。",
           ("seepage",), action="维修"),
    Clause("SW-02", RAIL, "第3.5.3条", "析出物清理",
           "析出物应清理并观察是否复现；伴随裂缝的渗水按裂缝条款处理。",
           ("seepage",), action="观察", keywords=("泛碱", "析出")),
    # ── 蜂窝麻面 ────────────────────────────────────────────────────────
    Clause("HC-01", RAIL, "第3.6.2条", "蜂窝麻面封闭",
           "蜂窝麻面等外观缺陷应作表面封闭处理，面积较大者采用聚合物砂浆修补。",
           ("honeycomb",), action="观察"),
    # ── 支座滑移 ────────────────────────────────────────────────────────
    Clause("BR-01", RAIL, "第4.2.1条", "支座检查项",
           "支座应检查位移、转角、脱空、剪切变形与钢材锈蚀，并复测支座位移量。",
           ("bearing",), ("C",), action="观察"),
    Clause("BR-02", RAIL, "第4.2.3条", "支座位移超限",
           "支座位移量超过设计允许值或出现明显脱空时，应顶升复位或更换支座。",
           ("bearing",), ("C",), action="加固", keywords=("位移", "脱空", "滑移")),
    # ── 伸缩缝错台 ──────────────────────────────────────────────────────
    Clause("JT-01", RAIL, "第4.3.2条", "伸缩装置维护",
           "伸缩装置应保持缝内清洁、伸缩自由，错台量与缝宽应定期量测记录。",
           ("joint_offset",), ("F",), action="观察"),
    Clause("JT-02", RAIL, "第4.3.4条", "错台超限处置",
           "伸缩缝错台量超过 5 mm 或影响行车平顺时，应清理杂物并调整伸缩装置。",
           ("joint_offset",), ("F",), limit_mm=5.0, action="维修",
           keywords=("错台", "平顺")),
    # ── 通用 / 作业条件 ─────────────────────────────────────────────────
    Clause("GN-01", HIGHWAY, "第5.1条", "技术状况评定",
           "按构件缺损的类型、程度与发展状况评定技术状况等级。", (), (), action="观察"),
    Clause("GN-02", HIGHWAY, "第5.3条", "缺损评分维度",
           "裂缝、剥落等缺损按「程度」与「发展」两个维度分别评分，两者独立不叠加。",
           ("crack", "spalling"), (), action="观察", keywords=("程度", "发展", "定级")),
    Clause("GN-03", HIGHWAY, "第4.2条", "检查分类与周期",
           "桥梁检查分为经常检查、定期检查与特殊检查，定期检查周期不超过 3 年。",
           (), (), action="观察", keywords=("周期", "定期检查")),
    Clause("GN-04", RAIL, "第2.1.4条", "检查记录要求",
           "检查记录应包含病害位置、尺寸、数量及影像资料，按同一基准比对历次记录。",
           (), (), action="观察", keywords=("记录", "基准", "比对")),
    Clause("GN-05", PRACTICE, "第19条", "同视场复拍条件",
           "同一航点复拍应保持相同站位与云台姿态，图像重叠率不低于 85%，"
           "以保证历次影像具备可比性。",
           (), (), limit_mm=0.85, action="维修",
           keywords=("重合率", "复拍", "同视场", "重叠率")),
)

CLAUSE_BY_ID = {c.cid: c for c in CLAUSES}

# 关键词同义词展开：用户/模型说的是口语，条文里写的是术语
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "crack": ("裂缝", "裂纹", "开裂", "龟裂"),
    "spalling": ("剥落", "掉块", "空鼓", "脱落"),
    "exposed_bar": ("露筋", "钢筋外露", "锈蚀", "锈胀"),
    "seepage": ("渗水", "泛碱", "湿渍", "析出", "漏水"),
    "honeycomb": ("蜂窝", "麻面", "外观缺陷"),
    "bearing": ("支座", "滑移", "位移", "脱空"),
    "joint_offset": ("错台", "伸缩缝", "伸缩装置", "接缝"),
}


def expand_keywords(terms: list[str] | tuple[str, ...]) -> list[str]:
    """把口语词展开成条文里可能出现的术语。"""
    out: list[str] = []
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        out.append(t)
        for _key, words in _SYNONYMS.items():
            if t in words or t == _key:
                out.extend(words)
                out.append(_key)
    # 去重但保持顺序
    seen: set[str] = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def search(classes: list[str] | tuple[str, ...] = (),
           components: list[str] | tuple[str, ...] = (),
           top_k: int = 3,
           extra_kw: list[str] | tuple[str, ...] = (),
           strict_component: bool = True) -> list[Clause]:
    """按类别 / 构件 / 关键词打分检索，返回最相关的 top_k 条。

    打分是三档加权：类别命中 +3、构件命中 +2、关键词命中 +1。通用条（不限类别、
    不限构件）给 0.5 保底分——它们在任何检索里都该垫底但不能缺席，因为像"检查
    记录要求""技术状况评定"这类条文对任何病害都成立。

    同分时**带具体限值的排前面**：能给出数字的条文对定级的支撑力度更大。

    strict_component 控制"限定了构件的条文，在调用方没给构件时怎么办"：
      True （默认）淘汰。适合**检索**场景——找不到构件就无从判断该条适不适用。
                       支座条文限定构件 C，用它去回一条桥墩的问题没有意义。
      False        保留。适合**引用**场景——已经知道是哪类病害了，要引的是
                       "这类病害按哪条处置"，与它长在哪个构件上无关。
    这个区分不是洁癖：伸缩缝条文（JT-01/JT-02）限定构件 F，演示库里却有一条
    错台长在 C03（支座）上。检索时带构件条件是对的，引用时若也带，
    JT 系列会整体消失，一条错台超限的结论最后只能引"同视场复拍条件"——引错了。
    """
    cls_set = {c for c in classes if c}
    comp_set = {(c or "").upper() for c in components if c}
    words = expand_keywords(list(extra_kw))

    scored: list[tuple[float, Clause]] = []
    for c in CLAUSES:
        score = 0.0
        if c.cls_keys:
            hit = cls_set & set(c.cls_keys)
            if not hit:
                # 类别不匹配且该条不是通用条 —— 直接淘汰。这条**没有**开关：
                # 不淘汰的话"渗水"的检索结果里会混进裂缝超限条文，
                # 研判智能体就可能拿裂缝的 0.30 mm 去定渗水的级。
                continue
            score += 3.0 * len(hit)
        else:
            score += 0.5
        if c.components:
            hit = comp_set & set(c.components)
            if not hit:
                if strict_component:
                    continue
                # 没给构件，无从判断适用性：不加分也不淘汰，靠类别分排序
            else:
                score += 2.0 * len(hit)
        elif comp_set:
            score += 0.2
        if words:
            hay = f"{c.title}{c.text}{c.code}{' '.join(c.keywords)}"
            score += sum(1.0 for w in words if w and w in hay)
        scored.append((score, c))

    # 同分时的排序依据，按影响力从大到小：
    #   1. 带具体限值的优先——能给出数字的条文对定级的支撑更硬；
    #   2. 只针对单一病害类别的优先——专指条文比兼述多类的条文更贴题。
    #      缺了这一条，「剥落掉块」会由《公路桥梁技术状况评定标准》第5.3条
    #      （同述裂缝与剥落两类、讲评分维度）排在《铁路桥隧建筑物修理规则》
    #      第3.3.2条（专讲剥落修补）前面：同为 3 分，靠 cid 字母序恰好蒙对一次，
    #      换个编号就会引出一条与处置无关的评定条款。
    #   3. 通用条（不限类别）的 cls_keys 为空，用 99 垫到最后。
    scored.sort(key=lambda sc: (-sc[0], -(sc[1].limit_mm or 0.0),
                                len(sc[1].cls_keys) or 99, sc[1].cid))
    return [c for _, c in scored[:max(1, top_k)]]


def for_disease(cls_key: str, component: str = "", top_k: int = 3) -> list[Clause]:
    """按病害类别取条文，供研判智能体给结论挂依据。

    **不做构件过滤**，即使调用方给了构件。这是引用而非检索：已经知道是哪类病害了，
    要引的是"这类病害按哪条处置"。构件过滤会把答案引错——演示库里 C03（支座）上有
    一条伸缩缝错台，错台条文限定构件 F，一带构件条件 JT-01/JT-02 就整体消失，
    一条"错台 10.92 mm 超限值 5 mm"的结论最后只能引到「同视场复拍条件」上去。
    需要按构件做适用性检索的场合请直接调 search（默认就是严格模式）。
    """
    return search([cls_key], [component] if component else [], top_k=top_k,
                  strict_component=False)


def limit_of(cls_key: str) -> float | None:
    """取该类别条文里出现的限值（与 config.DiseaseClass.width_limit_mm 互为印证）。"""
    for c in CLAUSES:
        if cls_key in c.cls_keys and c.limit_mm is not None:
            return c.limit_mm
    return None
