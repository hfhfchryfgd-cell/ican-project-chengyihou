"""智能体执行轨迹的数据结构。

这一层是桌面版（PyQt）与网页版（starlette + SSE）共用的**唯一**轨迹表示。约束有三条，
每条都对应一个真实会踩的坑：

1. **不依赖 PyQt、不依赖 asyncio。** 内核只产出事件，谁消费、怎么消费不关它的事。
   网页端靠任何线程池把生成器的 next() 挪出事件循环，桌面端靠 QThread 逐个取。
2. **data 必须能被 json.dumps 直接吃下。** 检测与分割的结果里有 numpy 数组
   （Detection 的坐标是 int、但 SegmentResult.mask 是整幅 ndarray），一旦原样塞进
   事件，SSE 推送会在序列化那一步炸掉整条连接。所有载荷过 _jsonable() 兜一遍。
3. **事件要能挂成树。** 比赛要求展示"智能体与其他协同工作情况"，因此调度器的
   delegate 事件是父节点，专家自己的 tool_call/tool_result 挂在它下面（parent=seq）。
   界面据此画出两级委派关系。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# 智能体花名册：key -> (中文名, 一句话职责, 界面配色)
# 配色沿用 theme.C 的语义色，网页端与桌面端取同一份，保证两边看起来是同一个系统。
AGENTS: dict[str, tuple[str, str, str]] = {
    "system":      ("系统",     "任务编排与状态广播",       "#5C6E8A"),
    "orchestrator": ("桥智·总控", "理解意图、编排专家、汇总结论", "#22D3EE"),
    "mission":     ("巡检定线", "把自然语言请求解析成结构化巡检范围", "#3B82F6"),
    "vision":      ("视觉检测", "识别七类病害并审计采集质量",     "#F87171"),
    "quant":       ("分割量化", "像素级提取轮廓并换算实际尺寸",   "#A78BFA"),
    "trend":       ("趋势研判", "跨批次比对并按规范条文定级",     "#FBBF24"),
    "refly":       ("复飞决策", "反解复飞航点与云台姿态",         "#FB923C"),
    "report":      ("报告编制", "装配巡检报告并登记产物",         "#34D399"),
}

# 事件类型。界面按 kind 决定图标、缩进与配色。
KINDS = (
    "run_start",    # 任务开始
    "thought",      # 思考（有 Key 来自模型，无 Key 来自规则模板）
    "delegate",     # 总控委派给某位专家
    "tool_call",    # 调用工具
    "tool_result",  # 工具返回
    "decision",     # 得出结论
    "artifact",     # 产出文件
    "warning",      # 需要人关注
    "error",        # 出错
    "run_end",      # 任务结束
)


def _jsonable(obj: Any) -> Any:
    """把任意对象递归转成 json.dumps 吃得下的形态。

    numpy 标量走 .item()，ndarray 只保留形状摘要——事件里绝不该出现整幅掩膜，
    需要看掩膜的场合一律先落盘、事件里只放路径。
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    item = getattr(obj, "item", None)          # numpy 标量
    if callable(item) and getattr(obj, "ndim", None) == 0:
        try:
            return item()
        except (ValueError, TypeError):
            pass
    shape = getattr(obj, "shape", None)         # ndarray
    if shape is not None:
        return f"<array{tuple(shape)}>"
    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(obj)


@dataclass
class AgentEvent:
    """执行轨迹上的一步。"""

    seq: int = 0
    ts: float = 0.0
    t_rel: float = 0.0
    agent: str = "system"
    kind: str = "thought"
    title: str = ""
    detail: str = ""
    data: dict = field(default_factory=dict)
    duration_ms: int = 0
    level: str = "info"                 # info | ok | warn | error
    parent: int | None = None           # 父事件 seq，用于委派树
    planner: str = "rule"               # llm | rule —— 界面据此标"规划来源"
    call_id: str = ""                   # 配对 tool_call 与 tool_result

    @property
    def agent_cn(self) -> str:
        return AGENTS.get(self.agent, (self.agent, "", ""))[0]

    @property
    def color(self) -> str:
        return AGENTS.get(self.agent, ("", "", "#5C6E8A"))[2]

    def to_dict(self) -> dict:
        d = {
            "seq": self.seq, "ts": self.ts, "t_rel": round(self.t_rel, 3),
            "agent": self.agent, "agent_cn": self.agent_cn, "color": self.color,
            "kind": self.kind, "title": self.title, "detail": self.detail,
            "data": _jsonable(self.data), "duration_ms": self.duration_ms,
            "level": self.level, "parent": self.parent,
            "planner": self.planner, "call_id": self.call_id,
        }
        return d

    def sse(self) -> str:
        """SSE 帧。ensure_ascii=False 让中文在浏览器 Network 面板里可读，
        调试时省掉一次 unicode 转义的心算。"""
        return "data: " + json.dumps(self.to_dict(), ensure_ascii=False) + "\n\n"


class AgentTracer:
    """事件收集器。seq / 时间戳 / 回调解耦都收在这里，规划器与专家只管 emit。"""

    def __init__(self, on_event: Callable[[AgentEvent], None] | None = None,
                 max_events: int = 400):
        self.events: list[AgentEvent] = []
        self._on_event = on_event
        self._max = max_events
        self._t0 = time.time()
        self._seq = 0
        self.planner = "rule"

    # -- 产出 ---------------------------------------------------------------
    def emit(self, kind: str, title: str, agent: str = "system", *,
             detail: str = "", data: dict | None = None, level: str = "info",
             parent: int | None = None, duration_ms: int = 0,
             call_id: str = "") -> AgentEvent:
        self._seq += 1
        ev = AgentEvent(
            seq=self._seq, ts=time.time(), t_rel=time.time() - self._t0,
            agent=agent, kind=kind, title=title, detail=detail,
            data=data or {}, level=level, parent=parent,
            planner=self.planner, call_id=call_id, duration_ms=duration_ms,
        )
        # 超过上限就不再回调了，但仍然记账。工具连续输出时界面不该被刷爆。
        if len(self.events) < self._max:
            self.events.append(ev)
            if self._on_event:
                self._on_event(ev)
        return ev

    # -- 计时 ---------------------------------------------------------------
    def timer(self) -> "StepTimer":
        return StepTimer()

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.events]

    def dumps(self, **kw) -> str:
        return json.dumps(self.to_list(), ensure_ascii=False, **kw)

    def save_jsonl(self, path) -> None:
        """把整条轨迹写盘。报告页可以附一份"本次智能体执行轨迹"，
        出问题时也是复盘材料。"""
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


class StepTimer:
    """with 块计时，毫秒。用于给 tool_call 填 duration_ms。"""

    def __init__(self):
        self.ms = 0

    def __enter__(self) -> "StepTimer":
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = int((time.perf_counter() - self._t) * 1000)
