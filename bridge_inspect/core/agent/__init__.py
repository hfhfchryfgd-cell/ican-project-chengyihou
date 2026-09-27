"""多智能体内核。

## 这是什么

把「导入 → 检测 → 分割量化 → 长周期趋势 → 复飞决策 → 报告」这条既有链路，重新组织成
**一次自然语言任务**：总控智能体解析请求，委派给六位专业智能体，每一步的工具调用、
判定依据与耗时都作为事件流实时产出，末尾汇总结论并落盘产物。

## 与原有 core 的关系

本包**不重写任何算法**。检测、分割、GSD 换算、趋势聚合、报告装配全部复用
`backend.py` / `segment.py` / `llm.py` / `report.py` 里已经跑通的函数，
经 `tools.py` 包成有名称、有参数模式、有中文结论的工具。这样桌面版与网页版共用同一套
内核，算法也只有一份实现、一处修改。

## 依赖边界

除 `config` 与 `db` 外不依赖 PyQt，也不依赖 asyncio。事件流是生成器（`yield`），
桌面版用 QThread 消费、网页版写进 SSE，同一份代码两条通路。

## 两条运行路径

| | 规划 | 执行 | 措辞 |
|---|---|---|---|
| 配置了 API Key | 兼容 API（DeepSeek / 通义千问等） | 确定性工具 | 模型 |
| 未配置 Key | 本地规则引擎 | **同一批**确定性工具 | 规则模板 |

两条路径产出的事件形态一致，界面上只需标注「规划来源」，不需要分支渲染逻辑。

## 快速开始

    from bridge_inspect.core.agent import Orchestrator

    for ev in Orchestrator().stream("检查 C 区支座三个批次的情况并生成报告"):
        print(ev.seq, ev.agent_cn, ev.kind, ev.title)
"""

from .events import AGENTS, KINDS, AgentEvent, AgentTracer, StepTimer
from .orchestrator import AgentRun, Orchestrator, run_task
from .planner import Plan, RulePlanner, Slot, make_planner, parse_slots
from .tools import TOOLS, ToolResult, ToolSpec, catalog, openai_tools

__all__ = [
    "AGENTS", "KINDS", "AgentEvent", "AgentTracer", "StepTimer",
    "Orchestrator", "AgentRun", "run_task",
    "Plan", "RulePlanner", "Slot", "make_planner", "parse_slots",
    "TOOLS", "ToolResult", "ToolSpec", "catalog", "openai_tools",
]
