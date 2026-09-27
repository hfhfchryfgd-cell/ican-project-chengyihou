"""功能页。

顺序即左侧导航顺序：先进入智能体工作台，再按巡检链路
"检测 → 分割量化 → 长周期监测"逐页展开，最后是历史与设置。

复飞决策与报告生成不做成独立页面：它们由智能体工作台按任务拆解自动调起，
产物则在检测历史里统一查看，能力本身完整保留在内核与工具层。

桌面版为当前交付版本，导航只保留现场巡检演示所需视图。
"""

from .p_agent import AgentPage
from .p_detect import DetectPage
from .p_history import HistoryPage
from .p_segment import SegmentPage
from .p_settings import SettingsPage
from .p_trend import TrendPage

__all__ = ["AgentPage", "DetectPage",
           "SegmentPage", "TrendPage", "HistoryPage", "SettingsPage"]
