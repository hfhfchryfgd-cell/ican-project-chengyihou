"""深色科技风主题：色彩令牌 + 全局 QSS + matplotlib 图表主题。

设计口径与参考系统《基于 YOLOv11 的无人机航拍小目标检测系统》保持一致：
深蓝底 + 青蓝霓虹强调 + 圆角卡片 + 渐变选中态。所有颜色只在 C 里定义一次，
QSS、图表、检测框绘制统一从这里取，避免出现两套色板。
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 色彩令牌
# --------------------------------------------------------------------------
C = {
    # 背景层次（由深到浅）
    "app":          "#0B1220",
    "sidebar_top":  "#0E1930",
    "sidebar_bot":  "#0A1120",
    "panel":        "#111C2E",
    "card":         "#16213A",
    "card_hi":      "#1B2946",
    "input":        "#0D1728",
    "hover":        "#1E2C4A",

    # 描边
    "border":       "#22314F",
    "border_hi":    "#2F4266",

    # 品牌强调
    "accent":       "#22D3EE",
    "accent_2":     "#3B82F6",
    "accent_dim":   "#155E75",
    "brand_a":      "#2563EB",
    "brand_b":      "#22D3EE",

    # 文字
    "text":         "#E8EEF9",
    "text_dim":     "#8FA3C0",
    "text_muted":   "#5C6E8A",
    "text_on":      "#04121F",

    # 语义色
    "ok":           "#34D399",
    "warn":         "#FBBF24",
    "danger":       "#F87171",
    "info":         "#60A5FA",
    "purple":       "#A78BFA",
}

# 中文字体族**只在这里定义一次**。这条清单原先在仓库里有五份副本（本文件、
# main.py 的 QFont、ui/graphics.py、p_agent.py），改字体会漏掉某
# 一处，表现是同一个界面里两种中文字体——比改错了还难查。其余四处一律 import 这里。
#
# 界面中文用宋体：宋体自带小字号点阵，10~14px 下笔画比雅黑锐利，屏读更清楚。
# 雅黑留在回退链里而不是直接落 serif —— 宋体缺失时该退回另一种中文字体，
# 落到 serif 只会得到一串方框（offline 环境下这个坑已经踩过一次）。
FONT_FAMILIES = ["SimSun", "宋体", "Songti SC",
                 "Microsoft YaHei UI", "Microsoft YaHei", "sans-serif"]
# 数字与路径仍走等宽：宋体没有等宽变体，表格里的数字换宋体后列会对不齐。
MONO_FAMILIES = ["Cascadia Mono", "Consolas", "Courier New", "monospace"]


def _stack(families: list[str]) -> str:
    """把族清单拼成 QSS 的 font-family 串。"""
    return ", ".join(f'"{f}"' for f in families)


FONT_STACK = _stack(FONT_FAMILIES)
MONO_STACK = _stack(MONO_FAMILIES)


def apply_app_font(app) -> None:
    """给 QApplication 设默认字体。

    桌面端有两个入口（main.py 与 ui/login.py 的 run_login），早先只有前者调了
    QApplication.setFont，后者漏了——于是 run_login 路径下所有**裸 QFont()**（如
    ui/widgets.py 的图注合成）都落到 Qt 默认族，同一个界面两种中文字体。抽到
    这里由两个入口共用，就不存在"漏调一处"的可能了。

    QSS 里带 font-family 的控件不依赖这个默认值；它管的是没被 QSS 规则命中的
    那些控件与 QPainter 自绘文字。
    """
    from PyQt6.QtGui import QFont

    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    # 9pt 是雅黑时代的取值；宋体的字面比雅黑小一号，同磅数看着更小，抬到 10pt。
    font.setPointSizeF(10)
    # 宋体是衬线体，全像素对齐在深色底上会把它渲染得毛糙。
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)


def _grad(a: str, b: str, horizontal: bool = True) -> str:
    """生成 QSS 线性渐变串。"""
    x2, y2 = ("1", "0") if horizontal else ("0", "1")
    return f"qlineargradient(x1:0, y1:0, x2:{x2}, y2:{y2}, stop:0 {a}, stop:1 {b})"


# --------------------------------------------------------------------------
# 全局样式表
# --------------------------------------------------------------------------
def build_qss() -> str:
    c = C
    return f"""
* {{
    font-family: {FONT_STACK};
    outline: none;
}}

QWidget {{
    background: transparent;
    color: {c['text']};
    font-size: 14px;
}}

QMainWindow, QDialog, #AppRoot {{
    background: {c['app']};
}}

/* ---------- 侧边栏 ---------- */
#Sidebar {{
    background: {_grad(c['sidebar_top'], c['sidebar_bot'], horizontal=False)};
    border-right: 1px solid {c['border']};
}}
#BrandTitle   {{ font-size: 18px; font-weight: 700; color: {c['text']}; }}
/* 下面这一组 ≤12px 的字号都比雅黑时代抬了 1px（11.5→12.5、10→11）：宋体是
   衬线体，同磅数的字面比雅黑小、笔画也更细，原样搬过来会整体发虚。 */
#BrandSub     {{ font-size: 11px;  color: {c['text_muted']}; letter-spacing: 1.5px; }}
#NavGroup     {{ font-size: 12.5px; color: {c['text_muted']}; letter-spacing: 1px;
                 padding: 0 6px; }}
#UserCard     {{ background: {c['card']}; border: 1px solid {c['border']};
                 border-radius: 10px; }}
#UserName     {{ font-size: 14px; font-weight: 600; color: {c['text']}; }}
#UserRole     {{ font-size: 12.5px; color: {c['text_muted']}; }}

/* ---------- 登录页 ---------- */
/* 底色与网页版 `.login-hero` 的 linear-gradient(160deg,#0E1930,#0A1120) 同一组色，
   斜向也是照抄的：纯竖直就成了侧边栏，两栏会看起来是同一个东西。
   那两团品牌色辉光不在这里——一条 QSS 规则只收一层画刷，网页版那面底是「两团
   radial + 一层 linear」叠出来的，叠不出来，由 HeroPanel.paintEvent 补画。

   这六条必须写在 build_qss() 里而不是控件上：build_qss() **没有裸 QLabel 规则**，
   只有 `QWidget {{ font-size: 13.5px }}`，不带 objectName 的标签拿不到字号与弱化色。
   另外注意别再用「没有选择器的样式表」——Qt 会把它应用到整棵子树，且优先级高于
   应用级样式表（早先登录页左栏就是这么写的，见 ui/login.py 的 HeroPanel 注释）。 */
#HeroPanel {{
    background: qlineargradient(x1:0, y1:0, x2:0.6, y2:1,
                                stop:0 {c['sidebar_top']}, stop:1 {c['sidebar_bot']});
    border-right: 1px solid {c['border']};
}}
/* 字号逐条抄网页版 app.css：h1 30px、.sub 13.5 改 12.5、.login-box h2 19px、
   .tip 12.5px。改这里要连 app.css 一起改，两边差 2px 就是"两套皮肤"。 */
#HeroTitle   {{ font-size: 30px; font-weight: 700; color: {c['text']}; }}
#HeroSub     {{ font-size: 12.5px; color: {c['text_dim']}; }}
#HeroPoint   {{ font-size: 12.5px; color: {c['text_dim']}; }}
#HeroFoot    {{ font-size: 12.5px; color: {c['text_muted']}; }}
#FormTitle   {{ font-size: 19px; font-weight: 700; color: {c['text']}; }}
#FormSub     {{ font-size: 12.5px; color: {c['text_muted']}; }}

QPushButton#NavItem {{
    text-align: left;
    /* 左侧 40px 是给线稿图标留的：图标由 NavButton.paintEvent 画在 x=15 处，
       宽 18px，文字由这条 padding 推到 40px 起——两个数值要一起改。
       早先这里没有图标，文字带两个前导空格凑出缩进。 */
    padding: 0 12px 0 40px;
    min-height: 40px;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 8px;
    color: {c['text_dim']};
    font-size: 14px;
}}
QPushButton#NavItem:hover {{
    background: {c['hover']};
    color: {c['text']};
}}
QPushButton#NavItem:checked {{
    background: {_grad('#1D3A63', '#123047')};
    border-left: 3px solid {c['accent']};
    color: {c['text']};
    font-weight: 600;
}}

/* ---------- 顶部横幅 ---------- */
/* 竖直渐变而不是纯色：横幅是每页第一眼看到的东西，一道极浅的渐变就够把它和
   下方的卡片区分开。刻度的齿由 PageHeader.paintEvent 画，不在这里。 */
#Header {{
    background: {_grad(c['card_hi'], c['panel'], horizontal=False)};
    border-bottom: 1px solid {c['border']};
}}
#PageTitle    {{ font-size: 18px; font-weight: 700; color: {c['text']}; }}
#PageSubtitle {{ font-size: 12.5px; color: {c['text_dim']}; }}

/* ---------- 面板 / 卡片 ---------- */
#Panel {{
    background: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 12px;
}}
#Card {{
    background: {c['card']};
    border: 1px solid {c['border']};
    border-radius: 10px;
}}
#Card:hover {{ border: 1px solid {c['border_hi']}; }}
#StatCard {{
    background: {_grad('#17253F', '#141F36', horizontal=False)};
    border: 1px solid {c['border']};
    border-radius: 10px;
}}
#StatCardAccent {{
    background: {_grad('#173A50', '#152A44', horizontal=False)};
    border: 1px solid {c['accent_dim']};
    border-radius: 10px;
}}
#CardTitle    {{ font-size: 14px; font-weight: 600; color: {c['text']}; }}
#PanelTitle   {{ font-size: 14px; font-weight: 600; color: {c['text_dim']};
                 letter-spacing: 1px; }}
#StatLabel    {{ font-size: 12.5px; color: {c['text_dim']}; }}
#StatValue    {{ font-size: 24px; font-weight: 700; color: {c['text']}; }}
#StatValueAccent {{ font-size: 24px; font-weight: 700; color: {c['accent']}; }}
#StatUnit     {{ font-size: 12.5px; color: {c['text_muted']}; }}
#Hint         {{ font-size: 12.5px; color: {c['text_muted']}; }}
#Mono         {{ font-family: {MONO_STACK}; font-size: 12.5px; color: {c['text_dim']}; }}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {c['card']};
    border: 1px solid {c['border_hi']};
    border-radius: 8px;
    padding: 7px 14px;
    color: {c['text_dim']};
    font-size: 12.5px;
}}
QPushButton:hover  {{ background: {c['hover']}; color: {c['text']};
                      border: 1px solid {c['accent_dim']}; }}
QPushButton:pressed {{ background: {c['input']}; }}
QPushButton:disabled {{ color: {c['text_muted']}; border: 1px solid {c['border']};
                        background: {c['panel']}; }}

QPushButton[accent="primary"] {{
    background: {_grad(c['brand_a'], '#1D4ED8')};
    border: 1px solid {c['brand_a']};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[accent="primary"]:hover {{
    background: {_grad('#3B82F6', c['brand_b'])};
    border: 1px solid {c['accent']};
}}
QPushButton[accent="primary"]:disabled {{
    background: {c['panel']}; color: {c['text_muted']};
    border: 1px solid {c['border']};
}}

QPushButton[accent="cyan"] {{
    background: {_grad('#0E7490', '#155E75')};
    border: 1px solid {c['accent']};
    color: #ECFEFF;
    font-weight: 600;
}}
QPushButton[accent="cyan"]:hover {{ background: {_grad('#0891B2', '#0E7490')}; }}

QPushButton[accent="danger"] {{
    background: transparent;
    border: 1px solid #7F2E38;
    color: {c['danger']};
}}
QPushButton[accent="danger"]:hover {{ background: #3A1A22; border-color: {c['danger']}; }}

QPushButton#Ghost {{ background: transparent; border: 1px solid transparent; }}

/* 可选中的小标签（复飞决策页的「审计范围」）。选中态必须与未选中拉开差距，
   否则一屏几十个 chip 谁被选中全靠记——那是控件没尽到责任。 */
QPushButton#Chip {{
    background: transparent;
    border: 1px solid {c['border_hi']};
    border-radius: 11px;
    padding: 2px 9px;
    color: {c['text_muted']};
    font-size: 12.5px;
}}
QPushButton#Chip:hover {{ border-color: {c['accent_dim']}; color: {c['text_dim']}; }}
QPushButton#Chip:checked {{
    background: {c['accent_dim']};
    border: 1px solid {c['accent']};
    color: {c['text_on']};
    font-weight: 600;
}}

/* ---------- 输入控件 ---------- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit {{
    background: {c['input']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 7px 10px;
    color: {c['text']};
    selection-background-color: {c['accent_dim']};
    selection-color: {c['text']};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus {{
    border: 1px solid {c['accent']};
}}
QLineEdit:disabled, QComboBox:disabled, QDateEdit:disabled {{ color: {c['text_muted']}; }}
QLineEdit[echoMode="2"] {{ letter-spacing: 2px; }}

QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {c['text_dim']};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {c['card']};
    border: 1px solid {c['border_hi']};
    border-radius: 8px;
    selection-background-color: {c['accent_dim']};
    color: {c['text']};
    padding: 4px;
    outline: none;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {c['card_hi']}; border: none; width: 16px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-bottom: 4px solid {c['text_dim']};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-top: 4px solid {c['text_dim']};
}}

QSlider::groove:horizontal {{
    height: 4px; background: {c['border']}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {_grad(c['brand_a'], c['accent'])}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #FFFFFF; width: 13px; height: 13px;
    margin: -5px 0; border-radius: 6px; border: 2px solid {c['accent']};
}}

QCheckBox {{ spacing: 7px; color: {c['text_dim']}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {c['border_hi']}; background: {c['input']};
}}
QCheckBox::indicator:checked {{
    background: {_grad(c['brand_a'], c['accent'])};
    border: 1px solid {c['accent']};
}}

/* ---------- 表格 ---------- */
QTableWidget, QTableView {{
    background: {c['card']};
    alternate-background-color: {c['card_hi']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    gridline-color: transparent;
    color: {c['text']};
    selection-background-color: {c['accent_dim']};
    selection-color: #FFFFFF;
}}
QTableWidget::item, QTableView::item {{ padding: 6px 8px; border: none; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background: {c['panel']};
    color: {c['text_dim']};
    border: none;
    border-bottom: 1px solid {c['border_hi']};
    padding: 7px 8px;
    font-size: 12.5px;
    font-weight: 600;
}}
QHeaderView::section:first {{ border-top-left-radius: 8px; }}
QHeaderView::section:last  {{ border-top-right-radius: 8px; }}
QTableCornerButton::section {{ background: {c['panel']}; border: none; }}

/* ---------- 标签页 ---------- */
QTabWidget::pane {{ border: none; background: transparent; }}
QTabBar::tab {{
    background: transparent;
    color: {c['text_muted']};
    padding: 7px 16px;
    margin-right: 4px;
    border: none;
    border-radius: 8px;
    font-size: 12.5px;
}}
QTabBar::tab:hover {{ color: {c['text']}; background: {c['card']}; }}
QTabBar::tab:selected {{
    background: {_grad('#1D3A63', '#123047')};
    color: {c['accent']};
    font-weight: 600;
}}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {c['border_hi']}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {c['accent_dim']}; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {c['border_hi']}; border-radius: 4px; min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {c['accent_dim']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---------- 其他 ---------- */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {c['border_hi']}; }}

QProgressBar {{
    background: {c['input']}; border: 1px solid {c['border']};
    border-radius: 6px; height: 8px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{
    background: {_grad(c['brand_a'], c['accent'])}; border-radius: 6px;
}}

QStatusBar {{ background: {c['panel']}; border-top: 1px solid {c['border']};
              color: {c['text_muted']}; font-size: 12.5px; }}
QStatusBar::item {{ border: none; }}

QToolTip {{
    background: {c['card']}; color: {c['text']};
    border: 1px solid {c['accent_dim']}; border-radius: 6px; padding: 5px 8px;
}}

QMenu {{
    background: {c['card']}; border: 1px solid {c['border_hi']};
    border-radius: 8px; padding: 5px;
}}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; color: {c['text_dim']}; }}
QMenu::item:selected {{ background: {c['accent_dim']}; color: #FFFFFF; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

QFrame[role="hline"] {{ background: {c['border']}; max-height: 1px; border: none; }}

/* 状态徽标 */
#Badge {{
    border-radius: 9px; padding: 2px 10px; font-size: 12.5px; font-weight: 600;
}}
#Badge[demo="true"]  {{ background: #3A2E12; color: {c['warn']};
                        border: 1px solid #6B5312; }}
#Badge[demo="false"] {{ background: #10322A; color: {c['ok']};
                        border: 1px solid #1C5E4A; }}
"""


# --------------------------------------------------------------------------
# matplotlib 图表主题（与 QSS 同一套色板）
# --------------------------------------------------------------------------
def _mpl_cjk_families() -> list[str]:
    """挑一组 matplotlib 真的能找到的中文字体族。

    matplotlib 的字体解析是**静默失败**的：`font.sans-serif` 里写一个装不上的族名，
    它不抛异常，只在 stderr 丢一句 findfont 警告，然后把中文全渲染成方框。界面上
    看起来像乱码、日志里什么也没有——本项目的 Qt 无窗口渲染已经踩过同一个坑。
    所以这里显式探一次：宋体探不到就退回雅黑链，而不是让图表变成一屏豆腐块。
    """
    from matplotlib import font_manager

    for fam in ("SimSun", "Microsoft YaHei", "SimHei"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=fam),
                                  fallback_to_default=False)
            # 英文与数字仍交给 DejaVu：宋体的西文字面偏窄，坐标轴刻度挤在一起。
            return [fam, "DejaVu Sans"]
        except Exception:                                  # noqa: BLE001
            continue
    return ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


def apply_mpl_theme() -> None:
    """把深色主题套到 matplotlib，使嵌入的图表与界面融为一体。"""
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor": C["card"],
        "axes.facecolor": C["card"],
        "savefig.facecolor": C["card"],
        "axes.edgecolor": C["border"],
        "axes.labelcolor": C["text_dim"],
        "axes.titlecolor": C["text"],
        "text.color": C["text"],
        "xtick.color": C["text_muted"],
        "ytick.color": C["text_muted"],
        "grid.color": C["border"],
        "grid.alpha": 0.55,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "legend.frameon": False,
        "legend.labelcolor": C["text_dim"],
        "figure.autolayout": True,
        "font.sans-serif": _mpl_cjk_families(),
        # 这一行不能删：宋体没有 U+2212（Unicode 减号），负号会渲染成方框。
        # 关掉它 matplotlib 改用 ASCII 连字符-减号，宋体有这个字形。
        "axes.unicode_minus": False,
    })
