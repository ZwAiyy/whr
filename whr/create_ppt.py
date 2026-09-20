"""
生成天气知识图谱项目 PPT
运行: uv run create_ppt.py
输出: weather_knowledge_graph.pptx

"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn

# ── 配色方案 ──
BG_DARK = RGBColor(0x1A, 0x1A, 0x2E)       # 深蓝黑背景
BG_CARD = RGBColor(0x22, 0x22, 0x3A)       # 卡片背景
ACCENT_BLUE = RGBColor(0x4E, 0xC9, 0xF0)   # 亮蓝强调
ACCENT_GREEN = RGBColor(0x2E, 0xCC, 0x71)  # 绿色
ACCENT_ORANGE = RGBColor(0xF3, 0x9C, 0x12) # 橙色
ACCENT_RED = RGBColor(0xE7, 0x4C, 0x3C)    # 红色
ACCENT_PURPLE = RGBColor(0x9B, 0x59, 0xB6) # 紫色
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xBD, 0xBD, 0xBD)
DARK_GRAY = RGBColor(0x7A, 0x7A, 0x7A)

FONT_CN = "Microsoft YaHei"


def set_cjk_font(run, font_name):
    """设置中文字体"""
    run.font.name = font_name
    rPr = run._r.get_or_add_rPr()
    successors = {
        "a:ea": ("a:cs", "a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"),
        "a:cs": ("a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"),
    }
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.insert_element_before(el, *successors[tag])
        el.set("typeface", font_name)


def add_bg(slide, color=BG_DARK):
    """设置深色背景"""
    rect = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
    )
    rect.fill.solid()
    rect.fill.fore_color.rgb = color
    rect.line.fill.background()


def add_text(slide, text, left, top, width, height, size=18, color=WHITE,
             bold=False, align=PP_ALIGN.LEFT, font=FONT_CN):
    """添加文字框"""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    r = p.runs[0]
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    set_cjk_font(r, font)
    return tb


def add_bullet_list(slide, items, left, top, width, height, size=16, color=WHITE, bullet_color=ACCENT_BLUE):
    """添加带圆点的列表"""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"  {item}"
        p.space_after = Pt(6)
        r = p.runs[0]
        r.font.size = Pt(size)
        r.font.color.rgb = color
        set_cjk_font(r, FONT_CN)
    return tb


def add_card(slide, left, top, width, height, color=BG_CARD):
    """添加圆角卡片"""
    rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    rect.fill.solid()
    rect.fill.fore_color.rgb = color
    rect.line.fill.background()
    return rect


def add_accent_line(slide, left, top, width, color=ACCENT_BLUE):
    """添加强调色线条"""
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(4))
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()


def add_table(slide, headers, data, left, top, width, row_height=Inches(0.45)):
    """添加表格"""
    rows = len(data) + 1
    cols = len(headers)
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, row_height * rows)
    tbl = table_shape.table

    # 表头
    for c, h in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0x2C, 0x3E, 0x50)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(14)
                r.font.bold = True
                r.font.color.rgb = ACCENT_BLUE
                set_cjk_font(r, FONT_CN)

    # 数据行
    for ri, row in enumerate(data, start=1):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG_CARD if ri % 2 == 1 else RGBColor(0x2A, 0x2A, 0x45)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size = Pt(13)
                    r.font.color.rgb = WHITE
                    set_cjk_font(r, FONT_CN)

    # 去掉边框
    for ri in range(rows):
        for ci in range(cols):
            cell = tbl.cell(ri, ci)
            for border in ['a:lnL', 'a:lnR', 'a:lnT', 'a:lnB']:
                tc = cell._tc
                tcPr = tc.find(qn('a:tcPr'))
                if tcPr is not None:
                    ln = tcPr.find(qn(border))
                    if ln is not None:
                        ln.set('w', '0')
    return tbl


# ── 创建 PPT ──
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ============================================================
# Slide 1: 封面
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_accent_line(slide, Inches(1.5), Inches(2.3), Inches(3), ACCENT_BLUE)
add_text(slide, "基于 Neo4j 与大模型的", Inches(1.5), Inches(2.5), Inches(10), Inches(0.8),
         size=24, color=LIGHT_GRAY)
add_text(slide, "天气知识图谱构建", Inches(1.5), Inches(3.2), Inches(10), Inches(1),
         size=44, bold=True, color=WHITE)
add_text(slide, "从 CSV 数据到知识可视化", Inches(1.5), Inches(4.3), Inches(10), Inches(0.6),
         size=20, color=ACCENT_BLUE)
add_text(slide, "天气数据集  |  1,097 条记录  |  15 个字段  |  2023.6 ~ 2026.3",
         Inches(1.5), Inches(5.5), Inches(10), Inches(0.5), size=14, color=DARK_GRAY)

# ============================================================
# Slide 2: 项目整体架构
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "项目整体架构", Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 三个流程卡片
cards = [
    ("Protege + OWL", "本体建模  ->  SPARQL 查询", ACCENT_PURPLE, Inches(0.5)),
    ("大模型知识抽取", "DeepSeek / 通义千问 / MiMo", ACCENT_BLUE, Inches(4.5)),
    ("Neo4j 图数据库", "知识存储  ->  可视化图谱", ACCENT_GREEN, Inches(8.5)),
]
for title, desc, color, left in cards:
    add_card(slide, left, Inches(1.8), Inches(3.8), Inches(2.2))
    add_text(slide, title, left + Inches(0.3), Inches(2.0), Inches(3.2), Inches(0.5),
             size=20, bold=True, color=color)
    add_text(slide, desc, left + Inches(0.3), Inches(2.6), Inches(3.2), Inches(0.8),
             size=14, color=LIGHT_GRAY)

# 数据源
add_text(slide, "CSV 天气数据 (weather_data.csv)", Inches(3.5), Inches(4.5), Inches(6), Inches(0.5),
         size=18, bold=True, color=ACCENT_ORANGE, align=PP_ALIGN.CENTER)

# 流程箭头说明
items = [
    "原始 CSV 数据作为三个流程的统一输入",
    "OWL 本体用于定义领域概念和推理规则",
    "大模型自动抽取实体、关系和气象规律",
    "Neo4j 将所有知识以图的形式存储和可视化",
]
add_bullet_list(slide, items, Inches(1), Inches(5.3), Inches(11), Inches(2), size=15)

# ============================================================
# Slide 3: Neo4j 安装与配置
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "Neo4j 安装与配置", Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 左侧：什么是 Neo4j
add_card(slide, Inches(0.5), Inches(1.6), Inches(5.8), Inches(5.2))
add_text(slide, "什么是 Neo4j？", Inches(0.8), Inches(1.8), Inches(5), Inches(0.5),
         size=22, bold=True, color=ACCENT_GREEN)
items = [
    "图数据库 — 用节点和关系存储数据",
    "不像 Excel 表格，更像蜘蛛网",
    "天然适合存储知识图谱",
    "自带可视化浏览器（Browser）",
]
add_bullet_list(slide, items, Inches(0.8), Inches(2.5), Inches(5.2), Inches(2), size=16)

# 右侧：关键配置
add_card(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(5.2))
add_text(slide, "关键配置", Inches(7.1), Inches(1.8), Inches(5), Inches(0.5),
         size=22, bold=True, color=ACCENT_ORANGE)
config_items = [
    "安装：Neo4j Desktop（免费下载）",
    "HTTP 端口：7474（浏览器界面）",
    "Bolt 端口：7687（程序连接）",
    "用户名：neo4j / 密码：自定义",
    "启动后状态显示绿色 Running",
]
add_bullet_list(slide, config_items, Inches(7.1), Inches(2.5), Inches(5.2), Inches(3), size=16)

# ============================================================
# Slide 4: 大模型知识抽取方案设计
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "大模型知识抽取 — 方案设计", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 目标
add_card(slide, Inches(0.5), Inches(1.6), Inches(6), Inches(2.5))
add_text(slide, "抽取目标", Inches(0.8), Inches(1.8), Inches(5), Inches(0.4),
         size=20, bold=True, color=ACCENT_BLUE)
items = [
    "实体：天气事件、时间段、地点",
    "关系：干旱 -> 降雨、高温伴随干旱",
    "规律：统计性气象规律和模式",
]
add_bullet_list(slide, items, Inches(0.8), Inches(2.3), Inches(5.4), Inches(1.5), size=15)

# 对比模型
add_card(slide, Inches(7), Inches(1.6), Inches(5.8), Inches(2.5))
add_text(slide, "对比模型", Inches(7.3), Inches(1.8), Inches(5), Inches(0.4),
         size=20, bold=True, color=ACCENT_GREEN)
items = [
    "DeepSeek-V3 — 国产高性能模型",
    "通义千问-Max — 阿里云旗舰模型",
    "MiMo-v2.5-pro — 小米大模型",
]
add_bullet_list(slide, items, Inches(7.3), Inches(2.3), Inches(5.2), Inches(1.5), size=15)

# 技术方案
add_card(slide, Inches(0.5), Inches(4.5), Inches(12.3), Inches(2.3))
add_text(slide, "技术方案", Inches(0.8), Inches(4.7), Inches(5), Inches(0.4),
         size=20, bold=True, color=ACCENT_PURPLE)
items = [
    "统一 Prompt 模板：定义 JSON 输出格式（实体/关系/规律）",
    "OpenAI 兼容接口：三个模型都支持 OpenAI SDK 调用",
    "自动修复 JSON：处理大模型输出的格式不规范问题",
    "样本数据：取 CSV 前 50 行作为抽取输入",
]
add_bullet_list(slide, items, Inches(0.8), Inches(5.2), Inches(11.5), Inches(1.5), size=15)

# ============================================================
# Slide 5: 调试 ① MiMo 模型名称错误
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "调试 ① MiMo 模型名称错误", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_RED)

# 问题
add_card(slide, Inches(0.5), Inches(1.6), Inches(3.8), Inches(3))
add_text(slide, "问题", Inches(0.8), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_RED)
add_text(slide, "Unsupported model\nMiMo-V2.5-Pro\n(400 Bad Request)",
         Inches(0.8), Inches(2.4), Inches(3.2), Inches(1.5), size=15, color=LIGHT_GRAY)

# 排查
add_card(slide, Inches(4.7), Inches(1.6), Inches(3.8), Inches(3))
add_text(slide, "排查", Inches(5), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_ORANGE)
add_text(slide, "调用 models.list()\n查询 API 支持的\n模型列表",
         Inches(5), Inches(2.4), Inches(3.2), Inches(1.5), size=15, color=LIGHT_GRAY)

# 修复
add_card(slide, Inches(8.9), Inches(1.6), Inches(3.8), Inches(3))
add_text(slide, "修复", Inches(9.2), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_GREEN)
add_text(slide, "MiMo-V2.5-Pro\n        ->\nmimo-v2.5-pro\n(改为小写)",
         Inches(9.2), Inches(2.4), Inches(3.2), Inches(1.5), size=15, color=LIGHT_GRAY)

# 教训
add_card(slide, Inches(0.5), Inches(5), Inches(12.3), Inches(1.8))
add_text(slide, "教训", Inches(0.8), Inches(5.2), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_BLUE)
add_text(slide, "不同 API 的模型命名规范不同，应先调用 models.list() 确认可用模型名称，避免大小写敏感问题。",
         Inches(0.8), Inches(5.7), Inches(11.5), Inches(0.8), size=16, color=LIGHT_GRAY)

# ============================================================
# Slide 6: 调试 ② JSON 解析失败
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "调试 ② JSON 解析失败", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_RED)

# 问题
add_card(slide, Inches(0.5), Inches(1.6), Inches(5.8), Inches(2.2))
add_text(slide, "问题表现", Inches(0.8), Inches(1.8), Inches(5), Inches(0.4),
         size=20, bold=True, color=ACCENT_RED)
items = [
    'DeepSeek: "streak": 1-5 (非法 JSON 值)',
    "MiMo: 输出过长被截断 (超过 4000 token)",
    "三个模型都出现不同程度的解析失败",
]
add_bullet_list(slide, items, Inches(0.8), Inches(2.3), Inches(5.2), Inches(1.2), size=14)

# 修复方案
add_card(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(2.2))
add_text(slide, "修复方案", Inches(7.1), Inches(1.8), Inches(5), Inches(0.4),
         size=20, bold=True, color=ACCENT_GREEN)
items = [
    "max_tokens: 4000 -> 8000",
    "去除 markdown 代码块标记",
    "修复尾部逗号 (trailing comma)",
    "裸值 1-5 替换为字符串",
    "补全截断的括号",
]
add_bullet_list(slide, items, Inches(7.1), Inches(2.3), Inches(5.2), Inches(1.8), size=14)

# 多层修复流程
add_card(slide, Inches(0.5), Inches(4.2), Inches(12.3), Inches(2.8))
add_text(slide, "多层 JSON 修复流程", Inches(0.8), Inches(4.4), Inches(5), Inches(0.4),
         size=20, bold=True, color=ACCENT_BLUE)

steps = [
    ("1. 直接解析", "json.loads(text) 尝试"),
    ("2. 基础修复", "去代码块 / 去尾逗号 / 修裸值"),
    ("3. 重新解析", "json.loads(fixed) 尝试"),
    ("4. 截断修复", "移除末尾不完整字段，补全括号"),
    ("5. 放弃", "返回 None，跳过该模型"),
]
for i, (step, desc) in enumerate(steps):
    x = Inches(0.8 + i * 2.4)
    add_text(slide, step, x, Inches(5.0), Inches(2.2), Inches(0.4),
             size=15, bold=True, color=ACCENT_ORANGE)
    add_text(slide, desc, x, Inches(5.4), Inches(2.2), Inches(0.8),
             size=12, color=LIGHT_GRAY)

# ============================================================
# Slide 7: 大模型抽取结果对比
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "大模型抽取结果对比", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

headers = ["指标", "DeepSeek-V3", "通义千问-Max", "MiMo"]
data = [
    ["耗时", "14.8s", "21.3s", "105.6s"],
    ["输入 tokens", "1,354", "1,404", "1,402"],
    ["输出 tokens", "2,894", "1,007", "5,924"],
    ["实体数量", "14", "6", "5"],
    ["关系数量", "14", "9", "3"],
    ["规律数量", "8", "3", "3"],
]
add_table(slide, headers, data, Inches(1.5), Inches(1.8), Inches(10.3), Inches(0.5))

# 结论
add_card(slide, Inches(1.5), Inches(5.7), Inches(10.3), Inches(1.2))
add_text(slide, "结论：DeepSeek-V3 综合表现最佳 — 速度快、抽取全面、规律发现最多",
         Inches(1.8), Inches(5.9), Inches(9.7), Inches(0.8),
         size=18, bold=True, color=ACCENT_GREEN, align=PP_ALIGN.CENTER)

# ============================================================
# Slide 8: 大模型抽取的知识示例
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "大模型抽取的知识示例（DeepSeek）", Inches(0.8), Inches(0.4), Inches(12), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 实体
add_card(slide, Inches(0.5), Inches(1.6), Inches(3.8), Inches(5))
add_text(slide, "实体 Entity", Inches(0.8), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_BLUE)
items = [
    "观测站（地点）",
    "干旱事件1：6.8~6.17",
    "降雨事件1：6.18",
    "干旱事件2：6.19~6.23",
    "高温事件：6.22 35.3C",
    "大风事件",
]
add_bullet_list(slide, items, Inches(0.8), Inches(2.4), Inches(3.2), Inches(3.5), size=14)

# 关系
add_card(slide, Inches(4.7), Inches(1.6), Inches(3.8), Inches(5))
add_text(slide, "关系 Relationship", Inches(5), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_GREEN)
items = [
    "干旱1 -[之后]-> 降雨1",
    "干旱2 -[之后]-> 降雨2",
    "干旱1 -[类似]-> 干旱2",
    "高温 -[伴随]-> 干旱2",
    "大风 -[伴随]-> 干旱3",
]
add_bullet_list(slide, items, Inches(5), Inches(2.4), Inches(3.2), Inches(3.5), size=14)

# 规律
add_card(slide, Inches(8.9), Inches(1.6), Inches(3.8), Inches(5))
add_text(slide, "规律 Rule", Inches(9.2), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_ORANGE)
items = [
    "干旱10天后降雨9.1mm",
    "干旱5天后降雨1.6mm",
    "干旱日80%是大风日",
    "降雨日湿度80%",
    "降雨日温度偏低",
    "干旱期30~35C",
]
add_bullet_list(slide, items, Inches(9.2), Inches(2.4), Inches(3.2), Inches(3.5), size=14)

# ============================================================
# Slide 9: Neo4j 数据导入设计
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "Neo4j 数据导入 — 图模型设计", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 节点类型
add_card(slide, Inches(0.5), Inches(1.6), Inches(5.8), Inches(5.2))
add_text(slide, "节点类型", Inches(0.8), Inches(1.8), Inches(5), Inches(0.4),
         size=22, bold=True, color=ACCENT_BLUE)
node_data = [
    ["WeatherRecord", "每日天气记录", "CSV"],
    ["WeatherEvent", "天气事件", "CSV + 大模型"],
    ["Month", "月份聚合", "自动聚合"],
    ["Rule", "气象规律", "大模型"],
    ["TimePeriod", "时间段", "大模型"],
    ["Location", "地点", "大模型"],
]
add_table(slide, ["节点", "说明", "来源"], node_data,
          Inches(0.8), Inches(2.4), Inches(5.2), Inches(0.4))

# 关系类型
add_card(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(5.2))
add_text(slide, "关系类型", Inches(7.1), Inches(1.8), Inches(5), Inches(0.4),
         size=22, bold=True, color=ACCENT_GREEN)
rel_data = [
    ["HAS_EVENT", "记录包含事件", "2,053"],
    ["NEXT_DAY", "时间序列", "1,096"],
    ["IN_MONTH", "属于某月", "1,097"],
    ["FOLLOWED_BY", "事件先后", "大模型"],
    ["SIMILAR_TO", "相似模式", "大模型"],
]
add_table(slide, ["关系", "说明", "数量"], rel_data,
          Inches(7.1), Inches(2.4), Inches(5.2), Inches(0.4))

# ============================================================
# Slide 10: 调试 ③ Neo4j 连接被拒绝
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "调试 ③ Neo4j 连接被拒绝", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_RED)

# 问题
add_card(slide, Inches(0.5), Inches(1.6), Inches(5.8), Inches(2))
add_text(slide, "ConnectionRefusedError: [WinError 10061]", Inches(0.8), Inches(1.8),
         Inches(5.2), Inches(0.4), size=16, bold=True, color=ACCENT_RED)
add_text(slide, "由于目标计算机积极拒绝，无法连接",
         Inches(0.8), Inches(2.3), Inches(5.2), Inches(0.5), size=15, color=LIGHT_GRAY)

# 原因
add_card(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(2))
add_text(slide, "原因分析", Inches(7.1), Inches(1.8), Inches(5), Inches(0.4),
         size=20, bold=True, color=ACCENT_ORANGE)
add_text(slide, '.env 配置: bolt://localhost:7474\n\n7474 = HTTP 浏览器端口\n7687 = Bolt 程序连接端口',
         Inches(7.1), Inches(2.3), Inches(5.2), Inches(1.2), size=14, color=LIGHT_GRAY)

# 修复
add_card(slide, Inches(0.5), Inches(4), Inches(12.3), Inches(2.8))
add_text(slide, "修复", Inches(0.8), Inches(4.2), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_GREEN)

add_card(slide, Inches(0.8), Inches(4.8), Inches(5), Inches(1.5), RGBColor(0x1E, 0x1E, 0x1E))
add_text(slide, "NEO4J_URI=bolt://localhost:7474", Inches(1), Inches(4.9), Inches(4.5), Inches(0.4),
         size=14, color=ACCENT_RED)
add_text(slide, "NEO4J_URI=bolt://localhost:7687", Inches(1), Inches(5.4), Inches(4.5), Inches(0.4),
         size=14, color=ACCENT_GREEN)

add_text(slide, "Neo4j 有两个端口，用途不同：7474 给浏览器用，7687 给程序用。",
         Inches(6.5), Inches(5.0), Inches(6), Inches(0.8), size=16, color=LIGHT_GRAY)

# ============================================================
# Slide 11: 调试 ④ 唯一性约束冲突
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "调试 ④ 唯一性约束冲突", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_RED)

# 问题
add_card(slide, Inches(0.5), Inches(1.6), Inches(12.3), Inches(1.5))
add_text(slide, 'ConstraintError: Node already exists with label WeatherRecord and property date = "2023-06-22"',
         Inches(0.8), Inches(1.9), Inches(11.5), Inches(0.8), size=15, color=ACCENT_RED)

# 原因
add_card(slide, Inches(0.5), Inches(3.4), Inches(5.8), Inches(3.5))
add_text(slide, "原因", Inches(0.8), Inches(3.6), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_ORANGE)
items = [
    "CSV 已导入 date='2023-06-22' 的节点",
    "通义千问抽取的实体名也是 '2023-06-22'",
    "date 属性有唯一性约束",
    "MERGE 时两个节点冲突",
]
add_bullet_list(slide, items, Inches(0.8), Inches(4.2), Inches(5.2), Inches(2), size=15)

# 修复
add_card(slide, Inches(6.8), Inches(3.4), Inches(5.8), Inches(3.5))
add_text(slide, "修复方案", Inches(7.1), Inches(3.6), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_GREEN)
items = [
    "为实体导入添加 try/except",
    "遇到约束冲突自动跳过",
    "记录成功/失败数量",
    "天气数据完整保留",
    "大模型知识部分保留",
]
add_bullet_list(slide, items, Inches(7.1), Inches(4.2), Inches(5.2), Inches(2.5), size=15)

# ============================================================
# Slide 12: Neo4j 最终数据库统计
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "Neo4j 最终数据库统计", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 节点统计
headers = ["节点类型", "数量", "来源"]
node_stats = [
    ["WeatherRecord", "1,098", "CSV 数据"],
    ["WeatherEvent", "19", "CSV + 大模型"],
    ["Month", "37", "自动聚合"],
    ["Rule", "14", "大模型抽取"],
    ["TimePeriod", "8", "大模型抽取"],
    ["Location", "2", "大模型抽取"],
]
add_table(slide, headers, node_stats, Inches(0.8), Inches(1.6), Inches(5.5), Inches(0.45))

# 关系统计
headers = ["关系类型", "数量", "说明"]
rel_stats = [
    ["HAS_EVENT", "2,053", "记录关联天气事件"],
    ["IN_MONTH", "1,097", "记录属于某月"],
    ["NEXT_DAY", "1,096", "时间序列连接"],
    ["FOLLOWED_BY", "多条", "事件先后顺序"],
    ["SIMILAR_TO", "多条", "相似天气模式"],
]
add_table(slide, headers, rel_stats, Inches(7), Inches(1.6), Inches(5.5), Inches(0.45))

# 大模型导入统计
add_card(slide, Inches(0.8), Inches(5.2), Inches(11.5), Inches(1.5))
add_text(slide, "大模型知识导入", Inches(1.1), Inches(5.3), Inches(3), Inches(0.4),
         size=18, bold=True, color=ACCENT_PURPLE)
add_text(slide, "DeepSeek: 14 实体 + 14 关系 + 8 规律    |    通义千问: 6 实体 + 9 关系 + 3 规律    |    MiMo: 4 实体 + 3 关系 + 3 规律",
         Inches(1.1), Inches(5.8), Inches(11), Inches(0.6), size=14, color=LIGHT_GRAY)

# ============================================================
# Slide 13: Neo4j 可视化展示
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "Neo4j 可视化查询", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 查询示例
queries = [
    ("查看全貌", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 200"),
    ("单日详情", 'MATCH (r:WeatherRecord {date:"2024-08-24"})-[rel]->(t) RETURN r,rel,t'),
    ("高温日", "MATCH (r:WeatherRecord)-[:HAS_EVENT]->(e:WeatherEvent {name:'HeatDay'}) RETURN r, e"),
    ("大模型规律", "MATCH (rule:Rule) RETURN rule.model, rule.description"),
    ("多事件叠加", "MATCH (r)-[:HAS_EVENT]->(e) WITH r, collect(e.name) AS ev, count(e) AS c WHERE c >= 2 RETURN r.date, ev, c"),
]

for i, (title, query) in enumerate(queries):
    y = Inches(1.6 + i * 1.1)
    add_card(slide, Inches(0.5), y, Inches(12.3), Inches(0.95))
    add_text(slide, title, Inches(0.8), y + Inches(0.1), Inches(2.5), Inches(0.35),
             size=16, bold=True, color=ACCENT_GREEN)
    add_text(slide, query, Inches(3.3), y + Inches(0.1), Inches(9.2), Inches(0.75),
             size=12, color=ACCENT_BLUE, font="Consolas")

# ============================================================
# Slide 14: 大模型知识在图谱中的体现
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "大模型知识在图谱中的体现", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 三个对比卡片
cards = [
    ("CSV（原材料）", "只有每天的数字\n看不出规律和关系\n需要人工统计分析", ACCENT_ORANGE),
    ("大模型（气象专家）", "读完数字后归纳出：\n- 10天干旱打包为事件\n- 发现干旱后下雨\n- 总结伴随关系", ACCENT_BLUE),
    ("Neo4j（关系图）", "把分析变成一目了然的\n连线图，鼠标点击即可\n探索数据中的规律", ACCENT_GREEN),
]
for i, (title, desc, color) in enumerate(cards):
    left = Inches(0.5 + i * 4.2)
    add_card(slide, left, Inches(1.6), Inches(3.8), Inches(3))
    add_text(slide, title, left + Inches(0.3), Inches(1.8), Inches(3.2), Inches(0.4),
             size=20, bold=True, color=color)
    add_text(slide, desc, left + Inches(0.3), Inches(2.4), Inches(3.2), Inches(1.8),
             size=14, color=LIGHT_GRAY)

# 实际用途
add_card(slide, Inches(0.5), Inches(5), Inches(12.3), Inches(2))
add_text(slide, "实际用途", Inches(0.8), Inches(5.2), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_PURPLE)
items = [
    '"干旱后多久下雨？"  ->  直接查 FOLLOWED_BY 关系，无需写复杂 SQL',
    '"干旱和大风有关系吗？"  ->  规律节点直接告诉你：80%的干旱日伴随大风',
    "新人看图谱就能理解数据规律，无需阅读原始 CSV",
]
add_bullet_list(slide, items, Inches(0.8), Inches(5.7), Inches(11.5), Inches(1.2), size=15)

# ============================================================
# Slide 15: 总结与反思
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text(slide, "总结与反思", Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
         size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(0.8), Inches(1.1), Inches(2), ACCENT_BLUE)

# 技术栈
add_card(slide, Inches(0.5), Inches(1.6), Inches(3.8), Inches(2.5))
add_text(slide, "技术栈", Inches(0.8), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_BLUE)
items = ["Python + OpenAI SDK", "Neo4j 图数据库", "Protege 本体建模", "SPARQL 查询"]
add_bullet_list(slide, items, Inches(0.8), Inches(2.3), Inches(3.2), Inches(1.5), size=15)

# 踩坑总结
add_card(slide, Inches(4.7), Inches(1.6), Inches(3.8), Inches(2.5))
add_text(slide, "踩坑总结", Inches(5), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_RED)
items = [
    "模型名称大小写敏感",
    "大模型 JSON 输出不可靠",
    "Neo4j 双端口要分清",
    "唯一性约束冲突需处理",
]
add_bullet_list(slide, items, Inches(5), Inches(2.3), Inches(3.2), Inches(1.5), size=15)

# 结论
add_card(slide, Inches(8.9), Inches(1.6), Inches(3.8), Inches(2.5))
add_text(slide, "核心结论", Inches(9.2), Inches(1.8), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_GREEN)
items = [
    "DeepSeek 综合最优",
    "图数据库适合知识存储",
    "大模型让知识抽取自动化",
]
add_bullet_list(slide, items, Inches(9.2), Inches(2.3), Inches(3.2), Inches(1.5), size=15)

# 改进方向
add_card(slide, Inches(0.5), Inches(4.5), Inches(12.3), Inches(2.5))
add_text(slide, "可改进方向", Inches(0.8), Inches(4.7), Inches(3), Inches(0.4),
         size=20, bold=True, color=ACCENT_PURPLE)
items = [
    "增加更多气象指标（气压、能见度、日照时数等）",
    "接入实时天气 API，实现动态知识更新",
    "开发前端可视化页面（D3.js / ECharts + Neo4j）",
    "引入 RAG 技术，支持自然语言问答",
    "对比更多大模型（GPT-4、Claude、文心一言等）",
]
add_bullet_list(slide, items, Inches(0.8), Inches(5.2), Inches(11.5), Inches(1.5), size=15)

# ── 保存 ──
output_path = "weather_knowledge_graph.pptx"
prs.save(output_path)
print(f"PPT 已生成: {output_path}")
print(f"共 {len(prs.slides)} 页")
