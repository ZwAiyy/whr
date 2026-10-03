# -*- coding: utf-8 -*-
"""项目全局配置：路径、请求头、标签体系、采样参数。

项目：多来源中文情感分类数据获取 / 清洗 / 特征分析 / 标注与一致性评价
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DIR_RAW = ROOT / "data" / "raw"            # 原始数据（未做任何修改）
DIR_RAW_HTML = DIR_RAW / "html"            # 爬虫原始 HTML 快照（可复现证据）
DIR_RAW_JSON = DIR_RAW / "json"            # API 原始 JSON 快照
DIR_RAW_DATASET = DIR_RAW / "public_dataset"
DIR_INTERIM = ROOT / "data" / "interim"    # 中间结果（单来源解析后）
DIR_PROCESSED = ROOT / "data" / "processed"  # 处理后结果（清洗/合并）
DIR_ANNOTATION = ROOT / "annotation"
DIR_FIGURES = ROOT / "outputs" / "figures"
DIR_REPORTS = ROOT / "outputs" / "reports"
DIR_LOGS = ROOT / "logs"
DIR_LS_DATA = ROOT / "label_studio_data"

for _d in (DIR_RAW, DIR_RAW_HTML, DIR_RAW_JSON, DIR_RAW_DATASET, DIR_INTERIM,
           DIR_PROCESSED, DIR_ANNOTATION, DIR_FIGURES, DIR_REPORTS, DIR_LOGS, DIR_LS_DATA):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 爬虫参数
# ----------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}
CRAWL_DELAY = (1.2, 2.6)     # 每次请求之间的随机休眠（秒），礼貌爬取
CRAWL_TIMEOUT = 20           # 单次请求超时（秒）
CRAWL_RETRY = 3              # 失败重试次数

# 豆瓣读书：目标书目（subject_id, 书名）。运行时校验标题，失效自动跳过。
DOUBAN_BOOKS = [
    (1084336, "小王子"),
    (1007305, "红楼梦"),
    (2567698, "三体"),
    (4913064, "活着"),
    (6082808, "百年孤独"),
    (1082154, "围城"),
]
DOUBAN_PAGES_PER_BOOK = 12   # 每本书抓取的短评页数（每页 20 条）

# 网易云音乐：目标歌曲（song_id, 备注）。运行时校验 total，无评论自动跳过。
NETEASE_SONGS = [
    (186016, "晴天-周杰伦"),
    (347230, "海阔天空-Beyond"),
    (1330348068, "the truth that you leave-Pianoboy"),
    (287035, "遇见-孙燕姿"),
    (202369, "十年-陈奕迅"),
    (418603077, "起风了-买辣椒也用券"),
]
NETEASE_PAGES = 8            # 每首歌翻页数（每页 100 条）
NETEASE_PAGE_SIZE = 100

# 公开数据集
PUBLIC_DATASETS = {
    "ChnSentiCorp_htl_all": {
        "url": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
        "filename": "ChnSentiCorp_htl_all.csv",
        "domain": "酒店评论",
        "desc": "ChnSentiCorp 酒店中文情感语料（二分类）",
    },
    "waimai_10k": {
        "url": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
        "filename": "waimai_10k.csv",
        "domain": "外卖评论",
        "desc": "外卖平台中文情感语料（二分类）",
    },
    "online_shoppers": {
        "url": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        "filename": "online_shoppers_intention.csv",
        "domain": "电商行为",
        "desc": "UCI Online Shoppers Purchasing Intention（结构化数值型数据集，用于特征分析对照）",
    },
}

# ----------------------------------------------------------------------------
# 标签体系（统一到 4 类：正面 / 负面 / 中性 / 无法判断）
# ----------------------------------------------------------------------------
LABELS = ["正面", "负面", "中性", "无法判断"]
LABEL_POS, LABEL_NEG, LABEL_NEU, LABEL_UNC = LABELS
# 二分类映射（用于与公开数据集 label 对齐）
BINARY_MAP = {"正面": 1, "负面": 0, "中性": None, "无法判断": None}


def douban_rating_to_label(star40: int) -> str:
    """豆瓣星级（10/20/30/40/50）→ 情感标签（弱标注，用于抽样分层与对照）。"""
    if star40 >= 40:
        return LABEL_POS
    if star40 == 30:
        return LABEL_NEU
    if star40 in (10, 20):
        return LABEL_NEG
    return LABEL_UNC


# 情感/噪声词典（清洗与规则弱标注使用）
POS_WORDS = set("""好看 经典 喜欢 推荐 精彩 感动 温暖 治愈 深刻 优秀 完美 震撼 惊喜 细腻 真诚 有力
宝贵 启发 值得 难忘 温柔 动人 舒服 流畅 有趣 幽默 智慧 丰富 扎实 出色 惊艳 高级 顶级 上乘 佳作
神作 必读 五星 好书 好剧 好电影 好看极了 受益匪浅 回味无穷 爱了 治愈系 宝藏 强推 力荐""".split())
NEG_WORDS = set("""难看 无聊 失望 垃圾 差劲 烂 拖沓 枯燥 乏味 幼稚 狗血 尴尬 生硬 空洞 肤浅 矫情
做作 崩坏 毁三观 浪费时间 后悔 劝退 一般 平庸 老套 套路 敷衍 粗糙 混乱 晦涩 冗长 不推荐 什么玩意
名不副实 盛名之下 失望透顶 无语 煎熬 弃了 一星 差评""".split())

# 停用词（精简版，覆盖助词/代词/连词/标点语气词等；清洗与特征分析共用）
STOPWORDS = set("""
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 那 他 她 它
们 个 中 为 与 及 或 而 且 但 但是 因为 所以 如果 虽然 然后 还是 已经 可以 这个 那个 这样 那样 什么 怎么
为什么 怎样 一些 一样 一直 一定 这些 那些 我们 你们 他们 她们 它 之 于 以 把 被 让 给 对 从 向 往 由 关于
除了 之外 的话 吧 呢 啊 呀 哦 嗯 哈 呵 吗 啦 么 嘛 唉 哎 哟 哇 咦 嘿 额 呃 唔 ～ ~ ！ ？ 。 ， 、 ； ： " " ' ' （ ） 《 》 ！ ？ ……
很 太 更 最 挺 蛮 超 非常 十分 特别 尤其 有点 有些 稍微 比较 相当 极其 格外 略 略略 真 真是 好 好像 似乎
没有 没 别 不要 不用 不是 不能 不会 就是 只是 还有 而且 或者 不过 其实 反正 毕竟 居然 竟然 果然 于是 因此
今天 昨天 明天 现在 当时 以前 以后 后来 最后 开始 结束 时候 时候 一次 一点 一下 一样 一切 大家 别人
觉得 认为 感觉 知道 觉得 应该 可能 也许 大概 差不多 好像 真的 实在 确实 的确 总是 一直 从来 曾经
""".split())

# ----------------------------------------------------------------------------
# 标注与一致性评价参数
# ----------------------------------------------------------------------------
ANNOTATION_SAMPLE_SIZE = 300        # 送入 Label Studio 的标注样本量
ANNOTATION_MIN_LEN = 4              # 样本字符长度下限
ANNOTATION_MAX_LEN = 120            # 样本字符长度上限
RANDOM_SEED = 20240501
ANNOTATORS = ["annotator_01", "annotator_02", "annotator_03"]
KAPPA_INTERPRET = [                 # Landis & Koch (1977) 一致性强度
    (0.00, 0.20, "极弱 slight"),
    (0.20, 0.40, "一般 fair"),
    (0.40, 0.60, "中等 moderate"),
    (0.60, 0.80, "较强 substantial"),
    (0.80, 1.00, "很强 almost perfect"),
]

# Label Studio
LS_URL = os.environ.get("LS_URL", "http://127.0.0.1:8080")
LS_EMAIL = "annotator@example.com"
LS_PASSWORD = "LabelStudio@2024"
LS_PROJECT_TITLE = "中文短文本情感分类标注（多来源语料）"
