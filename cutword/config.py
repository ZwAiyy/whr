# -*- coding: utf-8 -*-
"""项目全局配置：路径、标签体系、停用词、四种分词方法的采样与超参数。

项目：多来源中文情感分类数据获取 / 清洗 / 分词（四方法对比）
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[0]

DIR_INPUT = ROOT / "data" / "interim"        # 字段统一后的中间结果（清洗输入）
DIR_PROCESSED = ROOT / "data" / "processed"  # 清洗 / 分词结果（主交付）
DIR_REPORTS = ROOT / "outputs" / "reports"   # 人读版报告
DIR_MODELS = ROOT / "outputs" / "models"     # 深度学习分词模型参数
DIR_CACHE = ROOT / "outputs" / "cache"       # 大模型分词缓存（可断点续跑）
DIR_LOGS = ROOT / "logs"

for _d in (DIR_INPUT, DIR_PROCESSED, DIR_REPORTS, DIR_MODELS, DIR_CACHE, DIR_LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# 两个原始 CSV（允许放在项目根目录或 data/raw 下）
RAW_CSV_CANDIDATES = {
    "ctrip": ["source4_ctrip_raw.csv", "data/raw/source4_ctrip_raw.csv",
              "data/interim/source4_ctrip_raw.csv"],
    "dianping": ["adb_dazhongdianping_raw.csv", "data/raw/adb_dazhongdianping_raw.csv",
                 "data/interim/adb_dazhongdianping_raw.csv"],
}
UNIFIED_FILES = {
    "ctrip": DIR_INPUT / "ctrip_unified.csv",
    "dianping": DIR_INPUT / "dianping_unified.csv",
}

# ----------------------------------------------------------------------------
# 标签体系（统一到 4 类：正面 / 负面 / 中性 / 无法判断）
# ----------------------------------------------------------------------------
LABELS = ["正面", "负面", "中性", "无法判断"]
LABEL_POS, LABEL_NEG, LABEL_NEU, LABEL_UNC = LABELS
BINARY_MAP = {"正面": 1, "负面": 0, "中性": None, "无法判断": None}


def star_to_label(star: int) -> str:
    """携程星级（10~50，10 = 1 星）→ 情感弱标签。"""
    if star >= 40:
        return LABEL_POS
    if star == 30:
        return LABEL_NEU
    if star in (10, 20):
        return LABEL_NEG
    return LABEL_UNC


# 大众点评「评价档」→ 情感弱标签（该站点无数字星级）
DIANPING_LEVEL_MAP = {
    "超预期": LABEL_POS,
    "很棒": LABEL_POS,
    "满意": LABEL_POS,
    "不错": LABEL_POS,
    "一般": LABEL_NEU,
    "差": LABEL_NEG,
    "不满意": LABEL_NEG,
    "": LABEL_UNC,
}
# 「评价档」→ 虚拟星级（10~50），便于与携程口径统一
DIANPING_LEVEL_STAR = {"超预期": 50, "很棒": 50, "满意": 40, "不错": 40,
                       "一般": 30, "差": 20, "不满意": 10, "": 0}
DIANPING_LEVEL_MAP_STAR = DIANPING_LEVEL_STAR

# ----------------------------------------------------------------------------
# 清洗参数
# ----------------------------------------------------------------------------
MIN_LEN = 4            # 清洗后最短字符数（低于视为无效）
MAX_LEN = 500          # 清洗后最长字符数（长文截断而非丢弃）
AD_WORDS = ["加微信", "微信号", "加群", "代购", "私聊", "点击链接", "扫码", "优惠券",
            "免费领取", "兼职", "刷单", "V信", "QQ群", "出售", "低价出"]

# 情感词典（弱标注 / 规则分词的情感词兜底，供扩展使用）
POS_WORDS = set("""好看 经典 喜欢 推荐 精彩 感动 温暖 治愈 深刻 优秀 完美 震撼 惊喜 细腻 真诚 有力
宝贵 启发 值得 难忘 温柔 动人 舒服 流畅 有趣 幽默 智慧 丰富 扎实 出色 惊艳 高级 顶级 上乘 佳作
神作 必读 五星 好书 好剧 好电影 受益匪浅 回味无穷 宝藏 强推 力荐 超预期 满意 赞 棒 出片 打卡""".split())
NEG_WORDS = set("""难看 无聊 失望 垃圾 差劲 烂 拖沓 枯燥 乏味 幼稚 狗血 尴尬 生硬 空洞 肤浅 矫情
做作 崩坏 毁三观 浪费时间 后悔 劝退 一般 平庸 老套 套路 敷衍 粗糙 混乱 晦涩 冗长 不推荐 什么玩意
名不副实 失望透顶 无语 煎熬 一星 差评 宰客 坑人 不值""".split())

# ----------------------------------------------------------------------------
# 停用词（精简版，分词后过滤）
# ----------------------------------------------------------------------------
STOPWORDS = set("""
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 那 他 她 它
们 个 中 为 与 及 或 而 且 但 但是 因为 所以 如果 虽然 然后 还是 已经 可以 这个 那个 这样 那样 什么 怎么
为什么 怎样 一些 一样 一直 一定 这些 那些 我们 你们 他们 她们 之 于 以 把 被 让 给 对 从 向 往 由 关于
除了 之外 的话 吧 呢 啊 呀 哦 嗯 哈 呵 吗 啦 么 嘛 唉 哎 哟 哇 咦 嘿 额 呃 唔
很 太 更 最 挺 蛮 超 非常 十分 特别 尤其 有点 有些 稍微 比较 相当 极其 格外 略 真 真是 好像 似乎
没 别 不要 不用 不是 不能 不会 就是 只是 还有 而且 或者 不过 其实 反正 毕竟 居然 竟然 果然 于是 因此
今天 昨天 明天 现在 当时 以前 以后 后来 最后 开始 结束 时候 一次 一点 一下 一切 大家 别人
觉得 认为 感觉 知道 应该 可能 也许 大概 差不多 真的 实在 确实 的确 总是 一直 从来 曾经
""".split())

# ----------------------------------------------------------------------------
# 分词方法一：规则分词（jieba + 用户词典 + 正则兜底）
# ----------------------------------------------------------------------------
# 领域词表：把景点/酒庄领域的专名与常用搭配固化，避免被切碎
USER_DICT = [
    # 龙谕酒庄 / 贺兰山东麓 领域专名
    "龙谕酒庄", "龙谕", "贺兰山东麓", "贺兰山", "酒庄", "酒窖", "橡木桶", "葡萄酒",
    "酿酒葡萄", "品鉴", "品酒", "干红", "白葡萄酒", "赤霞珠", "霞多丽", "葡萄酒酿造",
    "恒温酒窖", "地下酒窖", "欧式古堡", "塞上江南", "银川", "宁夏", "西夏", "镇北堡",
    "门票", "讲解员", "讲解", "预约", "打卡", "出片", "氛围感", "性价比", "服务态度",
    "超出预期", "超预期", "值得推荐", "不虚此行", "值得一去", "回头客", "网红",
    "大众点评", "携程", "评价", "点评", "商家回复", "人均", "推荐菜", "口味", "环境", "服务",
    # 高频功能词搭配
    "整体感觉", "总体感觉", "工作人员", "服务员", "有意思", "没什么", "怎么样", "第一次",
]
RULE_DICT_MAX_WORD = max(len(w) for w in USER_DICT)

# ----------------------------------------------------------------------------
# 分词方法二：统计分词（LLR/PMI 无监督词发现 + 词频一元模型 + Viterbi）
# ----------------------------------------------------------------------------
STAT_NGRAM_ORDER = 3        # 报告用：字级 n-gram 阶数
STAT_MAX_WORD_LEN = 4       # 候选词最大长度
STAT_SMOOTH = 0.1           # 概率平滑系数
STAT_MIN_LLR = 4.0          # 词候选的 LLR（对数似然比 / PMI 累积）阈值
STAT_MIN_COUNT = 3          # 词候选最少出现次数
STAT_WORD_PENALTY = 0.0     # 词数惩罚 β 初值（运行时用开发集自动校准）
STAT_CALIB_MIN_DEV = 30     # 校准用开发集最小条数
STAT_CALIB_TARGET_LEN = 2.1  # 校准时期望的平均词长（中文分词经验值，用于筛掉退化切分）
STAT_CALIB_TOL = 0.45       # 平均词长容差
STAT_TOP_NGRAM = 4000       # 参与 LLR 统计的 n-gram 上限（控制内存/时间）
STAT_MAX_DEV = 80           # 校准开发集最大条数

# ----------------------------------------------------------------------------
# 分词方法三：深度学习分词（BiLSTM 序列标注）
#   backend: PyTorch（有 GPU 时自动用 CUDA）；无 torch 时自动回退纯 NumPy 实现
# ----------------------------------------------------------------------------
DL_FORCE_NUMPY = os.environ.get("DL_FORCE_NUMPY", "0").lower() in ("1", "true", "yes")
DL_EMBED_DIM = 128
DL_HIDDEN_DIM = 128
DL_EPOCHS = 20
DL_BATCH_SIZE = 64
DL_LR = 2e-3
DL_MAX_SEQ_LEN = 128
DL_SEED = 20240501
DL_TRAIN_RATIO = 0.9
DL_MODEL_FILE = DIR_MODELS / "bilstm_seg.npz"       # NumPy 后端模型文件
DL_MODEL_FILE_TORCH = DIR_MODELS / "bilstm_seg.pt"  # PyTorch 后端模型文件
DL_MAX_TRAIN_CHARS = 200000
DL_TAG_PENALTY = 0.0        # 解码时对 S（单字词）标签的惩罚，运行时自动校准
DL_TAG_PENALTY_GRID = [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

# ----------------------------------------------------------------------------
# 分词方法四：大模型分词（DeepSeek Chat API）
# ----------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.0
LLM_BATCH_SIZE = 8          # 单次请求内的文本条数
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT = 90
LLM_RETRY = 3
LLM_SLEEP = 0.4             # 请求间隔（秒），避免触发限流
LLM_PROMPT_VERSION = "v1"
LLM_SEG_DELIMITER = "/"     # 让模型用 "/" 连接词，便于解析

# ----------------------------------------------------------------------------
# 四方法统一采样（对比实验）
#   SAMPLE_N 可用环境变量覆盖：$env:SAMPLE_N="200"; python clean_pipeline.py
# ----------------------------------------------------------------------------
SAMPLE_N = int(os.environ.get("SAMPLE_N", 120))   # 每种方法参与对比的文本条数
SAMPLE_SEED = 20240501
SAMPLE_MIN_LEN = 8          # 采样时要求的最小字符长度（保证有足够切分点）
SAMPLE_MAX_LEN = 150        # 采样时要求的最大字符长度（控制大模型请求开销）
SAMPLE_PER_SOURCE = True    # 按来源分层采样（两来源尽量等量）

# 大模型评判（以 LLM 给出的“理想切分”作为准金标准，对四方法打分）
JUDGE_ENABLE = os.environ.get("JUDGE_ENABLE", "1").lower() not in ("0", "false", "no")
JUDGE_SAMPLE_N = int(os.environ.get("JUDGE_SAMPLE_N", 20))
JUDGE_MAX_LEN = 80          # 供评判的文本长度上限（保证单次请求可控）
JUDGE_BATCH_SIZE = 4        # 每次评判请求包含的文本条数
JUDGE_PROMPT_VERSION = "v3"

# 一致性评价（Landis & Koch 1977）
KAPPA_INTERPRET = [
    (0.00, 0.20, "极弱 slight"),
    (0.20, 0.40, "一般 fair"),
    (0.40, 0.60, "中等 moderate"),
    (0.60, 0.80, "较强 substantial"),
    (0.80, 1.00, "很强 almost perfect"),
]

# ----------------------------------------------------------------------------
# 凭据（大模型分词用）
# ----------------------------------------------------------------------------
_CRED_FILE = Path.home() / ".dsh" / ".credentials.yaml"


def _load_credentials() -> dict:
    """从 DSH 凭据文件读取 API Key，环境变量优先级更高。"""
    refs: dict[str, str] = {}
    if _CRED_FILE.exists():
        try:
            txt = _CRED_FILE.read_text(encoding="utf-8")
            refs = dict(re.findall(r"^\s{2}([A-Z_0-9]+):\s*(\S+)\s*$", txt, re.M))
        except Exception:  # noqa: BLE001
            refs = {}
    for name in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "XIAOMI_API_KEY"):
        if os.environ.get(name):
            refs[name] = os.environ[name]
    return refs


CREDENTIALS = _load_credentials()
LLM_API_KEY = CREDENTIALS.get("LLM_API_KEY") or CREDENTIALS.get("DEEPSEEK_API_KEY", "")
