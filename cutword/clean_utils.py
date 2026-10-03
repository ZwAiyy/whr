# -*- coding: utf-8 -*-
"""清洗规则库：两个来源的原始 CSV 共用同一套文本清洗与去重逻辑。

S0 字段统一 → S1 缺失剔除 → S2 HTML/实体 → S3 噪声符号（URL/@/#/邮箱）
→ S4 表情与方括号占位符 → S5 控制字符与空白归一 → S6 全角转半角
→ S7 繁体转简体 → S8 长度过滤 → S9 无效内容/广告过滤 → S10 精确去重
→ S11 近似去重（去标点/表情占位后一致）
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

import config as C

try:  # 繁体转简体（可选依赖）
    from opencc import OpenCC
    _cc = OpenCC("t2s")
except Exception:  # noqa: BLE001
    _cc = None

# ---------------------------------------------------------------- 正则
RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")
RE_ENTITY = re.compile(r"&[a-zA-Z]{2,8};|&#\d{2,6};")
RE_URL = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(com|cn|net|org)/\S*)")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_AT = re.compile(r"@[\w\u4e00-\u9fff\-_.]{1,30}")
RE_TOPIC = re.compile(r"#([^#\s]{1,30})#?")
RE_BRACKET_FACE = re.compile(r"\[[^\[\]]{1,12}\]")          # [大笑] / [比心] / [Lv-种草]
RE_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\u2190-\u21FF\u2B00-\u2BFF\uFE0F\u2600-\u26FF]")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
RE_WS = re.compile(r"\s+")
RE_NONKEEP = re.compile(r"[^\u4e00-\u9fff\u3040-\u30ffa-zA-Z0-9]")  # 非有效字符
RE_REPEAT = re.compile(r"(.)\1{6,}")                        # 6 个以上连续重复字符
RE_DIGITS_ONLY = re.compile(r"^[\d\W_]+$")
# 平台追加的元信息（点评末尾自动附带的“人均/推荐菜/图片数”等）
RE_META = re.compile(r"(人均\s*[¥￥]?\s*\d+\s*元?|推荐菜\s*[:：]\s*[^\s，,。]{1,20}"
                     r"|图片数\s*[:：]?\s*\d+|\d+\s*张图片|通过.{0,6}消费)")
# 起首或结尾的裸数字串 / 计数（“80”来自爬虫的图片计数列）
RE_BARE_NUM = re.compile(r"^[\d\s\W]{0,6}$")


def to_halfwidth(s: str) -> str:
    """全角 → 半角：仅处理全角数字/字母与全角空格，保留中文标点（。，！？等）。"""
    out = []
    for ch in s:
        o = ord(ch)
        if o == 0x3000:
            out.append(" ")
        elif 0xFF10 <= o <= 0xFF19 or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
            out.append(chr(o - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def clean_text(raw, stats: Counter | None = None) -> str:
    """单条文本清洗，返回清洗后文本（不含分词）。"""
    st = stats if stats is not None else Counter()
    t = "" if raw is None else str(raw)
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    before = t
    t = RE_HTML_TAG.sub(" ", t)
    t = RE_ENTITY.sub(" ", t)
    if t != before:
        st["S2_html_or_entity"] += 1

    before = t
    t = RE_URL.sub(" ", t)
    t = RE_EMAIL.sub(" ", t)
    t = RE_AT.sub(" ", t)
    t = RE_TOPIC.sub(r"\1", t)
    if t != before:
        st["S3_url_at_topic"] += 1

    before = t
    t = RE_BRACKET_FACE.sub(" ", t)
    if t != before:
        st["S4a_bracket_face"] += 1
    before = t
    t = RE_EMOJI.sub(" ", t)
    if t != before:
        st["S4b_emoji"] += 1

    before = t
    t = RE_META.sub(" ", t)
    if t != before:
        st["S5a_platform_meta"] += 1

    before = t
    t = RE_CTRL.sub("", t)
    t = unicodedata.normalize("NFC", t)
    t = to_halfwidth(t)
    t = RE_WS.sub(" ", t).strip()
    if t != before:
        st["S6_normalize"] += 1

    if _cc is not None and re.search(r"[\u4e00-\u9fff]", t):
        s = _cc.convert(t)
        if s != t:
            st["S7_t2s"] += 1
            t = s

    t = RE_REPEAT.sub(lambda m: m.group(1) * 3, t)   # 抑制刷屏式重复
    t = RE_WS.sub(" ", t).strip()
    t = t.strip(" \u3000.,;:!?~-—·、，。；：！？|/\\")
    return t.strip()


def is_invalid(t: str) -> str | None:
    """返回无效原因，None 表示有效。"""
    if len(t) < C.MIN_LEN:
        return "too_short"
    if RE_DIGITS_ONLY.match(t) or RE_BARE_NUM.match(t):
        return "digits_or_symbol_only"
    # 有效字符（汉字/假名/字母/数字）占比过低 → 基本是标点符号堆砌
    keep_ratio = len(RE_NONKEEP.sub("", t)) / max(len(t), 1)
    if keep_ratio < 0.15:
        return "punctuation_dominant"
    low = t.lower()
    if any(w.lower() in low for w in C.AD_WORDS):
        return "advertisement"
    return None


def dedup_key(t: str) -> str:
    """近似去重键：去掉所有标点/空白/表情占位后的小写字符串。"""
    return RE_NONKEEP.sub("", t).lower()


def clip_text(t: str, max_len: int | None = None) -> str:
    """超长文本按句末标点截断，避免破坏语义。"""
    max_len = max_len or C.MAX_LEN
    if len(t) <= max_len:
        return t
    cut = t[:max_len]
    for p in ("。", "！", "？", "，", "；", " "):
        i = cut.rfind(p)
        if i >= max_len * 0.6:
            return cut[:i].strip()
    return cut.strip()


def make_doc_id(*parts: str) -> str:
    """稳定 doc_id：内容 + 元信息的 md5 前 16 位。"""
    h = hashlib.md5("\u0001".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:16]
