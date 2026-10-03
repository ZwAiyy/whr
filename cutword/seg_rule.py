# -*- coding: utf-8 -*-
"""分词方法一：基于规则的分词。

规则来源分三层，从强到弱：
  R1 词典规则（用户词典 + jieba 最大概率路径，HMM 关闭以保证完全确定性）
  R2 形态规则（URL/邮箱/@/# 话题、数字+单位、英文词、连续标点等模式切分）
  R3 兜底规则（词典与形态都覆盖不到的字符，按字切分）

实现上用 jieba 的精确模式加载用户词典，再用 regex 对结果做后处理合并/拆分，
保证「规则」部分的确定性（关闭 HMM 新词发现）。
"""
from __future__ import annotations

import re
from collections import Counter

import jieba

import config as C

# jieba 前缀词典缓存改放到项目内，避免写入系统临时目录（沙箱/权限更友好）
_CACHE_DIR = C.DIR_CACHE / "jieba"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    jieba.dt.tmp_dir = str(_CACHE_DIR)
    jieba.dt.cache_file = "jieba.cache"
except Exception:  # noqa: BLE001
    pass

# ------------------------------------------------------------------ 词法规则
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_AT = re.compile(r"@[\w\u4e00-\u9fff\-_.]{1,30}")
RE_TOPIC = re.compile(r"#([^#\s]{1,30})#?")
RE_NUM_UNIT = re.compile(r"\d+(?:\.\d+)?(?:元|块|米|公里|千米|小时|分钟|天|年|月|日|人|次|杯|瓶|种|个|张|折|分)?")
RE_EN = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*")
RE_KANA = re.compile(r"[\u3040-\u30ff]+")
RE_CJK = re.compile(r"[\u4e00-\u9fff]+")
RE_PUNCT = re.compile(r"[^\u4e00-\u9fff\u3040-\u30ffA-Za-z0-9]+")
# 需要整体保留的“词内标点”搭配
RE_KEEP_PAIR = re.compile(r"^(?:一|两|三|几)\S{0,3}(?:二|三|四|五|六|七|八|九|十)$")

TOKEN_RE = re.compile(
    "|".join([
        RE_URL.pattern, RE_EMAIL.pattern, RE_AT.pattern,
        r"[\u4e00-\u9fff]+", r"[\u3040-\u30ff]+",
        r"[A-Za-z][A-Za-z0-9'’\-]*", r"\d+(?:\.\d+)?", r"[^\s]",
    ]))


class RuleTokenizer:
    """基于规则的分词器（词典规则 + 形态规则 + 兜底规则）。"""

    name = "rule"
    display = "规则分词"

    def __init__(self, user_dict: list[str] | None = None, use_hmm: bool = False):
        self.use_hmm = use_hmm
        self.user_dict = list(user_dict if user_dict is not None else C.USER_DICT)
        for w in self.user_dict:
            jieba.add_word(w, freq=10000, tag="nz")
        # 保留一份词典用于切分后处理
        self.vocab = {w for w in self.user_dict if len(w) >= 2}
        self.max_word_len = max((len(w) for w in self.vocab), default=2)

    # ------------------------------------------------------------ 形态切分
    @staticmethod
    def morphological_cut(text: str) -> list[str]:
        """R2+R3：按字符类型切出「同类型字符块」，标点单独成块（随后丢弃）。"""
        return TOKEN_RE.findall(text)

    # ------------------------------------------------------- 词典后处理合并
    def _merge_by_dict(self, blocks: list[str]) -> list[str]:
        """把被切碎的多字词按用户词典重新合并（可处理跨块边界的情况）。"""
        out: list[str] = []
        i = 0
        n = len(blocks)
        while i < n:
            if RE_CJK.fullmatch(blocks[i]):
                merged = False
                for L in range(min(self.max_word_len, n - i), 1, -1):
                    cand = "".join(blocks[i:i + L])
                    if cand in self.vocab and len(cand) == len("".join(blocks[i:i + L])):
                        # 只允许在汉字块边界处合并
                        if all(RE_CJK.fullmatch(b) for b in blocks[i:i + L]):
                            out.append(cand)
                            i += L
                            merged = True
                            break
                if merged:
                    continue
            out.append(blocks[i])
            i += 1
        return out

    # ------------------------------------------------------------ 主入口
    def cut(self, text: str) -> list[str]:
        if not text:
            return []
        # R1：jieba 精确模式（关闭 HMM → 纯词典/统计词典规则，可复现）
        segs = jieba.lcut(text, HMM=self.use_hmm)
        toks: list[str] = []
        for s in segs:
            if not s.strip():
                continue
            # R2：对每个 jieba 片段再做形态切分，避免 "80元" 之类被粘连
            pieces = self.morphological_cut(s) or [s]
            toks.extend(self._merge_by_dict(pieces))
        # 丢弃纯标点块，去除空白
        return [w for w in toks if w.strip() and not RE_PUNCT.fullmatch(w)]

    def cut_batch(self, texts) -> list[list[str]]:
        return [self.cut(t) for t in texts]


def boundaries(tokens: list[str]) -> list[int]:
    """返回切分边界下标集合（不含末尾），用于一致性评价。"""
    bs, pos = [], 0
    for w in tokens[:-1]:
        pos += len(w)
        bs.append(pos)
    return bs


def count_tokens(token_lists) -> Counter:
    return Counter(w for toks in token_lists for w in toks)
