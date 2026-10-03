# -*- coding: utf-8 -*-
"""分词方法二：基于统计的分词（无监督词发现 + 一元词模型 + Viterbi 解码）。

思路（不使用任何人工词典，全部由语料统计得到）：
  1) 词发现（LLR / PMI）：统计 1~L 元字符片段频次与字频，对每个候选片段 w 计算
         pmi(w) = mean_i log[ P(c_i c_{i+1}) / (P(c_i)·P(c_{i+1})) ]
         score(w) = freq(w) · pmi(w) · |w|^λ
     pmi 衡量内部相邻字「结伴出现」的程度（越高越像一个词），freq 保证它是常用搭配，
     |w|^λ 是对长词的正则化，避免把「集团位于」这类跨词串也当成词。
     取 score ≥ 阈值且频次达标的片段作为词表（如「酒庄」「橡木桶」「讲解员」会被自动发现）。
  2) 概率估计：用词频做一元语言模型  P(w) = count(w) / count(·)。
  3) 解码：Viterbi 求 max_{切分} Σ_w [ log P(w) + β ]，β 为词数惩罚，控制切分粒度。
  λ 与 β 均由开发集自动校准（见 calibrate）。

另附 BEMS-HMM（Baum-Welch 无监督训练 + Viterbi）作为统计方法的对照实现。
"""
from __future__ import annotations

import math
import re
from collections import Counter

import config as C

BOS = "\u0002"
EOS = "\u0003"

# 汉字串 / 连续数字 / 连续英文 / 单个其它字符
RE_ANY = re.compile(r"[\u4e00-\u9fff]+|[0-9]+|[A-Za-z]+|\S")

# HMM 状态：B 词首 / M 词中 / E 词尾 / S 单字词
HMM_STATES = ["B", "M", "E", "S"]
LOG2 = math.log(2.0)
_NEG = -1e18
RE_CJK = None


def _has_cjk(w: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in w)


def _all_cjk(w: str) -> bool:
    """整词必须由汉字组成（过滤「酒，」「。 」这类把标点粘进来的片段）。"""
    return bool(w) and all("\u4e00" <= ch <= "\u9fff" for ch in w)


def _logsumexp(vals) -> float:
    m = max(vals)
    if m <= _NEG / 2:
        return _NEG
    return m + math.log(sum(math.exp(v - m) for v in vals))


class NGramSegmenter:
    """基于统计的无监督分词器：LLR/PMI 词发现 + 词频一元模型 + Viterbi 解码。"""

    name = "stat"
    display = "统计分词"

    # 超参网格（开发集自动校准）
    LAMBDA_GRID = [0.0, 1.0, 2.0, 3.0, 4.0]
    PENALTY_GRID = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

    def __init__(self, max_word_len: int | None = None, smooth: float | None = None,
                 word_penalty: float | None = None, length_power: float = 1.5):
        self.max_word_len = max_word_len or C.STAT_MAX_WORD_LEN
        self.smooth = smooth if smooth is not None else C.STAT_SMOOTH
        self.word_penalty = (C.STAT_WORD_PENALTY if word_penalty is None
                             else word_penalty)
        self.length_power = length_power
        self.word_count: Counter = Counter()   # 词发现结果：词 -> 频次
        self.pmi: dict[str, float] = {}        # 词 -> 平均 PMI
        self.total_words = 1.0
        self.types = 0
        self.candidates = 0
        self.char_total = 0
        self.unk_logp = _NEG
        self.calibration: dict = {}

    # ------------------------------------------------- 词发现（LLR / PMI）
    def _count_ngrams(self, texts):
        one, two, multi = Counter(), Counter(), Counter()
        L = self.max_word_len
        for t in texts:
            n = len(t)
            for i in range(n):
                one[t[i]] += 1
                if i + 2 <= n:
                    two[t[i:i + 2]] += 1
                for k in range(2, L + 1):
                    if i + k <= n:
                        multi[t[i:i + k]] += 1
        keep = {w: c for w, c in multi.most_common(C.STAT_TOP_NGRAM)
                if c >= C.STAT_MIN_COUNT}
        return one, two, keep

    def _pmi_avg(self, w: str, two: Counter, one: Counter, total: int) -> float:
        s = 0.0
        for i in range(len(w) - 1):
            p_ab = two.get(w[i:i + 2], 0) / total
            p_a = one.get(w[i], 0) / total
            p_b = one.get(w[i + 1], 0) / total
            if p_ab <= 0 or p_a <= 0 or p_b <= 0:
                return _NEG
            s += math.log(p_ab / (p_a * p_b))
        return s / (len(w) - 1)

    def fit(self, texts, length_power: float | None = None, log=print) -> "NGramSegmenter":
        """无监督词发现：按 score(w)=freq·PMI·|w|^λ 筛选候选词，并估计 P(w)。"""
        lp = self.length_power if length_power is None else length_power
        texts = [t for t in texts if t]
        one, two, multi = self._count_ngrams(texts)
        total = sum(one.values()) or 1
        self.char_total = total
        self.candidates = len(multi)
        self._one, self._two, self._total = one, two, total
        self._multi_cache = multi
        # 记录所有候选的 PMI（供词表重建）
        pmi = {}
        for w in multi:
            v = self._pmi_avg(w, two, one, total)
            if v > _NEG / 2:
                pmi[w] = v
        self.pmi = pmi
        self.rebuild_vocab(lp)
        log(f"      统计分词词发现：候选 {self.candidates} 个片段 → 词表 {self.types} 个"
            f"（λ={lp}，阈值 score≥{C.STAT_MIN_LLR}，频次≥{C.STAT_MIN_COUNT}）")
        return self

    def rebuild_vocab(self, length_power: float) -> int:
        """按当前 λ 重建词表。"""
        self.length_power = length_power
        words = Counter()
        for w, c in self._count_cache().items():
            p = self.pmi.get(w, _NEG)
            if p <= _NEG / 2:
                continue
            if not _all_cjk(w):
                continue
            score = c * p * (len(w) ** length_power)
            if score >= C.STAT_MIN_LLR and c >= C.STAT_MIN_COUNT:
                words[w] = c
        self.word_count = words
        self.total_words = float(sum(words.values()) or 1)
        self.types = len(words)
        self.unk_logp = math.log(C.STAT_SMOOTH / (self.total_words + C.STAT_SMOOTH))
        return self.types

    def _count_cache(self):
        return self._multi_cache

    # ------------------------------------------------------- 概率查询
    def word_log_prob(self, text: str, i: int, j: int) -> float:
        c = self.word_count.get(text[i:j])
        if not c:
            return self.unk_logp
        return math.log(c) - math.log(self.total_words)

    def _logprob(self, w: str) -> float:
        c = self.word_count.get(w)
        return math.log(c) - math.log(self.total_words) if c else self.unk_logp

    # ------------------------------------------------------------ 解码
    def cut(self, text: str) -> list[str]:
        return self.cut_with(text, self.word_penalty)

    def cut_with(self, text: str, penalty: float = 0.0) -> list[str]:
        """Viterbi：max Σ_w [ log P(w) + β ]（β 为词数惩罚，控制切分粒度）。

        非汉字部分按形态规则切分：连续数字 / 连续英文各自成 token，标点与空白作为硬边界。
        标点是否保留由上层评价协议统一处理（见 seg_metrics.norm_tokens），
        这里保留标点 token 以便与其它方法保持同一套「字符级完整还原」约束。
        """
        if not text:
            return []
        toks: list[str] = []
        for m in RE_ANY.finditer(text):
            s = m.group(0)
            if "\u4e00" <= s[0] <= "\u9fff":
                toks.extend(self._cut_cjk(s, penalty))
            elif re.match(r"^[0-9]+$", s) or re.match(r"^[A-Za-z]+$", s):
                toks.append(s)                      # 连续数字/英文整体成 token
            elif not s.isspace():
                toks.append(s)                      # 标点等
        return toks

    def _cut_cjk(self, text: str, penalty: float = 0.0) -> list[str]:
        n, mw = len(text), self.max_word_len
        wc, tw, unk = self.word_count, self.total_words, self.unk_logp
        best = [_NEG] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            bj, bj_i = best[j], back[j]
            for i in range(max(0, j - mw), j):
                bi = best[i]
                if bi <= _NEG / 2:
                    continue
                c = wc.get(text[i:j])
                v = (math.log(c) - math.log(tw)) if c else unk
                score = bi + v + penalty
                if score > bj:
                    bj, bj_i = score, i
            best[j], back[j] = bj, bj_i
        toks, j = [], n
        while j > 0:
            i = back[j]
            toks.append(text[i:j])
            j = i
        toks.reverse()
        return toks

    def cut_batch(self, texts) -> list[list[str]]:
        return [self.cut(t) for t in texts]

    # ------------------------------------------------------- λ / β 联合校准
    def calibrate(self, texts, refs: dict[str, list[list[str]]],
                  lambda_grid: list[float] | None = None,
                  penalty_grid: list[float] | None = None, log=print) -> dict:
        """开发集联合校准：先在 (λ, β) 网格上按边界 F1 选最优，再用
        「平均词长接近中文经验值」筛掉退化切分（全切单字 / 整句成一个词）。"""
        import seg_metrics as M
        lg = lambda_grid or self.LAMBDA_GRID
        pg = penalty_grid or self.PENALTY_GRID
        target, tol = C.STAT_CALIB_TARGET_LEN, C.STAT_CALIB_TOL
        rows = []
        for lam in lg:
            self.rebuild_vocab(lam)
            for b in pg:
                hyps = [self.cut_with(t, b) for t in texts]
                n_tok = sum(len(h) for h in hyps)
                avg_len = sum(len(w) for h in hyps for w in h) / max(n_tok, 1)
                f1s = [M.evaluate(texts, ref, hyps)["boundary"]["f1"]
                       for ref in refs.values()]
                rows.append({"lambda": lam, "penalty": b, "vocab": self.types,
                             "boundary_f1_vs_refs": round(sum(f1s) / len(f1s), 4),
                             "avg_token_len": round(avg_len, 3),
                             "in_reasonable_range": bool(abs(avg_len - target) <= tol)})
        ok = [d for d in rows if d["in_reasonable_range"]] or rows
        best = max(ok, key=lambda d: (d["boundary_f1_vs_refs"], d["avg_token_len"]))
        self.rebuild_vocab(best["lambda"])
        self.word_penalty = best["penalty"]
        self.calibration = {"best": best, "grid": rows, "target_avg_len": target}
        log(f"    统计分词校准：λ={best['lambda']}（词表 {best['vocab']}），"
            f"β={best['penalty']}，与参照边界 F1={best['boundary_f1_vs_refs']}，"
            f"平均词长={best['avg_token_len']}")
        return self.calibration

    # 便于报告中展示模型规模
    def stats(self) -> dict:
        return {"max_word_len": self.max_word_len, "smooth": self.smooth,
                "min_score": C.STAT_MIN_LLR, "min_count": C.STAT_MIN_COUNT,
                "length_power": self.length_power,
                "candidates": self.candidates,
                "vocab_types": self.types, "total_tokens": int(self.total_words),
                "chars": self.char_total, "word_penalty": self.word_penalty,
                "avg_pmi": round(sum(self.pmi.values()) / max(len(self.pmi), 1), 3),
                "top_words": [w for w, _ in self.word_count.most_common(15)],
                "calibration": self.calibration.get("best", {})}


class HMMSegmenter:
    """BEMS-HMM 统计分词器：Baum-Welch 无监督训练 + Viterbi 解码（对照实现）。"""

    name = "stat_hmm"
    display = "统计分词-HMM"

    def __init__(self, max_iter: int = 8):
        self.max_iter = max_iter
        self.c2i: dict[str, int] = {}
        self.pi = [math.log(0.5), _NEG, _NEG, math.log(0.5)]
        self.trans = [[_NEG] * 4 for _ in range(4)]   # [prev][cur]
        self.emit: dict[int, list[float]] = {}
        self.log_likelihood = 0.0

    @staticmethod
    def _allowed(prev: int, cur: int) -> bool:
        ps, cs = HMM_STATES[prev], HMM_STATES[cur]
        if ps in ("B", "M"):
            return cs in ("M", "E")
        return cs in ("B", "S")   # prev 为 S / E（或句首）

    # ------------------------------------------------ 前后向 + Baum-Welch
    def fit(self, texts) -> "HMMSegmenter":
        chars = sorted({c for t in texts for c in t})
        self.c2i = {c: i for i, c in enumerate(chars)}
        V = max(len(chars), 1)
        seqs = [[self.c2i[c] for c in t] for t in texts if t]
        trans = [[math.log(0.25) for _ in range(4)] for _ in range(4)]
        emit = {vi: [math.log(1.0 / V)] * 4 for vi in range(V)}

        for _ in range(self.max_iter):
            t_num = [[0.0] * 4 for _ in range(4)]
            t_den = [0.0] * 4
            e_num = [[0.0] * 4 for _ in range(V)]   # [字符][状态]
            e_den = [0.0] * 4
            ll = 0.0
            for seq in seqs:
                T = len(seq)
                alpha = [[_NEG] * T for _ in range(4)]
                for s in range(4):
                    alpha[s][0] = self.pi[s] + emit[seq[0]][s]
                for t in range(1, T):
                    for s in range(4):
                        vals = [alpha[p][t - 1] + trans[p][s] for p in range(4)
                                if self._allowed(p, s)]
                        alpha[s][t] = _logsumexp(vals) + emit[seq[t]][s]
                beta = [[_NEG] * T for _ in range(4)]
                for s in range(4):
                    beta[s][T - 1] = 0.0
                for t in range(T - 2, -1, -1):
                    for s in range(4):
                        vals = [trans[s][n] + emit[seq[t + 1]][n] + beta[n][t + 1]
                                for n in range(4) if self._allowed(s, n)]
                        beta[s][t] = _logsumexp(vals)
                logZ = _logsumexp([alpha[s][T - 1] for s in range(4)])
                ll += logZ
                for t in range(T):
                    gam = [_logsumexp([alpha[s][t] + beta[s][t]]) - logZ for s in range(4)]
                    for s in range(4):
                        g = math.exp(gam[s])
                        e_num[seq[t]][s] += g
                        e_den[s] += g
                    if t + 1 < T:
                        for s in range(4):
                            for n in range(4):
                                if not self._allowed(s, n):
                                    continue
                                xi = math.exp(alpha[s][t] + trans[s][n] + emit[seq[t + 1]][n]
                                              + beta[n][t + 1] - logZ)
                                t_num[s][n] += xi
                                t_den[s] += xi
            for s in range(4):
                for n in range(4):
                    p = t_num[s][n] / t_den[s] if t_den[s] > 0 else 0.25
                    trans[s][n] = math.log(max(p, 1e-12))
            for vi in range(V):
                for s in range(4):
                    p = e_num[vi][s] / e_den[s] if e_den[s] > 0 else 1.0 / V
                    emit[vi][s] = math.log(max(p, 1e-12))
            self.log_likelihood = ll
        self.trans, self.emit = trans, emit
        self.pi = [math.log(0.5), _NEG, _NEG, math.log(0.5)]
        return self

    def cut(self, text: str) -> list[str]:
        if not text:
            return []
        T = len(text)
        ids = [self.c2i.get(c, -1) for c in text]
        dp = [[_NEG] * 4 for _ in range(T)]
        bk = [[0] * 4 for _ in range(T)]
        for s in range(4):
            e = self.emit[ids[0]][s] if ids[0] >= 0 else math.log(1e-12)
            dp[0][s] = self.pi[s] + e
        for t in range(1, T):
            for s in range(4):
                e = self.emit[ids[t]][s] if ids[t] >= 0 else math.log(1e-12)
                best, arg = _NEG, 0
                for p in range(4):
                    if not self._allowed(p, s):
                        continue
                    v = dp[t - 1][p] + self.trans[p][s]
                    if v > best:
                        best, arg = v, p
                dp[t][s] = best + e
                bk[t][s] = arg
        cand = [(dp[T - 1][s], s) for s in range(4) if HMM_STATES[s] in ("E", "S")]
        _, last = max(cand) if cand else (0.0, 3)
        tags = [""] * T
        cur = last
        for t in range(T - 1, -1, -1):
            tags[t] = HMM_STATES[cur]
            cur = bk[t][cur]
        toks, buf = [], ""
        for ch, tag in zip(text, tags):
            buf += ch
            if tag in ("E", "S"):
                toks.append(buf)
                buf = ""
        if buf:
            toks.append(buf)
        return toks

    def cut_batch(self, texts) -> list[list[str]]:
        return [self.cut(t) for t in texts]

    def stats(self) -> dict:
        return {"max_iter": self.max_iter, "char_types": len(self.c2i),
                "log_likelihood": round(self.log_likelihood, 2)}
