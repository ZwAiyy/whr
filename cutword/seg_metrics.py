# -*- coding: utf-8 -*-
"""分词结果一致性/质量评价指标。

四个层次：
  1) 词边界（boundary）P/R/F1：切分点是否落在同一位置——最常用的分词评价方式；
  2) 词语（segment）精确匹配 P/R/F1；
  3) 句子级完全一致率 / 近似一致率（F1 ≥ 0.9 视为近似一致）；
  4) 压缩率 / 平均词长 / OOV 率 / 词表规模等描述统计。

评价协议（对所有方法一致）：只对「词」打分——先剔除纯标点/空白 token，
再比较切分位置；这样「是否把标点单独成词」不会影响 F1（大模型倾向输出标点，
规则/统计/深度模型则丢弃标点，若不做归一化会系统性低估大模型）。
"""
from __future__ import annotations

import re
from collections import Counter

RE_PUNCT_ONLY = re.compile(r"^[^\u4e00-\u9fff\u3040-\u30ffA-Za-z0-9]+$")


def norm_tokens(tokens) -> list[str]:
    """剔除纯标点与空白 token（评价协议统一化）。"""
    out = []
    for t in tokens:
        w = str(t).strip()
        if not w or RE_PUNCT_ONLY.match(w):
            continue
        out.append(w)
    return out


def boundaries(text: str, tokens: list[str]) -> set[int]:
    """返回词边界位置集合（字符下标，不含 0 与句末）。"""
    bs, pos = set(), 0
    for w in tokens:
        pos += len(w)
        if pos < len(text):
            bs.add(pos)
    return bs


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def boundary_prf(text: str, ref: list[str], hyp: list[str]) -> dict:
    rb, hb = boundaries(text, ref), boundaries(text, hyp)
    return prf(len(rb & hb), len(hb - rb), len(rb - hb))


def segment_prf(ref: list[str], hyp: list[str]) -> dict:
    cr, ch = Counter(ref), Counter(hyp)
    tp = sum(min(cr[w], ch[w]) for w in cr.keys() & ch.keys())
    return prf(tp, len(hyp) - tp, len(ref) - tp)


def evaluate(texts, refs, hyps) -> dict:
    """对一组文本计算综合指标（先按评价协议剔除纯标点 token）。refs/hyps 与 texts 等长。"""
    n = len(texts)
    refs = [norm_tokens(r) for r in refs]
    hyps = [norm_tokens(h) for h in hyps]
    btp = bfp = bfn = stp = sfp = sfn = 0
    exact = approx = 0
    for t, r, h in zip(texts, refs, hyps):
        rb, hb = boundaries(t, r), boundaries(t, h)
        btp += len(rb & hb)
        bfp += len(hb - rb)
        bfn += len(rb - hb)
        cr, ch = Counter(r), Counter(h)
        tp = sum(min(cr[w], ch[w]) for w in cr.keys() & ch.keys())
        stp += tp
        sfp += len(h) - tp
        sfn += len(r) - tp
        if r == h:
            exact += 1
        seg = prf(tp, len(h) - tp, len(r) - tp)
        if seg["f1"] >= 0.9:
            approx += 1
    out = {"n": n}
    out["boundary"] = prf(btp, bfp, bfn)
    out["segment"] = prf(stp, sfp, sfn)
    out["exact_match"] = round(exact / max(n, 1), 4)
    out["approx_match_f1_ge_0.9"] = round(approx / max(n, 1), 4)
    return out


def describe(texts, hyps, ref_vocab: set[str] | None = None) -> dict:
    """描述统计：词数、平均词长、压缩率、单字词占比、OOV 率、词表规模。"""
    hyps = [norm_tokens(h) for h in hyps]
    n_tok = sum(len(t) for t in hyps)
    n_char = sum(len(t) for t in texts)
    uniq = Counter(w for toks in hyps for w in toks)
    oov = sum(c for w, c in uniq.items() if ref_vocab is not None and w not in ref_vocab)
    total = sum(uniq.values())
    lens = [len(w) for toks in hyps for w in toks]
    return {
        "tokens": n_tok,
        "chars": n_char,
        "avg_tokens_per_text": round(n_tok / max(len(texts), 1), 2),
        "avg_token_len": round(sum(lens) / max(len(lens), 1), 3),
        "single_char_ratio": round(sum(1 for x in lens if x == 1) / max(len(lens), 1), 4),
        "compression_ratio": round(n_char / max(n_tok, 1), 3),
        "vocab_size": len(uniq),
        "oov_rate": round(oov / max(total, 1), 4) if ref_vocab is not None else None,
        "top_tokens": uniq.most_common(20),
    }


def pairwise(texts, hyps: dict[str, list[list[str]]], ref_name: str | None = None) -> dict:
    """两两方法之间的一致性（boundary / segment F1）。"""
    names = list(hyps)
    ref_name = ref_name or names[0]
    matrix = {}
    for a in names:
        for b in names:
            if a >= b:
                continue
            btp = bfp = bfn = stp = sfp = sfn = 0
            for t, ra, rb in zip(texts, hyps[a], hyps[b]):
                ra, rb = norm_tokens(ra), norm_tokens(rb)
                ba, bb = boundaries(t, ra), boundaries(t, rb)
                btp += len(ba & bb)
                bfp += len(bb - ba)
                bfn += len(ba - bb)
                ca, cb = Counter(ra), Counter(rb)
                tp = sum(min(ca[w], cb[w]) for w in ca.keys() & cb.keys())
                stp += tp
                sfp += len(rb) - tp
                sfn += len(ra) - tp
            matrix[f"{a} vs {b}"] = {"boundary_f1": prf(btp, bfp, bfn)["f1"],
                                     "segment_f1": prf(stp, sfp, sfn)["f1"]}
    # 以某方法为参照
    against = {}
    for a in names:
        if a == ref_name:
            continue
        against[a] = evaluate(texts, hyps[ref_name], hyps[a])
    return {"matrix": matrix, "against_reference": against, "reference": ref_name}


def consensus_ref(texts, hyps: dict[str, list[list[str]]], min_votes: int = 3):
    """多数投票构造“共识参考切分”：某边界被 ≥ min_votes 个方法同时切出才被采纳。"""
    refs = []
    for i, t in enumerate(texts):
        votes: Counter = Counter()
        for name, toks in hyps.items():
            for b in boundaries(t, norm_tokens(toks[i])):
                votes[b] += 1
        keep = sorted(b for b, v in votes.items() if v >= min_votes)
        refs.append(cut_by_boundaries(t, keep))
    return refs


def cut_by_boundaries(text: str, bs: list[int]) -> list[str]:
    toks, prev = [], 0
    for b in sorted(bs):
        if 0 < b < len(text):
            toks.append(text[prev:b])
            prev = b
    if prev < len(text):
        toks.append(text[prev:])
    return [w for w in toks if w]
