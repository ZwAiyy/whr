# -*- coding: utf-8 -*-
"""数据清洗流水线：把三个来源的原始数据统一清洗为可建模/可标注的语料。

清洗步骤（每一步都统计数量，写入清洗报告）：
  S0 载入与字段统一 → S1 缺失剔除 → S2 HTML/实体 → S3 噪声符号（URL/@/#/邮箱）
  → S4 表情与方括号占位符 → S5 控制字符与空白归一 → S6 全角转半角
  → S7 繁体转简体 → S8 长度过滤 → S9 无效内容/广告过滤 → S10 精确去重
  → S11 近似去重（去标点后一致）→ S12 分词与停用词 → S13 弱标签映射

输出：
  data/processed/sentiment_all_clean.csv / .jsonl   处理后结果（主交付）
  data/processed/clean_report.json                  清洗质量报告
  outputs/reports/清洗报告.md                        人读版报告
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import jieba
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

try:
    from opencc import OpenCC
    _cc = OpenCC("t2s")
except Exception:  # noqa: BLE001
    _cc = None

# ---------------------------------------------------------------- 正则/词典
RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")
RE_ENTITY = re.compile(r"&[a-zA-Z]{2,8};|&#\d{2,6};")
RE_URL = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(com|cn|net|org)/\S*)")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_AT = re.compile(r"@[\w\u4e00-\u9fff\-_.]{1,30}")
RE_TOPIC = re.compile(r"#([^#\s]{1,30})#?")
RE_BRACKET_FACE = re.compile(r"\[[^\[\]]{1,12}\]")          # 网易云 [大笑] / 微信表情
RE_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2190-\u21FF\u2B00-\u2BFF\uFE0F]")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
RE_WS = re.compile(r"\s+")
RE_FULLWIDTH = re.compile(r"[\uff01-\uff5e]")
RE_NONKEEP = re.compile(r"[^\u4e00-\u9fff\u3040-\u30ffa-zA-Z0-9]")  # 匹配"非有效字符"（标点/符号/空白）
RE_REPEAT = re.compile(r"(.)\1{6,}")                        # 6 个以上连续重复字符
RE_DIGITS_ONLY = re.compile(r"^[\d\W_]+$")

AD_WORDS = ["加微信", "微信号", "加群", "代购", "私聊", "点击链接", "扫码", "优惠券",
            "免费领取", "兼职", "刷单", "V信", "QQ群", "广告", "出售", "低价出"]
SENSITIVE_MIN_LEN, SENSITIVE_MAX_LEN = 2, 400


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


def clean_text(raw: str, stats: Counter) -> str:
    """单条文本清洗，返回清洗后文本（不含分词）。"""
    t = "" if raw is None else str(raw)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    before = t
    t = RE_HTML_TAG.sub(" ", t)
    t = RE_ENTITY.sub(" ", t)
    if t != before:
        stats["S2_html_or_entity"] += 1

    before = t
    t = RE_URL.sub(" ", t)
    t = RE_EMAIL.sub(" ", t)
    t = RE_AT.sub(" ", t)
    t = RE_TOPIC.sub(r"\1", t)
    if t != before:
        stats["S3_url_at_topic"] += 1

    before = t
    t = RE_BRACKET_FACE.sub(" ", t)
    if t != before:
        stats["S4a_bracket_face"] += 1
    before = t
    t = RE_EMOJI.sub(" ", t)
    if t != before:
        stats["S4b_emoji"] += 1

    before = t
    t = RE_CTRL.sub("", t)
    t = unicodedata.normalize("NFC", t)
    t = to_halfwidth(t)
    t = RE_WS.sub(" ", t).strip()
    if t != before:
        stats["S5_normalize"] += 1

    if _cc is not None and re.search(r"[\u4e00-\u9fff]", t):
        s = _cc.convert(t)
        if s != t:
            stats["S7_t2s"] += 1
            t = s

    t = RE_REPEAT.sub(lambda m: m.group(1) * 3, t)   # 抑制刷屏式重复
    t = t.strip(" \u3000.,;:!?~-—·、，。；：！？")
    return t.strip()


def is_invalid(t: str) -> str | None:
    """返回无效原因，None 表示有效。"""
    if len(t) < SENSITIVE_MIN_LEN:
        return "too_short"
    if len(t) > SENSITIVE_MAX_LEN:
        return "too_long"
    if RE_DIGITS_ONLY.match(t):
        return "digits_or_symbol_only"
    # 有效字符（汉字/假名/字母/数字）占比过低 → 基本是标点符号堆砌
    keep_ratio = len(RE_NONKEEP.sub("", t)) / max(len(t), 1)
    if keep_ratio < 0.15:
        return "punctuation_dominant"
    low = t.lower()
    if any(w.lower() in low for w in AD_WORDS):
        return "advertisement"
    return None


def dedup_key(t: str) -> str:
    """近似去重键：去掉所有标点/空白/大小写差异后的字符串。"""
    return RE_NONKEEP.sub("", t).lower()


def main() -> None:
    stats: Counter = Counter()
    dropped: list[dict] = []

    # ---------------- S0 载入三来源原始数据 ----------------
    src_files = {
        "source1_douban_bs4_raw.csv": ("豆瓣读书短评", "web_crawl_beautifulsoup4"),
        "source2_public_raw.csv": ("公开数据集", "public_dataset"),
        "source3_netease_api_raw.csv": ("网易云音乐评论", "open_api_json"),
    }
    frames = []
    for fn, (disp, stype) in src_files.items():
        p = C.DIR_INTERIM / fn
        if not p.exists():
            print(f"[warn] 缺少 {p}，跳过")
            continue
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        df["src_file"] = fn
        frames.append(df)
        print(f"[load] {fn}: {len(df)} 行")
    raw = pd.concat(frames, ignore_index=True)
    stats["S0_loaded"] = len(raw)

    # ---------------- S1 缺失剔除 ----------------
    raw["text_raw"] = raw["text_raw"].fillna("").astype(str)
    empty_mask = raw["text_raw"].str.strip().str.len() == 0
    stats["S1_missing_text"] = int(empty_mask.sum())
    df = raw[~empty_mask].copy()
    stats["S1_kept"] = len(df)

    # ---------------- S2~S9 逐条清洗 ----------------
    cleaned, reasons = [], []
    for t in df["text_raw"]:
        ct = clean_text(t, stats)
        r = is_invalid(ct)
        cleaned.append(ct)
        reasons.append(r or "")
    df["text_clean"] = cleaned
    df["invalid_reason"] = reasons
    stats["S8_too_short"] = reasons.count("too_short")
    stats["S8_too_long"] = reasons.count("too_long")
    stats["S9_digits_only"] = reasons.count("digits_or_symbol_only")
    stats["S9_punct_dominant"] = reasons.count("punctuation_dominant")
    stats["S9_advertisement"] = reasons.count("advertisement")
    bad = df["invalid_reason"] != ""
    dropped.extend(df.loc[bad, ["doc_id", "source", "text_raw", "text_clean", "invalid_reason"]]
                   .head(50).to_dict("records"))
    df = df[~bad].copy()
    stats["S9_kept"] = len(df)

    # ---------------- S10 精确去重 ----------------
    before = len(df)
    df = df.drop_duplicates(subset=["text_clean"], keep="first").copy()
    stats["S10_exact_dup_removed"] = before - len(df)

    # ---------------- S11 近似去重（同来源内） ----------------
    before = len(df)
    df["dedup_key"] = df["text_clean"].map(dedup_key)
    df = df.drop_duplicates(subset=["source", "dedup_key"], keep="first").copy()
    stats["S11_near_dup_removed"] = before - len(df)

    # ---------------- S12 分词与停用词 ----------------
    tok_list, tok_counts = [], []
    for t in df["text_clean"]:
        toks = [w for w in jieba.lcut(t)
                if w.strip() and w not in C.STOPWORDS and not RE_DIGITS_ONLY.match(w)]
        tok_list.append(toks)
        tok_counts.append(len(toks))
    df["tokens"] = [" ".join(x) for x in tok_list]
    df["token_count"] = tok_counts
    df["char_len"] = df["text_clean"].str.len()

    # ---------------- S13 弱标签 ----------------
    def weak_label(row) -> str:
        if row["source_type"] == "public_dataset":
            v = str(row.get("rating_title", "")).strip()
            if v in ("0", "0.0"):
                return C.LABEL_NEG
            if v in ("1", "1.0"):
                return C.LABEL_POS
            return ""
        if row["source_type"] == "web_crawl_beautifulsoup4":
            v = str(row.get("rating_star", "")).strip()
            if v and v != "nan":
                return C.douban_rating_to_label(int(float(v)))
            return ""
        return ""

    df["weak_label"] = df.apply(weak_label, axis=1)
    df["binary_label"] = df["weak_label"].map(C.BINARY_MAP)

    # ---------------- 输出 ----------------
    keep_cols = ["doc_id", "source", "source_type", "site", "domain", "doc_url", "author",
                 "rating_star", "rating_title", "comment_time", "votes",
                 "text_raw", "text_clean", "char_len", "tokens", "token_count",
                 "weak_label", "binary_label", "crawl_time"]
    out_df = df[keep_cols].reset_index(drop=True)
    out_csv = C.DIR_PROCESSED / "sentiment_all_clean.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_df.to_json(C.DIR_PROCESSED / "sentiment_all_clean.jsonl",
                   orient="records", lines=True, force_ascii=False)

    # 每来源单独输出
    slug_map = {"豆瓣读书短评": "douban_book", "网易云音乐评论": "netease_music",
                "ChnSentiCorp_htl_all": "chnsenticorp", "waimai_10k": "waimai_10k"}
    per_source = {}
    for s, g in out_df.groupby("source"):
        slug = slug_map.get(s) or f"source_{len(per_source) + 1}"
        fn = C.DIR_PROCESSED / f"clean_{slug}.csv"
        g.to_csv(fn, index=False, encoding="utf-8-sig")
        per_source[s] = {"rows": int(len(g)), "file": str(fn),
                         "avg_char_len": round(float(g["char_len"].mean()), 2),
                         "avg_token_count": round(float(g["token_count"].mean()), 2),
                         "weak_label_dist": {str(k): int(v) for k, v in
                                             g["weak_label"].value_counts().items()}}

    top_words = Counter(w for toks in tok_list for w in toks).most_common(50)
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": dict(stats),
        "final_rows": int(len(out_df)),
        "retention_rate": round(len(out_df) / max(stats["S0_loaded"], 1), 4),
        "per_source": per_source,
        "source_distribution": {str(k): int(v) for k, v in out_df["source"].value_counts().items()},
        "weak_label_distribution": {str(k): int(v) for k, v in out_df["weak_label"].value_counts().items()},
        "char_len_stats": {k: round(float(v), 2) for k, v in
                           out_df["char_len"].describe().items()},
        "token_count_stats": {k: round(float(v), 2) for k, v in
                              out_df["token_count"].describe().items()},
        "top50_tokens": top_words,
        "vocab_size": int(len(set(w for toks in tok_list for w in toks))),
        "dropped_examples": dropped[:30],
        "opencc_available": _cc is not None,
    }
    (C.DIR_PROCESSED / "clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 清洗结果 =====")
    for k in sorted(stats):
        print(f"  {k:28s} {stats[k]}")
    print(f"  最终语料：{len(out_df)} 条（保留率 {report['retention_rate']:.1%}）")
    print(f"  输出：{out_csv}")


if __name__ == "__main__":
    main()
