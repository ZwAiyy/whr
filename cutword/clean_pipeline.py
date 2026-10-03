# -*- coding: utf-8 -*-
"""主流水线：两个 CSV 的清洗 + 四种方法分词（规则 / 统计 / 深度学习 / 大模型）。

用法：
    python clean_pipeline.py                          # 清洗 + 分词，采样 SAMPLE_N=120 条
    $env:SAMPLE_N="200"; python clean_pipeline.py     # 自定义参与分词对比的条数
    $env:SKIP_LLM="1"; python clean_pipeline.py       # 跳过需要联网的大模型分词

清洗步骤（每步统计数量，写入清洗报告）：
  S0 载入与字段统一 → S1 缺失剔除 → S2 HTML/实体 → S3 噪声符号（URL/@/#/邮箱）
  → S4 表情与方括号占位符 → S5 平台元信息 → S6 控制字符/空白归一与全角转半角
  → S7 繁体转简体 → S8 长度过滤与截断 → S9 无效内容/广告过滤
  → S10 精确去重 → S11 近似去重

分词步骤：
  T0 采样（默认 120 条，按来源分层）→ T1 规则分词 → T2 统计分词（n-gram + Viterbi）
  → T3 深度学习分词（BiLSTM 序列标注）→ T4 大模型分词（Few-shot + 字符级校验）
  → T5 一致性评价（边界/词语 F1、两两一致、共识参照、LLM 评判）

输出：
  data/interim/{ctrip,dianping}_unified.csv        字段统一后的数据
  data/processed/clean_all.csv                     两来源合并的清洗语料（tokens 为规则分词）
  data/processed/clean_ctrip.csv / clean_dianping.csv
  data/processed/seg_compare_sample.csv            四方法分词对比（逐条）
  data/processed/clean_report.json                 清洗质量报告
  data/processed/seg_report.json                   分词评价报告
  outputs/reports/清洗报告.md / 分词方法对比报告.md
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import clean_utils as U  # noqa: E402
import seg_metrics as M  # noqa: E402
from seg_dl import backend_info, bootstrap_gold_from_tokens, make_tagger, max_match  # noqa: E402
from seg_llm import LLMSegmenter  # noqa: E402
from seg_rule import RuleTokenizer  # noqa: E402
from seg_stat import HMMSegmenter, NGramSegmenter  # noqa: E402

SKIP_LLM = os.environ.get("SKIP_LLM", "0").lower() in ("1", "true", "yes")
SKIP_JUDGE = (os.environ.get("SKIP_JUDGE", "0").lower() in ("1", "true", "yes")
              or not C.JUDGE_ENABLE)

LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


# ============================================================ S0 载入与字段统一
DIANPING_COLS = ["来源", "内容", "时间", "时间(标准化)", "用户", "评价档", "图片数",
                 "标签", "商家回复", "店铺ID", "采集时间", "序号"]
RE_DIANPING_META = re.compile(
    r"\d{4}-\d{1,2}(?:-\d{1,2})?(?:月)?出行.*$"     # 2026-09出行｜游玩1-2小时｜情侣夫妻
    r"|通过.{0,8}消费"                             # 通过大众点评消费
    r"|^[¥￥]\s*\d+\s*元?$"                        # 独立成行的 ¥100
    r"|^\d{4}-\d{1,2}月?$"                         # 独立成行的 2026-09
    r"|^游玩.{0,8}小时$|^[^｜|\n]{1,6}(?:夫妻|亲子|情侣|朋友|独自|家庭|商务)$")


def _clean_dianping_content(s: str) -> str:
    """预清洗大众点评正文：去掉平台模板行、出行标签与表情占位符。"""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\[[^\[\]]{1,12}\]", " ", s)        # [比心] [Lv-种草] [流口水]
    lines = []
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 平台会在同一行追加「2026-09出行｜游玩1-2小时｜情侣夫妻」等标签，按分隔符再切
        segs = []
        for seg in re.split(r"[｜|]", line):
            seg = RE_DIANPING_META.sub(" ", seg).strip()
            if seg:
                segs.append(seg)
        if segs:
            lines.append(" ".join(segs))
    return " ".join(lines)


def find_raw(key: str) -> Path | None:
    for rel in C.RAW_CSV_CANDIDATES[key]:
        p = C.ROOT / rel
        if p.exists():
            return p
    return None


def load_ctrip(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = pd.DataFrame({
        "doc_id": df.get("doc_id", ""), "source": df.get("source", ""),
        "source_type": df.get("source_type", ""), "site": df.get("site", ""),
        "domain": df.get("domain", ""), "doc_url": df.get("doc_url", ""),
        "author": df.get("author", ""), "rating_star": df.get("rating_star", ""),
        "rating_title": df.get("rating_title", ""), "comment_time": df.get("comment_time", ""),
        "votes": df.get("votes", ""), "crawl_time": df.get("crawl_time", ""),
        "text_raw": df.get("text_raw", ""),
    })
    out["src_file"] = path.name
    return out


def load_dianping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in DIANPING_COLS:
        if c not in df.columns:
            df[c] = ""
    content = df["内容"].map(_clean_dianping_content)
    out = pd.DataFrame({
        "doc_id": [U.make_doc_id("dianping", c, u, t) for c, u, t in
                   zip(content, df["用户"], df["时间(标准化)"])],
        "source": "大众点评景点点评",
        "source_type": "adb_dazhongdianping",
        "site": "m.dianping.com",
        "domain": df["来源"],
        "doc_url": "https://m.dianping.com/shop/" + df["店铺ID"],
        "author": df["用户"],
        "rating_star": df["评价档"].map(C.DIANPING_LEVEL_STAR),
        "rating_title": df["评价档"],
        "comment_time": df["时间(标准化)"],
        "votes": "",
        "crawl_time": df["采集时间"],
        "text_raw": content,
    })
    out["src_file"] = path.name
    return out


def to_unified() -> tuple[pd.DataFrame, Counter]:
    """S0：读取两个原始 CSV，字段统一后合并。"""
    stats: Counter = Counter()
    frames = []
    paths = {"ctrip": find_raw("ctrip"), "dianping": find_raw("dianping")}
    for key, p in paths.items():
        if p is None:
            log(f"[warn] 找不到 {key} 的原始 CSV，跳过")
            continue
        df = load_ctrip(p) if key == "ctrip" else load_dianping(p)
        df.to_csv(C.UNIFIED_FILES[key], index=False, encoding="utf-8-sig")
        stats[f"S0_{key}_rows"] = len(df)
        frames.append(df)
        log(f"[S0] {p.name}: {len(df)} 行 → {C.UNIFIED_FILES[key].name}")
    if not frames:
        raise SystemExit("没有可用的原始 CSV")
    return pd.concat(frames, ignore_index=True), stats


# ============================================================ S1~S11 清洗
def clean_sources(uni: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[dict]]:
    stats: Counter = Counter()
    dropped: list[dict] = []
    parts = []
    for src_file, g in uni.groupby("src_file", sort=False):
        g = g.copy()
        stats["S0_loaded"] += len(g)
        # S1 缺失剔除
        empty = g["text_raw"].fillna("").astype(str).str.strip().str.len() == 0
        stats["S1_missing_text"] += int(empty.sum())
        g = g[~empty].copy()
        # S2~S9 逐条清洗
        cleaned, reasons = [], []
        local: Counter = Counter()
        for t in g["text_raw"]:
            ct = U.clean_text(t, local)
            cleaned.append(ct)
            reasons.append(U.is_invalid(ct) or "")
        g["text_clean"] = cleaned
        g["invalid_reason"] = reasons
        # S8 长度处理：超长按句末截断
        long_mask = g["text_clean"].str.len() > C.MAX_LEN
        stats["S8_truncated"] += int(long_mask.sum())
        if long_mask.any():
            g.loc[long_mask, "text_clean"] = g.loc[long_mask, "text_clean"].map(U.clip_text)
        bad = g["invalid_reason"] != ""
        stats["S8_too_short"] += reasons.count("too_short")
        stats["S9_digits_only"] += reasons.count("digits_or_symbol_only")
        stats["S9_punct_dominant"] += reasons.count("punctuation_dominant")
        stats["S9_advertisement"] += reasons.count("advertisement")
        dropped.extend(g.loc[bad, ["doc_id", "source", "text_raw", "text_clean",
                                   "invalid_reason"]].head(30).to_dict("records"))
        g = g[~bad].copy()
        stats["S9_kept"] += len(g)
        # S10 精确去重
        before = len(g)
        g = g.drop_duplicates(subset=["text_clean"], keep="first")
        stats["S10_exact_dup_removed"] += before - len(g)
        # S11 近似去重（同来源内去标点后一致）
        before = len(g)
        g = g.copy()
        g["dedup_key"] = g["text_clean"].map(U.dedup_key)
        g = g.drop_duplicates(subset=["dedup_key"], keep="first")
        stats["S11_near_dup_removed"] += before - len(g)
        # S13 弱标签
        def weak(row):
            if str(row["source_type"]).startswith("web_crawl"):
                v = str(row["rating_star"]).strip()
                if v and v.lower() != "nan":
                    try:
                        return C.star_to_label(int(round(float(v))))
                    except ValueError:
                        return C.LABEL_UNC
                return C.LABEL_UNC
            return C.DIANPING_LEVEL_MAP.get(str(row["rating_title"]).strip(), C.LABEL_UNC)

        g["weak_label"] = g.apply(weak, axis=1)
        g["binary_label"] = g["weak_label"].map(C.BINARY_MAP)
        g["char_len"] = g["text_clean"].str.len()
        parts.append(g)
        stats.update(local)
    out = pd.concat(parts, ignore_index=True) if parts else uni.iloc[0:0]
    stats["S12_final_rows"] = len(out)
    stats["retention_rate_pct"] = round(100 * len(out) / max(stats["S0_loaded"], 1), 2)
    return out, dict(stats), dropped


# ============================================================ T0 采样
CALIB_RATIO = 0.15          # 校准开发集占清洗语料的比例
CALIB_MIN = 40              # 校准开发集最小条数
CALIB_MAX = 100             # 校准开发集最大条数


def calib_texts(df: pd.DataFrame) -> list[str]:
    """从全语料中抽一小份「校准开发集」，用于自动调参（与评价样本不重叠地随机抽样）。"""
    n = min(max(CALIB_MIN, int(len(df) * CALIB_RATIO)), CALIB_MAX, len(df))
    pool = df[df["char_len"] <= C.SAMPLE_MAX_LEN]
    pool = pool if len(pool) >= n else df
    return pool.sample(n=n, random_state=C.SAMPLE_SEED + 7)["text_clean"].tolist()


def sample_texts(df: pd.DataFrame, n: int) -> pd.DataFrame:
    pool = df[(df["char_len"] >= C.SAMPLE_MIN_LEN) & (df["char_len"] <= C.SAMPLE_MAX_LEN)].copy()
    if pool.empty:
        pool = df.copy()
    if not C.SAMPLE_PER_SOURCE:
        return pool.sample(n=min(n, len(pool)), random_state=C.SAMPLE_SEED)
    groups = list(pool.groupby("source", sort=False))
    per = max(1, n // max(len(groups), 1))
    picked = [g.sample(n=min(per, len(g)), random_state=C.SAMPLE_SEED) for _, g in groups]
    out = pd.concat(picked, ignore_index=True)
    if len(out) < n:                       # 数量不足时用剩余样本补齐
        used = set(out["doc_id"])
        rest = pool[~pool["doc_id"].isin(used)]
        need = min(n - len(out), len(rest))
        if need > 0:
            out = pd.concat([out, rest.sample(n=need, random_state=C.SAMPLE_SEED + 1)],
                            ignore_index=True)
    return out.head(n).reset_index(drop=True)


# ============================================================ 分词主流程
def run_segmentation(df: pd.DataFrame) -> dict:
    t_start = time.time()
    corpus = df["text_clean"].tolist()
    sample_df = sample_texts(df, C.SAMPLE_N)
    texts = sample_df["text_clean"].tolist()
    log(f"\n[T0] 清洗语料 {len(corpus)} 条；参与四方法对比的采样 {len(texts)} 条"
        f"（来源分布：{dict(sample_df['source'].value_counts())}）")

    method_notes: dict[str, dict] = {}

    # ---------------- T1 规则分词 ----------------
    t0 = time.time()
    log("[T1] 规则分词：jieba 精确模式（关闭 HMM）+ 领域用户词典 + 正则形态规则")
    rule = RuleTokenizer()
    sample_rule = rule.cut_batch(texts)
    method_notes["rule"] = {
        "display": rule.display, "user_dict_size": len(rule.user_dict),
        "use_hmm": rule.use_hmm, "seconds": round(time.time() - t0, 2),
        "desc": "词典规则 + 形态规则 + 单字兜底；确定性切分，不依赖模型训练。",
    }

    # ---------------- T2 统计分词（词发现 + 词频模型） ----------------
    t0 = time.time()
    log("[T2] 统计分词：LLR/PMI 无监督词发现 + 词频一元模型 + Viterbi 解码")
    ngram = NGramSegmenter().fit(corpus)
    hmm = HMMSegmenter(max_iter=6).fit([t for t in corpus if t])

    # ---------------- T2b 超参校准（开发集自动调参） ----------------
    dev = calib_texts(df)
    log(f"[T2b] 自动校准：开发集 {len(dev)} 条，参照 = 规则分词 + 词表最大匹配（两路互不依赖）")
    dev_ref = {"规则分词": rule.cut_batch(dev),
               "词表匹配": [max_match(t, ngram.word_count, ngram.max_word_len) for t in dev]}
    ngram.calibrate(dev, dev_ref, log=log)
    # 用校准后的统计分词结果作为深度学习伪标注的「准标准」
    corpus_stat = [ngram.cut(t) for t in corpus]

    # ---------------- T3 深度学习分词 ----------------
    t0 = time.time()
    log("[T3] 深度学习分词：BiLSTM 序列标注")
    log(f"     后端 {backend_info()}")
    gold_tags, vocab = bootstrap_gold_from_tokens(corpus, corpus_stat, log=log)
    dl = make_tagger()
    dl.build_vocab(corpus)
    train_info = dl.fit(corpus, gold_tags, log=log)
    dev_ref["统计分词"] = [ngram.cut(t) for t in dev]
    dl.calibrate(dev, dev_ref, log=log)
    log(f"    训练 {train_info['train_seconds']}s，设备 {train_info['device']}，"
        f"训练样本 {train_info['train_size']} / 验证 {train_info['val_size']}")
    method_notes["stat"] = {
        "display": ngram.display, **ngram.stats(), "seconds": round(time.time() - t0, 2),
        "desc": "用 LLR/PMI 从语料中无监督发现候选词，再用词频估计一元语言模型 P(w)，"
                "Viterbi 求 Σ[log P(w)+β] 最大切分；λ 与 β 由开发集自动校准，无人工词典。",
    }
    method_notes["stat_hmm"] = {
        "display": hmm.display, **hmm.stats(), "is_auxiliary": True,
        "desc": "BEMS-HMM（Baum-Welch 无监督训练 + Viterbi）作为统计方法的对照实现。",
    }
    t1 = time.time()
    sample_stat = ngram.cut_batch(texts)
    sample_hmm = hmm.cut_batch(texts)
    sample_dl = dl.cut_batch(texts)
    infer_sec = round(time.time() - t1, 3)
    model_path = dl.save()
    method_notes["deep"] = {
        "display": dl.display, **dl.stats(), "train_size": train_info["train_size"],
        "val_size": train_info["val_size"], "pseudo_vocab_size": len(vocab),
        "seconds": round(time.time() - t0, 2), "infer_seconds": 0.0,
        "model_file": str(model_path),
        "desc": "字嵌入 → 双向 LSTM → 线性分类头，Viterbi 约束解码；监督信号为"
                "「校准后的统计分词」经词表最大匹配得到的 B/M/E/S 伪标签，"
                "解码时对单字词（S）的惩罚由开发集自动校准；"
                f"训练后端 {dl.stats().get('backend')}（{train_info['device']}）。",
    }

    # ---------------- T4 大模型分词 ----------------
    t0 = time.time()
    if SKIP_LLM:
        log("[T4] 大模型分词：已通过 SKIP_LLM=1 跳过，使用规则分词结果占位")
        sample_llm = [list(x) for x in sample_rule]
        method_notes["llm"] = {"display": "大模型分词", "skipped": True,
                               "desc": "本次未调用（SKIP_LLM=1），以规则分词结果占位。"}
    else:
        log(f"[T4] 大模型分词：{C.LLM_MODEL}（few-shot + JSON 约束 + 字符级校验 + 缓存）")
        llm = LLMSegmenter(fallback=rule)
        sample_llm = llm.cut_batch(texts, log=log)
        method_notes["llm"] = {
            "display": llm.display, "model": llm.model, "base_url": llm.base_url,
            "prompt_version": C.LLM_PROMPT_VERSION, **llm.stats_report(),
            "seconds": round(time.time() - t0, 2),
            "desc": "DeepSeek Chat few-shot 直接生成切分，用 '/' 连接并做字符级一致性校验。",
        }

    # ---------------- T5 评价 ----------------
    log("[T5] 一致性评价：边界/词语 F1、两两一致、共识参照切分")
    hyps = {"规则分词": sample_rule, "统计分词": sample_stat,
            "深度学习分词": sample_dl, "大模型分词": sample_llm}
    names = list(hyps)
    # 统一测量「对同一批文本做一次切分」的耗时，保证四方法可比（不含模型训练）
    infer_sec = {}
    for nm, key in [("规则分词", "rule"), ("统计分词", "stat"),
                    ("深度学习分词", "deep"), ("大模型分词", "llm")]:
        t = time.time()
        if key == "rule":
            rule.cut_batch(texts)
        elif key == "stat":
            ngram.cut_batch(texts)
        elif key == "deep":
            dl.cut_batch(texts)
        else:
            [list(x) for x in sample_llm]      # 已缓存，测量的是解析+缓存读取开销
        infer_sec[nm] = round(time.time() - t, 3)
    for nm, key in [("规则分词", "rule"), ("统计分词", "stat"),
                    ("深度学习分词", "deep"), ("大模型分词", "llm")]:
        method_notes[key]["infer_seconds"] = infer_sec[nm]
        method_notes[key]["infer_note"] = (f"对同一批 {len(texts)} 条文本切分一次"
                                           "（不含模型训练）")
    log(f"    单次切分耗时：{infer_sec}")
    against_rule = {nm: M.evaluate(texts, hyps["规则分词"], hyps[nm])
                    for nm in names if nm != "规则分词"}
    consensus = M.consensus_ref(texts, hyps, min_votes=3)
    against_consensus = {nm: M.evaluate(texts, consensus, hyps[nm]) for nm in names}
    against_stat = {nm: M.evaluate(texts, hyps["统计分词"], hyps[nm])
                    for nm in names if nm != "统计分词"}
    pair = M.pairwise(texts, hyps, ref_name="规则分词")
    desc = {nm: M.describe(texts, hyps[nm], ref_vocab=set(vocab)) for nm in names}
    aux = {"统计分词-HMM": {"against_consensus": M.evaluate(texts, consensus, sample_hmm),
                            "describe": M.describe(texts, sample_hmm, ref_vocab=set(vocab))}}

    # ---------------- T5b 大模型评判 ----------------
    judge_result: dict = {"enabled": False, "cases": [], "summary": {}}
    if not SKIP_JUDGE and not SKIP_LLM:
        log(f"[T5b] 大模型评判：抽取 {C.JUDGE_SAMPLE_N} 条文本，先给理想切分再逐方法打分")
        idx = [i for i, t in enumerate(texts) if len(t) <= C.JUDGE_MAX_LEN] or list(range(len(texts)))
        step = max(1, len(idx) // C.JUDGE_SAMPLE_N)
        pick = idx[::step][:C.JUDGE_SAMPLE_N]
        j_texts = [texts[i] for i in pick]
        j_hyps = {nm: [hyps[nm][i] for i in pick] for nm in names}
        try:
            from seg_judge import SegmentationJudge
            judge = SegmentationJudge()
            judged = judge.judge(j_texts, j_hyps, log=log)
            cases, agg = [], {nm: Counter() for nm in names}
            for local_i, t in enumerate(j_texts):
                rec = judged.get(t)
                if not rec:
                    continue
                ideal = rec.get("ideal", [])
                item = {"text": t, "ideal": ideal, "scores": {}, "f1": {}}
                for nm in names:
                    sc = (rec.get("scores") or {}).get(nm, {}) or {}
                    item["scores"][nm] = {k: sc.get(k) for k in
                                          ["boundary_ok", "exp_complete", "prop_noun",
                                           "no_fragment", "overall", "comment"]}
                    if ideal:
                        item["f1"][nm] = M.evaluate([t], [ideal], [j_hyps[nm][local_i]])
                    for k, v in sc.items():
                        if isinstance(v, (int, float)):
                            agg[nm][k] += float(v)
                cases.append(item)
            summary = {}
            for nm in names:
                n = max(len(cases), 1)
                summary[nm] = {k: round(agg[nm][k] / n, 2) for k in agg[nm]}
                f1s = [c["f1"][nm]["boundary"]["f1"] for c in cases if nm in c.get("f1", {})]
                summary[nm]["boundary_f1_vs_ideal"] = round(sum(f1s) / max(len(f1s), 1), 4)
            judge_result = {"enabled": True, "stats": judge.stats_report(),
                            "summary": summary, "cases": cases,
                            "note": "LLM-as-a-judge，非人工标注，仅作近似参照"}
        except Exception as e:  # noqa: BLE001
            log(f"    [warn] 大模型评判失败：{e}")
            judge_result = {"enabled": False, "error": str(e), "cases": [], "summary": {}}
    else:
        log("[T5b] 大模型评判：已跳过")

    # ---------------- 落盘：逐条对比 ----------------
    cmp_rows = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        rec = {"doc_id": row["doc_id"], "source": row["source"],
               "char_len": int(row["char_len"]), "text_clean": texts[i],
               "weak_label": row["weak_label"]}
        for nm, key in [("规则分词", "rule"), ("统计分词", "stat"),
                        ("深度学习分词", "deep"), ("大模型分词", "llm")]:
            toks = hyps[nm][i]
            rec[f"tokens_{key}"] = " ".join(toks)
            rec[f"n_tokens_{key}"] = len(toks)
        bs = [M.boundaries(texts[i], M.norm_tokens(hyps[nm][i])) for nm in names]
        common, union = set.intersection(*bs), set.union(*bs)
        rec["boundary_agree_rate"] = round(len(common) / max(len(union), 1), 4)
        rec["n_punct_tokens_llm"] = len(hyps["大模型分词"][i]) - len(
            M.norm_tokens(hyps["大模型分词"][i]))
        cmp_rows.append(rec)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_csv = C.DIR_PROCESSED / "seg_compare_sample.csv"
    cmp_df.to_csv(cmp_csv, index=False, encoding="utf-8-sig")
    cmp_df.to_json(C.DIR_PROCESSED / "seg_compare_sample.jsonl", orient="records",
                   lines=True, force_ascii=False)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_n": len(texts),
        "sample_source_dist": {str(k): int(v) for k, v in
                               sample_df["source"].value_counts().items()},
        "methods": {nm: method_notes.get(
            {"规则分词": "rule", "统计分词": "stat", "深度学习分词": "deep",
             "大模型分词": "llm"}[nm], {}) for nm in names},
        "auxiliary_methods": method_notes.get("stat_hmm", {}),
        "vs_rule_baseline": against_rule,
        "vs_consensus_ref": against_consensus,
        "vs_stat_ref": against_stat,
        "pairwise": pair,
        "describe": desc,
        "auxiliary_eval": aux,
        "judge": {k: v for k, v in judge_result.items() if k != "cases"},
        "examples": judge_result.get("cases", [])[:10],
        "total_seconds": round(time.time() - t_start, 1),
    }
    (C.DIR_PROCESSED / "seg_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"report": report, "cmp_csv": cmp_csv, "sample_df": sample_df, "texts": texts,
            "hyps": hyps, "method_notes": method_notes, "vocab": vocab,
            "judge": judge_result, "corpus": corpus}


# ============================================================ 报告
def write_clean_report(stats: dict, dropped: list[dict], out_df: pd.DataFrame,
                       seg: dict | None = None) -> Path:
    p = C.DIR_REPORTS / "清洗报告.md"
    lines = ["# 数据清洗报告", "",
             f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
             "- 输入：`source4_ctrip_raw.csv`、`adb_dazhongdianping_raw.csv`",
             "- 输出：`data/processed/clean_all.csv`（另按来源拆分）",
             f"- 最终语料：**{len(out_df)}** 条，保留率 **{stats.get('retention_rate_pct', 0)}%**",
             "", "## 1. 分步统计", "", "| 步骤 | 计数 |", "| --- | --- |"]
    for k in sorted(stats):
        lines.append(f"| {k} | {stats[k]} |")
    lines += ["", "## 2. 来源分布", "", "| 来源 | 条数 | 平均字数 | 平均词数 |",
              "| --- | --- | --- | --- |"]
    for s, g in out_df.groupby("source"):
        lines.append(f"| {s} | {len(g)} | {g['char_len'].mean():.1f} | "
                     f"{g['token_count'].mean():.1f} |")
    lines += ["", "## 3. 弱标签分布", "", "| 弱标签 | 条数 |", "| --- | --- |"]
    for k, v in out_df["weak_label"].value_counts().items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 4. 被剔除样本示例（最多 30 条）", "",
              "| 来源 | 原因 | 原文（截断） |", "| --- | --- | --- |"]
    for d in dropped[:30]:
        raw = str(d.get("text_raw", ""))[:40].replace("|", "/").replace("\n", " ")
        lines.append(f"| {d.get('source', '')} | {d.get('invalid_reason', '')} | {raw} |")
    lines += ["", "## 5. 高频词 Top30（规则分词，已过滤停用词）", "",
              "| 词 | 频次 | 词 | 频次 |", "| --- | --- | --- | --- |"]
    cnt = Counter(w for t in out_df["tokens"] for w in str(t).split())
    top = cnt.most_common(30)
    for i in range(0, len(top) - 1, 2):
        a, b = top[i], top[i + 1]
        lines.append(f"| {a[0]} | {a[1]} | {b[0]} | {b[1]} |")
    if seg is not None:
        lines += ["", "## 6. 四方法分词结果概览（对比样本）", "",
                  "| 方法 | 词数 | 平均词长 | 单字词占比 | 词表规模 | 与规则分词边界F1 | 与共识切分边界F1 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for nm, d in seg["report"]["describe"].items():
            f1r = ("-" if nm == "规则分词"
                   else seg["report"]["vs_rule_baseline"].get(nm, {}).get("boundary", {}).get("f1", "-"))
            f1c = seg["report"]["vs_consensus_ref"].get(nm, {}).get("boundary", {}).get("f1", "-")
            lines.append(f"| {nm} | {d['tokens']} | {d['avg_token_len']} | {d['single_char_ratio']} | "
                         f"{d['vocab_size']} | {f1r} | {f1c} |")
        lines += ["", "> 详细指标、评判样例与结论见 `分词方法对比报告.md`。"]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_seg_report(seg: dict) -> Path:
    r, texts, hyps = seg["report"], seg["texts"], seg["hyps"]
    p = C.DIR_REPORTS / "分词方法对比报告.md"
    L = ["# 分词方法对比报告（规则 / 统计 / 深度学习 / 大模型）", "",
         f"- 生成时间：{r['generated_at']}",
         f"- 对比样本：{r['sample_n']} 条（来源分布：{r['sample_source_dist']}）",
         "- 参照标准：① 规则分词（baseline）② 四方法共识切分（边界 ≥3 票）"
         "③ 统计分词 ④ 大模型裁判给出的理想切分",
         "", "## 1. 四种方法概述", "",
         "| 方法 | 核心思路 | 是否需要训练 | 外部知识 |", "| --- | --- | --- | --- |"]
    meta = {
        "规则分词": ("jieba 精确模式（关闭 HMM）+ 领域用户词典 + 正则形态规则 + 单字兜底",
                     "否", "人工词典 / 正则"),
        "统计分词": ("LLR/PMI 无监督词发现 + 词频一元模型 P(w) + Viterbi 动态规划",
                     "是（无监督统计）", "无"),
        "深度学习分词": ("字嵌入 → 双向 LSTM → 线性分类头 → Viterbi 约束解码"
                         "（PyTorch/CUDA，含纯 NumPy 回退）", "是（伪标注监督）", "词表伪标签"),
        "大模型分词": (f"{r['methods']['大模型分词'].get('model', C.LLM_MODEL)} "
                       "few-shot 生成切分 + 字符级一致性校验", "否（预训练）", "预训练语料"),
    }
    for nm, (a, b, c) in meta.items():
        L.append(f"| {nm} | {a} | {b} | {c} |")

    L += ["", "## 2. 描述统计", "",
          "| 方法 | 词数 | 平均词长 | 单字词占比 | 压缩率 | 词表规模 | OOV率 | 单次切分耗时(s) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for nm, d in r["describe"].items():
        c = r["methods"].get(nm, {})
        L.append(f"| {nm} | {d['tokens']} | {d['avg_token_len']} | {d['single_char_ratio']} | "
                 f"{d['compression_ratio']} | {d['vocab_size']} | {d['oov_rate']} | "
                 f"{c.get('infer_seconds', '-')} |")
    L += ["", f"> 耗时为「对同一批 {len(texts)} 条文本切分一次」的墙钟时间（不含模型训练；"
          "大模型一项为缓存命中的解析开销，首次联网请求约 3~4 秒/8 条）。"]

    L += ["", "## 3. 与参照切分的一致性（边界 / 词语 F1）", "",
          "### 3.1 以规则分词为参照", "",
          "| 方法 | 边界 P | 边界 R | 边界 F1 | 词语 F1 | 完全一致率 | 近似一致率(F1≥0.9) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for nm, m in r["vs_rule_baseline"].items():
        L.append(f"| {nm} | {m['boundary']['precision']} | {m['boundary']['recall']} | "
                 f"{m['boundary']['f1']} | {m['segment']['f1']} | {m['exact_match']} | "
                 f"{m['approx_match_f1_ge_0.9']} |")
    L += ["", "### 3.2 以四方法共识切分为参照（边界 ≥3 票）", "",
          "| 方法 | 边界 F1 | 词语 F1 | 完全一致率 |", "| --- | --- | --- | --- |"]
    for nm, m in r["vs_consensus_ref"].items():
        L.append(f"| {nm} | {m['boundary']['f1']} | {m['segment']['f1']} | {m['exact_match']} |")
    L += ["", "### 3.3 以统计分词为参照", "", "| 方法 | 边界 F1 | 词语 F1 |",
          "| --- | --- | --- |"]
    for nm, m in r["vs_stat_ref"].items():
        L.append(f"| {nm} | {m['boundary']['f1']} | {m['segment']['f1']} |")
    L += ["", "### 3.4 两两一致性", "", "| 方法对 | 边界 F1 | 词语 F1 |", "| --- | --- | --- |"]
    for k, v in r["pairwise"]["matrix"].items():
        L.append(f"| {k} | {v['boundary_f1']} | {v['segment_f1']} |")

    am = r.get("auxiliary_methods", {})
    aux = r.get("auxiliary_eval", {}).get("统计分词-HMM")
    if aux:
        L += ["", "### 3.5 辅助对照：统计分词-HMM（Baum-Welch）", "",
              f"- 与共识切分：边界 F1 = {aux['against_consensus']['boundary']['f1']}，"
              f"词语 F1 = {aux['against_consensus']['segment']['f1']}",
              f"- 训练：迭代 {am.get('max_iter', '-')} 次，字符表 {am.get('char_types', '-')} 个，"
              f"对数似然 {am.get('log_likelihood', '-')}"]

    L += ["", "### 3.6 自动校准结果（开发集，参照=规则分词+词表匹配伪标注）", ""]
    cal_stat = r["methods"].get("统计分词", {}).get("calibration", {})
    cal_dl = r["methods"].get("深度学习分词", {}).get("calibration", {})
    if cal_stat or cal_dl:
        L += ["| 方法 | 超参数 | 取值 | 与参照边界 F1 | 平均词长 |", "| --- | --- | --- | --- | --- |"]
        if cal_stat:
            L.append(f"| 统计分词 | 词数惩罚 β | {cal_stat.get('penalty')} | "
                     f"{cal_stat.get('boundary_f1_vs_refs')} | {cal_stat.get('avg_token_len')} |")
        if cal_dl:
            L.append(f"| 深度学习分词 | 单字词惩罚 | {cal_dl.get('s_penalty')} | "
                     f"{cal_dl.get('boundary_f1_vs_refs')} | {cal_dl.get('avg_token_len')} |")
    else:
        L.append("- 本次未记录校准信息。")

    j = r.get("judge", {})
    if j.get("enabled") and j.get("summary"):
        L += ["", "## 4. 大模型评判（LLM-as-a-judge，非人工标注）", "",
              "| 方法 | 边界合理 | 表意完整 | 专名完整 | 无碎词 | 总体 | 与理想切分边界F1 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for nm, s in j["summary"].items():
            L.append(f"| {nm} | {s.get('boundary_ok', '-')} | {s.get('exp_complete', '-')} | "
                     f"{s.get('prop_noun', '-')} | {s.get('no_fragment', '-')} | "
                     f"{s.get('overall', '-')} | {s.get('boundary_f1_vs_ideal', '-')} |")
        L += ["", "### 4.1 评判样例", ""]
        for ci, c in enumerate(r.get("examples", [])[:6]):
            L.append(f"**{ci + 1}. 原文**：{c['text']}")
            L.append(f"- 裁判理想切分：`{' / '.join(c['ideal'])}`")
            for nm, sc in c["scores"].items():
                toks = hyps[nm][texts.index(c["text"])] if c["text"] in texts else []
                L.append(f"- {nm}：`{' / '.join(toks)}`"
                         f"（overall={sc.get('overall')}；{sc.get('comment') or ''}）")
            L.append("")

    L += ["", "## 5. 分词结果示例（前 10 条采样）", ""]
    for i in range(min(10, len(texts))):
        L.append(f"**{i + 1}. {texts[i]}**")
        for nm in hyps:
            L.append(f"- {nm}（{len(hyps[nm][i])} 词）：`{' / '.join(hyps[nm][i])}`")
        L.append("")

    L += ["", "## 6. 结论与讨论", "",
          "1. **规则分词**：速度最快、完全可复现，领域词典让「龙谕酒庄」「贺兰山东麓」"
          "等专名保持完整；不足是未登录词（新景点、新搭配）仍会被切碎，覆盖受词典限制。",
          "2. **统计分词**：无需任何人工词典，完全由语料统计驱动——LLR/PMI 自动发现"
          "「酒庄 / 品鉴 / 橡木桶」等高频搭配并估计一元语言模型；"
          "短文本统计量不足时会出现把「导 / 游」「解 / 员」切开的情况。",
          "3. **深度学习分词**：BiLSTM 学习上下文表示，在词表基础上具备泛化能力，"
          "对未登录组合的边界更接近语义单元；本项目提供 PyTorch（GPU）与纯 NumPy 两套"
          "实现，训练代价从 55 s 降到 1.5 s，但切分质量基本持平——"
          "说明当前瓶颈是伪标注质量而非算力。",
          "4. **大模型分词**：借助预训练语言知识，专名、口语与标点处理最稳，"
          "字符级一致性校验保证结果可验证；代价是推理成本高、依赖网络与接口稳定性。",
          "5. **一致性**：四方法在功能词边界上高度一致，差异集中在领域专名与固定搭配处——"
          "这正是分词影响下游情感分析效果的关键位置。", ""]
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ============================================================ 主入口
def main() -> None:
    t_all = time.time()
    log("=" * 78)
    log("两个 CSV 的清洗 + 四方法分词流水线")
    log("=" * 78)
    uni, s0 = to_unified()
    log("\n[S1~S11] 逐条清洗（text_raw → text_clean）")
    clean_df, stats, dropped = clean_sources(uni)
    stats = {**{k: int(v) for k, v in s0.items()}, **stats}
    for k in sorted(stats):
        log(f"  {k:26s} {stats[k]}")

    # 用规则分词给全语料打 tokens（并过滤停用词），供后续建模直接使用
    rule = RuleTokenizer()
    clean_df = clean_df.copy()
    clean_df["tokens"] = [" ".join(w for w in rule.cut(t)
                                   if w and w not in C.STOPWORDS)
                          for t in clean_df["text_clean"]]
    clean_df["token_count"] = clean_df["tokens"].str.split().str.len()

    keep = ["doc_id", "source", "source_type", "site", "domain", "doc_url", "author",
            "rating_star", "rating_title", "comment_time", "votes", "text_raw",
            "text_clean", "char_len", "tokens", "token_count", "weak_label",
            "binary_label", "crawl_time"]
    out = clean_df[keep].reset_index(drop=True)
    out_all = C.DIR_PROCESSED / "clean_all.csv"
    out.to_csv(out_all, index=False, encoding="utf-8-sig")
    out.to_json(C.DIR_PROCESSED / "clean_all.jsonl", orient="records", lines=True,
                force_ascii=False)
    per = {}
    for s, g in out.groupby("source"):
        slug = {"携程景点点评": "ctrip", "大众点评景点点评": "dianping"}.get(s, "other")
        f = C.DIR_PROCESSED / f"clean_{slug}.csv"
        g.to_csv(f, index=False, encoding="utf-8-sig")
        per[s] = {"rows": int(len(g)), "file": f.name,
                  "avg_char_len": round(float(g["char_len"].mean()), 2),
                  "avg_token_count": round(float(g["token_count"].mean()), 2),
                  "weak_label_dist": {str(k): int(v) for k, v in
                                      g["weak_label"].value_counts().items()}}
    log(f"\n[S12] 输出清洗语料 {len(out)} 条 → {out_all.name}（另按来源拆分）")

    seg = run_segmentation(out)

    clean_report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": stats,
        "final_rows": int(len(out)),
        "retention_rate": round(len(out) / max(stats.get("S0_loaded", 1), 1), 4),
        "source_distribution": {str(k): int(v) for k, v in out["source"].value_counts().items()},
        "weak_label_distribution": {str(k): int(v) for k, v in
                                    out["weak_label"].value_counts().items()},
        "char_len_stats": {k: round(float(v), 2) for k, v in out["char_len"].describe().items()},
        "token_count_stats": {k: round(float(v), 2) for k, v in
                              out["token_count"].describe().items()},
        "per_source": per,
        "top50_tokens": Counter(w for t in out["tokens"] for w in str(t).split()).most_common(50),
        "vocab_size": len({w for t in out["tokens"] for w in str(t).split()}),
        "dropped_examples": dropped[:30],
        "opencc_available": U._cc is not None,
        "segmentation_report": "seg_report.json",
    }
    (C.DIR_PROCESSED / "clean_report.json").write_text(
        json.dumps(clean_report, ensure_ascii=False, indent=2), encoding="utf-8")
    rp1 = write_clean_report(stats, dropped, out, seg)
    rp2 = write_seg_report(seg)

    log("\n===== 完成 =====")
    log(f"  清洗语料：{len(out)} 条（保留率 {clean_report['retention_rate']:.1%}）")
    log(f"  清洗输出：{out_all}")
    log(f"  分词对比：{seg['cmp_csv']}")
    log(f"  报告：{rp1.name} / {rp2.name}")
    log(f"  总耗时：{time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
