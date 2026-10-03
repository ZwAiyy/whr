# -*- coding: utf-8 -*-
"""公开数据集特征分析 + 多来源语料对比分析

输入：
  data/raw/public_dataset/*.csv                原始公开数据集
  data/processed/sentiment_all_clean.csv       清洗后的多来源语料
输出：
  outputs/reports/公开数据集特征分析.md          文字报告
  outputs/figures/*.png                        全部图表
  data/processed/feature_dataset_profile.csv   数据集画像表
  data/processed/feature_length_stats.csv      长度统计表
  data/processed/feature_top_words.csv         词频表
  data/processed/feature_logodds_words.csv     类区分度词表
  data/processed/feature_summary.json          汇总 JSON
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 绘图环境
for cand in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

RE_KEEP = re.compile(r"[^\u4e00-\u9fff\u3040-\u30ffa-zA-Z0-9]")


def dedup_key(t: str) -> str:
    return RE_KEEP.sub("", str(t)).lower()


def tokenize(t: str) -> list[str]:
    return [w for w in jieba.lcut(str(t))
            if w.strip() and w not in C.STOPWORDS and not re.fullmatch(r"[\d\W_]+", w)]


def save(fig, name: str) -> str:
    p = C.DIR_FIGURES / name
    fig.savefig(p)
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- 基础统计
def series_text(df: pd.DataFrame, col: str) -> pd.Series:
    """统一的文本列读取：缺失值 → 空串（兼容 pandas 3.x 的 str 类型 NA 语义）。"""
    return df[col].fillna("").astype(str)


def profile_df(name: str, df: pd.DataFrame, text_col: str, label_col: str | None) -> dict:
    info = {
        "数据集": name,
        "样本量": int(len(df)),
        "字段数": int(df.shape[1]),
        "字段": "|".join(map(str, df.columns[:20])),
        "缺失值总数": int(df.isna().sum().sum()),
        "完全重复行": int(df.duplicated().sum()),
    }
    if label_col and label_col in df.columns:
        vc = df[label_col].value_counts()
        info["类别数"] = int(len(vc))
        info["标签分布"] = json.dumps({str(k): int(v) for k, v in vc.items()}, ensure_ascii=False)
        info["不平衡比(最大/最小)"] = round(float(vc.max() / max(vc.min(), 1)), 3)
    if text_col and text_col in df.columns:
        s = series_text(df, text_col)
        L = s.str.len()
        info["平均字符数"] = round(float(L.mean()), 2)
        info["字符数中位数"] = int(L.median())
        info["最长字符数"] = int(L.max())
        toks = [t for x in s.sample(min(len(s), 4000), random_state=C.RANDOM_SEED) for t in tokenize(x)]
        c = Counter(toks)
        info["抽样词表规模"] = len(c)
        info["hapax比例"] = round(sum(1 for w, n in c.items() if n == 1) / max(len(c), 1), 4)
        info["近似重复率"] = round(float(s.map(dedup_key).duplicated().mean()), 4)
    return info


def length_stats(df: pd.DataFrame, text_col: str, label_col: str | None, ds: str) -> pd.DataFrame:
    s = series_text(df, text_col)
    rows = [{"数据集": ds, "分组": "全部", "样本数": len(s),
             "均值": round(s.str.len().mean(), 2), "标准差": round(s.str.len().std(), 2),
             "最小": int(s.str.len().min()), "25%": float(s.str.len().quantile(.25)),
             "中位数": float(s.str.len().median()), "75%": float(s.str.len().quantile(.75)),
             "最大": int(s.str.len().max())}]
    if label_col and label_col in df.columns:
        for lab, g in df.groupby(label_col):
            L = series_text(g, text_col).str.len()
            rows.append({"数据集": ds, "分组": f"label={lab}", "样本数": len(g),
                         "均值": round(L.mean(), 2), "标准差": round(L.std(), 2),
                         "最小": int(L.min()), "25%": float(L.quantile(.25)),
                         "中位数": float(L.median()), "75%": float(L.quantile(.75)),
                         "最大": int(L.max())})
    return pd.DataFrame(rows)


def log_odds(df: pd.DataFrame, text_col: str, label_col: str, min_count: int = 8) -> pd.DataFrame:
    """类别区分度：带先验平滑的对数几率比（log-odds ratio with informative Dirichlet prior）。"""
    labs = sorted(df[label_col].unique())
    if len(labs) != 2:
        return pd.DataFrame()
    cnt = {l: Counter(t for x in series_text(df[df[label_col] == l], text_col) for t in tokenize(x))
           for l in labs}
    total = {l: sum(cnt[l].values()) for l in labs}
    vocab = set(cnt[labs[0]]) | set(cnt[labs[1]])
    prior = Counter()
    for l in labs:
        prior.update(cnt[l])
    prior_total = sum(prior.values()) or 1
    rows = []
    for w in vocab:
        y1, y2 = cnt[labs[0]].get(w, 0), cnt[labs[1]].get(w, 0)
        if y1 + y2 < min_count:
            continue
        a0 = 0.01 * prior_total
        p1 = (y1 + a0 * prior[w] / prior_total) / (total[labs[0]] + a0)
        p2 = (y2 + a0 * prior[w] / prior_total) / (total[labs[1]] + a0)
        delta = math.log(p1 / (1 - p1)) - math.log(p2 / (1 - p2))
        var = 1 / (y1 + a0 * prior[w] / prior_total) + 1 / (y2 + a0 * prior[w] / prior_total)
        rows.append({"数据集": "", "词": w, f"计数_label{labs[0]}": y1, f"计数_label{labs[1]}": y2,
                     "log_odds": round(delta, 4), "z_score": round(delta / math.sqrt(var), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 图表
def fig_label_dist(datasets: dict[str, tuple[pd.DataFrame, str]]) -> str:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, (df, lab)) in zip(axes, datasets.items()):
        vc = df[lab].value_counts().sort_index()
        bars = ax.bar([str(i) for i in vc.index], vc.values, color=PALETTE[:len(vc)])
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n{v/vc.sum():.1%}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{name} 标签分布", fontsize=10)
        ax.set_ylim(0, vc.max() * 1.25)
        ax.set_xlabel("情感标签"); ax.set_ylabel("样本数")
    return save(fig, "fig01_label_distribution.png")


def fig_length_hist(datasets: dict[str, tuple[pd.DataFrame, str]]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for i, (name, (df, lab)) in enumerate(datasets.items()):
        L = series_text(df, "review").str.len()
        ax.hist(L.clip(upper=250), bins=50, alpha=.6, label=f"{name} (mean={L.mean():.1f})",
                color=PALETTE[i])
    ax.set_xlabel("评论文本字符数（截断至 250）"); ax.set_ylabel("样本数")
    ax.set_title("公开数据集文本长度分布对比"); ax.legend()
    return save(fig, "fig02_length_hist.png")


def fig_length_box(df: pd.DataFrame, text_col: str, lab: str, name: str) -> str:
    data = [series_text(df[df[lab] == l], text_col).str.len().values for l in sorted(df[lab].unique())]
    fig, ax = plt.subplots(figsize=(5.4, 4))
    bp = ax.boxplot(data, tick_labels=[str(l) for l in sorted(df[lab].unique())],
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(c); patch.set_alpha(.7)
    ax.set_xlabel("情感标签"); ax.set_ylabel("字符数")
    ax.set_title(f"{name} 各情感类别文本长度箱线图")
    return save(fig, f"fig03_length_box_{re.sub(r'[^0-9A-Za-z]+', '_', name)}.png")


def fig_top_words(df: pd.DataFrame, text_col: str, name: str, topn: int = 25) -> tuple[str, list]:
    c = Counter(t for x in series_text(df, text_col) for t in tokenize(x))
    common = c.most_common(topn)
    fig, ax = plt.subplots(figsize=(7, 5))
    words = [w for w, _ in common][::-1]
    vals = [v for _, v in common][::-1]
    ax.barh(words, vals, color="#4C72B0")
    ax.set_title(f"{name} 高频词 Top{topn}")
    ax.set_xlabel("出现次数")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", fontsize=7)
    return save(fig, f"fig04_top_words_{re.sub(r'[^0-9A-Za-z]+', '_', name)}.png"), common


def fig_zipf(dfs: dict[str, pd.DataFrame], text_col: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (name, df) in enumerate(dfs.items()):
        c = Counter(t for x in series_text(df.sample(min(len(df), 6000), random_state=1), text_col)
                    for t in tokenize(x))
        freqs = np.array(sorted(c.values(), reverse=True))
        ax.loglog(np.arange(1, len(freqs) + 1), freqs, label=name, color=PALETTE[i], lw=1.4)
    ax.set_xlabel("词频排名 (log)"); ax.set_ylabel("出现次数 (log)")
    ax.set_title("词频分布（Zipf 定律检验）"); ax.legend()
    ax.grid(alpha=.3, which="both")
    return save(fig, "fig05_zipf.png")


def fig_logodds(lo: pd.DataFrame, name: str, topn: int = 15) -> str:
    lo = lo.sort_values("log_odds")
    neg = lo.head(topn); pos = lo.tail(topn)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.barh(neg["词"], neg["log_odds"], color="#C44E52")
    ax.barh(pos["词"], pos["log_odds"], color="#55A868")
    ax.axvline(0, color="k", lw=.8)
    ax.set_title(f"{name} 类别区分度词（log-odds ratio）")
    ax.set_xlabel("log-odds（负 ← → 正）")
    return save(fig, f"fig06_logodds_{re.sub(r'[^0-9A-Za-z]+', '_', name)}.png")


def fig_source_compare(clean: pd.DataFrame) -> str:
    g = clean.groupby("source").agg(样本数=("doc_id", "count"), 平均字符数=("char_len", "mean"),
                                    平均词数=("token_count", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, c in zip(axes, ["样本数", "平均字符数", "平均词数"], PALETTE):
        bars = ax.bar(g.index, g[col], color=c)
        ax.set_title(f"各来源 {col}", fontsize=10)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        for b, v in zip(bars, g[col]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.1f}" if col != "样本数" else f"{int(v):,}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, g[col].max() * 1.2)
    fig.suptitle("多来源语料规模与文本长度对比", fontsize=11)
    return save(fig, "fig07_source_compare.png")


def fig_online_shoppers(df: pd.DataFrame) -> tuple[str, str]:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.6, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Online Shoppers 数值特征相关系数矩阵")
    fig.colorbar(im, ax=ax, shrink=.8)
    f1 = save(fig, "fig08_corr_heatmap_online_shoppers.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    if "Revenue" in df.columns:
        vc = df["Revenue"].value_counts()
        axes[0].bar([str(i) for i in vc.index], vc.values, color=PALETTE[:2])
        axes[0].set_title("Revenue 分布（购买意图）")
        for i, v in enumerate(vc.values):
            axes[0].text(i, v, f"{v}\n{v/vc.sum():.1%}", ha="center", va="bottom", fontsize=8)
    if "VisitorType" in df.columns:
        vc2 = df["VisitorType"].value_counts()
        axes[1].barh([str(i) for i in vc2.index], vc2.values, color="#8172B3")
        axes[1].set_title("访客类型分布")
    if "Month" in df.columns:
        order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        vc3 = df["Month"].value_counts().reindex([m for m in order if m in set(df["Month"])])
        axes[2].plot(vc3.index, vc3.values, marker="o", color="#DD8452")
        axes[2].set_title("月度访问量")
    f2 = save(fig, "fig09_online_shoppers_categorical.png")
    return f1, f2


def fig_tfidf(df: pd.DataFrame, text_col: str, lab: str, name: str, topn: int = 12) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = {str(l): " ".join(tokenize(" ".join(
        series_text(df[df[lab] == l], text_col).tolist()[:3000]))) for l in sorted(df[lab].unique())}
    keys = list(docs)
    vec = TfidfVectorizer(max_features=4000)
    X = vec.fit_transform([docs[k] for k in keys])
    terms = np.array(vec.get_feature_names_out())
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.2))
    axes = np.atleast_1d(axes)
    for ax, k, row in zip(axes, keys, X.toarray()):
        idx = row.argsort()[::-1][:topn]
        ax.barh(terms[idx][::-1], row[idx][::-1], color="#55A868")
        ax.set_title(f"{name} 类别「{k}」TF-IDF 关键词", fontsize=10)
    return save(fig, f"fig10_tfidf_{re.sub(r'[^0-9A-Za-z]+', '_', name)}.png")


# ---------------------------------------------------------------- 主流程
def main() -> None:
    summary: dict = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "datasets": {}}
    profiles: list[dict] = []
    length_tables: list[pd.DataFrame] = []
    top_word_tables: list[pd.DataFrame] = []
    logodds_tables: list[pd.DataFrame] = []

    d1 = pd.read_csv(C.DIR_RAW_DATASET / "ChnSentiCorp_htl_all.csv")
    d2 = pd.read_csv(C.DIR_RAW_DATASET / "waimai_10k.csv")
    d3 = pd.read_csv(C.DIR_RAW_DATASET / "online_shoppers_intention.csv")

    for nm, df, tc, lc in [("ChnSentiCorp酒店评论", d1, "review", "label"),
                           ("waimai外卖评论", d2, "review", "label"),
                           ("OnlineShoppers", d3, None, "Revenue")]:
        profiles.append(profile_df(nm, df, tc, lc))

    # 可视化用（文本数据集）
    text_ds = {"ChnSentiCorp": (d1, "label"), "waimai_10k": (d2, "label")}

    figs = [fig_label_dist(text_ds),
            fig_length_hist({k: (v[0], v[1]) for k, v in text_ds.items()})]

    for nm, (df, lab) in text_ds.items():
        length_tables.append(length_stats(df, "review", "label", nm))
        f, common = fig_top_words(df, "review", nm)
        figs.append(f)
        top_word_tables.append(pd.DataFrame(common, columns=["词", "频次"]).assign(数据集=nm))
        lo = log_odds(df, "review", "label")
        if not lo.empty:
            lo["数据集"] = nm
            logodds_tables.append(lo)
            figs.append(fig_logodds(lo, nm))
            try:
                figs.append(fig_tfidf(df, "review", "label", nm))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] TF-IDF 图失败 {nm}: {e}")

    figs.append(fig_length_box(d1, "review", "label", "ChnSentiCorp"))
    figs.append(fig_length_box(d2, "review", "label", "waimai_10k"))
    figs.append(fig_zipf({"ChnSentiCorp": d1, "waimai_10k": d2}, "review"))
    f1, f2 = fig_online_shoppers(d3)
    figs += [f1, f2]

    # 多来源对比
    clean = pd.read_csv(C.DIR_PROCESSED / "sentiment_all_clean.csv")
    figs.append(fig_source_compare(clean))
    profiles.append(profile_df("多来源清洗后语料", clean, "text_clean", "weak_label"))

    # ---------------- 落盘 ----------------
    pd.DataFrame(profiles).to_csv(C.DIR_PROCESSED / "feature_dataset_profile.csv",
                                  index=False, encoding="utf-8-sig")
    length_df = pd.concat(length_tables, ignore_index=True)
    length_df.to_csv(C.DIR_PROCESSED / "feature_length_stats.csv", index=False, encoding="utf-8-sig")
    pd.concat(top_word_tables, ignore_index=True).to_csv(
        C.DIR_PROCESSED / "feature_top_words.csv", index=False, encoding="utf-8-sig")
    if logodds_tables:
        pd.concat(logodds_tables, ignore_index=True).to_csv(
            C.DIR_PROCESSED / "feature_logodds_words.csv", index=False, encoding="utf-8-sig")

    # 类条件高频词（写入 JSON，供报告引用）
    for nm, (df, lab) in text_ds.items():
        cond = {}
        for l in sorted(df[lab].unique()):
            c = Counter(t for x in series_text(df[df[lab] == l], "review") for t in tokenize(x))
            cond[str(l)] = c.most_common(15)
        summary["datasets"][nm] = {"class_top_words": cond}

    summary["profiles"] = profiles
    summary["figures"] = figs
    (C.DIR_PROCESSED / "feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("生成图表：")
    for f in figs:
        print("  ", f)
    print("画像表：", C.DIR_PROCESSED / "feature_dataset_profile.csv")


if __name__ == "__main__":
    main()
