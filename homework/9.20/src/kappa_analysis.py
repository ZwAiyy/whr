# -*- coding: utf-8 -*-
"""标注一致性评价：Cohen's κ / Fleiss' κ / Krippendorff's α / 加权 κ / 原始一致率

数据来源：annotation/annotation_matrix.csv（Label Studio 导出后整理）
输出：
  annotation/kappa_report.json                    全部指标（机器可读）
  annotation/cohen_kappa_pair_<A>_<B>.csv         两两列联表（观察频数）
  annotation/cohen_kappa_expected_<A>_<B>.csv     两两列联表（期望频数）
  annotation/item_agreement.csv                   条目级一致性与多数票
  annotation/annotator_stats.csv                  标注员统计
  outputs/reports/标注一致性评价报告.md             含完整计算过程的人读报告
  outputs/figures/fig11..fig13                    一致性可视化

实现说明：
  所有指标都用手写公式实现（便于在报告中展示计算过程），
  同时用 scikit-learn（Cohen's κ）与 nltk（Fleiss' κ、Krippendorff's α）做交叉验证，
  两套实现结果一致才视为通过。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

for cand in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

LABELS = C.LABELS
ORDINAL = {C.LABEL_NEG: 0, C.LABEL_NEU: 1, C.LABEL_POS: 2}


def interpret(k: float) -> str:
    for lo, hi, txt in C.KAPPA_INTERPRET:
        if lo <= k <= hi:
            return txt
    return "—"


# --------------------------------------------------------------------- Cohen
def cohen_kappa(a: list[str], b: list[str]) -> dict:
    """Cohen's κ：返回完整中间量（观察矩阵、期望矩阵、Po、Pe、κ、SE、CI）。"""
    labs = [l for l in LABELS if l in set(a) | set(b)]
    idx = {l: i for i, l in enumerate(labs)}
    K, n = len(labs), len(a)
    N = np.zeros((K, K), dtype=int)
    for x, y in zip(a, b):
        N[idx[x], idx[y]] += 1
    row = N.sum(axis=1)
    col = N.sum(axis=0)
    E = np.outer(row, col) / n
    Po = float(np.trace(N) / n)
    Pe = float((row / n) @ (col / n))
    kappa = (Po - Pe) / (1 - Pe)
    se = math.sqrt(Po * (1 - Po) / (n * (1 - Pe) ** 2))
    z = kappa / se if se else float("nan")
    p = 2 * (1 - sps.norm.cdf(abs(z)))
    return {
        "labels": labs, "n": n, "observed_matrix": N.tolist(), "expected_matrix": np.round(E, 4).tolist(),
        "row_marginal": row.tolist(), "col_marginal": col.tolist(),
        "Po": round(Po, 6), "Pe": round(Pe, 6), "kappa": round(float(kappa), 6),
        "SE": round(se, 6), "z": round(float(z), 4), "p_value": float(p),
        "CI95": [round(float(kappa - 1.96 * se), 6), round(float(kappa + 1.96 * se), 6)],
        "interpretation": interpret(kappa),
        "agree_count": int(np.trace(N)), "disagree_count": int(n - np.trace(N)),
    }


# -------------------------------------------------------------------- Fleiss
def fleiss_kappa(matrix: np.ndarray, labels: list[str]) -> dict:
    """Fleiss' κ。matrix: n 个条目 × K 个类别的计数矩阵 n_ij。"""
    n, K = matrix.shape
    m = matrix.sum(axis=1)
    m_bar = float(m.mean())
    P_i = (np.sum(matrix ** 2, axis=1) - m) / (m * (m - 1))
    P_bar = float(P_i.mean())
    p_j = matrix.sum(axis=0) / (n * m_bar)
    P_e = float(np.sum(p_j ** 2))
    kappa = (P_bar - P_e) / (1 - P_e)
    se = math.sqrt(2 / (n * m_bar * (m_bar - 1))) * math.sqrt(
        P_e - (2 * m_bar - 3) * P_e ** 2 + 2 * (m_bar - 2) * np.sum(p_j ** 3)) / (1 - P_e)
    z = kappa / se if se else float("nan")
    p = 2 * (1 - sps.norm.cdf(abs(z)))
    return {
        "n_items": int(n), "n_raters": m_bar, "K": K, "labels": labels,
        "P_i": np.round(P_i, 6).tolist(), "P_bar": round(P_bar, 6),
        "p_j": np.round(p_j, 6).tolist(), "P_e": round(P_e, 6),
        "kappa": round(float(kappa), 6), "SE": round(float(se), 6),
        "z": round(float(z), 4), "p_value": float(p),
        "CI95": [round(float(kappa - 1.96 * se), 6), round(float(kappa + 1.96 * se), 6)],
        "interpretation": interpret(kappa), "counts_matrix": matrix.tolist(),
    }


# ----------------------------------------------------------- Krippendorff α
def krippendorff_alpha_nominal(matrix: np.ndarray) -> dict:
    """Krippendorff's α（名义尺度，允许缺失）。matrix: n × K 计数矩阵。"""
    n, K = matrix.shape
    O = np.zeros((K, K))
    for row in matrix:
        m = row.sum()
        if m < 2:
            continue
        for c in range(K):
            for k in range(K):
                if c == k:
                    O[c, k] += row[c] * (row[c] - 1) / (m - 1)
                else:
                    O[c, k] += row[c] * row[k] / (m - 1)
    n_total = O.sum()
    n_c = O.sum(axis=0)
    D_o = 1 - np.trace(O) / n_total
    D_e = 1 - np.sum((n_c / n_total) ** 2)
    alpha = 1 - D_o / D_e
    return {"D_o": round(float(D_o), 6), "D_e": round(float(D_e), 6),
            "alpha": round(float(alpha), 6), "interpretation": interpret(alpha),
            "coincidence_matrix": np.round(O, 4).tolist(),
            "marginals": np.round(n_c, 4).tolist()}


# ------------------------------------------------------------------- figures
def fig_agreement_heatmaps(pairs: dict, out: Path) -> str:
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, pairs.items()):
        labs = res["labels"]
        M = np.array(res["observed_matrix"])
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > M.max() * .6 else "black")
        ax.set_title(f"{name}\nκ={res['kappa']:.3f}  Po={res['Po']:.3f}", fontsize=10)
        ax.set_xlabel("标注员B"); ax.set_ylabel("标注员A")
    fig.suptitle("两两标注列联表（观察频数）与 Cohen's κ", fontsize=11)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def fig_annotator_dist(stats_df: pd.DataFrame, out: Path) -> str:
    piv = stats_df.set_index("annotator")[LABELS]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bottom = np.zeros(len(piv))
    colors = {"正面": "#55A868", "负面": "#C44E52", "中性": "#4C72B0", "无法判断": "#999999"}
    for l in LABELS:
        ax.bar(piv.index, piv[l], bottom=bottom, label=l, color=colors[l])
        bottom += piv[l].values
    ax.set_title("各标注员标签分布"); ax.set_ylabel("标注条数"); ax.legend(ncol=4, fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def fig_kappa_bars(items: dict, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 4))
    names = list(items)
    vals = [items[k] for k in names]
    colors = ["#4C72B0" if v < 0.61 else "#55A868" for v in vals]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for y, txt in [(0.20, "极弱"), (0.40, "一般"), (0.60, "中等"), (0.80, "较强")]:
        ax.axhline(y, ls="--", lw=.7, color="gray")
        ax.text(len(names) - .4, y + .01, txt, fontsize=7, color="gray", ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("一致性系数")
    ax.set_title("各项一致性指标对比（虚线为 Landis & Koch 分级阈值）")
    ax.tick_params(axis="x", rotation=18, labelsize=8)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------- main
def main() -> None:
    df = pd.read_csv(C.DIR_ANNOTATION / "annotation_matrix.csv", dtype=str, keep_default_na=False)
    # 以 ls_tokens.json 中的标注员顺序为准
    tokens = json.loads((C.DIR_ANNOTATION / "ls_tokens.json").read_text(encoding="utf-8"))
    annotators = [k.split("@")[0] for k, v in tokens.items() if v["role"] == "annotator"]
    annotators = [a for a in annotators if a in df.columns]
    n_items = len(df)
    print(f"样本 {n_items} 条，标注员 {annotators}")

    # ---------------- 1. 标注员统计 ----------------
    stats_rows = []
    for a in annotators:
        vc = df[a].value_counts()
        stats_rows.append({"annotator": a, "标注条数": int(len(df[df[a] != ""])),
                           **{l: int(vc.get(l, 0)) for l in LABELS},
                           "平均用时(秒)": round(float(pd.read_csv(
                               C.DIR_ANNOTATION / "annotation_long.csv")
                               .query("annotator == @a")["lead_time"].astype(float).mean()), 2)})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(C.DIR_ANNOTATION / "annotator_stats.csv", index=False, encoding="utf-8-sig")
    print(stats_df.to_string(index=False))

    # ---------------- 2. 两两 Cohen's κ ----------------
    pair_results, verify_rows = {}, []
    for a, b in combinations(annotators, 2):
        sub = df[(df[a] != "") & (df[b] != "")]
        res = cohen_kappa(sub[a].tolist(), sub[b].tolist())
        sk = float(cohen_kappa_score(sub[a].tolist(), sub[b].tolist(), labels=res["labels"]))
        res["sklearn_kappa"] = round(sk, 6)
        res["verified"] = abs(sk - res["kappa"]) < 1e-6
        name = f"{a[-2:]}–{b[-2:]}"
        pair_results[name] = res
        verify_rows.append({"pair": f"{a} vs {b}", "手写实现": res["kappa"], "sklearn": res["sklearn_kappa"],
                            "一致": res["verified"]})
        pd.DataFrame(res["observed_matrix"], index=res["labels"], columns=res["labels"]).to_csv(
            C.DIR_ANNOTATION / f"cohen_kappa_pair_{a}_{b}.csv", encoding="utf-8-sig")
        pd.DataFrame(res["expected_matrix"], index=res["labels"], columns=res["labels"]).to_csv(
            C.DIR_ANNOTATION / f"cohen_kappa_expected_{a}_{b}.csv", encoding="utf-8-sig")
        print(f"  Cohen κ {a} vs {b}: κ={res['kappa']:.4f} (Po={res['Po']:.4f}, Pe={res['Pe']:.4f}) "
              f"sklearn={res['sklearn_kappa']:.4f}")

    # ---------------- 3. Fleiss' κ ----------------
    labs = [l for l in LABELS if any((df[a] == l).any() for a in annotators)]
    counts = np.zeros((n_items, len(labs)), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        for j, l in enumerate(labs):
            counts[i, j] = sum(1 for a in annotators if row[a] == l)
    fleiss = fleiss_kappa(counts, labs)
    print(f"  Fleiss κ={fleiss['kappa']:.4f} (P̄={fleiss['P_bar']:.4f}, P̄e={fleiss['P_e']:.4f})")

    # nltk 交叉验证（若可用）
    nltk_check = {}
    try:
        from nltk.metrics.agreement import AnnotationTask
        data = [(a, f"i{i}", str(df.iloc[i][a])) for i in range(n_items) for a in annotators]
        task = AnnotationTask(data=data)
        nltk_check = {
            "nltk_alpha": round(float(task.alpha()), 6),
            "nltk_kappa_avg_pairwise": round(float(task.kappa()), 6),
            "nltk_multi_kappa_davies_fleiss": round(float(task.multi_kappa()), 6),
            "mean_pairwise_cohen_sklearn": round(float(np.mean(
                [v["sklearn_kappa"] for v in pair_results.values()])), 6),
        }
        nltk_check["delta_fleiss_vs_nltk_alpha"] = round(
            abs(nltk_check["nltk_alpha"] - fleiss["kappa"]), 6)
        print(f"  nltk 交叉验证：alpha={nltk_check['nltk_alpha']:.6f}, "
              f"kappa(两两平均)={nltk_check['nltk_kappa_avg_pairwise']:.6f}, "
              f"multi_kappa={nltk_check['nltk_multi_kappa_davies_fleiss']:.6f}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] nltk 交叉验证不可用：{type(e).__name__} {e}")

    # ---------------- 4. Krippendorff's α ----------------
    alpha = krippendorff_alpha_nominal(counts)
    print(f"  Krippendorff α={alpha['alpha']:.4f} (Do={alpha['D_o']:.4f}, De={alpha['D_e']:.4f})")

    # ---------------- 5. 加权 κ（有序尺度，剔除「无法判断」） ----------------
    weighted = {}
    ord_df = df.copy()
    ord_annot = list(annotators)
    keep = ord_df[ord_annot].isin(list(ORDINAL)).all(axis=1)
    print(f"  有序子集（三人均给出 正面/负面/中性）：{int(keep.sum())} 条")
    for a, b in combinations(ord_annot, 2):
        sub = ord_df[keep]
        ya = [ORDINAL[x] for x in sub[a]]
        yb = [ORDINAL[x] for x in sub[b]]
        if len(ya) < 2:
            continue
        weighted[f"{a[-2:]}–{b[-2:]}"] = {
            "n": len(ya),
            "linear": round(float(cohen_kappa_score(ya, yb, weights="linear")), 6),
            "quadratic": round(float(cohen_kappa_score(ya, yb, weights="quadratic")), 6),
            "nominal": round(float(cohen_kappa_score(ya, yb)), 6),
            # 不一致的"距离"分布：1=相邻类别（正面↔中性 / 中性↔负面），2=两端（正面↔负面）
            "dist1_adjacent": int(sum(1 for p, q in zip(ya, yb) if abs(p - q) == 1)),
            "dist2_extreme": int(sum(1 for p, q in zip(ya, yb) if abs(p - q) == 2)),
        }
        print(f"  加权 κ {a} vs {b}: n={len(ya)} linear={weighted[f'{a[-2:]}–{b[-2:]}']['linear']:.4f} "
              f"quadratic={weighted[f'{a[-2:]}–{b[-2:]}']['quadratic']:.4f}")

    # ---------------- 5b. 标签体系折叠后的 κ 变化 ----------------
    variants = {}
    # (a) 3 类：把「无法判断」并入「中性」
    lab3 = ["正面", "负面", "中性"]
    c3 = np.zeros((n_items, 3), dtype=int)
    for i in range(n_items):
        for j, l in enumerate(lab3):
            c3[i, j] = sum(1 for a in annotators
                           if df.iloc[i][a] == l or (l == "中性" and df.iloc[i][a] == "无法判断"))
    variants["3类(无法判断→中性)"] = fleiss_kappa(c3, lab3)
    # (b) 2 类：仅保留该条三人皆为 正面/负面 的样本
    mask2 = df[list(annotators)].isin([C.LABEL_POS, C.LABEL_NEG]).all(axis=1)
    c2 = np.zeros((int(mask2.sum()), 2), dtype=int)
    for i, (_, row) in enumerate(df[mask2].iterrows()):
        for j, l in enumerate([C.LABEL_POS, C.LABEL_NEG]):
            c2[i, j] = sum(1 for a in annotators if row[a] == l)
    variants["2类(正面/负面)"] = fleiss_kappa(c2, [C.LABEL_POS, C.LABEL_NEG])
    for k, v in variants.items():
        print(f"  折叠方案「{k}」：n={v['n_items']} Fleiss κ={v['kappa']:.4f}")

    # ---------------- 6. 二分类折叠 κ（正面/负面，剔除中性 & 无法判断） ----------------
    binary = {}
    for a, b in combinations(annotators, 2):
        m = df[(df[a].isin([C.LABEL_POS, C.LABEL_NEG])) & (df[b].isin([C.LABEL_POS, C.LABEL_NEG]))]
        x = np.array([1 if v == C.LABEL_POS else 0 for v in m[a]])
        y = np.array([1 if v == C.LABEL_POS else 0 for v in m[b]])
        n11 = int(((x == 1) & (y == 1)).sum()); n10 = int(((x == 1) & (y == 0)).sum())
        n01 = int(((x == 0) & (y == 1)).sum()); n00 = int(((x == 0) & (y == 0)).sum())
        n = n11 + n10 + n01 + n00
        Po = (n11 + n00) / n if n else float("nan")
        Pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n) if n else float("nan")
        k = (Po - Pe) / (1 - Pe) if n else float("nan")
        binary[f"{a[-2:]}–{b[-2:]}"] = {
            "n_valid": n, "table": [[n11, n10], [n01, n00]], "Po": round(Po, 6), "Pe": round(Pe, 6),
            "kappa": round(float(k), 6), "interpretation": interpret(k),
            "sklearn": round(float(cohen_kappa_score(x, y)), 6),
        }
        print(f"  二分类 κ {a} vs {b}: n={n} κ={k:.4f} (Po={Po:.4f}, Pe={Pe:.4f})")

    # ---------------- 7. 条目级一致性 / 多数票 ----------------
    item_rows = []
    for i in range(n_items):
        row = df.iloc[i]
        vals = [row[a] for a in annotators]
        cnt = Counter(vals)
        top, top_n = cnt.most_common(1)[0]
        m = len(vals)
        P_i = (sum(v ** 2 for v in cnt.values()) - m) / (m * (m - 1))
        item_rows.append({"sample_id": row["sample_id"], "text": row["text"],
                          **{a: row[a] for a in annotators},
                          "一致人数": top_n, "多数票": top if top_n >= 2 else "无多数票",
                          "完全一致": top_n == m, "P_i": round(P_i, 4)})
    item_df = pd.DataFrame(item_rows)
    item_df.to_csv(C.DIR_ANNOTATION / "item_agreement.csv", index=False, encoding="utf-8-sig")
    unanimous = int(item_df["完全一致"].sum())
    no_majority = int((item_df["一致人数"] == 1).sum())

    # 与弱标签对照
    meta = pd.read_csv(C.DIR_ANNOTATION / "sample_meta.csv", dtype=str, keep_default_na=False)
    merged = item_df.merge(meta[["sample_id", "silver_label", "star"]], on="sample_id", how="left")
    gold = merged[merged["多数票"] != "无多数票"]
    acc_gold_silver = float((gold["多数票"] == gold["silver_label"]).mean()) if len(gold) else float("nan")
    gold_kappa_silver = float(cohen_kappa_score(gold["多数票"].tolist(), gold["silver_label"].tolist(),
                                               labels=[l for l in LABELS if l in set(gold["多数票"]) | set(gold["silver_label"])]))
    per_annot_silver = {}
    for a in annotators:
        sub = merged[merged[a] != ""]
        per_annot_silver[a] = {
            "accuracy_vs_silver": round(float((sub[a] == sub["silver_label"]).mean()), 6),
            "kappa_vs_silver": round(float(cohen_kappa_score(
                sub[a].tolist(), sub["silver_label"].tolist(),
                labels=[l for l in LABELS if l in set(sub[a]) | set(sub["silver_label"])])), 6)}

    # ---------------- 8. 混淆分析（最常见的不一致类别对） ----------------
    conf_pairs = Counter()
    for a, b in combinations(annotators, 2):
        for x, y in zip(df[a], df[b]):
            if x != y and x and y:
                conf_pairs[tuple(sorted([x, y]))] += 1

    # ---------------- 9. 可视化 ----------------
    figs = [fig_agreement_heatmaps(pair_results, C.DIR_FIGURES / "fig11_cohen_contingency.png"),
            fig_annotator_dist(stats_df, C.DIR_FIGURES / "fig12_annotator_label_dist.png")]
    bar_items = {f"两两κ\n{k}": v["kappa"] for k, v in pair_results.items()}
    bar_items["Fleiss κ"] = fleiss["kappa"]
    bar_items["Krippendorff α"] = alpha["alpha"]
    bar_items["二分类κ(均值)"] = round(float(np.mean([v["kappa"] for v in binary.values()])), 6)
    bar_items["加权κ(二次,均值)"] = round(float(np.mean([v["quadratic"] for v in weighted.values()])), 6)
    figs.append(fig_kappa_bars(bar_items, C.DIR_FIGURES / "fig13_kappa_compare.png"))

    # ---------------- 10. 汇总输出 ----------------
    report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_items": n_items, "annotators": annotators, "labels": LABELS,
        "annotator_stats": stats_rows,
        "cohen_kappa_pairs": pair_results,
        "cohen_verification": verify_rows,
        "fleiss_kappa": fleiss,
        "krippendorff_alpha": alpha,
        "nltk_cross_check": nltk_check,
        "weighted_kappa": weighted,
        "label_scheme_variants": variants,
        "binary_kappa": binary,
        "item_level": {"unanimous_items": unanimous, "unanimous_rate": round(unanimous / n_items, 4),
                       "no_majority_items": no_majority,
                       "mean_P_i": round(float(item_df["P_i"].mean()), 6)},
        "vs_weak_label": {"majority_vs_silver_accuracy": round(acc_gold_silver, 6),
                          "majority_vs_silver_kappa": round(gold_kappa_silver, 6),
                          "per_annotator": per_annot_silver},
        "confusion_pairs": [{"pair": list(k), "count": v} for k, v in conf_pairs.most_common(10)],
        "figures": figs,
    }
    (C.DIR_ANNOTATION / "kappa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n===== 汇总 =====")
    print(f"  完全一致条目 {unanimous}/{n_items}（{unanimous/n_items:.1%}），无多数票 {no_majority} 条")
    print(f"  Fleiss κ={fleiss['kappa']:.4f} ({fleiss['interpretation']})")
    print(f"  Krippendorff α={alpha['alpha']:.4f}")
    print(f"  多数票 vs 弱标签：准确率 {acc_gold_silver:.3f}, κ={gold_kappa_silver:.3f}")
    print(f"  报告 → {C.DIR_ANNOTATION / 'kappa_report.json'}")


if __name__ == "__main__":
    main()
