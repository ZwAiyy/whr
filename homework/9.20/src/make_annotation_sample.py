# -*- coding: utf-8 -*-
"""标注样本构建：从清洗后语料中分层抽样，生成待标注任务 + 弱标签对照。

抽样策略：
  * 主样本来自「豆瓣读书短评」——该来源同时带有星级弱标签，可用于抽样分层，
    也便于把多标注员结果与星级弱标签做对照分析；
  * 按弱标签（正面/中性/负面/未知）分层随机抽样，保证类别均衡；
  * 抽取文本长度在 [4, 120] 字符之间的样本，避免过短/过长影响判断。

输出：
  annotation/tasks_to_annotate.json   Label Studio 导入格式（仅含 text，不泄露元数据）
  annotation/sample_meta.csv          抽样元数据（doc_id / 来源 / 星级 / 弱标签，仅供分析，不对标注员可见）
  annotation/silver_labels.csv        规则弱标签（用于模拟标注者与对照评价）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def lexicon_label(text: str) -> str:
    """词典规则弱标注（用于无星级样本兜底）。"""
    pos = sum(1 for w in C.POS_WORDS if w in text)
    neg = sum(1 for w in C.NEG_WORDS if w in text)
    if pos > neg:
        return C.LABEL_POS
    if neg > pos:
        return C.LABEL_NEG
    if pos == neg == 0:
        return C.LABEL_NEU
    return C.LABEL_NEU


def main() -> None:
    df = pd.read_csv(C.DIR_PROCESSED / "sentiment_all_clean.csv", dtype=str, keep_default_na=False)
    d = df[df["source"] == "豆瓣读书短评"].copy()
    d["char_len"] = d["text_clean"].str.len().astype(int)
    d = d[(d["char_len"] >= C.ANNOTATION_MIN_LEN) & (d["char_len"] <= C.ANNOTATION_MAX_LEN)]
    print(f"豆瓣来源可用样本：{len(d)} 条（长度 {C.ANNOTATION_MIN_LEN}-{C.ANNOTATION_MAX_LEN}）")

    def silver(row) -> str:
        w = row["weak_label"]
        if w in (C.LABEL_POS, C.LABEL_NEG, C.LABEL_NEU):
            return w
        return lexicon_label(row["text_clean"])

    d["silver_label"] = d.apply(silver, axis=1)
    d["star"] = d["rating_star"].replace("", "NA")

    rng = np.random.default_rng(C.RANDOM_SEED)
    per_class = C.ANNOTATION_SAMPLE_SIZE // 3
    picked = []
    for lab in [C.LABEL_POS, C.LABEL_NEG, C.LABEL_NEU]:
        pool = d[d["silver_label"] == lab]
        take = min(per_class, len(pool))
        idx = rng.choice(pool.index.values, size=take, replace=False)
        picked.append(pool.loc[idx])
        print(f"  分层 {lab}: 候选 {len(pool)} 条 → 抽样 {take} 条")

    # 若某层候选不足，用其余样本（按类别轮询）补齐到目标量
    chosen_ids = set(pd.concat(picked)["doc_id"]) if picked else set()
    rest = d[~d["doc_id"].isin(chosen_ids)]
    need = C.ANNOTATION_SAMPLE_SIZE - sum(len(p) for p in picked)
    if need > 0 and len(rest):
        rest = rest.sort_values(["silver_label", "doc_id"]).groupby("silver_label", sort=True).head(
            int(np.ceil(need / 3)) )
        idx = rng.choice(rest.index.values, size=min(need, len(rest)), replace=False)
        picked.append(rest.loc[idx])
        print(f"  补齐 {min(need, len(rest))} 条（原始层候选不足）")

    sample = pd.concat(picked).drop_duplicates(subset=["doc_id"])
    sample = sample.sample(frac=1.0, random_state=C.RANDOM_SEED).reset_index(drop=True)

    sample["sample_id"] = [f"T{i+1:04d}" for i in range(len(sample))]
    sample = sample.rename(columns={"text_clean": "text"})

    tasks = [{"data": {"text": r["text"], "sample_id": r["sample_id"]}} for _, r in sample.iterrows()]
    (C.DIR_ANNOTATION / "tasks_to_annotate.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")

    meta_cols = ["sample_id", "doc_id", "text", "star", "rating_title", "weak_label", "silver_label"]
    sample[meta_cols].to_csv(C.DIR_ANNOTATION / "sample_meta.csv", index=False, encoding="utf-8-sig")
    sample[["sample_id", "doc_id", "silver_label"]].to_csv(
        C.DIR_ANNOTATION / "silver_labels.csv", index=False, encoding="utf-8-sig")

    print(f"\n抽样完成：{len(sample)} 条 → {C.DIR_ANNOTATION / 'tasks_to_annotate.json'}")
    print("弱标签分布：", sample["silver_label"].value_counts().to_dict())
    print("星级分布：", sample["star"].value_counts().to_dict())


if __name__ == "__main__":
    main()
