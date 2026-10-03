# -*- coding: utf-8 -*-
"""小样本验证大模型评判（LLM-as-a-judge）。用法：python tests/test_judge.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg_dl import TaggerClass  # noqa: E402
from seg_judge import SegmentationJudge  # noqa: E402
from seg_rule import RuleTokenizer  # noqa: E402
from seg_stat import NGramSegmenter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/clean_all.csv", dtype=str, keep_default_na=False)
corpus = df["text_clean"].tolist()
texts = corpus[:6]
rule = RuleTokenizer()
ng = NGramSegmenter().fit(corpus)
dl = TaggerClass.load()
hyps = {
    "规则分词": rule.cut_batch(texts),
    "统计分词": ng.cut_batch(texts),
    "深度学习分词": dl.cut_batch(texts),
    "大模型分词": rule.cut_batch(texts),     # 占位：验证评判链路
}
j = SegmentationJudge()
res = j.judge(texts, hyps, log=print)
for t in texts:
    r = res.get(t)
    print("=" * 70)
    print("原文:", t[:60])
    if not r:
        print("  (无结果)")
        continue
    print("  理想:", " / ".join(r["ideal"]))
    for nm, sc in (r.get("scores") or {}).items():
        if isinstance(sc, dict):
            print(f"  {nm}: overall={sc.get('overall')} boundary={sc.get('boundary_ok')} "
                  f"专名={sc.get('prop_noun')} 无碎词={sc.get('no_fragment')} "
                  f"| {sc.get('comment')}")
print("stats:", j.stats_report())
