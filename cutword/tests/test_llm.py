# -*- coding: utf-8 -*-
"""小样本验证大模型分词（真实调用 API）。用法：python tests/test_llm.py"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from seg_llm import LLMSegmenter  # noqa: E402
from seg_rule import RuleTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
print("model:", C.LLM_MODEL, "| base:", C.LLM_BASE_URL, "| key set:", bool(C.LLM_API_KEY))
df = pd.read_csv(ROOT / "data/processed/clean_all.csv", dtype=str, keep_default_na=False)
texts = df["text_clean"].tolist()[:8]
rule = RuleTokenizer()
llm = LLMSegmenter(fallback=rule)
t0 = time.time()
out = llm.cut_batch(texts, log=print)
print(f"耗时 {time.time() - t0:.1f}s")
for t, o in zip(texts, out):
    ok = "".join(o).replace(" ", "") == t.replace(" ", "")
    print(f"[{'OK ' if ok else 'BAD'}] {t[:40]}")
    print("      规则:", " / ".join(rule.cut(t)[:16]))
    print("      大模型:", " / ".join(o[:16]))
print("统计:", llm.stats_report())
