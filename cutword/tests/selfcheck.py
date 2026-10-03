# -*- coding: utf-8 -*-
"""自检：模块导入 + 单元级校验（梯度、指标、清洗规则）。

用法：python tests/selfcheck.py
"""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 项目根目录加入 sys.path

MODS = ["config", "clean_utils", "seg_rule", "seg_stat", "seg_dl", "seg_llm",
        "seg_metrics", "seg_judge", "clean_pipeline"]


def test_imports():
    ok = True
    for m in MODS:
        try:
            importlib.import_module(m)
            print(f"  OK   {m}")
        except Exception:
            ok = False
            print(f"  FAIL {m}")
            traceback.print_exc()
    return ok


def test_clean():
    import clean_utils as U
    cases = [
        ("<p>好喝&nbsp;！</p>https://a.com @张三 #话题# [大笑] 😀 全角１２３",
         ["好喝", "123", "全角123"]),
        ("通过大众点评消费\n酒庄很漂亮[比心]\n2026-09出行｜游玩1-2小时｜情侣夫妻",
         ["酒庄很漂亮"]),
        ("這是一個測試", ["这是一个测试"]),
    ]
    ok = True
    for raw, expects in cases:
        out = U.clean_text(raw)
        good = all(e in out for e in expects)
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {raw[:26]!r} -> {out!r}")
    # 平台标签必须被清掉
    from clean_pipeline import _clean_dianping_content
    pre = _clean_dianping_content("好喝！\n2026-09出行｜游玩1-2小时｜情侣夫妻")
    good = "出行" not in pre and "游玩" not in pre and "情侣夫妻" not in pre and "好喝" in pre
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} 平台标签清除 -> {pre!r}")
    return ok


def test_metrics():
    import seg_metrics as M
    text = "酒庄的风景很好"
    a = ["酒庄", "的", "风景", "很", "好"]
    b = ["酒庄", "的", "风景", "很好"]
    e = M.evaluate([text], [a], [b])
    print(f"  boundary F1={e['boundary']['f1']} segment F1={e['segment']['f1']} "
          f"exact={e['exact_match']}")
    # 标点不应影响打分
    e2 = M.evaluate([text], [a + ["，"]], [b + ["，", "。"]])
    same = e2["boundary"]["f1"] == e["boundary"]["f1"]
    print(f"  {'OK ' if same else 'BAD'} 标点归一化生效（F1={e2['boundary']['f1']}）")
    return same


def test_gradients():
    import numpy as np

    import seg_dl as D
    rng = np.random.default_rng(3)
    lstm = D.LSTM(4, 4, rng)
    x = rng.normal(size=(2, 5, 4))
    w = rng.normal(size=(2, 5, 4))
    lstm.forward(x)
    dx = lstm.backward(w)
    eps, worst = 1e-6, 0.0
    for _ in range(6):
        i, j = (int(rng.integers(s)) for s in lstm.Wh.shape)
        old = lstm.Wh[i, j]
        lstm.Wh[i, j] = old + eps
        lp = float((lstm.forward(x) * w).sum())
        lstm.Wh[i, j] = old - eps
        lm = float((lstm.forward(x) * w).sum())
        lstm.Wh[i, j] = old
        num = (lp - lm) / (2 * eps)
        ana = float(lstm.grads["Wh"][i, j])
        worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1e-12))
    print(f"  LSTM dWh 最大相对误差 {worst:.2e} -> {'PASS' if worst < 1e-5 else 'FAIL'}")
    return worst < 1e-5


def main():
    print("[1] 模块导入")
    r = [test_imports()]
    print("[2] 清洗规则")
    r.append(test_clean())
    print("[3] 评价指标")
    r.append(test_metrics())
    print("[4] 深度学习梯度校验")
    r.append(test_gradients())
    print("\n结论:", "全部通过" if all(r) else "存在失败项")


if __name__ == "__main__":
    main()
