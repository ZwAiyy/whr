# -*- coding: utf-8 -*-
"""PyTorch 后端冒烟测试：后端选择、训练、解码、保存/加载。

用法：python tests/test_dl_torch.py
（无 torch 环境会自动跳过并提示）
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seg_dl as D  # noqa: E402

TEXTS = ["龙谕酒庄真的超出预期", "酒窖和品酒都在城堡里面", "讲解员专业细致值得推荐"] * 8
GOLD = [["龙谕酒庄", "真的", "超出", "预期"],
        ["酒窖", "和", "品酒", "都", "在", "城堡", "里面"],
        ["讲解员", "专业", "细致", "值得", "推荐"]] * 8


def main():
    info = D.backend_info()
    print("后端信息:", info)
    if info["backend"] != "torch":
        print("当前不在 torch 后端（DL_FORCE_NUMPY=1 或未安装 torch），跳过本测试")
        return
    tags = [D.tokens_to_tags(t) for t in GOLD]
    m = D.make_tagger(seed=20240501)
    m.build_vocab(TEXTS)
    res = m.fit(TEXTS, tags, epochs=40, log=lambda *_: None)
    print(f"训练：{res['train_seconds']}s  设备={res['device']}  "
          f"最终 val_tag_acc={res['history'][-1]['val_tag_acc']}")
    assert res["history"][-1]["val_tag_acc"] > 0.95, "过拟合小样本失败，模型可能有问题"
    for t in ["龙谕酒庄真的超出预期", "讲解员专业细致"]:
        print(f"cut({t}) =", m.cut(t))
    with tempfile.TemporaryDirectory() as d:
        p = m.save(str(Path(d) / "m.pt"))
        m2 = D.TaggerClass.load(p)
        same = m2.cut("龙谕酒庄真的超出预期") == m.cut("龙谕酒庄真的超出预期")
        print(f"保存/加载一致: {same}")
        assert same
    print("\nPyTorch 后端测试通过")


if __name__ == "__main__":
    main()
