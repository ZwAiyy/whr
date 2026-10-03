# -*- coding: utf-8 -*-
"""纯 NumPy 后端的数值梯度校验 + 小规模训练冒烟。

用法：python tests/test_dl_grad.py
（PyTorch 后端由 torch.autograd 保证梯度正确，因此这里校验的是 seg_dl_numpy）
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seg_dl as D  # noqa: E402
import seg_dl_numpy as DN  # noqa: E402


def loss_and_grads(model, batch):
    idx, tag, mask = model._pad(batch)
    logits = model.forward(idx)
    ls = DN._log_softmax(logits)
    loss = -(ls[np.arange(len(batch))[:, None], np.arange(idx.shape[1])[None, :], tag]
             * mask).sum() / mask.sum()
    dlogits = np.exp(ls)
    dlogits[np.arange(len(batch))[:, None], np.arange(idx.shape[1])[None, :], tag] -= 1
    dlogits = dlogits * mask[:, :, None] / mask.sum()
    model.backward(dlogits)
    grads = {
        ("emb", "W"): model.emb_grad,
        ("fwd", "Wx"): model.fwd.grads["Wx"], ("fwd", "Wh"): model.fwd.grads["Wh"],
        ("fwd", "b"): model.fwd.grads["b"],
        ("bwd", "Wx"): model.bwd.grads["Wx"], ("bwd", "Wh"): model.bwd.grads["Wh"],
        ("bwd", "b"): model.bwd.grads["b"],
        ("head", "W"): model.head.grads["W"], ("head", "b"): model.head.grads["b"],
    }
    return float(loss), grads


def main():
    model = DN.BiLSTMTagger(embed_dim=4, hidden_dim=5, seed=1)
    model.build_vocab(["abcde", "fghij"], min_count=1)
    model._build()
    batch = [([2, 3, 4, 5], [0, 2, 0, 2]), ([6, 7, 8], [3, 0, 2])]
    # 让损失对隐藏状态的依赖更强：放大 head 权重
    model.head.W *= 5.0
    loss0, grads = loss_and_grads(model, batch)
    print("loss =", round(loss0, 5))
    eps = 1e-4
    worst = 0.0
    for (layer_name, attr), g in grads.items():
        layer = getattr(model, layer_name)
        P = getattr(layer, attr)
        flat = np.random.default_rng(7).choice(P.size, size=min(6, P.size), replace=False)
        flat_idx = [tuple(int(v) for v in np.unravel_index(f, P.shape)) for f in flat]
        for ix in flat_idx:
            old = P[ix]
            P[ix] = old + eps
            lp, _ = loss_and_grads(model, batch)
            P[ix] = old - eps
            lm, _ = loss_and_grads(model, batch)
            P[ix] = old
            num = (lp - lm) / (2 * eps)
            ana = float(g[ix])
            denom = max(abs(num), abs(ana), 1e-8)
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            print(f"  {layer_name}.{attr}{ix} num={num:+.6f} ana={ana:+.6f} rel={rel:.2e}")
    print("worst relative error =", f"{worst:.3e}", "->", "PASS" if worst < 1e-4 else "FAIL")

    # 小规模训练冒烟
    texts = ["龙谕酒庄真的超出预期", "酒窖和品酒都在城堡里面", "讲解员专业细致"] * 4
    tags = [D.tokens_to_tags(t) for t in texts]
    m2 = DN.BiLSTMTagger(seed=2)
    m2.build_vocab(texts)
    info = m2.fit(texts, tags, epochs=2, log=print)
    print("train info:", info)
    print("cut:", m2.cut("龙谕酒庄真的超出预期"))
    print("\n当前默认后端：", D.backend_info())


if __name__ == "__main__":
    main()
