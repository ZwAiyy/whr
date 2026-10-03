# -*- coding: utf-8 -*-
"""分词方法三的回退实现：BiLSTM 序列标注（纯 NumPy，手写前向/反向传播）。

本项目默认使用 PyTorch 后端（`seg_dl.py`，CUDA 可用时自动用 GPU）。
若运行环境没有 torch，`seg_dl.TaggerClass` 会自动回退到本模块，
接口（fit / cut / cut_batch / calibrate / save / load / stats）完全一致。

实现要点：
  网络：字嵌入 → 双向 LSTM（前向 + 反向）→ 线性分类头 → softmax
  训练：Adam 优化 + 全局梯度裁剪，反向传播（含 BPTT）全部手写
  解码：Viterbi + 合法转移约束（B/M/E/S）+ 可校准的单字词惩罚
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

import config as C

N_TAGS, TAG_NAMES = 4, ["B", "M", "E", "S"]
_NEG = -1e18
TRANS_MASK = np.array([
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
], dtype=bool)
START_ALLOWED = np.array([1, 0, 0, 1], dtype=bool)
END_ALLOWED = np.array([0, 0, 1, 1], dtype=bool)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


class Embedding:
    def __init__(self, vocab_size: int, dim: int, rng: np.random.Generator):
        self.W = rng.normal(0, 0.1, size=(vocab_size, dim)).astype(np.float64)
        self.dim = dim

    def forward(self, idx: np.ndarray) -> np.ndarray:
        self.idx = idx
        return self.W[idx]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.idx.reshape(-1), dout.reshape(-1, self.dim))
        return dW


class LSTM:
    """单向 LSTM 层，输入 (B,T,D)，输出 (B,T,H)。"""

    def __init__(self, in_dim: int, hidden: int, rng: np.random.Generator):
        self.hidden, self.in_dim = hidden, in_dim
        k = 1.0 / math.sqrt(hidden)
        self.Wx = rng.uniform(-k, k, size=(in_dim, 4 * hidden))
        self.Wh = rng.uniform(-k, k, size=(hidden, 4 * hidden))
        self.b = np.zeros(4 * hidden)
        self.b[hidden:2 * hidden] = 1.0          # forget gate bias = 1
        self.grads: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        H = self.hidden
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        self.cache = []
        hs = np.zeros((B, T, H))
        self.x = x
        xW = x @ self.Wx + self.b
        for t in range(T):
            h_prev = h
            z = xW[:, t, :] + h_prev @ self.Wh
            i = _sigmoid(z[:, 0:H])
            f = _sigmoid(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H])
            o = _sigmoid(z[:, 3 * H:4 * H])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            hs[:, t, :] = h
            self.cache.append((i, f, g, o, tc, c_prev, h_prev, x[:, t, :]))
        return hs

    def backward(self, dh_seq: np.ndarray) -> np.ndarray:
        B, T, H = dh_seq.shape
        dx = np.zeros_like(self.x)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in range(T - 1, -1, -1):
            i, f, g, o, tc, c_prev, h_prev, x_t = self.cache[t]
            dh = dh_seq[:, t, :] + dh_next
            do = dh * tc
            dc = dh * o * (1 - tc ** 2) + dc_next
            di, dg, df = dc * g, dc * i, dc * c_prev
            dc_prev = dc * f
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
            db += dz.sum(axis=0)
            dWx += x_t.T @ dz
            dWh += h_prev.T @ dz
            dx[:, t, :] = dz @ self.Wx.T
            dh_next = dz @ self.Wh.T
            dc_next = dc_prev
        self.grads = {"Wx": dWx, "Wh": dWh, "b": db}
        return dx


class Linear:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        k = 1.0 / math.sqrt(in_dim)
        self.W = rng.uniform(-k, k, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.grads = {"W": np.tensordot(self.x, dout, axes=([0, 1], [0, 1])),
                      "b": dout.sum(axis=(0, 1))}
        return dout @ self.W.T


class BiLSTMTagger:
    """纯 NumPy 版 BiLSTM 序列标注分词器（无 torch 时的回退实现）。"""

    name = "deep"
    display = "深度学习分词"
    backend = "numpy"

    def __init__(self, embed_dim: int | None = None, hidden_dim: int | None = None,
                 seed: int | None = None, tag_penalty: float | None = None):
        self.embed_dim = embed_dim or C.DL_EMBED_DIM
        self.hidden_dim = hidden_dim or C.DL_HIDDEN_DIM
        self.rng = np.random.default_rng(seed if seed is not None else C.DL_SEED)
        self.tag_penalty = C.DL_TAG_PENALTY if tag_penalty is None else tag_penalty
        self.calibration: dict = {}
        self.built = False

    # ---------------------------------------------------------- 词表
    def build_vocab(self, texts, min_count: int = 1):
        cnt = Counter(c for t in texts for c in t)
        chars = [c for c, n in cnt.most_common() if n >= min_count]
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id, self.unk_id = 0, 1
        return self

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        max_len = max_len or C.DL_MAX_SEQ_LEN
        return [self.stoi.get(c, self.unk_id) for c in text[:max_len]]

    def _build(self):
        self.emb = Embedding(len(self.itos), self.embed_dim, self.rng)
        self.fwd = LSTM(self.embed_dim, self.hidden_dim, self.rng)
        self.bwd = LSTM(self.embed_dim, self.hidden_dim, self.rng)
        self.head = Linear(2 * self.hidden_dim, N_TAGS, self.rng)
        self.opt_state: dict = {}
        self.step = 0
        self.built = True

    # ---------------------------------------------------------- 前向/反向
    def forward(self, idx: np.ndarray) -> np.ndarray:
        e = self.emb.forward(idx)
        hf = self.fwd.forward(e)
        hb = self.bwd.forward(e[:, ::-1, :])[:, ::-1, :]
        return self.head.forward(np.concatenate([hf, hb], axis=2))

    def backward(self, dlogits: np.ndarray):
        dh = self.head.backward(dlogits)
        dhf, dhb = dh[:, :, :self.hidden_dim], dh[:, :, self.hidden_dim:]
        de_f = self.fwd.backward(dhf)
        de_b = self.bwd.backward(dhb[:, ::-1, :])[:, ::-1, :]
        self.emb_grad = self.emb.backward(de_f + de_b)

    # ---------------------------------------------------------- 训练
    def fit(self, texts, gold_tags, log=print, epochs: int | None = None) -> dict:
        epochs = epochs or C.DL_EPOCHS
        if not self.built:
            self._build()
        data = [(self.encode(t), g[:C.DL_MAX_SEQ_LEN])
                for t, g in zip(texts, gold_tags) if t]
        n_val = max(1, int(len(data) * (1 - C.DL_TRAIN_RATIO)))
        val, train = data[:n_val], data[n_val:]
        hist = []
        import time
        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.rng.shuffle(train)
            tot, nb = 0.0, 0
            for bi in range(0, len(train), C.DL_BATCH_SIZE):
                batch = train[bi:bi + C.DL_BATCH_SIZE]
                idx, tag, mask = self._pad(batch)
                logits = self.forward(idx)
                ls = _log_softmax(logits)
                loss = -(ls[np.arange(len(batch))[:, None],
                            np.arange(idx.shape[1])[None, :], tag] * mask).sum() / mask.sum()
                tot += float(loss)
                nb += 1
                dlogits = np.exp(ls)
                dlogits[np.arange(len(batch))[:, None],
                        np.arange(idx.shape[1])[None, :], tag] -= 1
                self.backward(dlogits * mask[:, :, None] / mask.sum())
                self._adam_step()
            acc = self.tag_accuracy(val) if val else 0.0
            hist.append({"epoch": ep, "train_loss": round(tot / max(nb, 1), 4),
                         "val_tag_acc": round(acc, 4)})
            log(f"    epoch {ep}/{epochs}  loss={tot / max(nb, 1):.4f}  "
                f"val_tag_acc={acc:.4f}")
        self.history = hist
        self.train_seconds = round(time.time() - t0, 2)
        self.param_count = (self.emb.W.size + self.fwd.Wx.size + self.fwd.Wh.size
                            + self.fwd.b.size + self.bwd.Wx.size + self.bwd.Wh.size
                            + self.bwd.b.size + self.head.W.size + self.head.b.size)
        return {"history": hist, "train_size": len(train), "val_size": len(val),
                "train_seconds": self.train_seconds, "device": "cpu(numpy)"}

    def _pad(self, batch):
        L = max(len(x) for x, _ in batch)
        idx = np.zeros((len(batch), L), dtype=np.int64)
        tag = np.zeros((len(batch), L), dtype=np.int64)
        mask = np.zeros((len(batch), L), dtype=np.float64)
        for r, (x, g) in enumerate(batch):
            idx[r, :len(x)] = x
            tag[r, :len(g)] = g
            mask[r, :len(x)] = 1.0
        return idx, tag, mask

    def _adam_step(self, lr=None, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        lr = lr or C.DL_LR
        self.step += 1
        grads = {"emb.W": self.emb_grad, "fwd.Wx": self.fwd.grads["Wx"],
                 "fwd.Wh": self.fwd.grads["Wh"], "fwd.b": self.fwd.grads["b"],
                 "bwd.Wx": self.bwd.grads["Wx"], "bwd.Wh": self.bwd.grads["Wh"],
                 "bwd.b": self.bwd.grads["b"], "head.W": self.head.grads["W"],
                 "head.b": self.head.grads["b"]}
        total = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()) + 1e-12)
        scale = min(1.0, clip / (total + 1e-12))
        obj_map = {"emb": self.emb, "fwd": self.fwd, "bwd": self.bwd, "head": self.head}
        for name, g in grads.items():
            obj, attr = name.split(".")
            layer = obj_map[obj]
            g = g * scale
            m = self.opt_state.setdefault(name + ".m", np.zeros_like(g))
            v = self.opt_state.setdefault(name + ".v", np.zeros_like(g))
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            self.opt_state[name + ".m"], self.opt_state[name + ".v"] = m, v
            mh = m / (1 - beta1 ** self.step)
            vh = v / (1 - beta2 ** self.step)
            setattr(layer, attr, getattr(layer, attr) - lr * mh / (np.sqrt(vh) + eps))

    def tag_accuracy(self, data) -> float:
        if not data:
            return 0.0
        idx, tag, mask = self._pad(data)
        pred = self.forward(idx).argmax(-1)
        return float(((pred == tag) * mask).sum() / max(mask.sum(), 1))

    # ---------------------------------------------------------- 解码
    def viterbi(self, logits: np.ndarray, s_penalty: float | None = None) -> list[int]:
        pen = self.tag_penalty if s_penalty is None else s_penalty
        adj = logits.copy()
        adj[:, 3] -= pen
        T = adj.shape[0]
        dp = np.full((T, N_TAGS), _NEG)
        bk = np.zeros((T, N_TAGS), dtype=np.int64)
        dp[0] = np.where(START_ALLOWED, adj[0], _NEG)
        for t in range(1, T):
            for s in range(N_TAGS):
                cand = dp[t - 1] + np.where(TRANS_MASK[:, s], 0.0, _NEG)
                p = int(np.argmax(cand))
                dp[t, s] = cand[p] + adj[t, s]
                bk[t, s] = p
        cand = dp[T - 1] + np.where(END_ALLOWED, 0.0, _NEG)
        cur = int(np.argmax(cand))
        tags = [0] * T
        for t in range(T - 1, -1, -1):
            tags[t] = cur
            cur = int(bk[t, cur])
        return tags

    def cut_with(self, text: str, s_penalty: float | None = None) -> list[str]:
        from seg_dl import tags_to_tokens
        text = text[:C.DL_MAX_SEQ_LEN]
        if not text:
            return []
        logits = self.forward(np.array([self.encode(text)]))[0]
        return tags_to_tokens(text, self.viterbi(logits, s_penalty))

    def cut(self, text: str) -> list[str]:
        return self.cut_with(text, self.tag_penalty)

    def cut_batch(self, texts) -> list[list[str]]:
        return [self.cut(t) for t in texts]

    def cut_batch_with(self, texts, s_penalty: float) -> list[list[str]]:
        return [self.cut_with(t, s_penalty) for t in texts]

    def calibrate(self, texts, refs: dict[str, list[list[str]]],
                  grid: list[float] | None = None, log=print) -> dict:
        import seg_metrics as M
        grid = grid if grid is not None else C.DL_TAG_PENALTY_GRID
        target, tol = C.STAT_CALIB_TARGET_LEN, C.STAT_CALIB_TOL
        scores = []
        for b in grid:
            hyps = self.cut_batch_with(texts, b)
            avg_len = sum(len(w) for h in hyps for w in h) / max(sum(len(h) for h in hyps), 1)
            f1s = [M.evaluate(texts, ref, hyps)["boundary"]["f1"] for ref in refs.values()]
            scores.append({"s_penalty": b, "boundary_f1_vs_refs": round(sum(f1s) / len(f1s), 4),
                           "avg_token_len": round(avg_len, 3),
                           "in_reasonable_range": bool(abs(avg_len - target) <= tol)})
        ok = [d for d in scores if d["in_reasonable_range"]] or scores
        best = max(ok, key=lambda d: (d["boundary_f1_vs_refs"], d["avg_token_len"]))
        self.tag_penalty = best["s_penalty"]
        self.calibration = {"best": best, "grid": scores, "target_avg_len": target}
        log(f"    深度学习 S 惩罚校准：penalty={best['s_penalty']}，"
            f"与参照边界 F1={best['boundary_f1_vs_refs']}，平均词长={best['avg_token_len']}")
        return self.calibration

    # ---------------------------------------------------------- 持久化
    def save(self, path=None):
        path = str(path or C.DL_MODEL_FILE)
        np.savez(path, emb=self.emb.W, fwd_Wx=self.fwd.Wx, fwd_Wh=self.fwd.Wh, fwd_b=self.fwd.b,
                 bwd_Wx=self.bwd.Wx, bwd_Wh=self.bwd.Wh, bwd_b=self.bwd.b,
                 head_W=self.head.W, head_b=self.head.b,
                 itos=np.array(self.itos, dtype=object),
                 tag_penalty=self.tag_penalty, backend="numpy")
        return path

    @classmethod
    def load(cls, path=None, **kw):
        path = str(path or C.DL_MODEL_FILE)
        z = np.load(path, allow_pickle=True)
        m = cls(**kw)
        m.itos = list(z["itos"])
        m.stoi = {c: i for i, c in enumerate(m.itos)}
        m.pad_id, m.unk_id = 0, 1
        m.embed_dim = z["emb"].shape[1]
        m.hidden_dim = z["fwd_Wx"].shape[1] // 4
        m.tag_penalty = float(z["tag_penalty"]) if "tag_penalty" in z else 0.0
        m._build()
        m.emb.W = z["emb"]
        m.fwd.Wx, m.fwd.Wh, m.fwd.b = z["fwd_Wx"], z["fwd_Wh"], z["fwd_b"]
        m.bwd.Wx, m.bwd.Wh, m.bwd.b = z["bwd_Wx"], z["bwd_Wh"], z["bwd_b"]
        m.head.W, m.head.b = z["head_W"], z["head_b"]
        return m

    def stats(self) -> dict:
        return {"backend": "numpy", "device": "cpu", "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim, "vocab_size": len(self.itos),
                "params": int(getattr(self, "param_count", 0)),
                "tag_penalty": self.tag_penalty,
                "train_seconds": getattr(self, "train_seconds", None),
                "calibration": self.calibration.get("best", {}),
                "history": getattr(self, "history", [])}
