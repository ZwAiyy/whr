# -*- coding: utf-8 -*-
"""分词方法三：基于深度学习的分词（BiLSTM 序列标注，PyTorch 实现）。

任务形式：把分词建模为字符级序列标注（B/M/E/S 四标签）
  网络结构：字嵌入 Embedding → 双向 LSTM → 线性分类头 → Viterbi 约束解码
  训练：Adam + 交叉熵（忽略 padding），梯度裁剪，训练/验证划分
  解码：Viterbi + 合法转移约束（B→M/E，M→M/E，E→B/S，S→B/S）+ 句首/句末约束，
        并对单字词（S）加可校准的惩罚，用于控制切分粒度

监督信号：无监督学习得到的「词表约束」伪标注——
  1) 用统计分词器（校准后的 LLR/PMI 词发现 + 词频模型）对全语料切分；
  2) 取其高频词构成词表 L（出现 ≥2 次、长度 2~4），对每条文本用 L 做最大匹配；
  3) 最大匹配结果转成 B/M/E/S 伪标签，BiLSTM 在此基础上学习上下文泛化，
     从而比纯词表匹配召回更多未登录搭配。

设备：自动使用 CUDA（有 GPU 时）；`torch` 不可用时自动回退到
     `seg_dl_numpy.BiLSTMTagger`（纯 NumPy 手写反向传播版本，接口完全一致）。
"""
from __future__ import annotations

import importlib.util
import math
import time
from collections import Counter

import numpy as np

import config as C

# 标签：0=B 1=M 2=E 3=S
N_TAGS = 4
TAG_NAMES = ["B", "M", "E", "S"]
TAGS_TO_ID = {t: i for i, t in enumerate(TAG_NAMES)}
TRANS_MASK = np.array([
    [0, 1, 1, 0],   # B -> M, E
    [0, 1, 1, 0],   # M -> M, E
    [1, 0, 0, 1],   # E -> B, S
    [1, 0, 0, 1],   # S -> B, S
], dtype=bool)
START_ALLOWED = np.array([1, 0, 0, 1], dtype=bool)   # 句首只能是 B / S
END_ALLOWED = np.array([0, 0, 1, 1], dtype=bool)     # 句尾只能是 E / S

# 是否可用 PyTorch
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import torch.nn as nn

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ 工具
def tags_to_tokens(text: str, tags: list[int]) -> list[str]:
    toks, buf = [], ""
    for ch, tg in zip(text, tags):
        buf += ch
        if TAG_NAMES[tg] in ("E", "S"):
            toks.append(buf)
            buf = ""
    if buf:
        toks.append(buf)
    return toks


def tokens_to_tags(tokens: list[str]) -> list[int]:
    tags = []
    for w in tokens:
        if len(w) == 1:
            tags.append(3)                                  # S
        else:
            tags.extend([0] + [1] * (len(w) - 2) + [2])     # B M... E
    return tags


def max_match(text: str, vocab: set[str], max_len: int = 4) -> list[str]:
    """词典最大匹配（用于构造伪标签）。"""
    toks, i, n = [], 0, len(text)
    while i < n:
        hit = None
        for L in range(min(max_len, n - i), 1, -1):
            if text[i:i + L] in vocab:
                hit = text[i:i + L]
                break
        if hit:
            toks.append(hit)
            i += len(hit)
        else:
            j = i
            while j < n and not any(text[j:j + L] in vocab
                                    for L in range(min(max_len, n - j), 1, -1)):
                j += 1
            toks.extend(list(text[i:j]))
            i = j
    return toks


# ------------------------------------------------------------------ 伪标注
def bootstrap_gold(texts, segmenter, min_freq: int = 2,
                   min_len: int = 2, max_len: int = 4, log=print):
    """用统计分词器的切分结果构造词表约束伪标签，返回 (gold_tags, vocab)。"""
    freq = Counter()
    for t in texts:
        for w in segmenter.cut(t):
            freq[w] += 1
    vocab = {w for w, n in freq.items() if n >= min_freq and min_len <= len(w) <= max_len}
    gold = [tokens_to_tags(max_match(t, vocab, max_len)) for t in texts]
    covered = sum(1 for t, g in zip(texts, gold) if len(g) == len(t))
    avg_len = sum(len(w) for w in vocab) / max(len(vocab), 1)
    log(f"    伪标签词表 {len(vocab)} 词（平均词长 {avg_len:.2f}），"
        f"伪标签覆盖率 {covered}/{len(texts)} 条文本")
    return gold, vocab


def bootstrap_gold_from_tokens(texts, token_lists, min_freq: int = 2,
                               min_len: int = 2, max_len: int = 4, log=print):
    """直接用给定切分结果构造伪标签（先用校准后的统计分词切分的场景）。"""
    freq = Counter(w for toks in token_lists for w in toks)
    vocab = {w for w, n in freq.items() if n >= min_freq and min_len <= len(w) <= max_len}
    gold = [tokens_to_tags(max_match(t, vocab, max_len)) for t in texts]
    n_single = sum(1 for toks in token_lists for w in toks if len(w) == 1)
    total = sum(len(toks) for toks in token_lists) or 1
    avg_len = sum(len(w) for toks in token_lists for w in toks) / total
    log(f"    伪标签词表 {len(vocab)} 词（平均词长 {avg_len:.2f}，"
        f"单字词占比 {n_single / total:.1%}）")
    return gold, vocab


# ================================================================== PyTorch 实现
if HAS_TORCH:

    class _BiLSTMTaggerNet(nn.Module):
        """字级 BiLSTM 序列标注网络。"""

        def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                     dropout: float = 0.1):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                                bidirectional=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(2 * hidden_dim, N_TAGS)

        def forward(self, idx, lengths=None):
            x = self.drop(self.emb(idx))
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                out, _ = self.lstm(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(
                    out, batch_first=True, total_length=idx.shape[1])
            else:
                out, _ = self.lstm(x)
            return self.head(self.drop(out))


class BiLSTMTagger:
    """BiLSTM 序列标注分词模型（PyTorch + CUDA 可选）。"""

    name = "deep"
    display = "深度学习分词"
    backend = "torch"

    def __init__(self, embed_dim: int | None = None, hidden_dim: int | None = None,
                 seed: int | None = None, tag_penalty: float | None = None,
                 device: str | None = None):
        if not HAS_TORCH:
            raise RuntimeError("未安装 torch，请使用 seg_dl_numpy 或安装 torch")
        self.embed_dim = embed_dim or C.DL_EMBED_DIM
        self.hidden_dim = hidden_dim or C.DL_HIDDEN_DIM
        self.seed = C.DL_SEED if seed is None else seed
        self.tag_penalty = C.DL_TAG_PENALTY if tag_penalty is None else tag_penalty
        self.calibration: dict = {}
        self.device = torch.device(device) if device else DEVICE
        self.model: _BiLSTMTaggerNet | None = None
        self._trans_mask = torch.tensor(TRANS_MASK, dtype=torch.bool, device=self.device)
        self._start = torch.tensor(START_ALLOWED, dtype=torch.bool, device=self.device)
        self._end = torch.tensor(END_ALLOWED, dtype=torch.bool, device=self.device)

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
        torch.manual_seed(self.seed)
        self.model = _BiLSTMTaggerNet(len(self.itos), self.embed_dim,
                                      self.hidden_dim).to(self.device)

    # ---------------------------------------------------------- 批处理
    def _pad(self, batch):
        L = max(len(x) for x, _ in batch)
        idx = np.zeros((len(batch), L), dtype=np.int64)
        tag = np.zeros((len(batch), L), dtype=np.int64)
        mask = np.zeros((len(batch), L), dtype=np.float32)
        for r, (x, g) in enumerate(batch):
            idx[r, :len(x)] = x
            tag[r, :len(g)] = g
            mask[r, :len(x)] = 1.0
        return (torch.from_numpy(idx).to(self.device),
                torch.from_numpy(tag).to(self.device),
                torch.from_numpy(mask).to(self.device))

    # ---------------------------------------------------------- 训练
    def fit(self, texts, gold_tags, log=print, epochs: int | None = None) -> dict:
        epochs = epochs or C.DL_EPOCHS
        if self.model is None:
            self._build()
        data = [(self.encode(t), g[:C.DL_MAX_SEQ_LEN])
                for t, g in zip(texts, gold_tags) if t]
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(data))
        data = [data[i] for i in order]
        n_val = max(1, int(len(data) * (1 - C.DL_TRAIN_RATIO)))
        val, train = data[:n_val], data[n_val:]

        opt = torch.optim.Adam(self.model.parameters(), lr=C.DL_LR)
        lossf = nn.CrossEntropyLoss(reduction="none")
        hist = []
        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.model.train()
            rng.shuffle(train)
            tot, nb = 0.0, 0
            for bi in range(0, len(train), C.DL_BATCH_SIZE):
                batch = train[bi:bi + C.DL_BATCH_SIZE]
                idx, tag, mask = self._pad(batch)
                logits = self.model(idx)
                loss = (lossf(logits.reshape(-1, N_TAGS), tag.reshape(-1))
                        * mask.reshape(-1)).sum() / mask.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach())
                nb += 1
            acc = self.tag_accuracy(val) if val else 0.0
            hist.append({"epoch": ep, "train_loss": round(tot / max(nb, 1), 4),
                         "val_tag_acc": round(acc, 4)})
            log(f"    epoch {ep}/{epochs}  loss={tot / max(nb, 1):.4f}  "
                f"val_tag_acc={acc:.4f}")
        self.history = hist
        self.train_seconds = round(time.time() - t0, 2)
        self.param_count = sum(p.numel() for p in self.model.parameters())
        return {"history": hist, "train_size": len(train), "val_size": len(val),
                "train_seconds": self.train_seconds, "device": str(self.device)}

    @torch.no_grad()
    def tag_accuracy(self, data) -> float:
        if not data:
            return 0.0
        self.model.eval()
        correct = total = 0.0
        for bi in range(0, len(data), C.DL_BATCH_SIZE):
            batch = data[bi:bi + C.DL_BATCH_SIZE]
            idx, tag, mask = self._pad(batch)
            pred = self.model(idx).argmax(-1)
            correct += float(((pred == tag).float() * mask).sum())
            total += float(mask.sum())
        return correct / max(total, 1.0)

    # ---------------------------------------------------------- 解码
    @torch.no_grad()
    def viterbi(self, logits: np.ndarray, s_penalty: float | None = None) -> list[int]:
        """带转移约束的 Viterbi 解码（NumPy 实现，避免逐步 GPU 调度开销）。"""
        pen = self.tag_penalty if s_penalty is None else s_penalty
        adj = logits.copy()
        adj[:, 3] -= pen
        T = adj.shape[0]
        NEG = -1e18
        dp = np.full((T, N_TAGS), NEG)
        bk = np.zeros((T, N_TAGS), dtype=np.int64)
        dp[0] = np.where(START_ALLOWED, adj[0], NEG)
        for t in range(1, T):
            for s in range(N_TAGS):
                cand = dp[t - 1] + np.where(TRANS_MASK[:, s], 0.0, NEG)
                p = int(np.argmax(cand))
                dp[t, s] = cand[p] + adj[t, s]
                bk[t, s] = p
        cand = dp[T - 1] + np.where(END_ALLOWED, 0.0, NEG)
        cur = int(np.argmax(cand))
        tags = [0] * T
        for t in range(T - 1, -1, -1):
            tags[t] = cur
            cur = int(bk[t, cur])
        return tags

    @torch.no_grad()
    def predict_logits(self, texts: list[str]) -> list[np.ndarray]:
        self.model.eval()
        outs = []
        for bi in range(0, len(texts), 64):
            chunk = texts[bi:bi + 64]
            enc = [self.encode(t) for t in chunk]
            L = max(len(x) for x in enc)
            idx = np.zeros((len(chunk), L), dtype=np.int64)
            for r, x in enumerate(enc):
                idx[r, :len(x)] = x
            logits = self.model(torch.from_numpy(idx).to(self.device)).cpu().numpy()
            outs.extend(logits[r, :len(x)] for r, x in enumerate(enc))
        return outs

    def cut_with(self, text: str, s_penalty: float | None = None) -> list[str]:
        text = text[:C.DL_MAX_SEQ_LEN]
        if not text:
            return []
        logits = self.predict_logits([text])[0]
        return tags_to_tokens(text, self.viterbi(logits, s_penalty))

    def cut(self, text: str) -> list[str]:
        return self.cut_with(text, self.tag_penalty)

    def cut_batch(self, texts) -> list[list[str]]:
        out = []
        for logits, t in zip(self.predict_logits(list(texts)), texts):
            t = t[:C.DL_MAX_SEQ_LEN]
            out.append(tags_to_tokens(t, self.viterbi(logits, self.tag_penalty)))
        return out

    def cut_batch_with(self, texts, s_penalty: float) -> list[list[str]]:
        out = []
        for logits, t in zip(self.predict_logits(list(texts)), texts):
            t = t[:C.DL_MAX_SEQ_LEN]
            out.append(tags_to_tokens(t, self.viterbi(logits, s_penalty)))
        return out

    # ------------------------------------------------- S 标签惩罚校准
    def calibrate(self, texts, refs: dict[str, list[list[str]]],
                  grid: list[float] | None = None, log=print) -> dict:
        """在候选惩罚网格上以「与参照切分的边界 F1 均值」为准则选最优 S 惩罚。"""
        import seg_metrics as M
        grid = grid if grid is not None else C.DL_TAG_PENALTY_GRID
        target, tol = C.STAT_CALIB_TARGET_LEN, C.STAT_CALIB_TOL
        logits = self.predict_logits(list(texts))
        scores = []
        for b in grid:
            hyps = [tags_to_tokens(t, self.viterbi(lg, b))
                    for lg, t in zip(logits, texts)]
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
        path = str(path or C.DL_MODEL_FILE_TORCH)
        torch.save({"state_dict": self.model.state_dict(),
                    "itos": self.itos,
                    "embed_dim": self.embed_dim, "hidden_dim": self.hidden_dim,
                    "tag_penalty": self.tag_penalty,
                    "backend": "torch", "seed": self.seed}, path)
        return path

    @classmethod
    def load(cls, path=None, **kw):
        path = str(path or C.DL_MODEL_FILE_TORCH)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"],
                tag_penalty=ckpt.get("tag_penalty", C.DL_TAG_PENALTY), **kw)
        m.itos = list(ckpt["itos"])
        m.stoi = {c: i for i, c in enumerate(m.itos)}
        m.pad_id, m.unk_id = 0, 1
        m._build()
        m.model.load_state_dict(ckpt["state_dict"])
        m.model.eval()
        return m

    def stats(self) -> dict:
        n_param = (sum(p.numel() for p in self.model.parameters())
                   if self.model is not None else 0)
        return {"backend": "torch", "device": str(self.device),
                "embed_dim": self.embed_dim, "hidden_dim": self.hidden_dim,
                "vocab_size": len(self.itos), "params": int(n_param),
                "tag_penalty": self.tag_penalty,
                "train_seconds": getattr(self, "train_seconds", None),
                "calibration": self.calibration.get("best", {}),
                "history": getattr(self, "history", [])}


# ================================================================== 后端选择
def _select_backend():
    """优先 PyTorch；不可用时回退到纯 NumPy 实现（接口一致）。"""
    if HAS_TORCH and not C.DL_FORCE_NUMPY:
        return BiLSTMTagger
    from seg_dl_numpy import BiLSTMTagger as NumPyTagger
    return NumPyTagger


TaggerClass = _select_backend()
BACKEND = "torch" if TaggerClass is BiLSTMTagger else "numpy"


def make_tagger(**kw):
    """按当前后端创建分词模型（便于在报告里统一记录后端）。"""
    return TaggerClass(**kw)


def backend_info() -> dict:
    info = {"backend": BACKEND, "torch_installed": HAS_TORCH}
    if BACKEND == "torch":
        info["torch_version"] = torch.__version__
        info["device"] = str(DEVICE)
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["device"] = "cpu(numpy)"
        info["note"] = "未使用 torch：DL_FORCE_NUMPY=1 或环境缺少 torch"
    return info
