"""
一个最小可运行的 Transformer 实现，使用 PyTorch。

包含：
1. 多头自注意力 (MultiHeadSelfAttention)
2. 前馈网络 (FeedForward)
3. 位置编码 (PositionalEncoding)
4. 编码器块 (TransformerBlock)
5. 一个字符级"序列预测"演示，训练后 loss 应明显下降

运行: python minimal_transformer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 多头自注意力
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 一次性生成 Q、K、V 的投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, T, C = x.shape

        # 投影并拆成多头: (B, n_heads, T, d_k)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力分数: Q·K^T / sqrt(d_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))

        # 加权求和并合并多头
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 2. 前馈网络
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. 位置编码（正弦/余弦）
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，不参与梯度更新
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# 4. 编码器块（残差连接 + LayerNorm）
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 残差连接: x + 子层输出
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 5. 完整模型：词嵌入 + 位置编码 + 若干编码器块 + 输出头
# ---------------------------------------------------------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, block_size=32, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, block_size)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.block_size = block_size

    def forward(self, idx):
        # idx: (batch, seq_len) 的 token 索引
        x = self.token_embedding(idx)
        x = self.pos_encoding(x)
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.head(x)  # (batch, seq_len, vocab_size)


# ---------------------------------------------------------------------------
# 演示：字符级"下一个字符"预测（类似 GPT 的训练目标）
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 用小语料训练
    text = (
        "the quick brown fox jumps over the lazy dog. "
        "a journey of a thousand miles begins with a single step. "
        "practice makes perfect. knowledge is power. "
    ) * 5

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    block_size = 32
    batch_size = 16

    def get_batch():
        # 随机采样 (input, target)，target 是 input 右移一位
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = MiniTransformer(vocab_size, d_model=64, n_heads=4,
                            n_layers=2, d_ff=128, block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"词表大小: {vocab_size}, 参数量: {sum(p.numel() for p in model.parameters())}")

    for step in range(500):
        xb, yb = get_batch()
        logits = model(xb)  # (B, T, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step:4d}  loss = {loss.item():.4f}")

    # 用训练后的模型生成一小段文本
    model.eval()
    prompt = "the"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated = list(encode(prompt))
    for _ in range(60):
        # 只取最后 block_size 个 token 作为输入
        x = context[:, -block_size:]
        logits = model(x)
        probs = F.softmax(logits[0, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

    print("\n生成示例:")
    print(decode(generated))


if __name__ == "__main__":
    main()
