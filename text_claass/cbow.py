import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# ================= 1. 文本预处理与数据准备 =================

# 准备一小段示例语料（实际应用中通常会读取大型文本文件）
raw_text = """
    Natural language processing is a subfield of linguistics and computer science 
    concerned with the interactions between computers and human language.
"""

# 简单的文本清洗：转小写并按空格分词
tokens = raw_text.lower().split()

# 构建词汇表（Word to Index 映射）
# 使用 set 去除重复单词，sorted 保证每次运行顺序一致
vocab = sorted(list(set(tokens)))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)  # 词汇表大小

# 定义上下文窗口大小（例如 window_size=2，表示取目标词前后各2个词）
CONTEXT_SIZE = 2

# 构建训练数据对：(上下文词索引列表, 目标词索引)
# 例如：对于句子 "A B C D E"，如果目标词是 "C"，上下文就是 ["A", "B", "D", "E"]
cbow_data = []
for i in range(CONTEXT_SIZE, len(tokens) - CONTEXT_SIZE):
    # 提取上下文词
    context_words = tokens[i-CONTEXT_SIZE:i] + tokens[i+1:i+1+CONTEXT_SIZE]
    # 提取目标词（中心词）
    target_word = tokens[i]
    
    # 将文本转换为索引张量
    context_idxs = torch.tensor([word_to_idx[w] for w in context_words], dtype=torch.long)
    target_idx = torch.tensor([word_to_idx[target_word]], dtype=torch.long)
    
    cbow_data.append((context_idxs, target_idx))


# ================= 2. 定义 CBOW 模型 =================

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        初始化 CBOW 模型
        :param vocab_size: 词汇表大小
        :param embedding_dim: 词向量的维度（例如 100 维）
        """
        super(CBOWModel, self).__init__()
        
        # Embedding 层：将离散的词索引转换为稠密的连续向量
        # 训练完成后，self.embeddings.weight 就是我们要的词向量
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # 线性层（全连接层）：将词向量映射回词汇表大小的维度，用于计算概率
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, inputs):
        """
        前向传播过程
        :param inputs: 上下文词的索引张量，形状为 (batch_size, context_window * 2)
        :return: 目标词的对数概率分布
        """
        # 1. 查找上下文词的词向量
        # embeds 形状: (batch_size, context_window * 2, embedding_dim)
        embeds = self.embeddings(inputs)
        
        # 2. CBOW 核心操作：将上下文中所有词的向量求平均（或求和）
        # 这里使用 mean 在维度 1 上进行平均，压缩为一个向量
        # avg_embeds 形状: (batch_size, embedding_dim)
        avg_embeds = torch.mean(embeds, dim=1)
        
        # 3. 通过线性层计算得分（Logits）
        # out 形状: (batch_size, vocab_size)
        out = self.linear(avg_embeds)
        
        # 4. 使用 LogSoftmax 计算对数概率，配合 NLLLoss 使用
        log_probs = F.log_softmax(out, dim=1)
        
        return log_probs


# ================= 3. 模型训练 =================

# 超参数设置
EMBEDDING_DIM = 50  # 词向量维度
EPOCHS = 100        # 训练轮数
LEARNING_RATE = 0.001  # 学习率

# 初始化模型、损失函数和优化器
model = CBOWModel(vocab_size, EMBEDDING_DIM)
criterion = nn.NLLLoss()  # 负对数似然损失，常用于分类问题
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam 优化器

print("开始训练 CBOW 模型...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    
    # 遍历每一个训练样本
    for context, target in cbow_data:
        # 1. 梯度清零（PyTorch 默认会累积梯度，所以每次迭代前必须清零）
        optimizer.zero_grad()
        
        # 2. 前向传播：将上下文输入模型，得到预测的对数概率
        log_probs = model(context.unsqueeze(0))  # unsqueeze(0) 增加 batch 维度
        
        # 3. 计算损失：比较预测概率与真实目标词
        loss = criterion(log_probs, target)
        
        # 4. 反向传播：计算梯度
        loss.backward()
        
        # 5. 优化器更新模型参数
        optimizer.step()
        
        total_loss += loss.item()
    
    # 每 20 轮打印一次损失，观察收敛情况
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

print("训练完成！")


# ================= 4. 提取并测试词向量 =================

# 获取训练好的词向量权重
word_vectors = model.embeddings.weight.data.numpy()

# 测试：计算两个词之间的余弦相似度（这里简单打印向量看看）
test_word = "language"
if test_word in word_to_idx:
    print(f"\n'{test_word}' 的词向量 (前10维): {word_vectors[word_to_idx[test_word]][:10]}")

model.eval()

# --- 测试 1：预测单词（给定上下文，预测中心词） ---
print("\n===== 单词预测测试 =====")
# 假设上下文是 "natural" 和 "processing"，我们让模型预测中间缺失的词
test_context_words = ["natural", "processing"] 
test_context_idxs = torch.tensor([word_to_idx[w] for w in test_context_words], dtype=torch.long).unsqueeze(0)

# 前向传播获取预测概率
with torch.no_grad():  # 推理时不需要计算梯度，节省内存
    log_probs = model(test_context_idxs)
    probs = torch.exp(log_probs)  # 将对数概率转回正常概率

# 找到概率最大的词的索引
predicted_idx = torch.argmax(probs, dim=1).item()
predicted_word = idx_to_word[predicted_idx]

print(f"输入上下文: {test_context_words}")
print(f"模型预测的中心词: '{predicted_word}'")


# --- 测试 2：预测句子（句子补全） ---
print("\n===== 句子补全测试 =====")
# 我们给出一个句子的开头，让模型预测接下来的几个词
sentence_prefix = ["natural", "language", "processing", "is"]
generated_sentence = sentence_prefix.copy()

# 连续预测 5 个词
for _ in range(5):
    # 取最后 CONTEXT_SIZE * 2 个词作为上下文（这里简单取最后2个词）
    context_words = generated_sentence[-CONTEXT_SIZE:]
    context_idxs = torch.tensor([word_to_idx.get(w, 0) for w in context_words], dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        log_probs = model(context_idxs)
        probs = torch.exp(log_probs)
    
    # 贪心解码：每次选概率最高的词
    next_idx = torch.argmax(probs, dim=1).item()
    next_word = idx_to_word[next_idx]
    
    generated_sentence.append(next_word)

print(f"输入前缀: {sentence_prefix}")
print(f"模型生成的续写: {' '.join(generated_sentence)}")