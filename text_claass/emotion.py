import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# 1. 准备数据
text_data = [
    ("I love this movie, it is fantastic!", 1),
    ("This film is terrible and boring", 0),
    ("It was okay, nothing special", 1),
    ("Worst film ever, a complete waste of time", 0),
    ("Absolutely brilliant and heartwarming", 1),
    ("I hated every minute of it", 0),
    # 新增数据
    ("A masterpiece of storytelling and visuals", 1),
    ("The acting was wooden and the plot was predictable", 0),
    ("Really enjoyed the twists and turns, kept me hooked", 1),
    ("Such a dull and uninspired piece of work", 0),
    ("Charming characters and a delightful script", 1),
    ("I fell asleep halfway through, so boring", 0),
    ("Exceeded all my expectations, simply amazing", 1),
    ("Overhyped and disappointing, don't bother", 0),
    ("Beautiful cinematography and a moving score", 1),
    ("The dialogue was cringey and the pacing was off", 0),
    ("A fun ride from start to finish, highly recommend", 1),
    ("Pretentious and slow, not my cup of tea", 0),
    ("Heartfelt performances that stay with you", 1),
    ("Cheesy and cliché, seen it all before", 0),
    ("Incredible world-building and a satisfying ending", 1),
    ("Too long and self-indulgent, lost my interest", 0),
    ("Laugh-out-loud funny and surprisingly touching", 1),
    ("The special effects were cheap and the story was weak", 0),
    ("A thought-provoking film that lingers in your mind", 1),
    ("Messy plot and unlikeable characters", 0),
    ("Pure joy from beginning to end", 1),
    ("Forgettable and mediocre at best", 0),
    ("A stunning achievement in every aspect", 1),
    ("Regrettable waste of talent and time", 0),
    ("A cinematic gem that deserves all the awards", 1),
    ("The chemistry between the leads is electric and captivating", 1),
    ("A rollercoaster of emotions, I was on the edge of my seat", 1),
    ("Visually stunning and intellectually stimulating", 1),
    ("The director’s vision is bold and brilliantly executed", 1),
    ("An uplifting story that restores your faith in cinema", 1),
    ("Sharp writing and a killer soundtrack make this a must-watch", 1),
    ("Every scene is crafted with care and precision", 1),
    ("I left the theater with a huge smile on my face", 1),
    ("A perfect blend of humor, drama, and suspense", 1),
    ("The performances are so raw and genuine, it feels real", 1),
    ("A fresh take on a classic genre, absolutely refreshing", 1),
    ("Mesmerizing from the opening shot to the final frame", 1),
    ("Witty, clever, and deeply satisfying", 1),
    ("One of those rare films that gets better with each viewing", 1),

    # 负面 (0)
    ("Dreary, lifeless, and a total snoozefest", 0),
    ("The plot holes are big enough to drive a truck through", 0),
    ("The lead actor mumbles through the entire film", 0),
    ("A shallow attempt at depth that fails miserably", 0),
    ("I wanted to walk out but was too bored to move", 0),
    ("The jokes fall flat and the drama is laughable", 0),
    ("A disjointed mess that goes nowhere", 0),
    ("The CGI is terrible and the acting is worse", 0),
    ("Overlong, overblown, and underwhelming", 0),
    ("A soulless cash grab that insults the audience", 0),
    ("The characters are cardboard cutouts with no motivation", 0),
    ("I couldn't care less about anyone on screen", 0),
    ("The script is full of clunky exposition and forced twists", 0),
    ("A disappointing sequel that ruins the original's legacy", 0),
    ("So predictable that I guessed the ending in the first five minutes", 0)
]

def simple_tokenizer(text):
    text = text.lower()
    for char in ['.', ',', '!', '?', "'", '"']:
        text = text.replace(char, '')
    return text.split()

# 统计词频并构建词汇表
counter = Counter()
for text, _ in text_data:
    counter.update(simple_tokenizer(text))

# 创建词汇表字典，添加 <unk> 和 <pad>
vocab = {'<pad>': 0, '<unk>': 1}
for word, _ in counter.most_common():
    vocab[word] = len(vocab)

# 文本转索引的辅助函数
def text_pipeline(text):
    return [vocab.get(word, vocab['<unk>']) for word in simple_tokenizer(text)]

# 3. 定义 LSTM 情感分类模型
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :] 
        output = self.fc(last_hidden)
        return self.sigmoid(output)

    
if __name__ == '__main__':
# 4. 初始化模型、损失函数和优化器
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1

    model = SimpleLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 训练模型
    EPOCHS = 50
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for text, label in text_data:
            text_tensor = torch.tensor([text_pipeline(text)], dtype=torch.long)
            label_tensor = torch.tensor([label], dtype=torch.float)
            
            optimizer.zero_grad()
            predictions = model(text_tensor).squeeze(1)
            loss = criterion(predictions, label_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:02d} | Loss: {epoch_loss/len(text_data):.4f}')
    # 训练结束后，保存模型参数
    SAVE_PATH = 'sentiment_model.pth'
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"模型已成功保存至: {SAVE_PATH}")


    # 6. 模型预测测试
    def predict_sentiment(model, sentence):
        model.eval()
        with torch.no_grad():
            text_tensor = torch.tensor([text_pipeline(sentence)], dtype=torch.long)
            prediction = model(text_tensor).item()
            return "正面" if prediction > 0.5 else "负面"

    print("\n--- 预测结果 ---")
    print(f"'I love PyTorch': {predict_sentiment(model, 'I love PyTorch')}")
    print(f"'This is so good': {predict_sentiment(model, 'This is so good')}")
    print(f"'i love you': {predict_sentiment(model, 'i love you')}")
    print(f"'The seat is soft': {predict_sentiment(model, 'The seat is soft')}")