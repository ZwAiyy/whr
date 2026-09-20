from datasets import load_dataset
from collections import Counter

dataset = load_dataset("lansinuote/ChnSentiCorp")

# 查看数据集结构
print('=' * 50)
print(dataset)
print('=' * 50)
print(dataset["train"][1])
print('=' * 50)
# 划分数据集，固定种子
train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)

train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]
test_dataset = dataset["test"]
train_counts = Counter(train_dataset["label"])
val_counts = Counter(val_dataset["label"])
test_counts = Counter(test_dataset["label"])
print("=== 训练集类别样本数 ===")
for label, count in sorted(train_counts.items()):
    print(f"类别 {label}: {count} 个样本")
print("\n=== 验证集类别样本数 ===")
for label, count in sorted(val_counts.items()):
    print(f"类别 {label}: {count} 个样本")
print("\n=== 测试集类别样本数 ===")
for label, count in sorted(test_counts.items()):
    print(f"类别 {label}: {count} 个样本")