import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('weather_data.csv')

# 查看数据基本信息
print("数据形状:", df.shape)
print("\n列名:")
print(df.columns.tolist())
print("\n数据类型:")
print(df.dtypes)
print("\n缺失值:")
print(df.isnull().sum())

# ============================================
# 2. 特征工程
# ============================================

# 创建目标变量：是否有灾害（至少2个灾害字段同时为True）
disaster_columns = ['frost_day', 'heat_day', 'severe_heat_day', 'dry_day',
                    'strong_wind_day', 'dust_storm_risk', 'rainy_day']
df['has_disaster'] = (df[disaster_columns].sum(axis=1) >= 2).astype(int)

# 选择特征
feature_columns = [
    'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
    'wind_speed_10m_max', 'relative_humidity_2m_mean'
]

# 创建衍生特征
df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
df['temp_avg'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2

# 更新特征列表
feature_columns.extend(['temp_range', 'temp_avg'])

# 准备数据
X = df[feature_columns].values
y = df['has_disaster'].values

print(f"\n特征维度: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"灾害比例: {y.mean():.2%}")

# ============================================
# 3. 数据集划分
# ============================================

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # 计算距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # 获取k个最近邻的索引
            k_indices = np.argsort(distances)[:self.k]
            # 获取对应的标签
            k_labels = self.y_train[k_indices]
            # 投票
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)

# 4.2 逻辑回归（带梯度下降）
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X, threshold=0.5):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return (y_predicted >= threshold).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 初始化模型
knn = KNNClassifier(k=5)
lr = LogisticRegression(learning_rate=0.01, n_iterations=1000)

# 训练模型
print("\n训练模型...")
knn.fit(X_train_scaled, y_train)
lr.fit(X_train_scaled, y_train)

# 预测
y_pred_knn = knn.predict(X_test_scaled)
y_pred_lr = lr.predict(X_test_scaled)

# 评估函数
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{model_name}:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"  混淆矩阵:\n{cm}")

    return accuracy, precision, recall, f1

# 评估所有模型
results = {}
results['KNN'] = evaluate_model(y_test, y_pred_knn, 'K近邻 (KNN)')
results['Logistic Regression'] = evaluate_model(y_test, y_pred_lr, '逻辑回归')


# 保存结果
results_df = pd.DataFrame(results).T
results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1']
results_df.to_csv('model_results.csv')
