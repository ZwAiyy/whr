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

# ============================================
# 4. 从零实现分类算法
# ============================================

# 4.1 K近邻算法（KNN）
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

# 4.3 决策树（简化版）
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p0 = np.sum(y == 0) / len(y)
        p1 = np.sum(y == 1) / len(y)
        return 1 - p0**2 - p1**2

    def _split(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_idx, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                gini_total = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                if gini_total < best_gini:
                    best_gini = gini_total
                    best_split = (feature_idx, threshold, X_left, y_left, X_right, y_right)

        return best_split

    def _build_tree(self, X, y, depth):
        # 停止条件
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            return {'type': 'leaf', 'value': np.bincount(y).argmax()}

        # 寻找最佳分割
        split = self._best_split(X, y)
        if split is None:
            return {'type': 'leaf', 'value': np.bincount(y).argmax()}

        feature_idx, threshold, X_left, y_left, X_right, y_right = split

        # 递归构建子树
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return {
            'type': 'node',
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, x, node):
        if node['type'] == 'leaf':
            return node['value']

        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# ============================================
# 5. 模型训练和评估
# ============================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 初始化模型
knn = KNNClassifier(k=5)
lr = LogisticRegression(learning_rate=0.01, n_iterations=1000)
dt = DecisionTree(max_depth=5)

# 训练模型
print("\n训练模型...")
knn.fit(X_train_scaled, y_train)
lr.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)

# 预测
y_pred_knn = knn.predict(X_test_scaled)
y_pred_lr = lr.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test_scaled)

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
results['Decision Tree'] = evaluate_model(y_test, y_pred_dt, '决策树')

# ============================================
# 6. 可视化
# ============================================

# 6.1 模型性能对比
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
models = list(results.keys())

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    values = [results[model][i] for model in models]
    plt.bar(models, values)
    plt.title(metric)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# 6.2 特征重要性（逻辑回归系数）
plt.figure(figsize=(10, 6))
coefficients = lr.weights
feature_names = feature_columns
plt.barh(feature_names, coefficients)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# 6.3 决策边界可视化（选择两个特征）
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model, feature_idx1, feature_idx2, title):
    # 选择两个特征
    X_2d = X[:, [feature_idx1, feature_idx2]]

    # 创建网格
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    # 预测网格点
    grid = np.c_[xx.ravel(), yy.ravel()]
    X_full = np.zeros((grid.shape[0], X.shape[1]))
    X_full[:, feature_idx1] = grid[:, 0]
    X_full[:, feature_idx2] = grid[:, 1]

    if hasattr(model, 'predict'):
        Z = model.predict(X_full)
    else:
        Z = np.array([model.predict_one(x, model.tree) for x in X_full])

    Z = Z.reshape(xx.shape)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']),
                edgecolor='k', s=20)
    plt.xlabel(feature_names[feature_idx1])
    plt.ylabel(feature_names[feature_idx2])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'decision_boundary_{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

# 绘制逻辑回归的决策边界（选择两个最重要的特征）
plot_decision_boundary(X_test_scaled, y_test, lr, 0, 1, 'Logistic Regression Decision Boundary')

# ============================================
# 7. 交叉验证（可选）
# ============================================

from sklearn.model_selection import cross_val_score

print("\n交叉验证（5折）:")
for model_name, model in [('KNN', knn), ('Logistic Regression', lr), ('Decision Tree', dt)]:
    # 注意：交叉验证需要重新训练，这里简化处理
    # 实际使用时需要实现交叉验证函数
    pass

# ============================================
# 8. 保存模型和结果
# ============================================

import pickle

# 保存模型
models = {'knn': knn, 'lr': lr, 'dt': dt, 'scaler': scaler}
with open('weather_disaster_models.pkl', 'wb') as f:
    pickle.dump(models, f)

# 保存结果
results_df = pd.DataFrame(results).T
results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1']
results_df.to_csv('model_results.csv')

print("\n模型已保存到 weather_disaster_models.pkl")
print("结果已保存到 model_results.csv")