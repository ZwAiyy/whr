import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class ClusteringFramework:
    def __init__(self, data_path='weather_data.csv'):
        self.data_path = data_path
        self.df = None
        self.features = None
        self.scaled_features = None
        self.labels = {}
        self.models = {}

    def load_data(self):
        self.df = pd.read_csv(self.data_path)

        # 选择特征（与分类模型相同的特征）
        feature_columns = [
            'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
            'wind_speed_10m_max', 'relative_humidity_2m_mean'
        ]

        # 创建衍生特征
        self.df['temp_range'] = self.df['temperature_2m_max'] - self.df['temperature_2m_min']
        self.df['temp_avg'] = (self.df['temperature_2m_max'] + self.df['temperature_2m_min']) / 2

        # 更新特征列表
        feature_columns.extend(['temp_range', 'temp_avg'])

        self.features = self.df[feature_columns].values

        # 标准化特征
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(self.features)

        print(f"数据形状: {self.df.shape}")
        print(f"特征数量: {len(feature_columns)}")
        print(f"特征名称: {feature_columns}")

        return self

    def run_kmeans(self, n_clusters=3, random_state=42):
        """K-means聚类"""
        print(f"\n=== K-means聚类 (k={n_clusters}) ===")

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(self.scaled_features)

        self.labels['kmeans'] = labels
        self.models['kmeans'] = kmeans

        # 计算评估指标
        silhouette = silhouette_score(self.scaled_features, labels)
        calinski = calinski_harabasz_score(self.scaled_features, labels)
        davies = davies_bouldin_score(self.scaled_features, labels)

        print(f"轮廓系数: {silhouette:.4f}")
        print(f"Calinski-Harabasz指数: {calinski:.4f}")
        print(f"Davies-Bouldin指数: {davies:.4f}")

        return labels

    def run_dbscan(self, eps=0.5, min_samples=5):
        """DBSCAN聚类"""
        print(f"\n=== DBSCAN聚类 (eps={eps}, min_samples={min_samples}) ===")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.scaled_features)

        self.labels['dbscan'] = labels
        self.models['dbscan'] = dbscan

        # 计算评估指标（排除噪声点）
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:  # 至少有2个簇
            # 只计算非噪声点
            mask = labels != -1
            if np.sum(mask) > 1:
                silhouette = silhouette_score(self.scaled_features[mask], labels[mask])
                calinski = calinski_harabasz_score(self.scaled_features[mask], labels[mask])
                davies = davies_bouldin_score(self.scaled_features[mask], labels[mask])

                print(f"轮廓系数: {silhouette:.4f}")
                print(f"Calinski-Harabasz指数: {calinski:.4f}")
                print(f"Davies-Bouldin指数: {davies:.4f}")

        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)
        print(f"簇数量: {n_clusters}")
        print(f"噪声点数量: {n_noise}")

        return labels

    def run_hierarchical(self, n_clusters=3, linkage='ward'):
        """层次聚类"""
        print(f"\n=== 层次聚类 (k={n_clusters}, linkage={linkage}) ===")

        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hierarchical.fit_predict(self.scaled_features)

        self.labels['hierarchical'] = labels
        self.models['hierarchical'] = hierarchical

        # 计算评估指标
        silhouette = silhouette_score(self.scaled_features, labels)
        calinski = calinski_harabasz_score(self.scaled_features, labels)
        davies = davies_bouldin_score(self.scaled_features, labels)

        print(f"轮廓系数: {silhouette:.4f}")
        print(f"Calinski-Harabasz指数: {calinski:.4f}")
        print(f"Davies-Bouldin指数: {davies:.4f}")

        return labels

    def find_optimal_k(self, max_k=10):
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, labels))
        # 找出最优K值
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"基于轮廓系数的最优K值: {optimal_k}")
        print(f"最大轮廓系数: {max(silhouette_scores):.4f}")

        return optimal_k


    def analyze_clusters(self, method='kmeans'):
        """分析聚类结果"""
        if method not in self.labels:
            print(f"错误: 未找到 {method} 的聚类结果")
            return

        labels = self.labels[method]
        unique_labels = np.unique(labels)

        print(f"\n=== {method.upper()} 聚类分析 ===")
        print(f"簇数量: {len(unique_labels[unique_labels != -1])}")

        # 统计每个簇的样本数量
        cluster_stats = pd.DataFrame({
            '簇': unique_labels,
            '样本数量': [np.sum(labels == label) for label in unique_labels],
            '占比': [np.mean(labels == label) * 100 for label in unique_labels]
        })

        print("\n各簇统计:")
        print(cluster_stats.to_string(index=False))

        # 分析每个簇的特征均值
        print("\n各簇特征均值:")
        feature_names = ['温度最大值', '温度最小值', '降水量', '最大风速', '平均湿度', '温度范围', '平均温度']

        for label in unique_labels:
            if label != -1:
                mask = labels == label
                cluster_mean = self.features[mask].mean(axis=0)
                print(f"\n簇 {label}:")
                for i, name in enumerate(feature_names):
                    print(f"  {name}: {cluster_mean[i]:.2f}")

    def compare_methods(self):
        methods = ['kmeans', 'hierarchical']
        if 'dbscan' in self.labels:
            methods.append('dbscan')

        results = []
        for method in methods:
            labels = self.labels[method]
            unique_labels = np.unique(labels)

            # 计算评估指标（排除噪声）
            mask = labels != -1 if method == 'dbscan' else np.ones(len(labels), dtype=bool)

            if np.sum(mask) > 1:
                silhouette = silhouette_score(self.scaled_features[mask], labels[mask])
                calinski = calinski_harabasz_score(self.scaled_features[mask], labels[mask])
                davies = davies_bouldin_score(self.scaled_features[mask], labels[mask])

                results.append({
                    '方法': method.upper(),
                    '轮廓系数': silhouette,
                    'Calinski-Harabasz': calinski,
                    'Davies-Bouldin': davies,
                    '簇数量': len(unique_labels[unique_labels != -1])
                })

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        return results_df


def main():
    cluster = ClusteringFramework('weather_data.csv')# 创建聚类框架
    cluster.load_data()# 加载数据
    optimal_k = cluster.find_optimal_k(max_k=10)# 寻找最优K值
    cluster.run_kmeans(n_clusters=optimal_k) # K-means
    cluster.run_hierarchical(n_clusters=optimal_k, linkage='ward') # 层次聚类
    cluster.run_dbscan(eps=0.5, min_samples=5)# DBSCAN
    cluster.analyze_clusters('kmeans')# 分析聚类结果
    cluster.compare_methods()# 比较不同方法

if __name__ == '__main__':
    main()
