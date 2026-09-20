"""
聚类算法框架
基于天气灾害预测数据的聚类分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ClusteringFramework:
    """
    聚类算法框架
    支持多种聚类算法和评估指标
    """

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
        """寻找最优K值（肘部法则）"""
        print("\n=== 寻找最优K值 ===")

        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, labels))

        # 绘制肘部法则图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 肘部法则
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('K值')
        axes[0].set_ylabel('惯性（Inertia）')
        axes[0].set_title('肘部法则 - 寻找最优K值')
        axes[0].grid(True)

        # 轮廓系数
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('K值')
        axes[1].set_ylabel('轮廓系数')
        axes[1].set_title('轮廓系数 - 寻找最优K值')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('optimal_k_selection.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 找出最优K值
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"基于轮廓系数的最优K值: {optimal_k}")
        print(f"最大轮廓系数: {max(silhouette_scores):.4f}")

        return optimal_k

    def visualize_clusters(self, method='kmeans', save_path=None):
        """可视化聚类结果"""
        if method not in self.labels:
            print(f"错误: 未找到 {method} 的聚类结果")
            return

        labels = self.labels[method]

        # 使用PCA降维到2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.scaled_features)

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 聚类散点图（PCA）
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                # 噪声点（DBSCAN）
                axes[0, 0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                                 c='black', s=20, alpha=0.5, label='噪声')
            else:
                axes[0, 0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                                 c=colors[i], s=30, alpha=0.7, label=f'簇 {label}')

        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        axes[0, 0].set_title(f'{method.upper()} 聚类结果 (PCA)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. t-SNE可视化
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(self.scaled_features)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                axes[0, 1].scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                                 c='black', s=20, alpha=0.5, label='噪声')
            else:
                axes[0, 1].scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                                 c=colors[i], s=30, alpha=0.7, label=f'簇 {label}')

        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        axes[0, 1].set_title(f'{method.upper()} 聚类结果 (t-SNE)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 簇分布统计
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        axes[1, 0].bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
        axes[1, 0].set_xlabel('簇')
        axes[1, 0].set_ylabel('样本数量')
        axes[1, 0].set_title('各簇样本数量分布')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 在柱子上显示数值
        for i, (idx, count) in enumerate(cluster_counts.items()):
            axes[1, 0].text(i, count + 5, str(count), ha='center', va='bottom')

        # 4. 特征均值对比
        feature_names = ['温度最大值', '温度最小值', '降水量', '最大风速', '平均湿度', '温度范围', '平均温度']
        cluster_means = []
        for label in unique_labels:
            if label != -1:  # 排除噪声
                mask = labels == label
                cluster_means.append(self.features[mask].mean(axis=0))

        if cluster_means:
            cluster_means = np.array(cluster_means)
            x = np.arange(len(feature_names))
            width = 0.8 / len(unique_labels[unique_labels != -1])

            for i, label in enumerate(unique_labels[unique_labels != -1]):
                axes[1, 1].bar(x + i * width - width * (len(unique_labels[unique_labels != -1]) - 1) / 2,
                             cluster_means[i], width, label=f'簇 {label}')

            axes[1, 1].set_xlabel('特征')
            axes[1, 1].set_ylabel('均值')
            axes[1, 1].set_title('各簇特征均值对比')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

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
        """比较不同聚类方法"""
        print("\n=== 聚类方法比较 ===")

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

        # 可视化比较
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['轮廓系数', 'Calinski-Harabasz', 'Davies-Bouldin']
        for i, metric in enumerate(metrics):
            axes[i].bar(results_df['方法'], results_df[metric])
            axes[i].set_title(metric)
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3, axis='y')

            # 在柱子上显示数值
            for j, val in enumerate(results_df[metric]):
                axes[i].text(j, val + (max(results_df[metric]) * 0.01),
                           f'{val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        return results_df


def main():
    """主函数 - 示例用法"""
    print("=" * 60)
    print("天气灾害数据聚类分析框架")
    print("=" * 60)

    # 创建聚类框架
    cluster = ClusteringFramework('weather_data.csv')

    # 加载数据
    cluster.load_data()

    # 寻找最优K值
    optimal_k = cluster.find_optimal_k(max_k=10)

    # 运行不同聚类算法
    print("\n" + "=" * 60)
    print("运行聚类算法")
    print("=" * 60)

    # K-means
    cluster.run_kmeans(n_clusters=optimal_k)

    # 层次聚类
    cluster.run_hierarchical(n_clusters=optimal_k, linkage='ward')

    # DBSCAN
    cluster.run_dbscan(eps=0.5, min_samples=5)

    # 可视化结果
    print("\n" + "=" * 60)
    print("可视化聚类结果")
    print("=" * 60)

    cluster.visualize_clusters('kmeans', 'kmeans_clustering.png')
    cluster.visualize_clusters('hierarchical', 'hierarchical_clustering.png')

    # 分析聚类结果
    cluster.analyze_clusters('kmeans')

    # 比较不同方法
    cluster.compare_methods()

    print("\n" + "=" * 60)
    print("聚类分析完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("- optimal_k_selection.png: K值选择图")
    print("- kmeans_clustering.png: K-means聚类可视化")
    print("- hierarchical_clustering.png: 层次聚类可视化")
    print("- clustering_comparison.png: 聚类方法比较")


if __name__ == '__main__':
    main()
