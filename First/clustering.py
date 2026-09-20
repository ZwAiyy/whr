from WHR.First.clustering_framework import ClusteringFramework
import numpy as np
import pandas as pd

def basic_clustering():
    # 创建聚类框架
    cluster = ClusteringFramework('weather_data.csv')
    cluster.load_data()
    # 寻找最优K值
    optimal_k = cluster.find_optimal_k(max_k=10)
    # 运行K-means聚类
    cluster.run_kmeans(n_clusters=optimal_k)
    # 分析聚类结果
    cluster.analyze_clusters('kmeans')
    return cluster


def compare_methods():
    cluster = ClusteringFramework('weather_data.csv')
    cluster.load_data()

    cluster.run_kmeans(n_clusters=3)
    cluster.run_hierarchical(n_clusters=3, linkage='ward')
    cluster.run_dbscan(eps=0.5, min_samples=5)
    # 比较方法
    results = cluster.compare_methods()

    return cluster, results

def save_results():
    cluster = ClusteringFramework('weather_data.csv')
    cluster.load_data()
    # 运行聚类
    cluster.run_kmeans(n_clusters=3)
    # 保存聚类标签到DataFrame
    cluster.df['cluster_label'] = cluster.labels['kmeans']
    # 保存到CSV
    output_file = 'weather_data_with_clusters.csv'
    cluster.df.to_csv(output_file, index=False)
    return cluster
def main():
    

    cluster1 = basic_clustering()
    cluster2, results2 = compare_methods()
    cluster6 = save_results()
    return {
        'basic_clustering': cluster1,
        'method_comparison': (cluster2, results2),
        'save_results': cluster6
    }


if __name__ == '__main__':
    results = main()
