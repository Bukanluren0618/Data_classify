# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @Time:   2025/3/21 17:22
   @Software: PyCharm
   @File Name:  cluster.py
   @Description:
            
-------------------------------------------------
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def find_optimal_clusters(features, max_k=8):
    sse = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k)
        clusters = model.fit_predict(features)

        sse.append(model.inertia_)
        silhouette_scores.append(silhouette_score(features, clusters))

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(range(2, max_k + 1), sse, 'bo-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")

    plt.subplot(122)
    plt.plot(range(2, max_k + 1), silhouette_scores, 'rs--')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()
    plt.show()


def plot_tsne(features, clusters):
    """t-SNE降维可视化"""
    # tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)

    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=clusters,
                    palette="viridis", alpha=0.8, s=60)
    plt.title("t-SNE Projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title='Cluster')


def plot_feature_distribution(features, clusters):
    """特征分布雷达图"""
    from matplotlib.patches import Circle
    # 选择重要特征（示例选取前5个）
    selected_features = features.columns[:5]
    stats = features[selected_features].groupby(clusters).mean().T

    # 标准化到0-1范围
    normalized = (stats - stats.min()) / (stats.max() - stats.min())

    # 设置极坐标
    angles = np.linspace(0, 2 * np.pi, len(selected_features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    fig = plt.gcf()
    ax = fig.add_subplot(132, polar=True)

    # 绘制每个簇的雷达图
    for cluster in normalized.columns:
        values = normalized[cluster].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)

    # 设置极坐标标签
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), selected_features)

    # 添加辅助网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title("Feature Radar Chart")


def plot_cluster_centers(features, centers):
    """聚类中心热力图"""
    plt.figure(figsize=(10, 6))
    center_df = pd.DataFrame(centers, columns=features.columns)

    # 选择最具区分度的特征（按方差排序）
    top_features = center_df.var().sort_values(ascending=False)[:10].index
    sns.heatmap(center_df[top_features].T,
                cmap="YlGnBu",
                annot=True,
                fmt=".1f",
                cbar_kws={'label': 'Z-Score'})
    plt.title("Cluster Centers (Top 10 Discriminative Features)")
    plt.xlabel("Cluster")
    plt.ylabel("Features")
    plt.xticks(rotation=0)


def plot_power_curves(df):
    """各簇的典型用电曲线"""
    plt.figure(figsize=(12, 6))

    # 按用户聚类结果分组
    for cluster in sorted(df['cluster'].unique()):
        # 获取该簇所有用户的日均值
        cluster_users = df[df['cluster'] == cluster]['user_id'].unique()
        cluster_data = df[df['user_id'].isin(cluster_users)]
        daily_mean = cluster_data.groupby('timestamp')['total_power'].mean()

        # 平滑处理
        smoothed = daily_mean.rolling(7, center=True).mean()

        plt.plot(smoothed.index, smoothed,
                 label=f'Cluster {cluster}',
                 linewidth=2.5)

    plt.title("Typical Daily Power Consumption Patterns")
    plt.xlabel("Date")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# 执行可视化
# plot_power_curves(df)


def visualize_clusters(features, clusters, centers):
    """综合可视化函数"""
    plt.figure(figsize=(18, 5))

    # 子图1：降维可视化
    plt.subplot(131)
    plot_tsne(features, clusters)

    # 子图2：特征分布
    plt.subplot(132)
    plot_feature_distribution(features, clusters)

    # 子图3：聚类中心
    plt.subplot(133)
    plot_cluster_centers(features, centers)

    plt.tight_layout()
    plt.show()


def cluster_analysis(features, n_clusters=3):
    """
    执行聚类分析并可视化
    :param features: 标准化后的特征矩阵
    :param n_clusters: 预设的聚类数量
    """
    # 1. 聚类算法选择, 可以调用find_optimal_clusters进行选择最好的n_clusters
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(features)

    # 2. 评估指标计算
    silhouette_avg = silhouette_score(features, clusters)
    print(f"Silhouette Score: {silhouette_avg:.2f}")

    # 3. 可视化展示
    visualize_clusters(features, clusters, model.cluster_centers_)

    return clusters


# 使用示例

if __name__ == '__main__':
    pass
