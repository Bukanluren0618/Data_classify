# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @Time:   2025/3/23 10:31
   @Software: PyCharm
   @File Name:  data_process.py
   @Description:
            
-------------------------------------------------
"""
import json
import pandas as pd


def df_remove_outliers(group):
    """
        处理异常值（使用3σ原则）
    :param group:
    :return:
    """
    mean, std = group['total_power'].mean(), group['total_power'].std()
    return group[(group['total_power'] > mean - 3 * std) & (group['total_power'] < mean + 3 * std)]


def deal_data(file_path, is_day=False):
    """
        处理数据
    :return:
    """
    # 加载数据
    df = pd.read_csv(file_path, parse_dates=['time_5min'], header=0)
    print('df shape', df.shape)

    # 处理缺失值（线性插值填充）
    df = df.sort_values(by=['dataid', 'time_5min'], ascending=[True, True])
    df.set_index('time_5min', inplace=True)
    print('df.index', df.index)
    # df = df.groupby('dataid').apply(lambda x: x.interpolate(method='linear'))

    # 处理极端值
    # df = df.groupby('user_id').apply(df_remove_outliers)

    # 分钟改成天
    if is_day:
        day_df = df.groupby('dataid').resample('D').sum()  # 此时为多层索引
        day_df = day_df.rename(columns={'dataid': 'temp_dataid'})  # 重新命名
        day_df = day_df.reset_index(level='dataid')  # 只保留time_5min索引
        del day_df['temp_dataid']  # 删除旧索引
        df = day_df
    return df


def extract_features(df, save_path):
    """
        序列特征抽取
    :return:
    """
    # 初始化特征抽取器
    from time_feature import ParallelPowerFeatureExtractor
    extractor = ParallelPowerFeatureExtractor(
        # device_cols=['total_power', 'ac_power', 'fridge_power'],
        device_cols=['total_power', "air",  "lights"],
        # device_cols=['total_power', "air", "bathroom", "bedroom", "garage", "heater", "kitchen", "kitchen_app", "livingroom", "oven", "lights"],
        user_col='dataid',
        time_col='time_5min',
        rolling_windows=[7, 30],
        n_jobs=2
    )

    # 执行特征抽取
    all_features = extractor.extract_all_features(df=df, save_path=save_path)

    # 拼接user_id字段
    user_list = [user_id for user_id, group_data in df.groupby('dataid')]

    # 查看特征维度
    print(f"特征矩阵形状：{all_features.shape}")
    print("\n前5个样本的特征示例：")
    print(all_features.head().T)
    return all_features, user_list


def main():
    """
        核心函数
    :return:
    """
    # 处理数据
    deal_list = [("austin_final.csv", 'austin_features.csv'), ("california_final.csv", "california_features.csv"),
                 ("newyork_final.csv", "newyork_features.csv")]
    all_features_list = []
    user_id_list = []
    for (file_name_path, result_name_path) in deal_list:
        file_path = f'./data/ori/{file_name_path}'
        feature_path = f'./data/process/{result_name_path}'

        df = deal_data(file_path, is_day=True)
        # 抽取特征
        cur_features, user_list = extract_features(df, save_path=feature_path)
        all_features_list.append(cur_features)
        user_id_list.extend(user_list)
    all_features = pd.concat(all_features_list, axis=0)
    print(f"all_features: {all_features.shape}")
    # 聚类特征 & # 可视化
    from cluster_analyse import cluster_analysis
    # all_features_numpy_array = all_features.to_numpy()
    clusters = cluster_analysis(all_features, n_clusters=4)

    # 保存最后结果
    print(type(clusters))
    cluster_id_result = {}
    for user_id, cluster_id in zip(user_id_list, clusters):
        cluster_id = str(cluster_id)
        if cluster_id not in cluster_id_result:
            cluster_id_result[cluster_id] = [user_id]
        else:
            cluster_id_result[cluster_id].append(user_id)
    with open('./data/result/all_user_all_power_cluster_4.json', 'w') as file:
        json.dump(cluster_id_result, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
    pass
