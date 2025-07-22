# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @Time:   2025/3/23 16:47
   @Software: PyCharm
   @File Name:  t.py
   @Description:
            
-------------------------------------------------
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pywt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks, welch
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series
from scipy.stats import skew, kurtosis


class ParallelPowerFeatureExtractor(object):
    def __init__(self,
                 device_cols=['total_power', 'ac_power', 'fridge_power'],
                 user_col='user_id',
                 time_col='timestamp',
                 rolling_windows=[7, 30],
                 n_jobs=-1):
        """
        并行功耗特征提取器

        参数：
        - device_cols: 设备列列表（必须包含总功耗）
        - user_col: 用户ID列名
        - time_col: 时间列名
        - rolling_windows: 滑动窗口尺寸列表（天）
        - n_jobs: 并行进程数（-1表示使用所有核心）
        """
        self.device_cols = device_cols
        self.user_col = user_col
        self.time_col = time_col
        self.rolling_windows = rolling_windows
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

        # 校验必须包含总功耗列
        if 'total_power' not in device_cols:
            raise ValueError("device_cols must contain 'total_power'")

    def _base_features_single_device(self, task_param):
        """单个设备基础特征提取"""
        device, df = task_param
        grouped = df.groupby(self.user_col)[device]

        # 基础统计量
        stats = grouped.agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('skew', lambda x: skew(x)),  # 计算偏度
            ('kurtosis', lambda x: kurtosis(x)),  # 计算峰度
            # ('q95', ('quantile', 0.95))  # 分位值
        ])
        stats.columns = [f"{device}_{col}" for col in stats.columns]

        # 时域特征
        def extract_features(series):
            # 自相关特征
            acf_values = acf(series, nlags=7)[1:]  # 取滞后1-7天的自相关

            # 滞后特征
            lags = [series.shift(i) for i in [1, 7, 30]]
            lag_corr = [series.corr(lag) for lag in lags if not lag.isnull().all()]

            return pd.Series({
                **{f'acf_lag{i + 1}': v for i, v in enumerate(acf_values)},
                **{f'lag_{[1, 7, 30][i]}_corr': v for i, v in enumerate(lag_corr)}
            })

        temp_features = grouped.apply(extract_features).unstack()
        temp_features.columns = [f"{device}_{col}" for col in temp_features.columns]

        # 频域特征
        def extract_freq(series):
            # FFT特征
            fft = np.fft.fft(series - series.mean())
            amp = np.abs(fft)[1:6]  # 前5个主要频率分量

            # 小波能量特征
            coeffs = pywt.wavedec(series, 'db4', level=3)
            energy = [np.sum(c ** 2) for c in coeffs]

            return pd.Series({
                **{f'fft_amp_{i + 1}': v for i, v in enumerate(amp)},
                f'wavelet_energy_ratio': energy[0] / sum(energy)
            })

        freq_features = grouped.apply(extract_freq).unstack()
        freq_features.columns = [f"{device}_{col}" for col in freq_features.columns]

        # 模式特征
        def extract_patterns(series):
            # STL分解
            stl = STL(series, period=7).fit()
            # 峰谷特征
            peaks, _ = find_peaks(series, prominence=series.std())
            valleys, _ = find_peaks(-series, prominence=series.std())

            return pd.Series({
                'seasonal_strength': 1 - (np.var(stl.resid) / np.var(stl.trend + stl.resid)),
                'peaks_count': len(peaks),
                'peak_interval_std': np.std(np.diff(peaks)) if len(peaks) > 1 else 0,
                'peak_interval_mean': np.mean(np.diff(peaks)) if len(peaks) > 1 else 0,
                'peak_valley_ratio': len(peaks) / (len(valleys) + 1e-6),
            })

        patterns_features = grouped.apply(extract_patterns).unstack()
        patterns_features.columns = [f"{device}_{col}" for col in patterns_features.columns]

        # 事件特征
        def extract_events(series):
            # 异常检测（3σ原则）
            residuals = series - series.rolling(7, min_periods=1).mean()
            anomalies = (np.abs(residuals) > 3 * series.std()).sum()

            # 周末特征
            weekday_mean = series[series.index.weekday < 5].mean()
            weekend_mean = series[series.index.weekday >= 5].mean()

            return pd.Series({
                'anomalies_count': anomalies,
                'weekend_ratio': weekend_mean / (weekday_mean + 1e-6)
            })

        events_features = grouped.apply(extract_events).unstack()
        events_features.columns = [f"{device}_{col}" for col in events_features.columns]

        return pd.concat([stats, temp_features, freq_features, patterns_features, events_features], axis=1)

    def extract_base_features(self, df):
        """并行提取基础特征"""
        tasks = [(device, df) for device in self.device_cols]

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(self._base_features_single_device, tasks))

        return pd.concat(results, axis=1)

    def _rolling_features_single(self, params):
        """单个滚动窗口特征提取"""
        device, window, df = params

        rolling_stats = (
            df.groupby(self.user_col)[device]
            .rolling(window=window, min_periods=1)  # 允许窗口不满7天时计算（min_periods=1）
            .agg(['mean', 'std', 'min', 'max'])  # 一次性计算多个统计量
            .rename(columns={
                'mean': f'{window}d_mean',
                'std': f'{window}d_std',
                'min': f'{window}d_min',
                'max': f'{window}d_max'
            })  # 重命名列名
            .reset_index(level=0, drop=True)  # 丢弃分组索引，保留时间索引
        )
        rolling_stats = rolling_stats.fillna(0)

        return features

    def extract_rolling_features(self, df):
        """并行提取滑动窗口特征"""
        params = [(device, window, df)
                  for device in self.device_cols
                  for window in self.rolling_windows]

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            features_list = list(executor.map(self._rolling_features_single, params))

        return pd.concat(features_list, axis=1)

    def extract_interaction_features(self, df):
        """设备间交互特征"""

        def calc_interaction(group):
            features = {}
            total = group['total_power']

            # 设备占比特征
            for device in set(self.device_cols) - {'total_power'}:
                if device not in group.columns: continue
                dev_power = group[device]
                features[f'{device}_ratio'] = (dev_power / (total + 1e-6)).mean()
                features[f'{device}_corr'] = total.corr(dev_power)

            # 设备间交叉相关性
            devices = [d for d in self.device_cols if d != 'total_power']
            for i in range(len(devices)):
                for j in range(i + 1, len(devices)):
                    d1, d2 = devices[i], devices[j]
                    if d1 in group.columns and d2 in group.columns:
                        corr = group[d1].corr(group[d2])
                        features[f'{d1}_{d2}_corr'] = corr if not np.isnan(corr) else 0

            return pd.Series(features)

        return df.groupby(self.user_col).apply(calc_interaction)

    def extract_all_features(self, df, save_path=None, use_rolling_features=False):
        """综合特征提取入口"""
        # 基础特征
        print("开始抽取基础特征...")  # 总共25个人
        base_features = self.extract_base_features(df)
        base_features = base_features.fillna(0)
        print(f"完成抽取基础特征, base_features shape: {base_features.shape}")
        # 滑动窗口特征
        print("开始抽取滑动窗口特征...")
        rolling_features = None
        if use_rolling_features:
            rolling_features = self.extract_rolling_features(df)

        # 交互特征
        if len(self.device_cols) > 1:
            print("开始抽取交互特征...")
            interaction_features = self.extract_interaction_features(df)
            interaction_features = interaction_features.fillna(0)
            # 合并所有特征
            features_list = [item for item in [base_features, rolling_features, interaction_features] if
                             item is not None]
            all_features = pd.concat(features_list, axis=1).fillna(0)

        else:
            features_list = [item for item in [base_features, rolling_features] if
                             item is not None]
            all_features = pd.concat(features_list, axis=1).fillna(0)

        # 标准化
        scaler = StandardScaler()
        res_df = pd.DataFrame(
            scaler.fit_transform(all_features),
            index=all_features.index,
            columns=all_features.columns
        )
        if save_path:
            res_df.to_csv(save_path)
        return res_df


# ================= 测试用例 =================
if __name__ == "__main__":
    # 生成模拟数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    users = ['user_1', 'user_2', 'user_3']
    data = {
        'user_id': np.repeat(users, len(dates)),
        'timestamp': np.tile(dates, len(users)),
        'total_power': np.abs(np.random.normal(50, 15, len(users) * len(dates))),
        'ac_power': np.abs(np.random.normal(20, 5, len(users) * len(dates))),
        'fridge_power': np.abs(np.random.normal(10, 3, len(users) * len(dates)))
    }
    df = pd.DataFrame(data).sort_values(['user_id', 'timestamp'])

    # 初始化特征提取器
    extractor = ParallelPowerFeatureExtractor(
        device_cols=['total_power', 'ac_power', 'fridge_power'],
        n_jobs=2
    )

    # 执行特征提取
    features = extractor.extract_all_features(df)

    # 验证输出
    print(f"特征矩阵形状: {features.shape}")
    print("\n样例特征:")
    print(features.iloc[:2, :5].T)
    print("\n交互特征样例:")
    print(features[['ac_power_ratio', 'fridge_power_ratio', 'ac_power_fridge_power_corr']].head())
