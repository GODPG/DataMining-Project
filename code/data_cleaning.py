import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize  # 导入 winsorize

def winsorize_dataframe(df: pd.DataFrame, numeric_cols: list, limits: list = [0.01, 0.01]) -> pd.DataFrame:
    df_copy = df.copy()  # 创建 DataFrame 的副本，以避免修改原始 DataFrame
    for col in numeric_cols:
        df_copy[col] = winsorize(df_copy[col], limits=limits)
    return df_copy


def prepare_unsupervised_time_series_data(
        file_path, seq_length,
        scaler=None, return_labels=False):
    """
    加载数据，剔除标签列，可选复用或新建 StandardScaler，构造滑动窗口序列。

    参数:
        file_path: CSV 文件路径
        seq_length: 滑动窗口长度
        scaler: 如果提供，则使用它进行 transform，否则新建并 fit
        return_labels: 如果 True，返回与每个窗口对应的标签
    返回:
        X: (样本数, seq_length, 特征数) 数组
        scaler: StandardScaler 对象
        labels: (样本数,) 数组（或 None）
    """
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载：{file_path}")
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
        return None, None, None

    has_label = 'label' in df.columns
    if has_label:
        raw_labels = df['label'].values
        df = df.drop(columns=['label'])
    else:
        raw_labels = None

    # 数值列、缺失值填充、去重
    num_cols = df.select_dtypes(include=np.number).columns
    df_num  = df[num_cols].fillna(df[num_cols].mean()).drop_duplicates()

    # 标准化
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_num)
        print("已 fit_transform 标准化。")
    else:
        data_scaled = scaler.transform(df_num)
        print("已 transform 标准化。")

    # 滑动窗口 + 可选标签窗口
    X, Y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        if return_labels and raw_labels is not None:
            # 窗口内任一点为异常则视为异常窗口
            Y.append(int(raw_labels[i:i+seq_length].any()))
    X = np.array(X)
    Y = np.array(Y) if Y else None

    print(f"生成窗口数：{X.shape[0]}, 序列长：{seq_length}, 特征数：{X.shape[2]}")
    return X, scaler, Y

def add_features(df: pd.DataFrame, seq_length: int, numeric_cols: list) -> pd.DataFrame:
    for col in numeric_cols:
        df[f'{col}_diff1'] = df[col].diff().fillna(0)
        df[f'{col}_diff2'] = df[f'{col}_diff1'].diff().fillna(0)

    # 滑动统计特征
    for col in numeric_cols:
        df[f'{col}_roll_mean'] = (
            df[col]
              .rolling(seq_length)
              .mean()
              .fillna(method='bfill')
        )
        df[f'{col}_roll_std'] = (
            df[col]
              .rolling(seq_length)
              .std()
              .fillna(method='bfill')
        )

    # 频域主频特征（只示例第一个 numeric_cols）
    def dominant_freq(x: np.ndarray) -> float:
        fft_vals = np.fft.rfft(x)
        mags     = np.abs(fft_vals)
        freqs    = np.fft.rfftfreq(len(x))
        return float(freqs[np.argmax(mags)])

    df[f'{numeric_cols[0]}_dom_freq'] = (
        df[numeric_cols[0]]
          .rolling(seq_length)
          .apply(dominant_freq, raw=True)
          .fillna(method='bfill')
    )

    return df


def prepare_data_with_features(
        file_path: str,
        seq_length: int,
        scaler: StandardScaler = None,
        return_labels: bool = False,
        winsorize_clip: bool = False  # 添加 winsorize_clip 参数
    ):
    # 1. 读取 CSV
    df = pd.read_csv(file_path)
    raw_labels = None
    if 'label' in df.columns:
        raw_labels = df['label'].values
        df = df.drop(columns=['label'])

    # 2. 保留数值列，填充缺失
    df = df.select_dtypes(include=np.number)
    df = df.fillna(method='ffill').fillna(method='bfill')

    # 3. 特征工程
    numeric_cols = df.columns.tolist()
    df = add_features(df, seq_length, numeric_cols)

    # 4. Winsorize 处理 (在标准化之前)
    if winsorize_clip:
        df = winsorize_dataframe(df, numeric_cols)  # 调用新的 winsorize_dataframe 函数

    # 5. 标准化
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)
    else:
        data_scaled = scaler.transform(df)

    # 6. 滑动窗口切分
    X, Y = [], []
    n = data_scaled.shape[0]
    for i in range(n - seq_length):
        X.append(data_scaled[i:i+seq_length])
        if return_labels and raw_labels is not None:
            Y.append(int(raw_labels[i:i+seq_length].any()))
    X = np.array(X)
    Y = np.array(Y) if return_labels and raw_labels is not None else None

    return X, scaler, Y
