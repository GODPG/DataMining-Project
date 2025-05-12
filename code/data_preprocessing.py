import os
import pandas as pd
import numpy as np
import joblib
from data_cleaning import prepare_data_with_features  # 读取/特征/标准化/滑窗

# 数据预处理相关函数，例如 prepare_data_with_features
def load_data(train_file, test_file, seq_length, scaler=None, winsorize_clip=True):
    X_train, _, _ = prepare_data_with_features(
        train_file, seq_length, scaler=scaler, return_labels=False, winsorize_clip=winsorize_clip
    )

    X_all, _, y_all = prepare_data_with_features(
        test_file, seq_length, scaler=scaler, return_labels=True, winsorize_clip=winsorize_clip
    )
    return X_train, X_all, y_all, scaler

def load_and_save_smd(category, filename, dataset, dataset_folder, output_folder):
    """加载 SMD 数据集并保存为 CSV 格式"""
    os.makedirs(os.path.join(output_folder, filename.split('.')[0]), exist_ok=True)
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    # print(dataset, category, filename, temp.shape)
    fea_len = len(temp[0, :])
    header_list = []
    for i in range(fea_len):
        header_list.append("col_%d"%i)
    data = pd.DataFrame(temp, columns=header_list).reset_index()
    data.rename(columns={'index': 'timestamp'}, inplace=True)
    if category == "test":
        temp1 = np.genfromtxt(os.path.join(dataset_folder, "test_label", filename),
                         dtype=np.float32,
                         delimiter=',')
        data1 = pd.DataFrame(temp1, columns=["label"]).reset_index()
        data1.rename(columns={'index': 'timestamp'}, inplace=True)
        data = pd.merge(data, data1, how="left", on='timestamp')

    print(dataset, category, filename, temp.shape)
    data.to_csv(os.path.join(output_folder,  filename.split('.')[0], dataset + "_" + category + ".csv"), index=False)

def prepare_smd_data(dataset, dataset_folder, output_folder):
    """准备 SMD 数据集，将其转换为 CSV 格式"""
    if dataset == 'SMD':
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save_smd('train', filename, filename.strip('.txt'), dataset_folder, output_folder)
                load_and_save_smd('test', filename, filename.strip('.txt'), dataset_folder, output_folder)
