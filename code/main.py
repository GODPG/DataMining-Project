import os
import torch
from model import ConvBiGRUAutoencoder
from data_preprocessing import load_data, prepare_smd_data  # 导入 prepare_smd_data
from train import load_or_train
from evaluate import evaluate_model
from visualize import plot_original_vs_reconstructed, plot_roc_curve, plot_precision_recall_curve, plot_threshold_vs_f1,plot_mse_vs_timestamp
import joblib

# ——— 超参数 & 路径 ———
seq_length       = 20
hidden_size      = 64
num_layers       = 2
dropout          = 0.2
lr               = 1e-3
batch_size       = 64
num_epochs       = 50
alpha            = 1.0  # 加权放大系数

train_file       = '../Processed_data/machine-1-1/machine-1-1_train.csv'  #请注意修改这里
test_file        = '../Processed_data/machine-1-1/machine-1-1_test.csv'    #请注意修改这里
anomaly_log_file = '../ServerMachineDataset/interpretation_label/machine-1-1.txt'  # 请改为你的 txt 文件路径
save_dir         = '../saved_models'
model_path = os.path.join(save_dir, 'conv_bi_gru_autoencoder.pth')
scaler_path = os.path.join(save_dir, 'scaler_new.pkl')
try:
    scaler = joblib.load(scaler_path)
    print("成功加载 scaler11111111111111111111")
except Exception as e:
    print(f"加载 scaler 时出错：{e}")

# winsorize clip
winsorize_clip = True

# --- 数据集相关路径 ---
dataset = 'SMD'  # 指定数据集
dataset_folder = '../ServerMachineDataset'
output_folder = '../Processed_data'
os.makedirs(output_folder, exist_ok=True)

def check_processed_data(output_folder):
    """检查 processed data 文件夹中是否存在处理后的 CSV 文件"""
    for subdir in os.listdir(output_folder):
        subdir_path = os.path.join(output_folder, subdir)
        if os.path.isdir(subdir_path):
            if not any(file.endswith('.csv') for file in os.listdir(subdir_path)):
                return False  # 至少有一个子文件夹缺少 CSV 文件
    return True  # 所有子文件夹都包含 CSV 文件

if __name__ == '__main__':
    # 检查是否需要准备 SMD 数据
    if not check_processed_data(output_folder):
        print("Processed data 不存在或不完整，准备数据...")
        prepare_smd_data(dataset, dataset_folder, output_folder)
    else:
        print("Processed data 已存在，跳过数据准备...")

    # 加载或训练模型
    model, scaler, device = load_or_train(train_file, test_file, anomaly_log_file, seq_length, hidden_size, num_layers, dropout, lr, batch_size, num_epochs, alpha, save_dir, model_path, scaler_path, winsorize_clip)

    # 评估模型
    # 评估模型
    y_test, Xt_test, recon_t, mse_t = evaluate_model(model, train_file, test_file, seq_length, scaler, device,
                                                     winsorize_clip)

    # 可视化结果
    plot_original_vs_reconstructed(y_test, Xt_test, recon_t)
    plot_roc_curve(y_test, mse_t)
    plot_precision_recall_curve(y_test, mse_t)
    plot_threshold_vs_f1(y_test, mse_t)
    plot_mse_vs_timestamp(mse_t, y_test)
