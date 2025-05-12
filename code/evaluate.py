import os
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from model import ConvBiGRUAutoencoder
from data_preprocessing import load_data  # 导入数据预处理函数

def evaluate_model(model, train_file, test_file, seq_length, scaler, device, winsorize_clip=True):  # 修改 evaluate_model 的参数列表
    """加载模型并评估"""
    # ——— 验证/测试评估 ———
    #save_dir = '../saved_models'
    #scaler_path = os.path.join(save_dir, 'scaler_new.pkl')
    #print(f"scaler_path in evaluate_model: {scaler_path}")
    _, X_all, y_all, _ = load_data(train_file, test_file, seq_length, scaler=scaler, winsorize_clip=winsorize_clip)  # 使用加载的数据预处理函数
    X_val, X_test, y_val, y_test = train_test_split(
        X_all, y_all, test_size=0.5, stratify=y_all, random_state=42
    )

    # 验证集上搜索最佳阈值
    with torch.no_grad():
        Xt_val = torch.from_numpy(X_val).float().to(device)
        recon_v = model(Xt_val)
        mse_v = torch.mean((Xt_val - recon_v) ** 2, dim=(1, 2)).cpu().numpy()

    best_thr, best_f1 = None, -1
    for q in np.linspace(0.5, 0.99, 50):
        thr = np.quantile(mse_v, q)
        yv = (mse_v > thr).astype(int)
        f1 = f1_score(y_val, yv)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"[验证集] best_threshold={best_thr:.6f}, F1={best_f1:.4f}")

    # 最终测试评估
    with torch.no_grad():
        Xt_test = torch.from_numpy(X_test).float().to(device)
        recon_t = model(Xt_test)
        mse_t = torch.mean((Xt_test - recon_t) ** 2, dim=(1, 2)).cpu().numpy()

    y_pred = (mse_t > best_thr).astype(int)
    print(f"[测试集] 阈值={best_thr:.6f}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    return y_test, Xt_test, recon_t, mse_t
