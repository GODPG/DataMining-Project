import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from model import ConvBiGRUAutoencoder
from data_cleaning import prepare_data_with_features

def train_model(model, X_train, anomaly_log_file, device, lr=1e-3, batch_size=64, num_epochs=50, alpha=1.0, model_path=None):
    """训练模型并保存"""
    # ——— 统计静态权重 ———
    inp_dim = X_train.shape[2]
    cnt = Counter()
    with open(anomaly_log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, dims_str = line.split(':')
            dims = [int(x) for x in dims_str.split(',')]
            cnt.update(dims)
    # 正确计算最大频次
    max_freq = max(cnt.values()) if cnt else 1

    # 构造权重向量，并转为 (1,1,C)
    w_list = [1.0 + alpha * (cnt.get(i, 0) / max_freq) for i in range(inp_dim)]
    weight_tensor = torch.tensor(w_list, dtype=torch.float32, device=device).view(1, 1, inp_dim)

    # 优化器、调度、损失
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = nn.MSELoss(reduction='none')

    loader = torch.utils.data.DataLoader(
        torch.from_numpy(X_train).float(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4  # 启用多线程数据加载
    )

    # ——— 训练循环 ———
    best_loss, wait = float('inf'), 0
    model.train()
    train_losses = []

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon = model(batch)  # (B, T, C)
            mse_elem = loss_fn(recon, batch)  # (B, T, C)
            weighted_mse = mse_elem * weight_tensor  # 广播加权
            loss = weighted_mse.mean()  # 最终 loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}/{num_epochs}, loss={avg_loss:.6f}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            if model_path:
                torch.save(model.state_dict(), model_path)
                print(f"模型保存到 {model_path}")
        else:
            wait += 1
            if wait >= 10:
                print("Early stopping. 训练结束。")
                break

    print("模型训练并保存完毕。")
    model.eval()  # 训练完毕后，将模型设置为评估模式
    return model

def load_or_train(train_file, test_file, anomaly_log_file, seq_length, hidden_size, num_layers, dropout, lr, batch_size, num_epochs, alpha, save_dir, model_path, scaler_path, winsorize_clip=True):
    """加载已存在的模型或训练新模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # 如果已有模型 & scaler，直接加载
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("加载已有模型与 scaler…")
        scaler = joblib.load(scaler_path)
        X_train, _, _ = prepare_data_with_features(
            train_file, seq_length, scaler=scaler, return_labels=False, winsorize_clip=winsorize_clip  #  使用 winsorize_clip=True
        )
        inp_dim = X_train.shape[2]
        model   = ConvBiGRUAutoencoder(inp_dim, hidden_size, num_layers, dropout)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model.to(device).eval(), scaler, device


    # 否则训练新模型
    print("训练新模型…")
    from data_preprocessing import load_data
    # 加载训练数据和测试数据
    X_train, _, _, scaler = load_data(train_file, test_file, seq_length, scaler_path, winsorize_clip=winsorize_clip)

    inp_dim = X_train.shape[2]
    model = ConvBiGRUAutoencoder(inp_dim, hidden_size, num_layers, dropout)
    model = model.to(device)

    # 训练模型
    model = train_model(model, X_train, anomaly_log_file, device, lr, batch_size, num_epochs, alpha, model_path)
    if model_path:
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

    print("新模型训练并保存完毕。")
    model.eval()  # 训练完毕后，将模型设置为评估模式
    return model, scaler, device
