import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import joblib
import os
from torch.utils.data import Dataset

# ========== 模型定义 ==========
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout, bidirectional=True
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out



class CustomSeqDataset(Dataset):
    def __init__(self, data):
        self.data = data  # data 是 (x, y) 的列表

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y
    
# ========== 特征构建 ==========
def add_derived_features(df):
    df["Return_1"] = df["Close"].pct_change()
    df["Volatility_3"] = df["Close"].rolling(window=3).std()
    df["Volatility_5"] = df["Close"].rolling(window=5).std()
    df["MA_3"] = df["Close"].rolling(window=3).mean()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["Volume_Change_1"] = df["Volume"] / df["Volume"].shift(1)
    df["Turnover_Rate_Change_1"] = df["TurnoverRate"] / df["TurnoverRate"].shift(1)
    df["Amplitude_Normalized"] = df["Amplitude"] / df["Close"]
    df["High_Low_Spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Spread"] = (df["Close"] - df["Open"]) / df["Open"]
    return df

def process_data(npdf, stp=32):
    ret = []
    for i in range(npdf.shape[0] - stp):
        train_seq = npdf[i: i + stp]
        train_label = npdf[i + stp]
        train_seq = torch.tensor(train_seq, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.float32).view(-1)
        ret.append((train_seq, train_label))
    return ret

# ========== 训练模型 ==========
def train_model():
    # 设备自动选择
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple M1 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")

    os.makedirs("model", exist_ok=True)

    feature = pd.read_csv("data/feature_train_up.csv")
    feature = feature.sort_values(["StockCode", "Date"]).reset_index(drop=True)
    feature = feature.groupby("StockCode", group_keys=False).apply(add_derived_features)

    feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature = feature.ffill().bfill()

    drop_cols = ["StockCode", "Date", "year", "month", "day", "DateIndex"]
    feature_cols = [col for col in feature.columns if col not in drop_cols and feature[col].dtype in [np.float64, np.int64, np.float32]]

    scaler = MinMaxScaler()
    feature_scaled = feature.copy()
    feature_scaled[feature_cols] = scaler.fit_transform(feature_scaled[feature_cols])
    joblib.dump(scaler, "model/scaler_up.save")

    stockcodes = feature_scaled["StockCode"].drop_duplicates().tolist()

    train_data = []
    for stockcode in tqdm(stockcodes, desc="构建训练样本"):
        stock_data = feature_scaled[feature_scaled["StockCode"] == stockcode]
        stock_data = stock_data[feature_cols].values
        if len(stock_data) < 32:
            continue
        train_data += process_data(stock_data, stp=32)

    target_idx = feature_cols.index("Close")
    train_data = [(x.to(device), y[target_idx].to(device)) for x, y in train_data]

    print("🔥 即将构建 Dataset")
    # 创建 Dataset 和 Dataloader
    train_dataset = CustomSeqDataset(train_data)
    print("✅ Dataset 构建完成")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=(device.type != "cpu")  # GPU/MPS 时启用加速
    )

    model = BiLSTMWithAttention(len(feature_cols), 256, 2, 1, 0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(20):
        model.train()
        tot_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()
        print(f"📈 Epoch {epoch+1}/20 - Loss: {tot_loss:.6f}")

    torch.save(model.state_dict(), "model/model_Close_BiLSTM_Attn_up.pth")
    print("✅ 模型训练完成，已保存至 model/")

if __name__ == "__main__":
    train_model()
