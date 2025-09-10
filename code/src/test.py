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
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        ret.append((train_seq, train_label))
    return ret

def process_data_for_predict(npdf, stp=32):
    ret = []
    for i in range(npdf.shape[0] - stp):
        seq = npdf[i: i + stp]
        seq = torch.FloatTensor(seq)
        ret.append(seq)
    return ret

# ========== 预测流程 ==========
def predict():
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = pd.read_csv("data/feature_test_up.csv")
    data = data.sort_values(["StockCode", "Date"]).reset_index(drop=True)
    data = data.groupby("StockCode", group_keys=False).apply(add_derived_features)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.ffill().bfill()

    drop_cols = ["StockCode", "Date", "year", "month", "day", "DateIndex"]
    feature_cols = [col for col in data.columns if col not in drop_cols and data[col].dtype in [np.float64, np.int64]]

    scaler = joblib.load("model/scaler_up.save")
    data[feature_cols] = scaler.transform(data[feature_cols])

    stockcodes = data["StockCode"].drop_duplicates().tolist()
    stp = 32

    model = BiLSTMWithAttention(len(feature_cols), 256, 2, 1, 0.2).to(device)
    model.load_state_dict(torch.load("model/model_Close_BiLSTM_Attn_up.pth", map_location=device))
    model.eval()

    all_preds = []
    for stockcode in stockcodes:
        stock_data = data[data["StockCode"] == stockcode].sort_values("Date").reset_index(drop=True)
        feature_data = stock_data[feature_cols].values
        if feature_data.shape[0] < stp + 1:
            continue
        sequences = process_data_for_predict(feature_data, stp)
        sequences_tensor = torch.stack(sequences).to(device)

        with torch.no_grad():
            preds = model(sequences_tensor).squeeze(-1).cpu().numpy()
        pred = preds[-1]
        all_preds.append((stockcode, pred))

    max_date = data["Date"].max()
    pricechangerate = []
    for stockcode, pred in all_preds:
        preClose = data[(data["StockCode"] == stockcode) & (data["Date"] == max_date)]["Close"].values
        if len(preClose) == 0:
            continue
        preClose = preClose[0]
        change_pct = (pred - preClose) / preClose * 100
        pricechangerate.append((stockcode, change_pct))

    pricechangerate = sorted(pricechangerate, key=lambda x: x[1], reverse=True)
    top_max = pricechangerate[:10]
    top_min = pricechangerate[-10:]
    # top_min = top_min[::-1]
    result_df = pd.DataFrame({
        "涨幅最大股票代码": [str(x[0]).zfill(6) for x in top_max],
        # "预测涨幅(%)": [round(x[1], 2) for x in top_max],
        "涨幅最小股票代码": [str(x[0]).zfill(6) for x in top_min],
        # "预测跌幅(%)": [round(x[1], 2) for x in top_min],
    })

    os.makedirs("output", exist_ok=True)
    result_df.to_csv("output/result.csv", index=False, encoding="utf-8")
    print("✅ 预测完成，结果保存在 output/result.csv")

    all_df = pd.DataFrame(pricechangerate, columns=["StockCode", "PredictedChangePct"])
    all_df["StockCode"] = all_df["StockCode"].astype(str).str.zfill(6)
    all_df.to_csv("output/all_predictions.csv", index=False, encoding="utf-8")
    print("✅ 所有预测保存在 output/all_predictions.csv")


if __name__ == "__main__":
    predict()
