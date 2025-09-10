import pandas as pd
import numpy as np

def inputdata(path):
    return pd.read_csv(path, header=0, sep=",", encoding="utf-8")


def outputdata(path, data, is_index=False):
    data.to_csv(path, index=is_index, header=True, sep=",", mode="w", encoding="utf-8")


def transcolname(df, column_mapping):
    return df.rename(columns=column_mapping)


def trans_datetime(df):
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    except Exception as e:
        print("日期格式错误：", e)
        raise

    df.dropna(subset=["Date"], inplace=True)
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    df = df.sort_values(by=["StockCode", "Date"]).reset_index(drop=True)
    unique_dates = df["Date"].sort_values().unique()
    date_mapping = {date: i + 1 for i, date in enumerate(unique_dates)}
    df["Date"] = df["Date"].map(date_mapping)

    return df


def add_technical_indicators(df):
    df = df.sort_values(by=["StockCode", "Date"]).reset_index(drop=True)
    grouped = df.groupby("StockCode")

    df["MA5"] = grouped["Close"].transform(lambda x: x.rolling(window=5).mean())
    df["MA10"] = grouped["Close"].transform(lambda x: x.rolling(window=10).mean())
    df["EMA12"] = grouped["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df["EMA26"] = grouped["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["MACD"] = df["EMA12"] - df["EMA26"]

    delta = grouped["Close"].transform(lambda x: x.diff())
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def processing_feature(x):
    if x == 1:
        data = inputdata("data/train.csv")
    else:
        data = inputdata("data/test.csv")

    column_mapping = {
        "股票代码": "StockCode",
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Turnover",
        "振幅": "Amplitude",
        "涨跌额": "PriceChange",
        "换手率": "TurnoverRate",
        "涨跌幅": "PriceChangePercentage",
    }

    data = transcolname(data, column_mapping)
    data.drop(columns=["PriceChangePercentage"], inplace=True, errors="ignore")
    data = trans_datetime(data)
    data = add_technical_indicators(data)
    return data


if __name__ == "__main__":
    feature_train = processing_feature(1)
    feature_test = processing_feature(2)
    outputdata("data/feature_train_up.csv", feature_train)
    outputdata("data/feature_test_up.csv", feature_test)
