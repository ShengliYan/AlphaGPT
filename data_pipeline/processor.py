import numpy as np


class DataProcessor:
    @staticmethod
    def clean_ohlcv(df):
        if df.empty:
            return df

        key = "symbol" if "symbol" in df.columns else "address"
        df = df.drop_duplicates(subset=["time", key], keep="last")
        df = df.sort_values("time")
        df["close"] = df["close"].ffill()
        df["open"] = df["open"].fillna(df["close"])
        df["high"] = df["high"].fillna(df["close"])
        df["low"] = df["low"].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0)
        if "quote_volume" in df.columns:
            df["quote_volume"] = df["quote_volume"].fillna(0)

        df = df[df["close"] > 1e-15]
        return df

    @staticmethod
    def add_basic_factors(df):
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_ret"].rolling(window=20).std()
        vol_ma = df["volume"].rolling(window=20).mean() + 1e-6
        df["vol_shock"] = df["volume"] / vol_ma
        ma_long = df["close"].rolling(window=60).mean()
        df["trend"] = np.where(df["close"] > ma_long, 1, -1)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df
