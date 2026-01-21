import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class SMABacktester:
    """
    A simple SMA crossover backtesting framework.
    """

    def __init__(self, symbol, sma_short, sma_long, start, end):
        self.symbol = symbol
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.start = start
        self.end = end
        self.results = None
        self.data = None

        self.get_data()

    def get_data(self):
        """Downloads historical price data and computes indicators."""
        df = yf.download(self.symbol, start=self.start, end=self.end)

        if df.empty:
            raise ValueError("No data downloaded. Check symbol or internet connection.")

        data = df[["Close"]].copy()
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["SMA_S"] = data["Close"].rolling(self.sma_short).mean()
        data["SMA_L"] = data["Close"].rolling(self.sma_long).mean()
        data.dropna(inplace=True)

        self.data = data
        return data

    def run_backtest(self):
        """Runs the SMA crossover backtest."""
        data = self.data.copy()

        # Long / Short positions
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)

        # Strategy returns (shift to avoid look-ahead bias)
        data["strategy"] = data["position"].shift(1) * data["returns"]

        # Cumulative returns
        data["returns_bh"] = data["returns"].cumsum().apply(np.exp)
        data["returns_strategy"] = data["strategy"].cumsum().apply(np.exp)

        self.results = data

        perf = data["returns_strategy"].iloc[-1]
        bh_perf = data["returns_bh"].iloc[-1]
        outperf = perf - bh_perf

        print(f"Strategy return: {perf:.6f}")
        print(f"Buy & Hold return: {bh_perf:.6f}")
        print(f"Outperformance: {outperf:.6f}")

        return perf, outperf

    def plot_results(self):
        """Plots strategy vs buy-and-hold performance."""
        if self.results is None:
            raise RuntimeError("Run run_backtest() first.")

        title = (
            f"{self.symbol} | "
            f"SMA Short = {self.sma_short}, "
            f"SMA Long = {self.sma_long}"
        )

        self.results[["returns_bh", "returns_strategy"]].plot(
            title=title, figsize=(12, 6)
        )
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.show()


if __name__ == "__main__":
    bt = SMABacktester(
        symbol="AAPL",
        sma_short=20,
        sma_long=50,
        start="2019-01-01",
        end="2024-01-01",
    )

    bt.run_backtest()
    bt.plot_results()
