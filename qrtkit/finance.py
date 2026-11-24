"""Optional diagnostics on verified time series; no live trading or data access."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def _prices(frame):
    if frame.shape[1] != 2 or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Pair prices require two columns and a unique increasing time index")
    if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
        raise ValueError("Pair prices must be finite and strictly positive")


@dataclass
class PairModel:
    columns: list
    intercept: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    train_end: object
    cointegration_pvalue: float | None

    def transform(self, prices):
        _prices(prices)
        if list(prices.columns) != self.columns:
            raise ValueError("Pair column order changed")
        logs = np.log(prices)
        spread = logs.iloc[:, 0] - self.intercept - self.hedge_ratio * logs.iloc[:, 1]
        return pd.DataFrame({"spread": spread, "zscore": (spread - self.spread_mean) / self.spread_std})


def fit_pair(train_prices, cointegration=False):
    """Fit log-price hedge and z-score normalization on a training period only."""
    _prices(train_prices)
    if len(train_prices) < 10:
        raise ValueError("Pair fitting needs at least ten training observations")
    logs = np.log(train_prices)
    design = np.column_stack([np.ones(len(logs)), logs.iloc[:, 1]])
    intercept, beta = np.linalg.lstsq(design, logs.iloc[:, 0], rcond=None)[0]
    spread = logs.iloc[:, 0] - intercept - beta * logs.iloc[:, 1]
    std = float(spread.std())
    if std < 1e-12:
        raise ValueError("Pair spread has zero variance")
    pvalue = None
    if cointegration:
        from statsmodels.tsa.stattools import coint
        pvalue = float(coint(logs.iloc[:, 0], logs.iloc[:, 1])[1])
    return PairModel(list(train_prices.columns), float(intercept), float(beta), float(spread.mean()),
                     std, train_prices.index[-1], pvalue)


def backtest_signal(signal, forward_returns, cost_bps=0., periods_per_year=252):
    """signal[t] is already known BEFORE forward_returns[t] is realized.

    For same-bar returns, shift the signal first. Returns and positions are
    unit-notional; transaction costs apply to absolute position changes.
    This is a diagnostic, not a portfolio simulator. No annualization assumed
    unless periods_per_year matches the supplied data frequency.
    """
    if not signal.index.equals(forward_returns.index):
        raise ValueError("Signal/return indices must match")
    if not signal.index.is_unique or not signal.index.is_monotonic_increasing:
        raise ValueError("Backtest requires unique chronological observations")
    if cost_bps < 0 or periods_per_year <= 0:
        raise ValueError("Costs must be nonnegative and annualization positive")
    if not np.isfinite(signal.to_numpy()).all() or not np.isfinite(forward_returns.to_numpy()).all():
        raise ValueError("Backtest inputs must be finite")
    position = signal.clip(-1, 1)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    pnl = position * forward_returns - turnover * cost_bps / 10000
    equity = (1 + pnl).cumprod()
    equity_with_initial = pd.concat([pd.Series([1.]), equity.reset_index(drop=True)], ignore_index=True)
    drawdown = equity_with_initial / equity_with_initial.cummax() - 1
    std = pnl.std()
    return {"pnl": pnl, "equity": equity, "mean_return": float(pnl.mean()),
            "sharpe": float(pnl.mean() / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "max_drawdown": float(drawdown.min()), "turnover": float(turnover.mean())}
