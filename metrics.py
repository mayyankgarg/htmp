import numpy as np
import pandas as pd
import pandas.api.types

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


class ParticipantVisibleError(Exception):
    pass


def hull_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "date_id") -> float:
    """
    Exact copy of the competition metric for local use.
    solution: DataFrame with at least ['forward_returns', 'risk_free_rate'] (and optionally date_id)
    submission: DataFrame with column 'prediction' in [0, 2]
    """

    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    sol = solution.copy()
    sol['position'] = submission['prediction'].values

    if sol['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {sol["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if sol['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {sol["position"].min()} below minimum of {MIN_INVESTMENT}')

    # Strategy returns = mix of risk-free and market
    sol['strategy_returns'] = sol['risk_free_rate'] * (1 - sol['position']) + sol['position'] * sol['forward_returns']

    # Strategy Sharpe (excess over risk-free)
    strategy_excess_returns = sol['strategy_returns'] - sol['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = strategy_excess_cumulative ** (1 / len(sol)) - 1
    strategy_std = sol['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Market stats
    market_excess_returns = sol['forward_returns'] - sol['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = market_excess_cumulative ** (1 / len(sol)) - 1
    market_std = sol['forward_returns'].std()
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Vol penalty (for >120% of market vol)
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Return penalty (for underperforming market excess returns)
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap ** 2) / 100

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)
