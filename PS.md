# Kaggle Competition

# Hull Tactical - Market Prediction: Can you predict market predictability?


## Overview
Your task is to predict the stock market returns as represented by the excess returns of the S&P 500 while also managing volatility constraints. Your work will test the Efficient Market Hypothesis and challenge common tenets of personal finance.

## Description
Wisdom from most personal finance experts would suggest that it's irresponsible to try and time the market. The Efficient Market Hypothesis (EMH) would agree: everything knowable is already priced in, so don’t bother trying.

But in the age of machine learning, is it irresponsible to not try and time the market? Is the EMH an extreme oversimplification at best and possibly just…false?

This competition is about more than predictive modeling. Predicting market returns challenges the assumptions of market efficiency. Your work could help reshape how investors and academics understand financial markets. Participants could uncover signals others overlook, develop innovative strategies, and contribute to a deeper understanding of market behavior—potentially rewriting a fundamental principle of modern finance. Most investors don’t beat the S&P 500. That failure has been used for decades to prop up EMH: If even the professionals can’t win, it must be impossible. This observation has long been cited as evidence for the Efficient Market Hypothesis the idea that prices already reflect all available information and no persistent edge is possible. This story is tidy, but reality is less so. Markets are noisy, messy, and full of behavioral quirks that don’t vanish just because academic orthodoxy said they should.

Data science has changed the game. With enough features, machine learning, and creativity, it’s possible to uncover repeatable edges that theory says shouldn’t exist. The real challenge isn’t whether they exist—it’s whether you can find them and combine them in a way that is robust enough to overcome frictions and implementation issues.

Our current approach blends a handful of quantitative models to adjust market exposure at the close of each trading day. It points in the right direction, but with a blurry compass. Our model is clearly a sub-optimal way to model a complex, non-linear, adaptive system. This competition asks you to do better: to build a model that predicts excess returns and includes a betting strategy designed to outperform the S&P 500 while staying within a 120% volatility constraint. We’ll provide daily data that combines public market information with our proprietary dataset, giving you the raw material to uncover patterns most miss.

Unlike many Kaggle challenges, this isn’t just a theoretical exercise. The models you build here could be valuable in live investment strategies. And if you succeed, you’ll be doing more than improving a prediction engine—you’ll be helping to demonstrate that financial markets are not fully efficient, challenging one of the cornerstones of modern finance, and paving the way for better, more accessible tools for investors.

## Evaluation
The competition's metric is a variant of the Sharpe ratio that penalizes strategies that take on significantly more volatility than the underlying market or fail to outperform the market's return. The metric code is available here:

```py
## Hull Competition Metric Score:

import numpy as np
import pandas as pd
import pandas.api.types

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).

    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    Returns:
        float: The calculated adjusted Sharpe ratio.
    """

    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution = solution
    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)
```

## Submission File
You must submit to this competition using the provided evaluation API, which ensures that models do not peek forward in time. For each trading day, you must predict an optimal allocation of funds to holding the S&P500. As some leverage is allowed, the valid range covers 0 to 2. See this example notebook for more details

```py
"""
The evaluation API requires that you set up a server which will respond to inference requests.
We have already defined the server; you just need write the predict function.
When we evaluate your submission on the hidden test set the client defined in `default_gateway` will run in a different container
with direct access to the hidden test set and hand off the data timestep by timestep.

Your code will always have access to the published copies of the copmetition files.
"""

import os

import pandas as pd
import polars as pl

import kaggle_evaluation.default_inference_server


def predict(test: pl.DataFrame) -> float:
    """Replace this function with your inference code.
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    Each batch of predictions (except the very first) must be returned within 5 minutes of the batch features being provided.
    """
    return 0.0


# When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 1 minute response deadline.
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
```
## Dataset Description
This competition challenges you to predict the daily returns of the S&P 500 index using a tailored set of market data features.

## Competition Phases and Data Updates
The competition will proceed in two phases:

- A model training phase with a test set of six months of historical data. Because these prices are publicly available leaderboard scores during this phase are not meaningful.
- A forecasting phase with a test set to be collected after submissions close. You should expect the scored portion of the test set to be about the same size as the scored portion of the test set in the first phase.
During the forecasting phase the evaluation API will serve test data from the beginning of the public set to the end of the private set. This includes trading days before the submission deadline, which will not be scored. The first `date_id` served by the API will remain constant throughout the competition.

## Files
**train.csv** Historic market data. The coverage stretches back decades; expect to see extensive missing values early on.

- `date_id` - An identifier for a single trading day.
- `M*` - Market Dynamics/Technical features.
- `E*` - Macro Economic features.
- `I*` - Interest Rate features.
- `P*` - Price/Valuation features.
- `V*` - Volatility features.
- `S*` - Sentiment features.
- `MOM*` - Momentum features.
- `D*` - Dummy/Binary features.
- `forward_returns` - The returns from buying the S&P 500 and selling it a day later. Train set only.
- `risk_free_rate` - The federal funds rate. Train set only.
- `market_forward_excess_returns` - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4. Train set only.

**test.csv** A mock test set representing the structure of the unseen test set. The test set used for the public leaderboard set is a copy of the last 180 date IDs in the train set. As a result, the public leaderboard scores are not meaningful. The unseen copy of this file served by the evaluation API may be updated during the model training phase.

- `date_id`
- `[feature_name]` - The feature columns are the same as in train.csv.
- `is_scored` - Whether this row is included in the evaluation metric calculation. During the model training phase this will be true for the first 180 rows only. Test set only.
- `lagged_forward_returns` - The returns from buying the S&P 500 and selling it a day later, provided with a lag of one day.
- `lagged_risk_free_rate` - The federal funds rate, provided with a lag of one day.
- `lagged_market_forward_excess_returns` - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4, provided with a lag of one day.

**kaggle_evaluation/** Files used by the evaluation API. See the demo submission for an illustration of how to use the API.

Once the competition ends, we will periodically publish our data on our website, and you're welcome to use it for your own trading

