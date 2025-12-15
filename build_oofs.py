import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from inf import load_model
import os
from fit_models import add_rolling_features, prepare_datasets, winsorize_mad


# =========================
# LOAD WHAT YOU ALREADY HAVE
# =========================
# These must exist from your previous run
# Load models
model_path = os.path.join("artifacts", "models")

fitted_models = {}
for fname in os.listdir(model_path):
    model_name = os.path.splitext(fname)[0]
    if model_name not in fitted_models:
        fitted_models[model_name] = load_model(model_name, model_path)

train = pd.read_csv("data/train.csv").sort_values("date_id").reset_index(drop=True)

y = train["market_forward_excess_returns"].values
fwd = train["forward_returns"].values
rf  = train["risk_free_rate"].values

EXCLUDE_COLS = {
        "date_id",
        "market_forward_excess_returns",
        "forward_returns",
        "risk_free_rate",
        "lagged_forward_returns",
        "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns",
        "is_scored",
    }

# Load prepared feature matrices
train = pd.read_csv('data/train.csv')
train = train.sort_values("date_id").reset_index(drop=True)
train = add_rolling_features(train, price_col='forward_returns')
pre = prepare_datasets(train, None, exclude_cols=EXCLUDE_COLS, early_frac=0.25, rolling_window=252)

X_for_trees  = pre["X_for_trees"]
X_for_linear = pre["X_for_linear"]

tree_models = {"hgb", "rf", "et", "lgb", "xgb", "cat"}

# =========================
# REBUILD OOFs (NO TUNING)
# =========================
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

oof_dict = {}

for name, mdl_fold in fitted_models.items():
    print(f"Rebuilding OOF for {name}")
    X = X_for_trees if name in tree_models else X_for_linear

    oof = np.zeros(len(X))

    for tr_idx, val_idx in tscv.split(X):
        # mdl = clone(pipeline)       
        mdl_fold.fit(X.iloc[tr_idx], y[tr_idx])
        oof[val_idx] = mdl_fold.predict(X.iloc[val_idx])

    # optional but recommended
    oof = winsorize_mad(oof, n_mad=3.0)

    oof_dict[name] = oof

oof_df = pd.DataFrame(oof_dict)

# =========================
# SAVE FOR FINAL STRATEGY
# =========================
oof_df.to_csv("artifacts/oof/oof_df.csv", index=False)
np.save("artifacts/oof/forward_returns.npy", fwd)
np.save("artifacts/oof/risk_free_rate.npy", rf)

print("OOFs rebuilt without Optuna.")
