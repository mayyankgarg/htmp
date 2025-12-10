import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone

from metrics import hull_score, ParticipantVisibleError
# ==========================
# Config
# ==========================

TARGET_COL = "market_forward_excess_returns"
FORWARD_RET_COL = "forward_returns"
RISK_FREE_COL = "risk_free_rate"

# Columns we never use as features
EXCLUDE_COLS = {
    "date_id",
    TARGET_COL,
    FORWARD_RET_COL,
    RISK_FREE_COL,
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
    "is_scored",
}

# ==========================
# Data utilities
# ==========================

def load_data(train_path, test_path=None):
    """Load and sort train/test by date_id."""
    train = pd.read_csv(train_path)
    train = train.sort_values("date_id").reset_index(drop=True)
    test = None
    if test_path is not None:
        test = pd.read_csv(test_path)
        test = test.sort_values("date_id").reset_index(drop=True)
    return train, test


def get_feature_cols(df: pd.DataFrame):
    """All columns except targets, ids, and lagged columns."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ==========================
# Modeling: base models & stacking
# ==========================

def make_base_models(random_state=42):
    """
    Define a few diverse, relatively small models.
    You can add LightGBM/XGBoost/CatBoost here later.
    """
    models = {}

    # Linear, L2 regularized
    models["ridge"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )

    # Linear, L1+L2 regularized
    models["enet"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state)),
        ]
    )

    # Small gradient boosting tree model
    models["hgb"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=300,
                random_state=random_state,
            )),
        ]
    )

    return models


def generate_oof_predictions(models, X: pd.DataFrame, y: np.ndarray, n_splits=5):
    """
    Time-series OOF predictions for each base model.
    Returns:
      oof_df: DataFrame with one column per model (OOF preds)
      fold_indices: which fold each row belonged to
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_samples = X.shape[0]
    model_names = list(models.keys())
    oof_preds = {name: np.zeros(n_samples, dtype=float) for name in model_names}
    fold_indices = np.full(n_samples, -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]

        for name, model in models.items():
            mdl = clone(model)
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_val)
            oof_preds[name][val_idx] = preds

        fold_indices[val_idx] = fold
        print(f"[OOF] Finished fold {fold + 1}/{n_splits}")

    oof_df = pd.DataFrame({name: oof_preds[name] for name in model_names})
    return oof_df, fold_indices


def fit_meta_model(oof_df: pd.DataFrame, y: np.ndarray, random_state=42):
    """
    Simple linear meta-model on top of base model predictions.
    """
    meta = Ridge(alpha=0.1, random_state=random_state)
    meta.fit(oof_df.values, y)
    meta_oof = meta.predict(oof_df.values)
    return meta, meta_oof


# ==========================
# Strategy / betting engine
# ==========================

def evaluate_strategy(preds, forward_returns, risk_free_rate, dates, params):
    """
    Map model predictions -> positions in [0, 2],
    then compute the true competition score using hull_score().
    Also returns a few stats for debugging.
    """
    preds = np.asarray(preds, dtype=float)
    fwd = np.asarray(forward_returns, dtype=float)
    rf = np.asarray(risk_free_rate, dtype=float)

    # Mask out NaNs
    mask = ~np.isnan(preds) & ~np.isnan(fwd) & ~np.isnan(rf)
    preds = preds[mask]
    fwd = fwd[mask]
    rf = rf[mask]

    if preds.size == 0:
        return -1e9, None

    std = preds.std()
    if std == 0 or np.isnan(std):
        return -1e9, None

    # z-score of predictions
    z = preds / (std + 1e-8)

    # Optionally smooth predictions (EMA)
    alpha_p = params.get("pred_ema_alpha", 0.0)
    if alpha_p > 0:
        z_smoothed = np.zeros_like(z)
        z_smoothed[0] = z[0]
        for i in range(1, len(z)):
            z_smoothed[i] = alpha_p * z[i] + (1 - alpha_p) * z_smoothed[i - 1]
    else:
        z_smoothed = z

    # Non-linear mapping z -> position in [0, 2]
    k = params.get("k", 1.0)
    scale = params.get("scale", 1.0)
    max_leverage = params.get("max_leverage", 2.0)
    min_leverage = params.get("min_leverage", 0.0)

    alloc_raw = 1.0 + k * np.tanh(z_smoothed / scale)
    alloc_raw = np.clip(alloc_raw, min_leverage, max_leverage)

    # Smooth allocations (EMA)
    alpha_a = params.get("alloc_ema_alpha", 0.0)
    if alpha_a > 0:
        alloc = np.zeros_like(alloc_raw)
        alloc[0] = alloc_raw[0]
        for i in range(1, len(alloc_raw)):
            alloc[i] = alpha_a * alloc_raw[i] + (1 - alpha_a) * alloc[i - 1]
    else:
        alloc = alloc_raw

    # Build solution/submission for metric
    solution = pd.DataFrame({
        "forward_returns": fwd,
        "risk_free_rate": rf,
    })
    submission = pd.DataFrame({
        "prediction": alloc,
    })

    try:
        score = hull_score(solution, submission, row_id_column_name="date_id")
    except ParticipantVisibleError:
        score = -1e9

    # Strategy vs market returns
    strat_ret = alloc * fwd
    market_ret = fwd

    mu_s = strat_ret.mean()
    mu_m = market_ret.mean()
    sig_s = strat_ret.std()
    sig_m = market_ret.std()

    if sig_s == 0 or np.isnan(sig_s) or np.isnan(sig_m):
        return -1e9, sig_s, sig_m, np.nan, alloc

    sharpe_like = (mu_s - mu_m) / (sig_s + 1e-8)
    vol_ratio = sig_s / (sig_m + 1e-8)
    excess = mu_s - mu_m

    return score, vol_ratio, sharpe_like, excess, alloc



def optimize_mapping(preds, forward_returns, risk_free_rate, dates=None):
    """
    Grid search over mapping hyperparameters to maximize the true competition metric.
    """
    param_grid = []
    for k in [0.5, 1.0, 1.5]:
        for scale in [0.5, 1.0, 2.0]:
            for alpha_p in [0.0, 0.1, 0.3]:
                for alpha_a in [0.0, 0.2]:
                    param_grid.append({
                        "k": k,
                        "scale": scale,
                        "pred_ema_alpha": alpha_p,
                        "alloc_ema_alpha": alpha_a,
                        "max_leverage": 2.0,
                        "min_leverage": 0.0,
                        "vol_cap": 1.2,
                        "penalty": 2.0,
                    })

    best_score = -1e9
    best_params = None
    best_stats = None

    print(f"Searching over {len(param_grid)} mapping configs (exact metric)...")

    for i, params in enumerate(param_grid, 1):
        score, vol_ratio, sharpe_like, excess, _ = evaluate_strategy(
            preds,
            forward_returns,
            risk_free_rate,
            dates,
            params,
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_stats = {
                "vol_ratio": vol_ratio,
                "sharpe_like": sharpe_like,
                "excess": excess,
            }

        if i % 20 == 0:
            print(f"  Checked {i}/{len(param_grid)} configs, best_score={best_score:.4f}")

    return best_params, best_score, best_stats



# ==========================
# Fit ensemble on full data
# ==========================

def fit_full_ensemble(models, X: pd.DataFrame, y: np.ndarray):
    fitted = {}
    for name, model in models.items():
        mdl = clone(model)
        mdl.fit(X, y)
        fitted[name] = mdl
    return fitted


def ensemble_predict(fitted_models, meta_model, X: pd.DataFrame):
    model_names = list(fitted_models.keys())
    base_preds = []
    for name in model_names:
        base_preds.append(fitted_models[name].predict(X))
    base_preds = np.column_stack(base_preds)
    meta_preds = meta_model.predict(base_preds)
    return meta_preds


# ==========================
# Inference mapping
# ==========================

def build_allocation_from_preds(
    preds,
    mapping_params,
    pred_std=None,
    prev_pred_ema=None,
    prev_alloc_ema=None,
):
    """
    Apply the same mapping logic at inference time.

    In the live Kaggle API loop you will:
      - Keep pred_std fixed from training
      - Carry prev_pred_ema, prev_alloc_ema across days
    """
    preds = np.asarray(preds, dtype=float)

    if pred_std is None:
        pred_std = preds.std()
    if pred_std == 0 or np.isnan(pred_std):
        pred_std = 1.0

    z = preds / (pred_std + 1e-8)

    # EMA smoothing on preds
    alpha_p = mapping_params.get("pred_ema_alpha", 0.0)
    if alpha_p > 0:
        z_smoothed = np.zeros_like(z)
        if prev_pred_ema is None:
            z_smoothed[0] = z[0]
        else:
            z_smoothed[0] = alpha_p * z[0] + (1 - alpha_p) * prev_pred_ema
        for i in range(1, len(z)):
            z_smoothed[i] = alpha_p * z[i] + (1 - alpha_p) * z_smoothed[i - 1]
    else:
        z_smoothed = z

    k = mapping_params.get("k", 1.0)
    scale = mapping_params.get("scale", 1.0)
    max_leverage = mapping_params.get("max_leverage", 2.0)
    min_leverage = mapping_params.get("min_leverage", 0.0)

    alloc_raw = 1.0 + k * np.tanh(z_smoothed / scale)
    alloc_raw = np.clip(alloc_raw, min_leverage, max_leverage)

    # EMA smoothing on allocations
    alpha_a = mapping_params.get("alloc_ema_alpha", 0.0)
    if alpha_a > 0:
        alloc = np.zeros_like(alloc_raw)
        if prev_alloc_ema is None:
            alloc[0] = alloc_raw[0]
        else:
            alloc[0] = alpha_a * alloc_raw[0] + (1 - alpha_a) * prev_alloc_ema
        for i in range(1, len(alloc_raw)):
            alloc[i] = alpha_a * alloc_raw[i] + (1 - alpha_a) * alloc[i - 1]
    else:
        alloc = alloc_raw

    return alloc


# ==========================
# Main training entrypoint
# ==========================

def main(train_path, test_path=None, n_splits=5):
    # 1) Load
    train, test = load_data(train_path, test_path)
    feature_cols = get_feature_cols(train)

    X = train[feature_cols]
    y = train[TARGET_COL].values
    fwd = train[FORWARD_RET_COL].values
    dates = train["date_id"].values

    # 2) Base models + OOF
    models = make_base_models()
    oof_df, fold_indices = generate_oof_predictions(models, X, y, n_splits=n_splits)

    # 3) Meta-model on OOF
    meta_model, meta_oof = fit_meta_model(oof_df, y)

    # 4) Optimize strategy mapping on OOF predictions
    best_params, best_score, best_stats = optimize_mapping(
        meta_oof,
        fwd,                     # train['forward_returns'].values
        train[RISK_FREE_COL].values,
        dates,
    )
    pred_std = meta_oof.std()

    print("\n=== FINAL MAPPING (from OOF) ===")
    print("Best mapping params:", best_params)
    print("Best CV-like score:", best_score)
    print("Stats (vol_ratio, sharpe_like, excess):", best_stats)

    # 5) Refit ensemble on full data
    fitted_models = fit_full_ensemble(models, X, y)

    submission = None
    if test is not None:
        # 6) Example: generate allocations for offline test.csv
        X_test = test[feature_cols]
        meta_test_preds = ensemble_predict(fitted_models, meta_model, X_test)

        alloc_test = build_allocation_from_preds(
            meta_test_preds,
            best_params,
            pred_std=pred_std,
        )

        submission = pd.DataFrame({
            "date_id": test["date_id"],
            "allocation": alloc_test,
        })
        print("\n=== Sample of offline test allocations ===")
        print(submission.head())

    # Return everything you’ll need for the live API notebook
    return fitted_models, meta_model, best_params, pred_std, submission


if __name__ == "__main__":
    # Paths for local run; in Kaggle use "../input/..."
    train_path = "/mnt/data/train.csv"   # change to "../input/.../train.csv" on Kaggle
    test_path = "/mnt/data/test.csv"     # optional
    fitted_models, meta_model, best_params, pred_std, submission = main(
        train_path,
        test_path,
        n_splits=4,   # 4–6 is reasonable; increase if you want more stability
    )
