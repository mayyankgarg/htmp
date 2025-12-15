import optuna
import numpy as np
import pandas as pd

# ------------------------
# CONFIG
# ------------------------
USE_MODELS = [
    "cat",
    "enet",
    "lgb",
    "xgb",
    "lasso",
    "quantile",
    "bayesridge"
]

oof_sub = oof_df[USE_MODELS].values
fwd = np.asarray(fwd, dtype=float)
rf = np.asarray(rf, dtype=float)

# ------------------------
# ONE-SHOT STRATEGY SEARCH
# ------------------------
def objective(trial):
    # ----- convex weights -----
    raw = np.array([
        trial.suggest_float(f"w_{i}", 0.0, 1.0)
        for i in range(oof_sub.shape[1])
    ])
    if raw.sum() == 0:
        weights = np.ones_like(raw) / len(raw)
    else:
        weights = raw / raw.sum()

    preds = oof_sub @ weights
    pred_std = preds.std() if preds.std() > 0 else 1.0

    # ----- mapping params -----
    k = trial.suggest_float("k", 0.4, 2.0)
    scale = trial.suggest_float("scale", 0.5, 2.0)
    pred_ema = trial.suggest_float("pred_ema", 0.0, 0.2)
    alloc_ema = trial.suggest_float("alloc_ema", 0.0, 0.2)
    max_lev = trial.suggest_float("max_lev", 1.2, 1.8)
    vol_window = trial.suggest_int("vol_window", 40, 80)
    vol_mult = trial.suggest_float("vol_mult", 0.9, 1.3)

    params = dict(
        k=k,
        scale=scale,
        pred_ema_alpha=pred_ema,
        alloc_ema_alpha=alloc_ema,
        max_leverage=max_lev,
        min_leverage=0.0,
        vol_target=True,
        vol_window=vol_window,
        vol_mult=vol_mult,
        market_window=vol_window,
        lag=1
    )

    alloc = build_alloc_from_preds_array(
        preds,
        params,
        pred_std=pred_std,
        forward_returns=fwd
    )

    sol = pd.DataFrame({
        "forward_returns": fwd,
        "risk_free_rate": rf
    })
    sub = pd.DataFrame({"prediction": alloc})

    try:
        return hull_score(sol, sub)
    except Exception:
        return -1e9


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=150, n_jobs=1)

print("BEST SCORE:", study.best_value)
print("BEST PARAMS:", study.best_params)
