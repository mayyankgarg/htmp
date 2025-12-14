import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd

from script_with_optims_datafix import (
    add_rolling_features,
    build_alloc_from_preds_array,
    ensemble_predict_from_fitted,
)

# Optional boosters
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import catboost as cb
except Exception:
    cb = None


MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0


# --------------------------
# Model loading helpers
# --------------------------

def load_model(model_name, model_dir):
    # XGBoost
    if os.path.exists(os.path.join(model_dir, f"{model_name}.json")):
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(model_dir, f"{model_name}.json"))
        return model

    # LightGBM
    if os.path.exists(os.path.join(model_dir, f"{model_name}.txt")):
        return lgb.Booster(model_file=os.path.join(model_dir, f"{model_name}.txt"))

    # CatBoost
    if os.path.exists(os.path.join(model_dir, f"{model_name}.cbm")):
        model = cb.CatBoostRegressor()
        model.load_model(os.path.join(model_dir, f"{model_name}.cbm"))
        return model

    # sklearn / others
    with open(os.path.join(model_dir, f"{model_name}.pkl"), "rb") as f:
        return pickle.load(f)


# --------------------------
# Artifact loader
# --------------------------

def load_artifacts(base_path):
    meta_path = os.path.join(base_path, "metadata")
    prep_path = os.path.join(base_path, "preprocessors")
    model_path = os.path.join(base_path, "models")

    with open(os.path.join(meta_path, "best_weights.json")) as f:
        best_weights = json.load(f)

    with open(os.path.join(meta_path, "best_mapping.json")) as f:
        best_mapping = json.load(f)

    pred_std = np.load(os.path.join(meta_path, "pred_std.npy")).item()

    with open(os.path.join(meta_path, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    with open(os.path.join(meta_path, "regime_feature_cols.json")) as f:
        regime_feature_cols = json.load(f)

    with open(os.path.join(meta_path, "introduced_flags.json")) as f:
        introduced_flags = json.load(f)

    with open(os.path.join(prep_path, "pca.pkl"), "rb") as f:
        pca_model = pickle.load(f)

    with open(os.path.join(prep_path, "regime_scaler.pkl"), "rb") as f:
        regime_scaler = pickle.load(f)

    # Load models
    fitted_models = {}
    for fname in os.listdir(model_path):
        model_name = os.path.splitext(fname)[0]
        if model_name not in fitted_models:
            fitted_models[model_name] = load_model(model_name, model_path)

    return {
        "fitted_models": fitted_models,
        "best_weights": best_weights,
        "best_mapping": best_mapping,
        "pred_std": pred_std,
        "feature_cols": feature_cols,
        "regime_feature_cols": regime_feature_cols,
        "introduced_flags": introduced_flags,
        "pca_model": pca_model,
        "regime_scaler": regime_scaler,
    }


# --------------------------
# Inference
# --------------------------

def run_inference(test_path: str, artifacts_path: str = "artifacts"):
    artifacts = load_artifacts(artifacts_path)

    fitted_models = artifacts["fitted_models"]
    best_weights = artifacts["best_weights"]
    best_mapping = artifacts["best_mapping"]
    pred_std = artifacts["pred_std"]
    pca_model = artifacts["pca_model"]
    feature_cols = artifacts["feature_cols"]
    introduced_flags = artifacts["introduced_flags"]
    regime_feature_cols = artifacts["regime_feature_cols"]
    regime_scaler = artifacts["regime_scaler"]

    test = pd.read_csv(test_path)
    test = test.sort_values("date_id").reset_index(drop=True)

    test = test.rename(
        columns={
            "lagged_forward_returns": "forward_returns",
            "lagged_risk_free_rate": "risk_free_rate",
            "lagged_market_forward_excess_returns": "market_forward_excess_returns",
        }
    )

    # Rolling features (causal)
    test = add_rolling_features(test, price_col="forward_returns")

    # Missing indicators + introduced flags
    for c in feature_cols:
        test[f"{c}__miss"] = test[c].isna().astype(int)
        test[f"{c}__introduced_late"] = int(introduced_flags.get(c, 0))

    cats = [f"{c}__miss" for c in feature_cols] + [f"{c}__introduced_late" for c in feature_cols]

    X_test_for_trees = pd.concat([test[feature_cols], test[cats]], axis=1)

    # Linear fallback imputation (safe, causal)
    X_test_linear_num = test[feature_cols].fillna(0.0)
    X_test_for_linear = pd.concat([X_test_linear_num, test[cats]], axis=1)

    # PCA
    pca_input = X_test_linear_num.fillna(0.0).values
    comps = pca_model.transform(pca_input)
    for i in range(comps.shape[1]):
        X_test_for_trees[f"pca_{i}"] = comps[:, i]
        X_test_for_linear[f"pca_{i}"] = comps[:, i]

    # Regime inference
    if regime_feature_cols:
        reg_X = test[regime_feature_cols].fillna(0.0).values
        reg_X_std = regime_scaler.transform(reg_X)
        test["regime"] = 0  # default
        try:
            test["regime"] = artifacts.get("hmm_model", None).predict(reg_X_std)
        except Exception:
            pass

    tree_models = {"hgb", "rf", "et", "lgb", "xgb", "cat"}
    X_by_model = {
        name: (X_test_for_trees if name in tree_models else X_test_for_linear)
        for name in fitted_models.keys()
    }

    regimes_arr = test["regime"].to_numpy() if "regime" in test.columns else None
    blended = ensemble_predict_from_fitted(
        fitted_models, best_weights, X_by_model, regimes=regimes_arr
    )

    # Mapping
    if isinstance(best_mapping, dict) and "per_regime" in best_mapping:
        alloc = np.zeros_like(blended)
        regs = regimes_arr
        for r, params in best_mapping["per_regime"].items():
            mask = regs == int(r)
            if mask.any():
                alloc[mask] = build_alloc_from_preds_array(
                    blended[mask],
                    params,
                    pred_std=pred_std,
                    forward_returns=test["forward_returns"].values[mask],
                )
    else:
        alloc = build_alloc_from_preds_array(
            blended,
            best_mapping,
            pred_std=pred_std,
            forward_returns=test["forward_returns"].values,
        )

    return pd.DataFrame(
        {"date_id": test["date_id"].values, "allocation": alloc}
    )


# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="data/test.csv")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    args = parser.parse_args()

    sub = run_inference(args.test, args.artifacts)
    print(sub.head(50))
    sub.to_csv("submission.csv", index=False)
