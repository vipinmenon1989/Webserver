# treecrispr/models.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd

class FeatureMismatchError(RuntimeError):
    def __init__(self, model_name: str, have_count: int, expected_count: Optional[int],
                 missing: Optional[list[str]] = None, extra: Optional[list[str]] = None):
        self.model_name = model_name
        self.have_count = have_count
        self.expected_count = expected_count
        self.missing = missing or []
        self.extra = extra or []
        parts = [f"{model_name}: feature mismatch",
                 f"have={have_count}",
                 f"expected={expected_count if expected_count is not None else 'unknown'}"]
        if self.missing: parts.append(f"missing={len(self.missing)} (e.g. {', '.join(self.missing[:10])})")
        if self.extra:   parts.append(f"extra={len(self.extra)} (e.g. {', '.join(self.extra[:10])})")
        super().__init__(", ".join(parts))

def load_models(model_dir: Path, logger=None) -> Dict[str, object]:
    """Load ALL models from directory (.pkl/.joblib, case-insensitive)."""
    models: Dict[str, object] = {}
    try:
        import joblib
    except Exception as e:
        logger and logger.warning(f"joblib not available: {e}")
        return models
    if not model_dir or not model_dir.exists():
        logger and logger.warning(f"Model dir missing: {model_dir}")
        return models

    for p in sorted(model_dir.iterdir()):
        if not p.is_file(): 
            continue
        if p.suffix.lower() not in {".pkl", ".joblib"}:
            continue
        try:
            mdl = joblib.load(p)  # joblib.load is the right API for our pickles.  [oai_citation:0‡joblib.readthedocs.io](https://joblib.readthedocs.io/en/latest/generated/joblib.load.html?utm_source=chatgpt.com)
            models[p.stem] = mdl
            logger and logger.info(f"Loaded model: {p.name}")
        except Exception as e:
            logger and logger.warning(f"Failed loading {p.name}: {e}")
    logger and logger.info(f"Total models loaded: {len(models)} from {model_dir}")
    return models

def _booster_from(model):
    try:
        getb = getattr(model, "get_booster", None)
        if callable(getb):
            return getb()
    except Exception:
        pass
    try:
        import xgboost as xgb  # noqa
        from xgboost import Booster
        if isinstance(model, Booster):
            return model
    except Exception:
        pass
    return None

def _expected_feature_names(model, _Xcols: Sequence[str]) -> Optional[list[str]]:
    names = getattr(model, "feature_names_in_", None)  # sklearn stores names when fit on DataFrame.  [oai_citation:1‡Scikit-learn](https://scikit-learn.org/stable/developers/develop.html?utm_source=chatgpt.com)
    if names is not None:
        return list(names)
    booster = _booster_from(model)
    if booster is not None:
        try:
            names = booster.feature_names
            if names:
                return list(names)
        except Exception:
            pass
    return None

def _expected_feature_count(model) -> Optional[int]:
    n = getattr(model, "n_features_in_", None)
    if isinstance(n, (int, np.integer)) and n > 0:
        return int(n)
    booster = _booster_from(model)
    if booster is not None:
        try:
            return int(booster.num_features())
        except Exception:
            pass
    return None

def _diagnose(X: pd.DataFrame, model, model_name: str) -> None:
    """Strict check: raise with details if names or counts don’t match."""
    # Only numeric features go to models (per pandas select_dtypes docs).  [oai_citation:2‡Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html?utm_source=chatgpt.com)
    Xnum = X.select_dtypes(include=[np.number]).copy()
    have_cols = list(Xnum.columns)
    have_count = len(have_cols)

    exp_names = _expected_feature_names(model, have_cols)
    exp_count = _expected_feature_count(model)

    if exp_names is not None:
        exp_set, have_set = set(exp_names), set(have_cols)
        missing = [n for n in exp_names if n not in have_set]  # preserve model order
        extra   = [c for c in have_cols if c not in exp_set]
        if missing or extra:
            raise FeatureMismatchError(model_name, have_count, exp_count, missing, extra)

    if isinstance(exp_count, int) and have_count != exp_count:
        raise FeatureMismatchError(model_name, have_count, exp_count)

def _predict_raw(model, X: pd.DataFrame):
    booster = _booster_from(model)
    if booster is not None:
        import xgboost as xgb
        # XGBoost expects a DMatrix / compatible input for predict.  [oai_citation:3‡XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/prediction.html?utm_source=chatgpt.com)
        dm = xgb.DMatrix(X.select_dtypes(include=[np.number]).values)
        return booster.predict(dm)
    Xnum = X.select_dtypes(include=[np.number]).copy()
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(Xnum)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.max(axis=1)
        except Exception:
            pass
    return model.predict(Xnum)

def score_with_models(
    df_features: pd.DataFrame,
    models: Dict[str, object],
    logger=None,
    model_dir=None,   # accepted for logging/back-compat
) -> pd.DataFrame:
    """
    Try to score with EVERY model. Never auto-align. 
    If a model mismatches or errors, log and skip it; continue with others.
    """
    if df_features.empty or not models:
        return pd.DataFrame(index=df_features.index)

    outputs = {}
    failed  = {}

    for name, mdl in models.items():
        try:
            _diagnose(df_features, mdl, name)   # raises on mismatch
            preds = _predict_raw(mdl, df_features)
            outputs[name] = np.asarray(preds, dtype=float)
        except FeatureMismatchError as e:
            failed[name] = str(e)
            logger and logger.warning(f"[SKIP] {e}")
        except Exception as e:
            failed[name] = repr(e)
            logger and logger.warning(f"[SKIP] {name}: {e}")

    if logger:
        logger.info(f"Scored with {len(outputs)}/{len(models)} models from {model_dir or '<unknown>'}")
        if failed:
            logger.warning("Failed models:\n  - " + "\n  - ".join(f"{k}: {v}" for k, v in failed.items()))

    return pd.DataFrame(outputs, index=df_features.index)