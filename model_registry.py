"""
Single source of truth for loading the trained model bundle.

Supports both metadata schemas (legacy: features_selected / best_model /
performance_metrics, current: features.names / selected_model / test) so the
API and the Streamlit app can never drift onto different keys again.
"""
import json
import os
from dataclasses import dataclass

import joblib
import pandas as pd

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
PARTS = ("best_model", "scaler", "label_encoder")


@dataclass
class ModelBundle:
    timestamp: str
    model: object
    scaler: object
    label_encoder: object
    metadata: dict
    features: list
    model_name: str
    metrics: dict
    dataset_size: int
    baseline: float | None


def _complete_timestamps(models_dir):
    stamps = []
    for f in os.listdir(models_dir):
        if f.startswith("metadata_") and f.endswith(".json"):
            ts = f[len("metadata_"):-len(".json")]
            if all(os.path.exists(f"{models_dir}/{p}_{ts}.pkl") for p in PARTS):
                stamps.append(ts)
    return sorted(stamps, reverse=True)


def normalize_metadata(meta):
    features = meta.get("features_selected") or meta.get("features", {}).get("names")
    if not features:
        raise KeyError("metadata has neither features_selected nor features.names")
    name = meta.get("selected_model") or meta.get("best_model") or "unknown"
    if isinstance(meta.get("test"), dict):
        t = meta["test"]
        metrics = {k: t[k] for k in ("accuracy", "f1_macro", "log_loss") if k in t}
    else:
        metrics = dict(meta.get("performance_metrics", {}).get(name, {}))
    baseline = meta.get("leakage_audit", {}).get("majority_baseline")
    return list(features), name, metrics, int(meta.get("dataset_size", 0)), baseline


def load_latest(models_dir=MODELS_DIR):
    stamps = _complete_timestamps(models_dir)
    if not stamps:
        raise FileNotFoundError(f"no complete model artifact set in {models_dir}/")
    ts = stamps[0]
    with open(f"{models_dir}/metadata_{ts}.json") as fh:
        meta = json.load(fh)
    features, name, metrics, size, baseline = normalize_metadata(meta)
    return ModelBundle(
        timestamp=ts,
        model=joblib.load(f"{models_dir}/best_model_{ts}.pkl"),
        scaler=joblib.load(f"{models_dir}/scaler_{ts}.pkl"),
        label_encoder=joblib.load(f"{models_dir}/label_encoder_{ts}.pkl"),
        metadata=meta,
        features=features,
        model_name=name,
        metrics=metrics,
        dataset_size=size,
        baseline=baseline,
    )


def add_derived(row):
    row = dict(row)
    if "T4" in row and "T4U" in row:
        row["FTI"] = row["T4"] / (row["T4U"] + 0.01)  # matches data_generator.py
    return row


def predict_one(bundle, row):
    row = add_derived(row)
    missing = [f for f in bundle.features if f not in row]
    if missing:
        raise ValueError(f"missing model features: {missing}")
    X = pd.DataFrame([{f: row[f] for f in bundle.features}])
    proba = bundle.model.predict_proba(bundle.scaler.transform(X))[0]
    labels = bundle.label_encoder.inverse_transform(bundle.model.classes_)
    probs = {str(lbl): float(p) for lbl, p in zip(labels, proba)}
    pred = max(probs, key=probs.get)
    return {"prediction": pred, "confidence": probs[pred], "probabilities": probs}


def unwrap_tree_model(model, max_depth=6):
    """Return the underlying XGBoost estimator for SHAP, looking through
    CalibratedClassifierCV, FrozenEstimator and voting/stacking wrappers."""
    m = model
    for _ in range(max_depth):
        if "XGB" in type(m).__name__:
            return m
        if hasattr(m, "named_estimators_"):
            for est in m.named_estimators_.values():
                if "XGB" in type(est).__name__:
                    return est
        if hasattr(m, "steps"):
            for step_name, step in m.steps[:-1]:
                if hasattr(step, "transform") and not hasattr(step, "fit_resample"):
                    raise ValueError(f"pipeline step {step_name!r} transforms features; SHAP would need its output")
            m = m.steps[-1][1]
            continue
        if hasattr(m, "calibrated_classifiers_"):
            m = m.calibrated_classifiers_[0].estimator
        elif hasattr(m, "estimator") and m.estimator is not None:
            m = m.estimator
        else:
            break
    return m


def headline_metrics(bundle):
    m = bundle.metrics
    return {
        "accuracy": m.get("accuracy"),
        "f1_macro": m.get("f1_macro", m.get("f1_score")),
        "baseline": bundle.baseline,
    }


def comparison_table(meta):
    """Per model metrics for the comparison chart: validation split in the
    current schema, performance_metrics in the legacy one."""
    if isinstance(meta.get("validation"), dict):
        return meta["validation"]
    return meta.get("performance_metrics", {})
