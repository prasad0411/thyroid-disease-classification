"""
Thyroid Disease Classification - training pipeline (methodologically corrected).

FOUR DEFECTS FIXED FROM THE PREVIOUS VERSION
--------------------------------------------
1. FEATURE SELECTION LEAK
   Old: selector.fit(X, y_encoded) ran on the FULL dataset, then split.
   Test rows influenced which features were chosen. evaluate.py had this right
   but train.py did not, and train.py is what wrote the deployed .pkl files.
   Now: split first, RFE fit on the training fold only.

2. SMOTE APPLIED OUTSIDE CROSS VALIDATION
   Old: SMOTE ran once on the whole training set before any CV.
   Synthetic minority points derived from a sample can land in the validation
   fold that scored it, inflating minority recall.
   Now: SMOTE lives inside an imblearn Pipeline so it refits per fold.

3. MODEL SELECTION ON THE TEST SET
   Old: best_model = max(results, key=test f1). Choosing among three models by
   test score makes the reported test number optimistically biased.
   Now: three way split. Selection on validation, test touched exactly once.

4. SINGLE SPLIT, SINGLE SEED, NO UNCERTAINTY
   Old: one train_test_split at random_state=42, no CV, no variance estimate.
   Now: StratifiedKFold 5 fold on the training set, reported as mean +/- std.

ALSO ADDED
   Probability calibration with CalibratedClassifierCV plus Brier score, so
   predicted probabilities are usable clinically rather than only ranked.
   Subgroup metrics by sex and age band for the model card.
   experiments.json emitted from code rather than hand authored.

Execution: python train.py
"""

import json
import logging
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             log_loss)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import *
from data_generator import AUDIT_COLUMNS, generate_medical_dataset, leakage_audit

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False


def setup_logging():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(OUTPUTS_DIR / "training.log"), logging.StreamHandler()])
    return logging.getLogger()


def build_estimators():
    est = {}
    if XGB_OK:
        est["XGBoost"] = xgb.XGBClassifier(**XGBOOST_PARAMS)
    est["RandomForest"] = RandomForestClassifier(**RANDOMFOREST_PARAMS)
    if len(est) > 1:
        est["Ensemble"] = VotingClassifier(
            estimators=[(k, v) for k, v in est.items()], voting="soft", n_jobs=-1)
    return est


def wrap(estimator):
    """SMOTE inside the pipeline so it refits per CV fold."""
    if SMOTE_OK:
        return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)),
                            ("clf", estimator)])
    return estimator


def subgroup_metrics(model, X, y_true, meta, classes):
    """Per subgroup accuracy and macro F1 for the model card."""
    pred = model.predict(X)
    out = {}
    for label, mask in [
        ("sex_0", meta["sex"] == 0), ("sex_1", meta["sex"] == 1),
        ("age_under_40", meta["age"] < 40),
        ("age_40_to_65", (meta["age"] >= 40) & (meta["age"] < 65)),
        ("age_65_plus", meta["age"] >= 65),
        ("subclinical", meta["subclinical"] == 1),
        ("overt", meta["subclinical"] == 0),
    ]:
        m = np.asarray(mask)
        if m.sum() < 40:
            continue
        out[label] = {"n": int(m.sum()),
                      "accuracy": float(accuracy_score(y_true[m], pred[m])),
                      "f1_macro": float(f1_score(y_true[m], pred[m], average="macro",
                                                 zero_division=0))}
    accs = [v["accuracy"] for v in out.values()]
    out["_max_disparity"] = float(max(accs) - min(accs)) if accs else 0.0
    return out


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("STAGE 1  data generation and leakage audit")
    df = generate_medical_dataset()
    audit = leakage_audit(df)

    meta_cols = df[["sex", "age", "subclinical"]].copy()
    X = df.drop(columns=AUDIT_COLUMNS + ["target"])
    le = LabelEncoder()
    y = le.fit_transform(df["target"])
    log.info(f"{len(X):,} rows, {X.shape[1]} features, classes {list(le.classes_)}")

    log.info("STAGE 2  three way split, stratified")
    X_tmp, X_te, y_tmp, y_te, m_tmp, m_te = train_test_split(
        X, y, meta_cols, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_tr, X_va, y_tr, y_va, m_tr, m_va = train_test_split(
        X_tmp, y_tmp, m_tmp, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_tmp)
    log.info(f"train {len(X_tr):,} | val {len(X_va):,} | test {len(X_te):,}")

    log.info("STAGE 3  RFE on the training fold only")
    sel = RFE(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
              n_features_to_select=N_FEATURES_SELECTED, step=1).fit(X_tr, y_tr)
    feats = X_tr.columns[sel.support_].tolist()
    X_tr, X_va, X_te = X_tr[feats], X_va[feats], X_te[feats]
    log.info(f"{X.shape[1]} -> {len(feats)} features: {feats}")

    log.info("STAGE 4  scaling fit on train only")
    scaler = StandardScaler().fit(X_tr)
    Xtr, Xva, Xte = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    log.info("STAGE 5  5 fold stratified CV on the training set, SMOTE inside folds")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, est in build_estimators().items():
        scores = cross_val_score(wrap(est), Xtr, y_tr, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        cv_results[name] = {"f1_macro_mean": float(scores.mean()),
                            "f1_macro_std": float(scores.std()),
                            "folds": [float(s) for s in scores]}
        log.info(f"  {name:14s} CV f1_macro {scores.mean():.4f} +/- {scores.std():.4f}")

    log.info("STAGE 6  fit on train, select on validation")
    val_results, fitted = {}, {}
    for name, est in build_estimators().items():
        m = wrap(est).fit(Xtr, y_tr)
        fitted[name] = m
        pv = m.predict(Xva)
        val_results[name] = {"accuracy": float(accuracy_score(y_va, pv)),
                             "f1_macro": float(f1_score(y_va, pv, average="macro",
                                                        zero_division=0))}
        log.info(f"  {name:14s} val acc {val_results[name]['accuracy']:.4f} "
                 f"f1_macro {val_results[name]['f1_macro']:.4f}")

    best_name = max(val_results, key=lambda k: val_results[k]["f1_macro"])
    log.info(f"selected on validation: {best_name}")

    log.info("STAGE 7  probability calibration on the validation split")
    # sklearn >= 1.6 removed cv="prefit" in favour of FrozenEstimator.
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(fitted[best_name]), method="isotonic")
    except ImportError:
        calibrated = CalibratedClassifierCV(
            fitted[best_name], method="isotonic", cv="prefit")
    calibrated.fit(Xva, y_va)

    log.info("STAGE 8  final evaluation, test set touched once")
    pred = calibrated.predict(Xte)
    proba = calibrated.predict_proba(Xte)
    test_acc = float(accuracy_score(y_te, pred))
    test_f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
    rep = classification_report(y_te, pred, target_names=le.classes_,
                                output_dict=True, zero_division=0)

    brier_raw, brier_cal = {}, {}
    raw_proba = fitted[best_name].predict_proba(Xte)
    for i, c in enumerate(le.classes_):
        yt = (y_te == i).astype(int)
        brier_raw[c] = float(brier_score_loss(yt, raw_proba[:, i]))
        brier_cal[c] = float(brier_score_loss(yt, proba[:, i]))

    log.info(f"TEST accuracy {test_acc:.4f} | macro F1 {test_f1:.4f} "
             f"| log loss {log_loss(y_te, proba):.4f}")
    for c in le.classes_:
        r = rep[c]
        log.info(f"  {c:14s} P {r['precision']:.3f} R {r['recall']:.3f} "
                 f"F1 {r['f1-score']:.3f} | brier {brier_raw[c]:.4f} -> {brier_cal[c]:.4f}")

    sub = subgroup_metrics(calibrated, Xte, y_te, m_te, le.classes_)
    log.info(f"max subgroup accuracy disparity: {sub['_max_disparity']:.4f}")

    log.info("STAGE 9  persistence")
    joblib.dump(calibrated, MODELS_DIR / f"best_model_{stamp}.pkl")
    joblib.dump(scaler, MODELS_DIR / f"scaler_{stamp}.pkl")
    joblib.dump(le, MODELS_DIR / f"label_encoder_{stamp}.pkl")

    record = {
        "experiment_id": f"exp_{stamp}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "generator": "causal v2, no feature derived from target",
        "leakage_audit": {"majority_baseline": audit["baseline"],
                          "clinical_flags_only": audit["flag_accuracy"],
                          "lift_over_baseline": audit["flag_accuracy"] - audit["baseline"]},
        "dataset_size": int(len(df)),
        "split": {"train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
        "features": {"original": int(X.shape[1]), "selected": len(feats), "names": feats},
        "class_balancing": "SMOTE inside CV folds" if SMOTE_OK else "none",
        "cv_5fold_train": cv_results,
        "validation": val_results,
        "selected_model": best_name,
        "selection_criterion": "validation macro F1",
        "calibration": {"method": "isotonic", "brier_raw": brier_raw,
                        "brier_calibrated": brier_cal},
        "test": {"accuracy": test_acc, "f1_macro": test_f1,
                 "log_loss": float(log_loss(y_te, proba)),
                 "per_class": {c: {k: float(v) for k, v in rep[c].items()}
                               for c in le.classes_},
                 "confusion_matrix": confusion_matrix(y_te, pred).tolist()},
        "subgroups": sub,
        "artifacts": {"model": f"best_model_{stamp}.pkl",
                      "scaler": f"scaler_{stamp}.pkl",
                      "label_encoder": f"label_encoder_{stamp}.pkl"},
    }

    path = OUTPUTS_DIR / "experiments.json"
    history = json.loads(path.read_text()) if path.exists() else []
    if not isinstance(history, list):
        history = []
    history.append(record)
    path.write_text(json.dumps(history, indent=2))
    (MODELS_DIR / f"metadata_{stamp}.json").write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 72)
    print(f"selected model      : {best_name}  (chosen on validation)")
    print(f"CV macro F1 (train) : {cv_results[best_name]['f1_macro_mean']:.4f}"
          f" +/- {cv_results[best_name]['f1_macro_std']:.4f}")
    print(f"TEST accuracy       : {test_acc:.4f}")
    print(f"TEST macro F1       : {test_f1:.4f}")
    print(f"leak audit lift     : {audit['flag_accuracy'] - audit['baseline']:+.4f}")
    print(f"subgroup disparity  : {sub['_max_disparity']:.4f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
