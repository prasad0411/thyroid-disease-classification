"""
TensorFlow (Keras) neural network baseline on the exact splits used by train.py.

Reads outputs/split_cache.pkl (90,000 train, 30,000 validation, 30,000 test rows,
the same 12 RFE selected features) and verifies the test set matches the one
behind the production XGBoost model before training, so the comparison is fair.

Model: a three layer MLP with batch norm and dropout, class weighted loss for the
two minority classes, early stopping on validation loss, fixed seeds.

Run:  python tf_baseline.py     (writes outputs/tf_baseline.json)
"""
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
OUT = Path("outputs/tf_baseline.json")


def load():
    d = pickle.loads(Path("outputs/split_cache.pkl").read_bytes())
    Xtr, Xva, Xte = (np.asarray(d[k], dtype="float32") for k in ("Xtr", "Xva", "Xte"))
    ytr, yva, yte = (np.asarray(d[k]) for k in ("ytr", "yva", "yte"))
    # The cache may hold raw or scaled features; scale with the training scaler if raw.
    if np.abs(Xtr.mean(axis=0)).max() > 0.1 or np.abs(Xtr.std(axis=0) - 1).max() > 0.1:
        Xtr, Xva, Xte = (d["scaler"].transform(x).astype("float32") for x in (Xtr, Xva, Xte))
    return Xtr, ytr, Xva, yva, Xte, yte, d["feats"], d["le"]


def check_same_test_set(yte, le, feats):
    meta_path = sorted(Path("models").glob("metadata_*.json"))[-1]
    meta = json.loads(meta_path.read_text())
    support = {k: int(v["support"]) for k, v in meta["test"]["per_class"].items()}
    ours = {cls: int((yte == i).sum()) for i, cls in enumerate(le.classes_)}
    if ours != support or list(feats) != meta["features"]["names"]:
        raise SystemExit(f"Split mismatch with {meta_path.name}: {ours} vs {support}")
    return meta


def main():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    Xtr, ytr, Xva, yva, Xte, yte, feats, le = load()
    meta = check_same_test_set(yte, le, feats)
    n_classes = len(le.classes_)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xtr.shape[1],)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=512, verbose=2,
        class_weight=dict(enumerate(weights)),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)],
    )
    train_s = time.time() - t0

    proba = model.predict(Xte, batch_size=4096, verbose=0)
    pred = proba.argmax(axis=1)
    rep = classification_report(yte, pred, target_names=le.classes_, output_dict=True, zero_division=0)
    xgb = meta["test"]
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "framework": f"TensorFlow {tf.__version__} (Keras)",
        "architecture": "dense layers 256, 128 and 64 with batch norm, dropout 0.25 and a class weighted loss",
        "parameters": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "train_seconds": round(train_s, 1),
        "test": {
            "accuracy": float(accuracy_score(yte, pred)),
            "f1_macro": float(f1_score(yte, pred, average="macro")),
            "log_loss": float(log_loss(yte, proba, labels=list(range(n_classes)))),
            "recall": {c: float(rep[c]["recall"]) for c in le.classes_},
        },
        "xgboost_test": {"accuracy": xgb["accuracy"], "f1_macro": xgb["f1_macro"], "log_loss": xgb["log_loss"]},
        "same_split_as": meta["experiment_id"],
    }
    OUT.write_text(json.dumps(result, indent=2))
    t = result["test"]
    print(f"\nTensorFlow MLP  accuracy {t['accuracy']:.4f}  macro F1 {t['f1_macro']:.4f}  log loss {t['log_loss']:.4f}")
    print(f"XGBoost         accuracy {xgb['accuracy']:.4f}  macro F1 {xgb['f1_macro']:.4f}  log loss {xgb['log_loss']:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
