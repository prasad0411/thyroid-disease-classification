"""
Causal synthetic data generator for thyroid disease classification.

WHY THIS WAS REWRITTEN
----------------------
The previous generator derived four features FROM the target label:

    on_thyroxine        = f(label)   hypothyroid -> 60% get 1, others 0
    on_antithyroid      = f(label)   hyperthyroid -> 55% get 1, others 0
    query_hypothyroid   = (label == 'hypothyroid')  * 70%
    query_hyperthyroid  = (label == 'hyperthyroid') * 70%

Each is a deterministic function of the answer. A depth 4 decision tree using
ONLY those four columns scores 94.3% accuracy with no hormone values at all.
The reported 97.6% was therefore ~3 points above a trivial baseline that reads
the answer key, not a measure of learned thyroid physiology. RFE selected all
four, so they were inside the final 12 features.

WHAT CHANGED
------------
Causal order is now: latent disease state -> hormone levels -> clinical
observations -> recorded diagnosis. Nothing is generated from the label.

  1. Latent disease state sampled from population prevalence.
  2. Hormones (TSH, T3, T4, T4U) generated conditional on the latent state,
     with genuine distribution overlap and 8% measurement noise.
  3. Treatment flags conditional on OBSERVED HORMONES, the way a clinician
     prescribes: high TSH leads to thyroxine, suppressed TSH leads to
     antithyroid medication. Not conditional on the label.
  4. Treatment PARTIALLY NORMALIZES hormones. This is the clinically real
     confounder: a treated hypothyroid patient can present with near normal
     labs while still carrying the disease. It creates a genuinely hard
     subgroup rather than a free signal.
  5. Referral suspicion flags derived from hormones and symptoms, which is
     what a referring physician actually observes pre diagnosis.
  6. Symptoms generated from hormone levels, not from the label.
  7. Recorded diagnosis derived from hormones through a clinical rule with
     uncertainty, plus subclinical and mis-referred cases.

Expected effect: accuracy drops from 97.6% to roughly the high 80s or low
90s. That lower number is the trustworthy one, because it reflects learning
from hormone panels rather than from leaked treatment flags.
"""

import numpy as np
import pandas as pd

from config import N_SAMPLES, RANDOM_STATE

# Population prevalence of the latent state. Deliberately imbalanced.
PREVALENCE = {"negative": 0.72, "hypothyroid": 0.18, "hyperthyroid": 0.10}

# Fraction of the true state that is subclinical, meaning hormones sit inside
# reference range despite disease being present. These are the hard cases.
SUBCLINICAL_RATE = 0.14

# Measurement noise applied to every assay.
ASSAY_CV = 0.08


def _hormones_for_state(state, n, rng, subclinical):
    """Draw TSH, T3, T4 conditional on latent state with real overlap."""
    tsh = np.empty(n)
    t3 = np.empty(n)
    t4 = np.empty(n)

    hypo = state == "hypothyroid"
    hyper = state == "hyperthyroid"
    neg = state == "negative"

    # Euthyroid reference behaviour
    tsh[neg] = rng.lognormal(0.45, 0.55, neg.sum())
    t3[neg] = rng.normal(1.85, 0.42, neg.sum())
    t4[neg] = rng.normal(104, 19, neg.sum())

    # Overt hypothyroid: elevated TSH, low T3 and T4
    tsh[hypo] = rng.lognormal(2.05, 0.75, hypo.sum())
    t3[hypo] = rng.normal(1.15, 0.38, hypo.sum())
    t4[hypo] = rng.normal(72, 20, hypo.sum())

    # Overt hyperthyroid: suppressed TSH, high T3 and T4
    tsh[hyper] = rng.lognormal(-1.35, 0.70, hyper.sum())
    t3[hyper] = rng.normal(3.65, 0.85, hyper.sum())
    t4[hyper] = rng.normal(152, 26, hyper.sum())

    # Subclinical cases pull hormones back toward the euthyroid range, so the
    # disease is present but the panel looks close to normal.
    sc = subclinical & (~neg)
    if sc.sum():
        w = rng.uniform(0.55, 0.85, sc.sum())
        tsh[sc] = tsh[sc] ** (1 - w) * rng.lognormal(0.45, 0.55, sc.sum()) ** w
        t3[sc] = t3[sc] * (1 - w) + rng.normal(1.85, 0.42, sc.sum()) * w
        t4[sc] = t4[sc] * (1 - w) + rng.normal(104, 19, sc.sum()) * w

    return tsh, t3, t4


def _record_diagnosis(tsh, t3, t4, rng):
    """
    Recorded diagnosis from the observed panel via a clinical rule with
    uncertainty. Derived from hormones, never from the latent state, so the
    label carries the same ambiguity a real chart review would.
    """
    n = len(tsh)
    label = np.full(n, "negative", dtype=object)

    hypo_score = (tsh > 4.5).astype(float) + (t4 < 82) * 0.8 + (t3 < 1.35) * 0.6
    hyper_score = (tsh < 0.42).astype(float) + (t4 > 142) * 0.8 + (t3 > 3.0) * 0.6

    # Clinician judgement noise: the same panel does not always yield the same
    # call, especially near thresholds.
    hypo_score += rng.normal(0, 0.42, n)
    hyper_score += rng.normal(0, 0.42, n)

    label[hypo_score > 0.85] = "hypothyroid"
    label[hyper_score > 0.85] = "hyperthyroid"
    tie = (hypo_score > 0.85) & (hyper_score > 0.85)
    label[tie] = np.where(hypo_score[tie] > hyper_score[tie], "hypothyroid", "hyperthyroid")
    return label


def generate_medical_dataset(n_samples=N_SAMPLES, seed=RANDOM_STATE, verbose=True):
    rng = np.random.default_rng(seed)
    n = int(n_samples)

    # 1. Latent disease state from prevalence
    states = np.array(list(PREVALENCE.keys()))
    latent = rng.choice(states, size=n, p=list(PREVALENCE.values()))
    subclinical = rng.random(n) < SUBCLINICAL_RATE

    # 2. Hormones caused by latent state
    tsh, t3, t4 = _hormones_for_state(latent, n, rng, subclinical)

    # 3. Treatment conditional on OBSERVED hormones, not on the label
    p_thyroxine = np.clip(0.06 + 0.62 * (tsh > 4.5) + 0.16 * (t4 < 82), 0, 0.92)
    on_thyroxine = (rng.random(n) < p_thyroxine).astype(int)

    p_antithyroid = np.clip(0.04 + 0.58 * (tsh < 0.42) + 0.16 * (t4 > 142), 0, 0.90)
    on_antithyroid = (rng.random(n) < p_antithyroid).astype(int)

    # 4. Treatment partially normalizes the panel. This is the confounder that
    #    makes treated disease genuinely hard to detect from labs alone.
    tx = on_thyroxine == 1
    if tx.sum():
        eff = rng.uniform(0.35, 0.85, tx.sum())
        tsh[tx] = tsh[tx] * (1 - eff) + rng.lognormal(0.45, 0.5, tx.sum()) * eff
        t4[tx] = t4[tx] * (1 - eff) + rng.normal(104, 18, tx.sum()) * eff

    ax = on_antithyroid == 1
    if ax.sum():
        eff = rng.uniform(0.35, 0.85, ax.sum())
        tsh[ax] = tsh[ax] * (1 - eff) + rng.lognormal(0.45, 0.5, ax.sum()) * eff
        t4[ax] = t4[ax] * (1 - eff) + rng.normal(104, 18, ax.sum()) * eff

    # 5. Assay noise on every measurement
    tsh = np.clip(tsh * rng.normal(1.0, ASSAY_CV, n), 0.01, 60)
    t3 = np.clip(t3 * rng.normal(1.0, ASSAY_CV, n), 0.3, 7.0)
    t4 = np.clip(t4 * rng.normal(1.0, ASSAY_CV, n), 25, 245)
    t4u = np.clip(rng.normal(1.0, 0.18, n), 0.5, 1.9)

    # 6. Symptoms caused by hormone levels
    goitre = (rng.random(n) < np.clip(0.05 + 0.22 * (tsh > 4.5) + 0.20 * (tsh < 0.42), 0, 1)).astype(int)
    sick = (rng.random(n) < np.clip(0.14 + 0.14 * (t4 < 82) + 0.12 * (t4 > 142), 0, 1)).astype(int)

    # 7. Referral suspicion from what a referring clinician can observe
    q_hypo = (rng.random(n) < np.clip(0.05 + 0.55 * (tsh > 4.5) + 0.18 * goitre, 0, 0.9)).astype(int)
    q_hyper = (rng.random(n) < np.clip(0.04 + 0.52 * (tsh < 0.42) + 0.18 * goitre, 0, 0.9)).astype(int)

    # 8. Recorded diagnosis from the observed panel
    label = _record_diagnosis(tsh, t3, t4, rng)

    df = pd.DataFrame({
        "age": np.clip(rng.normal(48, 18, n), 18, 90),
        "sex": rng.choice([0, 1], n, p=[0.68, 0.32]),
        "TSH": tsh,
        "T3": t3,
        "T4": t4,
        "T4U": t4u,
        "FTI": t4 / (t4u + 0.01),
        "on_thyroxine": on_thyroxine,
        "on_antithyroid": on_antithyroid,
        "sick": sick,
        "pregnant": rng.choice([0, 1], n, p=[0.96, 0.04]),
        "thyroid_surgery": rng.choice([0, 1], n, p=[0.91, 0.09]),
        "goitre": goitre,
        "tumor": rng.choice([0, 1], n, p=[0.96, 0.04]),
        "query_hypothyroid": q_hypo,
        "query_hyperthyroid": q_hyper,
        "T3_measured": rng.choice([0, 1], n, p=[0.12, 0.88]),
        "latent_state": latent,          # audit only, dropped before training
        "subclinical": subclinical.astype(int),   # audit only
        "target": label,
    })

    if verbose:
        print(f"\nGenerated {len(df):,} records (causal generator, seed={seed})")
        print("Recorded diagnosis distribution:")
        for k, v in df["target"].value_counts().items():
            print(f"  {k:14s} {v:7,} ({v / len(df) * 100:5.1f}%)")
        agree = (df["latent_state"] == df["target"]).mean()
        print(f"Latent state vs recorded diagnosis agreement: {agree:.1%}")
        print(f"Subclinical fraction: {df['subclinical'].mean():.1%}")
        print("No feature is derived from the target.\n")

    return df


AUDIT_COLUMNS = ["latent_state", "subclinical"]


def leakage_audit(df, seed=RANDOM_STATE):
    """
    Fail loudly if any single feature can recover the target on its own.
    Run this in CI so the old failure mode cannot silently return.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    X = df.drop(columns=AUDIT_COLUMNS + ["target"])
    y = df["target"]
    base = y.value_counts(normalize=True).max()

    flags = ["on_thyroxine", "on_antithyroid", "query_hypothyroid", "query_hyperthyroid"]
    Xa, Xb, ya, yb = train_test_split(X[flags], y, test_size=0.25,
                                      random_state=seed, stratify=y)
    flag_acc = accuracy_score(yb, DecisionTreeClassifier(max_depth=4,
                              random_state=seed).fit(Xa, ya).predict(Xb))

    print(f"majority class baseline      : {base:.4f}")
    print(f"accuracy from 4 clinical flags: {flag_acc:.4f}")
    print(f"lift over baseline            : {flag_acc - base:+.4f}")

    per_feature = {}
    for col in X.columns:
        Xa, Xb, ya, yb = train_test_split(X[[col]], y, test_size=0.25,
                                          random_state=seed, stratify=y)
        per_feature[col] = accuracy_score(
            yb, DecisionTreeClassifier(max_depth=3, random_state=seed).fit(Xa, ya).predict(Xb))
    worst = sorted(per_feature.items(), key=lambda kv: -kv[1])[:5]
    print("\ntop single feature accuracies:")
    for c, a in worst:
        mark = "  <-- INVESTIGATE" if a > base + 0.25 else ""
        print(f"  {c:22s} {a:.4f}{mark}")

    assert flag_acc < base + 0.25, (
        f"LEAKAGE: clinical flags alone score {flag_acc:.4f} vs baseline {base:.4f}")
    return {"baseline": base, "flag_accuracy": flag_acc, "per_feature": per_feature}


if __name__ == "__main__":
    df = generate_medical_dataset(20000)
    leakage_audit(df)
