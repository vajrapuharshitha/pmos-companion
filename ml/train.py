"""Leakage-aware PMOS/PCOS screening model.
Usage: python ml/train.py data/pcos.csv
Runs TWO experiments and reports both honestly:
  full           - all features (labels are derived from diagnostic criteria, so scores are inflated)
  criteria_blind - drops direct criteria proxies (cycle, follicles, hair growth, acne, AMH...) to test
                   whether the model finds signal beyond the diagnostic rules themselves
Outputs: ml/metrics.json, ml/model_<name>.pkl, ml/shap_<name>.png (if shap installed)."""
import sys, json, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score, confusion_matrix

SEED = 42
TARGET_HINT = "pcos"
PROXY_WORDS = ["cycle", "follicle", "avg. f size", "hair growth", "pimples", "skin darkening", "hair loss", "amh"]
ID_WORDS = ["sl. no", "patient file", "unnamed"]

def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    y_col = next(c for c in df.columns if TARGET_HINT in c.lower() and "y/n" in c.lower())
    df = df[[c for c in df.columns if not any(w in c.lower() for w in ID_WORDS)]]
    y = df.pop(y_col).astype(int)
    X = df.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    return X, y

def boot_ci(y, p, fn, n=1000):
    rng, vals = np.random.default_rng(SEED), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y.iloc[i])) < 2: continue
        vals.append(fn(y.iloc[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def run(name, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    models = {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gboost": make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=SEED)),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    cvauc = {k: float(cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean()) for k, m in models.items()}
    best = max(cvauc, key=cvauc.get)
    model = models[best].fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn_, tp = confusion_matrix(yte, pred).ravel()
    res = {"n_features": X.shape[1], "n_train": len(Xtr), "n_test": len(Xte), "cv_auc": cvauc, "best_model": best,
           "test_auc": float(roc_auc_score(yte, p)), "test_auc_95ci": boot_ci(yte, p, roc_auc_score),
           "test_pr_auc": float(average_precision_score(yte, p)), "brier": float(brier_score_loss(yte, p)),
           "sensitivity": float(recall_score(yte, pred)), "specificity": float(tn / (tn + fp)),
           "confusion": [[int(tn), int(fp)], [int(fn_), int(tp)]]}
    bmi = next((c for c in X.columns if c.lower() == "bmi"), None)
    if bmi:
        lean = (Xte[bmi] <= 25).values
        for lab, m in (("lean_bmi<=25", lean), ("bmi>25", ~lean)):
            if m.sum() > 10 and len(set(yte[m])) > 1:
                res["auc_" + lab] = float(roc_auc_score(yte[m], p[m]))
    pickle.dump({"model": model, "features": list(X.columns)}, open(f"ml/model_{name}.pkl", "wb"))
    try:
        import shap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Xs = pd.DataFrame(model[:-1].transform(Xte), columns=X.columns)
        clf = model[-1]
        ex = shap.TreeExplainer(clf) if best == "gboost" else shap.LinearExplainer(clf, Xs)
        shap.summary_plot(ex.shap_values(Xs), Xs, show=False); plt.tight_layout()
        plt.savefig(f"ml/shap_{name}.png", dpi=150); plt.close()
    except Exception as e:
        res["shap"] = f"skipped: {e}"
    return res

if __name__ == "__main__":
    X, y = load(sys.argv[1])
    blind = X[[c for c in X.columns if not any(w in c.lower() for w in PROXY_WORDS)]]
    out = {"class_balance": {str(k): int(v) for k, v in y.value_counts().items()}, "full": run("full", X, y), "criteria_blind": run("criteria_blind", blind, y)}
    json.dump(out, open("ml/metrics.json", "w"), indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
