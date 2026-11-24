"""
Real-data injury prediction: Lövdal et al. (IJSPP 2021) competitive-runner dataset
(74 athletes, 2012-2019; DataverseNL doi:10.34894/UWU9PV).

Protocols
  P1  Original hold-out: train on first 64 athletes to join, test on last 10 (reproduction)
  P2  Athlete-grouped stratified 5-fold CV x 3 repeats on all 74 athletes (primary)
  P3  Random event-level stratified 5-fold CV (leaky reference)
Models: ACWR rule, logistic regression, random forest, XGBoost, bagged XGBoost (Lövdal),
        GRU, CNN-GRU (day approach, 7 x 10 sequence).
Usage: python run_runners.py --data DIR_WITH_CSVs --out OUTDIR
"""
import argparse, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from xgboost import XGBClassifier
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); torch.set_num_threads(4)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../data/runners"); ap.add_argument("--out", default="../output/study3")
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args(); RES, FIG = f"{a.out}/results", f"{a.out}/figures"
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]; INK2 = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "savefig.dpi": 300, "savefig.bbox": "tight"})

def load(name):
    d = pd.read_csv(f"{a.data}/{name}_approach_maskedID_timeseries.csv")
    feats = [c for c in d.columns if c not in ("Athlete ID", "injury", "Date")]
    X = d[feats].astype(float).copy()
    # per-athlete z-score (label-free: all of the athlete's events), as in the original study's normalisation idea
    g = d["Athlete ID"]
    mu = X.groupby(g).transform("mean"); sd = X.groupby(g).transform("std").replace(0, 1).fillna(1)
    Xz = ((X - mu) / sd).fillna(0).values.astype(np.float32)
    return d, feats, Xz, X.values.astype(np.float32), d["injury"].values.astype(int), g.values

D, FD, XD, XD_raw, YD, GD = load("day")
W, FW, XW, XW_raw, YW, GW = load("week")
join = D.groupby("Athlete ID").Date.min().sort_values()
TEST_ATH = join.index[-10:].values
audit = dict(day_shape=D.shape, week_shape=W.shape, athletes=int(D["Athlete ID"].nunique()),
             day_injuries=int(YD.sum()), week_injuries=int(YW.sum()), prevalence_day=float(YD.mean()),
             test_athletes_P1=TEST_ATH.tolist(), test_injuries_P1_day=int(YD[np.isin(GD, TEST_ATH)].sum()),
             test_events_P1_day=int(np.isin(GD, TEST_ATH).sum()),
             injuries_per_athlete=D.groupby("Athlete ID").injury.sum().describe().round(2).to_dict())
json.dump(audit, open(f"{RES}/runner_audit.json", "w"), indent=2, default=int); print(audit)

# ----------------------------- models -----------------------------
class GRUNet(nn.Module):
    def __init__(s, f, cnn=False):
        super().__init__(); s.cnn = cnn
        if cnn: s.c = nn.Sequential(nn.Conv1d(f, 32, 3, padding=1), nn.ReLU()); f = 32
        s.r = nn.GRU(f, 32, batch_first=True); s.h = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(s, x):
        if s.cnn: x = s.c(x.transpose(1, 2)).transpose(1, 2)
        return s.h(s.r(x)[1][-1]).squeeze(-1)

def to_seq(X):  # day approach: columns are [10 feats day0][10 feats day1]...[day6]; order oldest -> newest
    return X.reshape(len(X), 7, 10)[:, ::-1, :].copy()

def fit_dl(Xtr, ytr, Xte, seed, cnn=False, epochs=12):
    torch.manual_seed(seed); np.random.seed(seed)
    m = GRUNet(10, cnn); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    Xt = torch.tensor(to_seq(Xtr)); yt = torch.tensor(ytr, dtype=torch.float32); lf = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    for _ in range(epochs):  # balanced mini-batches (undersample negatives each epoch)
        idx = np.concatenate([pos, rng.choice(neg, len(pos) * 4, replace=False)]); rng.shuffle(idx)
        m.train()
        for b in range(0, len(idx), 128):
            i = idx[b:b + 128]; opt.zero_grad(); lf(m(Xt[i]), yt[i]).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(to_seq(Xte)))).numpy()

def bagged_xgb(Xtr, ytr, gtr, Xte, seed, n_bags=9, n_each=2048):
    rng = np.random.RandomState(seed); pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]; P = []
    for b in range(n_bags):
        idx = np.concatenate([rng.choice(pos, n_each, True), rng.choice(neg, n_each, True)])
        m = XGBClassifier(n_estimators=512, max_depth=3, learning_rate=0.01, random_state=seed * 100 + b, verbosity=0, n_jobs=4)
        m.fit(Xtr[idx], ytr[idx]); P.append(m.predict_proba(Xte)[:, 1])
    return np.mean(P, 0)

def predict(model, Xtr, ytr, gtr, Xte, seed, approach):
    if model == "Logistic regression":
        return LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if model == "Random forest":
        return RandomForestClassifier(300, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=4,
                                      random_state=seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if model == "XGBoost":
        return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), random_state=seed, verbosity=0,
                             n_jobs=4).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if model == "Bagged XGBoost [Lövdal]": return bagged_xgb(Xtr, ytr, gtr, Xte, seed)
    if model == "GRU": return fit_dl(Xtr, ytr, Xte, seed)
    if model == "CNN-GRU": return fit_dl(Xtr, ytr, Xte, seed, cnn=True)

def rule_score(approach, Xraw, feats):
    if approach == "week":  # ACWR-like: acute (week 0) / chronic (mean of weeks 0-2) total km
        k = [feats.index(c) for c in ["total kms", "total kms.1", "total kms.2"]]
        chronic = Xraw[:, k].mean(1); return np.where(chronic > 0, Xraw[:, k[0]] / np.maximum(chronic, 1e-6), 0)
    k = [feats.index(f"total km.{i}") if i else feats.index("total km") for i in range(7)]  # 7-day load spike
    return Xraw[:, k[0]] - Xraw[:, k].mean(1)

def threshold_from_train(Xtr, ytr, gtr, model, seed, approach):
    """Pick a threshold on an inner athlete-held-out split of the training data (Youden J)."""
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
    i_tr, i_va = next(gss.split(Xtr, ytr, gtr))
    if ytr[i_va].sum() < 5: return 0.5
    p = predict(model, Xtr[i_tr], ytr[i_tr], gtr[i_tr], Xtr[i_va], seed, approach)
    f, t, th = roc_curve(ytr[i_va], p); return th[np.argmax(t - f)]

MODELS_DAY = ["Logistic regression", "Random forest", "XGBoost", "Bagged XGBoost [Lövdal]", "GRU", "CNN-GRU"]
MODELS_WEEK = ["Logistic regression", "Random forest", "XGBoost", "Bagged XGBoost [Lövdal]"]

def evaluate(approach, protocol, models, repeats, with_threshold=False):
    X, Xr, y, g, feats = (XD, XD_raw, YD, GD, FD) if approach == "day" else (XW, XW_raw, YW, GW, FW)
    rows, oof_store = [], {}
    for rep in range(repeats):
        if protocol == "P1":
            te = np.isin(g, TEST_ATH); splits = [(np.where(~te)[0], np.where(te)[0])]
        elif protocol == "P2":
            splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g))
        else:
            splits = list(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y))
        for m in ["Load-spike rule (ACWR-type)"] + models:
            oof = np.full(len(y), np.nan); pred = np.full(len(y), np.nan)
            for tr, te in splits:
                if m.startswith("Load-spike"):
                    oof[te] = rule_score(approach, Xr[te], feats); continue
                oof[te] = predict(m, X[tr], y[tr], g[tr], X[te], rep, approach)
                if with_threshold:
                    th = threshold_from_train(X[tr], y[tr], g[tr], m, rep, approach); pred[te] = oof[te] >= th
            mask = ~np.isnan(oof)
            r = dict(approach=approach, protocol=protocol, repeat=rep, model=m,
                     AUROC=roc_auc_score(y[mask], oof[mask]), AUPRC=average_precision_score(y[mask], oof[mask]),
                     prevalence=y[mask].mean())
            if with_threshold and not m.startswith("Load-spike"):
                pm = pred[mask].astype(bool); yt = y[mask]
                r.update(sensitivity=pm[yt == 1].mean(), specificity=(~pm[yt == 0]).mean())
            rows.append(r); oof_store[(m, rep)] = oof
            print(approach, protocol, rep, m, round(r["AUROC"], 3), flush=True)
    return pd.DataFrame(rows), oof_store

t0 = time.time(); allres = []; store = {}
for appr, models in [("day", MODELS_DAY), ("week", MODELS_WEEK)]:
    r1, _ = evaluate(appr, "P1", models, 5, with_threshold=True); r1.to_csv(f"{RES}/partial_P1_{appr}.csv", index=False); allres.append(r1)
    r2, s2 = evaluate(appr, "P2", models, a.repeats, with_threshold=False); allres.append(r2); store[appr] = s2; r2.to_csv(f"{RES}/partial_P2_{appr}.csv", index=False)
    r3, _ = evaluate(appr, "P3", [m for m in models if m in ("Logistic regression", "XGBoost", "GRU")], 2); allres.append(r3)
    print(f"[{time.time()-t0:.0f}s]")
R = pd.concat(allres); R.to_csv(f"{RES}/runner_all_runs.csv", index=False)
S = R.groupby(["approach", "protocol", "model"])[["AUROC", "AUPRC", "sensitivity", "specificity"]].agg(["mean", "std"]).round(4)
S.to_csv(f"{RES}/runner_summary.csv"); print(S.to_string())

# -------- athlete-level bootstrap CI (P2, repeat 0) and paired differences vs bagged XGBoost --------
rng = np.random.RandomState(0); ath = np.unique(GD); boot = {}
ref = "Bagged XGBoost [Lövdal]"
for appr, y, g in [("day", YD, GD), ("week", YW, GW)]:
    models = ["Load-spike rule (ACWR-type)"] + (MODELS_DAY if appr == "day" else MODELS_WEEK)
    idx_by = {k: np.where(g == k)[0] for k in ath}; B = {m: [] for m in models}; Dd = {m: [] for m in models}
    for b in range(1000):
        samp = np.concatenate([idx_by[k] for k in rng.choice(ath, len(ath), True)])
        if y[samp].sum() == 0: continue
        aref = roc_auc_score(y[samp], store[appr][(ref, 0)][samp])
        for m in models:
            v = roc_auc_score(y[samp], store[appr][(m, 0)][samp]); B[m].append(v); Dd[m].append(v - aref)
    for m in models:
        boot[(appr, m)] = dict(AUROC_pooled=roc_auc_score(y, store[appr][(m, 0)]),
                               CI_low=np.percentile(B[m], 2.5), CI_high=np.percentile(B[m], 97.5),
                               diff_vs_bagged=np.mean(Dd[m]), diff_CI_low=np.percentile(Dd[m], 2.5), diff_CI_high=np.percentile(Dd[m], 97.5))
BT = pd.DataFrame(boot).T.round(4); BT.to_csv(f"{RES}/runner_bootstrap_P2.csv"); print(BT.to_string())

# -------- per-athlete AUROC heterogeneity (P2 pooled, bagged XGB, day) --------
pa = []
for k in ath:
    i = GD == k
    if YD[i].sum() >= 3 and (YD[i] == 0).sum() >= 3: pa.append(dict(athlete=k, n_inj=int(YD[i].sum()), AUROC=roc_auc_score(YD[i], store["day"][(ref, 0)][i])))
PA = pd.DataFrame(pa); PA.to_csv(f"{RES}/runner_per_athlete_auroc.csv", index=False)

# -------- SHAP (XGBoost, day, all data) --------
import shap
mx = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                   scale_pos_weight=(YD == 0).sum() / YD.sum(), random_state=0, verbosity=0, n_jobs=4).fit(XD, YD)
sub = np.random.RandomState(0).choice(len(XD), 8000, replace=False)
sv = shap.TreeExplainer(mx).shap_values(XD[sub])
imp = pd.Series(np.abs(sv).mean(0), index=FD)
base = imp.groupby(lambda c: c.split(".")[0]).sum().sort_values(ascending=False)
lag = imp.groupby(lambda c: int(c.split(".")[1]) if "." in c else 0).sum()
base.to_csv(f"{RES}/runner_shap_by_variable.csv"); lag.to_csv(f"{RES}/runner_shap_by_day_lag.csv")

# -------- figures --------
# ROC curves (P2 repeat 0, day)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
for k, (appr, y) in enumerate([("day", YD), ("week", YW)]):
    models = ["Load-spike rule (ACWR-type)"] + (MODELS_DAY if appr == "day" else MODELS_WEEK)
    for j, m in enumerate(models):
        f, t, _ = roc_curve(y, store[appr][(m, 0)])
        ax[k].plot(f, t, color=C[j], lw=1.4 if "Bagged" in m or "GRU" in m else 1.0,
                   label=f"{m.replace(' [Lövdal]','')} ({roc_auc_score(y, store[appr][(m, 0)]):.3f})")
    ax[k].plot([0, 1], [0, 1], ls=":", color=INK2, lw=0.8); ax[k].set_title(f"{appr.capitalize()} approach", fontsize=9)
    ax[k].set_xlabel("1 − specificity"); ax[k].set_ylabel("Sensitivity"); ax[k].legend(frameon=False, fontsize=6.3, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/figR1_roc.png"); plt.close()

# protocol comparison
pm = R[R.model.isin(["Logistic regression", "XGBoost", "GRU"])].groupby(["approach", "protocol", "model"]).AUROC.mean().reset_index()
fig, ax = plt.subplots(figsize=(7.2, 2.9)); labs = []; x = 0
for appr in ["day", "week"]:
    for m in ["Logistic regression", "XGBoost", "GRU"]:
        sub_ = pm[(pm.approach == appr) & (pm.model == m)]
        if sub_.empty: continue
        for j, (p, lab) in enumerate([("P3", "Random event split (leaky)"), ("P2", "Athlete-grouped CV"), ("P1", "Original 64/10 hold-out")]):
            v = sub_[sub_.protocol == p].AUROC
            if len(v): ax.bar(x + (j - 1) * 0.27, v.values[0], 0.25, color=[C[1], C[0], C[2]][j], label=lab if x == 0 else None)
            if len(v): ax.text(x + (j - 1) * 0.27, v.values[0] + 0.01, f"{v.values[0]:.2f}", ha="center", fontsize=6.3)
        labs.append(f"{m}\n({appr})"); x += 1
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=7); ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUROC")
ax.axhline(0.5, color=INK2, ls=":", lw=0.8); ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{FIG}/figR2_protocol.png"); plt.close()

# SHAP by variable and by lag
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
v = base.sort_values(); ax[0].barh(range(len(v)), v.values, color=C[0], height=0.6); ax[0].set_yticks(range(len(v))); ax[0].set_yticklabels(v.index, fontsize=7)
ax[0].set_xlabel("Σ mean |SHAP| over 7 days")
ax[1].bar(lag.index, lag.values, color=C[0], width=0.6); ax[1].set_xlabel("Days before event (0 = previous day)"); ax[1].set_ylabel("Σ mean |SHAP|")
fig.tight_layout(); fig.savefig(f"{FIG}/figR3_shap.png"); plt.close()
print("DONE", time.time() - t0)
