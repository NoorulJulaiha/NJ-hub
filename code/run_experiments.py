"""
Experiments for: Multimodal wearable sensing for sprint-phase recognition and
physiological-strain anomaly detection (chest + wrist band).

Inputs : sprinter_synthetic_5000.csv (raw stream, 100 Hz)
         overall_ranges.csv          (channel min/max, used only for audit)
         normalized_features.csv     (prior feature table, used only for audit)
Outputs: results/*.csv, figures/*.png

Usage  : python run_experiments.py --data DIR --out OUTDIR
Colab  : !pip install xgboost shap torch ; then run the same command.
"""
import argparse, json, os, warnings, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import (f1_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, roc_auc_score, average_precision_score)
from xgboost import XGBClassifier
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../data")
ap.add_argument("--out", default="../output/study1")
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()
OUT = args.out; FIG = f"{OUT}/figures"; RES = f"{OUT}/results"
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

# ---------------- palette (validated categorical order) ----------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

def find(prefix):
    return [os.path.join(args.data, f) for f in os.listdir(args.data) if prefix in f][0]

# ======================= 0. LOAD & AUDIT =======================
raw = pd.read_csv(find("sprinter_synthetic"))
CH = list(raw.columns[3:16])
FS = 100
PHASES = ["Warmup", "Acceleration", "PeakSprint", "Deceleration", "Recovery"]
raw["phase_id"] = raw.phase_label.map({p: i for i, p in enumerate(PHASES)})
audit = {
    "n_samples": len(raw), "fs_hz": FS, "duration_s": len(raw) / FS,
    "sessions": raw.session_id.nunique(), "missing": int(raw.isna().sum().sum()),
    "phase_counts": raw.phase_label.value_counts().to_dict(),
    "anomaly_samples": int(raw.anomaly_flag.sum()),
    "anomaly_runs": int(((raw.anomaly_flag.diff() == 1)).sum() + raw.anomaly_flag.iloc[0]),
    "anomaly_start_idx": int(raw.index[raw.anomaly_flag == 1][0]),
    "anomaly_end_idx": int(raw.index[raw.anomaly_flag == 1][-1]),
    "r_wristEMG_chestEMG": float(raw.wrist_emg.corr(raw.chest_emg)),
}
rng_tab = pd.read_csv(find("overall_ranges"), index_col=0)
audit["ranges_match_raw"] = bool(np.allclose(rng_tab.loc["min", CH], raw[CH].min()) and
                                 np.allclose(rng_tab.loc["max", CH], raw[CH].max()))

# Reproduce the prior normalized_features table (5 s windows, 2.5 s hop)
nf = pd.read_csv(find("normalized_features"))
W5, H5 = 500, 250
rows = []
for s in range(0, len(raw) - W5 + 1, H5):
    w = raw[CH].iloc[s:s + W5]; t = np.arange(W5); f = {}
    for c in CH:
        f[c + "_mean"] = w[c].mean(); f[c + "_std"] = w[c].std(ddof=0)
        f[c + "_slope"] = np.polyfit(t, w[c].values, 1)[0]
    rows.append(f)
f5 = pd.DataFrame(rows)
buggy = f5.copy()
for c in CH:
    mn, mx = rng_tab.loc["min", c], rng_tab.loc["max", c]
    for k in ["_mean", "_std", "_slope"]:
        buggy[c + k] = (f5[c + k] - mn) / (mx - mn)
audit["prior_table_reproduced_maxabs_err"] = float((buggy[nf.columns] - nf).abs().max().max())
audit["prior_table_n_negative_values"] = int((nf < 0).sum().sum())
corrected = pd.DataFrame(StandardScaler().fit_transform(f5), columns=f5.columns)
corrected.to_csv(f"{RES}/normalized_features_CORRECTED_zscore.csv", index=False)
json.dump(audit, open(f"{RES}/data_audit.json", "w"), indent=2, default=str)
print(json.dumps(audit, indent=2, default=str))

# ======================= 1. CORRELATION & STATISTICS =======================
pear = raw[CH].corr("pearson"); spear = raw[CH].corr("spearman")
pear.to_csv(f"{RES}/corr_pearson.csv"); spear.to_csv(f"{RES}/corr_spearman.csv")

# Kruskal-Wallis across phases with epsilon^2 effect size
kw = []
for c in CH:
    g = [raw.loc[raw.phase_label == p, c].values for p in PHASES]
    H, p = stats.kruskal(*g)
    eps2 = (H - len(g) + 1) / (len(raw) - len(g))
    kw.append([c, H, p, eps2])
kw = pd.DataFrame(kw, columns=["channel", "H", "p", "epsilon2"]).sort_values("epsilon2", ascending=False)

# Anomaly association within PeakSprint only (removes phase confounding)
pk = raw[raw.phase_label == "PeakSprint"]
an = []
for c in CH:
    a, b = pk.loc[pk.anomaly_flag == 1, c], pk.loc[pk.anomaly_flag == 0, c]
    rpb = stats.pointbiserialr(pk.anomaly_flag, pk[c])[0]
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    U, p = stats.mannwhitneyu(a, b)
    an.append([c, rpb, d, p, a.mean(), b.mean()])
an = pd.DataFrame(an, columns=["channel", "r_pointbiserial", "cohens_d", "p_MWU",
                               "mean_anomaly", "mean_normal"]).sort_values("cohens_d", key=abs, ascending=False)
# effective sample size note: autocorrelation inflates significance
lag1 = {c: raw[c].autocorr(1) for c in CH}
kw["lag1_autocorr"] = kw.channel.map(lag1)
kw.to_csv(f"{RES}/phase_kruskal.csv", index=False); an.to_csv(f"{RES}/anomaly_association_peaksprint.csv", index=False)
raw.groupby("phase_label")[CH].agg(["mean", "std"]).round(3).T.to_csv(f"{RES}/phase_descriptives.csv")

# ======================= 2. WINDOWING & CV PROTOCOLS =======================
WIN, HOP = 100, 10   # 1 s windows, 0.1 s hop; label = last sample (causal)
X_raw = raw[CH].values.astype(np.float32)
ends = np.arange(WIN - 1, len(raw), HOP)
starts = ends - WIN + 1
y_phase = raw.phase_id.values[ends]
y_anom = raw.anomaly_flag.values[ends]

def feats(seq):
    t = np.arange(seq.shape[0]); tc = t - t.mean()
    slope = (tc[:, None] * (seq - seq.mean(0))).sum(0) / (tc ** 2).sum()
    return np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0), slope,
                           np.sqrt((seq ** 2).mean(0))])
FN = [f"{c}_{k}" for k in ["mean", "std", "min", "max", "slope", "rms"] for c in CH]
F = np.stack([feats(X_raw[s:e + 1]) for s, e in zip(starts, ends)])
SEQ = np.stack([X_raw[s:e + 1] for s, e in zip(starts, ends)])

# Segment = contiguous run of (phase, anomaly). Segment-stratified blocked CV:
seg_id = ((raw.phase_id.diff() != 0) | (raw.anomaly_flag.diff() != 0)).cumsum().values
K = 5
chunk = np.zeros(len(raw), int)
for s in np.unique(seg_id):
    idx = np.where(seg_id == s)[0]
    if len(idx) < K * 50:            # tiny segment (<2.5 s): keep whole in one fold
        chunk[idx] = s % K; continue
    for k, part in enumerate(np.array_split(idx, K)):
        chunk[part] = k
win_fold = chunk[ends]

def blocked_splits():
    for k in range(K):
        te = np.where(win_fold == k)[0]
        test_samples = np.zeros(len(raw), bool)
        test_samples[np.isin(chunk, [k])] = True
        # purge: training windows must not overlap any test sample; plus 0.5 s embargo
        EMB = 50
        guard = np.convolve(test_samples.astype(int), np.ones(2 * EMB + 1), "same") > 0
        tr = np.array([i for i in range(len(ends)) if win_fold[i] != k
                       and not guard[starts[i]:ends[i] + 1].any()])
        yield tr, te

def random_splits(seed=0):
    r = np.random.RandomState(seed); perm = r.permutation(len(ends))
    for k in range(K):
        te = perm[k::K]; tr = np.setdiff1d(perm, te); yield tr, te

# ======================= 3. MODELS =======================
class CNN(nn.Module):
    def __init__(s, c, o):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(c, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.h = nn.Linear(32, o)
    def forward(s, x): return s.h(s.f(x.transpose(1, 2)).squeeze(-1))
class GRU(nn.Module):
    def __init__(s, c, o):
        super().__init__(); s.r = nn.GRU(c, 32, batch_first=True); s.h = nn.Linear(32, o)
    def forward(s, x): return s.h(s.r(x)[1][-1])
class CNNGRU(nn.Module):
    def __init__(s, c, o):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(c, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        s.r = nn.GRU(32, 32, batch_first=True); s.h = nn.Linear(32, o)
    def forward(s, x): return s.h(s.r(s.c(x.transpose(1, 2)).transpose(1, 2))[1][-1])
class GRUAE(nn.Module):
    def __init__(s, c, z=8):
        super().__init__(); s.e = nn.GRU(c, z, batch_first=True); s.d = nn.GRU(z, 32, batch_first=True); s.o = nn.Linear(32, c)
    def forward(s, x):
        h = s.e(x)[1][-1]; return s.o(s.d(h.unsqueeze(1).repeat(1, x.shape[1], 1))[0])

def norm_seq(tr, te, S):
    mu = S[tr].reshape(-1, S.shape[2]).mean(0); sd = S[tr].reshape(-1, S.shape[2]).std(0) + 1e-6
    return (S[tr] - mu) / sd, (S[te] - mu) / sd

def train_clf(Model, Xtr, ytr, Xte, n_cls, seed, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(Xtr.shape[2], n_cls); opt = torch.optim.Adam(m.parameters(), 3e-3)
    w = torch.tensor(len(ytr) / (n_cls * np.maximum(np.bincount(ytr, minlength=n_cls), 1)), dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w); Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    for _ in range(epochs):
        m.train(); p = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 64):
            i = p[b:b + 64]; opt.zero_grad(); lf(m(Xt[i]), yt[i]).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return torch.softmax(m(torch.tensor(Xte)), 1).numpy()

def classical(name, seed):
    if name == "LogReg": return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "RandomForest": return RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "XGBoost": return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=seed, verbosity=0)

# ======================= 4. TASK A: SPRINT-PHASE RECOGNITION =======================
def run_phase(protocol, seeds, ch_idx=None, models=None, noise=0.0):
    models = models or ["LogReg", "RandomForest", "XGBoost", "1D-CNN", "GRU", "CNN-GRU"]
    ci = ch_idx if ch_idx is not None else list(range(len(CH)))
    fcols = [j for j, n in enumerate(FN) if CH.index(n.rsplit("_", 1)[0]) in ci]
    out = []; preds = {}
    for m in models:
        for sd in seeds:
            yt_all, yp_all = [], []
            spl = blocked_splits() if protocol == "blocked" else random_splits(sd)
            for tr, te in spl:
                if m in ["LogReg", "RandomForest", "XGBoost"]:
                    Ftr, Fte = F[tr][:, fcols], F[te][:, fcols]
                    if noise:
                        Fte = np.stack([feats(SEQ[i][:, ci] + np.random.RandomState(sd + i).normal(0, noise, (WIN, len(ci))) * X_raw[:, ci].std(0)) for i in te])
                    sc = StandardScaler().fit(Ftr); clf = classical(m, sd)
                    sw = None
                    if m == "XGBoost":
                        cw = len(tr) / (5 * np.bincount(y_phase[tr], minlength=5)); sw = cw[y_phase[tr]]
                    clf.fit(sc.transform(Ftr), y_phase[tr], **({"sample_weight": sw} if sw is not None else {}))
                    yp = clf.predict(sc.transform(Fte))
                else:
                    S = SEQ[:, :, ci].copy()
                    if noise:
                        S[te] = S[te] + np.random.RandomState(sd).normal(0, noise, S[te].shape).astype(np.float32) * X_raw[:, ci].std(0)
                    a, b = norm_seq(tr, te, S)
                    Mod = {"1D-CNN": CNN, "GRU": GRU, "CNN-GRU": CNNGRU}[m]
                    yp = train_clf(Mod, a.astype(np.float32), y_phase[tr], b.astype(np.float32), 5, sd).argmax(1)
                yt_all += list(y_phase[te]); yp_all += list(yp)
            yt_all, yp_all = np.array(yt_all), np.array(yp_all)
            out.append(dict(model=m, seed=sd, protocol=protocol,
                            macroF1=f1_score(yt_all, yp_all, average="macro"),
                            balAcc=balanced_accuracy_score(yt_all, yp_all),
                            kappa=cohen_kappa_score(yt_all, yp_all)))
            preds[(m, sd)] = (yt_all, yp_all)
    return pd.DataFrame(out), preds

seeds = list(range(args.seeds))
t0 = time.time()
pa_b, preds_b = run_phase("blocked", seeds)
pa_r, _ = run_phase("random", seeds)
phaseA = pd.concat([pa_b, pa_r])
phaseA.to_csv(f"{RES}/taskA_phase_all_runs.csv", index=False)
summA = phaseA.groupby(["protocol", "model"])[["macroF1", "balAcc", "kappa"]].agg(["mean", "std"]).round(4)
summA.to_csv(f"{RES}/taskA_phase_summary.csv"); print(summA, f"\n[{time.time()-t0:.0f}s]")

# Modality ablation for phase recognition (blocked, best classical + CNN-GRU)
MOD = {"IMU (accel+gyro)": [c for c in CH if "accel" in c or "gyro" in c],
       "EMG (wrist+chest)": ["wrist_emg", "chest_emg"], "Cardio (HR+SpO2+ECG)": ["wrist_hr_bpm", "wrist_spo2_pct", "chest_ecg"],
       "GSR": ["chest_gsr"], "EEG": ["chest_eeg"],
       "Wrist band only": ["wrist_emg", "wrist_hr_bpm", "wrist_spo2_pct"],
       "Chest band only": [c for c in CH if c.startswith("chest")], "All 13 channels": CH}
abl = []
for name, chs in MOD.items():
    r, _ = run_phase("blocked", seeds, [CH.index(c) for c in chs], ["XGBoost", "CNN-GRU"])
    r["modality"] = name; abl.append(r)
ablA = pd.concat(abl); ablA.to_csv(f"{RES}/taskA_modality_ablation_runs.csv", index=False)
ablA_s = ablA.groupby(["modality", "model"]).macroF1.agg(["mean", "std"]).round(4).unstack()
ablA_s.to_csv(f"{RES}/taskA_modality_ablation.csv"); print(ablA_s)

# Noise robustness
nz = []
for lvl in [0.0, 0.1, 0.25, 0.5, 1.0]:
    r, _ = run_phase("blocked", seeds, None, ["XGBoost", "CNN-GRU"], noise=lvl); r["noise_sigma_x_channel_sd"] = lvl; nz.append(r)
nzA = pd.concat(nz); nzA_s = nzA.groupby(["noise_sigma_x_channel_sd", "model"]).macroF1.agg(["mean", "std"]).round(4).unstack()
nzA_s.to_csv(f"{RES}/taskA_noise_robustness.csv"); print(nzA_s)

# ======================= 5. TASK B: STRAIN-ANOMALY DETECTION =======================
# Semi-supervised novelty detection: detectors are fitted ONLY on normal windows of training folds.
def event_metrics(score, thr):
    # map window scores back to sample timeline (window end), causal
    alarm = score > thr
    ev = np.where(y_anom == 1)[0]
    first = ev[alarm[ev]] if alarm[ev].any() else None
    latency = (ends[first[0]] - ends[ev[0]]) / FS if first is not None else np.nan
    fa = alarm & (y_anom == 0)
    # count false-alarm episodes (rising edges)
    fa_ep = int(((fa.astype(int)[1:] - fa.astype(int)[:-1]) == 1).sum() + fa[0])
    normal_minutes = (y_anom == 0).sum() * HOP / FS / 60
    return latency, fa_ep / normal_minutes

def run_anomaly(ch_names, seeds, detectors=None, noise=0.0):
    detectors = detectors or ["HR z-score (rule)", "Mahalanobis", "IsolationForest", "OC-SVM", "LOF", "GRU-AE", "XGBoost (supervised)"]
    ci = [CH.index(c) for c in ch_names]
    fcols = [j for j, n in enumerate(FN) if n.rsplit("_", 1)[0] in ch_names]
    out = []; scores_store = {}
    for det in detectors:
        if det == "HR z-score (rule)" and "wrist_hr_bpm" not in ch_names: continue
        for sd in (seeds if det in ["IsolationForest", "GRU-AE", "XGBoost (supervised)"] else [0]):
            score = np.zeros(len(ends)); thr_vec = np.zeros(len(ends))
            for tr, te in blocked_splits():
                trn = tr[y_anom[tr] == 0]
                Fte = F[te][:, fcols]
                if noise:
                    Fte = np.stack([feats(SEQ[i][:, ci] + np.random.RandomState(sd + i).normal(0, noise, (WIN, len(ci))) * X_raw[:, ci].std(0)) for i in te])
                if det == "HR z-score (rule)":
                    j = FN.index("wrist_hr_bpm_mean"); mu, s = F[trn, j].mean(), F[trn, j].std()
                    hrte = Fte[:, fcols.index(j)] if noise else F[te, j]
                    st, ste = (F[trn, j] - mu) / s, (hrte - mu) / s
                elif det == "GRU-AE":
                    S = SEQ[:, :, ci].copy()
                    if noise: S[te] = S[te] + np.random.RandomState(sd).normal(0, noise, S[te].shape).astype(np.float32) * X_raw[:, ci].std(0)
                    a, b = norm_seq(trn, te, S); a, b = a.astype(np.float32), b.astype(np.float32)
                    torch.manual_seed(sd); m = GRUAE(len(ci)); opt = torch.optim.Adam(m.parameters(), 3e-3)
                    Xt = torch.tensor(a)
                    for _ in range(30):
                        p = torch.randperm(len(Xt))
                        for k in range(0, len(Xt), 64):
                            i = p[k:k + 64]; opt.zero_grad(); ((m(Xt[i]) - Xt[i]) ** 2).mean().backward(); opt.step()
                    with torch.no_grad():
                        st = ((m(Xt) - Xt) ** 2).mean((1, 2)).numpy(); tb = torch.tensor(b)
                        ste = ((m(tb) - tb) ** 2).mean((1, 2)).numpy()
                else:
                    sc = StandardScaler().fit(F[trn][:, fcols]); A, B = sc.transform(F[trn][:, fcols]), sc.transform(Fte)
                    if det == "Mahalanobis":
                        cov = EmpiricalCovariance().fit(A); st, ste = cov.mahalanobis(A), cov.mahalanobis(B)
                    elif det == "IsolationForest":
                        m = IsolationForest(n_estimators=300, random_state=sd).fit(A); st, ste = -m.score_samples(A), -m.score_samples(B)
                    elif det == "OC-SVM":
                        m = OneClassSVM(nu=0.01, gamma="scale").fit(A); st, ste = -m.score_samples(A), -m.score_samples(B)
                    elif det == "LOF":
                        m = LocalOutlierFactor(20, novelty=True).fit(A); st, ste = -m.score_samples(A), -m.score_samples(B)
                    elif det == "XGBoost (supervised)":
                        sc2 = StandardScaler().fit(F[tr][:, fcols])
                        pos = max(y_anom[tr].sum(), 1)
                        m = XGBClassifier(n_estimators=200, max_depth=3, scale_pos_weight=(len(tr) - pos) / pos, random_state=sd, verbosity=0)
                        m.fit(sc2.transform(F[tr][:, fcols]), y_anom[tr])
                        st = m.predict_proba(sc2.transform(F[trn][:, fcols]))[:, 1]; ste = m.predict_proba(sc2.transform(Fte))[:, 1]
                # threshold from TRAINING normal scores only (99.5th percentile) -> no test peeking
                score[te] = ste; thr_vec[te] = np.percentile(st, 99.5)
            alarm = score > thr_vec
            lat, far = event_metrics(score - thr_vec, 0)
            out.append(dict(detector=det, seed=sd, AUROC=roc_auc_score(y_anom, score), AUPRC=average_precision_score(y_anom, score),
                            F1=f1_score(y_anom, alarm), recall=alarm[y_anom == 1].mean(), FPR=alarm[y_anom == 0].mean(),
                            latency_s=lat, false_alarms_per_min=far))
            scores_store[(det, sd)] = (score, thr_vec)
    return pd.DataFrame(out), scores_store

anB, scoresB = run_anomaly(CH, seeds)
anB.to_csv(f"{RES}/taskB_anomaly_all_runs.csv", index=False)
summB = anB.groupby("detector")[["AUROC", "AUPRC", "F1", "recall", "FPR", "latency_s", "false_alarms_per_min"]].agg(["mean", "std"]).round(4)
summB.to_csv(f"{RES}/taskB_anomaly_summary.csv"); print(summB)

# Leave-out ablation for anomaly detection: what if the obvious channels are removed?
ABL_B = {"All 13 channels": CH, "Without HR": [c for c in CH if c != "wrist_hr_bpm"],
         "Without HR, EMG": [c for c in CH if c not in ["wrist_hr_bpm", "wrist_emg", "chest_emg"]],
         "Without HR, EMG, GSR": [c for c in CH if c not in ["wrist_hr_bpm", "wrist_emg", "chest_emg", "chest_gsr"]],
         "IMU only": [c for c in CH if "accel" in c or "gyro" in c],
         "Wrist band only": ["wrist_emg", "wrist_hr_bpm", "wrist_spo2_pct"],
         "Chest band only": [c for c in CH if c.startswith("chest")]}
ab = []
for name, chs in ABL_B.items():
    r, _ = run_anomaly(chs, seeds, ["Mahalanobis", "IsolationForest", "GRU-AE"]); r["channels"] = name; ab.append(r)
ablB = pd.concat(ab); ablB.to_csv(f"{RES}/taskB_ablation_runs.csv", index=False)
ablB_s = ablB.groupby(["channels", "detector"]).AUROC.agg(["mean", "std"]).round(4).unstack()
ablB_s.to_csv(f"{RES}/taskB_ablation_auroc.csv"); print(ablB_s)
ablB_f = ablB.groupby(["channels", "detector"]).false_alarms_per_min.mean().round(3).unstack()
ablB_f.to_csv(f"{RES}/taskB_ablation_false_alarms.csv"); print(ablB_f)

nzb = []
for lvl in [0.0, 0.25, 0.5, 1.0]:
    r, _ = run_anomaly(CH, seeds, ["Mahalanobis", "IsolationForest", "GRU-AE"], noise=lvl); r["noise"] = lvl; nzb.append(r)
nzB = pd.concat(nzb); nzB_s = nzB.groupby(["noise", "detector"])[["AUROC", "false_alarms_per_min"]].mean().round(4).unstack()
nzB_s.to_csv(f"{RES}/taskB_noise_robustness.csv"); print(nzB_s)

# ======================= 6. EXPLAINABILITY (SHAP on XGBoost phase model) =======================
import shap
sc = StandardScaler().fit(F); clf = classical("XGBoost", 0).fit(sc.transform(F), y_phase)
sv = shap.TreeExplainer(clf).shap_values(sc.transform(F))
sv = np.array(sv)
if sv.ndim == 3 and sv.shape[0] == len(F): sv = sv.transpose(2, 0, 1)   # -> (class, n, feat)
imp_feat = pd.Series(np.abs(sv).mean((0, 1)), index=FN)
imp_ch = imp_feat.groupby(lambda n: n.rsplit("_", 1)[0]).sum().sort_values(ascending=False)
imp_ch.to_csv(f"{RES}/shap_channel_importance_phase.csv"); imp_feat.sort_values(ascending=False).head(20).to_csv(f"{RES}/shap_top20_features_phase.csv")

# ======================= 7. FIGURES =======================
t = np.arange(len(raw)) / FS
seg_bounds = raw.groupby((raw.phase_label != raw.phase_label.shift()).cumsum()).apply(lambda g: (g.index[0], g.index[-1], g.phase_label.iloc[0]))
a0, a1 = audit["anomaly_start_idx"] / FS, audit["anomaly_end_idx"] / FS

# Fig 1: multimodal stream
show = [("wrist_hr_bpm", "HR (bpm)"), ("chest_emg", "Chest EMG (a.u.)"), ("chest_gsr", "GSR (µS)"),
        ("chest_gyro_y", "Gyro-y (°/s)"), ("wrist_spo2_pct", "SpO₂ (%)")]
fig, ax = plt.subplots(len(show), 1, figsize=(7.2, 6.2), sharex=True)
for k, (c, lab) in enumerate(show):
    for (s, e, p) in seg_bounds:
        ax[k].axvspan(s / FS, e / FS, color=C[PHASES.index(p)], alpha=0.10, lw=0)
    ax[k].axvspan(a0, a1, color="#e34948", alpha=0.18, lw=0)
    ax[k].plot(t, raw[c], color=INK, lw=0.6); ax[k].set_ylabel(lab, fontsize=8)
for (s, e, p) in seg_bounds:
    ax[0].text((s + e) / 2 / FS, ax[0].get_ylim()[1], p, ha="center", va="bottom", fontsize=7.5, color=INK2)
ax[1].text((a0 + a1) / 2, ax[1].get_ylim()[1] * 0.92, "strain event", ha="center", fontsize=7.5, color="#b02a2a")
ax[-1].set_xlabel("Time (s)"); fig.align_ylabels(); fig.savefig(f"{FIG}/fig1_multimodal_stream.png"); plt.close()

# Fig 2: correlation heatmap (diverging blue-gray-red)
from matplotlib.colors import LinearSegmentedColormap
div = LinearSegmentedColormap.from_list("d", ["#184f95", "#86b6ef", "#f0efec", "#ef9a99", "#a32a2a"])
short = [c.replace("wrist_", "W-").replace("chest_", "C-").replace("_pct", "").replace("_bpm", "") for c in CH]
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(spear.values, cmap=div, vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(13)); ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7.5)
ax.set_yticks(range(13)); ax.set_yticklabels(short, fontsize=7.5)
for i in range(13):
    for j in range(13):
        v = spear.values[i, j]
        if abs(v) >= 0.3 and i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8, color="white" if abs(v) > 0.7 else INK)
cb = fig.colorbar(im, fraction=0.046, pad=0.03); cb.set_label("Spearman ρ"); cb.outline.set_visible(False)
fig.savefig(f"{FIG}/fig2_spearman_correlation.png"); plt.close()

# Fig 3: phase-wise distributions (Kruskal effect sizes)
top = kw.channel.head(6).tolist()
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
for k, c in enumerate(top):
    a = ax.flat[k]
    bp = a.boxplot([raw.loc[raw.phase_label == p, c] for p in PHASES], patch_artist=True, widths=0.6, showfliers=False,
                   medianprops=dict(color=INK, lw=1))
    for j, b in enumerate(bp["boxes"]): b.set_facecolor(C[j]); b.set_alpha(0.75); b.set_edgecolor("white")
    a.set_xticks(range(1, 6)); a.set_xticklabels(["Warm", "Acc", "Peak", "Dec", "Rec"], fontsize=7)
    e2 = kw.set_index("channel").loc[c, "epsilon2"]
    a.set_title(f"{c}\nε² = {e2:.2f}", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_phase_distributions.png"); plt.close()

# Fig 4: protocol comparison (leakage)
pm = phaseA.groupby(["protocol", "model"]).macroF1.agg(["mean", "std"]).reset_index()
order = ["LogReg", "RandomForest", "XGBoost", "1D-CNN", "GRU", "CNN-GRU"]
fig, ax = plt.subplots(figsize=(7.2, 3.0)); xw = np.arange(len(order)); bw = 0.38
for k, (prot, lab) in enumerate([("random", "Random window split (leaky)"), ("blocked", "Segment-stratified blocked CV + purge")]):
    d = pm[pm.protocol == prot].set_index("model").loc[order]
    ax.bar(xw + (k - 0.5) * bw, d["mean"], bw - 0.03, yerr=d["std"], color=[C[1], C[0]][k], label=lab, error_kw=dict(lw=0.8, ecolor=INK2))
    for x, v in zip(xw + (k - 0.5) * bw, d["mean"]): ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(xw); ax.set_xticklabels(order); ax.set_ylabel("Macro-F1 (phase)"); ax.set_ylim(0, 1.1)
ax.legend(frameon=False, fontsize=7.5, loc="lower right"); fig.savefig(f"{FIG}/fig4_protocol_leakage.png"); plt.close()

# Fig 5: confusion matrix of best blocked deep model
best = pa_b.groupby("model").macroF1.mean().idxmax()
yt, yp = preds_b[(best, 0)]
cm = confusion_matrix(yt, yp, normalize="true")
seq_cmap = LinearSegmentedColormap.from_list("s", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
fig, ax = plt.subplots(figsize=(4.2, 3.6)); ax.imshow(cm, cmap=seq_cmap, vmin=0, vmax=1); ax.grid(False)
for i in range(5):
    for j in range(5): ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.6 else INK)
ax.set_xticks(range(5)); ax.set_xticklabels(["Warm", "Acc", "Peak", "Dec", "Rec"]); ax.set_yticks(range(5)); ax.set_yticklabels(["Warm", "Acc", "Peak", "Dec", "Rec"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{best} (blocked CV, seed 0)", fontsize=8.5)
fig.savefig(f"{FIG}/fig5_confusion_phase.png"); plt.close()

# Fig 6: anomaly score timelines
fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.8), sharex=True)
for k, det in enumerate(["Mahalanobis", "IsolationForest", "GRU-AE"]):
    s, th = scoresB[(det, 0)]; tt = ends / FS
    ax[k].axvspan(a0, a1, color="#e34948", alpha=0.18, lw=0)
    ax[k].plot(tt, s / th, color=C[k], lw=1); ax[k].axhline(1, color=INK2, ls="--", lw=0.8)
    ax[k].set_yscale("log"); ax[k].set_ylabel("score / threshold", fontsize=7.5); ax[k].set_title(det, fontsize=8.5, loc="left")
ax[-1].set_xlabel("Time (s)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig6_anomaly_scores.png"); plt.close()

# Fig 7: SHAP channel importance
fig, ax = plt.subplots(figsize=(5.2, 3.6))
v = imp_ch.sort_values(); ax.barh(range(len(v)), v.values, color=C[0], height=0.6)
ax.set_yticks(range(len(v))); ax.set_yticklabels(v.index, fontsize=7.5); ax.set_xlabel("Σ mean |SHAP| (phase model)")
fig.savefig(f"{FIG}/fig7_shap_channels.png"); plt.close()

# Fig 8: anomaly ablation
fig, ax = plt.subplots(figsize=(7.2, 3.2)); names = list(ABL_B.keys()); xw = np.arange(len(names)); bw = 0.27
for k, det in enumerate(["Mahalanobis", "IsolationForest", "GRU-AE"]):
    d = ablB.groupby(["channels", "detector"]).AUROC.mean().unstack().loc[names, det]
    ax.bar(xw + (k - 1) * bw, d.values, bw - 0.03, color=C[k], label=det)
ax.set_xticks(xw); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7.5); ax.set_ylabel("AUROC (strain event)")
ax.set_ylim(0, 1.05); ax.axhline(0.5, color=INK2, ls=":", lw=0.8); ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
fig.savefig(f"{FIG}/fig8_anomaly_ablation.png"); plt.close()
print("DONE", time.time() - t0)
