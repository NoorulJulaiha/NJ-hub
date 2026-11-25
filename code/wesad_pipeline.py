"""
WESAD (Schmidt et al., ICMI 2018) validation of the chest + wrist sensing concept.
Chest RespiBAN (700 Hz): ECG, EMG, EDA, Resp, Temp, ACC.  Wrist Empatica E4: BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz.
Labels in SX.pkl: 1 baseline, 2 stress, 3 amusement (others ignored).

Protocols: LOSO (primary) vs random window split (leaky reference).
Tasks: 3-class (baseline/stress/amusement) and binary (stress vs non-stress).
Modality sets: chest, wrist, chest+wrist, device-matched subset, single modalities.
Usage: python wesad_pipeline.py --root /content/WESAD --out /content/out_wesad [--seeds 3]
"""
import argparse, os, glob, pickle, json, time, warnings
import numpy as np, pandas as pd
from scipy import signal as sg
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", default="out_wesad")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--win", type=int, default=60); ap.add_argument("--step", type=int, default=5)
ap.add_argument("--dl_epochs", type=int, default=25)
ap.add_argument("--abl_seeds", type=int, default=1)
ap.add_argument("--abl_models", default="XGBoost", help="comma list of models for modality ablation")
a = ap.parse_args(); DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", DEV); RES, FIG = f"{a.out}/results", f"{a.out}/figures"
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
FS_C, FS = 700, 4          # chest native rate; common derived-signal rate (Hz)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]; INK2 = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "savefig.dpi": 300, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- signal processing
def bp(x, lo, hi, fs, order=3):
    b, a_ = sg.butter(order, [lo / (fs / 2), hi / (fs / 2)], "band"); return sg.filtfilt(b, a_, x)
def lp(x, hi, fs, order=3):
    b, a_ = sg.butter(order, hi / (fs / 2), "low"); return sg.filtfilt(b, a_, x)
def to_grid(x, fs_in, n_out):
    """Resample a signal to FS (4 Hz) by block averaging / interpolation onto n_out points."""
    t_in = np.arange(len(x)) / fs_in; t_out = np.arange(n_out) / FS
    return np.interp(t_out, t_in, x)
def peaks_to_hr(peaks, fs, n_out):
    """Instantaneous HR (bpm) on the 4 Hz grid, plus RR list (s) with peak times."""
    if len(peaks) < 3: return np.full(n_out, np.nan), np.array([]), np.array([])
    tp = peaks / fs; rr = np.diff(tp); tm = tp[1:]
    ok = (rr > 0.3) & (rr < 2.0); rr, tm = rr[ok], tm[ok]
    if len(rr) < 3: return np.full(n_out, np.nan), rr, tm
    return np.interp(np.arange(n_out) / FS, tm, 60 / rr), rr, tm

def derive_subject(d):
    """Return dict of 4 Hz derived channels + RR series for one subject pkl."""
    ch, wr = d["signal"]["chest"], d["signal"]["wrist"]
    lab = np.asarray(d["label"]).ravel(); n_s = len(lab); n_out = int(n_s / FS_C * FS)
    out = {}
    ecg = bp(np.asarray(ch["ECG"]).ravel(), 5, 15, FS_C)
    pk, _ = sg.find_peaks(ecg, distance=int(0.33 * FS_C), height=np.percentile(ecg, 90) * 0.5)
    out["C_HR"], rr_c, tm_c = peaks_to_hr(pk, FS_C, n_out)
    emg = np.asarray(ch["EMG"]).ravel(); emg = emg - lp(emg, 20, FS_C)          # high-pass ~20 Hz
    blk = FS_C // FS; nb = len(emg) // blk
    out["C_EMG_RMS"] = np.sqrt((emg[:nb * blk].reshape(nb, blk) ** 2).mean(1))[:n_out]
    out["C_EDA"] = to_grid(lp(np.asarray(ch["EDA"]).ravel(), 1, FS_C), FS_C, n_out)
    out["C_RESP"] = to_grid(lp(np.asarray(ch["Resp"]).ravel(), 1, FS_C), FS_C, n_out)
    out["C_TEMP"] = to_grid(np.asarray(ch["Temp"]).ravel(), FS_C, n_out)
    acc = np.asarray(ch["ACC"]); out["C_ACC"] = to_grid(lp(np.linalg.norm(acc, axis=1), 1.5, FS_C), FS_C, n_out)
    bvp = bp(np.asarray(wr["BVP"]).ravel(), 0.7, 3.5, 64)
    pk, _ = sg.find_peaks(bvp, distance=int(0.33 * 64))
    out["W_HR"], rr_w, tm_w = peaks_to_hr(pk, 64, n_out)
    out["W_EDA"] = to_grid(np.asarray(wr["EDA"]).ravel(), 4, n_out)
    out["W_TEMP"] = to_grid(np.asarray(wr["TEMP"]).ravel(), 4, n_out)
    wacc = np.asarray(wr["ACC"], dtype=float); out["W_ACC"] = to_grid(np.linalg.norm(wacc, axis=1), 32, n_out)
    lab4 = lab[(np.arange(n_out) * FS_C / FS).astype(int)]
    for k in out: out[k] = pd.Series(out[k]).interpolate(limit_direction="both").values[:n_out]
    return out, lab4, (rr_c, tm_c), (rr_w, tm_w)

CHANNELS = ["C_HR", "C_EMG_RMS", "C_EDA", "C_RESP", "C_TEMP", "C_ACC", "W_HR", "W_EDA", "W_TEMP", "W_ACC"]
def slope(x): t = np.arange(len(x)) - (len(x) - 1) / 2; return (t * (x - x.mean())).sum() / (t ** 2).sum()
def window_features(sig, rrc, rrw, s, e):
    f = {}
    for k in CHANNELS:
        x = sig[k][s:e]
        f[f"{k}_mean"], f[f"{k}_std"], f[f"{k}_min"], f[f"{k}_max"], f[f"{k}_slope"] = x.mean(), x.std(), x.min(), x.max(), slope(x)
    for tag, (rr, tm) in [("C", rrc), ("W", rrw)]:
        m = (tm >= s / FS) & (tm < e / FS); r = rr[m]
        f[f"{tag}_HRV_RMSSD"] = np.sqrt(np.mean(np.diff(r) ** 2)) if len(r) > 3 else np.nan
        f[f"{tag}_HRV_SDNN"] = r.std() if len(r) > 3 else np.nan
    for k in ["C_EDA", "W_EDA"]:
        x = sig[k][s:e]; f[f"{k}_nSCR"] = len(sg.find_peaks(x, prominence=max(0.01, 0.05 * x.std()))[0])
    r = sig["C_RESP"][s:e] - sig["C_RESP"][s:e].mean(); fr, P = sg.periodogram(r, FS)
    band = (fr > 0.1) & (fr < 0.7); f["C_RESP_rate"] = fr[band][np.argmax(P[band])] * 60 if band.any() else np.nan
    f["C_EMG_RMS_p90"] = np.percentile(sig["C_EMG_RMS"][s:e], 90)
    return f

# ---------------------------------------------------------------- build window table
t0 = time.time()
CACHE = f"{RES}/_cache_windows.npz"
pk_files = sorted(glob.glob(os.path.join(a.root, "S*", "S*.pkl")))
assert pk_files, f"No S*/S*.pkl under {a.root}"
rows, seqs = [], []
W, ST = a.win * FS, a.step * FS
if os.path.exists(CACHE) and os.path.exists(f"{RES}/wesad_window_features.csv"):
    print("Loading cached windows", flush=True); pk_files = []
for pf in pk_files:
    sid = os.path.basename(pf)[:-4]
    with open(pf, "rb") as fh: d = pickle.load(fh, encoding="latin1")
    sig, lab4, rrc, rrw = derive_subject(d); del d
    for s in range(0, len(lab4) - W + 1, ST):
        L = lab4[s:s + W]
        if L[0] not in (1, 2, 3) or (L != L[0]).any(): continue
        f = window_features(sig, rrc, rrw, s, s + W); f.update(subject=sid, label=int(L[0]), start_s=s / FS)
        rows.append(f); seqs.append(np.stack([sig[k][s:s + W] for k in CHANNELS], 1).astype(np.float32))
    print(sid, "windows so far:", len(rows), f"[{time.time()-t0:.0f}s]", flush=True)
if rows:
    T = pd.DataFrame(rows); SEQ = np.stack(seqs)
    FEATS = [c for c in T.columns if c not in ("subject", "label", "start_s")]
    T[FEATS] = T[FEATS].fillna(T[FEATS].median())
    T.to_csv(f"{RES}/wesad_window_features.csv", index=False); np.savez_compressed(CACHE, SEQ=SEQ)
else:
    T = pd.read_csv(f"{RES}/wesad_window_features.csv"); SEQ = np.load(CACHE)["SEQ"]
    FEATS = [c for c in T.columns if c not in ("subject", "label", "start_s")]
# per-subject z-score (label-free, standard in WESAD work)
Z = T.copy(); Z[FEATS] = T.groupby("subject")[FEATS].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
SEQZ = SEQ.copy()
for s in T.subject.unique():
    i = (T.subject == s).values; mu = SEQ[i].reshape(-1, len(CHANNELS)).mean(0); sd = SEQ[i].reshape(-1, len(CHANNELS)).std(0) + 1e-6
    SEQZ[i] = (SEQ[i] - mu) / sd
info = dict(subjects=T.subject.nunique(), windows=len(T), per_label=T.label.value_counts().to_dict(), win_s=a.win, step_s=a.step, n_features=len(FEATS))
json.dump(info, open(f"{RES}/wesad_audit.json", "w"), indent=2, default=int); print(info)

# ---------------------------------------------------------------- models
class CNNGRU(nn.Module):
    def __init__(s, c, o):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(c, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        s.r = nn.GRU(32, 32, batch_first=True); s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(32, o))
    def forward(s, x): return s.h(s.r(s.c(x.transpose(1, 2)).transpose(1, 2))[1][-1])

def fit_predict(model, tr, te, y, ncls, ch_idx, fcols, seed):
    if model == "CNN-GRU":
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr, Xte = torch.tensor(SEQZ[tr][:, :, ch_idx]).to(DEV), torch.tensor(SEQZ[te][:, :, ch_idx]).to(DEV)
        m = CNNGRU(len(ch_idx), ncls).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
        w = torch.tensor(len(tr) / (ncls * np.bincount(y[tr], minlength=ncls).clip(1)), dtype=torch.float32).to(DEV)
        lf = nn.CrossEntropyLoss(weight=w); yt = torch.tensor(y[tr]).to(DEV)
        for _ in range(a.dl_epochs):
            m.train(); p = torch.randperm(len(tr), device=DEV)
            for b in range(0, len(tr), 64):
                i = p[b:b + 64]; opt.zero_grad(); lf(m(Xtr[i]), yt[i]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): return m(Xte).argmax(1).cpu().numpy()
    Xtr, Xte = Z.iloc[tr][fcols].values, Z.iloc[te][fcols].values
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if model == "Logistic regression": clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
    elif model == "Random forest": clf = RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed)
    else: clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=0, n_jobs=-1)
    sw = None
    if model == "XGBoost": cw = len(tr) / (ncls * np.bincount(y[tr], minlength=ncls)); sw = cw[y[tr]]
    clf.fit(Xtr, y[tr], **({"sample_weight": sw} if sw is not None else {})); return clf.predict(Xte)

MODSETS = {
    "Chest + wrist (all)": CHANNELS,
    "Chest only": [c for c in CHANNELS if c.startswith("C_")],
    "Wrist only": [c for c in CHANNELS if c.startswith("W_")],
    "Device-matched (ECG-HR, EMG, EDA, chest ACC, wrist HR, wrist ACC)": ["C_HR", "C_EMG_RMS", "C_EDA", "C_ACC", "W_HR", "W_ACC"],
    "Cardiac only (chest HR + wrist HR)": ["C_HR", "W_HR"],
    "EDA only (chest + wrist)": ["C_EDA", "W_EDA"],
    "EMG only": ["C_EMG_RMS"],
    "Motion only (chest + wrist ACC)": ["C_ACC", "W_ACC"],
}
def cols_for(chs):
    pref = tuple(chs) + tuple({c.split("_")[0] + "_HRV" for c in chs if c.endswith("_HR")}) + (("C_RESP_rate",) if "C_RESP" in chs else ()) + (("C_EMG_RMS_p90",) if "C_EMG_RMS" in chs else ())
    return [f for f in FEATS if f.startswith(pref)]

RUNS = f"{RES}/wesad_all_runs.csv"
DONE_KEYS = set()
if os.path.exists(RUNS):
    _p = pd.read_csv(RUNS); DONE_KEYS = set(zip(_p.task, _p.protocol, _p.modality, _p.model, _p.seed)); print("Resuming:", len(DONE_KEYS), "runs already done", flush=True)
def save_row(r):
    pd.DataFrame([r]).to_csv(RUNS, mode="a", header=not os.path.exists(RUNS), index=False)
def run(task, protocol, models, modset, seeds):
    y = T.label.values - 1 if task == "3-class" else (T.label.values == 2).astype(int)
    ncls = 3 if task == "3-class" else 2; chs = MODSETS[modset]; ci = [CHANNELS.index(c) for c in chs]; fc = cols_for(chs)
    subj = T.subject.values; out = []
    for m in models:
        for sd in seeds:
            if (task, protocol, modset, m, sd) in DONE_KEYS: continue
            if protocol == "LOSO": splits = [(np.where(subj != s)[0], np.where(subj == s)[0]) for s in np.unique(subj)]
            else: splits = list(StratifiedKFold(5, shuffle=True, random_state=sd).split(T, y))
            pred = np.zeros(len(y), int); persub = []
            for tr, te in splits:
                pred[te] = fit_predict(m, tr, te, y, ncls, ci, fc, sd)
                if protocol == "LOSO": persub.append(dict(subject=subj[te][0], macroF1=f1_score(y[te], pred[te], average="macro")))
            out.append(dict(task=task, protocol=protocol, modality=modset, model=m, seed=sd, accuracy=accuracy_score(y, pred),
                            macroF1=f1_score(y, pred, average="macro"), balAcc=balanced_accuracy_score(y, pred)))
            if protocol == "LOSO" and modset == "Chest + wrist (all)" and sd == 0:
                pd.DataFrame(persub).to_csv(f"{RES}/persubject_{task}_{m.replace(' ', '_')}.csv", index=False)
            save_row(out[-1]); print(task, protocol, modset[:22], m, sd, round(out[-1]["macroF1"], 3), f"[{time.time()-t0:.0f}s]", flush=True)
    return out

seeds = list(range(a.seeds)); MODELS = ["Logistic regression", "Random forest", "XGBoost", "CNN-GRU"]; res = []
for task in ["3-class", "binary"]:
    res += run(task, "LOSO", MODELS, "Chest + wrist (all)", seeds)
    res += run(task, "Random split", ["Logistic regression", "XGBoost", "CNN-GRU"], "Chest + wrist (all)", seeds)
    for ms in list(MODSETS)[1:]:
        res += run(task, "LOSO", a.abl_models.split(","), ms, list(range(a.abl_seeds)))
R = pd.read_csv(RUNS)
S = R.groupby(["task", "protocol", "modality", "model"])[["accuracy", "macroF1", "balAcc"]].agg(["mean", "std"]).round(4)
S.to_csv(f"{RES}/wesad_summary.csv"); print(S.to_string())

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for k, task in enumerate(["3-class", "binary"]):
    d = R[(R.task == task) & (R.modality == "Chest + wrist (all)")].groupby(["protocol", "model"]).macroF1.mean().unstack(0)
    x = np.arange(len(d));
    for j, (p, lab) in enumerate([("Random split", "Random window split (leaky)"), ("LOSO", "Leave-one-subject-out")]):
        if p in d: ax[k].bar(x + (j - 0.5) * 0.38, d[p].values, 0.35, color=[C[1], C[0]][j], label=lab)
    ax[k].set_xticks(x); ax[k].set_xticklabels(d.index, fontsize=7, rotation=15); ax[k].set_title(task, fontsize=9); ax[k].set_ylim(0, 1.05)
ax[0].set_ylabel("Macro-F1"); ax[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(1.05, 1.07))
fig.savefig(f"{FIG}/figW1_protocol.png"); plt.close()

fig, ax = plt.subplots(figsize=(7.2, 3.4)); names = list(MODSETS)
d = R[(R.protocol == "LOSO") & (R.model == "XGBoost")].groupby(["modality", "task"]).macroF1.mean().unstack().loc[names]
y_ = np.arange(len(names))
for j, task in enumerate(["3-class", "binary"]): ax.barh(y_ + (j - 0.5) * 0.38, d[task].values, 0.35, color=C[j], label=task)
ax.set_yticks(y_); ax.set_yticklabels([n.split(" (")[0] if len(n) > 30 else n for n in names], fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("Macro-F1 (LOSO, XGBoost)"); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{FIG}/figW2_modality.png"); plt.close()
print("DONE", time.time() - t0)
