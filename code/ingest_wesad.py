"""Turn Colab WESAD results (wesad_all_runs.csv) into manuscript values: python ingest_wesad.py <results_dir> <out_json>"""
import sys, json, pandas as pd
R = pd.read_csv(f"{sys.argv[1]}/wesad_all_runs.csv")
g = R.groupby(["task", "protocol", "modality", "model"]).macroF1.agg(["mean", "std"])
def v(task, prot, mod, model):
    try: m, s = g.loc[(task, prot, mod, model)]
    except KeyError: return "—"
    return f"{m:.3f}" if pd.isna(s) else f"{m:.3f} ± {s:.3f}"
ALL, DEV = "Chest + wrist (all)", "Device-matched (ECG-HR, EMG, EDA, chest ACC, wrist HR, wrist ACC)"
o = {}
for t, tk in [("3-class", "3c"), ("binary", "bin")]:
    for mdl, k in [("Logistic regression", "lr"), ("Random forest", "rf"), ("XGBoost", "xgb"), ("CNN-GRU", "cnn")]:
        o[f"{k}_{tk}_loso"] = v(t, "LOSO", ALL, mdl); o[f"{k}_{tk}_rand"] = v(t, "Random split", ALL, mdl)
    o[f"xgb_{tk}_chest"] = v(t, "LOSO", "Chest only", "XGBoost"); o[f"xgb_{tk}_wrist"] = v(t, "LOSO", "Wrist only", "XGBoost"); o[f"xgb_{tk}_dev"] = v(t, "LOSO", DEV, "XGBoost")
lo = g.xs(("3-class", "LOSO", ALL), level=[0, 1, 2])["mean"]; ra = g.xs(("3-class", "Random split", ALL), level=[0, 1, 2])["mean"]
lb = g.xs(("binary", "LOSO", ALL), level=[0, 1, 2])["mean"]
o["loso_best_3c"] = f"{lo.max():.3f}"; o["rand_best_3c"] = f"{ra.max():.3f}"
mods = g.xs(("3-class", "LOSO"), level=[0, 1])["mean"].unstack()["XGBoost"].drop(ALL, errors="ignore").sort_values(ascending=False)
o["wesad_text"] = (f"Under LOSO the best three-class macro-F1 was {lo.max():.3f} ({lo.idxmax()}) and the best binary macro-F1 {lb.max():.3f} ({lb.idxmax()}); "
                   f"random window splitting raised three-class macro-F1 to {ra.max():.3f}. The CNN-GRU reached {lo.get('CNN-GRU', float('nan')):.3f} (three-class) "
                   f"against {lo.drop('CNN-GRU', errors='ignore').max():.3f} for the best feature-based model. In the modality ablation (XGBoost, three-class), "
                   f"the most informative subset was {mods.index[0].split(' (')[0].lower()} ({mods.iloc[0]:.3f}) and the least {mods.index[-1].split(' (')[0].lower()} ({mods.iloc[-1]:.3f}); "
                   f"the device-matched sensor set reached {g.loc[('3-class','LOSO',DEV,'XGBoost'),'mean']:.3f}. These values are compared with the original WESAD benchmark in Table 3.")
json.dump(o, open(sys.argv[2], "w"), ensure_ascii=False, indent=1); print(json.dumps(o, ensure_ascii=False, indent=1)[:1500])
