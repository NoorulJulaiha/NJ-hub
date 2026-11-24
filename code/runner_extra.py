import sys, numpy as np, pandas as pd, re
sys.argv=['x']
src=open('run_runners.py').read().split("t0 = time.time(); allres = []")[0]
exec(src)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
# ---- feature-group ablation (day approach, athlete-grouped CV, bagged XGBoost and GRU-free tabular) ----
base=lambda c: re.sub(r"\.\d+$","",c)
SUBJ={"perceived exertion","perceived trainingSuccess","perceived recovery"}
groups={"Objective load only (7 vars)":[i for i,c in enumerate(FD) if base(c) not in SUBJ],
        "Subjective ratings only (3 vars)":[i for i,c in enumerate(FD) if base(c) in SUBJ],
        "Distance and intensity only (5 vars)":[i for i,c in enumerate(FD) if base(c) in {"total km","km Z3-4","km Z5-T1-T2","km sprinting","nr. sessions"}],
        "Last 2 days only (all vars)":[i for i,c in enumerate(FD) if (int(re.search(r'\.(\d+)$',c).group(1)) if re.search(r'\.(\d+)$',c) else 0)<=1],
        "All 70 features":list(range(len(FD)))}
rows=[]
for name,cols in groups.items():
    for rep in range(3):
        oof=np.zeros(len(YD))
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(XD,YD,GD):
            oof[te]=bagged_xgb(XD[tr][:,cols],YD[tr],GD[tr],XD[te][:,cols],rep)
        rows.append(dict(group=name,rep=rep,AUROC=roc_auc_score(YD,oof),AUPRC=average_precision_score(YD,oof)))
        print(name,rep,round(rows[-1]['AUROC'],3),flush=True)
A=pd.DataFrame(rows); A.to_csv(f"{RES}/runner_feature_group_ablation.csv",index=False)
print(A.groupby('group')[['AUROC','AUPRC']].agg(['mean','std']).round(3))
# ---- calibration: bagged XGB OOF (P2 rep0) raw vs isotonic (fitted out-of-fold by athlete) ----
O=pd.read_csv(f"{RES}/runner_oof_P2_day.csv"); p=O["Bagged XGBoost [Lövdal]"].values; y=O.y.values; g=O.athlete.values
pc=np.zeros_like(p)
for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=1).split(p,y,g):
    pc[te]=IsotonicRegression(out_of_bounds="clip").fit(p[tr],y[tr]).predict(p[te])
cal=dict(brier_raw=brier_score_loss(y,p),brier_cal=brier_score_loss(y,pc),brier_null=brier_score_loss(y,np.full_like(p,y.mean())),
         mean_pred_raw=p.mean(),mean_pred_cal=pc.mean(),prevalence=y.mean())
# top-decile lift
q=np.quantile(pc,0.9); cal["top10_rate"]=y[pc>=q].mean(); cal["bottom50_rate"]=y[pc<=np.quantile(pc,0.5)].mean()
pd.Series(cal).to_csv(f"{RES}/runner_calibration.csv"); print(pd.Series(cal).round(4))
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0))
for k,(pp,lab) in enumerate([(p,"Raw (balanced bags)"),(pc,"Isotonic, athlete-grouped")]):
    fr,mp=calibration_curve(y,pp,n_bins=10,strategy="quantile"); ax[0].plot(mp,fr,"o-",ms=4,color=C[k],label=lab)
ax[0].plot([0,0.06],[0,0.06],ls=":",color=INK2); ax[0].set_xlim(0,max(0.06,p.max()) if False else 0.06); ax[0].set_ylim(0,0.06)
ax[0].set_xlabel("Predicted injury probability"); ax[0].set_ylabel("Observed injury rate"); ax[0].legend(frameon=False,fontsize=7)
ax[0].set_title("Calibration (deciles)",fontsize=9)
PA=pd.read_csv(f"{RES}/runner_per_athlete_auroc.csv").sort_values("AUROC")
ax[1].scatter(PA.n_inj,PA.AUROC,s=14,color=C[0]); ax[1].axhline(0.5,ls=":",color=INK2); ax[1].axhline(PA.AUROC.median(),ls="--",color=C[1],lw=1)
ax[1].text(PA.n_inj.max(),PA.AUROC.median()+0.01,f"median {PA.AUROC.median():.2f}",ha="right",fontsize=7,color=INK2)
ax[1].set_xlabel("Injuries per athlete"); ax[1].set_ylabel("Per-athlete AUROC"); ax[1].set_title("Between-athlete variability",fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/figR4_calibration_athletes.png"); plt.close()
print("DONE")
