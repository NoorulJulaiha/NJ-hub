import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C=["#2a78d6","#eb6834","#1baf7a"]; INK,INK2="#0b0b0b","#52514e"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#e4e3df","savefig.dpi":300,"savefig.bbox":"tight"})
R="../output/study1/results/"; Fg="../output/study1/figures/"
pa=pd.read_csv(R+"taskA_phase_all_runs.csv"); pm=pa.groupby(["protocol","model"]).macroF1.agg(["mean","std"]).reset_index()
order=["LogReg","RandomForest","XGBoost","1D-CNN","GRU","CNN-GRU"]
fig,ax=plt.subplots(figsize=(7.2,3.1)); x=np.arange(6); bw=.38
for k,(p,l) in enumerate([("random","Random window split (leaky)"),("blocked","Segment-stratified blocked CV + purge")]):
    d=pm[pm.protocol==p].set_index("model").loc[order]
    ax.bar(x+(k-.5)*bw,d["mean"],bw-.03,yerr=d["std"],color=[C[1],C[0]][k],label=l,error_kw=dict(lw=.8,ecolor=INK2))
    for xx,v in zip(x+(k-.5)*bw,d["mean"]): ax.text(xx,v+.015,f"{v:.3f}",ha="center",fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("Macro-F1 (phase)"); ax.set_ylim(0,1.08)
ax.legend(frameon=False,fontsize=7.5,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.0)); fig.savefig(Fg+"fig4_protocol_leakage.png"); plt.close()
ab=pd.read_csv(R+"taskB_ablation_runs.csv").groupby(["channels","detector"]).AUROC.mean().unstack()
names=["All 13 channels","Without HR","Without HR, EMG","Without HR, EMG, GSR","IMU only","Wrist band only","Chest band only"]
fig,ax=plt.subplots(figsize=(7.2,3.3)); x=np.arange(len(names)); bw=.27
for k,d in enumerate(["Mahalanobis","IsolationForest","GRU-AE"]):
    ax.bar(x+(k-1)*bw,ab.loc[names,d],bw-.03,color=C[k],label=d)
ax.set_xticks(x); ax.set_xticklabels(names,rotation=20,ha="right",fontsize=7.5); ax.set_ylabel("AUROC (strain event)"); ax.set_ylim(0,1.05)
ax.axhline(.5,color=INK2,ls=":",lw=.8); ax.text(len(names)-.45,.51,"chance",fontsize=7,color=INK2,ha="right",va="bottom")
ax.legend(frameon=False,fontsize=7.5,ncol=3,loc="lower center",bbox_to_anchor=(.5,1.0)); fig.savefig(Fg+"fig8_anomaly_ablation.png"); plt.close()
