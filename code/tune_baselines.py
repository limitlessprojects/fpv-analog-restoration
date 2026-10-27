"""Best-known baseline (median3 + temporal NLM) on the TUNE clip -> floor for gate G0."""
import numpy as np, json
from baselines import nlm_multi, med3
from metrics2 import psnr
dg = np.load("tune_deg.npy"); cl = np.load("tune_clean.npy")
out = nlm_multi(list(med3(dg)), 18, 18)
p = float(np.mean([psnr(out[i], cl[i]) for i in range(len(cl))]))
json.dump({"med3_nlm_psnr_all": p}, open("tune_baselines.json", "w"))
print("tune med3+nlm PSNR:", round(p, 2))
