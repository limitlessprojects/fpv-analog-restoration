"""One-shot local state bootstrap (session 2026-10-27 continuation):
- copy tracked champion into code/ for the iterate loop
- rebuild tune_baselines.json (med3+nlm floor) if missing
- write champion_out.npy + screen_A_champ.npy on the tune clip (input for diag_target_check)
Idempotent: skips work already present."""
import json, shutil, os, time
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists("champion.json"):
    shutil.copy("../results/champion.json", "champion.json")
champ = json.load(open("champion.json"))

if not os.path.exists("tune_baselines.json"):
    from baselines import nlm_multi, med3
    from metrics2 import psnr
    dg = np.load("tune_deg.npy"); cl = np.load("tune_clean.npy")
    out = nlm_multi(list(med3(dg)), 18, 18)
    p = float(np.mean([psnr(out[i], cl[i]) for i in range(len(cl))]))
    json.dump({"med3_nlm_psnr_all": p}, open("tune_baselines.json", "w"))
    print("tune med3+nlm floor:", round(p, 2), flush=True)
else:
    print("tune floor:", json.load(open("tune_baselines.json")))

if not os.path.exists("champion_out.npy"):
    from restore import restore
    dg = np.load("tune_deg.npy")
    t0 = time.time()
    out = restore(dg, cfg=champ["cfg"])
    np.save("champion_out.npy", out); np.save("screen_A_champ.npy", out)
    print(f"champion rebuilt on tune in {time.time()-t0:.0f}s", flush=True)
else:
    print("champion_out.npy present")
