"""Baseline the no-invention gate metrics on the TUNE clip with the CHAMPION config:
rerun restore with loo=1 + save_masks=1, save the log + output, then score.
This establishes the measured reference point future candidates will be gated against."""
import json, pickle, time
import numpy as np
import sim, restore as R

rgb = sim.load_clean(start=150)
cl = sim.clean_frames(rgb)
dg, meta = sim.degrade(cl, seed=11)
champ = json.load(open("../results/champion.json"))["cfg"]

log = {}
cfg = {**champ, "loo": 1, "save_masks": 1}
t0 = time.time()
out = R.restore(dg, log=log, cfg=cfg)
print(f"restore {time.time()-t0:.0f}s, masks captured: {log.get('_masks') is not None}", flush=True)

np.save("ni_champ_out.npy", out)
pickle.dump({"provenance": log["provenance"], "loo": log.get("loo"),
             "_masks": log.get("_masks")}, open("ni_champ_log.pkl", "wb"))
loo = log.get("loo") or []
print(f"loo entries: {len(loo)}", flush=True)
print("masks captured:", log.get("_masks") is not None, flush=True)
print("provenance mean observed:", round(float(np.mean([p['observed_frac'] for p in log['provenance']])), 4), flush=True)
