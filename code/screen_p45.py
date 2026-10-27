"""Prompt 4 + 5A screening on the TUNE clip (start=150, seed=11).
A: champion (reproduce)  B: +fz  C: +stab  D: +fz+stab.
Screening only: no VLM judge here; the winner goes through iterate.py (gates + blind judge).
Usage: python3 screen_p45.py [--quick]"""
import json, time, sys
import numpy as np, pickle
import sim, restore as R
from metrics2 import psnr, flicker_summary
from skimage.metrics import structural_similarity as ssim

quick = "--quick" in sys.argv
champ = json.load(open("../results/champion.json"))["cfg"]
variants = {
    "A_champ": {},
    "B_fz": {"fz": 1},
    "C_stab": {"stab": 1},
    "D_fz_stab": {"fz": 1, "stab": 1},
}
if quick:
    variants = {k: v for k, v in variants.items() if k != "A_champ"}

rgb = sim.load_clean(start=150)
cl = sim.clean_frames(rgb)
dg, meta = sim.degrade(cl, seed=11)
q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
groups = {"all": np.ones(len(cl), bool), "good": (q > 0.7) & ~lost & ~roll,
          "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}

results = {}
for name, delta in variants.items():
    cfg = {**champ, **delta}
    t0 = time.time()
    out = R.restore(dg, cfg=cfg)
    rt = time.time() - t0
    p = {g: float(np.mean([psnr(out[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(out[i], cl[i], channel_axis=2, data_range=255)
                           for i in np.where(m)[0]])) for g, m in groups.items()}
    f = flicker_summary(out, cl, meta)
    results[name] = dict(delta=delta, p=p, s=s, f=f, runtime=rt)
    np.save(f"screen_{name}.npy", out)
    print(f"{name:12s} " + " ".join(f"{g}:{p[g]:6.2f}" for g in ("all", "good", "fades", "roll", "lost"))
          + f" | flick_static_good {f['flick_static_good']:7.3f} tdiff {f['flick_tdiff_good']:6.3f}"
          + f" | ssim {s['all']:.3f} | {rt:.0f}s", flush=True)

json.dump(results, open("screen_p45.json", "w"), indent=1)
print("\nsaved screen_p45.json")
