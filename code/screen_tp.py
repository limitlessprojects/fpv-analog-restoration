"""Detail-band temporal_post screening on the TUNE clip (start=150, seed=11).
Hypothesis (session 2026-10-27): the metric-domain pair-diff variance on good/static pairs
is 64% detail band (diag_target_check: 69.6 detail vs 39.4 lowpass, re-verified this
session); a flow-gated temporal median restricted to the DETAIL band reduces
flick_static_good / flick_tdiff_good without touching the low band, so PSNR/SSIM survive.
Arms are parameterizations of that one lever; T4 is the legacy full-band arm kept for
attribution (is the band restriction doing anything?).
Screening only: no VLM judge here; the winner goes through iterate.py (gates + blind judge).
Usage: python3 screen_tp.py [--quick]"""
import json, time, sys
import numpy as np, pickle
import sim, restore as R
from metrics2 import psnr, flicker_summary
from skimage.metrics import structural_similarity as ssim

quick = "--quick" in sys.argv
champ = json.load(open("../results/champion.json"))["cfg"]
variants = {
    "A_champ": {},
    "T1_band_r2_f030": {"temporal_post": True, "tp_band": 1},
    "T2_band_r2_f045": {"temporal_post": True, "tp_band": 1, "tp_flow": 0.45},
    "T3_band_r3_f030": {"temporal_post": True, "tp_band": 1, "tp_rad": 3},
    "T4_full_r2_f030": {"temporal_post": True, "tp_band": 0},
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
    try:
        out = R.restore(dg, cfg=cfg)
    except Exception as e:
        print(f"{name:16s} CRASH {e}", flush=True)
        results[name] = dict(delta=delta, crash=str(e))
        continue
    rt = time.time() - t0
    p = {g: float(np.mean([psnr(out[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(out[i], cl[i], channel_axis=2, data_range=255)
                           for i in np.where(m)[0]])) for g, m in groups.items()}
    f = flicker_summary(out, cl, meta)
    results[name] = dict(delta=delta, p=p, s=s, f=f, runtime=rt)
    np.save(f"screen_{name}.npy", out)
    print(f"{name:16s} " + " ".join(f"{g}:{p[g]:6.2f}" for g in ("all", "good", "fades", "roll", "lost"))
          + f" | flick_static_good {f['flick_static_good']:7.3f} tdiff {f['flick_tdiff_good']:6.3f}"
          + f" | ssim {s['all']:.3f} | {rt:.0f}s", flush=True)

json.dump(results, open("screen_tp.json", "w"), indent=1)
print("\nsaved screen_tp.json")
