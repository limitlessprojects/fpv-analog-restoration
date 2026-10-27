"""Diagnose: (1) where does residual flicker live - pre or post polish?
(2) what flow magnitudes does the tune clip actually have (gate fraction)?"""
import numpy as np, cv2, pickle, json
from restore import restore_core, spatial_polish, temporal_post_stage, guide_of, flow, warp, DEFAULTS
from metrics2 import flicker_summary, psnr

cl = np.load("tune_clean.npy"); dg = np.load("tune_deg.npy")
meta = pickle.load(open("tune_meta.pkl", "rb"))

cur, sigs, valid = restore_core(dg, cfg={})
nop = np.array([cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB) for y in cur])
print("pre-polish :", {k: round(v, 3) for k, v in flicker_summary(nop, cl, meta).items()})
pol = np.array(spatial_polish(cur, sigs, valid, {**DEFAULTS}))
print("post-polish:", {k: round(v, 3) for k, v in flicker_summary(pol, cl, meta).items()})

# flow magnitudes on the polished output guides (what temporal_post sees)
guides = [guide_of(cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32), 0.8) for f in pol]
mags = []
for t in range(1, len(pol) - 1):
    if not valid[t]: continue
    fl = flow(guides[t], guides[t + 1])
    mags.append(np.hypot(fl[..., 0], fl[..., 1]))
m = np.concatenate([x.ravel() for x in mags])
for thr in (0.3, 0.75, 1.5, 3.0):
    print(f"flow< {thr}: {(m < thr).mean() * 100:.1f}% of pixels")
