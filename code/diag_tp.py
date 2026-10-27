"""Why did the detail-band temporal median not move flick_static_good?
Two candidate mechanisms:
  (a) gate coverage: the flow gate (mag < thr) passes almost everything, so it is not the
      limiter, OR passes almost nothing
  (b) correlated residual: on gated GT-static pixels the post-polish residual is shared
      across the +-rad window (the fusion stages reuse the same warped data), so the
      window median carries the same error as the current frame and removes nothing.
Method: recompute the tp gate on the champion output (post-polish), then compare
pair-diff std / variance decomposition before (A) and after (T1) on gated static pixels.
Also: how much of the static-pair std survives if we median-shift the actual residual
(oracle: replace out_t with out_{t-1} on gated static pixels -> std should drop to ~0)."""
import numpy as np, cv2, pickle
import restore as R
from metrics2 import static_std_pairs

out = np.load("screen_A_champ.npy").astype(np.float32)
out1 = np.load("screen_T1_band_r2_f030.npy").astype(np.float32)
cl = np.load("tune_clean.npy").astype(np.float32)
meta = pickle.load(open("tune_meta.pkl", "rb"))
q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
good = (q > 0.7) & ~lost & ~roll
T, H, W = out.shape[:3]

dpair_gt = np.abs(cl[1:] - cl[:-1]).max(3)
stat = dpair_gt <= 1.0

guides = [R.guide_of(f.astype(np.uint8), 0.8) for f in out]
rad, thr = 2, 0.30
gates = []
for t in range(T):
    mag = np.zeros((H, W), np.float32)
    for d in range(-rad, rad + 1):
        n = t + d
        if d == 0 or n < 0 or n >= T: continue
        fl = R.flow(guides[t], guides[n])
        mag = np.maximum(mag, np.hypot(fl[..., 0], fl[..., 1]))
    gates.append(mag < thr)
gates = np.stack(gates)  # (T, H, W); pair t-1->t both need gate[t-1] & gate[t]... gate is per-frame
gfrac = [float(g.mean()) for g in gates]
print(f"gate coverage per frame: min {min(gfrac):.3f} med {np.median(gfrac):.3f} max {max(gfrac):.3f}")

# pair analysis on good pairs, gated pixels
res_a = out - cl
res_b = out1 - cl
lo_a = np.stack([cv2.GaussianBlur(res_a[t], (0, 0), 8) for t in range(T)])
det_a = res_a - lo_a
stats = []
for t in range(1, T):
    if not (good[t] and good[t - 1]): continue
    m = stat[t - 1] & gates[t] & gates[t - 1]
    if m.sum() < 500: continue
    stats.append((t, m, m.sum()))
print(f"good pairs with >=500 gated static px: {len(stats)}")
da, db, ddet_a, ddet_b = [], [], [], []
for t, m, _ in stats:
    da.append((out[t][m] - out[t - 1][m]).var())
    db.append((out1[t][m] - out1[t - 1][m]).var())
    ddet_a.append((det_a[t][m] - det_a[t - 1][m]).var())
print(f"pair-diff variance on gated static px: champ {np.mean(da):.2f} | +T1 {np.mean(db):.2f}")
print(f"detail-band part (champ): {np.mean(ddet_a):.2f}")

# oracle: if the residual were INDEPENDENT per frame, median of 5 would cut variance ~2.2x.
res_lum = res_a[:, :, :, 0]
res_var = float(np.mean([res_lum[t][m].var() for t, m, _ in stats]))
print(f"per-frame luma residual variance on gated static px: {res_var:.2f} "
      f"(iid median-of-5 floor: {res_var/2.2:.2f})")
cors = []
for t, m, _ in stats:
    a = res_lum[t][m]; b = res_lum[t - 1][m]
    if a.std() > 1e-3 and b.std() > 1e-3:
        cors.append(float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std())))
print(f"residual correlation across consecutive frames (gated static): mean {np.mean(cors):.3f}")
