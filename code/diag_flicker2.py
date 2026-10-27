"""Prompt 3 FLICKER DECOMPOSITION (test clip, mine_final.npy).
Residual 2.67 static-flicker number decomposed by:
  A. mechanism (residual out-GT at long-static pixels, luma): temporal band <1Hz / 1-5Hz / >5Hz,
     spatial scale (16px lowpass vs rest), event proximity (+-3 frames of impairment on/off)
  B. metric domain (pair diffs at GT-static pixels, pooled channels = the 2.67): event vs
     calm pairs, lowpass vs highpass pair-diff images
  C. EFS activation rate on the REAL clip (real_mine.npy): does the keeper ever fire there?
Then: ONE fix target = largest bucket."""
import numpy as np, cv2, pickle, json
from metrics2 import static_std_pairs
from restore import flow, guide_of, DEFAULTS

cl = np.load("test_clean.npy").astype(np.float32)
out = np.load("mine_final.npy").astype(np.float32)
meta = pickle.load(open("test_meta.pkl", "rb"))
T, H, W = cl.shape[:3]
BL, VROWS = 12, 8
scene = np.zeros((H, W), bool); scene[VROWS:, BL:] = True   # exclude pilot strip + vsync rows

# ---------------- A. residual-domain decomposition (long-static pixels, luma)
clY = np.array([cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2YCrCb)[..., 0] for f in cl]).astype(np.float32)
ouY = np.array([cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2YCrCb)[..., 0] for f in out]).astype(np.float32)
dpair_gt = np.abs(clY[1:] - clY[:-1])
stat = dpair_gt <= 1.0                          # (T-1,H,W) static per pair
cnt = stat.sum(0)
long_static = (cnt >= 0.5 * (T - 1)) & scene     # pixels static in >=50% of pairs, scene-only
print(f"long-static scene pixels: {int(long_static.sum())} ({100*long_static.mean():.1f}% of scene)")

R = (ouY - clY)[:, long_static].T.copy()        # (P, T) residual series per pixel
# fill gaps: pixels occasionally non-static -> interpolate in time
bad = ~stat[:, long_static].T                   # (P, T-1) pairs where not static
bad = np.concatenate([bad, bad[:, -1:]], axis=1)  # align to T samples
idx = np.arange(T)
for p in range(R.shape[0]):
    b = bad[p]
    if b.any() and (~b).any():
        R[p] = np.interp(idx, idx[~b], R[p, ~b])
# temporal bands via FFT (fs=30), along time axis
Fs = 30.0
F = np.fft.rfft(R, axis=1)                      # (P, T//2+1)
freqs = np.fft.rfftfreq(T, d=1 / Fs)
def band_var(lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    x = np.fft.irfft(F * m[None, :], n=T, axis=1)
    return float(x.var())
v_low, v_mid, v_hi = band_var(0, 1), band_var(1, 5), band_var(5, Fs / 2 + 1)
# spatial scale: lowpass each frame's residual map at ~16px scale
res_full = ouY - clY
low = np.stack([cv2.GaussianBlur(f, (0, 0), 8) for f in res_full])
v_sp_low = float(low[:, long_static].var())
v_sp_hi = float((res_full - low)[:, long_static].var())
# event proximity
q = meta["q"]; roll = np.array(meta["roll"]) > 0; lost = np.array(meta["lost"])
ev = set()
for arr in (roll.astype(int), lost.astype(int), (q < 0.5).astype(int), (q < 0.35).astype(int),
            ((np.arange(T) >= 28) & (np.arange(T) <= 44)).astype(int)):
    tr = np.where(np.diff(arr) != 0)[0] + 1
    for t0 in tr:
        ev.update(range(max(0, t0 - 3), min(T, t0 + 4)))
ev_mask = np.zeros(T, bool); ev_mask[list(ev)] = True
v_ev = float(R[:, ev_mask].var()); v_calm = float(R[:, ~ev_mask].var())
tot = float(R.var())
print(f"\nA. residual variance budget (luma, long-static px): total {tot:.2f} gray^2")
print(f"   temporal: <1Hz {v_low:5.2f} ({100*v_low/tot:.0f}%) | 1-5Hz {v_mid:5.2f} ({100*v_mid/tot:.0f}%) | >5Hz {v_hi:5.2f} ({100*v_hi/tot:.0f}%)")
print(f"   spatial : lowpass16px {v_sp_low:5.2f} ({100*v_sp_low/tot:.0f}%) | detail {v_sp_hi:5.2f} ({100*v_sp_hi/tot:.0f}%)")
print(f"   event   : within+-3 of on/off {v_ev:5.2f} ({100*v_ev/tot:.0f}%, {ev_mask.sum()} frames) | calm {v_calm:5.2f} ({100*v_calm/tot:.0f}%)")

# ---------------- B. metric domain (the 2.67): pair diffs at GT-static pixels
ss = static_std_pairs(out.astype(np.uint8), cl.astype(np.uint8))  # per pair, pooled ch
print(f"metric (mean-of-pairs) full frame: {np.nanmean(ss):.3f}")
# same metric with sync strip excluded (proper masked std, not zero-fill)
from metrics2 import pair_ok
ss2 = []
out8 = out.astype(np.uint8); cl8 = cl.astype(np.uint8)
good = (q > 0.7) & ~lost & ~roll
for t in range(1, T):
    m = stat[t - 1] & scene
    if m.sum() < 500: ss2.append(np.nan); continue
    ss2.append((out8[t][m].astype(np.float32) - out8[t - 1][m].astype(np.float32)).std())
ss2 = np.array(ss2)
gok = pair_ok(ss2, set(np.where(good)[0]))
print(f"metric scene-only (strip excluded): all {np.nanmean(ss2):.3f} | good {np.nanmean(ss2[gok]):.3f}")
dimg = out[1:] - out[:-1]                                        # (T-1,H,W,3) pair diffs
ev_pair = np.array([bool(ev_mask[t] or ev_mask[t - 1]) for t in range(1, T)])
def pooled_std(mask_pairs, img_filter=None):
    acc = []
    for t in np.where(mask_pairs)[0]:
        d = dimg[t][stat[t]]
        if img_filter is not None:
            d = img_filter(d)
        acc.append(d.ravel())
    v = np.concatenate(acc)
    return float(v.std()), len(v)
s_all, n_all = pooled_std(np.ones(T - 1, bool))
s_ev, n_ev = pooled_std(ev_pair)
s_cal, n_cal = pooled_std(~ev_pair)
lp = lambda d: d - cv2.GaussianBlur(d, (0, 0), 8)  # highpass
s_lo, _ = pooled_std(np.ones(T - 1, bool), lambda d: cv2.GaussianBlur(d, (0, 0), 8))
s_hi, _ = pooled_std(np.ones(T - 1, bool), lp)
print(f"\nB. pair-diff std (metric domain): overall {s_all:.3f} (metric quotes {np.nanmean(ss):.3f} as mean-of-pairs)")
print(f"   event pairs: {s_ev:.3f} ({100*n_ev/n_all:.0f}% of pooled px) | calm pairs: {s_cal:.3f}")
print(f"   pair-diff spatial: lowpass16px {s_lo:.3f} | highpass {s_hi:.3f}")

# ---------------- C. EFS activation on the real clip
try:
    real = np.load("real_mine.npy")
    rmeta = pickle.load(open("real_meta.pkl", "rb"))
    valid = [bool(v) for v in rmeta.get("valid", [True] * len(real))]
    cfg = {**DEFAULTS, "efs_w": 0.15}
    k_mad = cfg["efs_k"]; floor = cfg["efs_floor"]; thr_fl = cfg["tp_flow"]
    guides = [guide_of(cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32), 0.8) for f in real]
    n_pairs = gate3000 = spiked = 0
    gate_store = {}
    for t in range(1, len(real)):
        if not (valid[t] and valid[t - 1]): continue
        n_pairs += 1
        f1 = flow(guides[t - 1], guides[t]); f2 = flow(guides[t], guides[t - 1])
        g = np.maximum(np.hypot(f1[..., 0], f1[..., 1]), np.hypot(f2[..., 0], f2[..., 1])) < thr_fl
        for u in (t - 1, t + 1):
            pass
        gate_store[t] = g
    for t, g in gate_store.items():
        gg = g.copy()
        for u in (t - 1, t + 1):
            if u in gate_store: gg &= gate_store[u]
        if gg.sum() >= 3000: gate3000 += 1
    if gate3000:
        out3 = [f.astype(np.float32) for f in real]
        dvals = {}
        for t, g in gate_store.items():
            gg = g.copy()
            for u in (t - 1, t + 1):
                if u in gate_store: gg &= gate_store[u]
            if gg.sum() < 3000: continue
            dm = np.abs(out3[t] - out3[t - 1]).mean(2)[gg]
            if float((dm > max(3 * np.median(dm), 12.0)).mean()) < cfg["efs_frac"]: continue
            dvals[t] = dm.mean()
        if len(dvals) >= 8:
            vals = np.array(list(dvals.values()))
            med = np.median(vals); mad = 1.4826 * np.median(np.abs(vals - med)) + 1e-3
            sthr = med + k_mad * mad + floor
            spiked = int((vals > sthr).sum())
    print(f"\nC. EFS on REAL clip ({len(real)} frames): pairs {n_pairs}, pairs with >=3000 triple-static px: {gate3000}, spike pairs: {spiked}")
    print(f"   activation rate vs metric pairs: {100*spiked/max(n_pairs,1):.1f}%")
except FileNotFoundError as e:
    print("\nC. real clip not available:", e)
