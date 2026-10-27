"""Is the Prompt 5A target (slow low-band drift) still the dominant flicker bucket on the
TUNE clip with the CURRENT champion? Same decomposition as diag_flicker2, different input."""
import numpy as np, cv2, pickle
from metrics2 import static_std_pairs

cl = np.load("tune_clean.npy").astype(np.float32)
out = np.load("screen_A_champ.npy").astype(np.float32)
meta = pickle.load(open("tune_meta.pkl", "rb"))
T, H, W = cl.shape[:3]
BL, VROWS = 12, 8
scene = np.zeros((H, W), bool); scene[VROWS:, BL:] = True

clY = np.array([cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2YCrCb)[..., 0] for f in cl]).astype(np.float32)
ouY = np.array([cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2YCrCb)[..., 0] for f in out]).astype(np.float32)
dpair_gt = np.abs(clY[1:] - clY[:-1])
stat = dpair_gt <= 1.0
cnt = stat.sum(0)
long_static = (cnt >= 0.5 * (T - 1)) & scene
R = (ouY - clY)[:, long_static].T.copy()
bad = ~stat[:, long_static].T
bad = np.concatenate([bad, bad[:, -1:]], axis=1)
idx = np.arange(T)
for p in range(R.shape[0]):
    b = bad[p]
    if b.any() and (~b).any():
        R[p] = np.interp(idx, idx[~b], R[p, ~b])
Fs = 30.0
F = np.fft.rfft(R, axis=1)
freqs = np.fft.rfftfreq(T, d=1 / Fs)
def band_var(lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    x = np.fft.irfft(F * m[None, :], n=T, axis=1)
    return float(x.var())
v_low, v_mid, v_hi = band_var(0, 1), band_var(1, 5), band_var(5, Fs / 2 + 1)
res_full = ouY - clY
low = np.stack([cv2.GaussianBlur(f, (0, 0), 8) for f in res_full])
v_sp_low = float(low[:, long_static].var()); v_sp_hi = float((res_full - low)[:, long_static].var())
q = meta["q"]; roll = np.array(meta["roll"]) > 0; lost = np.array(meta["lost"])
ev = set()
for arr in (roll.astype(int), lost.astype(int), (q < 0.5).astype(int)):
    tr = np.where(np.diff(arr) != 0)[0] + 1
    for t0 in tr: ev.update(range(max(0, t0 - 3), min(T, t0 + 4)))
ev_mask = np.zeros(T, bool); ev_mask[list(ev)] = True
tot = float(R.var())
v_ev = float(R[:, ev_mask].var()); v_calm = float(R[:, ~ev_mask].var())
print(f"tune clip, current champion, residual variance budget: total {tot:.2f} gray^2")
print(f"  temporal: <1Hz {v_low:5.2f} ({100*v_low/tot:.0f}%) | 1-5Hz {v_mid:5.2f} ({100*v_mid/tot:.0f}%) | >5Hz {v_hi:5.2f} ({100*v_hi/tot:.0f}%)")
print(f"  spatial : lowpass16px {v_sp_low:5.2f} ({100*v_sp_low/tot:.0f}%) | detail {v_sp_hi:5.2f} ({100*v_sp_hi/tot:.0f}%)")
print(f"  event   : {v_ev:5.2f} ({100*v_ev/tot:.0f}%, {ev_mask.sum()} frames) | calm {v_calm:5.2f} ({100*v_calm/tot:.0f}%)")
ss = static_std_pairs(out.astype(np.uint8), cl.astype(np.uint8))
lo_pair = np.stack([cv2.GaussianBlur((out[i] - out[i - 1]).astype(np.float32), (0, 0), 8) for i in range(1, T)])
hi_pair = np.stack([(out[i] - out[i - 1]).astype(np.float32) - lo_pair[i - 1] for i in range(1, T)])
good = (q > 0.7) & ~lost & ~roll
from metrics2 import pair_ok
g = pair_ok(ss, set(np.where(good)[0]))
lo_v = np.array([lo_pair[i - 1][stat[i - 1]].var() if stat[i - 1].sum() > 500 else np.nan for i in range(1, T)])
hi_v = np.array([hi_pair[i - 1][stat[i - 1]].var() if stat[i - 1].sum() > 500 else np.nan for i in range(1, T)])
print(f"  metric-domain pair-diff variance at good/static: lowpass {np.nanmean(lo_v[g]):.2f} | detail {np.nanmean(hi_v[g]):.2f}")
