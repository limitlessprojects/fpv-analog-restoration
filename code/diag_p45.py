"""Diagnose WHY the P4/P5A screening candidates lost vs champion (no new pipeline runs).
1. C_stab: per-pair static-std series A vs C, split event/calm; and the temporal roughness
   of the correction field b itself (does the fix inject new pair-diff variance?).
2. B_fz: per-frame PSNR delta (B-A) split by frame groups; mean fusion weight stats."""
import json, pickle
import numpy as np, cv2
import sim, restore as R
from metrics2 import psnr, static_std_pairs, pair_ok

cl = np.load("tune_clean.npy"); dg = np.load("tune_deg.npy")
meta = pickle.load(open("tune_meta.pkl", "rb"))
q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
A = np.load("screen_A_champ.npy").astype(np.float32)
B = np.load("screen_B_fz.npy").astype(np.float32)
C = np.load("screen_C_stab.npy").astype(np.float32)
champ = json.load(open("../results/champion.json"))["cfg"]

# ---- events: frames adjacent (<=3) to impairment on/off
ev = np.zeros(60, bool)
for arr in (roll.astype(int), lost.astype(int), (q < 0.5).astype(int)):
    tr = np.where(np.diff(arr) != 0)[0] + 1
    for t0 in tr: ev[max(0, t0 - 3):t0 + 4] = True

ssA = static_std_pairs(A.astype(np.uint8), cl.astype(np.uint8))
ssC = static_std_pairs(C.astype(np.uint8), cl.astype(np.uint8))
good = (q > 0.7) & ~lost & ~roll
idx = set(np.where(good)[0])
ok = np.array([i in idx and (i + 1) in idx for i in range(59)])
ev_ok = ok & np.array([ev[i] or ev[i + 1] for i in range(59)])
cal_ok = ok & ~np.array([ev[i] or ev[i + 1] for i in range(59)])
print("== C_stab static-std by pair class (good pairs) ==")
print(f"event pairs : A {np.nanmean(ssA[ev_ok]):7.3f}  C {np.nanmean(ssC[ev_ok]):7.3f}")
print(f"calm  pairs : A {np.nanmean(ssA[cal_ok]):7.3f}  C {np.nanmean(ssC[cal_ok]):7.3f}")

# ---- reconstruct the b-field the stabilizer applied to A (stab runs on champion output)
ycc = [cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32) for f in A]
cfg = {**R.DEFAULTS, **champ, "stab": 1}
sig, rad, tile = cfg["stab_sigma"], int(cfg["stab_rad"]), int(cfg["stab_tile"])
H, W = 360, 480; sh, sw = H // 2, W // 2
lows = [cv2.resize(cv2.GaussianBlur(f[:, :, 0], (0, 0), sig), (sw, sh), interpolation=cv2.INTER_AREA) for f in ycc]
gxs, gys = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
valid = [not l for l in lost]
bs = []
for t in range(60):
    if not valid[t]: bs.append(np.zeros((sh, sw), np.float32)); continue
    refs, mags = [], []
    for d in range(-rad, rad + 1):
        u = t + d
        if d == 0 or u < 0 or u >= 60 or not valid[u]: continue
        f = cv2.calcOpticalFlowFarneback(lows[t], lows[u], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        refs.append(cv2.remap(lows[u], gxs + f[..., 0], gys + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        mags.append(np.hypot(f[..., 0], f[..., 1]))
    L = np.median(np.stack(refs), 0)
    dev = cv2.GaussianBlur(cv2.medianBlur((lows[t] - L).astype(np.float32), 5), (0, 0), sig / 2)
    trust = np.median(np.stack(mags), 0) < cfg["stab_flow"]
    nt_y, nt_x = -(-sh // tile), -(-sw // tile)
    pad = np.full((nt_y * tile, nt_x * tile), np.nan, np.float32); pad[:sh, :sw] = dev
    tb = np.nanmedian(pad.reshape(nt_y, tile, nt_x, tile), axis=(1, 3))
    gate = np.kron(np.abs(tb) >= cfg["stab_dev"], np.ones((tile, tile), np.float32))[:sh, :sw] * trust
    bs.append(np.clip(dev * gate, -cfg["stab_max"], cfg["stab_max"]))
bs = np.stack(bs)
dpair_gt = np.abs(cl[1:].astype(np.float32) - cl[:-1].astype(np.float32)).max(3)
statmask = dpair_gt <= 1.0
bfull = np.abs(bs[1:] - bs[:-1])           # (59, sh, sw) per-pixel correction change
print("\n== b-field self-roughness |b_t - b_{t-1}| at GT-static px (gray) ==")
stat_s = np.stack([cv2.resize(statmask[i].astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.5
                   for i in range(59)])
m_ev = np.zeros_like(bfull, bool); m_ev[ev_ok] = stat_s[ev_ok]
m_ca = np.zeros_like(bfull, bool); m_ca[cal_ok] = stat_s[cal_ok]
print(f"event pairs {bfull[m_ev].mean():.3f}   calm pairs {bfull[m_ca].mean():.3f}")
sd = (bs[:, :, :][:, statmask[0]] if False else None)
raw_sd = np.array([bs[t][stat_s[min(t, 58)]].std() for t in range(1, 60)])
print(f"b spatial std per pair: mean {raw_sd[ok].mean():.3f} (adds directly to pair diffs if uncorrelated frame-to-frame)")

# ---- B_fz: per-frame PSNR delta by group
pA = np.array([psnr(A[i], cl[i]) for i in range(60)])
pB = np.array([psnr(B[i], cl[i]) for i in range(60)])
d = pB - pA
print("\n== B_fz per-frame PSNR delta (B-A) ==")
for name, m in [("good", good & ~ev), ("good@event", good & ev), ("fades", (q < 0.5) & ~lost),
                ("roll", roll & ~lost), ("lost", lost)]:
    print(f"{name:11s} mean {d[m].mean():+.3f} dB")
tex = np.array([float(cv2.Sobel(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 1, 1, 3).std()) for f in cl])
mt = np.array([float(np.hypot(cv2.calcOpticalFlowFarneback(
    cv2.cvtColor(cl[i], cv2.COLOR_RGB2GRAY).astype(np.uint8),
    cv2.cvtColor(cl[i + 1], cv2.COLOR_RGB2GRAY).astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)[..., 0],
    cv2.calcOpticalFlowFarneback(
    cv2.cvtColor(cl[i], cv2.COLOR_RGB2GRAY).astype(np.uint8),
    cv2.cvtColor(cl[i + 1], cv2.COLOR_RGB2GRAY).astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)[..., 1]).mean())
    for i in range(59)] + [0.0])
print(f"corr(delta, texture) = {np.corrcoef(d, tex)[0, 1]:+.3f}   corr(delta, motion) = {np.corrcoef(d, mt)[0, 1]:+.3f}")
