"""Chroma-dominant damage profile for the FAIR competitor test (Gap 1, session 2026-10-27).

Motivation: flugbild (FPV_DVR_CLEANER) and AnaHD are chroma-focused tools with no sync/roll
recovery. Scoring them on the roll/dropout-heavy default simulator tests them outside their
design envelope. This profile removes every impairment their design does not claim to touch
(roll, lost frames, dropout bands, line jitter) and instead stresses the chroma path:
hue instability, saturation pumping, heavy chroma noise, chroma streaks, chroma-only
multipath echo, and comb-filter leakage (rainbow shimmer) at sharp luma edges.
Luma keeps mild noise + the shared bandwidth limit so frames are not synthetic-clean.

The clean-frame pipeline (pilot, vsync rows, quality curve) is shared with sim.py so the
existing metrics/groups/flicker code works unchanged (roll=0, lost=none everywhere).
"""
import numpy as np, cv2
from sim import load_clean, clean_frames, quality_curve, to_ycc, to_rgb, W, H, BL, PILOT, VSYNC


def degrade_chroma(clean_rgb, seed=7):
    rng = np.random.default_rng(seed)
    n = len(clean_rgb); q = quality_curve(n, rng)
    out = []; meta = dict(q=q, roll=[0] * n, lost=[False] * n)
    theta = 0.0; sat = 1.0
    for t in range(n):
        y = to_ycc(clean_rgb[t]); qq = q[t]; badness = 1 - qq
        # --- chroma-only multipath echo (multipath hits colour difference signals with a
        # different delay than luma in a real analog chain)
        if 28 <= t <= 44:
            k, a = 6, 0.42
            c = y[:, :, 1:]
            g = np.concatenate([np.repeat(c[:, :1], k, 1), c[:, :-k]], 1)
            y[:, :, 1:] = (c + a * g) / (1 + a)
        # --- bandwidth limit (shared with luma sim)
        y[:, :, 0] = cv2.GaussianBlur(y[:, :, 0], (0, 0), sigmaX=0.9, sigmaY=0.01)
        y[:, :, 1:] = cv2.GaussianBlur(y[:, :, 1:], (0, 0), sigmaX=3.0, sigmaY=0.01)
        # --- comb-filter leakage ("rainbow"): high-frequency chroma shimmer scaled by
        # sharp luma edges (the classic cross-luma artefact derainbow filters target)
        edges = np.abs(cv2.Sobel(cv2.GaussianBlur(y[:, :, 0], (0, 0), 1.0),
                                 cv2.CV_32F, 1, 1, ksize=3))
        edges = np.minimum(edges, 220.0) / 220.0
        leak = rng.normal(0, 1, (H, W, 2)) * (18 + 34 * badness) * edges[:, :, None]
        y[:, :, 1:] += leak.astype(np.float32)
        # --- gaussian noise: mild on luma, heavy on chroma (the asymmetry we test)
        sy = 2 + 7 * badness
        sc = 9 + 52 * badness
        y[:, :, 0] += rng.normal(0, sy, (H, W))
        y[:, :, 1:] += rng.normal(0, sc, (H, W, 2))
        # --- chroma streaks: FM clicks that land mostly in the colour channels
        ns = int(2 + 240 * badness ** 2.2)
        for _ in range(ns):
            r = rng.integers(0, H); x = rng.integers(BL, W - 2)
            L = rng.integers(2, 18); amp = rng.choice([-1, 1]) * rng.uniform(70, 170)
            xs = np.arange(L); prof = amp * np.exp(-xs / (L / 2.5))
            xe = min(W, x + L)
            y[r, x:xe, 1:] += prof[:xe - x, None]
            # horizontal smear of the chroma row segment (bleeding)
            y[r, :, 1:] = cv2.blur(y[r, :, 1:], (5, 1))
        # --- hue instability: fast alternating component (temporal tools can win this)
        # + slow drift (nobody can win this; shared handicap)
        theta = 0.9 * theta + rng.normal(0, 10.0)
        theta = float(np.clip(theta, -40, 40))
        hue_fast = rng.normal(0, 11.0)          # per-frame zero-mean jump
        ang = np.deg2rad(theta + hue_fast)
        cr, cb = y[:, :, 1].copy(), y[:, :, 2].copy()
        y[:, :, 1] = np.cos(ang) * cr - np.sin(ang) * cb
        y[:, :, 2] = np.sin(ang) * cr + np.cos(ang) * cb
        # --- saturation pumping (chroma AGC wobble)
        sat = 0.9 * sat + rng.normal(0, 0.05) + 0.1
        g = float(np.clip(sat, 0.55, 1.5))
        y[:, :, 1:] *= g
        # --- pilot stays clean (no sync damage in this profile)
        y[:, :BL, 0] = PILOT[None, :]; y[:, :BL, 1:] = 128
        y[0:4, BL:, 0] = VSYNC[0]; y[4:8, BL:, 0] = VSYNC[1]; y[0:8, BL:, 1:] = 128
        out.append(to_rgb(y))
    return np.array(out), meta


def build(name, start, seed):
    rgb = load_clean(start=start); cl = clean_frames(rgb)
    dg, meta = degrade_chroma(cl, seed=seed)
    np.save(f"{name}_clean.npy", cl); np.save(f"{name}_deg.npy", dg)
    import pickle; pickle.dump(meta, open(f"{name}_meta.pkl", "wb"))
    print(name, cl.shape, "chroma profile, no roll/lost", flush=True)


if __name__ == "__main__":
    import sys
    build(sys.argv[1], start=int(sys.argv[2]), seed=int(sys.argv[3]))
