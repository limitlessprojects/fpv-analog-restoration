"""Flicker / temporal-consistency metrics. These are GT-assisted evaluation metrics only
(the pipeline never sees them at run time)."""
import numpy as np

def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)

def static_std_pairs(x, clean, thr=1.0):
    """For each consecutive pair (t-1,t): std of output diff over pixels where GT is static.
    Residual noise + flicker on non-moving regions; the eye locks onto exactly these."""
    xs = x.astype(np.float32); cs = clean.astype(np.float32)
    d = np.abs(cs[1:] - cs[:-1]).max(3)          # T-1,H,W
    mask = d <= thr
    vals = []
    for t in range(1, len(xs)):
        m = mask[t - 1]
        if m.sum() < 500: vals.append(np.nan); continue
        vals.append((xs[t][m] - xs[t - 1][m]).std())
    return np.array(vals)                        # per pair t-1 -> t

def tdiff_excess_pairs(x, clean, cap=30.0):
    """For each consecutive pair: mean ||out_t - out_{t-1}| - |gt_t - gt_{t-1}||.
    Penalizes added flicker AND lost real motion/detail (both directions bad)."""
    xs = x.astype(np.float32); cs = clean.astype(np.float32)
    do = np.abs(xs[1:] - xs[:-1]).mean(3)
    dc = np.minimum(np.abs(cs[1:] - cs[:-1]).mean(3), cap)
    return np.abs(do - dc).mean(axis=(1, 2))     # per pair

def pair_ok(pairs, idxset):
    """pairs t-1->t kept iff both t-1 and t in idxset"""
    return np.array([i for i in range(len(pairs)) if i in idxset and (i + 1) in idxset], int)

def flicker_summary(x, clean, meta):
    q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
    good = (q > 0.7) & ~lost & ~roll
    ss = static_std_pairs(x, clean)
    te = tdiff_excess_pairs(x, clean)
    g = pair_ok(ss, set(np.where(good)[0]))
    a = pair_ok(ss, {i for i in range(len(ss))} - set(np.where(lost)[0]))
    return dict(
        flick_static_good=float(np.nanmean(ss[g])),
        flick_tdiff_good=float(np.mean(te[g])),
        flick_static_all=float(np.nanmean(ss[a])),
        flick_tdiff_all=float(np.mean(te[a])),
    )
