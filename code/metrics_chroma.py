"""Chroma-specific metrics for the fair competitor test (Gap 1).
All GT-assisted; none of this is used at restoration time."""
import numpy as np, cv2
from metrics2 import psnr, static_std_pairs, pair_ok


def to_ycc_list(rgb):
    return [cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32) for f in rgb]


def plane_psnr(x, cl):
    """mean PSNR per YCrCb plane"""
    xs = to_ycc_list(x); cs = to_ycc_list(cl)
    out = {}
    for i, name in enumerate(["Y", "Cr", "Cb"]):
        m = np.mean([(xs[t][:, :, i] - cs[t][:, :, i]) ** 2 for t in range(len(xs))])
        out[name] = 99.0 if m == 0 else float(10 * np.log10(255 ** 2 / m))
    out["chroma"] = float(np.mean([out["Cr"], out["Cb"]]))
    return out


def chroma_flicker(x, clean, meta):
    """static_std_pairs computed on chroma planes (mean of Cr,Cb), good pairs only."""
    xs = to_ycc_list(x); cs = to_ycc_list(clean)
    q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
    good = (q > 0.7) & ~lost & ~roll
    per_plane = []
    for i in (1, 2):
        xc = np.stack([f[:, :, i] for f in xs])[:, :, :, None]
        cc = np.stack([f[:, :, i] for f in cs])[:, :, :, None]
        per_plane.append(static_std_pairs(xc, cc))
    ss = (per_plane[0] + per_plane[1]) / 2
    g = pair_ok(ss, set(np.where(good)[0]))
    if len(g) == 0: return float("nan")
    return float(np.nanmean(ss[g]))


def hue_sat_error(x, clean):
    """hue error (deg, circular mean over chroma-active pixels, all frames) and
    saturation ratio (mean ||C_out|| / ||C_clean||)."""
    xs = to_ycc_list(x); cs = to_ycc_list(clean)
    ang_errs = []; sat_r = []
    for t in range(len(xs)):
        cx, cc = xs[t][:, :, 1:] - 128, cs[t][:, :, 1:] - 128
        mag_c = np.hypot(cc[:, :, 0], cc[:, :, 1])
        act = mag_c > 15
        if act.sum() < 100: continue
        ax, bx = cx[act, 0], cx[act, 1]
        ac, bc = cc[act, 0], cc[act, 1]
        dot = ax * ac + bx * bc
        cross = ax * bc - bx * ac
        ang_errs.append(np.arctan2(cross, dot))          # signed
        sat_r.append(np.hypot(ax, bx) / (np.hypot(ac, bc) + 1e-6))
    if not ang_errs: return {"hue_err_deg": float("nan"), "sat_ratio": float("nan")}
    ae = np.concatenate(ang_errs)
    # circular mean of signed angle error; spread from the mean resultant length
    hue = float(np.degrees(np.arctan2(np.sin(ae).mean(), np.cos(ae).mean())))
    R = float(np.hypot(np.sin(ae).mean(), np.cos(ae).mean()))
    spread = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))
    return {"hue_bias_deg": hue, "hue_spread_deg": spread,
            "sat_ratio": float(np.mean(np.concatenate(sat_r)))}


def chroma_summary(x, cl, meta):
    out = plane_psnr(x, cl)
    out["chroma_flicker_good"] = chroma_flicker(x, cl, meta)
    out.update(hue_sat_error(x, cl))
    return out
