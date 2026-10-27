"""No-invention gate scorer (session 2026-10-27, groundwork for loop priority #3).

Inputs: restoration output (npy), clean + degraded + meta of the same clip, and the
pipeline log written by baseline_ni.py (provenance per frame + LOO fill stats + masks).

Metrics (all GT-assisted, evaluation-side only):
  1. provenance  : per-frame observed / warped / interpolated fractions (from the restore log)
  2. loo_fill    : leave-one-out fill instability (mean / p95 over masked pixels, pooled
                   pass1+pass2). High p95 = fills that lean on a single observation = risk.
  3. hf_retention: high-frequency energy std(out - blur) / std(clean - blur) on OBSERVED
                   pixels only (the pipeline may not destroy detail where it changed nothing).
                   Also reported vs degraded for reference (link attenuation floor).
  4. lpips       : optional (needs torch+lpips); primary perceptual metric once installed.

Gate proposal is produced by baseline_ni.py after measuring the champion + known-bad outputs.
"""
import os, sys, json, pickle
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def hf_energy(x, sigma=2.0):
    """per-frame std of the highpass (x - blur) on luma"""
    ys = [cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32) for f in x]
    return np.array([float((y - cv2.GaussianBlur(y, (0, 0), sigma)).std()) for y in ys])


def score(out_npy, clean_npy, deg_npy, meta_pkl, log_pkl):
    out = np.load(out_npy); cl = np.load(clean_npy); dg = np.load(deg_npy)
    meta = pickle.load(open(meta_pkl, "rb")); log = pickle.load(open(log_pkl, "rb"))
    masks = log.get("_masks")
    q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
    good = (q > 0.7) & ~lost & ~roll
    T = len(out)

    prov = log.get("provenance", [])
    prov_agg = dict(
        interp_frames=int(sum(1 for p in prov if p["interpolated"])),
        mean_observed=float(np.mean([p["observed_frac"] for p in prov])) if prov else None,
        mean_warped=float(np.mean([p["warped_frac"] for p in prov])) if prov else None)

    loo = log.get("loo") or []
    loo_agg = dict(frames_with_fill=len(loo),
                   mean=float(np.mean([e["mean"] for e in loo])) if loo else None,
                   p95_mean=float(np.mean([e["p95"] for e in loo])) if loo else None,
                   p95_max=float(np.max([e["p95"] for e in loo])) if loo else None)

    hf_out, hf_cl, hf_dg = hf_energy(out), hf_energy(cl), hf_energy(dg)
    ret_frames = []
    for t in range(T):
        if masks is None or masks[t] is None or not good[t]:
            continue
        obs = ~masks[t]
        if obs.sum() < 500: continue
        yo = cv2.cvtColor(out[t], cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
        yc = cv2.cvtColor(cl[t], cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
        ho = yo - cv2.GaussianBlur(yo, (0, 0), 2.0)
        hc = yc - cv2.GaussianBlur(yc, (0, 0), 2.0)
        ret_frames.append(ho[obs].std() / (hc[obs].std() + 1e-6))
    hf = dict(retention_good_observed=round(float(np.mean(ret_frames)), 4) if ret_frames else None,
              n_frames=len(ret_frames),
              hf_out_all=round(float(np.mean(hf_out)), 3),
              hf_clean_all=round(float(np.mean(hf_cl)), 3),
              hf_degraded_all=round(float(np.mean(hf_dg)), 3))

    lp = None
    try:
        import torch, lpips as _lp
        loss_fn = _lp.LPIPS(net="alex")
        xs = torch.tensor(np.array(out).transpose(0, 3, 1, 2) / 127.5 - 1, dtype=torch.float32)
        cs = torch.tensor(np.array(cl).transpose(0, 3, 1, 2) / 127.5 - 1, dtype=torch.float32)
        with torch.no_grad():
            v = loss_fn(xs, cs).mean().item()
        lp = {"lpips_out_vs_clean": round(v, 4)}
    except Exception as e:
        lp = {"lpips": "unavailable", "why": str(e)[:80]}

    return dict(provenance=prov_agg, loo_fill=loo_agg, hf_retention=hf, lpips=lp)


if __name__ == "__main__":
    if len(sys.argv) >= 6:
        out_npy, clean_npy, deg_npy, meta_pkl, log_pkl = sys.argv[1:6]
        d, tail = ".", None
        if len(sys.argv) > 6: tail = sys.argv[6]
    else:
        d = sys.argv[1] if len(sys.argv) > 1 else "."
        out_npy, clean_npy, deg_npy = f"{d}/out.npy", f"{d}/clean.npy", f"{d}/deg.npy"
        meta_pkl, log_pkl = f"{d}/meta.pkl", f"{d}/log.pkl"
        tail = sys.argv[2] if len(sys.argv) > 2 else None
    res = score(out_npy, clean_npy, deg_npy, meta_pkl, log_pkl)
    print(json.dumps(res, indent=1))
    if tail: json.dump(res, open(tail, "w"), indent=1)
