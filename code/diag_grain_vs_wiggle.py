"""Is the residual flicker GRAIN (flat pixels) or GEOMETRIC WIGGLE (edges)?
Split static-pixel frame-diffs by local GT gradient; also check row-level level flicker."""
import numpy as np, cv2, pickle
from metrics2 import psnr

cl = np.load("tune_clean.npy")
meta = pickle.load(open("tune_meta.pkl", "rb"))
champ = np.load("champion_out.npy")
dg = np.load("tune_deg.npy")
q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
good = (q > 0.7) & ~lost & ~roll
pairs = [t for t in range(1, 60) if good[t - 1] and good[t]]

g = cv2.cvtColor(cl[10], cv2.COLOR_RGB2GRAY).astype(np.float32)
grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
edges = grad > 60
flat = grad < 8
print(f"pixels: edges {edges.mean()*100:.0f}%  flat {flat.mean()*100:.0f}%")

def dstat(x, name):
    xs = x.astype(np.float32); cs = cl.astype(np.float32)
    d = np.abs(cs[1:] - cs[:-1]).max(3)
    res = {"gt_static_edge": [], "gt_static_flat": []}
    for t in pairs:
        m = d[t - 1] <= 1.0
        res["gt_static_edge"].append((xs[t][m & edges] - xs[t - 1][m & edges]).std())
        res["gt_static_flat"].append((xs[t][m & flat] - xs[t - 1][m & flat]).std())
    e = np.nanmean(res["gt_static_edge"]); f = np.nanmean(res["gt_static_flat"])
    p = psnr(x[pairs], cl[pairs])
    print(f"{name:12s} static-edge diffstd {e:6.2f} | static-flat diffstd {f:6.2f} | good-pair PSNR {p:.2f}")

dstat(cl, "GT")
dstat(dg, "degraded")
dstat(champ, "champion")

# level flicker: per-frame mean over static flat GT pixels, minus GT mean, std across time
xs = champ.astype(np.float32); cs = cl.astype(np.float32)
d = np.abs(cs[1:] - cs[:-1]).max(3) <= 1.0
flatm = d & flat[None]
lm = np.array([xs[t][flatm[t - 1]].mean() - cs[t][flatm[t - 1]].mean() for t in pairs])
print("champion level flicker (std of static-flat mean offset):", round(float(lm.std()), 3))
lg = np.array([dg[t][flatm[t - 1]].mean() - cs[t][flatm[t - 1]].mean() for t in pairs])
print("degraded level flicker:", round(float(lg.std()), 3))
