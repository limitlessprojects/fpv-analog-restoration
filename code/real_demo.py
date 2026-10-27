"""Real-content demo: mixkit drone clip (real footage) pushed through the SAME simulated
analog link, restored with the locked config. GT = marker-added clean, so scores are
honest but note: damage is simulated, content is real."""
import numpy as np, cv2, pickle, json
import sim, restore as R
from metrics2 import psnr, flicker_summary
from skimage.metrics import structural_similarity as ssim

W, H, N = sim.W, sim.H, sim.N
cap = cv2.VideoCapture("/tmp/mixkit_test.mp4")
frames = []
while len(frames) < N:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
v = np.array(frames[:N])
out = []
for f in v:
    h, w = f.shape[:2]
    cw = int(h * W / H); x0 = (w - cw) // 2
    out.append(cv2.resize(f[:, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA))
rgb = np.array(out)
cl = sim.clean_frames(rgb)
dg, meta = sim.degrade(cl, seed=21)
np.save("real_clean.npy", cl); np.save("real_deg.npy", dg)
pickle.dump(meta, open("real_meta.pkl", "wb"))
print("real clip degraded. roll:", [(i, r) for i, r in enumerate(meta["roll"]) if r],
      "lost:", [i for i, l in enumerate(meta["lost"]) if l], flush=True)

cfg = {"efs": True, "efs_w": 0.15, "efs_max": 6}
mine = R.restore(dg, cfg=cfg)
np.save("real_mine.npy", mine)

def rep(x, name):
    q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
    groups = {"all": np.ones(N, bool), "good": (q > 0.7) & ~lost & ~roll,
              "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}
    p = {g: float(np.mean([psnr(x[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(x[i], cl[i], channel_axis=2, data_range=255) for i in np.where(m)[0]]))
         for g, m in groups.items()}
    fl = flicker_summary(x, cl, meta)
    print(f"{name:14s}" + "".join(f"{p[g]:9.2f}/{s[g]:.3f}" for g in groups) +
          f"  flick {fl['flick_static_good']:.2f}/{fl['flick_tdiff_good']:.2f}")
    return dict(p=p, s=s, fl=fl)

print(f"{'':14s}" + "".join(f"{g:>14s}" for g in ["all", "good", "fades", "roll", "lost"]))
r = {}
r["degraded"] = rep(dg, "degraded")
r["mine"] = rep(mine, "mine")
json.dump(r, open("real_scores.json", "w"), indent=1)

# side-by-side video: degraded | restored, 2x slow
import imageio.v2 as iio
rows = [np.hstack([dg[t], mine[t]]) for t in range(N)]
iio.mimsave("real_sidebyside.mp4", rows, fps=12, macro_block_size=1)
print("wrote real_sidebyside.mp4")
