"""Round 2: decompose the failed P4 into its two components + stab-v2 (trajectory smooth).
B1 = stable guide only (old weights)   B2 = z^2 only (noisy guide)
C2 = stab with temporally smoothed offset trajectory (the spec step round 1 omitted)"""
import json, time
import numpy as np, sim, restore as R
from metrics2 import psnr, flicker_summary
from skimage.metrics import structural_similarity as ssim

champ = json.load(open("../results/champion.json"))["cfg"]
variants = {
    "B1_guide": {"fz": 1, "fz_z2": 0},
    "B2_z2": {"fz": 1, "fz_guide": 0},
    "C2_stab_ts": {"stab": 1},
}
rgb = sim.load_clean(start=150)
cl = sim.clean_frames(rgb)
dg, meta = sim.degrade(cl, seed=11)
q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
groups = {"all": np.ones(len(cl), bool), "good": (q > 0.7) & ~lost & ~roll,
          "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}

results = {}
for name, delta in variants.items():
    cfg = {**champ, **delta}
    t0 = time.time()
    out = R.restore(dg, cfg=cfg)
    rt = time.time() - t0
    p = {g: float(np.mean([psnr(out[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(out[i], cl[i], channel_axis=2, data_range=255)
                           for i in np.where(m)[0]])) for g, m in groups.items()}
    f = flicker_summary(out, cl, meta)
    results[name] = dict(delta=delta, p=p, s=s, f=f, runtime=rt)
    np.save(f"screen_{name}.npy", out)
    print(f"{name:12s} " + " ".join(f"{g}:{p[g]:6.2f}" for g in ("all", "good", "fades", "roll", "lost"))
          + f" | flick_static_good {f['flick_static_good']:7.3f} tdiff {f['flick_tdiff_good']:6.3f}"
          + f" | ssim {s['all']:.3f} | {rt:.0f}s", flush=True)
    print("   (A_champ reference: all 27.36 good 28.66 flick 7.477 tdiff 6.224 ssim 0.814)")
json.dump(results, open("screen_p45_r2.json", "w"), indent=1)
print("\nsaved screen_p45_r2.json")
