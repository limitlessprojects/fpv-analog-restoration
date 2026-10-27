"""Final evaluation on the TEST clip (scored once, config locked from tuning clip).
PSNR/SSIM by group + flicker metrics for every method with an npy present."""
import numpy as np, pickle, os, json
from skimage.metrics import structural_similarity as ssim
from metrics2 import psnr, flicker_summary

cl = np.load("test_clean.npy"); meta = pickle.load(open("test_meta.pkl", "rb"))
q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
groups = {"all": np.ones(60, bool), "good": (q > 0.7) & ~lost & ~roll,
          "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}

SETS = [("degraded input", "degraded.npy".replace("degraded", "test_deg")), ("hqdn3d default", "b_hqdn3d_def.npy"),
        ("hqdn3d strong", "b_hqdn3d_strong.npy"), ("temporal NLM h12", "b_nlm_h12.npy"),
        ("temporal NLM h25", "b_nlm_h25.npy"), ("median3 + NLM", "b_med3_nlm.npy"),
        ("MINE (locked)", "mine_final.npy")]

res = {}
print(f"{'method':18s}" + "".join(f"{g:>14s}" for g in groups) + "   (PSNR dB / SSIM)")
for name, f in SETS:
    if not os.path.exists(f): continue
    x = np.load(f)
    p = {g: float(np.mean([psnr(x[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(x[i], cl[i], channel_axis=2, data_range=255) for i in np.where(m)[0]]))
         for g, m in groups.items()}
    fl = flicker_summary(x, cl, meta)
    res[name] = dict(p=p, s=s, f=fl)
    print(f"{name:18s}" + "".join(f"{p[g]:9.2f}/{s[g]:.3f}" for g in groups))
print()
print(f"{'method':18s}{'flick_static_good':>19s}{'flick_tdiff_good':>18s}{'flick_static_all':>18s}{'flick_tdiff_all':>17s}")
for name, _ in SETS:
    if name not in res: continue
    f = res[name]["f"]
    print(f"{name:18s}{f['flick_static_good']:19.3f}{f['flick_tdiff_good']:18.3f}{f['flick_static_all']:18.3f}{f['flick_tdiff_all']:17.3f}")
json.dump(res, open("final_scores.json", "w"), indent=1)
