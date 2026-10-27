"""Prompt 1 ATTRIBUTION RUN (session 2026-10-27).
RUN A: best baseline (med3 + temporal NLM h18, exact run_baselines.py params) PRECEDED by
our sync recovery stage. Also scores sync-only (no denoise) as a diagnostic.
Question: is the margin a roll corrector (sync), or do stages 2-9 earn their keep?
Protocol: test clip, scored once, no tuning here."""
import numpy as np, cv2, json, pickle, time
from skimage.metrics import structural_similarity as ssim
from restore import sync_recover, to_ycc
from baselines import nlm_multi, med3
from metrics2 import psnr, flicker_summary

t0 = time.time()
cl = np.load("test_clean.npy"); dg = np.load("test_deg.npy")
meta = pickle.load(open("test_meta.pkl", "rb"))
q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
n = len(cl)
groups = {"all": np.ones(n, bool), "good": (q > 0.7) & ~lost & ~roll,
          "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}

# --- sync recovery stage only (diagnostic)
sl = []
ycc2, valid, quals = sync_recover([to_ycc(f) for f in dg], sl)
synced = np.array([cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB) for y in ycc2])
np.save("abl_sync_only.npy", synced)
print(f"sync done, valid={sum(valid)}/{len(valid)}, {time.time()-t0:.0f}s", flush=True)

# --- RUN A: sync -> med3 -> NLM (exact baseline params)
runA = nlm_multi(list(med3(synced)), 18, 18)
np.save("abl_run_a.npy", runA)
print(f"run A done {time.time()-t0:.0f}s", flush=True)

res = {}
for name, x in [("sync_only", synced), ("runA_sync_med3nlm", runA)]:
    p = {g: float(np.mean([psnr(x[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(x[i], cl[i], channel_axis=2, data_range=255) for i in np.where(m)[0]]))
         for g, m in groups.items()}
    f = flicker_summary(x, cl, meta)
    res[name] = dict(p=p, s=s, f=f, runtime_s=round(time.time() - t0, 1))
    print(name, {k: round(v, 2) for k, v in p.items()}, flush=True)
    print("  ssim", {k: round(v, 3) for k, v in s.items()}, flush=True)
    print("  flick", {k: round(v, 2) for k, v in f.items()}, flush=True)
json.dump(res, open("run_a.json", "w"), indent=1)

# --- attribution table (prior numbers from final_scores.json, labeled by source)
try:
    fs = json.load(open("final_scores.json"))
    keys = ("degraded input", "median3 + NLM", "MINE (locked)")
    prior = {k: fs[k]["p"]["all"] for k in keys if k in fs}
    ss = {k: fs[k]["s"]["all"] for k in keys if k in fs}
except Exception:
    prior, ss = {}, {}
print("\n--- ATTRIBUTION (test clip, PSNR/SSIM all frames) ---", flush=True)
rows = [("sync_only (new)", res["sync_only"]["p"]["all"], res["sync_only"]["s"]["all"], "MEASURED"),
        ("runA sync+med3nlm", res["runA_sync_med3nlm"]["p"]["all"], res["runA_sync_med3nlm"]["s"]["all"], "MEASURED")]
if prior:
    rows = ([("degraded input", prior["degraded input"], ss["degraded input"], "PRIOR"),
             ("median3 + NLM", prior["median3 + NLM"], ss["median3 + NLM"], "PRIOR")] + rows +
            [("MINE (locked)", prior["MINE (locked)"], ss["MINE (locked)"], "PRIOR")])
for name, p, s, tag in rows:
    print(f"{name:20s} {p:6.2f} / {s:.3f}   [{tag}]", flush=True)
if "median3 + NLM" in prior and "MINE (locked)" in prior:
    mA = res["runA_sync_med3nlm"]["p"]["all"]
    print(f"\nsync value for the baseline  : {mA - prior['median3 + NLM']:+.2f} dB", flush=True)
    print(f"stages 2-9 over sync+baseline: {prior['MINE (locked)'] - mA:+.2f} dB", flush=True)
else:
    print("(prior final_scores.json not present; margins computed locally)", flush=True)
