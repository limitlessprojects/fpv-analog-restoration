"""Gap 1 FAIR chroma test (session 2026-10-27): flugbild (FPV_DVR_CLEANER) and AnaHD
classic on CHROMA-DOMINANT damage - their design envelope - against our pipeline and the
naive/bar baselines. Split verdict is the expected outcome; goal is to report WHERE they win.

Protocol: clip 0 is the tune clip (both tools get their setting grid tuned here, selected on
CHROMA PSNR - their home metric); the remaining clips are scored ONCE with the chosen
settings. Systems: degraded, naive med3+nlm, bar sync+med3+nlm, ours (champion P22),
flugbild, anahd. Metrics: chroma_summary (plane PSNR, chroma flicker, hue/sat) + all-frame
PSNR/SSIM/flicker for context.

Usage: python run_chroma.py OUTDIR S0 S1 S2 ...   (S = "start:seed" pairs; first is tune)
Env: REFS_DIR points at the dir containing fpv_dvr_cleaner/ and anahd/ clones."""
import os, sys, json, time, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_chroma
from baselines import nlm_multi, med3
from restore import restore, sync_recover, to_ycc
from metrics2 import psnr, flicker_summary
from metrics_chroma import chroma_summary
from skimage.metrics import structural_similarity as ssim
import cv2
import run_refs  # write_mp4 / read_mp4 / run_flugbild / run_anahd

CLIPS = [tuple(s.split(":")) for s in sys.argv[2:]]
CLIPS = [(int(a), int(b)) for a, b in CLIPS]
OUTDIR = sys.argv[1]; os.makedirs(OUTDIR, exist_ok=True)

def groups_of(meta):
    n = len(meta["q"]); q = meta["q"]
    lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
    return {"all": np.ones(n, bool), "good": (q > 0.7) & ~lost & ~roll}

def full_score(x, cl, meta):
    g = groups_of(meta)
    p = {k: float(np.mean([psnr(x[i], cl[i]) for i in np.where(m)[0]])) for k, m in g.items()}
    s = {k: float(np.mean([ssim(x[i], cl[i], channel_axis=2, data_range=255)
                           for i in np.where(m)[0]])) for k, m in g.items()}
    out = dict(p=p, s=s, f=flicker_summary(x, cl, meta))
    out["chroma"] = chroma_summary(x, cl, meta)
    return out

def build_clip(start, seed):
    rgb = sim_chroma.load_clean(start=start)
    cl = sim_chroma.clean_frames(rgb)
    dg, meta = sim_chroma.degrade_chroma(cl, seed=seed)
    return cl, dg, meta

def naive(dg):   return np.array(nlm_multi(list(med3(dg)), 18, 18))

def bar(dg):
    ycc, valid, _ = sync_recover([to_ycc(f) for f in dg], [])
    synced = [cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB) for y in ycc]
    return np.array(nlm_multi(list(med3(synced)), 18, 18))

results = {}
t_start = time.time()
for ci, (start, seed) in enumerate(CLIPS):
    tag = f"c{ci}_s{start}_k{seed}"
    cl, dg, meta = build_clip(start, seed)
    tmp = os.path.abspath(f"{OUTDIR}/in_{tag}.mp4"); run_refs.write_mp4(dg, tmp)
    r = {"degraded": full_score(dg, cl, meta)}
    r["naive"] = full_score(naive(dg), cl, meta)
    r["bar"] = full_score(bar(dg), cl, meta)
    champ = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "champion.json")))
    t0 = time.time(); x = restore(dg, cfg=champ["cfg"]); rt = time.time() - t0
    r["ours"] = full_score(x, cl, meta); r["ours"]["runtime_s"] = round(rt, 1)
    if ci == 0:
        grid = {}
        for rm, fb in [(50, 50), (80, 70)]:
            t0 = time.time()
            x = run_refs.run_flugbild(tmp, os.path.abspath(f"{OUTDIR}/flug_{rm}_{fb}.mp4"), rm, fb)
            grid[f"flugbild rm{rm} fb{fb}"] = (full_score(x, cl, meta), round(time.time() - t0, 1))
        for nl in (True, False):
            t0 = time.time()
            x = run_refs.run_anahd(tmp, os.path.abspath(f"{OUTDIR}/anahd_{int(nl)}.mp4"), nl)
            grid[f"anahd no_look={nl}"] = (full_score(x, cl, meta), round(time.time() - t0, 1))
        r["tune_grid"] = {k: v[0] for k, v in grid.items()}
        r["tune_grid_runtime"] = {k: v[1] for k, v in grid.items()}
        flug_best = max((k for k in grid if k.startswith("flug")),
                        key=lambda k: grid[k][0]["chroma"]["chroma"])
        anahd_best = max((k for k in grid if k.startswith("anahd")),
                         key=lambda k: grid[k][0]["chroma"]["chroma"])
        choice = {"flugbild": flug_best, "anahd": anahd_best}
        json.dump(choice, open(f"{OUTDIR}/chroma_choice.json", "w"), indent=1)
        r["chosen"] = choice
        results[tag] = r
        results["_choice"] = choice
    else:
        choice = json.load(open(f"{OUTDIR}/chroma_choice.json"))
        fb_set = choice["flugbild"]; rm, fb = int(fb_set.split("rm")[1].split()[0]), int(fb_set.split("fb")[1])
        x = run_refs.run_flugbild(tmp, os.path.abspath(f"{OUTDIR}/flug_test_{tag}.mp4"), rm, fb)
        r["flugbild"] = full_score(x, cl, meta)
        x = run_refs.run_anahd(tmp, os.path.abspath(f"{OUTDIR}/anahd_test_{tag}.mp4"), "True" in choice["anahd"])
        r["anahd"] = full_score(x, cl, meta)
        results[tag] = r
    os.remove(tmp)
    json.dump(results, open(f"{OUTDIR}/chroma_scores.json", "w"), indent=1)
    keys = ["degraded", "naive", "bar", "ours", "flugbild", "anahd"]
    line = f"{tag} ({time.time()-t_start:.0f}s): "
    for k in keys:
        if k in r: line += f"{k} cPSNR {r[k]['chroma']['chroma']:.2f} | "
    print(line, flush=True)

# aggregate
print("\n=== CHROMA TEST AGGREGATE (tune clip excluded from competitor means) ===", flush=True)
sysnames = ["degraded", "naive", "bar", "ours", "flugbild", "anahd"]
test_tags = [t for t in results if t.startswith("c") and not t.startswith("c0_")]
agg = {}
for s in sysnames:
    vals = {}
    for metric in [("chroma", "chroma"), ("chroma", "Y"), ("chroma", "chroma_flicker_good"),
                   ("chroma", "hue_spread_deg"), ("chroma", "sat_ratio"), ("p", "all"), ("s", "all"),
                   ("f", "flick_static_good")]:
        v = [results[t][s][metric[0]][metric[1]] for t in test_tags
             if s in results[t] and metric[1] in results[t][s].get(metric[0], {})]
        if v: vals["/".join(metric)] = round(float(np.mean(v)), 3)
    agg[s] = vals
    print(s, vals, flush=True)
json.dump({"per_clip": results, "aggregate": agg}, open(f"{OUTDIR}/chroma_final.json", "w"), indent=1)
print("done ->", f"{OUTDIR}/chroma_final.json", flush=True)
