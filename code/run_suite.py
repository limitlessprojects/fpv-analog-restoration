"""Gap 3 N-expansion suite (session 2026-10-27): 24 clips x 3 seeds, 3 systems,
paired runs. Clip = 60-frame window; windows come from cockatoo.mp4 (stride 20) and
chelsea.mp4 so content is varied while everything stays CI-reproducible from pinned
imageio sample assets.

Usage: python run_suite.py OUTDIR --shard I --nshards N
       python run_suite.py OUTDIR --only 7        (single clip index, local smoke)
Deterministic clip list: index -> (source, start, seed). Same index = same damage+content
for every system, so deltas are paired at (window, seed) level.

Systems: naive (med3+nlm), bar (sync+med3+nlm, == run A two-stage), ours (champion P22).
Per clip writes OUTDIR/clip_III.json with per-system p/s/f. Aggregation lives in
aggregate_suite.py (cluster bootstrap over non-overlapping 60-frame super-windows)."""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
from baselines import nlm_multi, med3
from restore import restore, sync_recover, to_ycc
from metrics2 import psnr, flicker_summary
from skimage.metrics import structural_similarity as ssim
import cv2

SEEDS = [7, 11, 23]

def clip_list():
    """23 windows x 3 seeds = 69 paired runs. imageio ships exactly ONE standard video
    (cockatoo.mp4, 280 frames), so windows are stride-10 slices of it. Known limitation,
    recorded in the summary: content diversity comes from ONE source; the 3 seeds sample
    damage randomness, not scene diversity. Adjacent windows share up to 50 frames, which
    aggregate_suite handles with super-window clustering (and says out loud)."""
    import imageio.v3 as iio
    n1 = len(iio.imread("imageio:cockatoo.mp4"))
    starts = [s for s in range(0, n1 - 59, 10)][:23]
    clips = []
    for st in starts:
        for seed in SEEDS:
            clips.append((len(clips), "cockatoo", st, seed))
    return clips

_RAW = {}
def load_src(name):
    if name not in _RAW:
        import imageio.v3 as iio
        _RAW[name] = iio.imread(f"imageio:{name}.mp4")
    return _RAW[name]

def build_clip(src, start, seed):
    v = load_src(src)
    frames = v[start:start + 60]
    out = []
    for f in frames:
        h, w = f.shape[:2]
        cw = int(h * sim.W / sim.H); x0 = (w - cw) // 2
        g = cv2.resize(f[:, x0:x0 + cw], (sim.W, sim.H), interpolation=cv2.INTER_AREA)
        out.append(g)
    rgb = np.array(out)
    cl = sim.clean_frames(rgb)
    dg, meta = sim.degrade(cl, seed=seed)
    return cl, dg, meta

def groups_of(meta):
    n = len(meta["q"]); q = meta["q"]
    lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
    return {"all": np.ones(n, bool), "good": (q > 0.7) & ~lost & ~roll,
            "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}

def score(x, cl, meta):
    g = groups_of(meta)
    p = {k: float(np.mean([psnr(x[i], cl[i]) for i in np.where(m)[0]])) for k, m in g.items()}
    s = {k: float(np.mean([ssim(x[i], cl[i], channel_axis=2, data_range=255)
                           for i in np.where(m)[0]])) for k, m in g.items()}
    f = flicker_summary(x, cl, meta)
    return dict(p=p, s=s, f=f)

def naive(dg):   return np.array(nlm_multi(list(med3(dg)), 18, 18))

def bar(dg):
    ycc, valid, _ = sync_recover([to_ycc(f) for f in dg], [])
    synced = [cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB) for y in ycc]
    return np.array(nlm_multi(list(med3(synced)), 18, 18))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--only", type=int, default=-1)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    clips = clip_list()
    todo = [c for c in clips if c[0] % a.nshards == a.shard] if a.only < 0 else [clips[a.only]]
    print(f"shard {a.shard}/{a.nshards}: {len(todo)} clips of {len(clips)}", flush=True)
    champ = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "champion.json")))
    t0 = time.time()
    for (idx, src, start, seed) in todo:
        outp = f"{a.outdir}/clip_{idx:03d}.json"
        if os.path.exists(outp): continue
        cl, dg, meta = build_clip(src, start, seed)
        r = dict(idx=idx, src=src, start=start, seed=seed,
                 roll=[i for i, v in enumerate(meta["roll"]) if v],
                 lost=[i for i, v in enumerate(meta["lost"]) if v])
        r["degraded"] = score(dg, cl, meta)
        r["naive"] = score(naive(dg), cl, meta)
        r["bar"] = score(bar(dg), cl, meta)
        t1 = time.time(); x = restore(dg, cfg=champ["cfg"])
        r["ours"] = score(x, cl, meta); r["ours"]["runtime_s"] = round(time.time() - t1, 1)
        json.dump(r, open(outp, "w"), indent=1)
        print(f"clip {idx:03d} {src} s{start} k{seed} ({time.time()-t0:.0f}s) "
              f"dg {r['degraded']['p']['all']:.2f} naive {r['naive']['p']['all']:.2f} "
              f"bar {r['bar']['p']['all']:.2f} ours {r['ours']['p']['all']:.2f}", flush=True)

if __name__ == "__main__":
    main()
