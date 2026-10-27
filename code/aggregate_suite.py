"""Aggregate the Gap-3 suite: paired deltas ours-vs-bar (and vs naive), cluster bootstrap
over non-overlapping 60-frame super-windows (adjacent stride-20 windows share content, so
the cluster is the honest unit), per-clip win rates, per-damage margins.

Usage: python aggregate_suite.py SUITE_DIR [OUT.json]
Reads SUITE_DIR/clip_*.json (written by run_suite.py)."""
import os, sys, json, glob
import numpy as np

def load(dirp):
    clips = {}
    for f in sorted(glob.glob(os.path.join(dirp, "clip_*.json"))):
        r = json.load(open(f)); clips[r["idx"]] = r
    return clips

def cluster_map(clips):
    """window -> super-window: starts within the same 60-frame span (same src) form one
    cluster. Proper gap-based grouping: sort starts, cut a new cluster when the gap to the
    previous start is >= 60 (no shared content)."""
    by_src = {}
    for r in clips.values():
        by_src.setdefault(r["src"], []).append(r["start"])
    def cluster_of(r):
        for s0 in cluster_starts[r["src"]]:
            if r["start"] < s0 + 60: return (r["src"], s0)
        return (r["src"], r["start"])
    cluster_starts = {}
    for src, starts in by_src.items():
        starts = sorted(set(starts)); cs = []
        for st in starts:
            if not cs or st - cs[-1] >= 60: cs.append(st)
        cluster_starts[src] = cs
    return {idx: cluster_of(r) for idx, r in clips.items()}

def paired_stats(deltas_by_clip, clusters, n_boot=10000, seed=5):
    """deltas_by_clip: {idx: d}. Cluster bootstrap: resample clusters (with their clips),
    then resample clips within clusters. Returns mean, lo, hi (95%), win_rate."""
    rng = np.random.default_rng(seed)
    items = sorted(deltas_by_clip.items())
    by_cl = {}
    for idx, d in items: by_cl.setdefault(clusters[idx], []).append(d)
    cl_keys = sorted(by_cl)
    obs = float(np.mean([d for _, d in items]))
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(cl_keys), size=len(cl_keys))
        pool = []
        for j in pick:
            arr = by_cl[cl_keys[j]]
            pool.extend(rng.choice(arr, size=len(arr), replace=True))
        means.append(np.mean(pool))
    lo, hi = np.percentile(means, [2.5, 97.5])
    wins = sum(1 for c in cl_keys if np.mean(by_cl[c]) > 0)
    return dict(mean=round(obs, 3), lo=round(float(lo), 3), hi=round(float(hi), 3),
                win_clusters=wins, n_clusters=len(cl_keys),
                win_clips=round(wins / len(cl_keys), 3), n_pairs=len(items))

METRICS = [("p", "all"), ("p", "good"), ("p", "fades"), ("p", "roll"), ("p", "lost"),
           ("s", "all"), ("s", "good"),
           ("f", "flick_static_good"), ("f", "flick_tdiff_good")]

def main():
    dirp = sys.argv[1]; outp = sys.argv[2] if len(sys.argv) > 2 else f"{dirp}/suite_summary.json"
    clips = load(dirp)
    if not clips: sys.exit("no clip_*.json found")
    clusters = cluster_map(clips)
    systems = ["naive", "bar", "ours"]
    means = {s: {} for s in systems}
    for s in systems:
        for a, b in METRICS:
            v = [r[s][a][b] for r in clips.values() if b in r[s][a]]
            means[s]["/".join((a, b))] = round(float(np.mean(v)), 4)
    # paired deltas: ours - bar (the Gap-2 question), ours - naive, bar - naive
    pairs = {}
    for name, (x, y) in [("ours_minus_bar", ("ours", "bar")),
                          ("ours_minus_naive", ("ours", "naive")),
                          ("bar_minus_naive", ("bar", "naive"))]:
        d = {}
        for a, b in METRICS:
            dd = {}
            for idx, r in clips.items():
                if b in r[x][a] and b in r[y][a]:
                    dd[idx] = r[x][a][b] - r[y][a][b]
            if dd:
                st = paired_stats(dd, clusters)
                st["metric"] = f"{a}.{b}"
                d[f"{a}.{b}"] = st
        pairs[name] = d
    out = dict(n_clips=len(clips),
               systems=means,
               paired=pairs,
               runtime_mean_s={s: round(float(np.mean([r[s].get("runtime_s", np.nan)
                                   for r in clips.values()])), 1)
                               for s in ["ours"] if "runtime_s" in clips[list(clips)[0]].get(s, {})})
    json.dump(out, open(outp, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
