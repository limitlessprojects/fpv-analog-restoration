"""NI metric distribution across suite windows (champion cfg, loo + save_masks on).
Collects provenance / loo_fill / hf_retention per window so gate thresholds come from
a distribution, not a single clip (rule: no single-clip thresholds).
Usage: python3 ni_dist.py <start> [<start> ...]   (suite window starts, seed 7)"""
import sys, os, json, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NI_SKIP_LPIPS", "1")
from run_suite import build_clip
import restore as R
import no_invention as NI

champ = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "champion.json")))["cfg"]
os.makedirs("../results/ni_dist", exist_ok=True)

for start in [int(a) for a in sys.argv[1:]]:
    tag = f"w{start:03d}_s7"
    outp = f"../results/ni_dist/{tag}.json"
    if os.path.exists(outp):
        print(tag, "exists, skip", flush=True); continue
    cl, dg, meta = build_clip("cockatoo", start, 7)
    log = {}
    cfg = {**champ, "loo": 1, "save_masks": 1}
    t0 = time.time()
    out = R.restore(dg, log=log, cfg=cfg)
    rt = time.time() - t0
    np.save(f"/tmp/ni_{tag}_out.npy", out)
    pickle.dump({"provenance": log["provenance"], "loo": log.get("loo"),
                 "_masks": log.get("_masks")}, open(f"/tmp/ni_{tag}_log.pkl", "wb"))
    np.save("/tmp/ni_cl.npy", cl); np.save("/tmp/ni_dg.npy", dg)
    pickle.dump(meta, open("/tmp/ni_meta.pkl", "wb"))
    m = NI.score(f"/tmp/ni_{tag}_out.npy", "/tmp/ni_cl.npy", "/tmp/ni_dg.npy",
                 "/tmp/ni_meta.pkl", f"/tmp/ni_{tag}_log.pkl")
    m.update(start=start, seed=7, runtime_s=round(rt, 1))
    json.dump(m, open(outp, "w"), indent=1)
    loo = m["loo_fill"]
    print(f"{tag} obs={m['provenance']['mean_observed']:.3f} "
          f"loo mean={loo['mean'] and round(loo['mean'],2)} p95max={loo['p95_max'] and round(loo['p95_max'],1)} "
          f"hf={m['hf_retention']['retention_good_observed']} ({rt:.0f}s)", flush=True)
