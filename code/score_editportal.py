"""Score editportal.net Quality Boost (Classic) output against the paired baselines
for suite clip 005 (cockatoo window 10, seed 23). Uses the IDENTICAL scoring stack
as run_suite.py (same groups, same flicker_summary) so the comparison is apples to
apples. Records the site settings that produced the output."""
import sys, os, json, pickle
import numpy as np
import imageio.v2 as iio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_suite import score  # same groups + flicker as every suite clip

CL = np.load("/tmp/ep_suite/cl.npy")
DG = np.load("/tmp/ep_suite/dg.npy")
META = pickle.load(open("/tmp/ep_suite/meta.pkl", "rb"))
BASE = json.load(open("/tmp/ep_suite/clip_005.json"))

out = np.array([f for f in iio.get_reader("/home/z/my-project/download/editportal_out_classic.mp4")])
assert out.shape == DG.shape, f"shape mismatch {out.shape} vs {DG.shape}"

ep = score(out, CL, META)
ep["runtime_s"] = None

systems = {"degraded": BASE["degraded"], "naive": BASE["naive"], "bar": BASE["bar"],
           "ours": BASE["ours"], "editportal_classic": ep}

print(f"{'system':20s} {'P all':>7} {'P good':>7} {'P fades':>8} {'P roll':>7} {'P lost':>7} "
      f"{'S all':>7} {'S good':>7} {'flk stat':>9} {'flk tdiff':>10}")
for name, r in systems.items():
    p, s, f = r["p"], r["s"], r["f"]
    print(f"{name:20s} {p['all']:7.2f} {p['good']:7.2f} {p['fades']:8.2f} {p['roll']:7.2f} {p['lost']:7.2f} "
          f"{s['all']:7.3f} {s['good']:7.3f} {f['flick_static_good']:9.2f} {f['flick_tdiff_good']:10.2f}")

result = {
    "what": "editportal.net Quality Boost, Classic mode (denoise 35, sharpen 60, contrast 6, vibrance 10, scale Original), browser-run via Playwright; AI Fast/AI Pro disabled without WebGPU (button disabled), so not scored",
    "clip": {"idx": 5, "src": "cockatoo", "start": 10, "seed": 23, "frames": 60, "size": "480x360"},
    "input_sha_note": "input = suite degraded clip (sync/roll/lost/dropout damage intact; editportal has no analog-specific handling)",
    "scores": systems,
}
os.makedirs("../results", exist_ok=True)
json.dump(result, open("../results/editportal_scores.json", "w"), indent=1)
print("\nwrote results/editportal_scores.json")
