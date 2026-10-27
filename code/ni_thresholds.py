"""Derive no-invention gate thresholds from the 8-window NI distribution, then
SELF-CHECK: the tune-clip champion (the standing reference) must pass the gate.
Writes results/ni_thresholds.json."""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rows = [json.load(open(p)) for p in sorted(glob.glob("../results/ni_dist/*.json"))]
assert len(rows) >= 8, f"only {len(rows)} distribution rows"
obs = [r["provenance"]["mean_observed"] for r in rows]
warp = [r["provenance"]["mean_warped"] for r in rows]
interp = [r["provenance"]["interp_frames"] for r in rows]
loo_max = [r["loo_fill"]["p95_max"] for r in rows if r["loo_fill"]["p95_max"] is not None]
hf = [r["hf_retention"]["retention_good_observed"] for r in rows if r["hf_retention"]["retention_good_observed"] is not None]

def stats(v):
    return dict(min=round(float(np.min(v)), 4), median=round(float(np.median(v)), 4),
                max=round(float(np.max(v)), 4))

dist = dict(observed=stats(obs), warped=stats(warp), interp_frames=stats(interp),
            loo_p95_max=stats(loo_max), hf_retention=stats(hf))

gate = dict(
    observed_floor=round(float(np.min(obs)) - 0.02, 4),
    interp_ceiling=int(max(2, np.max(interp))),
    loo_p95_max_ceiling=round(float(np.max(loo_max)) * 1.2, 1),
    hf_band=[round(float(np.min(hf)) - 0.08, 4), 1.02],
)
gate["rationale"] = {
    "observed_floor": "dist_min - 0.02: candidate may not observe less of the signal than champion ever does",
    "interp_ceiling": "max(2, dist_max): champion legitimately interpolates <=2 fade frames",
    "loo_p95_max_ceiling": "dist_max * 1.2: fills leaning on single observations beyond champion's worst window",
    "hf_band": "[dist_min - 0.08, 1.02]: below = detail destruction; above 1.0 = more HF than clean on untouched pixels = invention",
}

# ---- self-check: tune-clip champion must pass ----
def gate_ok(m, gate):
    checks = []
    checks.append(("observed", m["provenance"]["mean_observed"] >= gate["observed_floor"]))
    checks.append(("interp", m["provenance"]["interp_frames"] <= gate["interp_ceiling"]))
    if m["loo_fill"]["p95_max"] is not None:
        checks.append(("loo_p95_max", m["loo_fill"]["p95_max"] <= gate["loo_p95_max_ceiling"]))
    hf = m["hf_retention"]["retention_good_observed"]
    if hf is not None:
        checks.append(("hf_band", gate["hf_band"][0] <= hf <= gate["hf_band"][1]))
    return checks

tune = json.load(open("../results/ni_champ_metrics.json")) if os.path.exists("../results/ni_champ_metrics.json") else json.load(open("ni_champ_metrics.json"))
checks = gate_ok(tune, gate)
print("SELF-CHECK tune champion:", checks)
assert all(ok for _, ok in checks), f"champion fails own gate: {checks}"

out = dict(n_clips=len(rows), source="cockatoo windows 0..210 stride 30, seed 7, local run, champion cfg + loo+masks",
           caveat="single content source; thresholds are status-quo guards (catch invention worse than champion), not perceptual absolutes; LPIPS not gated (no distribution)",
           distribution=dist, gate=gate)
json.dump(out, open("../results/ni_thresholds.json", "w"), indent=1)
print(json.dumps(out, indent=1))
