"""Append P4/P5A screening verdicts to the ledger (JSONL)."""
import json

E = [
 {"event": "screen", "name": "P4_fz_full", "clip": "tune s150 k11", "cfg_delta": {"fz": 1},
  "scores": {"p_all": 26.02, "flick_static_good": 8.656, "ssim": 0.798},
  "vs_champ": {"dp": -1.34, "dflick": 1.18}, "verdict": "REVERT",
  "why": "stable guide + z2 weighting loses everywhere; loss correlates +0.68 with texture "
         "(grad*delta tolerance admits misaligned texture = melting risk realised); lost +0.19 only"},
 {"event": "screen", "name": "P4a_fz_guide_only", "clip": "tune s150 k11", "cfg_delta": {"fz": 1, "fz_z2": 0},
  "scores": {"p_all": 26.45, "flick_static_good": 8.850, "ssim": 0.805},
  "vs_champ": {"dp": -0.91, "dflick": 1.37}, "verdict": "REVERT",
  "why": "fused-median guide is neighbour-biased: diffs shrink for wrong neighbours too "
         "(circularity); guide noise is not the flicker culprit"},
 {"event": "screen", "name": "P4b_z2_only", "clip": "tune s150 k11", "cfg_delta": {"fz": 1, "fz_guide": 0},
  "scores": {"p_all": 26.45, "flick_static_good": 8.711, "ssim": 0.803},
  "vs_champ": {"dp": -0.91, "dflick": 1.23}, "verdict": "REVERT",
  "why": "z2 component alone equals guide-only damage: on this data Farneback error on texture "
         "samples wrong texels, so |grad|*delta tolerance is the wrong sign"},
 {"event": "screen", "name": "P5A_stab_raw", "clip": "tune s150 k11", "cfg_delta": {"stab": 1},
  "scores": {"p_all": 27.10, "flick_static_good": 7.915, "ssim": 0.808},
  "vs_champ": {"dp": -0.26, "dflick": 0.44}, "verdict": "REVERT",
  "why": "b-field self-roughness 0.81 gray/frame on calm static pairs injects the flicker it "
         "removes; event pairs unchanged (+0.025)"},
 {"event": "screen", "name": "P5A_stab_traj_smooth", "clip": "tune s150 k11",
  "cfg_delta": {"stab": 1, "stab_ts": 2},
  "scores": {"p_all": 27.19, "flick_static_good": 7.553, "ssim": 0.810},
  "vs_champ": {"dp": -0.17, "dflick": 0.076}, "verdict": "REVERT",
  "why": "trajectory smoothing recovers most self-flicker but target is wrong on tune clip: "
         "metric-domain pair-diff variance is 64% detail band (69.6 vs 39.4 lowpass), lowpass "
         "only ~41% of spatial residual; stabilizer caps at ~1/3 of metric and loses to its own "
         "injection"},
 {"event": "diagnostic", "name": "target_recheck_tune",
  "finding": "tune clip + current champion residual: 90% slow temporally (48% <1Hz, 42% 1-5Hz) "
             "but only ~41% lowpass spatially; metric dominated by detail band. Test-clip "
             "93%-lowpass finding does NOT transfer to tune+P22. Next flicker hypothesis must "
             "re-aim at the detail band (temporal_post family), not another low-band stabilizer"},
]
with open("../results/ledger.json", "a") as f:
    for e in E:
        f.write(json.dumps(e) + "\n")

lines = open("../results/ledger.json").read().strip().split("\n")
for l in lines:
    json.loads(l)
print(f"ledger valid, {len(lines)} entries")
