"""Loop driver: propose -> apply -> gate-check -> blind-judge -> keep or revert.
Usage:
  python3 iterate.py NAME '<json cfg>'            # full loop vs current champion
  python3 iterate.py --seed                       # (re)build champion from DEFAULTS
State: champion.json (cfg+scores), champion_out.npy, ledger.json (append-only).
Hard gates run BEFORE scoring/judging; a failed gate = immediate revert.
Keep rule: all gates pass AND VLM does not prefer champion AND score improves."""
import sys, json, time, os, traceback
import numpy as np
from skimage.metrics import structural_similarity as ssim
from restore import restore, DEFAULTS
from metrics2 import psnr, flicker_summary
from judge import judge

CL = None; DG = None; META = None

def load_data():
    global CL, DG, META
    import pickle
    CL = np.load("tune_clean.npy"); DG = np.load("tune_deg.npy")
    META = pickle.load(open("tune_meta.pkl", "rb"))

def score_all(x):
    q = META["q"]; lost = np.array(META["lost"]); roll = np.array(META["roll"]) > 0
    groups = {"all": np.ones(len(x), bool), "good": (q > 0.7) & ~lost & ~roll,
              "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}
    p = {g: float(np.mean([psnr(x[i], CL[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(x[i], CL[i], channel_axis=2, data_range=255) for i in np.where(m)[0]]))
         for g, m in groups.items()}
    return p, s, flicker_summary(x, CL, META)

def run_gates(cand, champ, base_psnr, runtime, out):
    g = []
    def add(name, ok, why=""):
        g.append({"gate": name, "ok": bool(ok), **({"why": why} if why else {})})
    add("G1_shape_finite", out.shape == CL.shape and out.dtype == np.uint8
        and np.isfinite(out.astype(np.float64)).all())
    if base_psnr is not None:
        add("G0_floor_baseline", cand["p"]["all"] >= base_psnr - 1e-6,
            f"{cand['p']['all']:.2f} vs baseline {base_psnr:.2f}")
    add("G2_core_roll", cand["p"]["roll"] >= champ["p"]["roll"] - 0.5,
        f"{cand['p']['roll']:.2f} vs champ {champ['p']['roll']:.2f}")
    add("G2_core_lost", cand["p"]["lost"] >= champ["p"]["lost"] - 0.5,
        f"{cand['p']['lost']:.2f} vs champ {champ['p']['lost']:.2f}")
    add("G3_psnr_floor", cand["p"]["all"] >= champ["p"]["all"] - 0.10,
        f"{cand['p']['all']:.2f} vs champ {champ['p']['all']:.2f}")
    add("G4_flicker_tdiff", cand["f"]["flick_tdiff_good"] <= champ["f"]["flick_tdiff_good"] * 1.05,
        f"{cand['f']['flick_tdiff_good']:.3f} vs champ {champ['f']['flick_tdiff_good']:.3f}")
    add("G4_flicker_static", cand["f"]["flick_static_good"] <= champ["f"]["flick_static_good"] * 1.15,
        f"{cand['f']['flick_static_good']:.3f} vs champ {champ['f']['flick_static_good']:.3f}")
    add("G5_runtime", runtime <= 300, f"{runtime:.0f}s")
    return g

def score_ok(c, h):
    dp = c["p"]["all"] - h["p"]["all"]; ds = c["s"]["all"] - h["s"]["all"]
    df = c["f"]["flick_tdiff_good"] - h["f"]["flick_tdiff_good"]
    if dp > 0.02: return True, f"psnr +{dp:.3f}"
    if abs(dp) <= 0.02 and (ds > 0.001 or df < -0.01):
        return True, f"tie psnr ({dp:+.3f}) with ssim {ds:+.4f} / flicker {df:+.4f}"
    # flicker-trade clause: a large flicker reduction may buy a small PSNR cost
    if df < -0.15 and dp >= -0.05 and ds >= -0.002:
        return True, f"flicker trade: dflick {df:+.3f} for dpsnr {dp:+.3f}, dssim {ds:+.4f}"
    return False, f"no improvement (dpsnr {dp:+.3f}, dssim {ds:+.4f}, dflick {df:+.4f})"

def ledger(e):
    with open("ledger.json", "a") as f: f.write(json.dumps(e) + "\n")

def main():
    if len(sys.argv) < 2: sys.exit("usage: iterate.py NAME CFG | iterate.py --seed")
    load_data()
    base_psnr = None
    if os.path.exists("tune_baselines.json"):
        base_psnr = json.load(open("tune_baselines.json")).get("med3_nlm_psnr_all")

    if sys.argv[1] == "--seed":
        t0 = time.time(); out = restore(DG, cfg={}); rt = time.time() - t0
        p, s, f = score_all(out)
        champ = dict(name="champion_seed", cfg={}, p=p, s=s, f=f, runtime=rt)
        json.dump(champ, open("champion.json", "w"), indent=1)
        np.save("champion_out.npy", out)
        ledger({"event": "seed", "scores": {"p": p, "s": s, "f": f}, "runtime": rt})
        print(json.dumps({"p": p, "s": s, "f": f, "runtime": rt}, indent=1))
        return

    name = sys.argv[1]; cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    champ = json.load(open("champion.json"))
    # variants build on the champion: merge new knobs over the champion's non-default cfg
    from restore import DEFAULTS as DEF
    base = {k: v for k, v in champ.get("cfg", {}).items() if DEF.get(k) != v}
    cfg = {**base, **cfg}
    if not base and not cfg and name != "champion_seed":
        print("variant == seed champion; skipping"); return

    print(f"[{name}] running pipeline ...", flush=True)
    t0 = time.time()
    try:
        out = restore(DG, cfg=cfg)
    except Exception as e:
        ledger({"event": "revert", "name": name, "cfg": cfg, "reason": f"crash: {e}"})
        print("CRASH -> revert:", e); traceback.print_exc(); return
    rt = time.time() - t0
    try:
        p, s, f = score_all(out)
    except Exception as e:
        ledger({"event": "revert", "name": name, "cfg": cfg, "reason": f"metric crash: {e}"})
        print("METRIC CRASH -> revert:", e); return
    cand = dict(p=p, s=s, f=f)

    # ---- HARD GATES FIRST (vs champion), then score, then blind judge
    gates = run_gates(cand, champ, base_psnr, rt, out)
    failed = [g for g in gates if not g["ok"]]
    if failed:
        ledger({"event": "revert", "name": name, "cfg": cfg, "gates": gates,
                "scores": {"p": p, "s": s, "f": f}, "runtime": rt})
        print(f"[{name}] GATE FAIL -> revert: {[(g['gate'], g.get('why')) for g in failed]}")
        return
    ok, why = score_ok(cand, champ)
    v = {"votes": None, "details": None}
    if ok:
        champ_out = np.load("champion_out.npy")
        print(f"[{name}] gates pass, score ok ({why}) -> blind judge ...", flush=True)
        v = judge(out, champ_out, CL)
        vlm_ok = v["votes"]["cand"] >= v["votes"]["champ"]
        if not vlm_ok:
            ledger({"event": "revert", "name": name, "cfg": cfg, "gates": gates,
                    "scores": {"p": p, "s": s, "f": f}, "judge": v["votes"],
                    "reason": f"VLM prefers champion ({why})", "runtime": rt})
            print(f"[{name}] VLM REJECT -> revert. votes={v['votes']}")
            for d in v["details"]: print("   ", d.get("trial"), "->", d.get("better"), "|", d.get("reason", d.get("raw", ""))[:80])
            return
        ledger({"event": "keep", "name": name, "cfg": cfg, "gates": gates,
                "scores": {"p": p, "s": s, "f": f}, "judge": v["votes"],
                "reason": why, "runtime": rt})
        json.dump(dict(name=name, cfg={**DEFAULTS, **cfg}, p=p, s=s, f=f, runtime=rt),
                  open("champion.json", "w"), indent=1)
        np.save("champion_out.npy", out); np.save(f"results/variants/{name}.npy", out)
        print(f"[{name}] KEEP. votes={v['votes']}")
        for d in v["details"]: print("   ", d.get("trial"), "->", d.get("better"), "|", d.get("reason", d.get("raw", ""))[:80])
    else:
        ledger({"event": "revert", "name": name, "cfg": cfg, "gates": gates,
                "scores": {"p": p, "s": s, "f": f}, "reason": f"score gate: {why}", "runtime": rt})
        print(f"[{name}] REVERT (no improvement): {why}")
        for d in [(g["gate"], g.get("why")) for g in gates]: print("   ", d)

if __name__ == "__main__":
    main()
