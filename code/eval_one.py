"""Build one clip (start, seed), restore with a config, score it, write JSON.
Config = DEFAULTS (restore.py) + champion cfg (optional) + delta (optional).
Usage:
  python3 eval_one.py OUT.json --start 150 --seed 11 [--champion results/champion.json] \
      [--delta '{"efs_w":0.2}'] [--label my-label]
No state is shared with the local loop: this is the screening/robustness path for
CI runners. It never writes npy files and never touches the held-out test clip.
"""
import argparse, json, time
import numpy as np
import sim, restore as R
from metrics2 import psnr, flicker_summary
from skimage.metrics import structural_similarity as ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--champion", default=None)
    ap.add_argument("--delta", default="{}")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    cfg = {}
    if a.champion:
        cfg.update(json.load(open(a.champion)).get("cfg", {}))
    cfg.update(json.loads(a.delta))

    rgb = sim.load_clean(start=a.start)
    cl = sim.clean_frames(rgb)
    dg, meta = sim.degrade(cl, seed=a.seed)

    t0 = time.time()
    out = R.restore(dg, cfg=cfg)
    rt = time.time() - t0

    q = meta["q"]; lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
    groups = {"all": np.ones(len(out), bool), "good": (q > 0.7) & ~lost & ~roll,
              "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}
    p = {g: float(np.mean([psnr(out[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(out[i], cl[i], channel_axis=2, data_range=255) for i in np.where(m)[0]]))
         for g, m in groups.items()}
    f = flicker_summary(out, cl, meta)
    pin = {g: float(np.mean([psnr(dg[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    res = {"label": a.label or f"s{a.start}k{a.seed}", "start": a.start, "seed": a.seed,
           "cfg": cfg, "p": p, "s": s, "f": f, "p_degraded": pin, "runtime": rt}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"label": res["label"], "p": p, "flick_static_good": f["flick_static_good"],
                      "flick_tdiff_good": f["flick_tdiff_good"], "runtime": rt}))


if __name__ == "__main__":
    main()
