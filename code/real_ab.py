"""Real-footage blind A/B (the arbiter): mixkit drone content through the SAME simulated
analog link, then blind VLM judges on randomized A/B sheets:
  AB1: bar (sync+med3+nlm, SSIM/flicker winner in the suite)  vs  ours (champion P22)
  AB2: degraded (status quo, no restoration)                  vs  ours
Ties go to the second argument (champion/status-quo bias: a noisy judge must EVIDENTIALLY
prefer the candidate). Judge never knows which is which.
Usage: real_ab.py build | real_ab.py judge"""
import sys, os, json, time, pickle, random
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, restore as R
from run_suite import bar, score
from judge import judge

os.chdir(os.path.dirname(os.path.abspath(__file__)))
CHAMP = json.load(open("../results/champion.json"))["cfg"]

if sys.argv[1] == "build":
    cap = cv2.VideoCapture("/tmp/mixkit_test.mp4")
    frames = []
    while len(frames) < 60:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    v = np.array(frames)
    out = []
    for f in v:
        h, w = f.shape[:2]
        cw = int(h * sim.W / sim.H); x0 = (w - cw) // 2
        out.append(cv2.resize(f[:, x0:x0 + cw], (sim.W, sim.H), interpolation=cv2.INTER_AREA))
    rgb = np.array(out)
    cl = sim.clean_frames(rgb)
    dg, meta = sim.degrade(cl, seed=21)
    np.save("/tmp/rf_cl.npy", cl); np.save("/tmp/rf_dg.npy", dg)
    pickle.dump(meta, open("/tmp/rf_meta.pkl", "wb"))
    print("roll frames:", [i for i, r in enumerate(meta["roll"]) if r],
          "lost:", [i for i, l in enumerate(meta["lost"]) if l], flush=True)
    t0 = time.time(); b = bar(dg); print(f"bar {time.time()-t0:.0f}s", flush=True)
    t0 = time.time(); mine = R.restore(dg, cfg=CHAMP); print(f"ours {time.time()-t0:.0f}s", flush=True)
    np.save("/tmp/rf_bar.npy", b); np.save("/tmp/rf_ours.npy", mine)
    r = dict(degraded=score(dg, cl, meta), bar=score(b, cl, meta), ours=score(mine, cl, meta))
    print(f"{'':10s}{'P all':>8}{'P good':>8}{'P lost':>8}{'P roll':>8}{'S all':>8}{'S good':>8}{'flkS':>7}{'flkT':>7}")
    for k, v2 in r.items():
        print(f"{k:10s}{v2['p']['all']:8.2f}{v2['p']['good']:8.2f}{v2['p']['lost']:8.2f}{v2['p']['roll']:8.2f}"
              f"{v2['s']['all']:8.3f}{v2['s']['good']:8.3f}{v2['f']['flick_static_good']:7.2f}{v2['f']['flick_tdiff_good']:7.2f}")
    json.dump(r, open("/tmp/rf_scores.json", "w"), indent=1)

elif sys.argv[1] == "judge":
    random.seed(20261027)  # reproducible A/B assignment
    cl = np.load("/tmp/rf_cl.npy"); dg = np.load("/tmp/rf_dg.npy")
    b = np.load("/tmp/rf_bar.npy"); mine = np.load("/tmp/rf_ours.npy")
    ab1 = judge(b, mine, cl, workdir="/tmp/rf_crops1")   # does bar visibly beat ours?
    ab2 = judge(dg, mine, cl, workdir="/tmp/rf_crops2")  # does restoration visibly beat status quo?
    scores = json.load(open("/tmp/rf_scores.json"))
    res = dict(
        what="real-footage blind A/B: mixkit drone content, simulated analog link (seed 21), 60 frames 480x360",
        protocol="5 randomized trials per A/B (2 static flicker, 2 motion detail, 1 motion flicker); ties favor the second system (status-quo bias)",
        ab1_bar_vs_ours=ab1, ab2_degraded_vs_ours=ab2, scores=scores)
    json.dump(res, open("../results/real_ab.json", "w"), indent=1)
    print("AB1 bar(cand) vs ours(champ):", ab1["votes"], flush=True)
    for d in ab1["details"]: print("  ", d.get("trial"), d.get("better"), "|", d.get("reason", "")[:60], flush=True)
    print("AB2 degraded(cand) vs ours(champ):", ab2["votes"], flush=True)
    for d in ab2["details"]: print("  ", d.get("trial"), d.get("better"), "|", d.get("reason", "")[:60], flush=True)
    # side-by-side + zoom sheets for the user
    rows = [np.hstack([dg[t], b[t], mine[t]]) for t in range(60)]
    import imageio.v2 as iio
    iio.mimsave("../media/real_ab_sidebyside.mp4", rows, fps=12, macro_block_size=1)
    for src, stem in [("/tmp/rf_crops1", "ab1"), ("/tmp/rf_crops2", "ab2")]:
        import glob, shutil
        for p in sorted(glob.glob(src + "/*.png")):
            shutil.copy(p, f"../media/sheet_{stem}_{os.path.basename(p)}")
    print("wrote media/real_ab_sidebyside.mp4 + judge sheets", flush=True)
