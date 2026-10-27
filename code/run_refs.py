"""Strong prior-art baselines on the simulated clips: flugbild engine (FPV_DVR_CLEANER)
and AnaHD classic (CPU). Settings chosen on the TUNE clip, then ONE scored run on test.
Route: npy -> lossless mp4 (libx264rgb qp0) -> tool -> mp4 -> npy (AnaHD downscaled back
to 480x360; flugbild runs at 100% size). Both outputs go through their normal encoder
(crf 10 / crf 12), noted as a caveat."""
import numpy as np, cv2, json, pickle, subprocess, os, sys, time
from metrics2 import psnr, flicker_summary
from skimage.metrics import structural_similarity as ssim

REFS = os.environ.get("REFS_DIR", "/home/z/my-project/refs")
W, H = 480, 360

def write_mp4(frames, path):
    n, h, w = frames.shape[:3]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264rgb", "-qp", "0",
           "-preset", "fast", path]
    p = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if p.returncode != 0:
        print("libx264rgb failed, trying ffv1:", p.stderr.decode()[-200:], flush=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "bgr0", path]
        p = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
        assert p.returncode == 0, p.stderr.decode()[-500:]

def read_mp4(path):
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                        capture_output=True, text=True)
    w, h = map(int, pr.stdout.strip().split(","))
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    n = len(a) // (h * w * 3)
    return a[:n * h * w * 3].reshape(n, h, w, 3)

def run_flugbild(src, dst, rm, fb, aufwand="best"):
    req = {"action": "export", "source": src, "target": dst, "mode": "manual",
           "recipe": {"signalart": "analog", "farbberuhigung": fb, "rauschminderung": rm,
                      "osd_entfernen": False, "aufwand": aufwand,
                      "vergroesserung_prozent": 100, "kompressionswert": 14,
                      "dateiformat": "mp4-h264"}}
    rj = f"/tmp/req_flug_{os.getpid()}.json"; json.dump(req, open(rj, "w"))
    env = dict(os.environ, FLUGBILD_GPU="0", PYTHONPATH=f"{REFS}/fpv_dvr_cleaner/src")
    p = subprocess.run([sys.executable, f"{REFS}/fpv_dvr_cleaner/packaging/engine_entry.py", rj],
                       env=env, cwd=f"{REFS}/fpv_dvr_cleaner", capture_output=True)
    if p.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("flugbild failed:\n" + p.stderr.decode()[-800:] + p.stdout.decode()[-400:])
    return read_mp4(dst)

def run_anahd(src, dst, no_look):
    cmd = [sys.executable, "-m", "anahd.cli", src, "--cpu", "--force", "-o", dst]
    if no_look: cmd.append("--no-look")
    p = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=f"{REFS}/anahd/src"),
                       cwd=f"{REFS}/anahd", capture_output=True)
    if p.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("anahd failed:\n" + p.stderr.decode()[-800:] + p.stdout.decode()[-400:])
    x = read_mp4(dst)
    if x.shape[1] != H or x.shape[2] != W:
        x = np.array([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in x])
    return x

def score(x, cl, meta, groups):
    p = {g: float(np.mean([psnr(x[i], cl[i]) for i in np.where(m)[0]])) for g, m in groups.items()}
    s = {g: float(np.mean([ssim(x[i], cl[i], channel_axis=2, data_range=255) for i in np.where(m)[0]]))
         for g, m in groups.items()}
    return dict(p=p, s=s, f=flicker_summary(x, cl, meta))

def groups_of(meta):
    n = len(meta["q"]); q = meta["q"]
    lost = np.array(meta["lost"]); roll = np.array(meta["roll"]) > 0
    return {"all": np.ones(n, bool), "good": (q > 0.7) & ~lost & ~roll,
            "fades": (q < 0.5) & ~lost, "roll": roll & ~lost, "lost": lost}

def bench(tag, cl, meta, runner, out_npy):
    t0 = time.time(); x = runner()
    res = score(x, cl, meta, groups_of(meta))
    res["runtime_s"] = round(time.time() - t0, 1)
    np.save(out_npy, x)
    print(f"{tag:28s} all {res['p']['all']:6.2f}/{res['s']['all']:.3f}  "
          f"good {res['p']['good']:6.2f}  fades {res['p']['fades']:6.2f}  "
          f"roll {res['p']['roll']:6.2f}  lost {res['p']['lost']:6.2f}  "
          f"flickG {res['f']['flick_static_good']:6.2f}  ({res['runtime_s']}s)", flush=True)
    return res

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "tune"
    tmp = "/tmp/refbase"; os.makedirs(tmp, exist_ok=True)
    if phase == "tune":
        cl = np.load("tune_clean.npy"); dg = np.load("tune_deg.npy")
        meta = pickle.load(open("tune_meta.pkl", "rb"))
        src = f"{tmp}/tune_in.mp4"; write_mp4(dg, src)
        out = {}
        for rm, fb in [(50, 50), (80, 70)]:
            dst = f"{tmp}/flug_{rm}_{fb}.mp4"
            out[f"flugbild rm{rm} fb{fb}"] = bench(
                f"flugbild rm{rm} fb{fb}", cl, meta,
                lambda r=rm, f_=fb, d=dst: run_flugbild(src, d, r, f_), f"abl_flug_{rm}_{fb}.npy")
        for nl in (True, False):
            dst = f"{tmp}/anahd_{int(nl)}.mp4"
            out[f"anahd no_look={nl}"] = bench(
                f"anahd no_look={nl}", cl, meta,
                lambda n_=nl, d=dst: run_anahd(src, d, n_), f"abl_anahd_{int(nl)}.npy")
        json.dump(out, open("tune_refs.json", "w"), indent=1)
    else:
        cl = np.load("test_clean.npy"); dg = np.load("test_deg.npy")
        meta = pickle.load(open("test_meta.pkl", "rb"))
        src = f"{tmp}/test_in.mp4"; write_mp4(dg, src)
        choice = json.load(open("tune_refs.json"))
        # pick each tool's best tune setting by PSNR-all
        flug_best = max((k for k in choice if k.startswith("flugbild")), key=lambda k: choice[k]["p"]["all"])
        rm, fb = int(flug_best.split("rm")[1].split()[0]), int(flug_best.split("fb")[1])
        anahd_nl = "no_look=True" if choice.get("anahd no_look=True", {}).get("p", {}).get("all", 0) \
            >= choice.get("anahd no_look=False", {}).get("p", {}).get("all", 0) else "no_look=False"
        print(f"chosen from tune clip: flugbild '{flug_best}' ({rm},{fb}), anahd '{anahd_nl}'", flush=True)
        out = {}
        out[flug_best + " [TEST]"] = bench(
            f"flugbild({rm},{fb}) [TEST]", cl, meta,
            lambda: run_flugbild(src, f"{tmp}/flug_test.mp4", rm, fb), "b_flugbild.npy")
        out[anahd_nl + " [TEST]"] = bench(
            f"anahd [{anahd_nl}] [TEST]", cl, meta,
            lambda: run_anahd(src, f"{tmp}/anahd_test.mp4", "True" in anahd_nl), "b_anahd.npy")
        json.dump(out, open("refs_scores.json", "w"), indent=1)
