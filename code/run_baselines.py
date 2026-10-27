"""Baselines on the TEST clip (bundle protocol). Run once at the end."""
import numpy as np, cv2, os, subprocess, shutil
from baselines import nlm_multi, med3
dg = np.load("degraded.npy")
np.save("b_med3_nlm.npy", nlm_multi(list(med3(dg)), 18, 18)); print("med3+nlm done", flush=True)
np.save("b_nlm_h25.npy", nlm_multi(list(dg), 25, 25)); print("nlm25 done", flush=True)
np.save("b_nlm_h12.npy", nlm_multi(list(dg), 12, 12)); print("nlm12 done", flush=True)
d = "tmp_hq"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
for i, f in enumerate(dg): cv2.imwrite(f"{d}/in_{i:03d}.png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{d}/in_%03d.png", "-vf", "hqdn3d=4:3:6:4.5", f"{d}/o_%03d.png"], check=True)
np.save("b_hqdn3d_def.npy", np.array([cv2.cvtColor(cv2.imread(f"{d}/o_{i+1:03d}.png"), cv2.COLOR_BGR2RGB) for i in range(len(dg))]))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{d}/in_%03d.png", "-vf", "hqdn3d=12:10:20:15", f"{d}/s_%03d.png"], check=True)
np.save("b_hqdn3d_strong.npy", np.array([cv2.cvtColor(cv2.imread(f"{d}/s_{i+1:03d}.png"), cv2.COLOR_BGR2RGB) for i in range(len(dg))]))
shutil.rmtree(d); print("hqdn3d done", flush=True)
