"""CI smoke test: full-length tiny pipeline run through the default config.
Fails the job if the pipeline crashes, produces wrong output, or fails to
meaningfully beat the degraded input. Runs on GitHub runners on every push."""
import numpy as np
import sim, restore as R
from metrics2 import psnr

rgb = sim.load_clean(start=60)
cl = sim.clean_frames(rgb)
dg, meta = sim.degrade(cl, seed=1)
out = R.restore(dg, cfg={})
assert out.shape == cl.shape and out.dtype == np.uint8
assert np.isfinite(out.astype(np.float64)).all()

groups_ok = True
for m, name in [(~np.array(meta["lost"]), "non-lost"), (np.ones(len(out), bool), "all")]:
    p = float(np.mean([psnr(out[i], cl[i]) for i in np.where(m)[0]]))
    pd = float(np.mean([psnr(dg[i], cl[i]) for i in np.where(m)[0]]))
    print(f"smoke[{name}]: degraded {pd:.2f} dB -> restored {p:.2f} dB")
    if p < pd + 1.0:
        groups_ok = False
assert groups_ok, "pipeline failed to improve over degraded input"
p_all = float(np.mean([psnr(out[i], cl[i]) for i in range(len(out))]))
assert p_all > 15.0, f"restored PSNR below sanity floor: {p_all:.2f}"
print("SMOKE OK")
