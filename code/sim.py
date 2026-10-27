"""Physically-motivated analog FPV link simulator.
Impairments: FM-threshold click noise (horizontal streaks w/ decay tail), gaussian noise
(luma + stronger chroma), partial-line dropout bands, vsync loss (vertical roll), complete
signal loss frames, multipath ghost (delayed echo), per-line horizontal jitter (no TBC).
A reference pilot pattern sits at the start of every line (stands in for sync/back-porch)."""
import numpy as np, cv2, imageio.v3 as iio

W, H, N = 480, 360, 60
PILOT = np.r_[np.full(8, 16), np.full(4, 200)].astype(np.float32)  # 12 px reference
BL = len(PILOT)
VSYNC = (235.0, 16.0)   # 4 bright rows then 4 dark rows at the top of every field

def load_clean(n=N, start=60):
    v = iio.imread("imageio:cockatoo.mp4")[start:start + n]
    out = []
    for f in v:
        h, w = f.shape[:2]
        cw = int(h * W / H); x0 = (w - cw) // 2
        g = cv2.resize(f[:, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
        out.append(g)
    out = np.array(out)
    return out  # RGB uint8

def add_pilot_ycc(ycc):
    ycc[:, :BL, 0] = PILOT[None, :]
    ycc[:, :BL, 1:] = 128
    ycc[0:4, BL:, 0] = VSYNC[0]; ycc[4:8, BL:, 0] = VSYNC[1]; ycc[0:8, BL:, 1:] = 128   # vertical sync/blanking marker
    return ycc

def to_ycc(rgb):  return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
def to_rgb(ycc): return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

def clean_frames(rgb):
    out = []
    for f in rgb:
        y = add_pilot_ycc(to_ycc(f)); out.append(to_rgb(y))
    return np.array(out)

def quality_curve(n, rng):
    q = np.full(n, 0.78)
    q += 0.08 * np.sin(np.linspace(0, 5, n))
    for c, wdt, depth in [(14, 5, 0.55), (31, 6, 0.45), (47, 4, 0.7)]:
        q -= depth * np.exp(-((np.arange(n) - c) / wdt) ** 2)
    return np.clip(q, 0.02, 1)

def degrade(clean_rgb, seed=7):
    rng = np.random.default_rng(seed)
    n = len(clean_rgb); q = quality_curve(n, rng)
    out = []; meta = dict(q=q, roll=[0] * n, lost=[False] * n)
    lost_idx = {int(np.argmin(q[40:55])) + 40 - 1, int(np.argmin(q[40:55])) + 40 + 1}
    roll_idx = {12, 15, 29, 33, 46, 49}
    jit = np.zeros(H); jit_state = 0.0
    for t in range(n):
        y = to_ycc(clean_rgb[t]); qq = q[t]; badness = 1 - qq
        # --- multipath ghost (frames 28-44)
        if 28 <= t <= 44:
            a, k = 0.28, 9
            g = np.concatenate([np.repeat(y[:, :1], k, 1), y[:, :-k]], 1)
            y = (y + a * g) / (1 + a)
            # (pilot gets echo too -> realistic)
        # --- bandwidth limit
        y[:, :, 0] = cv2.GaussianBlur(y[:, :, 0], (0, 0), sigmaX=0.9, sigmaY=0.01)
        y[:, :, 1:] = cv2.GaussianBlur(y[:, :, 1:], (0, 0), sigmaX=2.5, sigmaY=0.01)
        # --- gaussian noise
        sy = 3 + 28 * badness ** 1.3
        y[:, :, 0] += rng.normal(0, sy, (H, W))
        y[:, :, 1:] += rng.normal(0, sy * 1.7, (H, W, 2))
        # --- FM click noise: streaks with decaying tails
        ns = int(2 + 220 * badness ** 2.2)
        for _ in range(ns):
            r = rng.integers(0, H); x = rng.integers(BL, W - 2)
            L = rng.integers(2, 16); amp = rng.choice([-1, 1]) * rng.uniform(110, 255)
            xs = np.arange(L); prof = amp * np.exp(-xs / (L / 2.5))
            xe = min(W, x + L)
            y[r, x:xe, 0] += prof[:xe - x]
            if rng.random() < 0.5: y[r, x:xe, 1:] += rng.normal(0, 60, (xe - x, 2))
        # --- dropout bands (deep fades)
        if qq < 0.35:
            for _ in range(rng.integers(1, 4)):
                r0 = rng.integers(0, H - 30); hh = rng.integers(4, 28); x0 = rng.integers(BL, W // 2)
                y[r0:r0 + hh, x0:, 0] = 110 + rng.normal(0, 55, (hh, W - x0))
                y[r0:r0 + hh, x0:, 1:] = 128 + rng.normal(0, 40, (hh, W - x0, 2))
        # --- complete loss
        if t in lost_idx:
            y[:, :, 0] = 100 + rng.normal(0, 70, (H, W)); y[:, :, 1:] = 128 + rng.normal(0, 45, (H, W, 2))
            y[:, :BL, 0] = np.where(rng.random((H, BL)) < 0.5, y[:, :BL, 0], PILOT[None, :])  # pilot swamped
            meta["lost"][t] = True
        # --- horizontal line jitter (no TBC): OU random walk, sub-pixel
        for r in range(H):
            jit_state = 0.985 * jit_state + rng.normal(0, 0.45); jit[r] = jit_state
        jit = np.clip(jit, -4.5, 4.5)
        mapx = (np.arange(W)[None, :] - jit[:, None]).astype(np.float32)
        mapy = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, 1)
        y = cv2.remap(y, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # --- vsync loss: vertical roll
        if t in roll_idx:
            s = int(rng.integers(40, 200)); y = np.roll(y, s, axis=0); meta["roll"][t] = s
        out.append(to_rgb(y))
    return np.array(out), meta

def build(name, start, seed):
    rgb = load_clean(start=start); cl = clean_frames(rgb)
    dg, meta = degrade(cl, seed=seed)
    np.save(f"{name}_clean.npy", cl); np.save(f"{name}_deg.npy", dg)
    import pickle; pickle.dump(meta, open(f"{name}_meta.pkl", "wb"))
    print(name, cl.shape, "roll", [(i, r) for i, r in enumerate(meta["roll"]) if r],
          "lost", [i for i, l in enumerate(meta["lost"]) if l])

if __name__ == "__main__":
    build("test", start=60, seed=7)    # held-out test clip: score ONCE at the end
    build("tune", start=150, seed=11)  # tuning clip: all tuning happens here
