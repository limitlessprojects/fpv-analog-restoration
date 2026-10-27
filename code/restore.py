"""Config-driven restoration pipeline (from the FPV restoration bundle, parameterized).
All defaults reproduce the bundle's locked behavior exactly; variants pass a cfg dict."""
import numpy as np, cv2
from math import erf
from sim import W, H, BL, PILOT

DEFAULTS = dict(
    sync=True, pass2=True, ghost=False,
    rad1=2, h1_base=6.0, h1_scale=0.6,
    rad2=3, h2_base=5.0, h2_scale=0.35,
    mask_abs=38.0, mask_sig=4.2, mask_row_k=3.0, mask_row_floor=12.0,
    mask_res=0.0,                # blend frame-sigma -> residual-MAD-sigma for mask thr (0=off)
    mask_row_grow=0, mask_dilate_right=9, mask_dilate_left=5,
    fill_q=0,                    # fill median only from the N best-sync-quality neighbours (0=off)
    chroma_base=0.5, chroma_scale=0.035, chroma_min=0.8, chroma_max=3.0,
    polish_ky=4.0, polish_kc=4.0, polish_ymax=32.0, polish_cmax=32.0,
    polish_multi=False,          # windowed (3-frame) joint NLM at polish stage
    temporal_post=False,         # flow-gated temporal median (flicker kill)
    tp_after_polish=True,        # True: apply AFTER NLM polish (NLM re-adds independent residual)
    tp_rad=2, tp_flow=0.30, tp_mix=1.0,
    chroma_temporal=0.0,         # blend chroma toward warped-neighbour median
    lost_smooth=0.0,             # extra smoothing blend on interpolated frames
    ghost_smooth=False,          # cross-fade multipath echo on/off boundaries (flicker)
    gsb_fade=1,
    efs=False,                   # event-flicker smoother (static-gated pair-diff spikes)
    efs_k=4.0, efs_floor=2.0, efs_w=0.35, efs_max=6, efs_frac=0.0,
    flow_damp=0.0,               # temporal-weight damping by flow magnitude (px sigma; 0=off)
    fill_mag=0.0,                # fill-median only from neighbours with mean flow <= this (0=off)
    vfill_frac=0.0,              # rows with mask fraction above this get vertical inpaint fill (0=off)
    sharpen=0.0,                 # unsharp amount on final luma
)

# ---------------------------------------------------------------- sync / timing recovery
_GRID = np.arange(-5.5, 5.51, 0.25).astype(np.float32)
_X = np.arange(24, dtype=np.float32)
_PHI = np.vectorize(lambda z: 0.5 * (1 + erf(z / np.sqrt(2))))

def pilot_fit(y):
    """per-row sub-pixel shift of the reference pilot's rising edge + fit RMS"""
    rows = y[:, :24]
    T = 16 + 184 * _PHI((_X[None, :] - 7.5 - _GRID[:, None]) / 0.9)
    Wt = (_X[None, :] < 10 + _GRID[:, None]).astype(np.float32)
    d = (rows[:, None, :] - T[None]) ** 2 * Wt[None]
    cost = d.sum(2) / Wt.sum(1)[None]
    i = cost.argmin(1)
    ii = np.clip(i, 1, len(_GRID) - 2); r = np.arange(len(y))
    c0, c1, c2 = cost[r, ii - 1], cost[r, ii], cost[r, ii + 1]
    den = (c0 - 2 * c1 + c2); off = np.where(den > 1e-6, 0.5 * (c0 - c2) / den, 0)
    s = _GRID[ii] + np.clip(off, -1, 1) * 0.25
    return s, np.sqrt(cost[r, i])

_VT = np.r_[np.full(4, 235.0), np.full(4, 16.0)].astype(np.float32)

def find_roll(y):
    """lock onto the vertical sync marker (what a real decoder's sync separator does):
    circular template-match of the row-median profile (left strip only)."""
    p = np.median(y[:, BL:BL + 60, 0], axis=1).astype(np.float32)
    cost = np.array([np.mean((np.roll(p, -s)[:8] - _VT) ** 2) for s in range(H)])
    k = int(cost.argmin())
    return (k if cost[k] < 55.0 ** 2 else 0), float(np.sqrt(cost[k]))

def _align(y, s_row, roll):
    y = np.roll(y, -roll, axis=0); s_row = np.roll(s_row, -roll)
    m3 = np.median(np.stack([np.roll(s_row, 1), s_row, np.roll(s_row, -1)]), 0)
    s_row = np.where(np.abs(s_row - m3) > 1.2, m3, s_row).astype(np.float32)
    mapx = (np.arange(W, dtype=np.float32)[None, :] + s_row[:, None])
    mapy = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, 1)
    return cv2.remap(y, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def sync_recover(frames_ycc, log=None):
    """lock vertical sync (unroll), re-time every line from the pilot, flag lost frames.
    Also returns per-frame quality (median pilot-fit RMS; inf for unlocked frames)."""
    n = len(frames_ycc); out = []; valid = []; rolls = []; quals = []
    for t, y in enumerate(frames_ycc):
        s_row, rms = pilot_fit(y[:, :, 0])
        med_rms = float(np.median(rms))
        locked = bool(med_rms < 30 and noise_sigma(y[:, :, :1]) < 28)
        roll = 0
        if locked: roll, _ = find_roll(y)
        out.append(_align(y, s_row, roll) if locked else y); valid.append(locked); rolls.append(roll)
        quals.append(med_rms if locked else 1e9)
    if log is not None: log.extend((t, valid[t], rolls[t], 0.0) for t in range(n))
    return out, valid, quals

# ---------------------------------------------------------------- motion-compensated stacking
def guide_of(y, blur=1.3):
    g = np.clip(y[:, :, 0], 0, 255).astype(np.uint8)
    g = cv2.medianBlur(g, 5)
    return cv2.GaussianBlur(g, (0, 0), blur)

def flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 7, 1.5, 0)

_gx, _gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
def warp(img, fl):
    return cv2.remap(img, _gx + fl[..., 0], _gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def build_stack(t, frames, valid, guides, rad, cache):
    ws, ds = [], []
    for d in range(-rad, rad + 1):
        n = t + d
        if d == 0 or n < 0 or n >= len(frames) or not valid[n]: continue
        key = (t, n)
        if key not in cache: cache[key] = flow(guides[t], guides[n])
        ws.append(warp(frames[n], cache[key])); ds.append(d)
    return ws, ds

def noise_sigma(y):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(y[:, BL:], -1, k)
    return 1.4826 * np.median(np.abs(r)) / 6.0

def impulse_mask(y, ref_y, sig, cfg):
    R = np.abs(y[:, :, 0] - ref_y[:, :, 0])
    if cfg["mask_res"] > 0:
        # residual-MAD sigma adapts to what the fill actually leaves behind (fades),
        # frame sigma alone overshoots there (4.2*28 ~ 118 misses dim band pixels)
        rsig = 1.4826 * float(np.median(R[:, BL:]))
        sig = (1 - cfg["mask_res"]) * sig + cfg["mask_res"] * rsig
    thr = max(cfg["mask_abs"], cfg["mask_sig"] * sig)
    m = (R > thr).astype(np.uint8); m[:, :BL] = 0
    E = np.minimum(R, 120)[:, BL:].mean(1)
    med = np.median(E); mad = 1.4826 * np.median(np.abs(E - med)) + 1e-6
    bad = E > med + max(cfg["mask_row_k"] * mad, cfg["mask_row_floor"])
    if cfg["mask_row_grow"] > 0 and bad.any():
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((1 + 2 * cfg["mask_row_grow"], 1), np.uint8)).astype(bool)
    if bad.any():
        sm = cv2.blur(np.minimum(R, 120), (9, 1))
        for r in np.where(bad)[0]:
            hit = np.where(sm[r, BL:] > med + 10)[0]
            x0 = BL + (hit[0] if len(hit) else 0)
            m[r, max(BL, x0 - 4):] = 1
    right = cv2.dilate(m, np.ones((1, cfg["mask_dilate_right"]), np.uint8), anchor=(cfg["mask_dilate_right"] - 1, 0))
    left = cv2.dilate(m, np.ones((1, cfg["mask_dilate_left"]), np.uint8), anchor=(cfg["mask_dilate_left"] - 1, 0))
    return np.maximum(right, left).astype(bool)

def interpolate_lost(t, frames, valid, guides):
    p = next((i for i in range(t - 1, -1, -1) if valid[i]), None)
    n = next((i for i in range(t + 1, len(frames)) if valid[i]), None)
    if p is None: return frames[n].copy()
    if n is None: return frames[p].copy()
    a = (t - p) / (n - p); f = flow(guides[p], guides[n])
    A = warp(frames[p], -a * f); B = warp(frames[n], (1 - a) * f)
    return (1 - a) * A + a * B

def conceal_and_denoise(frames, valid, guides, ref_for_mask, rad, h_base, h_scale, cfg, tag="", quality=None):
    n = len(frames); cache = {}; out = []; sigs = []; masks = []
    fd = cfg["flow_damp"]; fmax = cfg["fill_mag"]
    for t in range(n):
        if not valid[t]:
            f = interpolate_lost(t, frames, valid, guides)
            if cfg["lost_smooth"] > 0:
                f = (1 - cfg["lost_smooth"]) * f + cfg["lost_smooth"] * cv2.GaussianBlur(f, (0, 0), 1.5)
            out.append(f); sigs.append(0); masks.append(None); continue
        y = frames[t].copy()
        ws, ds = build_stack(t, frames, valid, guides, rad, cache)
        sig = noise_sigma(y)
        if ref_for_mask is None:
            near = [w for w, d in zip(ws, ds) if abs(d) <= 2]
            ref = np.median(np.stack(near + [y]), 0) if near else y
        else:
            ref = ref_for_mask[t]
        m = impulse_mask(y, ref, sig, cfg)
        m3 = np.repeat(m[:, :, None], 3, 2)
        if ws:
            if fmax > 0:
                sel = [w for w, d in zip(ws, ds)
                       if float(np.hypot(cache[(t, t + d)][..., 0], cache[(t, t + d)][..., 1]).mean()) <= fmax]
                fill_src = sel if sel else ws[:1]
            else:
                fill_src = ws
            if cfg["fill_q"] > 0 and quality is not None:
                # neighbours carry their own damage: fill only from the N best-quality
                # (lowest pilot-fit RMS) warped neighbours
                cand = sorted(zip(ds, ws), key=lambda x: quality[t + x[0]])
                fill_src = [w for _, w in cand[:int(cfg["fill_q"])]]
            fill = np.median(np.stack(fill_src), 0)
            y = np.where(m3, fill, y)
        if cfg["vfill_frac"] > 0:
            # VCR-style dropout compensation (AFTER the warped fill): rows the mask flagged
            # as bands get a vertical inpaint (TELEA) instead of trusting warped neighbours
            # that carry their own fade damage
            rowfrac = m[:, BL:].mean(1)
            ridx = np.where(rowfrac > cfg["vfill_frac"])[0]
            if len(ridx):
                mb = np.zeros((H, W), np.uint8); mb[ridx] = m[ridx].astype(np.uint8)
                if mb.sum() > 0:
                    for c in range(3):
                        yy = np.clip(y[:, :, c], 0, 255).astype(np.uint8)
                        y[mb.astype(bool), c] = cv2.inpaint(yy, mb, 3, cv2.INPAINT_TELEA)[mb.astype(bool)]
        masks.append(m)
        sig2 = noise_sigma(y) if m.mean() < 0.4 else sig
        h = h_base + h_scale * sig2
        Gt = guides[t].astype(np.float32)
        acc = y.copy(); wsum = np.ones((H, W, 1), np.float32)
        for w, d in zip(ws, ds):
            nb = t + d
            Gn = warp(guides[nb].astype(np.float32), cache[(t, nb)])
            diff = np.abs(Gt - Gn)
            wt = np.exp(-(diff ** 2) / (2 * h ** 2))[:, :, None] * (0.85 ** (abs(d) - 1))
            if fd > 0:
                mag2 = cache[(t, nb)][..., 0] ** 2 + cache[(t, nb)][..., 1] ** 2
                wt = wt * np.exp(-mag2 / (2 * fd * fd))[:, :, None]
            acc += wt * w; wsum += wt
        y = acc / wsum
        sigs.append(sig2)
        cs = float(np.clip(cfg["chroma_base"] + cfg["chroma_scale"] * sig2, cfg["chroma_min"], cfg["chroma_max"]))
        y[:, :, 1:] = cv2.GaussianBlur(y[:, :, 1:], (0, 0), cs)
        out.append(y)
    return out, sigs, masks

# ---------------------------------------------------------------- ghost segment detection
def hp_ac(y, maxlag=32):
    """echo statistic: autocorrelation of the horizontal derivative, white-noise power removed
    from lag 0. For y=(x+a*x_k)/(1+a): rho(k) ~= a/(1+a^2)."""
    d = np.clip(np.diff(y[:, BL:, 0], axis=1), -100, 100)
    d = d - d.mean(1, keepdims=True)
    ac = np.array([(d[:, :d.shape[1] - k] * d[:, k:]).mean() for k in range(maxlag + 1)])
    sig = noise_sigma(y[:, :, :1])
    return ac / max(ac[0] - 2 * sig ** 2, 0.25 * ac[0])

def ghost_segment(frames, valid):
    """frames where a multipath echo is active (persistence-verified lag-profile z-score).
    Used NOT to invert the echo but to smooth the on/off boundary flicker."""
    idx = [i for i, v in enumerate(valid) if v]
    if len(idx) < 8: return set()
    acs = np.array([hp_ac(f) for f in frames])
    base = np.median(acs[idx], 0); D = acs - base
    mad = 1.4826 * np.median(np.abs(D[idx] - np.median(D[idx], 0)), 0) + 1e-3
    Z = D / mad
    Zs = (Z[:, 5:34] + np.roll(Z[:, 5:34], 1, 1) + np.roll(Z[:, 5:34], -1, 1)) / np.sqrt(3)
    cand = {}
    for t in idx:
        j = int(Zs[t].argmax()); k = j + 5
        if 6 <= k <= 32 and Zs[t, j] > 3.0 and D[t, k] > 0.02: cand[t] = (k, float(Zs[t, j]), float(D[t, k]))
    if not cand: return set()
    ks = np.array([v[0] for v in cand.values()]); kmode = int(np.bincount(ks).argmax())
    flagged = {t for t, v in cand.items() if abs(v[0] - kmode) <= 1}
    seg = set()
    for t in idx:
        nb = sum(1 for u in range(t - 3, t + 4) if u in flagged)
        if (t in flagged and nb >= 3) or (nb >= 3 and Zs[t, kmode - 5] > 1.0): seg.add(t)
    return seg

def ghost_boundary_smooth(out_rgb, seg, fade):
    """cross-fade the echo on/off transitions so the segment appears/disappears as a short
    ramp instead of a single-frame texture step (perceptual flicker kill; PSNR-neutral:
    blending halves the echo bias on the blended frames)."""
    if not seg: return out_rgb
    out = [f.astype(np.float32) for f in out_rgb]
    for j in range(1, fade + 1):
        w = 0.5 ** j
        for t in sorted(seg):
            if t - 1 not in seg and t - j >= 0 and t - j not in seg:
                out[t] = (1 - w) * out[t] + w * out[t - j]      # echo side leans to pre-echo
            if t + 1 not in seg and t + j < len(out) and t + j not in seg:
                out[t] = (1 - w) * out[t] + w * out[t + j]      # echo side leans to post-echo
    return [np.clip(f, 0, 255).astype(np.uint8) for f in out]

def event_flicker_smooth(out_rgb, valid, cfg):
    """Detect pairs whose long-window flow-gated static-region diff spikes (impairment on/off
    steps) and cross-fade both frames. Gate requires each pixel static over 3 consecutive
    pairs, so fast real motion (under-tracked by flow) cannot trigger a blend."""
    k_mad = cfg["efs_k"]; floor = cfg["efs_floor"]; w = cfg["efs_w"]; thr = cfg["tp_flow"]
    max_pairs = cfg["efs_max"]; frac_min = cfg["efs_frac"]
    guides = [guide_of(cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32), 0.8) for f in out_rgb]
    n = len(out_rgb)
    gate = {}
    for t in range(1, n):
        if not (valid[t] and valid[t - 1]): continue
        f1 = flow(guides[t - 1], guides[t]); f2 = flow(guides[t], guides[t - 1])
        gate[t] = np.maximum(np.hypot(f1[..., 0], f1[..., 1]), np.hypot(f2[..., 0], f2[..., 1])) < thr
    out = [f.astype(np.float32) for f in out_rgb]
    d = {}
    for t in sorted(gate):
        g = gate[t].copy()
        for u in (t - 1, t + 1):
            if u in gate: g &= gate[u]
        if g.sum() < 3000: continue
        dm = np.abs(out[t] - out[t - 1]).mean(2)[g]
        # spread test: a global impairment step lifts diffs everywhere on static pixels;
        # a flow failure on a moving object concentrates them
        if float((dm > max(3 * np.median(dm), 12.0)).mean()) < frac_min: continue
        d[t] = (dm.mean(), int(g.sum()))
    if len(d) < 8: return [np.clip(f, 0, 255).astype(np.uint8) for f in out]
    vals = np.array([v[0] for v in d.values()])
    med = np.median(vals); mad = 1.4826 * np.median(np.abs(vals - med)) + 1e-3
    spike_thr = med + k_mad * mad + floor
    pairs = sorted([t for t, v in d.items() if v[0] > spike_thr], key=lambda t: -d[t][0])[:max_pairs]
    for t in pairs:
        a, b = out[t - 1], out[t]
        out[t] = (1 - w) * b + w * a
        out[t - 1] = (1 - w) * a + w * b
    return [np.clip(f, 0, 255).astype(np.uint8) for f in out]

# ---------------------------------------------------------------- optional post stages
def temporal_post_stage(frames, valid, cfg):
    """flow-gated temporal median on static pixels: kills residual grain flicker where
    the scene is not moving; motion pixels are left untouched."""
    rad = cfg["tp_rad"]; thr = cfg["tp_flow"]; mixv = cfg["tp_mix"]
    guides = [guide_of(f, 0.8) for f in frames]
    out = []
    for t, y in enumerate(frames):
        if not valid[t]:
            out.append(y); continue
        ws = []; mag = np.zeros((H, W), np.float32)
        for d in range(-rad, rad + 1):
            n = t + d
            if d == 0 or n < 0 or n >= len(frames) or not valid[n]: continue
            fl = flow(guides[t], guides[n])
            mag = np.maximum(mag, np.hypot(fl[..., 0], fl[..., 1]))
            ws.append(warp(frames[n], fl))
        if not ws:
            out.append(y); continue
        med = np.median(np.stack(ws + [y]), 0)
        gate = (mag < thr)[:, :, None]
        out.append(np.where(gate, mixv * med + (1 - mixv) * y, y))
    return out

def chroma_temporal_stage(frames, valid, cfg):
    """blend chroma toward the median chroma of warped neighbours (chroma flicker fix)"""
    wgt = cfg["chroma_temporal"]
    guides = [guide_of(f, 1.0) for f in frames]
    out = []
    for t, y in enumerate(frames):
        if not valid[t] or wgt <= 0:
            out.append(y); continue
        ws = []
        for d in (-1, 1):
            n = t + d
            if 0 <= n < len(frames) and valid[n]:
                ws.append(warp(y, flow(guides[t], guides[n])))
        if not ws:
            out.append(y); continue
        med = np.median(np.stack(ws), 0)
        y = y.copy(); y[:, :, 1:] = (1 - wgt) * y[:, :, 1:] + wgt * med[:, :, 1:]
        out.append(y)
    return out

def spatial_polish(frames, sigs, valid, cfg):
    ky, kc = cfg["polish_ky"], cfg["polish_kc"]
    ymax, cmax = cfg["polish_ymax"], cfg["polish_cmax"]
    res = []
    bgr = [cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR) for y in frames]
    if cfg["polish_multi"]:
        r = 1; pad = [bgr[0]] * r + bgr + [bgr[-1]] * r
        for t, (y, s, v) in enumerate(zip(frames, sigs, valid)):
            rem = noise_sigma(y)
            hY = float(np.clip(ky * rem, 1.0, ymax)); hC = float(np.clip(kc * rem, 1.0, cmax))
            d = cv2.fastNlMeansDenoisingColoredMulti(pad[t:t + 3], 1, 3, None, hY, hC, 5, 15)
            res.append(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))
        return res
    for y, s, v in zip(frames, sigs, valid):
        rgb = cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        rem = noise_sigma(y)
        hY = float(np.clip(ky * rem, 1.0, ymax)); hC = float(np.clip(kc * rem, 1.0, cmax))
        d = cv2.fastNlMeansDenoisingColored(rgb, None, hY, hC, 5, 15)
        res.append(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))
    return res

def sharpen_stage(frames_rgb, cfg):
    a = cfg["sharpen"]
    if a <= 0: return frames_rgb
    out = []
    for f in frames_rgb:
        b = cv2.GaussianBlur(f, (0, 0), 1.2).astype(np.float32)
        out.append(np.clip(f.astype(np.float32) + a * (f.astype(np.float32) - b), 0, 255).astype(np.uint8))
    return out

def to_ycc(rgb): return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)

def restore_core(degraded_rgb, log=None, cfg=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    log = {} if log is None else log
    frames = [to_ycc(f) for f in degraded_rgb]
    if cfg["sync"]:
        sl = []; frames, valid, quals = sync_recover(frames, sl); log["sync"] = sl
    else:
        valid = [True] * len(frames); quals = None
    if cfg["ghost_smooth"]:
        log["ghost_seg"] = sorted(ghost_segment(frames, valid))
    guides = [guide_of(f) for f in frames]
    p1, sigs, masks = conceal_and_denoise(frames, valid, guides, None, cfg["rad1"], cfg["h1_base"], cfg["h1_scale"], cfg, quality=quals)
    log["sigma"] = sigs; log["maskfrac"] = [None if m is None else float(m.mean()) for m in masks]
    cur = p1
    if cfg["pass2"]:
        filled = [np.where(m[:, :, None], p1[t], frames[t]) if (valid[t] and m is not None) else frames[t]
                  for t, m in enumerate(masks)]
        g2 = [guide_of(f, 0.8) for f in p1]
        cur, sigs, _ = conceal_and_denoise(filled, valid, g2, p1, cfg["rad2"], cfg["h2_base"], cfg["h2_scale"], cfg, quality=quals)
    return cur, sigs, valid

def restore(degraded_rgb, log=None, cfg=None):
    t_all = __import__("time").time()
    cfg = {**DEFAULTS, **(cfg or {})}
    if log is None: log = {}
    cur, sigs, valid = restore_core(degraded_rgb, log, cfg)
    if cfg["temporal_post"] and not cfg["tp_after_polish"]:
        cur = temporal_post_stage(cur, valid, cfg)
    if cfg["chroma_temporal"] > 0:
        cur = chroma_temporal_stage(cur, valid, cfg)
    out = np.array(spatial_polish(cur, sigs, valid, cfg))
    if cfg["temporal_post"] and cfg["tp_after_polish"]:
        ycc = [cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32) for f in out]
        ycc = temporal_post_stage(ycc, valid, cfg)
        out = np.array([cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB) for y in ycc])
    if cfg["ghost_smooth"]:
        out = np.array(ghost_boundary_smooth(list(out), set(log.get("ghost_seg", [])), cfg["gsb_fade"]))
    if cfg["efs"]:
        out = np.array(event_flicker_smooth(list(out), valid, cfg))
    out = sharpen_stage(out, cfg)
    out = np.asarray(out)
    cfg["runtime_s"] = round(__import__("time").time() - t_all, 1)
    return out
