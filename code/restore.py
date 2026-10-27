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
    tp_band=0,                   # 0=full-band median (legacy), 1=detail-band only: median the
                                 # high-freq component, never touch the low band (the stab
                                 # lesson: the low band carries content; the metric-domain
                                 # residual flicker on calm static pairs is 64% detail band)
    tp_lowsig=8.0,               # detail-band separator Gaussian sigma (px)
    loo=0,                       # leave-one-out fill check: refill each masked region without
                                 # the most central neighbour; |fill - fill_loo| = how much the
                                 # fill depends on a single observation (invention risk)
    save_masks=0,                # put per-frame impulse masks into log["_masks"] (no-invention
                                 # detail-retention floor needs observed-pixel locations)
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
    fz=0,                        # Prompt 4: z^2 fusion weighting (noise vs flow-error separation)
    fz_guide=1,                  #   weight guide = median(current, warped neighbours) (time-stable)
    fz_z2=1,                     #   z^2 weighting: noise vs |grad|*flow-error separation + occ cap
    fz_occ=1.0,                  #   FB flow disagreement (px) counted as occlusion
    fz_occ_max=0.3,              #   weight cap on occluded pixels (melting safety)
    fz_smooth=2.0,               #   Gaussian sigma smoothing the FB disagreement field
    stab=0.0,                    # Prompt 5A: band-split temporal stabilizer strength (0=off)
    stab_sigma=4.0,              #   low-band Gaussian sigma (~16px scale: 93% of residual lives here)
    stab_tile=32,                #   tile size for the persistence gate (px)
    stab_rad=4,                  #   temporal window (frames each side)
    stab_max=8.0,                #   max applied offset (gray levels)
    stab_flow=1.5,               #   median flow magnitude (px) below which a pixel is trusted
    stab_dev=1.0,                #   deviation (gray) below which no correction is applied
    stab_persist=4,              #   frames of same-sign deviation before it passes as content
    stab_ts=2,                   #   temporal smoothing (frames, sigma) of the offset trajectory
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

def fb_delta(guides, t, n, cache, fbd, smooth):
    """Prompt 4: forward-backward flow disagreement for pair (t,n), in px.
    delta(p) = |f_tn(p) + f_nt(p + f_tn(p))|: ~0 when the two directions agree, large at
    occlusions and flow failures. Guide diffs there mean 'wrong content', not noise, so
    the fusion weight must not read them as texture. Both directions cached per pair."""
    key = (t, n) if t < n else (n, t)
    d = fbd.get(key)
    if d is not None: return d
    fa = cache.get((t, n))
    if fa is None: fa = flow(guides[t], guides[n]); cache[(t, n)] = fa
    fb = cache.get((n, t))
    if fb is None: fb = flow(guides[n], guides[t]); cache[(n, t)] = fb
    back = warp(fb, fa)
    delta = np.hypot(fa[..., 0] + back[..., 0], fa[..., 1] + back[..., 1])
    if smooth > 0: delta = cv2.GaussianBlur(delta, (0, 0), smooth)
    fbd[key] = delta
    return delta

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
    grow = int(cfg["mask_row_grow"])
    if grow > 0 and bad.any():
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((1 + 2 * grow, 1), np.uint8)).astype(bool)
    if bad.any():
        sm = cv2.blur(np.minimum(R, 120), (9, 1))
        for r in np.where(bad)[0]:
            hit = np.where(sm[r, BL:] > med + 10)[0]
            x0 = BL + (hit[0] if len(hit) else 0)
            m[r, max(BL, x0 - 4):] = 1
    dr = int(cfg["mask_dilate_right"]); dl = int(cfg["mask_dilate_left"])
    right = cv2.dilate(m, np.ones((1, dr), np.uint8), anchor=(dr - 1, 0))
    left = cv2.dilate(m, np.ones((1, dl), np.uint8), anchor=(dl - 1, 0))
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
    fz = int(cfg["fz"]); fbd = {}
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
            if cfg["loo"] and cfg.get("_loo") is not None:
                # invention-risk probe: drop the most central neighbour and refill; a fill
                # that moves a lot when one observation is removed was mostly INVENTED from
                # the fusion machinery, not observed
                sel = [w for w, d in zip(ws, ds) if d != min(ds, key=abs)]
                if sel:
                    fill2 = np.median(np.stack(sel), 0)
                    dful = np.abs(fill - fill2)[m]
                    if dful.size:
                        cfg["_loo"].append(dict(t=t, n=int(dful.size),
                                                mean=float(dful.mean()),
                                                p95=float(np.percentile(dful, 95))))
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
        if fz and cfg["fz_guide"] and ws:
            # Prompt 4a: weight guide from the FUSED data (mask-filled current + raw warped
            # neighbours), not the single noisy frame: the old guide re-noises every frame,
            # so guide noise lands at a new position each frame and the fusion tracks it.
            Gt = guide_of(np.median(np.stack([y] + ws), 0))
        acc = y.copy(); wsum = np.ones((H, W, 1), np.float32)
        if fz and cfg["fz_z2"]:
            gr_x = cv2.Sobel(Gt, cv2.CV_32F, 1, 0, ksize=3)
            gr_y = cv2.Sobel(Gt, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.magnitude(gr_x, gr_y)
            occ_cap = float(cfg["fz_occ_max"]) if cfg["fz_occ_max"] < 1.0 else 1.0
        for w, d in zip(ws, ds):
            nb = t + d
            Gn = warp(guides[nb].astype(np.float32), cache[(t, nb)])
            diff = np.abs(Gt - Gn)
            if fz and cfg["fz_z2"]:
                # Prompt 4b: separate the two error sources inside the guide diff.
                # SCREENING VERDICT (tune clip): this term LOSES -0.68 corr with texture;
                # on our data a guide diff on textured pixels IS mostly content mismatch
                # (Farneback error samples wrong texels), so tolerance there = melting.
                # Kept behind fz_z2 for the record; default runs use 4a only.
                delta = fb_delta(guides, t, nb, cache, fbd, cfg["fz_smooth"])
                eff2 = h * h + (grad * delta) ** 2
                wt = np.exp(-(diff ** 2) / (2 * eff2))
                if cfg["fz_occ_max"] < 1.0:
                    wt = np.minimum(wt, np.where(delta > cfg["fz_occ"], occ_cap, 1.0))
            else:
                wt = np.exp(-(diff ** 2) / (2 * h ** 2))
            wt = wt[:, :, None] * (0.85 ** (abs(d) - 1))
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
    max_pairs = int(cfg["efs_max"]); frac_min = cfg["efs_frac"]
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

# ---------------------------------------------------------------- band-split temporal stabilizer
def stabilizer_stage(frames, valid, cfg):
    """Prompt 5A: band-split temporal stabilizer. The flicker breakdown put ~92% of the
    residual variance in the low-spatial band as slow drift (41% <1 Hz + 51% 1-5 Hz),
    concentrated +-3 frames around impairment on/off events. Fix: measure the luma offset
    against a flow-aligned temporal median of the low band, clean it spatially, gate it by
    flow trust (real motion must pass untouched) and a per-tile persistence test (a
    same-sign deviation lasting >= stab_persist frames is content, not drift), then
    subtract from the full-res luma. Chroma untouched: the residual budget is luma drift."""
    sig = float(cfg["stab_sigma"]); rad = int(cfg["stab_rad"])
    mx = float(cfg["stab_max"]); flow_thr = float(cfg["stab_flow"])
    dev_min = float(cfg["stab_dev"]); persist = int(cfg["stab_persist"])
    ts = float(cfg.get("stab_ts", 2.0)); tile = int(cfg["stab_tile"])
    n = len(frames)
    sh, sw = H // 2, W // 2
    lows = [cv2.resize(cv2.GaussianBlur(f[:, :, 0], (0, 0), sig), (sw, sh),
                       interpolation=cv2.INTER_AREA) for f in frames]
    gxs, gys = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
    flcache = {}
    def getf(t, d):
        key = (t, d)
        if key not in flcache:
            flcache[key] = cv2.calcOpticalFlowFarneback(
                lows[t], lows[t + d], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flcache[key]
    nt_y, nt_x = -(-sh // tile), -(-sw // tile)
    run = np.zeros((nt_y, nt_x), np.int32); last_sign = np.zeros((nt_y, nt_x), np.int8)
    raw = [None] * n          # pass 1: gated per-frame offset fields
    for t in range(n):
        y = frames[t]
        if not valid[t]:
            run[:] = 0; last_sign[:] = 0
            continue
        refs, mags = [], []
        for d in range(-rad, rad + 1):
            q = t + d
            if d == 0 or q < 0 or q >= n or not valid[q]: continue
            f = getf(t, d)
            refs.append(cv2.remap(lows[q], gxs + f[..., 0], gys + f[..., 1],
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
            mags.append(np.hypot(f[..., 0], f[..., 1]))
        if not refs:
            continue
        L = np.median(np.stack(refs), 0)
        dev = cv2.GaussianBlur(cv2.medianBlur((lows[t] - L).astype(np.float32), 5), (0, 0), sig / 2)
        trust = np.median(np.stack(mags), 0) < flow_thr
        # per-tile persistence: same-sign significant deviation that has already lasted
        # >= persist frames is content (flow failure, real shading change), not drift
        pad = np.full((nt_y * tile, nt_x * tile), np.nan, np.float32)
        pad[:sh, :sw] = dev
        tb = np.nanmedian(pad.reshape(nt_y, tile, nt_x, tile), axis=(1, 3))
        sig_t = np.abs(tb) >= dev_min
        sn = np.sign(tb).astype(np.int8)
        run = np.where((sn == last_sign) & sig_t, run + 1,
                       np.where(sig_t, 1, 0)).astype(np.int32)
        last_sign = np.where(sig_t, sn, 0)
        gate_t = ~((run >= persist) & sig_t)
        gate = np.kron(gate_t, np.ones((tile, tile), np.float32))[:sh, :sw] * trust
        raw[t] = np.clip(dev * gate, -mx, mx).astype(np.float32)
    # pass 2: smooth the offset TRAJECTORY in time (DIAGNOSIS: the raw field wobbles
    # 0.81 gray/frame on calm static pairs and ADDS flicker; the drift it corrects moves
    # at 1-5 Hz, so a ~2-frame smooth kills the wobble and keeps the correction)
    if ts > 0:
        radius = max(1, int(round(3 * ts)))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / ts) ** 2)
        kernel /= kernel.sum()
        for t in range(n):
            if raw[t] is None: continue
            acc = np.zeros_like(raw[t]); wsum = np.zeros((sh, sw), np.float32)
            for j, w in zip(range(-radius, radius + 1), kernel):
                u = t + j
                if u < 0 or u >= n or raw[u] is None: continue
                acc += w * raw[u]; wsum += w
            raw[t] = acc / np.maximum(wsum, 1e-6)
    out = []
    for t in range(n):
        y = frames[t]
        if raw[t] is None:
            out.append(y); continue
        b = cv2.resize(raw[t], (W, H), interpolation=cv2.INTER_LINEAR)
        y = y.copy(); y[:, :, 0] -= b
        out.append(y)
    return out

# ---------------------------------------------------------------- optional post stages
def temporal_post_stage(frames, valid, cfg):
    """flow-gated temporal median on static pixels: kills residual grain flicker where
    the scene is not moving; motion pixels are left untouched."""
    rad = cfg["tp_rad"]; thr = cfg["tp_flow"]; mixv = cfg["tp_mix"]
    rad = int(rad)
    band = int(cfg.get("tp_band", 0)); lowsig = float(cfg.get("tp_lowsig", 8.0))
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
        if band:
            # detail-band aim: replace only the high-frequency component with the median's
            # high-frequency component; the low band (shading, fades, exposure) passes
            # through from the current frame untouched.
            low_y = cv2.GaussianBlur(y, (0, 0), lowsig)
            low_m = cv2.GaussianBlur(med, (0, 0), lowsig)
            fused = low_y + mixv * (med - low_m) + (1 - mixv) * (y - low_y)
        else:
            fused = mixv * med + (1 - mixv) * y
        gate = (mag < thr)[:, :, None]
        out.append(np.where(gate, fused, y))
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
    p1, sigs, masks = conceal_and_denoise(frames, valid, guides, None, int(cfg["rad1"]), cfg["h1_base"], cfg["h1_scale"], cfg, quality=quals)
    log["sigma"] = sigs; log["maskfrac"] = [None if m is None else float(m.mean()) for m in masks]
    if cfg.get("save_masks"): log["_masks"] = masks
    if cfg.get("_loo") is not None: log["loo"] = cfg["_loo"]
    log["provenance"] = [
        dict(t=t, interpolated=bool(not valid[t]),
             warped_frac=0.0 if not valid[t] else (0.0 if masks[t] is None else float(masks[t].mean())),
             observed_frac=0.0 if not valid[t] else (1.0 if masks[t] is None else 1.0 - masks[t].mean()))
        for t in range(len(frames))]
    cur = p1
    if cfg["pass2"]:
        filled = [np.where(m[:, :, None], p1[t], frames[t]) if (valid[t] and m is not None) else frames[t]
                  for t, m in enumerate(masks)]
        g2 = [guide_of(f, 0.8) for f in p1]
        cur, sigs, _ = conceal_and_denoise(filled, valid, g2, p1, int(cfg["rad2"]), cfg["h2_base"], cfg["h2_scale"], cfg, quality=quals)
    return cur, sigs, valid

def restore(degraded_rgb, log=None, cfg=None):
    t_all = __import__("time").time()
    cfg = {**DEFAULTS, **(cfg or {})}
    if log is None: log = {}
    cfg.setdefault("_loo", [] if cfg["loo"] else None)
    cfg.setdefault("_masks", None)
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
    if cfg["stab"] > 0:
        ycc = [cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32) for f in out]
        ycc = stabilizer_stage(ycc, valid, cfg)
        out = np.array([cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB) for y in ycc])
    out = sharpen_stage(out, cfg)
    out = np.asarray(out)
    cfg["runtime_s"] = round(__import__("time").time() - t_all, 1)
    return out
