"""Blind VLM judge. Builds anonymized [reference | A | B] comparison sheets from two
restorations of the SAME tuning clip, asks the vision model which restoration is better,
5 independent calls (2x flicker@static, 2x detail@motion, 1x flicker@motion) with
randomized A/B assignment per call, majority vote; ties go to the champion (status-quo
bias: a noisy judge must EVIDENTIALLY prefer the candidate). The judge never knows
which one is the champion."""
import subprocess, json, re, os, random
import numpy as np, cv2

SIZE = 150
SCALE = 2

def _integral_argmax(score, size):
    integ = cv2.integral(score.astype(np.float32))
    best, by, bx = -1e18, 0, 0
    H, W = score.shape
    for y0 in range(6, H - size - 6, 10):
        for x0 in range(6, W - size - 6, 10):
            s = integ[y0 + size, x0 + size] - integ[y0, x0 + size] - integ[y0 + size, x0] + integ[y0, x0]
            if s > best: best, by, bx = s, y0, x0
    return by, bx

def pick_regions(clean):
    cs = clean.astype(np.float32)
    tmap = np.abs(cs[1:] - cs[:-1]).max(3).mean(0)          # temporal change
    g = cv2.cvtColor(clean[10], cv2.COLOR_RGB2GRAY).astype(np.float32)
    tex = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3))
    static = tex / (tmap + 2.0)
    motion = tmap * (tex / (tex.max() + 1e-6) + 0.3)
    sy, sx = _integral_argmax(static, SIZE)
    my, mx = _integral_argmax(motion, SIZE)
    return (sy, sx), (my, mx)

def _col(img, y0, x0):
    c = img[y0:y0 + SIZE, x0:x0 + SIZE]
    return cv2.resize(c, (SIZE * SCALE, SIZE * SCALE), interpolation=cv2.INTER_NEAREST)

def _sheet(a, b, clean, region, ts):
    y0, x0 = region
    rows = [np.hstack([_col(clean[t], y0, x0), _col(a[t], y0, x0), _col(b[t], y0, x0)]) for t in ts]
    return np.vstack(rows)

PROMPT_FLICK = """You judge two restorations A and B of the same noisy analog FPV video.
Each image has 3 columns: REFERENCE (the clean original), A, B. The 3 rows show the SAME image region at CONSECUTIVE times t, t+1, t+2 (top to bottom).
Compare A against B ONLY (ignore which column the reference sits in):
1. Temporal flicker: does residual grain/sparkle change noticeably from row to row? Less change is better.
2. Which one matches the reference region's true texture better (not its noise)?
Answer ONLY compact JSON:
{"better": "A"|"B"|"tie", "flicker": "A"|"B"|"tie", "detail": "A"|"B"|"tie", "reason": "<15 words max>"}"""

PROMPT_DETAIL = """You judge two restorations A and B of the same noisy analog FPV video.
Each image has 2 rows; each row has 3 columns: REFERENCE (clean original), A, B, showing the SAME moving region at two different times (top, bottom).
Compare A against B ONLY (ignore which column the reference sits in):
1. Which keeps more REAL detail that the reference shows (edges, texture)?
2. Which introduces fewer fake structures, blotches or smearing?
Answer ONLY compact JSON:
{"better": "A"|"B"|"tie", "detail": "A"|"B"|"tie", "artifacts": "A"|"B"|"tie", "reason": "<15 words max>"}"""

def _ask(png_path, prompt, tag):
    out = f"/tmp/judge_{tag}.json"
    if os.path.exists(out): os.remove(out)
    r = subprocess.run(["z-ai", "vision", "-p", prompt, "-i", png_path, "-o", out],
                       capture_output=True, text=True, timeout=240)
    if not os.path.exists(out):
        return {"better": "tie", "raw": (r.stderr or r.stdout)[-200:]}
    try:
        d = json.load(open(out))
        txt = d.get("content") or d.get("choices", [{}])[0].get("message", {}).get("content", "") or json.dumps(d)
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {"better": "tie", "raw": txt[:200]}
    except Exception as e:
        return {"better": "tie", "raw": f"parse fail {e}"}

def judge(out_cand, out_champ, clean, workdir="crops"):
    os.makedirs(workdir, exist_ok=True)
    (sy, sx), (my, mx) = pick_regions(clean)
    n = len(clean)
    trials = [
        ("flick_static_a", "sheet1", PROMPT_FLICK, (sy, sx), (20, 21, 22)),
        ("flick_static_b", "sheet4", PROMPT_FLICK, (sy, sx), (10, 11, 12)),
        ("detail_motion_a", "sheet2", PROMPT_DETAIL, (my, mx), (18, 24)),
        ("detail_motion_b", "sheet5", PROMPT_DETAIL, (my, mx), (30, 38)),
        ("flick_motion", "sheet3", PROMPT_FLICK, (my, mx), (8, 9, 10)),
    ]
    trials = [t for t in trials if all(u < n for u in t[4])]
    votes = {"cand": 0, "champ": 0, "tie": 0}; details = []
    for name, stem, prompt, region, ts in trials:
        cand_is_A = random.random() < 0.5
        A, B = (out_cand, out_champ) if cand_is_A else (out_champ, out_cand)
        sheet = _sheet(A, B, clean, region, ts)
        p = f"{workdir}/{stem}.png"
        cv2.imwrite(p, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        d = _ask(p, prompt, stem)
        d["trial"] = name; d["cand_is"] = "A" if cand_is_A else "B"
        v = d.get("better", "tie")
        if v == "tie" or v not in ("A", "B"): votes["tie"] += 1
        elif v == d["cand_is"]: votes["cand"] += 1
        else: votes["champ"] += 1
        details.append(d)
    return {"votes": votes, "details": details}
