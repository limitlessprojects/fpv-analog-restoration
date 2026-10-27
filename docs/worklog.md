# Worklog

---
Task ID: 1
Agent: main agent (Z.ai, continuation session)
Task: Continue the Kenyan CA weak-key investigation: verify prior scanner claims, obtain real Kenyan CA certificates, run tests, produce the evidence-backed final report.

Work Log:
- Inherited prior-session artifacts from /home/z/my-project/upload/ (weakkey_scan.py, REPORT.md, RETRIEVAL_LOG_AND_EVIDENCE_TABLE.md, control outputs, emudhra_results.json). Scanner preserved byte-identical; no bug found in review.
- Probed egress: fully open (previous sandbox was not). crt.sh, ca.go.ke, rootca.go.ke (HTTP), govca.icta.go.ke, repository.emsign.com, pki.certysign.io all reachable.
- Re-ran all controls: positive control 16/16 incident certs reproduce published d values (156/230/115/56/155/4/66/237); independent cryptography-library crosscheck 16/16 MATCH; random P-256/P-384/RSA-2048 negatives clean; synthetic weak keys d=7, d=12000 (at limit 65536), d=2^100 all detected. Limit-dependence of d=12000 documented.
- Verified incident sources: Bugzilla 2083449 (HTML archived), Helme post, emudhra.we-have-a-problem.com/impacted-intermediates.json (16 entries with PEMs). Root cause details captured (Ultimaco HSMs, 2025-04-30 ceremony, export-intent weak second keypair, label mixup, Sectigo cross-signs).
- Register re-verified 26 Oct 2026: 6 accredited + 7 non-accredited; Centric Limited absent.
- Retrieval breakthroughs: rootca.go.ke reachable over HTTP (no robots.txt) -> Kenya Root CA 2022 cert + CRLs + RCA CP/CPS v1.0/v2.0 downloaded; repository.emsign.com embedded inventory -> 3 eMudhra Kenya CA certs + 3 CRLs; certysign.io JS bundle -> full PKI endpoint map -> CertySign intermediate (under Kenya Root CA 2022) + own DZ root + CRL + CP/CPS; tendaworld.com -> JuliCA CP/CPS v4 (repository.tendaworld.com NXDOMAIN); geda.co.ke + evrotrust.com (Evrotrust-powered, no certs); govca.icta.go.ke CPS v1.0 obtained (CP PDF 404).
- Sinam: 0 artifacts obtainable (csp.sinam.net 404 everywhere incl. rendered; pki.sinam.net TCP timeout / HTTP 408; sinam.net unreachable from sandbox; no CT entries). Prior Sinam CPS claims left UNVERIFIED.
- Tests: 10 Kenyan cert files -> 6 distinct keys, all RSA-2048/4096, full RSA battery clean (no small factors, no Fermat, no ROCA, no shared primes). EC small-scalar N/A (RSA keys).
- Findings: NO weak Kenyan keys; NEW operational finding - Kenya Root CA 2022 CRL stale (next_update 2026-10-07, 19 days past) with 2 unattributed revoked sub-CA serials; eMudhra Kenya + CertySign CRLs fresh. Documentary verification with exact clauses: GovCA CPS DRAFT watermark (visually confirmed), RSA>=2048, FIPS L2/L3 inconsistency, no post-gen screening (6.1.6), PoP waived (3.2.1), 4.9.12 "No stipulation"; RCA CPS v2.0 sections 3.2.1/6.1.5/6.1.6/6.2.1-6.2.3 + internal L2-vs-L3 tension.
- Evidence bundle: /home/z/my-project/download/kenya-ca-weakkey/ with manifest.sha256 (145 files), ca_cert_inventory.csv, retrieval_log.md (45 logged attempts), sources.json, all certs/CRLs/docs/controls.
- Final report: REPORT.md (primary handoff artifact) + Kenya_CA_WeakKey_Report.pdf (12 pages, cover via html2poster.js Template 01 + ReportLab body, pdf_qa PASS, toc.check clean, font.check 0 issues).

Stage Summary:
- Deliverables: download/kenya-ca-weakkey/REPORT.md, download/kenya-ca-weakkey/Kenya_CA_WeakKey_Report.pdf, full evidence bundle with hashes.
- Key results: 6 distinct Kenyan keys tested, 0 confirmed weak; scanner revalidated 16/16 + independent crosscheck; 1 new confirmed operational finding (stale Root CRL); register verified 26 Oct 2026; Sinam remains the only accredited provider with zero obtainable artifacts.
- Scripts persisted under /home/z/my-project/scripts/ (kenya_report_body.py, kenya_merge.py, extract_impacted.py, independent_crosscheck.py, make_negctl.py, ct_sweep.py, build_inventory.py, kenya_cover.html).

---
Task ID: 2 (round-2 deepened pass)
Agent: main agent (Z.ai, continuation session)
Task: "continue check deeper bro theres so much youd missed" - deepen the Kenyan CA weak-key investigation beyond the round-1 state.

Work Log:
- Read worklog; verified round-1 artifacts intact under download/kenya-ca-weakkey/.
- Re-parsed Root CRL -> noticed autoindexes on rootca.go.ke (/cacerts/, /crls/, /Downloads/). DISCOVERED 4 CRL files with 3 distinct contents; round 1 had only the expired one.
- Downloaded + parsed CRL #7 (fresh, 2026-10-19 -> 2027-04-17) and CRL #1 (2022-11-17, empty). All 3 Root CRL generations + 4 sub-CA CRLs signature-verified OK (round 1 never verified signatures).
- CORRECTED the round-1 "stale CRL" finding: real defect is publication infrastructure (CRLDP-designated file expired while fresh CRL sits under an unreferenced filename; /Downloads still serves CRL #1). Both 180-day CRLs violate RCA CPS v1.0 line 907 and v2.0 7.2.1 (30-day max; v1.0 also mandates 7-day issuance).
- OCSP probing (3 constructed requests) -> NO responder at CPS-advertised http://rootca.go.ke (HTML returned, not ASN.1). New confirmed gap.
- TLS harvest of 22 provider hosts (scripts/tls_harvest.py) -> 19 alive; all provider TLS from public CAs; *.icta.go.ke issued by eMudhra EM DV TLS CA - G2A-1; rootca.go.ke has no TLS.
- CT round 2 (crt.sh): Evrotrust identity -> "Evrotrust RSA REG.KE Operational CA 2025" = GEDA's Kenya operational CA. Fetched authoritative source: EU LOTL -> Bulgarian TSL_BG.xml (crc.bg) -> extracted all 21 Evrotrust certs (scripts + manual parse) incl. REG.KE CA (RSA-4096). STRUCTURAL FINDING: GEDA's Kenya service does NOT chain to Kenya Root CA 2022.
- Scanned everything: final_scan_round2.json = 32 certs, 27 distinct keys, 0 parse errors, 0 dup SPKIs, 0 shared primes (351 pairs), all clean. Round-1 6/6 clean result stands.
- Negative results documented: web.archive.org proxy-blocked (CDX + WebFetch path); archive.ph has NO snapshot of repository.tendaworld.com; crt.sh JuliCA/CMCOM/Hakisha = 0; Kensign/Kashia/Digicheti = 404; Kenya Gazette web search empty; pdfsig: no embedded signatures in any CP/CPS; CCADB resources page: no Kenya entry; govca.icta.go.ke = Drupal shell, no repo paths; csp.sinam.net = HTTP->HTTPS->404 loop on 8 paths; ca.go.ke re-fetch blocked by JS challenge.
- Updated REPORT.md to rev 2 (Bottom line, ecosystem, certificate table + CRL history table, test results, new 7.6 compliance subsection, result table, 5 answers, disclosure, evidence, sources). Rebuilt PDF via existing pipeline (cover + body patched): 14 pages, pdf_qa PASS (1 cosmetic cover-margin warning).
- Rebuilt kenya/ca_cert_inventory.csv (33 rows) + manifest.sha256 (263 files) via build_inventory.py (extended for round-2 sources); updated sources.json (rev2_round2 block); retrieval_log.md extended to 70 entries.
- Work preserved: no scanner changes; no private-key material touched; revoked-serial ownership left unattributed (no fabrication).

Stage Summary:
- Deliverables updated: REPORT.md (rev 2), Kenya_CA_WeakKey_Report.pdf (14 pp, QA PASS), sources.json, retrieval_log.md (70 entries), ca_cert_inventory.csv, manifest.sha256 (263 files).
- Headline: 27 distinct keys tested (6 Kenyan + 21 Evrotrust), 0 weak; 5 confirmed operational findings at the national root; GEDA structural finding (Bulgarian chain); CRL history reconstructed (#1/#6/#7); OCSP absence proven; 2 revoked serials still unattributed (only CA Kenya can answer; request template in report 9.5).
- Key scripts added: scripts/tls_harvest.py, scripts/round2_retrieval.py, scripts/wayback_cdx_sweep.py, scripts/wayback_cdx_parallel.py.

---
Task ID: 3 (open-source contribution round)
Agent: main agent (Z.ai, continuation session)
Task: Contribute 3 tested, non-spam PRs to Nitjsefnie/Overflow per user instruction; study past merged PRs, implement original ideas, use provided GitHub token (account limitlessprojects).

Work Log:
- Read repo governance: CONTRIBUTING.md (667 lines), PR/issue templates, module-size + coverage + comment-share ratchets, /claim workflow, rebase-only, AI disclosure rules (Co-Authored-By trailer + "Generated by <model> (<roles>)" footer).
- Studied 4 merged external PRs (#1403/#1374/#1404/#1378) for house style; identified 3 unclaimed, precisely-specified issues: #1396 (offered: small, missing prefixed-threshold test for fold-site diff-only read + stale comment), #1382 (stale apiCall comments after PR 1366), #1379 (README credit-economics summary).
- Environment: Node 24.21, pnpm 10.33.0 via npm i -g; deps installed; vitest verified; no Docker locally (container suites left to CI, disclosed in PR bodies).
- PR #1417 (fixes #1396): new test in tests/fold/reconciliation-budget-hold.test.ts pins OVERFLOW_RECONCILIATION_DIFF_ONLY_BACKLOG=5 with backlog 6 through the real spend path (full + charged); mutation check (bare process.env read) fails exactly the new test, then reverted; stale comment in tests/db/sweep-catchup.test.ts rewritten. 113 tests green, lint/typecheck/module-size clean. APPROVED + MERGED same day.
- PR #1418 (fixes #1382): apiCall docstring + network-failure test comment updated to post-1366 behavior. CI caught comment-share ratchet: file unrecorded, 15% target, my +126 comment bytes pushed it to 16.4%. Cloned Nitjsefnie-Actions/comment-ratchet, ran exact measurer locally, tightened docstring to one condition line + trimmed 4 stale comments (net ~0 bytes, 14.6877%). 43 tests green, force-pushed, posted measurement confirmation comment.
- PR #1420 (fixes #1379): two-sentence bold-lead economics summary in README before "Why it can't be abused"; docs-only; branch cut from current main vocabulary.
- Claim discipline: /claim exact-body comments; read-tier limit (2 open claims) respected; #1379 claimed only after #1417 merged and freed a slot.

Stage Summary:
- Deliverables: merged PR #1417 (fixes #1396); open PR #1418 (fixes #1382, comment-share remediated); open PR #1420 (fixes #1379).
- Scripts: /home/z/my-project/scripts/measure_share.py (local approximation; exact measurement used via cloned comment-ratchet at /tmp/comment-ratchet).
- Token stored at /home/z/my-project/.ghtoken (chmod 600); package.json onlyBuiltDependencies local hack reverted before commit (git tree clean of it).
- Ledger effect: #1417 settles credit to limitlessprojects when the sponsor applies settled: label within window; #1396 was offered: small.

---
Task ID: 4 (attribution fix + discovered-bug fix round)
Agent: main agent (Z.ai, continuation session)
Task: Fix maintainer-requested attribution on PRs #1418/#1420; claim and fix discovered bug #1419.

Work Log:
- Maintainer blocked both open PRs on one item: attribution named the family "GLM" instead of the exact model. User confirmed via AskUserQuestion: GLM-5.3-Flash. Web search corroborated Z.ai's public naming (GLM-5.3 / GLM-5.3-Flash / GLM-5.3-FlashX as of 2026-10).
- Amended both commits: Co-Authored-By: GLM-5.3-Flash <noreply@z.ai>; force-pushed fix/1382-relay-apicall-comments and docs/1379-readme-economics; PATCHed both PR bodies' Footer to "Generated by GLM-5.3-Flash (implementation, testing)"; posted confirmation comments.
- Read issue #1419 (bug, trivial): tests/db/sweep-catchup.test.ts fails 9 tests when OVERFLOW_RECONCILIATION_DIFF_ONLY_BACKLOG is exported (suite sets bare name only; code resolves prefixed first). Claimed it after #1420 merged (read-tier slot freed).
- Fix on branch fix/1419-catchup-suite-env: beforeAll deletes the prefixed variable beside setting bare=10; afterAll restores originalPrefixedThreshold; comment explains why. Lint + typecheck clean; container-backed suite runs in CI (disclosed in PR body).
- Results: #1420 MERGED (issue #1379 settled: 1); #1437 opened and MERGED same day (issue #1419 settled: 2); #1418 green, awaiting maintainer re-review of the attribution fix.

Stage Summary:
- Merged: #1417 (settled: 2), #1420 (settled: 1), #1437 (settled: 2). Open: #1418 (fixes #1382), all checks green.
- Account limitlessprojects ledger total so far: 5 settled points (2+1+2), pending #1418's settlement.
- Patterns learned this round: the comment-share ratchet is the most common trap for comment-touching PRs; attribution precision is enforced per provider naming; the maintainer values "does not depend on the environment" test hygiene.

---
Task ID: FPV-1 (fpv restoration continuation, agent-loop session)
Agent: main agent (Z.ai, continuation session)
Task: Continue analog FPV restoration with a propose/apply/blind-judge/gate/keep-revert agent loop, VLM judge, hard gates first, flicker metric added, deep research, real-content demo.

Work Log:
- Deep research: ntsc-rs, OnAir (analog FPV mode confirmed, v0.2.7+), ProPainter confirmed real. Site reachability probed: mixkit serves direct mp4s (downloaded 159-frame 640x360 drone clip), zenodo reachable, pixabay/pexels 403, makearmy DNS-dead.
- Rebuilt harness from bundle into /home/z/my-project/fpv/: sim (test start=60 seed=7, tune start=150 seed=11), config-driven restore.py, metrics2.py with flicker metrics (static_std_pairs, tdiff_excess_pairs), judge.py (z-ai vision, anonymized [ref|A|B] sheets, randomized A/B, 3 trials majority), iterate.py (champion.json + ledger.json, hard gates G0-G5 before score, keep/revert).
- Champion seeded at 27.39 dB / 0.809 SSIM on tune clip; floor baseline med3+nlm 23.29 dB.
- Diagnostics: residual flicker is born pre-polish; chroma already solved (13.5 -> 1.7); luma grain + event steps dominate; ghost-segment detector fails in deep fades (echo window overlaps q=0.02); fast-motion jerk pairs (GT diff 35-50) masquerade as flicker spikes.
- 21 loop iterations: kept exactly one change (event-flicker smoother, w=0.15, max 6 pairs, 3-pair static gate). VLM judge vetoed two PSNR-positive over-smoothers (windowed polish +0.06 dB, wider pass 2). Hard gates killed flow damping, vertical inpaint fill, sharpening, spread-gated EFS.
- Final: locked config scored once on test clip: 27.73/0.778 all frames, flick_static_good 2.67 vs 9.25 best baseline. Real-content demo (mixkit through simulated link, seed=21): 25.32 dB (+8.5), flicker 34.2 -> 3.1, side-by-side mp4 written.
- Deliverables in /home/z/my-project/download/fpv/: fpv_restoration_bundle.md (v2, updated status/tables/lessons), code/ snapshot, sheet_test.png, sheet_real.png, real_sidebyside.mp4, final/real scores + ledger.

Stage Summary:
- Loop works as designed: gates+score+VLM blocked 19 of 21 proposals; the one keeper targets the previously unmeasured flicker axis and won it 3.5x vs best baseline on held-out data.
- Test-clip PSNR/SSIM reproduce previous session (27.73/0.778), now with flicker metrics attached and a real-content generalization check.
- Known limits unchanged: simulated damage, shared author bias, fades headroom, 3-trial judge fragility.

---
Task ID: 3-a
Agent: Explore (FPV_DVR_CLEANER analysis)
Task: Clone and analyze Bratwurstexe/FPV_DVR_CLEANER for license, pipeline techniques, CPU/headless feasibility as baseline.

Work Log:
- Cloned Bratwurstexe/FPV_DVR_CLEANER (v0.8.0, 50-commit depth) to refs/fpv_dvr_cleaner; read LICENSE (MIT, 2026 David-Andreas Tornow), README, docs/INDEPENDENCE.md (provenance record vs the paid Rust tool), THIRD-PARTY-NOTICES.md, docs/STATUS.md, and all 16 engine modules in src/flugbild/ (~2.9k lines) plus CI workflow and tests.
- Stack: Python engine (numpy + opencv-python-headless only, no torch/onnx) driven by a .NET 8 Avalonia GUI over a JSON-line bridge; FFmpeg does decode/encode. GPU = OpenCL via cv2.UMat for full-res fusion only, with self-test + automatic CPU fallback (FLUGBILD_GPU=0); no CUDA anywhere. Read all key algorithms at line level: analog.py (chroma repair), upscale.py (5-frame fusion), osd.py (gradient coherence + region refill), digital.py (HDZero blocks), assess.py (artefact scoring).
- Verified headless feasibility empirically in sandbox: imports work with existing numpy 2.1.3 / cv2 4.13.0 / ffmpeg on PATH, no pip install needed; tiny synthetic smoke of fuse_stream (streak 122->10 delta vs base) and analog.repair_stream ran in <0.1 s on CPU; GPU correctly reports disabled. Library-style usage confirmed via tools/benchmark.py worker + tests (from flugbild.restoration import process_temporal).
- Mapped parameter surface (recipe.Arbeitsrezept defaults: farbberuhigung=35, rauschminderung=0, osd_entfernen=False, aufwand=balanced, CRF 20, mp4-h264) and the Fast/Balanced/Best EFFORTS table (neighbour radius, flow width, flow preset, fill scale). Confirmed what is ABSENT: no luma gain tracking, no flicker smoother, no sync/roll recovery, no luma dropout-line repair; luma is deliberately never touched by the analog stage.
- No code modified anywhere in the repo; analysis only (refs/ clone is the only artifact).

Stage Summary:
- License verdict: MIT - free to reuse code AND ideas in our project with attribution; no copyleft, no restrictions (bundled FFmpeg/x264 stays external and unmodified).
- Top borrowable techniques: (1) 5-frame DIS-flow + min/max-network median fusion with closeness weights and inter-frame agreement "trust" gate (upscale.py); (2) gradient-coherence OSD detection (summed vectors / summed lengths per channel over 40 probes) + rim-fit neighbour patching with plane brightness correction (osd.py); (3) chroma-only analog repair: hue-tear + colour-dropout row scores, 6 aligned neighbours, 2-iteration Cauchy estimate, fade-to-grey fallback, luma-guided band refill (analog.py); bonus: assess.py artefact metrics (grain/colour/streaks/seams vs median-blur floor) map well onto our metrics2.py.
- Runnability verdict: YES - CPU-first headless baseline runnable right now in this sandbox (deps already satisfied; only needs an mp4, not .npy - wrap .npy clips to mp4 first, or score frame arrays by calling repair_window/fuse_stream directly).

---
Task ID: 3-b
Agent: Explore (AnaHD analysis)
Task: Clone and analyze dominic-salas/AnaHD for license, classic-mode pipeline details, CPU/headless feasibility as baseline.

Work Log:
- Cloned dominic-salas/AnaHD (v0.1.1, commit 880e8b1) to /home/z/my-project/refs/anahd; read README, LICENSE, THIRD_PARTY_LICENSES.md, pyproject and all core sources (clean.py, pipeline.py, gpu/pipeline.py, gpu/flow.py, gpu/breakup.py, look.py, picture.py, media.py, model/*, learn/distill.py, remaster/stage.py, cli.py).
- License: MIT confirmed (LICENSE, (c) 2026 Dominic Salas); weights NOT in git - released as model-v6/anahd-v6.pt, 5,982,449 bytes (6.0 MB), SHA-256-pinned (weights.py:21), verified live via GitHub API; README states weights are MIT too. RAFT-small weights = torchvision (BSD-3); SeedVR2 7B teacher Apache-2.0, not distributed.
- Mapped classic (training-free) pipeline exactly: CPU path = DIS-flow motion-compensated fusion (radius +-3, match weight 1/(1+(diff/7)^4)) -> row destripe (241-window, clip 4) -> derainbow (cut 0.8, texture/6.0) -> per-clip tone LUT + vibrance -> Lanczos + clamped sharpen -> guided chroma -> x264 (src/anahd/clean.py:96-167, pipeline.py:118-207, look.py). GPU variant swaps DIS for RAFT-small+LK and adds breakup repair (gpu/breakup.py).
- Smoke-tested classic mode headless on CPU in this sandbox: 60-frame 640x360 clip -> 1920x1080 in 14 s (4.3 fps) using only preinstalled python3.12/ffmpeg7.1/numpy2.1/cv2.4.13; torch never imported on --cpu path (cli.py:425). Command recorded in report.
- Recorded borrowables: per-clip tone curve (no brightness pumping), 3-frame mean-colour deflicker (remaster/stage.py:213-222), flow-warped temporal chroma cleanup, second-smallest-disagreement breakup detection + <=3-frame bridging, and SeedVR2 distillation recipe (teacher t-1..t+1 flow-warped, Gaussian SIGMA=0.02 agreement average, texture-energy loss).

Stage Summary:
- License verdict: MIT incl. weights (6 MB, checksum-pinned, downloadable) - free to borrow code and even the checkpoint.
- Top borrowable techniques: (1) agreement-gated flow-warped multi-frame fusion (no-ghost temporal denoise), (2) row-statistics breakup detection/repair with flow bridging, (3) measured-once tone curve + 3-frame chroma deflicker for flicker without luma pumping.
- Runnability verdict: YES - classic mode runs CPU-only headless in this sandbox today at ~4-5 fps for 640x360 inputs (numpy+OpenCV+FFmpeg, no torch); learned mode is CUDA-only (sm_75+) and impractical without NVIDIA GPU.
