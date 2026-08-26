# PS 26166 — Strategy & Team Plan
**Multi-modal, Sun-Angle and Scale-Invariant Image Correspondence using Chandrayaan-2 Optical Images (OHRC, TMC-2, IIRS)**
Organization: ISRO / Department of Space | Theme: Space Technology

---
# Members list (Abbreviations used later)
- **Member 1** - Suryansh
- **Member 2** - Manthan
- **Member 3** - Ambreesh
- **Member 4** - Sagar
- **Member 5** - Arya
- **Member 6** - Rituraj
---
## 0. Restating the problem in plain terms

You need software that takes two lunar images of (roughly) the same ground area — one from Chandrayaan-2 (OHRC / TMC-2 / IIRS) and one reference image (LRO NAC or SELENE) — and finds accurate **matching point pairs** between them, then aligns ("registers") them, even though:

- **The sun was in a different position** when each was taken (different shadows, brightness — "illumination variation").
- **The satellites saw the ground from different angles/orbits** ("viewpoint variation").
- **The two images have very different resolutions** — OHRC is ~25 cm/pixel, TMC-2 ~5 m/pixel, IIRS ~80 m/pixel, LRO NAC ~0.5–2 m/pixel ("scale variation").

Output required: matched point pairs at **sub-pixel accuracy**, spread evenly across the image (not clustered in one corner), plus evaluation metrics (RMSE, inlier count, inlier ratio).

This is a well-studied problem in remote-sensing literature called **multi-modal / multi-temporal image registration**, and it has known algorithmic building blocks — you are not starting from zero. The job is to assemble and tune the right pipeline, not invent new math from scratch.

---

## 1. Tech Stack

### Core language & environment
- **Python 3.11** for the entire CV/ML pipeline (best library support for planetary imagery + deep learning).
- **Git + GitHub** for version control, with a simple branch-per-member workflow.
- **Conda or venv** for a reproducible environment (`environment.yml` / `requirements.txt`).

### Image processing & classical CV
- **OpenCV (`opencv-python`)** — SIFT/ORB baseline, RANSAC, `findTransformECC`, `phaseCorrelate`, `logPolar` (Fourier-Mellin), pyramids.
- **scikit-image** — additional filters, transforms, metrics (SSIM, etc.).
- **NumPy / SciPy** — array math, FFT, interpolation.
- **phasepack** (`pip install phasepack`) — Kovesi's Phase Congruency implementation in Python — the core building block for illumination-invariant features (used in RIFT/HOPC-style descriptors).
- **Matplotlib** — visualization during development (checkerboards, heatmaps, quiver plots of matches).

### Deep-learning matchers (pretrained, no training needed)
- **PyTorch**
- **Kornia** (`kornia.feature`) — ready-to-use implementations of **LoFTR**, **DISK**, **SuperPoint + LightGlue**. These give you a strong deep-learning baseline in a few lines of code, no training required — critical given your time budget.

### Planetary / remote-sensing data handling
- **GDAL** + **rasterio** — reading georeferenced rasters, reprojecting, `gdaldem hillshade` for synthetic relighting.
- **pvl** (`pip install pvl`) — parses PDS3/PDS4 label (.LBL/.XML) files that accompany Chandrayaan-2 and LRO images.
- **pds4_tools** (if Chandrayaan-2 data is distributed as PDS4) — reading PDS4 image products.
- **spiceypy** (optional, stretch goal) — if you need camera pointing/geometry (SPICE kernels) for more rigorous geometric correction.

### Backend
- **FastAPI** + **Uvicorn** — REST API wrapping the registration pipeline.
- **Pydantic** — request/response schemas.
- **Background tasks** (FastAPI `BackgroundTasks`, or a simple in-process queue) — registration takes seconds—minutes per pair, so don't block requests.
- **Pillow** — serving/encoding result images.

### Frontend
- **React (Vite)** + **Tailwind CSS** — fast to build, matches your frontend/design member's skillset.
- **OpenSeadragon** or a custom canvas swipe-compare component — for viewing large lunar images with pan/zoom and a before/after slider.
- **Recharts** or simple HTML — displaying RMSE/inlier metrics.

### DevOps / Deployment
- **Docker** — one Dockerfile for backend (Python + GDAL base image), one for frontend (Node build → Nginx serve).
- **docker-compose.yml** — orchestrates both + a shared volume for image data.
- **GitHub Actions** (optional, stretch) — basic CI to catch broken builds.

### Presentation
- **Canva / PowerPoint** for the pitch deck.
- **OBS Studio** or phone screen-recording for the demo video.
- **Draw.io / Excalidraw** for the architecture diagram.

---

## 2. Team Role Breakdown

| # | Member | Primary Ownership | Concrete Deliverables |
|---|--------|--------------------|------------------------|
| 1 | Frontend & Design | React app, visual identity | Upload/demo-select page, results viewer with pan/zoom + before/after slider, matched-points overlay, metrics panel, pitch deck visuals |
| 2 | Backend (FastAPI) + Deployment | API + infra | FastAPI service wrapping the pipeline, job queue/status endpoints, Dockerfiles, docker-compose, deployment (Render/Railway or local demo box) |
| 3 | AI/ML (Lead algorithm) | Core registration pipeline | Illumination-invariant feature pipeline (Phase Congruency / RIFT-style), coarse-to-fine multi-scale strategy, RANSAC geometric verification |
| 4 | AI/ML (Co-lead algorithm) | Deep-learning + refinement | Kornia LoFTR/DISK integration & benchmarking against classical pipeline, sub-pixel refinement, ensembling/fallback logic between classical & deep methods |
| 5 | UI/UX + Research | Domain research, data, UX flow | Literature review (RIFT/HOPC/CFOG/LoFTR papers), PDS/ISRO data format research, dataset curation incl. synthetic data, ground-truth control points for evaluation, UX flow design, evaluation-metrics module (RMSE/inlier ratio/coverage) |
| 6 | Presentation + Docker + Coordination | Delivery & narrative | Docker support to Member 2, daily standup coordination, demo video, pitch deck narrative, judge Q&A prep, architecture diagram, README polish |

**Pairing logic:** 3+4 own the algorithm (the hardest, most judged part — two brains reduce risk). 1+2 own the product shell in parallel so integration isn't a last-day scramble. 5 is the glue — feeds data and domain understanding to 3/4, and feeds UX requirements to 1. 6 removes friction for everyone (infra, deadlines, story) and owns the thing judges actually see in the last hour: the pitch.

---

## 3. Step-by-Step Walkthrough: Research → Data → Prototype

Assume **5 working days** before your prototype needs to be demo-ready. Adjust proportionally if you have more/less time — but do not skip Phase 0 or Phase 1; a wrong assumption there wastes everyone's time downstream.

### Phase 0 — Research & Framing (Day 1, morning) — *Owner: Member 1 and 5, Others just read*
- Understand *why* plain SIFT/ORB fails here: they match on raw intensity gradients, which change non-linearly under different sun angles.
- Read up on (summaries, not full papers — use AI assistants here, see §4):
  - **Phase Congruency** (Kovesi) — illumination-invariant feature detection.
  - **RIFT** (Radiation-invariant Feature Transform) — SOTA classical method for multimodal remote-sensing matching.
  - **HOPC** (Histogram of Oriented Phase Congruency) — descriptor built on phase congruency.
  - **LoFTR** — detector-free deep learning matcher (Kornia has it ready-to-use).
  - Chandrayaan-2 payload specs (OHRC, TMC-2, IIRS resolutions/altitudes) and LRO NAC specs.
  - PDS3/PDS4 file format basics (what a `.IMG`/`.LBL` or `.xml` pair contains).
- **Output of this phase:** a one-page shared doc (Notion/Google Doc) with the chosen algorithmic approach and a list of exact resolutions (GSD) for each Chandrayaan-2 instrument vs LRO NAC — you'll need these numbers explicitly for scale normalization.

### Phase 1 — Data Acquisition & Environment Setup (Day 1, afternoon) — *Owner: Member 4 (data), Member 2 (env)*
- Register on ISSDC (`chmapbrowse.issdc.gov.in`) immediately — access approval can take time, so start this the moment you start the project, not when you need the data.
- **De-risk the timeline:** while waiting on real dataset access/approval, generate a **synthetic proxy dataset** so the algorithm team is never blocked:
  - Get a public lunar DEM (e.g., LOLA) + a base albedo mosaic (e.g., LRO WAC global mosaic).
  - Use `gdaldem hillshade -az <azimuth> -alt <altitude>` to render the *same* terrain under multiple different sun angles — this gives you unlimited illumination-varied image pairs with **known, perfect ground-truth correspondence** (since it's literally the same map, just relit), ideal for building and unit-testing your matching pipeline before real data even arrives.
- **Find a known-good real test case:** LRO NAC has publicly documented images of the Vikram Lander site, which Chandrayaan-2's OHRC has also imaged — a well-documented, small, findable overlap region. Use this as your first real-data test pair once ISSDC access comes through.
- Set up the shared Python environment (`requirements.txt`), repo structure, and a shared cloud folder / Git LFS or DVC for the (large) image files.

### Phase 2 — Baseline Pipeline: Prove the Problem (Day 1 evening – Day 2 morning) — *Owner: Member 3 with Member 4*
- Run plain OpenCV SIFT/ORB matching on your synthetic illumination-varied pairs.
- Show it degrading/failing as sun angle difference increases. This is genuinely useful for your pitch: "here's the baseline, here's why it breaks, here's what we built instead."

### Phase 3 — Illumination-Robust Matching Pipeline (Day 2) — *Owner: Member 3, Partner: Member 4*
- Implement Phase Congruency-based feature detection using `phasepack`.
- Build a HOPC/RIFT-style descriptor around detected keypoints (histogram of oriented phase-congruency gradients in a local patch).
- Test against the synthetic dataset — this should clearly outperform raw SIFT under illumination change.

### Phase 4 — Deep-Learning Matcher & Ensemble (Day 2–3) — *Owner: Member 4, Partner: Member 3*
- Integrate Kornia's `LoFTR` (pretrained `outdoor` weights) as a second, independent matcher.
- Benchmark it against the classical pipeline on the same synthetic pairs.
- Decide (with data, not guesswork) whether to use LoFTR alone, the classical pipeline alone, or run both and keep whichever gives more/better-distributed inliers per image pair — a simple, defensible ensemble strategy.

### Phase 5 — Scale Handling & Coarse-to-Fine Strategy (Day 3) — *Owner: Members 3 and 4*
- Use each product's known GSD (from Phase 0) to pre-resample images to a comparable scale **before** feature matching — don't rely on the descriptor alone to bridge >10x scale gaps.
- Build a Gaussian image pyramid; do coarse alignment at low resolution (phase correlation or ECC for a rough global shift/scale/rotation via Fourier-Mellin/log-polar transform), then refine with local feature matching at full resolution on the roughly-aligned crop.
- Apply RANSAC (or MAGSAC) to reject outlier matches and estimate a transform (start with similarity/affine; only go to full homography/TPS if you have time and see systematic residual distortion).

### Phase 6 — Sub-Pixel Refinement & Metrics (Day 3–4) — *Owner: Member 4*
- Refine each inlier match with normalized cross-correlation on an upsampled local patch or parabolic peak-fitting of the correlation surface.
- Implement evaluation outputs: **RMSE**, **inlier count**, **inlier ratio**, plus a spatial-coverage check (grid-based, so you can honestly claim "uniform distribution across the image" as the PS requires).
- Generate visualization artifacts: checkerboard mosaic of the two registered images, a difference heatmap, and a quiver plot of matches.

### Phase 7 — Wrap in API + Frontend (Day 3–4, in parallel with above) — *Owner: Members 1+2*
- Member 2 exposes the pipeline as a FastAPI service (`/register`, `/jobs/{id}`, `/jobs/{id}/result`).
- Member 1 builds the upload/demo-select page and the results viewer (overlay slider, matched points, metrics panel).
- Integrate early and often — don't wait until Day 5 to connect frontend and backend.

### Phase 8 — Dockerize, Test, Polish, Present (Day 4–5) — *Owner: Member 6 (leads), Presentation: Member 5 (ppt and flow charts), everyone contributes*
- Dockerize both services; verify `docker-compose up` works on a clean machine.
- Run the full pipeline end-to-end on: (a) synthetic pairs, (b) the real Vikram-site pair if data has arrived, (c) at least one deliberately hard case (large sun-angle difference) to show robustness honestly.
- Build the pitch deck and demo video. Rehearse the story: problem → why classical methods fail → your approach → results with metrics → live demo.

---

## 4. AI Assistant Usage Strategy (you have very little time — spend it on what only humans can do)

The core principle: **use AI heavily for anything that is boilerplate, research-summarization, or debugging; do NOT outsource algorithm correctness judgment or the final evaluation to AI without verifying against your own ground-truth data.**

| Task | Which AI tool | When | Why |
|---|---|---|---|
| Summarizing papers (RIFT, HOPC, Phase Congruency, LoFTR) into a working understanding | Claude / ChatGPT | Phase 0, Day 1 morning | Turns hours of paper-reading into a 30-minute briefing; ask for "explain like I need to implement this in Python this week," not just theory |
| Understanding PDS3/PDS4 format, ISSDC/LROC download quirks | Claude / ChatGPT + web search | Phase 1 | Saves time decoding unfamiliar planetary-data documentation |
| Scaffolding boilerplate (FastAPI app skeleton, React component skeletons, Dockerfiles, requirements.txt) | Claude / ChatGPT | Phases 1, 7, 8 | This is pure time-saving with low risk — boilerplate is easy to verify by running it |
| Inline coding while implementing the pipeline | GitHub Copilot / Cursor | Phases 2–6 | Fast autocomplete for repetitive NumPy/OpenCV code inside your editor, keeps flow state |
| Debugging error messages / stack traces (GDAL install issues, Kornia shape errors, etc.) | Claude / ChatGPT | Continuously | Fast turnaround beats StackOverflow-diving under time pressure |
| Generating the synthetic-data script (hillshade rendering pipeline) | Claude / ChatGPT | Phase 1 | A well-specified, verifiable script — good AI use case |
| **Algorithm design decisions** (which descriptor, which transform model, threshold tuning) | AI as a *sounding board*, humans decide | Phases 3–6 | Ask AI to explain trade-offs, but validate every choice against your own synthetic ground-truth numbers — don't ship an AI-suggested threshold untested |
| **Final evaluation numbers reported to judges** | Human-verified only | Phase 6 | Never let AI compute or fabricate a metric you report; run your own metrics code on real output and read the number yourself |
| Writing README, code comments, technical report sections | Claude / ChatGPT | Phase 8 | Good use of AI — you already know the content, it just needs to be written up clearly |
| Pitch deck copy, script for demo video | Claude / ChatGPT | Phase 8 | Speeds up wording; Member 6 still owns the actual delivery/rehearsal |
| Quick fact-checking on ISRO mission specs, instrument GSDs, sun-angle-related terminology | Perplexity or Claude with web search | Phase 0 | Fast, citable answers for facts that will appear in your pitch |

**Two hard rules:**
1. Never let an AI assistant invent or "smooth over" a metric, dataset detail, or capability claim in your final report/pitch — everything you present to judges must be something your own code actually produced and you can reproduce live if asked.
2. Whenever AI generates a non-trivial algorithm snippet (e.g., a RANSAC variant, a descriptor implementation), have Member 3 or 4 read it line-by-line and test it against the synthetic ground-truth pairs before it goes into the pipeline — CV code that "runs without errors" is not the same as CV code that's correct.

---

## 5. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| ISSDC data access approval is slow | Blocks real-data testing | Synthetic hillshade dataset (Phase 1) keeps the algorithm team working in parallel from Day 1 |
| Chandrayaan-2 and LRO NAC have little/no actual coverage overlap for an easy test region | No real validation pair | Target the well-documented Vikram Lander site overlap as a known-good first case |
| Raw (non-map-projected) images have too much geometric distortion for 2D image matching alone | Poor/impossible matching | Prefer RDR (calibrated, ideally map-projected) products over raw EDR; treat full photogrammetric correction with SPICE as a stretch goal, not a requirement |
| Deep-learning matcher (LoFTR) performs poorly out-of-domain on lunar imagery | Wasted integration time | Time-box the LoFTR experiment (half a day); fall back to the classical Phase-Congruency pipeline as your primary if it underperforms |
| Frontend/backend integration left too late | Demo doesn't work live | Phase 7 runs in parallel with Phases 3–6, not after |
| Team runs out of time before Docker/deployment works | Can't show a "real product" | Member 6 owns Docker from Day 1 as a standing task, tested incrementally, not attempted for the first time on the last day |

---
