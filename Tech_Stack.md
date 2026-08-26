### Image processing & classical CV (Ambreesh and Sagar) divide topics on your own.
- **OpenCV (`opencv-python`)** — SIFT/ORB baseline, RANSAC, `findTransformECC`, `phaseCorrelate`, `logPolar` (Fourier-Mellin), pyramids.
- **scikit-image** — additional filters, transforms, metrics (SSIM, etc.).
- **NumPy / SciPy** — array math, FFT, interpolation.
- **phasepack** (`pip install phasepack`) — Kovesi's Phase Congruency implementation in Python — the core building block for illumination-invariant features (used in RIFT/HOPC-style descriptors).
- **Matplotlib** — visualization during development (checkerboards, heatmaps, quiver plots of matches).
- **PyTorch**
- **Kornia** (`kornia.feature`) — ready-to-use implementations of **LoFTR**, **DISK**, **SuperPoint + LightGlue**. These give you a strong deep-learning baseline in a few lines of code, no training required — critical given your time budget.
- **GDAL** + **rasterio** — reading georeferenced rasters, reprojecting, `gdaldem hillshade` for synthetic relighting.
- **pvl** (`pip install pvl`) — parses PDS3/PDS4 label (.LBL/.XML) files that accompany Chandrayaan-2 and LRO images.
- **pds4_tools** (if Chandrayaan-2 data is distributed as PDS4) — reading PDS4 image products.
- **spiceypy** (optional, stretch goal) — if you need camera pointing/geometry (SPICE kernels) for more rigorous geometric correction.

### Backend (Manthan)
- **FastAPI** + **Uvicorn** — REST API wrapping the registration pipeline.
- **Pydantic** — request/response schemas.
- **Background tasks** (FastAPI `BackgroundTasks`, or a simple in-process queue) — registration takes seconds—minutes per pair, so don't block requests.
- **Pillow** — serving/encoding result images.

### Frontend (Suryansh)
- **React (Vite)** + **Tailwind CSS** — fast to build, matches your frontend/design member's skillset.
- **OpenSeadragon** or a custom canvas swipe-compare component — for viewing large lunar images with pan/zoom and a before/after slider.
- **Recharts** or simple HTML — displaying RMSE/inlier metrics.

### DevOps / Deployment (Rituraj)
- **Docker** — one Dockerfile for backend (Python + GDAL base image), one for frontend (Node build → Nginx serve).
- **docker-compose.yml** — orchestrates both + a shared volume for image data.
- **GitHub Actions** (optional, stretch) — basic CI to catch broken builds.

### Presentation (Arya Roy)
- **Canva** for the pitch deck.
- **Draw.io** for the architecture diagram.
