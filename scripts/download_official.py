#!/usr/bin/env python3
"""Download the public Transolver/Geo-FNO benchmark cases from Hugging Face."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(
    os.environ.get("CFD_BENCHMARK_DIR", ROOT / "data" / "cfd_benchmark")
).expanduser()
OFFICIAL = Path(
    os.environ.get("OFFICIAL_DATA_DIR", ROOT / "data" / "official")
).expanduser()
REPO = "OneScience-Group/cfd_benchmark"

PRIORITY = [
    "data/elasticity/Meshes/Random_UnitCell_sigma_10.npy",
    "data/elasticity/Meshes/Random_UnitCell_XY_10.npy",
    "data/ns/NavierStokes_V1e-5_N1200_T20.mat",
    "data/darcy/piececonst_r421_N1024_smooth1.mat",
    "data/darcy/piececonst_r421_N1024_smooth2.mat",
    "data/plas/plas_N987_T20.mat",
]


def layout() -> None:
    OFFICIAL.mkdir(parents=True, exist_ok=True)
    elas_src = LOCAL / "data" / "elasticity"
    elas_dst = OFFICIAL / "elasticity"
    if elas_src.exists():
        if elas_dst.is_symlink():
            elas_dst.unlink()
        if not elas_dst.exists():
            elas_dst.symlink_to(elas_src)

    ns_dir = OFFICIAL / "NavierStokes_V1e-5_N1200_T20"
    ns_dir.mkdir(parents=True, exist_ok=True)
    ns_src = LOCAL / "data" / "ns" / "NavierStokes_V1e-5_N1200_T20.mat"
    ns_dst = ns_dir / "NavierStokes_V1e-5_N1200_T20.mat"
    if ns_src.exists() and not ns_dst.exists():
        ns_dst.symlink_to(ns_src)

    for name in (
        "piececonst_r421_N1024_smooth1.mat",
        "piececonst_r421_N1024_smooth2.mat",
    ):
        src = LOCAL / "data" / "darcy" / name
        dst = OFFICIAL / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    plas_src = LOCAL / "data" / "plas" / "plas_N987_T20.mat"
    plas_dst = OFFICIAL / "plas_N987_T20.mat"
    if plas_src.exists() and not plas_dst.exists():
        plas_dst.symlink_to(plas_src)
    print("layout ready", OFFICIAL, flush=True)


def main() -> int:
    LOCAL.mkdir(parents=True, exist_ok=True)
    wanted = PRIORITY
    if len(sys.argv) > 1:
        wanted = [p for p in PRIORITY if any(tok in p for tok in sys.argv[1:])]
    for rel in wanted:
        dest = LOCAL / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"exists {rel} ({dest.stat().st_size} bytes)", flush=True)
            continue
        print(f"downloading {rel}", flush=True)
        path = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=rel,
            local_dir=str(LOCAL),
        )
        print(f"got {path}", flush=True)
        layout()
    layout()
    print("download done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
