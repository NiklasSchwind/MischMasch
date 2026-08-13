#!/usr/bin/env python3
"""Simple inference: sample from a trained MISCH-MASCH checkpoint given a GMT timeseries.

    # minimal
    python inference.py --checkpoint /path/to/best.pt --gmt gmt.npy

    # more control
    python inference.py \\
        --checkpoint /path/to/best.pt \\
        --gmt gmt.npy \\
        --members 10 \\
        --stride 48 \\
        --sample-steps 50 \\
        --out emulated.npy \\
        --device cuda

Input
-----
    --gmt   1-D numpy array (monthly, must be a multiple of 12) saved as .npy,
            or a plain text file with one value per line.
            Units must match the training data (K above pre-industrial baseline).

Output
------
    <out>.npy          object array of M arrays, each (1 + n_tas + n_pr, T)
                       row 0: GMT (echoed from input)
                       rows 1..n_tas: tas patterns (physical units)
                       rows n_tas+1..: pr patterns (physical units)

Load back with::

    sims = list(np.load("emulated.npy", allow_pickle=True))
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

from misch_masch import ScenarioSampler


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_gmt(path: str) -> np.ndarray:
    """Load a monthly GMT array from a .npy file or a plain-text file."""
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size % 12 != 0:
        raise ValueError(
            f"GMT length {arr.size} is not a multiple of 12 months. "
            f"Trim or pad to a whole number of years."
        )
    return arr


def save_object_list(path: str, arrays) -> None:
    obj = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        obj[i] = np.asarray(a)
    np.save(path, obj, allow_pickle=True)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------


def run_inference(
    checkpoint: str,
    gmt_monthly: np.ndarray,
    n_members: int = 5,
    stride: Optional[int] = None,
    sample_steps: Optional[int] = None,
    device: str = "cuda",
    use_ema: bool = True,
    seed: Optional[int] = None,
    area_weights: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Sample from a trained checkpoint conditioned on a monthly GMT trajectory.

    Returns a list of M arrays, each shaped (1 + n_tas + n_pr, T):
      row 0    : GMT (echoed from ``gmt_monthly``)
      rows 1.. : emulated climate fields in physical units
    """
    sampler = ScenarioSampler.from_checkpoint(
        checkpoint, device=device, use_ema=use_ema
    )
    cfg = sampler.cfg
    stride = stride if stride is not None else cfg.data.window // 2
    sample_steps = sample_steps if sample_steps is not None else cfg.diffusion.sample_steps

    T = gmt_monthly.size
    n_tas = cfg.data.n_tas
    n_pr = cfg.data.n_pr
    print(
        f"[inference] checkpoint : {checkpoint}\n"
        f"[inference] GMT length : {T} months ({T // 12} years)\n"
        f"[inference] members    : {n_members}\n"
        f"[inference] stride     : {stride}  sample_steps={sample_steps}\n"
        f"[inference] layout     : 1 GMT + {n_tas} tas + {n_pr} pr",
        flush=True,
    )

    ens = sampler.sample(
        gmt_monthly,
        n_members=n_members,
        stride=stride,
        steps=sample_steps,
        seed=seed,
        area_weights=area_weights,
        progress=True,
    )
    # ens: (M, n_channels, T)

    results = []
    for m in range(ens.shape[0]):
        arr = np.empty((1 + n_tas + n_pr, T), dtype=np.float32)
        arr[0] = gmt_monthly
        arr[1:] = ens[m]
        results.append(arr)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="path to a .pt checkpoint (best.pt or last.pt)")
    p.add_argument("--gmt", required=True,
                   help="monthly GMT timeseries: .npy array or plain-text file "
                        "(one value per line), length must be a multiple of 12")
    p.add_argument("--out", default="emulated.npy",
                   help="output path for the object array (default: emulated.npy)")
    p.add_argument("--members", type=int, default=5,
                   help="number of ensemble members to generate (default: 5)")
    p.add_argument("--stride", type=int, default=None,
                   help="window stride in months (default: window // 2)")
    p.add_argument("--sample-steps", type=int, default=None,
                   help="DDIM steps (default: value stored in checkpoint)")
    p.add_argument("--device", default="cuda",
                   help="torch device (default: cuda, falls back to cpu if "
                        "no GPU is available)")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw weights instead of EMA")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducibility")
    p.add_argument("--area-weights", default=None,
                   help="optional .npy file with per-region area weights (n_tas,) "
                        "to hard-constrain the area-weighted mean of tas to the GMT")
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"[fatal] checkpoint not found: {args.checkpoint}")

    gmt = load_gmt(args.gmt)
    area_weights = None
    if args.area_weights is not None:
        area_weights = np.load(args.area_weights).astype(np.float32)

    results = run_inference(
        checkpoint=args.checkpoint,
        gmt_monthly=gmt,
        n_members=args.members,
        stride=args.stride,
        sample_steps=args.sample_steps,
        device=args.device,
        use_ema=not args.no_ema,
        seed=args.seed,
        area_weights=area_weights,
    )

    out_path = args.out if args.out.endswith(".npy") else args.out + ".npy"
    save_object_list(out_path, results)

    n_tas = results[0].shape[0] - 1  # rough; sampler.cfg.data.n_tas is the truth
    print(f"[done] {len(results)} members -> {out_path}")
    print(f"[done] load with:  sims = list(np.load('{out_path}', allow_pickle=True))")


if __name__ == "__main__":
    main()
