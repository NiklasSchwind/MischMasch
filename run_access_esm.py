#!/usr/bin/env python3
"""Train MISCH-MASCH on ACCESS-ESM1-5 and emulate ssp245.

    python run_access_esm.py                      # train + emulate
    python run_access_esm.py --skip-train         # reuse existing checkpoint
    python run_access_esm.py --max-steps 50000    # shorter run

Outputs
-------
    <out-root>/models/<run-name>/last.pt        checkpoint (model + EMA +
                                                normaliser + config: this file
                                                alone is enough to reload)
    <out-root>/models/<run-name>/config.json
    <out-root>/test_data/ssp245_emulated.npy    object array of (117, T) arrays
                                                in the SAME layout emuvaluate
                                                hands out, physical units
    <out-root>/test_data/ssp245_emulated_tas.npy   (57, T) blocks
    <out-root>/test_data/ssp245_emulated_pr.npy    (59, T) blocks
    <out-root>/test_data/ssp245_reference.npy   the ESM ssp245 members, as-is
    <out-root>/test_data/metadata.json          provenance for every array

Load the emulated data back with::

    sims = list(np.load(path, allow_pickle=True))
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Sequence, Tuple

import numpy as np

from emuvaluate.data_preparation import load_scenarios

from misch_masch import Config, ScenarioSampler, train_from_sims

# --------------------------------------------------------------------------

MODEL = "ACCESS-ESM1-5"

TRAIN_SCENARIOS = [
    "flat10-from-025",
    "flat10-zec-from-025",
    "flat10-cdr-from-025",
    "ssp126",
    "esm-1pct-brch-1000pgc-from-025",
    "esm-1pct-brch-750PgC",
    "esm-1pct-brch-2000PgC",
    "ssp534-over",
    "ssp585",
    "ssp370",
    "ssp460",
    "1pctco2",
]
TEST_SCENARIOS = ["ssp245"]

#: row layout of one simulation: 1 GMT + N_TAS tas + N_PR pr
N_TAS, N_PR = 57, 59


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _as_array_list(obj) -> List[np.ndarray]:
    """Coerce whatever load_scenarios returns into a flat list of 2-D arrays."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [np.asarray(a) for a in obj.ravel()]
        if obj.ndim == 3:
            return [obj[i] for i in range(obj.shape[0])]
        if obj.ndim == 2:
            return [obj]
    if isinstance(obj, dict):
        out: List[np.ndarray] = []
        for v in obj.values():
            out.extend(_as_array_list(v))
        return out
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(a, np.ndarray) and a.ndim == 2 for a in obj):
            return [np.asarray(a) for a in obj]
        # e.g. (sims, metadata) or nested per-scenario lists
        out = []
        for el in obj:
            try:
                out.extend(_as_array_list(el))
            except TypeError:
                continue
        if out:
            return out
    raise TypeError(f"cannot interpret load_scenarios output of type {type(obj)}")


def _trim_to_whole_years(a: np.ndarray) -> np.ndarray:
    """MISCH-MASCH assumes column 0 is January and T % 12 == 0."""
    a = np.asarray(a, dtype=np.float32)
    T = a.shape[1]
    return a[:, : T - T % 12] if T % 12 else a


def load(scenarios: Sequence[str], model: str, model_path: str
         ) -> Tuple[List[np.ndarray], List[str]]:
    """Load one scenario at a time, so we know each simulation's group label.

    Group labels are what keep ensemble members of the same scenario -- and
    scenarios that branch off a shared run -- on the same side of the
    train/val split.  Loading everything in one call would lose that.
    """
    sims: List[np.ndarray] = []
    groups: List[str] = []
    for sc in dict.fromkeys(scenarios):          # de-duplicate, keep order
        t0 = time.time()
        try:
            raw = load_scenarios(
                model=model,
                indicators=["tas", "pr"],
                scenarios=[sc],
                model_path=model_path,
                pattern_scaling_residuals=False,
                ramp_down_corrected_ps=False,
                monthly_flag=True,
                use_smoothing=False,
                train_pattern_scaling_name=None,
            )
        except Exception as e:                    # noqa: BLE001
            print(f"[warn] skipping '{sc}': {type(e).__name__}: {e}")
            continue
        arrs = [_trim_to_whole_years(a) for a in _as_array_list(raw)]
        arrs = [a for a in arrs if a.shape[1] >= 96]
        if not arrs:
            print(f"[warn] '{sc}' returned no usable simulations")
            continue
        sims.extend(arrs)
        groups.extend([sc] * len(arrs))
        print(f"[load] {sc:<32s} {len(arrs):>3d} members, "
              f"T = {sorted({a.shape[1] for a in arrs})}  ({time.time()-t0:.1f}s)")
    if not sims:
        raise RuntimeError("no simulations loaded")
    return sims, groups


def verify_layout(sims: Sequence[np.ndarray], n_tas: int, n_pr: int) -> None:
    """Fail loudly if the row layout is not what the model assumes.

    Also checks the tas/pr split point by magnitude: tas is O(10), pr is
    O(1e-5), so the boundary is visible in the data.  Getting this wrong is
    silent and fatal, so it is worth the ten lines.
    """
    n_rows = sims[0].shape[0]
    if n_rows != 1 + n_tas + n_pr:
        raise ValueError(
            f"expected {1 + n_tas + n_pr} rows (1 GMT + {n_tas} tas + {n_pr} pr), "
            f"got {n_rows}. Adjust N_TAS / N_PR at the top of this script."
        )
    scale = np.mean([np.abs(a[1:]).mean(axis=1) for a in sims[:5]], axis=0)
    scale = np.maximum(scale, 1e-30)
    drop = int(np.argmax(np.log10(scale[:-1]) - np.log10(scale[1:]))) + 1
    ratio = scale[:n_tas].mean() / scale[n_tas:].mean()
    print(f"[check] mean|tas| / mean|pr| = {ratio:.3g}")
    if drop != n_tas:
        print(f"[warn] the largest magnitude break is after row {drop}, but "
              f"N_TAS = {n_tas}. Verify the tas/pr split before trusting "
              f"anything downstream.")
    if not all(a.shape[0] == n_rows for a in sims):
        raise ValueError("simulations have inconsistent row counts")


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------


def save_object_list(path: str, arrays: Sequence[np.ndarray]) -> None:
    obj = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        obj[i] = np.asarray(a)
    np.save(path, obj, allow_pickle=True)


# --------------------------------------------------------------------------


def build_config(args) -> Config:
    cfg = Config()
    cfg.data.n_tas, cfg.data.n_pr = N_TAS, N_PR
    cfg.data.window = args.window
    cfg.data.january_start = True
    cfg.data.context_lengths = tuple(range(12, args.window, 12))
    cfg.data.val_fraction = args.val_fraction
    cfg.data.seed = args.seed

    cfg.model.d_model = args.d_model
    cfg.model.depth = args.depth
    cfg.model.n_heads = args.n_heads
    cfg.model.gmt_max_years = 2048

    cfg.diffusion.sample_steps = args.sample_steps

    cfg.train.device = args.device
    cfg.train.batch_size = args.batch_size
    cfg.train.max_steps = args.max_steps
    cfg.train.lr = args.lr
    cfg.train.num_workers = args.num_workers
    cfg.train.amp = not args.no_amp
    cfg.train.val_every = args.val_every
    cfg.train.ckpt_every = args.ckpt_every
    cfg.train.out_dir = os.path.join(args.out_root, "models", args.run_name)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--model-path", default=None,
                   help="default: /projects/icigroup/CMIP6/cmip6-ng-inc-oceans/<model>")
    p.add_argument("--out-root", default="/hdrive/all_users/schwind/MischMasch")
    p.add_argument("--run-name", default="access-esm1-5")

    p.add_argument("--max-steps", type=int, default=150_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--val-every", type=int, default=2_000)
    p.add_argument("--ckpt-every", type=int, default=5_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--skip-train", action="store_true",
                   help="reuse the existing checkpoint and only emulate")
    p.add_argument("--members-per-gmt", type=int, default=5,
                   help="emulated members generated per ssp245 GMT trajectory")
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--stride", type=int, default=60)
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()

    model_path = args.model_path or f"/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{args.model}"
    model_dir = os.path.join(args.out_root, "models", args.run_name)
    test_dir = os.path.join(args.out_root, "test_data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    ckpt = os.path.join(model_dir, "last.pt")

    cfg = build_config(args)

    # ---------------------------------------------------------------- train
    if not args.skip_train:
        print("\n=== loading training scenarios " + "=" * 32)
        sims, groups = load(TRAIN_SCENARIOS, args.model, model_path)
        verify_layout(sims, N_TAS, N_PR)
        n_months = sum(a.shape[1] for a in sims)
        print(f"[load] {len(sims)} simulations, {n_months} months total, "
              f"~{n_months // cfg.data.window} effective independent windows")
        if n_months // cfg.data.window < 5_000:
            print("[note] that is a small effective sample count -- consider "
                  "--d-model 192 --depth 4, and watch the validation loss.")

        print("\n=== training " + "=" * 49)
        train_from_sims(sims, cfg, groups=groups)
        del sims
    else:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"--skip-train given but {ckpt} does not exist")
        print(f"[skip-train] reusing {ckpt}")

    # ----------------------------------------------------------- emulate
    print("\n=== loading test scenario " + "=" * 36)
    test_sims, test_groups = load(TEST_SCENARIOS, args.model, model_path)
    verify_layout(test_sims, N_TAS, N_PR)

    print("\n=== emulating " + "=" * 48)
    sampler = ScenarioSampler.from_checkpoint(
        ckpt, device=args.device, use_ema=not args.no_ema
    )

    emulated: List[np.ndarray] = []
    meta: List[dict] = []
    t0 = time.time()
    for i, sim in enumerate(test_sims):
        gmt = np.asarray(sim[0], dtype=np.float64)
        T = gmt.size
        print(f"[emulate] source member {i + 1}/{len(test_sims)} "
              f"({T} months, {T // 12} years) x {args.members_per_gmt} members")
        ens = sampler.sample(
            gmt,
            n_members=args.members_per_gmt,
            stride=args.stride,
            steps=args.sample_steps,
            seed=args.seed * 1000 + i,
        )
        for m in range(ens.shape[0]):
            arr = np.empty((1 + N_TAS + N_PR, T), dtype=np.float32)
            arr[0] = gmt                       # keep the GMT row: same layout in
            arr[1:] = ens[m]                   # and out, so emuvaluate can read it
            emulated.append(arr)
            meta.append(dict(
                index=len(emulated) - 1,
                scenario=test_groups[i],
                source_member=i,
                emulated_member=m,
                n_months=int(T),
                seed=args.seed * 1000 + i,
            ))
    print(f"[emulate] {len(emulated)} arrays in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------- save
    tag = "_".join(TEST_SCENARIOS)
    save_object_list(os.path.join(test_dir, f"{tag}_emulated.npy"), emulated)
    save_object_list(os.path.join(test_dir, f"{tag}_emulated_tas.npy"),
                     [a[1 : 1 + N_TAS] for a in emulated])
    save_object_list(os.path.join(test_dir, f"{tag}_emulated_pr.npy"),
                     [a[1 + N_TAS :] for a in emulated])
    save_object_list(os.path.join(test_dir, f"{tag}_reference.npy"), test_sims)

    with open(os.path.join(test_dir, "metadata.json"), "w") as f:
        json.dump(dict(
            model=args.model,
            model_path=model_path,
            train_scenarios=list(dict.fromkeys(TRAIN_SCENARIOS)),
            test_scenarios=TEST_SCENARIOS,
            checkpoint=ckpt,
            used_ema=not args.no_ema,
            sample_steps=args.sample_steps,
            stride=args.stride,
            members_per_gmt=args.members_per_gmt,
            row_layout=dict(gmt=[0, 1], tas=[1, 1 + N_TAS],
                            pr=[1 + N_TAS, 1 + N_TAS + N_PR]),
            units="physical, identical to the load_scenarios output",
            arrays=meta,
            config=cfg.to_dict(),
        ), f, indent=2)

    print(f"\n[done] checkpoint  -> {ckpt}")
    print(f"[done] test data   -> {test_dir}")
    print(f"[done] load with:  sims = list(np.load('{test_dir}/{tag}_emulated.npy', "
          f"allow_pickle=True))")


if __name__ == "__main__":
    main()
