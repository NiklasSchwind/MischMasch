#!/usr/bin/env python3
"""Train MISCH-MASCH on ACCESS-ESM1-5 and emulate ssp245.

    python run_access_esm.py                      # train + emulate
    python run_access_esm.py --skip-train         # reuse existing checkpoint
    python run_access_esm.py --max-steps 50000    # shorter run

`misch_masch/config.py` is the single source of truth. Every flag below
defaults to ``None`` and only overrides the corresponding config field when
you pass it explicitly, so editing config.py is enough -- nothing here
silently overwrites your settings. The resolved config is printed at startup
and saved next to the checkpoint.

Outputs
-------
    <out-root>/models/<run-name>/last.pt        checkpoint (model + EMA +
                                                normaliser + config: this file
                                                alone is enough to reload)
    <out-root>/models/<run-name>/config.json
    <out-root>/test_data/ssp245_emulated.npy    object array of (1+n_tas+n_pr, T)
                                                arrays in the SAME layout
                                                emuvaluate hands out, physical
                                                units
    <out-root>/test_data/ssp245_emulated_tas.npy   (n_tas, T) blocks
    <out-root>/test_data/ssp245_emulated_pr.npy    (n_pr, T) blocks
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
import torch

from emuvaluate.data_preparation import load_scenarios

from misch_masch import Config, ScenarioSampler, train_from_sims
from misch_masch.config import DataConfig, DiffusionConfig, ModelConfig, TrainConfig

# --------------------------------------------------------------------------

MODEL = "ACCESS-ESM1-5"

TRAIN_SCENARIOS = [
    "flat10-from-025",
    "flat10-zec-from-025",
    "flat10-cdr-from-025",
    "ssp245",
    "esm-1pct-brch-1000pgc-from-025",
    "esm-1pct-brch-750PgC",
    "esm-1pct-brch-2000PgC",
    "ssp534-over",
    "ssp585",
    "ssp370",
    "ssp460",
    "1pctco2",
]
TEST_SCENARIOS = ["ssp126"]

# NOTE: the row layout (n_tas / n_pr) is NOT defined here -- it comes from
# misch_masch/config.py, so there is exactly one place to change it.


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


def load(scenarios: Sequence[str], model: str, model_path: str, min_months: int
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
            print(f"[warn] skipping '{sc}': {type(e).__name__}: {e}", flush=True)
            continue
        arrs = [_trim_to_whole_years(a) for a in _as_array_list(raw)]
        dropped = sum(1 for a in arrs if a.shape[1] < min_months)
        arrs = [a for a in arrs if a.shape[1] >= min_months]
        if dropped:
            print(f"[warn] '{sc}': dropped {dropped} member(s) shorter than "
                  f"the {min_months}-month window", flush=True)
        if not arrs:
            print(f"[warn] '{sc}' returned no usable simulations", flush=True)
            continue
        sims.extend(arrs)
        groups.extend([sc] * len(arrs))
        print(f"[load] {sc:<32s} {len(arrs):>3d} members, "
              f"T = {sorted({a.shape[1] for a in arrs})}  ({time.time()-t0:.1f}s)",
              flush=True)
    if not sims:
        raise RuntimeError("no simulations loaded")
    return sims, groups


def verify_layout(sims: Sequence[np.ndarray], cfg: Config) -> None:
    """Fail loudly if the row layout is not what the config assumes.

    Also checks the tas/pr split point by magnitude: tas is O(10), pr is
    O(1e-5), so the boundary is visible in the data.  Getting this wrong is
    silent and fatal, so it is worth the ten lines.
    """
    d = cfg.data
    n_rows = sims[0].shape[0]
    if n_rows != d.n_rows:
        raise ValueError(
            f"expected {d.n_rows} rows (1 GMT + {d.n_tas} tas + {d.n_pr} pr), "
            f"got {n_rows}. Fix data.n_tas / data.n_pr in misch_masch/config.py."
        )
    if not all(a.shape[0] == n_rows for a in sims):
        raise ValueError("simulations have inconsistent row counts")

    scale = np.mean([np.abs(a[1:]).mean(axis=1) for a in sims[:5]], axis=0)
    scale = np.maximum(scale, 1e-30)
    drop = int(np.argmax(np.log10(scale[:-1]) - np.log10(scale[1:]))) + 1
    ratio = scale[: d.n_tas].mean() / scale[d.n_tas :].mean()
    print(f"[check] mean|tas| / mean|pr| = {ratio:.3g}")
    if drop != d.n_tas:
        print(f"[warn] the largest magnitude break is after row {drop}, but "
              f"data.n_tas = {d.n_tas}. Verify the tas/pr split in config.py "
              f"before trusting anything downstream.")


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------


def save_object_list(path: str, arrays: Sequence[np.ndarray]) -> None:
    obj = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        obj[i] = np.asarray(a)
    np.save(path, obj, allow_pickle=True)


# --------------------------------------------------------------------------
# config resolution
# --------------------------------------------------------------------------


def build_config(args) -> Config:
    """Start from config.py and apply ONLY the flags that were passed."""
    cfg = Config()

    def setopt(obj, name, value):
        if value is not None:
            setattr(obj, name, value)

    window_overridden = args.window is not None

    setopt(cfg.data, "window", args.window)
    setopt(cfg.data, "january_start", args.january_start)
    setopt(cfg.data, "val_fraction", args.val_fraction)
    setopt(cfg.data, "seed", args.seed)
    if window_overridden:
        # an explicit --window invalidates any hand-written context ladder;
        # empty means "derive every multiple of 12 up to window - 12"
        cfg.data.context_lengths = ()

    setopt(cfg.model, "d_model", args.d_model)
    setopt(cfg.model, "depth", args.depth)
    setopt(cfg.model, "n_heads", args.n_heads)
    setopt(cfg.model, "dropout", args.dropout)

    setopt(cfg.diffusion, "sample_steps", args.sample_steps)

    setopt(cfg.train, "device", args.device)
    setopt(cfg.train, "batch_size", args.batch_size)
    setopt(cfg.train, "max_steps", args.max_steps)
    setopt(cfg.train, "lr", args.lr)
    setopt(cfg.train, "num_workers", args.num_workers)
    setopt(cfg.train, "val_every", args.val_every)
    setopt(cfg.train, "ckpt_every", args.ckpt_every)
    setopt(cfg.train, "weight_decay", args.weight_decay)
    setopt(cfg.train, "warmup_steps", args.warmup_steps)
    setopt(cfg.train, "val_batches", args.val_batches)
    setopt(cfg.train, "early_stop_patience", args.early_stop_patience)
    setopt(cfg.train, "spike_abort_ratio", args.spike_abort_ratio)
    if args.no_amp:
        cfg.train.amp = False
    cfg.train.out_dir = os.path.join(args.out_root, "models", args.run_name)

    return cfg.finalize()


def check_device(cfg: Config, allow_cpu: bool) -> None:
    """Report the runtime, and refuse to start a 48-hour CPU run by accident."""
    avail = torch.cuda.is_available()
    print(f"[env] torch {torch.__version__}  cuda_available={avail}")
    if avail:
        print(f"[env] device: {torch.cuda.get_device_name(0)}  "
              f"bf16_supported={torch.cuda.is_bf16_supported()}")
        if cfg.train.amp and not torch.cuda.is_bf16_supported():
            print("[warn] bf16 autocast is enabled but this GPU does not "
                  "support bf16 -- pass --no-amp.")
    if cfg.train.device.startswith("cuda") and not avail:
        msg = ("train.device is 'cuda' but torch cannot see a GPU. Training "
               "would silently fall back to CPU and take ~100x longer.")
        if not allow_cpu:
            raise SystemExit(f"[fatal] {msg} Pass --allow-cpu to override.")
        print(f"[warn] {msg} Continuing because --allow-cpu was given.")


# --------------------------------------------------------------------------


def main() -> None:
    _d, _t, _f, _m = DataConfig(), TrainConfig(), DiffusionConfig(), ModelConfig()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--model-path", default=None,
                   help="default: /projects/icigroup/CMIP6/cmip6-ng-inc-oceans/<model>")
    p.add_argument("--out-root", default="/hdrive/all_users/schwind/MischMasch")
    p.add_argument("--run-name", default="access-esm1-5")

    g = p.add_argument_group(
        "config overrides",
        "all default to the value in misch_masch/config.py and are only "
        "applied when passed explicitly")
    g.add_argument("--max-steps", type=int, default=None,
                   help=f"config: {_t.max_steps}")
    g.add_argument("--batch-size", type=int, default=None,
                   help=f"config: {_t.batch_size}")
    g.add_argument("--lr", type=float, default=None, help=f"config: {_t.lr:g}")
    g.add_argument("--window", type=int, default=None,
                   help=f"config: {_d.window} (an explicit value also "
                        f"regenerates context_lengths)")
    g.add_argument("--january-start", dest="january_start", default=None,
                   action=argparse.BooleanOptionalAction,
                   help=f"config: {_d.january_start}")
    g.add_argument("--d-model", type=int, default=None, help=f"config: {_m.d_model}")
    g.add_argument("--depth", type=int, default=None, help=f"config: {_m.depth}")
    g.add_argument("--n-heads", type=int, default=None, help=f"config: {_m.n_heads}")
    g.add_argument("--dropout", type=float, default=None, help=f"config: {_m.dropout:g}")
    g.add_argument("--weight-decay", type=float, default=None,
                   help=f"config: {_t.weight_decay:g}")
    g.add_argument("--warmup-steps", type=int, default=None,
                   help=f"config: {_t.warmup_steps}")
    g.add_argument("--val-batches", type=int, default=None,
                   help=f"config: {_t.val_batches}")
    g.add_argument("--early-stop-patience", type=int, default=None,
                   help=f"validations without improvement before stopping, "
                        f"0 disables (config: {_t.early_stop_patience})")
    g.add_argument("--spike-abort-ratio", type=float, default=None,
                   help=f"abort if val exceeds best by this factor, 0 disables "
                        f"(config: {_t.spike_abort_ratio:g})")
    g.add_argument("--val-fraction", type=float, default=None,
                   help=f"config: {_d.val_fraction}")
    g.add_argument("--val-every", type=int, default=None, help=f"config: {_t.val_every}")
    g.add_argument("--ckpt-every", type=int, default=None, help=f"config: {_t.ckpt_every}")
    g.add_argument("--num-workers", type=int, default=None,
                   help=f"config: {_t.num_workers}")
    g.add_argument("--device", default=None, help=f"config: {_t.device}")
    g.add_argument("--seed", type=int, default=None, help=f"config: {_d.seed}")
    g.add_argument("--sample-steps", type=int, default=None,
                   help=f"config: {_f.sample_steps}")
    g.add_argument("--no-amp", action="store_true",
                   help="disable bf16 autocast (required on pre-Ampere GPUs)")

    r = p.add_argument_group("run options")
    r.add_argument("--skip-train", action="store_true",
                   help="reuse the existing checkpoint and only emulate")
    r.add_argument("--allow-cpu", action="store_true",
                   help="do not abort when device is cuda but no GPU is visible")
    r.add_argument("--members-per-gmt", type=int, default=5,
                   help="emulated members generated per ssp245 GMT trajectory")
    r.add_argument("--stride", type=int, default=None,
                   help="window stride in months (default: window // 2, "
                        "i.e. 50%% overlap)")
    r.add_argument("--no-ema", action="store_true",
                   help="sample from the raw weights instead of the EMA")
    r.add_argument("--checkpoint", default=None,
                   help="explicit checkpoint to emulate from; default is "
                        "best.pt (lowest validation loss), falling back to last.pt")
    r.add_argument("--use-last", action="store_true",
                   help="emulate from last.pt even when best.pt exists")
    args = p.parse_args()

    cfg = build_config(args)
    model_path = args.model_path or f"/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{args.model}"
    model_dir = cfg.train.out_dir
    test_dir = os.path.join(args.out_root, "test_data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    last_path = os.path.join(model_dir, "last.pt")
    best_path = os.path.join(model_dir, "best.pt")

    def resolve_checkpoint() -> str:
        """Prefer the lowest-validation-loss checkpoint, not the final one."""
        if args.checkpoint:
            return args.checkpoint
        if args.use_last:
            return last_path
        return best_path if os.path.exists(best_path) else last_path

    # stride / overlap
    stride = args.stride if args.stride is not None else cfg.data.window // 2
    overlap = cfg.data.window - stride
    if stride <= 0 or stride > cfg.data.window - 12:
        raise SystemExit(f"[fatal] --stride must be in [12, {cfg.data.window - 12}]")
    if cfg.data.january_start and stride % 12:
        raise SystemExit("[fatal] --stride must be a multiple of 12 when "
                         "data.january_start is True")

    print("\n=== configuration " + "=" * 44)
    check_device(cfg, args.allow_cpu)
    print(cfg.summary())
    print(f"  inference   : stride {stride}, overlap {overlap} months")
    if overlap not in cfg.data.context_lengths:
        print(f"[warn] the inference overlap ({overlap}) is not in "
              f"data.context_lengths -- the model will be asked for a context "
              f"length it never saw. Expect seams.")

    # ---------------------------------------------------------------- train
    if not args.skip_train:
        print("\n=== loading training scenarios " + "=" * 32)
        sims, groups = load(TRAIN_SCENARIOS, args.model, model_path, cfg.data.window)
        verify_layout(sims, cfg)
        n_months = sum(a.shape[1] for a in sims)
        n_eff = n_months // cfg.data.window
        print(f"[load] {len(sims)} simulations, {n_months} months total, "
              f"~{n_eff} effective independent windows")
        if n_eff < 5_000:
            print("[note] that is a small effective sample count -- consider "
                  "--d-model 192 --depth 4, and watch the validation loss.")

        print("\n=== training " + "=" * 49)
        out = train_from_sims(sims, cfg, groups=groups)
        del sims
        if out["best_val"] is not None:
            print(f"[train] best validation loss {out['best_val']:.4f} at step "
                  f"{out['best_step']}")
    else:
        print("\n[skip-train] reusing an existing checkpoint")

    ckpt = resolve_checkpoint()
    if not os.path.exists(ckpt):
        raise SystemExit(f"[fatal] checkpoint {ckpt} does not exist")
    print(f"[checkpoint] emulating from {ckpt}"
          + ("" if ckpt == best_path else "  (NOT the best-validation checkpoint)"))

    # ----------------------------------------------------------- emulate
    print("\n=== loading test scenario " + "=" * 36)
    test_sims, test_groups = load(TEST_SCENARIOS, args.model, model_path,
                                  cfg.data.window)
    verify_layout(test_sims, cfg)

    print("\n=== emulating " + "=" * 48)
    sampler = ScenarioSampler.from_checkpoint(
        ckpt, device=cfg.train.device, use_ema=not args.no_ema
    )
    # the checkpoint's own config governs the row layout of what comes out
    n_tas = sampler.cfg.data.n_tas
    n_pr = sampler.cfg.data.n_pr
    if (n_tas, n_pr) != (cfg.data.n_tas, cfg.data.n_pr):
        print(f"[warn] checkpoint was trained with n_tas={n_tas}, n_pr={n_pr}, "
              f"but config.py now says {cfg.data.n_tas}/{cfg.data.n_pr}. "
              f"Using the checkpoint's layout.")

    emulated: List[np.ndarray] = []
    meta: List[dict] = []
    t0 = time.time()
    for i, sim in enumerate(test_sims):
        gmt = np.asarray(sim[0], dtype=np.float64)
        T = gmt.size
        print(f"[emulate] source member {i + 1}/{len(test_sims)} "
              f"({T} months, {T // 12} years) x {args.members_per_gmt} members",
              flush=True)
        ens = sampler.sample(
            gmt,
            n_members=args.members_per_gmt,
            stride=stride,
            steps=cfg.diffusion.sample_steps,
            seed=cfg.data.seed * 1000 + i,
        )
        for m in range(ens.shape[0]):
            arr = np.empty((1 + n_tas + n_pr, T), dtype=np.float32)
            arr[0] = gmt                       # keep the GMT row: same layout in
            arr[1:] = ens[m]                   # and out, so emuvaluate can read it
            emulated.append(arr)
            meta.append(dict(
                index=len(emulated) - 1,
                scenario=test_groups[i],
                source_member=i,
                emulated_member=m,
                n_months=int(T),
                seed=cfg.data.seed * 1000 + i,
            ))
    print(f"[emulate] {len(emulated)} arrays in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------- save
    tag = "_".join(TEST_SCENARIOS)
    save_object_list(os.path.join(test_dir, f"{tag}_emulated.npy"), emulated)
    save_object_list(os.path.join(test_dir, f"{tag}_emulated_tas.npy"),
                     [a[1 : 1 + n_tas] for a in emulated])
    save_object_list(os.path.join(test_dir, f"{tag}_emulated_pr.npy"),
                     [a[1 + n_tas :] for a in emulated])
    save_object_list(os.path.join(test_dir, f"{tag}_reference.npy"), test_sims)

    with open(os.path.join(test_dir, "metadata.json"), "w") as f:
        json.dump(dict(
            model=args.model,
            model_path=model_path,
            train_scenarios=list(dict.fromkeys(TRAIN_SCENARIOS)),
            test_scenarios=TEST_SCENARIOS,
            checkpoint=ckpt,
            used_ema=not args.no_ema,
            sample_steps=cfg.diffusion.sample_steps,
            stride=stride,
            overlap=overlap,
            members_per_gmt=args.members_per_gmt,
            row_layout=dict(gmt=[0, 1], tas=[1, 1 + n_tas],
                            pr=[1 + n_tas, 1 + n_tas + n_pr]),
            units="physical, identical to the load_scenarios output",
            arrays=meta,
            config=sampler.cfg.to_dict(),
        ), f, indent=2)

    print(f"\n[done] checkpoint  -> {ckpt}")
    print(f"[done] test data   -> {test_dir}")
    print(f"[done] load with:  sims = list(np.load('{test_dir}/{tag}_emulated.npy', "
          f"allow_pickle=True))")


if __name__ == "__main__":
    main()
