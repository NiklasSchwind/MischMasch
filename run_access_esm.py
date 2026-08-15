#!/usr/bin/env python3
"""Train MISCH-MASCH on several CMIP6 ESMs and emulate ssp245 for each.

(The filename is a leftover from the single-model version -- rename it freely,
just update the sbatch line.)

    python run_access_esm.py                      # train + emulate all models
    python run_access_esm.py --skip-train         # reuse existing checkpoint
    python run_access_esm.py --models CanESM5,MIROC6

Models are distinguished by a learned ESM embedding (``model.n_esm``), which is
zero-initialised, so everything else about the pipeline is unchanged. Training
crops are sampled so every model is seen equally often, the train/val split
holds out scenarios *within* each model, and validation loss is reported per
model as well as as a balanced mean.

`misch_masch/config.py` is the single source of truth. Every flag below
defaults to ``None`` and only overrides the corresponding config field when
you pass it explicitly.

Outputs
-------
    <out-root>/models/<run-name>/best.pt         lowest-validation-loss
                                                 checkpoint; inference uses it
    <out-root>/models/<run-name>/last.pt
    <out-root>/models/<run-name>/config.json
    <out-root>/models/<run-name>/esm_ids.json    {model name: embedding index}
    <out-root>/test_data/ssp245_<MODEL>_emulated.npy      (1+n_tas+n_pr, T)
    <out-root>/test_data/ssp245_<MODEL>_emulated_tas.npy  (n_tas, T)
    <out-root>/test_data/ssp245_<MODEL>_emulated_pr.npy   (n_pr, T)
    <out-root>/test_data/ssp245_<MODEL>_reference.npy     ESM members, as-is
    <out-root>/test_data/metadata.json                    provenance for all

Load any of them back with::

    sims = list(np.load(path, allow_pickle=True))
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from emuvaluate.data_preparation import load_scenarios

from misch_masch import Config, ScenarioSampler, train_from_sims
from misch_masch.config import DataConfig, DiffusionConfig, ModelConfig, TrainConfig

# --------------------------------------------------------------------------

MODELS = [
    "CanESM5",
    "ACCESS-ESM1-5",
    "MPI-ESM1-2-LR",
    "MIROC6",
    "IPSL-CM6A-LR",
]

MODEL_ROOT = "/projects/icigroup/CMIP6/cmip6-ng-inc-oceans"

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


def load_one(model: str, scenario: str, model_root: str, min_months: int,
             max_members: int | None) -> List[np.ndarray]:
    """Load one (model, scenario). Returns [] and warns if it is unavailable."""
    t0 = time.time()
    try:
        raw = load_scenarios(
            model=model,
            indicators=["tas", "pr"],
            scenarios=[scenario],
            model_path=os.path.join(model_root, model),
            pattern_scaling_residuals=False,
            ramp_down_corrected_ps=False,
            monthly_flag=True,
            use_smoothing=False,
            train_pattern_scaling_name=None,
        )
    except Exception as e:                    # noqa: BLE001
        print(f"[warn] {model}/{scenario}: skipping ({type(e).__name__}: {e})",
              flush=True)
        return []

    arrs = [_trim_to_whole_years(a) for a in _as_array_list(raw)]
    short = sum(1 for a in arrs if a.shape[1] < min_months)
    arrs = [a for a in arrs if a.shape[1] >= min_months]
    if short:
        print(f"[warn] {model}/{scenario}: dropped {short} member(s) shorter "
              f"than the {min_months}-month window", flush=True)
    if max_members is not None and len(arrs) > max_members:
        print(f"[note] {model}/{scenario}: keeping {max_members} of "
              f"{len(arrs)} members (--max-members-per-scenario)", flush=True)
        arrs = arrs[:max_members]
    if not arrs:
        print(f"[warn] {model}/{scenario}: no usable simulations", flush=True)
        return []
    print(f"[load] {model:<16s} {scenario:<32s} {len(arrs):>3d} members, "
          f"T = {sorted({a.shape[1] for a in arrs})}  ({time.time()-t0:.1f}s)",
          flush=True)
    return arrs


def load_multi(models: Sequence[str], scenarios: Sequence[str], model_root: str,
               esm_index: Dict[str, int], min_months: int,
               max_members: int | None):
    """Load every (model, scenario) pair.

    Returns ``(sims, groups, esm_ids, strata)`` where
      * ``groups`` = ``"<model>/<scenario>"`` -- the unit the train/val split
        moves as a block, so ensemble members and branched scenarios never
        straddle it;
      * ``strata`` = the model, so ``val_fraction`` of scenarios is held out
        *within each* model rather than by a global draw that could leave one
        model with no validation data.
    """
    sims: List[np.ndarray] = []
    groups: List[str] = []
    esm_ids: List[int] = []
    strata: List[str] = []
    for model in dict.fromkeys(models):
        n_before = len(sims)
        for sc in dict.fromkeys(scenarios):
            arrs = load_one(model, sc, model_root, min_months, max_members)
            sims.extend(arrs)
            groups.extend([f"{model}/{sc}"] * len(arrs))
            esm_ids.extend([esm_index[model]] * len(arrs))
            strata.extend([model] * len(arrs))
        got = len(sims) - n_before
        if got == 0:
            raise RuntimeError(
                f"'{model}' yielded no simulations at all -- check the path "
                f"{os.path.join(model_root, model)} and the scenario names"
            )
        print(f"[load] {model:<16s} TOTAL {got} simulations", flush=True)
    if not sims:
        raise RuntimeError("no simulations loaded")

    nbytes = sum(a.nbytes for a in sims)
    print(f"[load] {len(sims)} simulations, {nbytes/2**30:.2f} GiB raw "
          f"(the dataset keeps a normalised float32 copy too)", flush=True)
    if nbytes / 2**30 > 12:
        print("[warn] that is a lot of memory; the CropDataset roughly doubles "
              "it and each dataloader worker may add more. Consider "
              "--max-members-per-scenario or --num-workers 2.", flush=True)
    return sims, groups, esm_ids, strata


def verify_layout(sims: Sequence[np.ndarray], cfg: Config, label: str = "") -> None:
    """Fail loudly if the row layout is not what the config assumes.

    Also checks the tas/pr split point by magnitude: tas is O(10), pr is
    O(1e-5), so the boundary is visible in the data.  Getting this wrong is
    silent and fatal, so it is worth the ten lines.
    """
    d = cfg.data
    tag = f" [{label}]" if label else ""
    n_rows = sims[0].shape[0]
    if n_rows != d.n_rows:
        raise ValueError(
            f"expected {d.n_rows} rows (1 GMT + {d.n_tas} tas + {d.n_pr} pr), "
            f"got {n_rows}{tag}. Fix data.n_tas / data.n_pr in "
            f"misch_masch/config.py."
        )
    if not all(a.shape[0] == n_rows for a in sims):
        raise ValueError(f"simulations have inconsistent row counts{tag}")

    scale = np.mean([np.abs(a[1:]).mean(axis=1) for a in sims[:5]], axis=0)
    scale = np.maximum(scale, 1e-30)
    drop = int(np.argmax(np.log10(scale[:-1]) - np.log10(scale[1:]))) + 1
    ratio = scale[: d.n_tas].mean() / scale[d.n_tas :].mean()
    print(f"[check]{tag} mean|tas| / mean|pr| = {ratio:.3g}")
    if drop != d.n_tas:
        print(f"[warn]{tag} the largest magnitude break is after row {drop}, "
              f"but data.n_tas = {d.n_tas}. Verify the tas/pr split in "
              f"config.py before trusting anything downstream.")


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
    setopt(cfg.data, "p_full_gmt_history", args.p_full_gmt_history)
    setopt(cfg.data, "seed", args.seed)
    if window_overridden:
        # an explicit --window invalidates any hand-written context ladder;
        # empty means "derive every multiple of 12 up to window - 12"
        cfg.data.context_lengths = ()

    setopt(cfg.model, "d_model", args.d_model)
    setopt(cfg.model, "depth", args.depth)
    setopt(cfg.model, "n_heads", args.n_heads)
    setopt(cfg.model, "dropout", args.dropout)
    setopt(cfg.model, "n_esm", args.n_esm)

    setopt(cfg.diffusion, "sample_steps", args.sample_steps)
    setopt(cfg.diffusion, "eta", args.eta)

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
    setopt(cfg.train, "balance_esms", args.balance_esms)
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
    p.add_argument("--models", default=None,
                   help=f"comma-separated ESMs (default: {','.join(MODELS)})")
    p.add_argument("--model-root", default=MODEL_ROOT,
                   help=f"parent directory of the per-model archives "
                        f"(default: {MODEL_ROOT})")
    p.add_argument("--out-root", default="/hdrive/all_users/schwind/MischMasch")
    p.add_argument("--run-name", default="cmip6-5models")
    p.add_argument("--max-members-per-scenario", type=int, default=None,
                   help="cap members per (model, scenario) to bound memory; "
                        "what is dropped is logged")

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
    g.add_argument("--n-esm", type=int, default=None,
                   help=f"size of the ESM embedding table (config: {_m.n_esm}); "
                        f"must be >= the number of models")
    g.add_argument("--balance-esms", dest="balance_esms", default=None,
                   action=argparse.BooleanOptionalAction,
                   help=f"equalise how often each ESM is sampled "
                        f"(config: {_t.balance_esms})")
    g.add_argument("--weight-decay", type=float, default=None,
                   help=f"config: {_t.weight_decay:g}")
    g.add_argument("--warmup-steps", type=int, default=None,
                   help=f"config: {_t.warmup_steps}")
    g.add_argument("--val-batches", type=int, default=None,
                   help=f"total validation batches, split across models "
                        f"(config: {_t.val_batches})")
    g.add_argument("--early-stop-patience", type=int, default=None,
                   help=f"consecutive validations with no new best before "
                        f"stopping, 0 disables (config: {_t.early_stop_patience})")
    g.add_argument("--spike-abort-ratio", type=float, default=None,
                   help=f"abort if val exceeds best by this factor, 0 disables "
                        f"(config: {_t.spike_abort_ratio:g})")
    g.add_argument("--p-full-gmt-history", type=float, default=None,
                   help=f"probability the GMT history starts at year 0; "
                        f"otherwise it starts at a random year before the crop "
                        f"(config: {_d.p_full_gmt_history:g})")
    g.add_argument("--val-fraction", type=float, default=None,
                   help=f"fraction of each model's scenarios held out "
                        f"(config: {_d.val_fraction})")
    g.add_argument("--val-every", type=int, default=None, help=f"config: {_t.val_every}")
    g.add_argument("--ckpt-every", type=int, default=None, help=f"config: {_t.ckpt_every}")
    g.add_argument("--num-workers", type=int, default=None,
                   help=f"config: {_t.num_workers}")
    g.add_argument("--device", default=None, help=f"config: {_t.device}")
    g.add_argument("--seed", type=int, default=None, help=f"config: {_d.seed}")
    g.add_argument("--sample-steps", type=int, default=None,
                   help=f"config: {_f.sample_steps}")
    g.add_argument("--eta", type=float, default=None,
                   help=f"sampler stochasticity, 0 = deterministic DDIM "
                        f"(config: {_f.eta:g})")
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

    models = [m.strip() for m in args.models.split(",")] if args.models else list(MODELS)
    models = list(dict.fromkeys(models))
    esm_index = {m: i for i, m in enumerate(models)}

    cfg = build_config(args)
    if cfg.model.n_esm < len(models):
        raise SystemExit(
            f"[fatal] {len(models)} models requested but model.n_esm = "
            f"{cfg.model.n_esm}. Set n_esm in misch_masch/config.py (or pass "
            f"--n-esm {len(models)})."
        )

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
    print(f"  models      : " + ", ".join(f"{m}={i}" for m, i in esm_index.items()))
    if overlap not in cfg.data.context_lengths:
        print(f"[warn] the inference overlap ({overlap}) is not in "
              f"data.context_lengths -- the model will be asked for a context "
              f"length it never saw. Expect seams.")
    print("[note] normalisation statistics are fitted POOLED over all models. "
          "Models differ in climatology, seasonal amplitude and variance, so "
          "the ESM embedding has to absorb part of that. Per-ESM normalisation "
          "is the obvious next improvement if results are uneven across models.")

    with open(os.path.join(model_dir, "esm_ids.json"), "w") as f:
        json.dump(esm_index, f, indent=2)

    # ---------------------------------------------------------------- train
    if not args.skip_train:
        print("\n=== loading training data " + "=" * 36)
        sims, groups, esm_ids, strata = load_multi(
            models, TRAIN_SCENARIOS, args.model_root, esm_index,
            cfg.data.window, args.max_members_per_scenario)
        verify_layout(sims, cfg, "train")
        n_months = sum(a.shape[1] for a in sims)
        n_eff = n_months // cfg.data.window
        print(f"[load] {n_months} months total, ~{n_eff} effective independent "
              f"windows across {len(models)} models")
        if n_eff < 5_000:
            print("[note] that is a small effective sample count -- watch the "
                  "per-model validation losses.")

        print("\n=== training " + "=" * 49)
        out = train_from_sims(
            sims, cfg, groups=groups, esm_ids=esm_ids, strata=strata,
            esm_names={i: m for m, i in esm_index.items()},
        )
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
    if sampler.cfg.model.n_esm < len(models):
        raise SystemExit(
            f"[fatal] checkpoint was trained with n_esm="
            f"{sampler.cfg.model.n_esm} but {len(models)} models requested"
        )

    all_meta: List[dict] = []
    tag = "_".join(TEST_SCENARIOS)
    for model in models:
        print(f"\n=== emulating {model} " + "=" * max(4, 40 - len(model)))
        test_sims = []
        for sc in TEST_SCENARIOS:
            test_sims.extend(load_one(model, sc, args.model_root,
                                      cfg.data.window,
                                      args.max_members_per_scenario))
        if not test_sims:
            print(f"[warn] {model}: no test simulations, skipping")
            continue
        verify_layout(test_sims, cfg, model)

        emulated: List[np.ndarray] = []
        t0 = time.time()
        for i, sim in enumerate(test_sims):
            gmt = np.asarray(sim[0], dtype=np.float64)
            T = gmt.size
            print(f"[emulate] {model} source member {i + 1}/{len(test_sims)} "
                  f"({T} months, {T // 12} years) x {args.members_per_gmt}",
                  flush=True)
            seed = cfg.data.seed * 100_000 + esm_index[model] * 1_000 + i
            ens = sampler.sample(
                gmt, n_members=args.members_per_gmt, stride=stride,
                steps=cfg.diffusion.sample_steps, esm_id=esm_index[model],
                seed=seed,
            )
            for m in range(ens.shape[0]):
                arr = np.empty((1 + n_tas + n_pr, T), dtype=np.float32)
                arr[0] = gmt          # keep the GMT row: same layout in and out
                arr[1:] = ens[m]
                emulated.append(arr)
                all_meta.append(dict(
                    model=model, esm_id=esm_index[model],
                    file=f"{tag}_{model}_emulated.npy",
                    index=len(emulated) - 1,
                    scenario=TEST_SCENARIOS[0],
                    source_member=i, emulated_member=m,
                    n_months=int(T), seed=seed,
                ))
        print(f"[emulate] {model}: {len(emulated)} arrays in {time.time()-t0:.1f}s")

        base = os.path.join(test_dir, f"{tag}_{model}")
        save_object_list(f"{base}_emulated.npy", emulated)
        save_object_list(f"{base}_emulated_tas.npy",
                         [a[1 : 1 + n_tas] for a in emulated])
        save_object_list(f"{base}_emulated_pr.npy",
                         [a[1 + n_tas :] for a in emulated])
        save_object_list(f"{base}_reference.npy", test_sims)
        del test_sims, emulated

    with open(os.path.join(test_dir, "metadata.json"), "w") as f:
        json.dump(dict(
            models=models,
            esm_ids=esm_index,
            model_root=args.model_root,
            train_scenarios=list(dict.fromkeys(TRAIN_SCENARIOS)),
            test_scenarios=TEST_SCENARIOS,
            checkpoint=ckpt,
            used_ema=not args.no_ema,
            sample_steps=cfg.diffusion.sample_steps,
            eta=cfg.diffusion.eta,
            stride=stride,
            overlap=overlap,
            members_per_gmt=args.members_per_gmt,
            max_members_per_scenario=args.max_members_per_scenario,
            row_layout=dict(gmt=[0, 1], tas=[1, 1 + n_tas],
                            pr=[1 + n_tas, 1 + n_tas + n_pr]),
            units="physical, identical to the load_scenarios output",
            normalisation="pooled over all models (not per-ESM)",
            arrays=all_meta,
            config=sampler.cfg.to_dict(),
        ), f, indent=2)

    print(f"\n[done] checkpoint  -> {ckpt}")
    print(f"[done] esm id map  -> {os.path.join(model_dir, 'esm_ids.json')}")
    print(f"[done] test data   -> {test_dir}")
    for model in models:
        print(f"[done]   {tag}_{model}_emulated.npy")


if __name__ == "__main__":
    main()
