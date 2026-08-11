"""Training loop for MISCH-MASCH."""

from __future__ import annotations

import math
import os
import time
from typing import Hashable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import (CropDataset, Normalizer, check_data, collate, group_split,
                   n_gmt_features)
from .diffusion import Diffusion
from .model import build_model


class EMA:
    """Exponential moving average of parameters -- essential for diffusion."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        return self.shadow


def lr_at(step: int, cfg: Config) -> float:
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / max(t.warmup_steps, 1)
    prog = (step - t.warmup_steps) / max(t.max_steps - t.warmup_steps, 1)
    return t.lr * (0.5 * (1 + math.cos(math.pi * min(prog, 1.0))) * 0.95 + 0.05)


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def save_checkpoint(path, model, ema, normalizer, cfg, step, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        dict(
            step=step,
            model=model.state_dict(),
            ema=ema.state_dict() if ema is not None else None,
            normalizer=normalizer.state_dict(),
            config=cfg.to_dict(),
            extra=extra or {},
        ),
        path,
    )


@torch.no_grad()
def validate(model, diffusion, loader, device, max_batches: int = 20) -> float:
    model.eval()
    tot, n = 0.0, 0
    g = torch.Generator(device=device).manual_seed(0)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        tot += float(diffusion.loss(model, batch, generator=g))
        n += 1
    model.train()
    return tot / max(n, 1)


def train_from_sims(
    sims: Sequence[np.ndarray],
    cfg: Optional[Config] = None,
    groups: Optional[Sequence[Hashable]] = None,
    esm_ids: Optional[Sequence[int]] = None,
    verbose: bool = True,
):
    """Train the diffusion model from a list of ``(117, T)`` arrays.

    Parameters
    ----------
    sims    : list of simulations, see :mod:`misch_masch.data`.
    cfg     : :class:`Config`; defaults are sensible for ~1e4-1e5 crops.
    groups  : one label per simulation used for the train/val split.  Use the
              scenario (or the parent run for branched scenarios) so that
              ensemble members of the same scenario never straddle the split.
    esm_ids : one integer per simulation if you train on several ESMs; also set
              ``cfg.model.n_esm``.
    """
    cfg = (cfg or Config()).finalize()
    device = torch.device(cfg.train.device if torch.cuda.is_available()
                          or cfg.train.device == "cpu" else "cpu")

    check_data(sims, cfg.data, verbose=verbose)
    normalizer = Normalizer.fit(sims, cfg.data)

    tr_idx, va_idx = group_split(len(sims), groups, cfg.data.val_fraction, cfg.data.seed)
    if verbose:
        print(f"[split] {len(tr_idx)} train sims / {len(va_idx)} val sims")

    ds_tr = CropDataset(sims, normalizer, cfg, tr_idx, esm_ids, train=True)
    ds_va = (CropDataset(sims, normalizer, cfg, va_idx, esm_ids, train=False)
             if va_idx else None)
    if verbose:
        n_eff = sum(np.asarray(sims[i]).shape[1] for i in tr_idx) // cfg.data.window
        print(f"[data] {len(ds_tr)} train crops"
              + (f" / {len(ds_va)} val crops" if ds_va else ""))
        print(f"[data] crops overlap heavily; effective independent samples "
              f"~= sum(T)/window = {n_eff}. Keep the model small if this is "
              f"below ~1e4.")

    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.train.num_workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.train.num_workers > 0,
    )
    dl_va = (DataLoader(ds_va, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate) if ds_va else None)

    model = build_model(cfg, n_gmt_features(cfg)).to(device)
    if verbose:
        print(f"[model] {model.n_params()/1e6:.2f} M parameters")
    diffusion = Diffusion(cfg.diffusion.n_train_steps, cfg.diffusion.schedule).to(device)
    ema = EMA(model, cfg.train.ema_decay)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr,
        betas=tuple(cfg.train.betas), weight_decay=cfg.train.weight_decay,
    )
    use_amp = cfg.train.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16

    os.makedirs(cfg.train.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.train.out_dir, "config.json"))

    step, t0, running = 0, time.time(), []
    model.train()
    while step < cfg.train.max_steps:
        for batch in dl_tr:
            if step >= cfg.train.max_steps:
                break
            batch = _to_device(batch, device)
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step, cfg)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = diffusion.loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            ema.update(model)

            running.append(loss.detach().item())
            step += 1

            if verbose and step % cfg.train.log_every == 0:
                dt = time.time() - t0
                print(f"step {step:>7d}  loss {np.mean(running):.4f}  "
                      f"lr {lr_at(step, cfg):.2e}  "
                      f"{cfg.train.log_every/dt:.1f} it/s", flush=True)
                running, t0 = [], time.time()

            if dl_va is not None and step % cfg.train.val_every == 0:
                vl = validate(model, diffusion, dl_va, device)
                if verbose:
                    print(f"step {step:>7d}  VAL loss {vl:.4f}", flush=True)
                t0 = time.time()

            if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
                save_checkpoint(os.path.join(cfg.train.out_dir, "last.pt"),
                                model, ema, normalizer, cfg, step)

    path = os.path.join(cfg.train.out_dir, "last.pt")
    save_checkpoint(path, model, ema, normalizer, cfg, step)
    if verbose:
        print(f"[done] checkpoint -> {path}")
    return dict(path=path, model=model, ema=ema, normalizer=normalizer, config=cfg)
