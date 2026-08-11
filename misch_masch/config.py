"""Configuration objects for MISCH-MASCH.

Everything that changes behaviour lives here so experiments are reproducible
from a single JSON file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    # ---- layout of one simulation array, shape (1 + n_tas + n_pr, T) ----
    gmt_row: int = 0
    n_tas: int = 58
    n_pr: int = 58
    #: calendar month of column 0 of every simulation (0 = January)
    start_month: int = 0

    # ---- cropping ----
    window: int = 240  # months per generated piece
    #: if True, crops start at X with X % 12 == 0 (i.e. always January).
    january_start: bool = False

    # ---- preprocessing ----
    #: "none" | "signed_cbrt".  Precipitation anomalies are heavy-tailed and
    #: ~1e-5 in magnitude; the signed cube root is monotone, exactly
    #: invertible, and tames the tails before standardisation.
    pr_transform: str = "signed_cbrt"

    # ---- context-conditioned outpainting (training-time masking) ----
    #: possible lengths (in months) of the clean prefix handed to the model.
    #: MUST include the inference overlap (window - stride).  The default
    #: covers every multiple of 12 up to window - 12, which also keeps the
    #: final (snapped) window of a scenario in-distribution.
    context_lengths: Tuple[int, ...] = (12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228)
    #: probability of a fully unconditional window (needed for the very first
    #: window of a scenario, which has no history to condition on).
    p_no_context: float = 0.20

    # ---- splitting ----
    val_fraction: float = 0.15
    seed: int = 0

    @property
    def n_channels(self) -> int:
        return self.n_tas + self.n_pr

    @property
    def n_rows(self) -> int:
        return 1 + self.n_channels


@dataclass
class ModelConfig:
    # ---- denoiser (1-D DiT over month tokens) ----
    d_model: int = 256
    depth: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_window: int = 256  # capacity of the learned positional table

    # ---- GMT history encoder ----
    gmt_d_model: int = 128
    gmt_depth: int = 4
    gmt_heads: int = 4
    gmt_max_years: int = 1024
    #: append elapsed-years as an explicit feature.  Helps with historical
    #: forcings that GMT alone does not explain (volcanoes, aerosols) but
    #: makes the model slightly more "calendar aware" and less purely
    #: path-driven.  Set False if you want conditioning on GMT path only.
    use_elapsed_time_feature: bool = True

    cond_dim: int = 512
    #: hook for later multi-ESM training; leave at 1 for a single model.
    n_esm: int = 1


@dataclass
class DiffusionConfig:
    n_train_steps: int = 1000
    schedule: str = "cosine"
    parameterization: str = "v"  # v-prediction is the stable choice here
    sample_steps: int = 100
    #: 1.0 = ancestral (DDPM-like) sampling, 0.0 = deterministic DDIM.
    #: Keep at 1.0: ensemble spread is the product here, not sample "quality".
    eta: float = 1.0
    #: clamp on the predicted x0 in normalised units.  None = no clamping,
    #: which preserves tails (important for extremes).  Set e.g. 10.0 only if
    #: sampling diverges.
    x0_clip: float | None = None


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.99)
    ema_decay: float = 0.999
    max_steps: int = 200_000
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    log_every: int = 100
    val_every: int = 2_000
    ckpt_every: int = 5_000
    num_workers: int = 4
    device: str = "cuda"
    amp: bool = True
    out_dir: str = "runs/mischmasch"
    #: NOTE: classifier-free guidance is deliberately NOT implemented.
    #: CFG shrinks sample diversity, which for an ensemble emulator destroys
    #: exactly the quantity you are trying to reproduce.


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            data=DataConfig(**{**asdict(DataConfig()), **d.get("data", {})}),
            model=ModelConfig(**{**asdict(ModelConfig()), **d.get("model", {})}),
            diffusion=DiffusionConfig(
                **{**asdict(DiffusionConfig()), **d.get("diffusion", {})}
            ),
            train=TrainConfig(**{**asdict(TrainConfig()), **d.get("train", {})}),
        )

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))
