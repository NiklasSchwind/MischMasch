"""Configuration objects for MISCH-MASCH.

This file is the single source of truth. Everything that changes behaviour
lives here, and `run_access_esm.py` only overrides a field when you pass the
corresponding flag explicitly -- editing the defaults below is enough.

`Config` validates itself on construction, and again via `finalize()` after
any programmatic mutation, so mismatched settings fail loudly at startup
instead of silently misbehaving twelve hours into a job.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Tuple

PR_TRANSFORMS = ("none", "signed_cbrt")


@dataclass
class DataConfig:
    # ---- layout of one simulation array, shape (1 + n_tas + n_pr, T) ----
    gmt_row: int = 0
    n_tas: int = 58
    n_pr: int = 58
    #: calendar month of column 0 of every simulation (0 = January)
    start_month: int = 0

    # ---- cropping ----
    window: int = 240  # months per generated piece (must be a multiple of 12)
    #: if True, crops start at X with X % 12 == 0 (i.e. always January).
    #: False gives 12x the crops as seasonal-phase augmentation; the model
    #: carries a per-token calendar-month embedding either way.
    january_start: bool = False

    # ---- preprocessing ----
    #: "none" | "signed_cbrt".  Precipitation anomalies are heavy-tailed and
    #: ~1e-5 in magnitude; the signed cube root is monotone, exactly
    #: invertible, and tames the tails before standardisation.
    pr_transform: str = "signed_cbrt"

    # ---- context-conditioned outpainting (training-time masking) ----
    #: possible lengths (in months) of the clean prefix handed to the model.
    #: MUST include the inference overlap (window - stride).
    #: Leave EMPTY to derive every multiple of 12 up to window - 12, which is
    #: what you want unless you are deliberately concentrating training on one
    #: overlap length.  Entries longer than window - 12 are dropped.
    context_lengths: Tuple[int, ...] = ()
    #: probability of a fully unconditional window (needed for the very first
    #: window of a scenario, which has no history to condition on).
    p_no_context: float = 0.20

    # ---- splitting ----
    val_fraction: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> None:
        """Coerce and range-check the fields that can be set inconsistently."""
        if self.window % 12 != 0:
            raise ValueError(f"data.window must be a multiple of 12, got {self.window}")
        if self.window < 24:
            raise ValueError(f"data.window must be >= 24, got {self.window}")
        if self.pr_transform not in PR_TRANSFORMS:
            raise ValueError(
                f"data.pr_transform must be one of {PR_TRANSFORMS}, "
                f"got {self.pr_transform!r}"
            )
        if not 0.0 <= self.p_no_context <= 1.0:
            raise ValueError(f"data.p_no_context must be in [0, 1], got {self.p_no_context}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"data.val_fraction must be in [0, 1), got {self.val_fraction}")

        cl = sorted({int(c) for c in self.context_lengths})
        cl = [c for c in cl if 0 < c <= self.window - 12]
        if not cl:
            cl = list(range(12, self.window, 12))
        self.context_lengths = tuple(cl)

    @property
    def n_channels(self) -> int:
        return self.n_tas + self.n_pr

    @property
    def n_rows(self) -> int:
        return 1 + self.n_channels

    @property
    def max_context(self) -> int:
        return self.context_lengths[-1]


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
    gmt_max_years: int = 2048
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

    def __post_init__(self) -> None:
        self.finalize()

    def finalize(self) -> "Config":
        """Re-validate after mutation.  Call this before using a Config you
        built by assigning to fields (``run_access_esm.py`` does)."""
        self.data.normalize()
        m, d = self.model, self.data
        if d.window > m.max_window:
            raise ValueError(
                f"data.window ({d.window}) exceeds model.max_window "
                f"({m.max_window}); raise max_window (the positional table) "
                f"or shorten the window."
            )
        if m.d_model % m.n_heads:
            raise ValueError(
                f"model.d_model ({m.d_model}) must be divisible by "
                f"model.n_heads ({m.n_heads})"
            )
        if m.gmt_d_model % m.gmt_heads:
            raise ValueError(
                f"model.gmt_d_model ({m.gmt_d_model}) must be divisible by "
                f"model.gmt_heads ({m.gmt_heads})"
            )
        if m.n_esm < 1:
            raise ValueError(f"model.n_esm must be >= 1, got {m.n_esm}")
        if self.diffusion.schedule not in ("cosine", "linear"):
            raise ValueError(f"diffusion.schedule: {self.diffusion.schedule!r}")
        return self

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        d, m, t = self.data, self.model, self.train
        cl = self.data.context_lengths
        return "\n".join([
            f"  rows        : 1 GMT + {d.n_tas} tas + {d.n_pr} pr = {d.n_rows}",
            f"  window      : {d.window} months ({d.window // 12} yr), "
            f"january_start={d.january_start}",
            f"  context     : {cl[0]}..{cl[-1]} step 12 ({len(cl)} lengths), "
            f"p_no_context={d.p_no_context}",
            f"  pr transform: {d.pr_transform}",
            f"  denoiser    : d_model={m.d_model} depth={m.depth} heads={m.n_heads}",
            f"  gmt encoder : d_model={m.gmt_d_model} depth={m.gmt_depth} "
            f"max_years={m.gmt_max_years}",
            f"  training    : {t.max_steps} steps, batch {t.batch_size}, lr {t.lr:g}, "
            f"amp={t.amp}, device={t.device}",
            f"  out_dir     : {t.out_dir}",
        ])

    # -- (de)serialisation -------------------------------------------------
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
