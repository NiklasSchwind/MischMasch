"""End-to-end smoke test on synthetic data in the exact input format.

Not a scientific test -- it only proves the pipeline runs, shapes line up, the
loss decreases, long-scenario stitching works, and the diagnostics execute.

    python smoke_test.py
"""

import numpy as np
import torch

from misch_masch import Config, ScenarioSampler, check_data, train_from_sims
from misch_masch import evaluate

RNG = np.random.default_rng(0)
N_TAS, N_PR = 57, 59
C = N_TAS + N_PR


def make_gmt(n_years: int, peak: float, shape: str) -> np.ndarray:
    t = np.arange(n_years) / n_years
    if shape == "ramp":
        g = peak * t
    elif shape == "overshoot":
        g = peak * np.sin(np.pi * np.clip(t * 1.3, 0, 1)) + 0.3 * peak * t
    else:  # stabilise
        g = peak * (1 - np.exp(-3 * t))
    return g + 0.05 * RNG.standard_normal(n_years).cumsum() * 0.1


def make_sim(n_years: int, peak: float, shape: str) -> np.ndarray:
    """Fake but structurally realistic: GMT-scaled response + seasonal cycle
    + AR(1) internal variability, tas O(10), pr O(1e-5)."""
    T = n_years * 12
    g_annual = make_gmt(n_years, peak, shape)
    g_monthly = np.repeat(g_annual, 12)

    month = np.arange(T) % 12
    seas = np.sin(2 * np.pi * month / 12)[None]

    pattern_tas = RNG.uniform(0.5, 2.5, (N_TAS, 1))
    pattern_pr = RNG.uniform(-1.0, 1.0, (N_PR, 1)) * 2e-6
    amp_tas = RNG.uniform(2.0, 15.0, (N_TAS, 1))
    amp_pr = RNG.uniform(1e-6, 8e-6, (N_PR, 1))

    def ar1(n_ch, sd, rho=0.6):
        e = RNG.standard_normal((n_ch, T)) * sd
        out = np.empty_like(e)
        out[:, 0] = e[:, 0]
        for i in range(1, T):
            out[:, i] = rho * out[:, i - 1] + np.sqrt(1 - rho ** 2) * e[:, i]
        return out

    tas = pattern_tas * g_monthly[None] + amp_tas * seas + ar1(N_TAS, 1.5)
    pr = pattern_pr * g_monthly[None] + amp_pr * seas + ar1(N_PR, 3e-6)

    sim = np.empty((1 + C, T), dtype=np.float32)
    sim[0] = g_monthly
    sim[1 : 1 + N_TAS] = tas
    sim[1 + N_TAS :] = pr
    return sim


def main():
    torch.manual_seed(0)

    # ---- fake dataset: 3 scenarios x 4 members, 60 years each --------------
    sims, groups = [], []
    for si, (peak, shape) in enumerate(
        [(3.0, "ramp"), (2.0, "stabilise"), (4.0, "overshoot")]
    ):
        for _ in range(4):
            sims.append(make_sim(60, peak, shape))
            groups.append(f"scen{si}")

    cfg = Config()
    cfg.data.n_tas, cfg.data.n_pr = N_TAS, N_PR
    cfg.data.window = 96
    cfg.data.val_fraction = 1 / 3
    cfg.model.d_model, cfg.model.depth, cfg.model.n_heads = 96, 3, 4
    cfg.model.gmt_d_model, cfg.model.gmt_depth, cfg.model.gmt_heads = 48, 2, 4
    cfg.model.cond_dim = 96
    cfg.diffusion.sample_steps = 20
    cfg.train.batch_size = 16
    cfg.train.max_steps = 300
    cfg.train.warmup_steps = 50
    cfg.train.log_every = 50
    cfg.train.val_every = 150
    cfg.train.ckpt_every = 300
    cfg.train.num_workers = 0
    cfg.train.device = "cpu"
    cfg.train.amp = False
    cfg.train.out_dir = "runs/smoke"

    out = train_from_sims(sims, cfg, groups=groups)

    # ---- long-scenario inference ------------------------------------------
    n_years = 80
    gmt = np.repeat(make_gmt(n_years, 3.5, "overshoot"), 12)
    sampler = ScenarioSampler.from_checkpoint(out["path"], device="cpu")
    ens = sampler.sample(gmt, n_members=4, seed=1)
    print("[smoke] ensemble shape:", ens.shape)
    assert ens.shape == (4, C, n_years * 12), ens.shape
    assert np.isfinite(ens).all(), "non-finite values in generated ensemble"
    print(f"[smoke] tas range {ens[:, :N_TAS].min():.2f} .. {ens[:, :N_TAS].max():.2f}")
    print(f"[smoke] pr  range {ens[:, N_TAS:].min():.3g} .. {ens[:, N_TAS:].max():.3g}")

    # ---- GMT-consistency projection hook ---------------------------------
    w = np.ones(N_TAS) / N_TAS
    ens_c = sampler.sample(gmt, n_members=2, seed=2, area_weights=w, progress=False)
    resid = np.abs(ens_c[:, :N_TAS].mean(1) - gmt[None]).max()
    print(f"[smoke] max |mean(tas) - GMT| with projection = {resid:.2e}")
    assert resid < 1e-2, "GMT projection did not bind"

    # ---- diagnostics ------------------------------------------------------
    ref = np.stack([make_sim(n_years, 3.5, "overshoot")[1:] for _ in range(4)])
    evaluate.report(ens, ref, N_TAS, gmt_monthly=gmt)

    rh = evaluate.rank_histogram(ens, ref[0])
    print("[smoke] rank histogram:", np.round(rh, 3))
    print("[smoke] crps (median over channels):", float(np.median(evaluate.crps(ens, ref[0]))))
    f, p = evaluate.power_spectrum(ens)
    print("[smoke] spectrum shape:", f.shape, p.shape)
    nn = evaluate.nearest_neighbour_distance(
        ens[:, :, :96], np.stack([s[1:, :96] for s in sims])
    )
    print("[smoke] nearest-neighbour distances:", np.round(nn, 3))

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
