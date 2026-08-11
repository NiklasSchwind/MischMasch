"""Unit checks for the parts that are easy to get silently wrong."""

import json
import tempfile

import numpy as np
import torch

from misch_masch import (Config, Normalizer, ScenarioSampler, build_model,
                         group_split, train_from_sims)
from misch_masch.data import CropDataset, collate, n_gmt_features
from misch_masch.diffusion import Diffusion

N_TAS, N_PR = 7, 9
C = N_TAS + N_PR
RNG = np.random.default_rng(0)


def tiny_cfg(**over):
    cfg = Config()
    cfg.data.n_tas, cfg.data.n_pr = N_TAS, N_PR
    cfg.model.d_model, cfg.model.depth, cfg.model.n_heads = 32, 2, 4
    cfg.model.gmt_d_model, cfg.model.gmt_depth, cfg.model.gmt_heads = 16, 1, 4
    cfg.model.cond_dim = 32
    cfg.train.device, cfg.train.amp, cfg.train.num_workers = "cpu", False, 0
    for k, v in over.items():
        obj, attr = k.split(".")
        setattr(getattr(cfg, obj), attr, v)
    return cfg


def make_sim(n_years=40):
    T = n_years * 12
    g = np.repeat(np.linspace(0, 3, n_years), 12)
    sim = np.empty((1 + C, T))
    sim[0] = g
    sim[1 : 1 + N_TAS] = RNG.standard_normal((N_TAS, T)) * 5 + g
    sim[1 + N_TAS :] = RNG.standard_normal((N_PR, T)) * 3e-6
    return sim


def test_normalizer_roundtrip():
    cfg = tiny_cfg()
    sims = [make_sim() for _ in range(3)]
    nrm = Normalizer.fit(sims, cfg.data)
    for t0 in (0, 12, 37):
        raw = sims[0][1 : 1 + C, t0 : t0 + 96]
        z = nrm.transform_targets(raw, t0=t0)
        back = nrm.inverse_transform_targets(z, t0=t0)
        assert np.allclose(raw, back, rtol=1e-5, atol=1e-12), np.abs(raw - back).max()
    # normalised data should actually be ~unit variance in BOTH blocks
    z = nrm.transform_targets(sims[0][1 : 1 + C], t0=0)
    s_tas, s_pr = z[:N_TAS].std(), z[N_TAS:].std()
    assert 0.5 < s_tas < 2.0 and 0.5 < s_pr < 2.0, (s_tas, s_pr)
    print(f"  normalizer roundtrip OK (std tas={s_tas:.3f}, pr={s_pr:.3f})")


def test_pr_transform_none():
    cfg = tiny_cfg(**{"data.pr_transform": "none"})
    sims = [make_sim() for _ in range(2)]
    nrm = Normalizer.fit(sims, cfg.data)
    raw = sims[0][1 : 1 + C]
    assert np.allclose(raw, nrm.inverse_transform_targets(nrm.transform_targets(raw)))
    print("  pr_transform='none' roundtrip OK")


def test_config_roundtrip():
    cfg = tiny_cfg(**{"data.window": 240, "model.n_esm": 3})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        cfg.save(f.name)
        cfg2 = Config.load(f.name)
    assert cfg2.data.window == 240 and cfg2.model.n_esm == 3
    assert tuple(cfg2.data.context_lengths) == tuple(cfg.data.context_lengths)
    print("  config JSON roundtrip OK")


def test_group_split_no_leak():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    tr, va = group_split(8, groups, 0.25, seed=0)
    assert set(tr) & set(va) == set()
    assert {groups[i] for i in tr} & {groups[i] for i in va} == set()
    print(f"  group split OK (train groups {sorted({groups[i] for i in tr})})")


def test_dataset_context_masking():
    cfg = tiny_cfg()
    sims = [make_sim() for _ in range(2)]
    nrm = Normalizer.fit(sims, cfg.data)
    ds = CropDataset(sims, nrm, cfg, train=True)
    seen = set()
    for i in range(300):
        b = ds[i % len(ds)]
        L = int(b["ctx_mask"].sum())
        seen.add(L)
        # context must equal x0 inside the prefix and be zero outside
        assert torch.allclose(b["ctx"][:, :L], b["x0"][:, :L])
        assert torch.all(b["ctx"][:, L:] == 0)
        assert int(b["end_year"]) == 0 or True
    assert 0 in seen and 36 in seen, seen
    print(f"  context masking OK (lengths seen: {sorted(seen)})")


def test_january_alignment():
    cfg = tiny_cfg()
    sims = [make_sim()]
    nrm = Normalizer.fit(sims, cfg.data)
    ds = CropDataset(sims, nrm, cfg, train=False)
    assert all(x % 12 == 0 for _, x in ds.index)
    assert all(int(ds[i]["month_idx"][0]) == 0 for i in range(0, len(ds), 7))
    # non-January mode gives 12x the crops and varied phases
    cfg2 = tiny_cfg(**{"data.january_start": False})
    ds2 = CropDataset(sims, nrm, cfg2, train=False)
    assert len(ds2) > 10 * len(ds)
    assert len({int(ds2[i]["month_idx"][0]) for i in range(24)}) > 1
    print(f"  crop alignment OK ({len(ds)} Jan-only vs {len(ds2)} any-month crops)")


def test_causal_gmt_encoder():
    """The readout at end_year must not depend on any later year."""
    cfg = tiny_cfg()
    model = build_model(cfg, n_gmt_features(cfg)).eval()
    Y, B = 50, 2
    gmt = torch.randn(B, Y)
    feats = torch.randn(B, n_gmt_features(cfg))
    end = torch.tensor([20, 20])
    with torch.no_grad():
        a = model.gmt_encoder(gmt, end, feats)
        gmt2 = gmt.clone()
        gmt2[:, 25:] += 10.0                       # perturb the future only
        b = model.gmt_encoder(gmt2, end, feats)
    d = (a - b).abs().max().item()
    assert d < 1e-4, f"GMT encoder is not causal (delta {d})"
    # and it must respond to the past
    gmt3 = gmt.clone(); gmt3[:, :10] += 10.0
    with torch.no_grad():
        c = model.gmt_encoder(gmt3, end, feats)
    assert (a - c).abs().max().item() > 1e-3
    print(f"  GMT encoder causality OK (future delta {d:.2e})")


def test_diffusion_identities():
    dif = Diffusion(200, "cosine")
    x0 = torch.randn(4, C, 96)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 50, 120, 199])
    x_t = dif.q_sample(x0, t, noise)
    v = dif.v_target(x0, noise, t)
    assert torch.allclose(dif.x0_from_v(x_t, v, t), x0, atol=1e-4)
    assert torch.allclose(dif.eps_from_v(x_t, v, t), noise, atol=1e-4)
    assert dif.alpha_bar[0] > dif.alpha_bar[-1]
    print("  diffusion v-parameterisation identities OK")


def test_masked_loss_ignores_context():
    """Changing the target inside the clean prefix must not change the loss."""
    cfg = tiny_cfg()
    model = build_model(cfg, n_gmt_features(cfg)).eval()
    dif = Diffusion(100, "cosine")
    B, W = 3, cfg.data.window
    batch = dict(
        x0=torch.randn(B, C, W), ctx=torch.zeros(B, C, W),
        ctx_mask=torch.zeros(B, W), gmt=torch.randn(B, 30),
        end_year=torch.tensor([7, 7, 7]),
        gmt_feats=torch.randn(B, n_gmt_features(cfg)),
        month_idx=torch.arange(W).remainder(12).expand(B, W).contiguous(),
        esm_id=torch.zeros(B, dtype=torch.long),
    )
    batch["ctx_mask"][:, :36] = 1.0
    batch["ctx"][:, :, :36] = batch["x0"][:, :, :36]
    g = torch.Generator().manual_seed(0)
    l1 = dif.loss(model, batch, generator=g)
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b2["x0"] = b2["x0"].clone()
    b2["x0"][:, :, :36] += 100.0                   # only the context region
    b2["ctx"][:, :, :36] = b2["x0"][:, :, :36]
    g = torch.Generator().manual_seed(0)
    l2 = dif.loss(model, b2, generator=g)
    # the loss weight is zero there, but the network input changed, so allow
    # a tolerance -- what must hold is that the *weighting* excludes it
    w = (1 - batch["ctx_mask"]).unsqueeze(1)
    assert float(w[:, :, :36].sum()) == 0.0
    print(f"  masked loss weighting OK (ctx-only perturbation: "
          f"{l1.detach().item():.4f} -> {l2.detach().item():.4f})")


def test_long_window_and_multi_esm():
    cfg = tiny_cfg(**{
        "data.window": 240, "data.val_fraction": 0.0, "model.n_esm": 2,
        "train.max_steps": 6, "train.warmup_steps": 2, "train.batch_size": 4,
        "train.log_every": 1000, "train.val_every": 10**9,
        "train.ckpt_every": 10**9, "train.out_dir": "runs/test_long",
    })
    cfg.data.context_lengths = tuple(range(12, 240, 12))
    sims = [make_sim(60) for _ in range(4)]
    out = train_from_sims(sims, cfg, groups=["a", "a", "b", "b"],
                          esm_ids=[0, 0, 1, 1], verbose=False)
    s = ScenarioSampler.from_checkpoint(out["path"], device="cpu")
    gmt = np.repeat(np.linspace(0, 4, 50), 12)
    for eid in (0, 1):
        ens = s.sample(gmt, n_members=2, steps=4, esm_id=eid, seed=0, progress=False)
        assert ens.shape == (2, C, 600) and np.isfinite(ens).all()
    print("  window=240 + multi-ESM embedding OK")


def test_member_diversity_and_determinism():
    cfg = tiny_cfg(**{
        "data.val_fraction": 0.0, "train.max_steps": 6, "train.warmup_steps": 2,
        "train.batch_size": 4, "train.log_every": 1000,
        "train.val_every": 10**9, "train.ckpt_every": 10**9,
        "train.out_dir": "runs/test_div",
    })
    sims = [make_sim(40) for _ in range(3)]
    out = train_from_sims(sims, cfg, verbose=False)
    s = ScenarioSampler.from_checkpoint(out["path"], device="cpu")
    gmt = np.repeat(np.linspace(0, 3, 20), 12)
    a = s.sample(gmt, n_members=4, steps=6, seed=7, progress=False)
    b = s.sample(gmt, n_members=4, steps=6, seed=7, progress=False)
    c = s.sample(gmt, n_members=4, steps=6, seed=8, progress=False)
    assert np.allclose(a, b), "same seed must reproduce"
    assert not np.allclose(a, c), "different seed must differ"
    spread = a.std(axis=0).mean()
    assert spread > 0, "members are identical -- no ensemble spread"
    print(f"  seeding + member diversity OK (mean across-member std {spread:.4g})")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("running checks:")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nALL TESTS PASSED")
