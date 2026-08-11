# MISCH-MASCH

A conditional diffusion emulator that generates monthly regional `tas` and `pr`
time series from a global-mean-temperature (GMT) trajectory.

* **Denoiser:** 1-D DiT (transformer, adaLN-Zero conditioning) over month tokens
* **Conditioning:** causal transformer over the *full* annual GMT history plus
  explicit path-dependence features
* **Long scenarios:** context-conditioned outpainting — 96-month windows,
  60-month stride, 36-month clean prefix
* **Probabilistic:** ensemble members differ by initial noise + ancestral
  sampling; no classifier-free guidance (deliberately)

---

## Install

```bash
pip install torch numpy
```

## Data format

A `list` of numpy arrays, one per simulation, each `(117, T)` with `T % 12 == 0`
and column 0 = January:

| rows      | contents                                             |
|-----------|------------------------------------------------------|
| `0`       | GMT — annual value repeated 12× within each year     |
| `1..57`   | monthly regional `tas` (57 IPCC regions)             |
| `58..116` | monthly regional `pr` (59 IPCC regions)              |

Lengths may differ between simulations. Set `cfg.data.n_tas` / `n_pr` if your
region counts differ.

## Train

```python
import numpy as np
from misch_masch import Config, train_from_sims

sims = [...]                                  # your list of (117, T) arrays
groups = [...]                                # one label per sim: the SCENARIO
                                              # (or parent run for branched runs)

cfg = Config()
cfg.train.out_dir   = "runs/v1"
cfg.train.device    = "cuda"
cfg.train.max_steps = 200_000
cfg.train.batch_size = 64

out = train_from_sims(sims, cfg, groups=groups)
```

`groups` matters. Ensemble members of the same scenario are near-duplicates and
scenarios that branch from a shared historical run share a long prefix — if they
straddle the train/val split, the validation loss is meaningless. Pass the
scenario name (or the parent-run id) and the split is done at the group level.

## Generate a new scenario

```python
from misch_masch import ScenarioSampler

s = ScenarioSampler.from_checkpoint("runs/v1/last.pt", device="cuda")

gmt = np.repeat(annual_gmt, 12)               # e.g. 3000 monthly values
ens = s.sample(gmt, n_members=20, seed=0)     # -> (20, 116, 3000), physical units
```

Rows of `ens` are ordered exactly like rows 1..116 of your inputs. All members
are generated in one batch, so member count is nearly free.

Optional hard constraint — force the area-weighted mean of the generated `tas`
to match the prescribed GMT exactly, by projecting `x0` at every denoising step:

```python
ens = s.sample(gmt, n_members=20, area_weights=region_area_fractions)
```

`region_area_fractions` is a length-`n_tas` array summing to 1. Only physically
meaningful if your `tas` regions tile the globe.

## Evaluate

```python
from misch_masch import evaluate
evaluate.report(ens, ref_ensemble, n_tas=57, gmt_monthly=gmt)
```

where `ref_ensemble` is `(M, 116, T)` of held-out ESM members for the same
scenario. See `misch_masch/evaluate.py` for the individual diagnostics.

## Smoke test

```bash
python smoke_test.py    # ~3 min on CPU, synthetic data, proves the pipeline runs
```

---

## Design decisions, and why

### Per-(channel, calendar-month) standardisation is not optional

Raw `tas` is O(10); raw `pr` is O(1e-5). Under an MSE denoising loss the `pr`
channels contribute ~1e-12 of the gradient — you would train a `tas`-only model
with 59 dead channels. `check_data` prints the magnitude ratio so this is
impossible to miss. Standardising per *calendar month* additionally removes the
climatological seasonal cycle, so capacity goes to anomalies instead of
re-learning summer and winter.

`pr` additionally gets a **signed cube root** before standardisation
(`cfg.data.pr_transform`): monotone, exactly invertible, symmetric, and it tames
the heavy tail. Set `"none"` to compare.

### One token per month, not 2-D image patches

The `(region × time)` matrix is not an image. Time is translation-invariant;
the region axis is an arbitrary permutation of IPCC regions. A 4×4 patch would
impose a spurious locality prior, mixing (say) Central Africa with Northern
Europe because they happen to be adjacent in the row index. So: one token per
month, with the full 116-vector as the token's channel dimension. Cross-region
structure comes from the embedding matrix and the residual stream (no false
locality), cross-time structure from full self-attention over 96 tokens.

### GMT is annual

Row 0 is constant within each calendar year, so it is encoded at annual
resolution — 12× cheaper and honest about the information content. The encoder
is *causal*, so one forward pass over an entire scenario yields a valid
embedding at every year end; long-scenario inference is one pass plus a gather.

### Explicit path-dependence features

Regional response depends on the GMT *path* mostly through ocean heat uptake,
which is a near-integral of the forcing. So the encoder gets, alongside its
learned readout: current level, 10-yr mean, 10-yr and 50-yr trends, running
mean, cumulative GMT, peak-so-far, and overshoot depth. A purely learned encoder
extrapolates badly to scenario shapes (deep overshoots, sharp stabilisations) it
never saw in training; these features are cheap insurance.

`cfg.model.use_elapsed_time_feature` also passes elapsed years. This helps with
historical forcings GMT alone does not explain (volcanoes, aerosols) but makes
the model slightly calendar-aware rather than purely path-driven. Set `False` if
you want conditioning on the GMT path only.

### Outpainting is *trained*, not bolted on

Windows after the first receive the previous 36 months as a clean prefix. The
model is trained that way — random prefix lengths, loss computed only on the
positions that must be generated, plus 20% fully-unconditional windows for the
start of a scenario. Untrained RePaint-style replacement at inference produces
seams that compound over dozens of windows; this does not.

`cfg.data.context_lengths` **must** contain `window - stride`. The default
covers every multiple of 12 up to `window - 12`, which also keeps the final
(snapped) window of an arbitrary-length scenario in-distribution.

### No classifier-free guidance

CFG with `w > 1` systematically contracts the sample distribution. For an image
generator that reads as "higher quality"; for an ensemble emulator it silently
destroys the ensemble spread, which is the entire product. It is not implemented
here, on purpose. Diversity comes from the initial noise and from ancestral
sampling (`eta = 1.0`).

---

## Known limitations — read before trusting a 250-year emulation

**1. The window is 8 years long.** Internal variability on timescales much
longer than that is represented only insofar as it is carried by the 36-month
overlap and the GMT conditioning. Multidecadal variability (AMV/PDO-like) will
tend to come out too weak. Check `evaluate.variance_by_timescale` — if
`var_ratio_20yr` and `var_ratio_50yr` fall away while `var_ratio_1yr` is fine,
this is what you are looking at. Two fixes:

* Raise `cfg.data.window` to 240 (20 years). Still only 240 tokens — cheap.
* **Two-stage cascade** (the better fix): stage A generates *annual* means for
  the whole scenario in one shot (250 tokens for 3000 months — no stitching, no
  drift, correct low-frequency spectrum); stage B generates the 12 monthly
  values within each year conditioned on the annual field. This also dissolves
  the long-generation problem entirely.

**2. Autoregressive drift.** Thirty-plus stitched windows can accumulate bias.
The GMT embedding re-anchors the model at every window, which is the main
defence, but verify with `evaluate.trend_drift` before trusting a long run. If
drift appears: increase overlap (reduce `stride`), or enable the `area_weights`
GMT projection.

**3. Data volume.** With 8-year windows the number of *effective* independent
samples is roughly `sum(T) / window`, not the number of crops — crops overlap
heavily. `train_from_sims` prints this. Below ~1e4, keep `d_model`/`depth` small
(defaults give ~5 M parameters) and check `evaluate.nearest_neighbour_distance`
for memorisation.

**4. January-only crops cost a factor of 12 in training data.** You asked for
this and it is the default. If data turns out to be the binding constraint, set
`cfg.data.january_start = False`: crops then start at any month, and the
per-token calendar-month embedding (already in the model) keeps the seasonal
cycle learnable. Everything else works unchanged.

**5. Physical constraints are not enforced.** Generated `pr` anomalies can fall
below `-climatology`, i.e. imply negative total precipitation. Check how often,
and if it matters, either model `pr` in a transformed space with the right
support or clip and re-close the budget.

**6. Volcanic and aerosol forcing is not in GMT.** Historical-period regional
patterns driven by stratospheric aerosol are not recoverable from a GMT path
alone. If they matter, add them as extra conditioning channels alongside GMT.

**7. Benchmark against MESMER-M.** If a diffusion model cannot beat the standard
monthly emulator on the diagnostics in `evaluate.py`, the extra machinery is not
earning its keep. That comparison is the honest headline result.

---

## Extensions with hooks already in place

* **Multiple ESMs.** Set `cfg.model.n_esm = <n>` and pass `esm_ids=[...]` (one
  per simulation) to `train_from_sims`; pass `esm_id=` to
  `ScenarioSampler.sample`. The embedding is zero-initialised, so a single-model
  run is unchanged. Group your train/val split by ESM if you want to test
  out-of-sample generalisation across models.
* **Axial attention over regions.** Alternate the existing time-attention
  blocks with attention over 116 region tokens plus learned region embeddings.
  Useful if you want the model to reason about regions explicitly, or to
  generalise to a different region set.
* **Flow matching instead of DDPM.** `diffusion.py` is small and self-contained;
  swapping the v-prediction objective for rectified flow is ~30 lines and often
  behaves better in the low-data regime with few sampling steps.

## Files

```
misch_masch/
  config.py      all hyperparameters, JSON round-trippable
  data.py        normalisation, crop dataset, GMT path features, group split
  model.py       causal GMT encoder + 1-D DiT denoiser
  diffusion.py   v-prediction cosine diffusion, masked loss, DDIM/ancestral sampler
  train.py       training loop, EMA, checkpointing
  sample.py      ScenarioSampler (outpainting) + GMT-consistency projector
  evaluate.py    spread calibration, spectra, drift, joint structure, memorisation
smoke_test.py    end-to-end run on synthetic data
```
