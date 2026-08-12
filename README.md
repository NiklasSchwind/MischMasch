# MISCH-MASCH

A conditional diffusion emulator that generates monthly regional `tas` and `pr`
time series from a global-mean-temperature (GMT) trajectory.

* **Denoiser:** 1-D DiT (transformer, adaLN-Zero conditioning) over month tokens
* **Conditioning:** causal transformer over the *full* annual GMT history plus
  explicit path-dependence features
* **Long scenarios:** context-conditioned outpainting — 240-month windows,
  120-month stride, 120-month clean prefix (all set in `config.py`)
* **Probabilistic:** ensemble members differ by initial noise + ancestral
  sampling; no classifier-free guidance (deliberately)

---

## Install

```bash
pip install torch numpy
```

`misch_masch/config.py` is the **single source of truth**. Every flag in
`run_access_esm.py` defaults to `None` and only overrides its config field when
passed explicitly, so editing the defaults in `config.py` is enough. `Config`
validates itself on construction and again via `finalize()` after mutation, so
inconsistent settings (window not a multiple of 12, window past `max_window`,
`d_model` not divisible by `n_heads`, a context ladder that no longer fits the
window) fail at startup instead of hours in. Leave `data.context_lengths`
empty to derive every multiple of 12 up to `window - 12`.

## Data format

A `list` of numpy arrays, one per simulation, each `(117, T)` with `T % 12 == 0`
and column 0 = January:

| rows      | contents                                             |
|-----------|------------------------------------------------------|
| `0`       | GMT — annual value repeated 12× within each year     |
| `1..58`   | monthly regional `tas` (58 IPCC regions)             |
| `59..116` | monthly regional `pr` (58 IPCC regions)              |

Lengths may differ between simulations. `data.n_tas` / `data.n_pr` in
`misch_masch/config.py` define the split and are the only place it is written
down; `run_access_esm.py` reads them from there.

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

## Multi-ESM driver script

`run_access_esm.py` (name is a leftover — rename freely, just update the sbatch
line) loads the training scenarios for **five CMIP6 models** via
`emuvaluate.data_preparation.load_scenarios`, trains one model over all of them,
then emulates `ssp245` for each:

```
CanESM5  ACCESS-ESM1-5  MPI-ESM1-2-LR  MIROC6  IPSL-CM6A-LR
```

Models are distinguished by the learned ESM embedding (`model.n_esm = 5`), which
is zero-initialised, so nothing else about the architecture changes.

```bash
python run_access_esm.py                       # train + emulate all five
python run_access_esm.py --skip-train          # reuse the checkpoint, re-emulate
python run_access_esm.py --models CanESM5,MIROC6
python run_access_esm.py --max-members-per-scenario 10   # bound memory
```

Writes to `/hdrive/all_users/schwind/MischMasch` (change with `--out-root`):

```
models/cmip6-5models/best.pt        lowest-validation-loss checkpoint
                                    (model + EMA + normaliser + config)
                                    -- this is what inference uses
models/cmip6-5models/last.pt        final step, for inspection
models/cmip6-5models/config.json
models/cmip6-5models/esm_ids.json   {model name: embedding index}
test_data/ssp245_<MODEL>_emulated.npy       object array of (117, T) arrays,
                                            same layout load_scenarios hands
                                            out, physical units, GMT row kept
test_data/ssp245_<MODEL>_emulated_tas.npy   (n_tas, T) blocks
test_data/ssp245_<MODEL>_emulated_pr.npy    (n_pr, T) blocks
test_data/ssp245_<MODEL>_reference.npy      that ESM's members, unmodified
test_data/metadata.json                     one file covering all models: which
                                            emulated array came from which
                                            source member, seeds, full config
```

```python
sims = list(np.load(".../ssp245_emulated.npy", allow_pickle=True))
```

Three things the script does on purpose:

* **Loads one scenario per `load_scenarios` call.** A single call for all of
  them would lose the scenario label, and the label is what keeps ensemble
  members of a scenario on the same side of the train/val split.
* **Skips scenarios that fail to load** with a warning rather than aborting
  after an hour of I/O — a few of the `esm-1pct-brch-*` runs may not exist for
  every model.
* **Verifies the tas/pr split by magnitude** (`verify_layout`). `tas` is O(10)
  and `pr` is O(1e-5), so the boundary is visible in the data; if the largest
  magnitude break is not at `data.n_tas` the script says so. Getting that wrong
  is silent and fatal, so it is worth the check.
* **Aborts if `train.device` is `cuda` but torch cannot see a GPU**, rather than
  falling back to CPU and burning a 48-hour allocation at 1/100 speed. Override
  with `--allow-cpu`.
* **Emulates from `best.pt`**, not the final step. `--use-last` or
  `--checkpoint PATH` to override.
* **Balances ESM sampling** (`train.balance_esms`). Member counts differ several
  fold between models, so uniform crop sampling would let one dominate the
  shared weights while the others rode on the embedding.
* **Splits train/val within each model** (`strata`), so `val_fraction` of
  *every* model's scenarios is held out. A global draw can leave one model with
  no validation data — which is the model you most wanted to check.
* **Reports validation loss per model** as well as as a balanced mean, and the
  balanced mean is what drives best-checkpoint selection and early stopping:

```
step   12000  VAL loss 0.5241  best 0.5241 @ 12000  *  [CanESM5=0.58 ACCESS-ESM1-5=0.51 ...]
```

  That per-model breakdown is the diagnostic for whether the embedding is
  earning its keep — a model whose loss sits well above the others is not being
  served by the shared weights.

**Known limitation of this configuration:** normalisation statistics are fitted
**pooled across all five models**. They differ in climatology, seasonal-cycle
amplitude and variance, so the embedding has to spend capacity undoing a
preprocessing choice. Per-ESM normalisation statistics are the obvious next
improvement and would likely matter more than the conditioning mechanism itself.
The driver prints this warning at startup so it does not get forgotten.

Emulated members are generated per source `ssp245` member (each conditioned on
that member's own GMT, matching how the model was trained),
`--members-per-gmt` of them each.

## Checkpoint selection and overfitting guards

The first full ACCESS-ESM1-5 run is why these exist. Validation loss bottomed at
**0.5099 at step 18k of 200k**, drifted up to 0.55 by 166k, then jumped
discontinuously to 0.80 at ~168k and stayed there — a training collapse, not
overfitting. The final model was worse than the model at step 2000, and because
only `last.pt` was written, the good one was gone.

So the loop now:

| setting | default | what it does |
|---|---|---|
| `train.save_best` | `True` | writes `best.pt` on every validation improvement |
| `train.early_stop_patience` | `20` | stops after 20 **consecutive** validations with no new best (0 disables). The counter resets only on a new best, so a single small rise costs one tick and can never stop a run by itself. |
| `train.spike_abort_ratio` | `1.25` | aborts if val exceeds the best by 25% — catches a collapse instead of training through it |
| `train.skip_nonfinite_grads` | `True` | drops an update whose gradient norm is not finite rather than letting one batch move the weights somewhere unrecoverable |
| `train.val_batches` | `40` | a **fixed random subset spread across all validation sims and start months** — iterating the first N batches in index order only ever saw the earliest months of the first validation run |

And in the model, `model.qk_norm = True` normalises q and k to unit RMS before
the attention dot product. The collapse was not a bad gradient — training loss
ramped 0.4609 → 0.8333 over ~1300 steps with finite gradients and a smoothly
decaying LR, then plateaued with train ≈ val, which is the signature of
attention entropy collapse (QK logits grow, softmax saturates toward one-hot,
gradients through attention vanish). QK-norm bounds the logits at ~√head\_dim
however large the projections grow: at 100× input scale the measured max logit
is 3.2 with it and 17,781 without. Costs two parameter vectors per attention
layer and no measurable time. Checkpoints are not interchangeable between
settings; ones saved before this existed load as `qk_norm=False` automatically.

Checkpoints are written atomically (`.tmp` then `os.replace`), so a killed job
never leaves a truncated file. The step log now also prints gradient-norm mean
and max, which is what you would look at first to diagnose a collapse.

If `best_step` lands in the first half of the run, the loop says so and suggests
a shorter `max_steps`.

## Reference point for the loss

The data is standardised to unit variance, so with v-prediction
`E[v²] = ᾱ + (1−ᾱ) = 1` exactly. **A model that outputs zero scores 1.0000.**
Read every loss against that: 0.82 means 18% of the v-variance explained, 0.51
means 49%. Precipitation is close to white noise after deseasonalising, so with
half your channels being `pr` the achievable floor is well above zero — the
absolute number matters far less than the diagnostics below.

## Evaluate

```python
from misch_masch import evaluate
evaluate.report(ens, ref_ensemble, n_tas=cfg.data.n_tas, gmt_monthly=gmt)
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
with 58 dead channels. `check_data` prints the magnitude ratio so this is
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
locality), cross-time structure from full self-attention over the window's month tokens.

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

**3. Data volume — this one bit.** The number of *effective* independent samples
is roughly `sum(T) / window`, not the number of crops; crops overlap heavily.
`train_from_sims` prints it. On ACCESS-ESM1-5 with a 240-month window that was
small enough that a 256/6 model (8.9 M params) overfit by step 18k of 200k.
The defaults are now 192/4 with `dropout 0.1` and `weight_decay 0.01`, and
`max_steps` is 25k rather than 200k. Since an overfit *generative* model
memorises training windows, `evaluate.nearest_neighbour_distance` is not
optional here.

**4. Crop alignment.** `data.january_start = False` (the default) lets crops
start at any month, giving 12x the crops as seasonal-phase augmentation; the
per-token calendar-month embedding keeps the seasonal cycle learnable. Set it
to `True` for strictly January-aligned crops. Note that neither setting changes
the number of *effective independent* windows, which is `sum(T) / window`.

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

* **More ESMs.** Add names to `MODELS` in the driver and raise
  `cfg.model.n_esm` to match (the driver refuses to start on a mismatch).
* **Per-ESM normalisation** — see the limitation noted above. Probably the
  highest-value next change for multi-model work.
* **Unseen-ESM generalisation.** The embedding cannot do this by construction.
  Two routes: freeze the network and fit a single new embedding vector on a
  small sample (textual-inversion style), or add a permutation-invariant
  encoder over crops from the target model that emits a vector into the same
  conditioning slot. Validate either with leave-one-ESM-out.
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
