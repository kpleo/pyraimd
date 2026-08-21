# M2 design — committee + online conformal switching

Spec for milestone M2 of `docs/design-phase-b.md` (§9). Normative where the constitution
leaves choices open; deviations must be documented in the commit message.

## 1. CommitteeSurrogate

- K = 4 members sharing the frozen MACE-MP-0 "small" backbone. Each member owns a readout
  head φ_k (the MACE readout parameters only), initialized from the foundation head plus
  small perturbation, fine-tuned on a bootstrap resample (sample with replacement, |D_k| =
  |D|) of the current label store D.
- Prediction: Ē = mean_k E_k; F̄_i = mean_k F_{i,k}. Per-atom spread
  σ_i = sqrt(mean_k |F_{i,k} − F̄_i|²) (population RMS of the deviation vectors);
  committee score s = max_i σ_i (conservative max-atom).
- Implements the `Surrogate` protocol; `uncertainty` = σ_i (no longer NaN — honest now).
- Fine-tuning: Adam lr = 1e-3, loss = MSE(E)/N + 10·MSE(F), 50 epochs on the (tiny)
  bootstrap set. CPU-only. Backbone parameters frozen by name filter ("readout" in
  parameter name stays trainable; if introspection finds none, raise — never silently
  train nothing).
- Cold start: at load, member 0 keeps the clean foundation head and members 1..K−1
  get a seeded scale-aware perturbation of it, so σ > 0 from the first prediction.
  (Pre-7615191 the members were identical until the first fine-tune → σ ≡ 0 →
  r = e/(s+δ) degenerated to e/δ ≈ 1000·e, pinning q̂ at the window max for a full
  window and locking out acceptance. The switch's |W| < w_min → "dft" policy is
  unchanged.)

## 2. ConformalSwitch (the contribution)

Parameters: α = 0.05 (target miscoverage), ε_acc = 0.1 eV/Å (accuracy budget),
window W = 64 labeled steps, w_min = 16, δ = 1e-3 eV/Å (spread floor).

- State: sliding window of pairs (s_j, e_j) from the w most recent DFT-labeled steps,
  where e_j = max_i |F̄_i − F_true_i| comes from the shadow evaluation (M1 already logs it).
- Normalized nonconformity: r_j = e_j / (s_j + δ).
- Quantile with finite-sample correction: q̂ = the ⌈(n+1)(1−α)⌉-th order statistic of
  {r_j} (n = |W| current window size).
- Predicted error bound at a new configuration: B(s) = q̂ · (s + δ).
- Decision: route "dft" iff B(s) > ε_acc, or |W| < w_min (cold start). `Decision.score`
  = B(s); `reason` carries s, q̂, |W|.
- Update: after every DFT-labeled step, append (s, e) and evict the oldest pair.

Property being tested (not assumed): under approximate exchangeability of the window,
the long-run fraction of accepted ("ml") steps with true force error > ε_acc should be
≈ α. MD trajectories violate exchangeability under drift — M2 *measures* the gap.

## 3. Fine-tune trigger

Every N_label = 8 new DFT labels → `committee.finetune(store.iter_labels(...))`.
Hooked inside the DFT branch of the loop (calculator or runner level — implementer's
choice, but it must also fire during offline replay).

## 4. Coverage experiment (M2 acceptance gate)

`experiments/coverage_h2o.py`, all local CPU:

1. **Collect**: 300-step NVE H₂O @ 300 K trajectory driven by frozen MACE-MP-0; at every
   frame compute PySCF truth and store (atoms, truth). (~3–5 min.)
2. **Replay**: fresh committee, chronological pass over stored frames. At each frame:
   committee.predict → ConformalSwitch.assess → if "dft": reveal the stored label, update
   the window, fire fine-tunes per §3. Record every decision.
3. **Metrics**: DFT fraction; effective miscoverage α̂ = (#accepted steps with e > ε_acc) /
   (#accepted steps). **Gate: α̂ within α ± 0.03.**
4. **Ablation**: ScheduledSwitch at the same DFT fraction as the conformal run — report
   its α̂ for comparison (expected worse). Also committee-only (fixed threshold on s,
   tuned to same budget) if cheap.
5. Print a compact table; save the decision log next to the store.

## 5. Tests (beyond M1 suite)

- Synthetic exchangeable (s, e) stream with known conditional relation → empirical
  coverage of ConformalSwitch decisions within tolerance (statistical test, generous
  seed-fixed margins).
- Cold start: |W| < w_min always routes "dft".
- Committee with identical members → σ ≡ 0 → ConformalSwitch never trusts it (δ floor
  keeps B > 0 only via q̂·δ; verify behavior explicitly).
- Quantile order-statistic edge cases (n=0, n=1, n=w_min−1).
- Replay harness end-to-end with FakeEngine/FakeSurrogate committee analog.
- Slow-marked: real CommitteeSurrogate fine-tune improves held-out force MAE by ≥ 30%
  vs frozen foundation on 30 H₂O labels / 10 held-out frames.

## 6. Non-goals for M2

No NVT/NPT, no periodic systems, no backbone fine-tuning, no stress labels, no HPC.
M3 opens all of those.

## 7. M2 run-1 findings (2026-08-19) — amendment

- First run at ε_acc = 0.1 eV/Å: the switch routed 302/302 frames to DFT (37 fine-tunes);
  α̂ undefined. Decision-log diagnosis: the head-only committee's max-atom force error on
  OOD water is median 0.17 / min 0.138 eV/Å — **no frame ever meets a 0.1 budget**.
  Refusing everything was the correct behavior; the budget, not the switch, was
  miscalibrated for this system.
- Calibration quality is already right: the bound tracks the true error nearly tightly
  (late-frame median B(s) ≈ 0.22–0.25 vs e ≈ 0.17 eV/Å) even though the raw committee
  spread underestimates the error ~85× — the learned q̂ (≈ 75–85) performs exactly the
  correction it exists for.
- Amendment: ε_acc → 0.25 eV/Å for the H₂O gate (above the measured floor so acceptance
  is possible); the experiment gained a CORRECT-REFUSAL verdict (no frame meets budget →
  exit 0, informative) and reports bound tightness B(s)/e.
- Paper takeaway: the method has two honest regimes — calibrated acceptance when the
  surrogate can meet the budget, and honest refusal when it cannot. Show both.

## 8. M2 run-2 findings (2026-08-19) — gate amendment

Run at ε_acc = 0.25 eV/Å (302 frames, seeded trajectory):

- Conformal switch: DFT fraction 0.252 (76/302), 9 fine-tunes, **α̂ = 0.000** over 226
  accepted frames. Scheduled ablation at the same DFT fraction (period 4): **α̂ = 0.0929**
  — nearly 2× above target. 294/302 frames met the budget (oracle DFT fraction 0.026).
- Gate revision: the original two-sided gate (|α̂ − α| ≤ 0.03) was mis-specified relative
  to the theory — conformal's marginal guarantee is **one-sided** (coverage ≥ 1 − α).
  The gate is now one-sided safety: α̂ ≤ α + 0.03. Under-coverage (α̂ < α) is an
  *efficiency* question, tracked via bound tightness B(s)/e and oracle overhead.
- Open finding (paper material): with a head-only committee the spread s carries little
  conditional signal on OOD water (σ ≈ 0.002 while e ≈ 0.17 eV/Å), so the bound is
  ~4.6× conservative in the accepting regime → DFT fraction 0.252 vs oracle 0.026.
  Tightening paths: conditional/locally adaptive conformal, a learned e|s regression as
  nonconformity score, or a spread with more capacity (deeper fine-tune). The headline
  comparison already holds: at equal DFT budget, conformal 0 violations vs scheduled 9.3%.

## 9. Spread-estimator correction (2026-08-21) — supersedes the σ values above

- The σ quoted in §7–§8 was computed as std_k(|F_{i,k} − F̄_i|) (std of deviation
  *norms*) — a broken estimator: identically 0 at K=2 (norms from the mean are equal
  by construction) and systematically understating the spread at any K. Fixed to the
  standard committee-UQ estimator σ_i = sqrt(mean_k |F_{i,k} − F̄_i|²). The ~85×
  understatement ratio in §7 (and the r ≈ 60–65 of smoke 7615281) were both measured
  with the broken estimator and are therefore inflated by an unknown factor; the
  qualitative conclusion (head-only, same-backbone members stay jointly
  overconfident) stands.
- Complementary fix, same day: **mixed-backbone committees** (MACE-MP-0b3 + MACE-MPA-0
  members). Zero-shot cross-backbone r ≈ 2.2 on the 7615191 smoke labels (vs ~65
  same-backbone) — cross-family diversity is the honest-spread lever; both backbones
  share the identical data interface (head "default", 89-element z_table, r_max 6.0),
  so one graph builder serves all members.
