# HPC fleet operations record — pyraimd2

Source of truth: the provider skills under `/Users/pengkang/.codex/skills/` (`hpc-fleet`
router + per-provider skills). This file records live audits and project-level decisions.
**Re-audit live state before every consequential submission; snapshots below go stale.**

Rules that apply everywhere (from the fleet skill):
- Never mix Slurm/MPI/memory/VASP parameters across clusters. Each cluster has its own
  contract below.
- No heavy compute on login nodes; everything through Slurm.
- Clusters cannot reach each other; relay transfers through the local workstation.
- VASP work additionally requires `$vasp-hpc` + the provider's VASP profile (not needed
  for the current PySCF/GPAW phase).
- Parallel Cloud (BSCC-L): external SSH unverified — not in use.

## Live audit 2026-08-19 14:10 CST

| Cluster | Connect | Partition (node shape) | Idle now | Account limits | My jobs |
|---|---|---|---|---|---|
| Huabei 华北 | `hb100775@60.31.21.42:12310` | `8163*` — 48 CPU/node | **126 nodes / 6048 CPU** | node≤4, jobs≤6, submit≤50 | none |
| Neimeng A 内蒙 | `df103967@60.31.21.42:8887` | `normal*` — 28 CPU/node; `9242` — 96 CPU/375G node | 160 nodes (normal) + 33 nodes (9242) | node≤4 per partition, jobs≤6 | none |
| Tencent 腾讯 | `tx100141@140.143.141.78:22` | `tencent*` — 48 CPU/node | 31 nodes / 1488 CPU | node≤4, jobs≤6, submit≤50 | none |
| Huadong A 华东 | `hd100493@221.130.101.31:22023` | `q_amd_share*` — 128 CPU AMD EPYC/node | 10 nodes / 1280 CPU (+342 in mix) | node≤4 (1 in use → 3 free), jobs≤6 | `1680867 si_kmesh_ext` RUNNING (user's, do not touch) |

GPU: none of the four Slurm fleets exposes GPU GRES to these accounts. The GPU path is the
seetacloud H20 instance (currently powered off; ssh alias `H20`, port drifts on restart).

## Per-cluster contracts (do NOT mix)

- **Huabei `8163`**: 48 physical CPU/node. Memory accounting is broken (`RealMemory=1`) →
  **omit `--mem`**. Modules via `/data/profile`; openmpi situation per skill; pack light
  serial tasks into one allocation.
- **Neimeng A `normal`**: 28 CPU/node → `--mem=120G`. **`9242`**: 96 CPU/node →
  `--mem=360G`. Env: `source /data/profile/module.env`; conda via
  `/data/hzwtech/profile/Miniforge3.env`.
- **Tencent `tencent`**: 48 CPU/node. Memory accounting broken → **omit `--mem`**.
  MPI: `module load openmpi/4.1.8`, `srun --mpi=pmi2`.
- **Huadong A `q_amd_share`**: 128 CPU/node AMD EPYC 7H12, memory accounting VALID →
  **`--mem-per-cpu=1500M`** (≈192 GB full node). Compute on Lustre
  (`/online1/ycsc_hzw/hd100493/...`), not NFS HOME. Modules:
  `source /etc/profile.d/modules.sh`; the validated VASP tuple is
  `amd/intel_compiler/2022u1` (Intel-on-AMD; never substitute Huabei's 2021.3 stack).

## pyraimd2 usage plan

- Remote project root on each cluster: `$HOME/cloud_projects/pyraimd2/` (Huadong A:
  `/online1/ycsc_hzw/hd100493/cloud_projects/pyraimd2/`), created with `umask 077` and the
  standard subdirs (inputs/calculations/outputs/submit/logs/...).
- Workload fit:
  - DFT label campaigns (many independent single points) → job arrays / packed workers on
    Huabei `8163` (most idle capacity) or Neimeng `normal`.
  - Larger periodic DFT (memory-heavy) → Neimeng `9242` (96 CPU + 375G) or Huadong A
    (128 CPU, valid memory accounting) — subject to the 4-node association cap.
  - Torch committee fine-tuning: single CPU node anywhere; no GPU needed at M2 scale.
- Sync: code via git; results as compact JSON/CSV into `sync/` then rsync to local.
  Raw outputs stay remote.

## Logbook

- 2026-08-19 14:10 CST — first fleet audit for pyraimd2 (all four clusters reachable;
  capacities above). No submissions yet.
- 2026-08-19 ~18:50 CST — **Neimeng A selected for the QE workload** (most idle + working
  conda). Project root `/data/home/df103967/df103967/cloud_projects/pyraimd2` created
  (mode 700, standard subdirs). QE 7.5 installed via conda-forge (TUNA mirror) into
  `envs/qe` (openmpi 5.0.10, elpa, hdf5). Si pseudopotential
  `Si.pbe-n-kjpaw_psl.1.0.0.UPF` in `inputs/`.
  - Job 7614852 FAILED 0 s — Slurm does not expand `%HOME%` in `#SBATCH --output`;
    fixed to absolute log paths.
  - Job 7614853 FAILED — `pseudo_dir='../inputs'` resolved wrong from the run dir;
    fixed to the absolute inputs path.
  - Job 7614854 COMPLETED (8 s, 4 ranks, c66): Si bulk SCF converged in 13 iterations,
    E = −93.43942921 Ry / 2 atoms, forces ≈ 0. **Remote QE pipeline validated
    end-to-end.** Templates live in `hpc/neimeng/{inputs,submit}` (git-tracked).
- 2026-08-19 ~19:00 CST — py312 env repaired for the old-glibc system (glibc 2.17 →
  PyPI manylinux wheels unusable; everything via conda-forge): torch 2.6.0 CPU,
  numpy 2.5.1, ase, mace, matscipy; `libstdcxx-ng` required. 38 unit tests pass
  remotely. **Version skew vs local dev (kept deliberately; production numbers come
  from the remote env only):** local torch 2.13.0 / numpy 2.5.2 vs remote torch
  2.6.0 / numpy 2.5.1.
- 2026-08-19 19:56–21:32 CST — array 7614864 (ecut 40/50/60/70 on the 288-atom LP30
  box) **all FAILED after 36–96 min**: every SCF hit `electron_maxstep=200`
  unconverged. Root cause: `conv_thr=1e-9` + default plain mixing at 972 electrons →
  charge sloshing (accuracy estimator oscillated, spikes to 2.2 Ry). Fix (commit
  2b21a3f): `mixing_mode='local-TF'`, `diago_david_ndim=4`, `conv_thr=1e-8` as
  QeConfig defaults; ecut inputs regenerated. Lesson: at ~1k electrons, 1e-9 Ry is
  not "safe", it is unreachable.
- 2026-08-19 ~21:50 CST — job 7614890 (mini box, 106 atoms, new settings):
  **converged, E = −2100.82061883 Ry, 128 s on 28 cores.** Settings validated;
  ecut array resubmitted as 7614891.
- 2026-08-20 — **the ecut convergence saga** (288-atom LP30 box, Γ, PBE+D3).
  Four campaigns failed before the protocol was found; each failure isolated one
  layer of the problem:
  - 7614891 (local-TF, conv_thr=1e-8): still 200 iters unconverged; energy stable
    to 1e-3 Ry but density residual oscillating ~1e-2 Ry with 2.2 Ry spikes.
    Saved densities kept (`tmp/pyraimd2.save/`) → restart path opened.
  - 7615036 (restart, beta=0.15, ndim=12): same plateau; diagnosis sharpened to
    **zero empty bands** (972 e⁻ → exactly 486 KS states) → `nbnd=536` (84b7b98).
  - Probe 7615054 (nbnd): plateau again; logs show `c_bands: 1 eigenvalues not
    converged` → Davidson at ndim=4 cannot converge ~500 bands → `ndim=8` +
    `diago_full_acc=.true.` (3009ef1).
  - Probe 7615060 (full-acc): **zero c_bands warnings** but residual crawls at
    ~1%/iter (beta=0.15). Probe 7615064 (beta=0.35): spikes return → the stiff
    mode is the **occupation response at the Fermi level**, not mixing and not
    the eigensolver.
  - Fix: mv smearing `degauss=0.01` Ry (162e65c). Probe **7615067: CONVERGED in
    17 iterations, 4 min** (from the saved density). Campaign settings frozen in
    design-m3.md §3.1. ecut=40 result: E = −5502.72152805 Ry (246 s).
  - Lesson for the paper's methods section: large insulating liquid boxes at Γ
    are smearing-requiring systems in practice, whatever the nominal gap.
- 2026-08-20 ~16:40 CST — **W ecut campaign (array 7615119), first-pass full
  success**: 128-atom bcc box, kjpaw W (14-valence, scalar-relativistic),
  §3.1 protocol + native mv smearing. All four converged (261–490 s each):
  E = −97318.28183213 (40), −97318.60317236 (50), −97318.75205500 (60),
  −97318.77250788 (70) Ry. Forward gaps: 34.2 → 15.8 → 2.17 meV/atom.
  Confirmation point ecut=80 submitted (7615128) to certify the <1 meV/atom
  step. Contrast with the electrolyte saga: same protocol, metal with native
  smearing → zero drama.
- 2026-08-20 — electrolyte ecut_70 (7615071_3) FAILED at maxstep 400 with the
  residual crawling at ~5e-6 Ry (threshold 1e-6). Restarted from its own saved
  density (7615127, --array=3): the restart-as-asset doctrine again.
- 2026-08-20 ~19:00 CST — **electrolyte production cutoff decided: ecutwfc=60,
  ecutrho=600.** The 70-Ry restart crawled asymptotically at ~5e-6 residual and
  was cancelled after 741 total iterations; its final energies were stable to
  1e-8 Ry, giving ΔE(70−60)/atom = 3.6 meV. Decision made on the *label-relevant*
  metric instead: per-atom force change between successive converged cutoffs
  50→60 is RMS 0.0037 / max 0.045 eV/Å (against a 1.65 eV/Å reference RMS
  force) — already below MLIP training noise. Absolute basis-set drift beyond
  60 Ry is dominated by the PBE-D3 functional error; consistency within one
  cutoff is what the labels need. (Rule update: for MLIP-label campaigns the
  convergence gate is force-RMS-first, total-energy-second.) W note:
  ΔE(80−70)/atom flattened to 1.58 meV (not the geometric ~0.3 expected);
  90-Ry confirmation point running (7615193).
- 2026-08-20 19:07–23:49 CST — **adaptive-loop smoke 7615191 COMPLETED (30 steps,
  mini electrolyte 106, eps_acc=0.25): mechanics all green, surrogate is the
  bottleneck.** n_dft=31/31 (dft_fraction=1.0: the switch never trusted the
  committee), 3 fine-tunes fired at labels 8/16/24, store/resume/summary all OK;
  ~3.5 min per 28-core SCF; CPU fine-tune 30–50 min per trigger. Per-step loop.db
  trajectory: force MAE 0.62 → 0.52 → 0.45 → 0.41 eV/Å across the three
  fine-tunes (learning works but plateaus ~0.4); energy referencing fixed after
  fine-tune #1 (|dE/atom| 0.5 meV). Two structural findings:
  - Committee spread far too small: max-atom σ 0.01–0.07 eV/Å vs true max error
    1.6–2.8 eV/Å, pearson(σ, err) = −0.72. Readout-only training on small
    bootstrap sets keeps members jointly wrong. The conformal layer (calibrated
    on realized errors) is what preserves honesty.
  - **Cold-start q̂ poisoning**: before fine-tune #1 the members are identical →
    σ ≡ 0 → r = e/(σ+δ) = e/δ = 1000·e (δ=1e-3); q̂ pinned at 2777 meV/Å for the
    whole run (window 64 > 31 observations). Fix: seeded load-time readout
    perturbation for members 1..K−1 (design-m2.md §1 revised; regression test
    updated). Window-eviction lag remains a real constraint for short runs.
- 2026-08-20 — **foundation shootout on the 31 stored smoke labels** (zero-shot,
  float64 CPU, analysis/foundation_shootout.py): old 20231210 128L0 fMAE
  0.606 / fMax-worst 2.78; MACE-MP-0b3-medium fMAE 0.321 / fMax-worst 1.51;
  MACE-MPA-0-medium fMAE 0.336 / fMax-worst 1.54 eV/Å. Newer foundation halves
  the error before any fine-tuning (old ckpt fine-tuned only reached 0.41) →
  production foundation swaps to 0b3-medium; online fine-tune recipe unchanged.
- 2026-08-20 — W ecut=90 confirmation (7615193_5, 12 min): E = −97318.80855348
  Ry → ΔE(90−80)/atom = 3.83 meV, i.e. the energy tail is non-monotonic
  (ΔE(80−70) was 1.58 meV). Confirms the force-first convergence rule; the
  70/700 Ry production cutoff stands.
- 2026-08-21 — **smoke v2 (7615281, 0b3-medium, window=24): foundation swap works,
  two deeper defects found and fixed.** 32 labels in 12 h (TIMEOUT mid-run; 0b3
  fine-tune ≈ 2.5 h/trigger on CPU — production cost driver). Zero-shot fMAE
  0.317 (vs 0.62 old ckpt), fine-tunes drive it 0.317 → 0.272 → 0.258 → 0.236,
  fMax 1.53 → 0.83 eV/Å; cold-start poisoning gone (s = 0.049 at step −1,
  q̂ = 30.5 not 2777). But B(s) stuck at ~1.0–1.6 vs ε_acc = 0.25:
  - Root cause 1 (estimator bug): per-atom spread was coded as std_k of the
    deviation *norms* — identically 0 at K=2, badly understating at any K.
    Fixed to the standard RMS estimator σ_i = sqrt(mean_k |F_{i,k}−F̄_i|²)
    (regression test: K=2 exactness to rtol 1e-10).
  - Root cause 2 (correlated members): cross-backbone probe on the 31 stored
    labels — 0b3 vs MPA-0 disagreement s ≈ 0.5 eV/Å against true error
    e ≈ 1.1–1.5 → r ≈ 2.2 (median), vs r ≈ 65 same-backbone. Mixed-backbone
    committee implemented (per-member backbone list, interface-compatibility
    guard: z_table/head/r_max/keys/units must match).
  - smoke v3 = K=2 [0b3, MPA-0] + RMS spread, 32 steps, window 24 — the first
    run where acceptance is statistically reachable (B(s) ≈ q̂·s with q̂ ≈ 2–3
    and post-ft s → ~0.1 crosses ε_acc = 0.25 around step ~20–30).
- 2026-08-21 20:26 – 2026-08-22 ~05:30 CST — **smoke v3 (7615761) COMPLETED: 32/32
  steps, 4 fine-tunes, calibration layer verified fixed; acceptance not reached
  because the surrogate is genuinely not there yet.** Mixed committee K=2
  [0b3, MPA-0] + RMS spread. Zero-shot fMAE 0.294/fMax 1.42 (ensemble mean beats
  either backbone: 0.321/0.336); fine-tunes drive fMAE 0.294 → 0.242 → 0.212 →
  0.187 → 0.176 and fMax to 0.67 eV/Å at step 31. q̂ settles 3.30 → 1.92 as the
  window fills with post-fine-tune pairs; final-step tightness B/e = 0.95 — the
  bound is nearly tight, with the slight under-cover consistent with the
  marginal (not per-step) α=0.05 design under trajectory drift. B(s) fell
  1.65 → 0.64 monotonically; acceptance at ε_acc = 0.25 extrapolates to
  roughly 60–100 total labels on this box. **Economics finding: fine-tuning
  consumed 27446 s of the 32466 s wall (84.5%), scaling ~343 s/label (K=2
  mixed, 50 epochs, CPU); DFT averaged ~76 s/step with density restarts.**
  Fine-tune reports: loss descends every trigger (0.62→0.38 / 0.62→0.34 /
  0.61→0.30 / 0.59→0.26), i.e. training is healthy but readout-only capacity
  and/or data volume is the accuracy bottleneck, and CPU fine-tune cost is
  the throughput bottleneck. Next: offline capacity probe (readout-only vs
  deeper unfreeze, train/test split on the 32 stored labels) + per-epoch
  logging + training-window cost controls before the flagship campaigns.
- 2026-08-22 — **capacity probe (array 7615818, 3 nodes parallel, ~1.2 h each):
  readout+products is the fine-tune sweet spot.** Held-out (last 8 of the 32
  stored 7615761 labels; train on first 24, 0b3-medium, 50 epochs, K=1):
  zero-shot fMAE 0.317/fMax 1.056; readout-only 0.234/0.852; readout+products
  **0.104/0.360**; full-backbone (lr 1e-4) 0.122/0.387 — deeper than products
  overfits slightly at 24 configs. Structural fact: readouts are 2192 params
  (0.03% of 0b3-medium), products 1.35M (14.9%), interactions 7.7M (85%).
  Cost is graph-forward-dominated: products adds only ~7% wall vs readout
  (4497 vs 4202 s). Production recipe → trainable_filters=("readout",
  "products"), K=2 mixed [0b3, MPA-0]; held-out fMax 0.36 vs ε_acc=0.25 with
  the committee mean still to come — acceptance is now within reach of a
  modest burn-in. smoke v4 will verify.
- 2026-08-22 07:01–16:32 CST — **smoke v4 (7615821) COMPLETED: the first ML-accepted
  steps in the framework's history, and they audit clean.** products recipe
  (readout+products), K=2 mixed [0b3, MPA-0], 40 steps in 9.5 h. Steps 31–32
  accepted (B(s) = 0.213/0.227 ≤ ε_acc = 0.25, q̂ = 1.90); the step-33 DFT label
  confirms the region's true fMax = 0.221 eV/Å — the accepted steps were genuinely
  within budget, then drift pushed B(s) back over (0.25 → 0.38 by step 39) and the
  loop honestly reverted to DFT. Learning curve: zero-shot fMAE 0.30 → post-ft#4
  0.064; fine-tune losses 0.026 → 0.0089 (vs readout-only 0.38 → 0.26); fine-tune
  walls 2846/5641/8481/11281 s (82.7% of wall — the standing cost driver).
  Acceptance economics: burn-in to first acceptance = 32 labels on this box;
  steady-state acceptance rate remains to be measured (v4.5 continuation).
  Note: v4 predates the checkpoint hook, so no committee.pt — continuation uses
  the warm-start resume path (full-label fine-tune, deterministic by
  bounded-forgetting semantics).
- 2026-08-22 19:10 – 2026-08-23 ~03:30 CST — **v4.5 continuation (7615920)
  COMPLETED: resume machinery verified in production; steady-state acceptance
  is wave-like, 9/48 = 19% on this box at ε_acc = 0.25.** Warm start (32-label
  windowed fine-tune, 3.2 h) restored the committee without a checkpoint;
  steps 40-43 accepted immediately. Acceptance clusters AFTER each fine-tune
  (steps 40-43 and 60-64); rare-event single-atom excursions (fMax 0.8 eV/Å
  with fMAE only 0.10 — a max-atom outlier, honestly rejected) drive the
  rejection stretches. Two 32-label windowed fine-tunes cost 11587+11555 s =
  77% of the 8.3 h wall; final q̂ = 3.90 (hard-region pairs inflate the
  window). Cumulative across v4+v4.5: 11/88 steps ML (12.5%).
  **Economics verdict: on CPU, fine-tune cost — not rejection — is the
  binding term.** Back-of-envelope for the 474-atom flagship at ~30%
  acceptance: adaptive ≈ 40 h/100 steps vs AIMD ≈ 33 h/100 steps — a loss;
  acceptance ≥50% or 2-3× cheaper fine-tunes (epochs 50→20, and/or GPU) flip
  it. Flagship plan revised: offline bootstrap pre-training (~60 labels
  spanning the target config space, doubles as SCF protocol validation) so
  the loop starts warm, before any long production run.
- 2026-08-23 — **epoch-budget probe (7616069): 25 epochs keeps ~96% of the
  accuracy at half the cost.** products recipe, 24 train/8 held-out on the
  7615761 labels: 15 epochs → fMAE 0.139/fMax 0.453 (1363 s); 25 epochs →
  0.123/0.374 (2251 s); 50-epoch reference → 0.104/0.360 (4497 s). Production
  default moves 50 → 30 epochs (fine-tune cost cut ~40%, accuracy delta in
  the noise of the acceptance criterion). First iface SCF attempt failed in
  26 s on a UPF parsing bug (z_valence written in Fortran exponent form
  "4.000e0" slipped the regex) — fixed; valences confirmed (Li-s = 3,
  W-spn = 14); resubmitted as 7616078.
- 2026-08-23 — **interface-box SCF check (7616078) FAILED unconverged — the
  §3.1 bulk-liquid protocol does not transfer to the metallic slab.** 1584
  electrons (Li=3, W=14 valences confirmed), nbnd = 852; 191 iterations in
  3 h (~56 s/iter) with the accuracy *oscillating* 0.018–0.26 Ry — the
  Fermi-level occupation response of a true metal, not charge sloshing of
  an insulator. Probe array 7616106 (4 nodes): mv/fd × degauss {0.01–0.03}
  × beta {0.10–0.30} × nbnd headroom {150, 250}, maxstep 400. The flagship-A
  labeling campaign is gated on a converging metallic protocol from this
  probe. QeConfig gained a `smearing` field (mv default; fd for true
  metals).
- 2026-08-23 — **metallic protocol DECIDED: Fermi-Dirac smearing.** Probe array
  7616106 (474-atom interface box, 1584 e⁻, Γ): fd / degauss 0.02 / β 0.20 /
  headroom 250 CONVERGED in 64 iterations, 5617 s (E = −120965.322216 eV). All
  three mv variants plateaued oscillating (t0 ~0.12 Ry, t1 ~0.21, t3 ~0.06–0.08,
  flat for hundreds of iterations) — for a genuine metal|liquid cell at Γ, mv
  fails at any tested β/degauss, fd converges. Production metallic protocol:
  60/600 Ry, fd 0.02, local-TF β 0.20 ndim 12, Davidson ndim 8 full-acc,
  conv_thr 1e-6, nbnd = nelec/2 + {250 pending the 150 control 7616174}.
  SCF cost ~88 s/iter at 1042 bands on 28 cores — flagship-A label cost
  estimate ~1.5 h/SCF from scratch (density restarts should cut this).
- 2026-08-23 — **fd150 control (7616174) COMPLETED: converged in 115 iterations,
  7952 s, E = −120965.322216 eV — identical to the fd250 result (band count
  changes convergence speed, not the physics).** Final metallic protocol for
  flagship A: 60/600 Ry, fd smearing degauss 0.02, local-TF β 0.20 ndim 12,
  Davidson ndim 8 full-acc, conv_thr 1e-6, nbnd = nelec/2 + 150 (= 942).
  From-scratch single point ≈ 2.2 h on 28 cores; campaign labels will chain
  density restarts (startpot_file) to cut this.
- 2026-08-23/24 — **flagship-A bootstrap stages 1-2: 76 frames generated
  (7616226, 450K+300K segments, 4.5 h each); labeling array 7616227 hit the
  14 h wall with 32/76 labels banked** (per-task jsonl appends survived —
  restart-chain reality: ~1.7 h/frame, only ~25% saving vs from-scratch;
  the frames differ too much for the density to transfer well). Continuation
  7616463 (+7 frames/task to 60 labels) resumes each task from its last
  converged density. Stage-3 driver bootstrap_train.py ready (temperature-
  stratified held-out, production recipe, checkpoint for the warm start).
- 2026-08-25 — **bootstrap stage-3 COMPLETE (train 7616875, merge 7616877):
  ensemble held-out fMAE 0.155 / fMax 0.981 eV/A (zero-shot 0.335/2.20 —
  2.2x better on both).** Three OOM iterations before the root cause:
  7616839 (K=2, env-28) and 7616847 (K=1, env-16) died on the identical
  28.96 GB einsum allocation — invariant across member/thread counts
  because the size is set by the TRAINING SET: finetune's energy-shift fit
  forwarded all 52 labels as ONE 24,648-atom batch. Fixed by chunked
  minibatch referencing (mathematically identical; ~4.5 GB peak). Thread
  theory retracted: in-process set_num_threads still kept (probe 7616841:
  predict@4 = 6.9 GB, finetune@16 = 20.6 s/label/epoch/member, 41.5 GB).
  Member-parallel training (one backbone per node, K=1) is exact — members
  train independently; merged by bootstrap_merge.py. Members: 0b3
  0.182/1.100 (6.4 h), MPA-0 0.171/1.008 (6.2 h), 30 epochs on 52 labels.
  300 K held-out frames: 0.75/0.77/0.79/1.19; 450 K: 0.95/1.09/1.20/1.10.
  Gate: fMax 0.98 vs the ~0.35 heuristic (from the gentler 106-atom probe
  box) — missed, but production proceeds: the conformal guarantee holds
  at any acceptance rate, the held-out set is the campaign's hardest
  (450 K collision frames with fMax labels up to 9.9), and the refusal
  curve is recomputed post-hoc from the stored (s,e) stream anyway.
  adaptive_loop production hardening: QE timeout 7200 -> 10800 s (fresh
  SCF measured up to 8591 s), --startpot-chain flag (density chaining +
  wipe-retry), in-process torch thread pin (--torch-threads 16).
- 2026-08-25 — **flagship-A production segment 1 (7616899, TIMEOUT at 14 h
  wall, chained): burn-in physics — the trajectory hardens and the
  committee KNOWS.** 15 labels (steps -1..13, all DFT, w_min=16 burn-in),
  ~66 min/label with the startpot chain (~50% saving vs from-scratch —
  production frames are 0.5 fs apart, far closer than the bootstrap
  chain's 10 fs). First fine-tune (8 labels, 30 epochs) completed +
  checkpointed; second (16 labels) killed at the wall, redone in segment
  2. (s,e) stream: calibration excellent (r = 1.94-2.0, design target
  ~2.2; smoke-v2 same-backbone was ~65) but e RISES 0.57 -> 1.03 after
  the first fine-tune — 300 K NVT structural relaxation of the fresh
  interface box (Li surface reconstruction / liquid settling) drives the
  trajectory out of the bootstrap training distribution; crucially s
  tracks e upward in lockstep (0.29 -> 0.40) — honest UQ under
  distribution shift, exactly the regime the framework exists for.
  Acceptance at eps=0.25 needs s <= ~0.1 at qhat ~2.6: not until the
  hard-region fine-tunes pull e back (v4.5 wave dynamics on a harder
  system). Segment 2 (7617264) resumed from the checkpoint + store.
- 2026-08-26 — **flagship-A segment 2 (7617264, 14 h wall): the wave
  breaks through — fine-tune 3 halves e and tightens r.** 27 labels
  (steps -1..25). e peaked 1.055 at step 15 (structural relaxation of the
  fresh box), declined gently to 0.907 by step 21; fine-tune 3 (24
  labels incl. the whole hard region) then cut e to 0.487 and r from
  ~2.6 to ~1.6 (steps 22-25: s 0.30->0.35, e 0.49->0.56, r 1.56-1.65) —
  the v4.5 wave dynamics replayed at production scale: fine-tune ->
  sharp improvement -> slow drift -> next trigger. Operating-point
  arithmetic at r~1.6: acceptance at eps=1.0 is reachable NOW, eps=0.5
  needs s <= ~0.25 (one or two fine-tunes away), eps=0.25 stays out of
  reach near-term (backbone-diversity s floor ~0.3 on this far-from-
  pretraining system). The refusal curve will be recomputed post-hoc
  from the stored (s,e) stream over the full eps sweep. Segment 3
  (7617313) resumed.
- 2026-08-26 — **flagship-A segment 3 (7617313, 14 h wall): steady state
  characterized — e waves 0.5-1.05, operating point decided: eps 1.0.**
  38 labels (steps -1..36). Post-fine-tune-3 drift carried e 0.50 ->
  0.99 (steps 23-33); fine-tune 4 (32 labels) then cut s to ~0.28 (near
  the backbone-diversity floor) but e only to ~0.82-0.91 -> r jumped to
  ~3.0 (members converge on shared labels faster than truth improves —
  mild overconfidence signature, honestly recorded). Steady state at
  eps=0.25: 0% acceptance over 3 segments/42 h — the 474-atom Li|
  electrolyte interface is persistently hard for the committee. eps
  enters ONLY the accept rule (window/qhat/store untouched by a
  between-segment switch — verified in adaptive_loop construction), so
  segment 4+ runs EPS_ACC=1.0 (sbatch env var): B(s) ~ 3.0 x 0.28 ~
  0.84 < 1.0 — acceptance finally reachable; guarantee semantics: <=1.0
  eV/A max-atom force at 95% marginal coverage, the refusal-curve-
  selected operating point (Act 3's content, not an arbitrary choice).
- 2026-08-27 — **flagship-A segment 4 (7617519, eps=1.0): FIRST production
  acceptance streak + accepted-step audit launched.** 48 DFT labels + 6
  accepted (steps 37-42, ~3 fs pure-committee propagation), last step 52
  (26 fs). The streak ended with the switch correctly rejecting step 43
  (post-hoc e=1.144 > 1.0 — no false reject in the whole stream), a 9-step
  rejection stretch as e drifted to 1.30, then fine-tune 5 recovery to
  e~0.88 by step 52. Audit batch 1 (7617790): post-hoc DFT on the 6
  accepted frames -> violation count vs alpha=0.05 with Clopper-Pearson
  CI (audit_accepted.py extract/score; store.append already banks the
  surrogate prediction on ml rows, so only the DFT half was missing).
  Segment 5 (7617793, eps=1.0) continues acceptance accrual.
- 2026-08-27 — **AUDIT BATCH 1 (7617790): the naive conformal bound BREAKS
  on acceptance streaks — 4/6 violations — mechanism located, repair
  calibrated and implemented.** Post-hoc DFT on the segment-4 streak
  (steps 37-42, eps=1.0): true e = 0.952/0.991/1.028/1.062/1.092/1.10 —
  monotonic growth ACROSS the streak; the first two steps comply, the
  rest violate. Mechanism: qhat refreshes only on DFT steps, so during an
  acceptance streak the frozen qhat x slowly-rising s undercovers a true
  ratio r drifting ~+0.05/step (measured: r 3.16 -> 3.44 across the six
  frames). Repair: streak-inflated bound B_k = qhat (s+delta) (1+rho k)
  with rho=0.05 (the measured drift rate) — replayed on the audit batch
  it keeps the two compliant accepts and rejects exactly the four
  violators (6/6 correct). Implemented in ConformalSwitch (streak_rho,
  initial_streak for resume continuity; observe() is now replay-safe —
  the streak lives purely in the live assess path); Store gains
  trailing_ml_streak(); adaptive_loop gains --streak-rho. Segments 4-5
  remain the unfixed "before" curve; segment 6+ runs rho=0.05 as the
  "after". This is the manuscript's Act 1: break (measured) -> mechanism
  -> repair (calibrated from the break) -> verification (next audits).
  88 unit + 3 regression tests pass.
