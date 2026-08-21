# M3 design — flagship demonstrations: certified reactive MD, two event extremes

Status: proposal v2 (supersedes v1 tier structure 2026-08-20; TBG+Li dropped in v1).
Decision record: v2 approved by the owner 2026-08-20. Three changes vs v1:
1. **Narrative inversion.** Bulk LP30 transport is demoted from headline to
   *credibility anchor* (BAMBOO-class bespoke potentials already do bulk transport
   well — redoing it proves nothing). The headline is the part where prior methods
   demonstrably fail: reactive/event-driven dynamics with a statistical contract.
2. **Decisive experiment added** (both flagships): the same trajectory run twice —
   bare committee online MD vs the calibrated supervisor — producing the
   side-by-side "prior art fails / ours works" figure.
3. **Second flagship added**: tungsten displacement-cascade-core physics
   (thermal-spike quench), the fast-violent-event extreme complementing the
   electrolyte's slow-chemistry extreme. One contract, both ends of the event
   spectrum.

## 1. Why these systems showcase the method

The method (Phase B): foundation-potential prior + committee online fine-tuning +
calibrated conformal switching + honest refusal, framed as *runtime assurance*
(Sec. framing in manuscript f2e161f). A flagship must have (a) a documented
failure of prior methods, (b) expensive ground truth so DFT-fallback economics
matter, (c) hard benchmarks. Both chosen systems have all three, and they bracket
the event-time spectrum: picosecond violence (cascade core) vs nanosecond
chemistry (SEI onset).

### Flagship A — Li(100)|electrolyte interface, initial SEI (slow chemistry)

- **Prior-method failure, documented**: MACE-MP-0 without dispersion makes carbonate
  liquid trajectories evaporate; even with D3, density ≈ 0.72 vs exp ≈ 1.2 g/cm³
  (Cambridge electrolyte-MLIP study). Universal potentials fail alkali-ion battery
  kinetics with migration-barrier MAE 312–638 meV (Guo et al., arXiv:2601.10938).
  Naive online MLMD collapses at bond-breaking (HAML paper documents collapse).
- **White space, narrowing**: no published work combines all four elements in any
  domain. Nearest occupants to cite and differentiate: Niu et al., J. Energy Chem.
  110, 356 (2025) (on-the-fly MLMD at Li interfaces — read in full before claiming
  novelty); Kundu et al., arXiv:2509.14067 (2025) (MACE-MP-0 fine-tuning for EC
  decomposition on Li; PBE vs hybrid rates differ up to 9 orders of magnitude —
  motivates calibrated budgeting); Ho et al., npj Comput. Mater. 12, 225 (2026)
  (conformal calibration on MACE-MP-0, but frame selection, no per-step gate, and
  the flexible variant relaxes strict coverage); Carli et al., arXiv:2506.10944
  (2025) (on-the-fly NequIP at a solid-state reactive interface, from scratch, no
  calibration).
- **The signature figure no one else can produce**: during interfacial SEI reactions
  the configurations leave the training distribution → the conformal bound inflates
  → the switch automatically falls back to DFT → fine-tunes → recovers.
  Reactive-event detection *for free*, with a measured coverage guarantee.

### Flagship B — W cascade-core quench (fast violent physics), NEW in v2

- **Prior-method failure, documented**: EAM/pair potentials fitted near equilibrium
  fail on melt physics and defect chemistry inside the cascade core; zero-shot
  foundation potentials face liquid-metal + off-stoichiometric defect double-OOD;
  bespoke MLIPs (GAP-W, Byggmästar et al. PRB 2019 [verify exact ref]) work but
  cost a full bespoke training campaign and carry no certificate. Pure AIMD is
  trapped at ~100 atoms / ps.
- **Our delta, one sentence**: foundation prior + certified switching reaches
  bespoke-class reliability at a fraction of the labels, and every step carries an
  auditable guarantee.
- **The signature figure**: the conformal bound during a spike is a sharp
  picosecond spike of its own — crystal confidence → core-melting refusal spike →
  recovery after quench. The most photogenic bound-inflation plot we can make.

## 2. Structure of each flagship (v2: inverted, three acts + anchor)

### Anchor (not the headline) — bulk validation

- **Electrolyte**: LP30 bulk box (288 atoms, 15 Å). Density 1.289 g/mL, Li–O
  RDF ~1.9 Å / CN ≈ 4, D_Li < D_PF6 ordering vs Hayamizu. Purpose: prove labels
  and the closed loop are trustworthy in this chemistry. NOT a BAMBOO-beating
  exercise.
- **Tungsten**: vacancy formation energy (~3.2 eV), SIA migration, melting point
  (3695 K) from the labeled loop. Same purpose.

### Act 1 — decisive experiment (the v2 centerpiece)

Same initial condition, same committee, two runs: (a) bare committee online MD
(calibration layer off); (b) the calibrated supervisor. Expected: (a) degrades at
the first event (SEI: bond-breaking collapse à la HAML; W: melt-core distortion);
(b) bound inflates → DFT fallback → fine-tune → physical recovery. One figure,
two traces: prior art vs this work. This is the cheapest and most persuasive
figure in the paper.

### Act 2 — benchmark distributions (not single-trajectory anecdotes)

- **SEI**: initial product census (LiF, Li₂CO₃, Li₂O, ROCO₂Li, ROLi, C₂H₄, CO₂, CO)
  vs QMTP/DP-QEq/XPS inventories over 3–5 independent trajectories; reaction-onset
  timescales qualitative (PBE-D3 barrier caveat, §3).
- **W**: N = 10–20 independent spikes; surviving Frenkel-pair distribution vs
  NRT/arc-dpa; defect-cluster census vs GAP-W literature and TEM; liquid-W
  structure factor during the spike vs experiment. Report distributions.

### Act 3 — refusal curve (the operating curve)

ε_acc sweep on each production system: engine fraction and α̂ vs ε_acc over ≥3
budgets. Two flagships, two curves, same contract → the operating curve is a
general product of the method.

### Build specs and claim boundaries (preserved from v1 + W additions)

- **Interface cell (flagship A, v2.1 anchored)**: Li(100) 3×3 slab, **4 layers**
  = 36 Li (10.53 × 10.53 Å, bottom layer fixed; a = 3.51 Å) + electrolyte
  21 EC + 17 DMC + 3 LiPF₆ (438 atoms) in ~4.7 nm → **~474 atoms total**, box
  ≈ 10.5 × 10.5 × 51 Å, PBC (two interfaces per slab). Scale-up variant
  (~800 atoms): Li(100) 4×4 ×4 layers = 64 Li + 36 EC + 28 DMC + 5 LiPF₆.
  Build: Packmol (tol 2.0 Å) → classical-FF bulk NPT pre-equilibration to
  measured density (UFF/Forcite precedent: Leung 2016; APPLE&P: Borodin & Smith
  JPCB 110, 4971 (2006)) → stack → NVT 300 K, dt = 0.5 fs (precedents use 1 fs;
  ours is the conservative direction); adopt DP-QEq's anisotropic ConstQ NPT
  stage (200 ps, 1 bar xy / 100 bar z) to absorb reaction-induced cavities.
  Seeding AIMD snapshots at **450 K** (Leung & Budzien PCCP 2010; Leung et al.
  JPCC 2016 — both 450 K "to avoid EC freezing", NOT 300/400 K).
- **Density anchor correction (v2.1)**: the 1.289 g/mL value could not be traced
  to a primary source; cite 1.28 g/cm³ (1 M LiPF₆ EC:DMC 1:1, Dinh-Nguyen et al.
  JES 2016, DOI 10.1149/2.0771605jes) or the measured simulation value.
- **Cascade cell (flagship B)**: bcc W, 7×7×7 unit cells = 686 atoms (fallback
  6×6×6 = 432 if label cost overruns), Γ point. Thermal-spike protocol:
  instantaneous ~8000 K velocity heating of a central core region (rest 300 K),
  NVE quench into the matrix, track to ~50–200 ps. Rationale: the full cascade
  needs 10⁵–10⁶ atoms (unlabelable); the spike quench isolates exactly the core
  physics where trust fails, at a DFT-labelable size. This is a focus, not a
  compromise.
- **Explicitly NOT claimed**: dendrite growth/morphology (µs+), steady-state SEI,
  rates/barriers (PBE-D3 caveat), voltage dependence (no potentiostat; neutral
  open-circuit interface only); full-cascade PKA statistics at engineering
  energies (box-size limit); Fe cascades (ferromagnetism doubles label cost —
  deferred).
- **Benchmark anchors for W** (verify each before manuscript use): NRT model
  N_F = 0.8·E_D/(2E_d) and arc-dpa corrections; W E_d ≈ 40–90 eV directional;
  vacancy formation ≈ 3.2 eV; melting 3695 K; GAP-W cascade database
  (Byggmästar et al. PRB 2019 [verify]); Wigner-Seitz defect analysis to be
  built (the one new analysis tool).
- **W label protocol**: pslibrary kjpaw PAW W pseudo + ecut convergence campaign
  (W is metallic — mv smearing native; §3.1 protocol applies directly). Estimated
  3–8 min/label for 686 atoms on 28 cores; 30–60 labels per spike run → 3–10 h
  per run; N = 10–20 spikes + controls ≈ days of Neimeng-A queue, parallel with
  the electrolyte campaigns. MACE inference for production goes GPU (H20) when
  powered; labels stay CPU.

### Parameter provenance (v2.1 literature audit, 2026-08-20)

Every design parameter was audited against the literature by two dedicated
scans; verdicts: ✅ strong anchor / 🟡 precedent with caveats / ❌ none —
self-proof required. Full audit trails are in the session record; the
load-bearing entries:

**Flagship A (electrolyte interface)**
- ✅ Li(100): Leung JPCC 2016 (DOI 10.1021/acs.jpcc.5b11719), Bertolini &
  Balbuena JPCC 2018 (10.1021/acs.jpcc.8b03046), Kundu arXiv:2509.14067,
  DP-QEq Nat. Commun. 16:7379 (2025) all use (100)/(001).
- 🟡→fixed Slab thickness: precedents are 4–6 layers (Kundu 4, DP-QEq 6);
  v1's 3 layers was too thin → **now 4 layers (36 Li, ~474 atoms)** + SI
  layer-convergence table (adsorption/work function vs 3/4/5 layers).
- ✅ Lateral 10.5 Å (Kundu used 10.13 Å 3×3); ✅ dual-interface PBC (Leung
  2010/2016, DP-QEq); ✅ NVT 300 K (HAML); dt=0.5 fs conservative vs the
  1 fs precedents (fine, one-sentence note).
- ✅ Packmol tol 2.0 Å (default; arXiv:2606.09422 precedent).
- ✅ Classical pre-equilibration: UFF/Forcite (Leung 2016), APPLE&P (Borodin
  & Smith JPCB 110, 4971 (2006)); DP-QEq pre-samples 200 ps ReaxFF.
- ✅ DP-QEq ConstQ stage verified verbatim: anisotropic NPT 200 ps, 1 bar xy /
  100 bar z (absorbs reaction cavities) — cite it as anisotropic, not
  "volume relaxation".
- ⚠️→fixed Seeding temperature: Leung AIMD is **450 K** ("to avoid EC
  freezing"), NOT 300/400 K → seed at 450 K and say so.
- 🟡 Density: 1.289 g/mL untraceable → cite 1.28 g/cm³ (Dinh-Nguyen et al.
  JES 2016, DOI 10.1149/2.0771605jes).
- ❌→mitigated Salt statistics: 3 LiPF₆ over two interfaces makes LiF
  formation single-event → SI salt-count sensitivity (3 vs 6) + scope LiF
  claims to "consistent with QMTP/Takenaka trends".
- ✅ Product census: QMTP (npj Comput. Mater. 11:121, 2025) observed LiF,
  Li₂CO₃, ROCO₂Li, ROLi, CO₂, C₂H₄ within 44 ps; CO first predicted by Leung
  & Budzien 2010; Li₂O innermost layer thermodynamics in Leung 2016. LEDC is
  a LATER-stage product — keep it out of the "initial" claims.
- 🟡 Trajectory count: Leung 3 restarts / QMTP 1+1 / Kundu dozens with error
  bars → report Poisson errors; wording "observational, not rate statistics".
- ✅ PBE+D3: Kundu used exactly QE+PBE+D3+PAW; DP-QEq used PBE+D3(BJ).
  Mandatory caveat: Debnath JPCA 127, 9178 (2023) + Kundu quantify barrier
  underestimation up to 9 orders of magnitude in rates → kinetic claims
  stay qualitative; plan ωB97X-D3 cluster spot-checks on 1–2 elementary
  steps (EC ring-opening, PF₆⁻ defluorination) to show the product sequence
  is unchanged.

**Flagship B (W cascade core)**
- 🟡 Thermal-spike protocol: 40-year lineage — Diaz de la Rubia, Averback,
  Benedek & King, PRL 59, 1930 (1987) showed heat-spike deposition reproduces
  real-cascade defect production; 2T-MD line: Caro & Victoria PRA 40, 2287
  (1989), Zarkadoula JPCM 27, 135401 (2015). **Naming discipline: always
  "thermal-spike surrogate of the cascade core", never "cascade simulation".**
- 🟡 Peak temperature: report as **energy density** (~1 eV/atom at 8000 K),
  inside the canonical 1–2 eV/atom cascade-core range (Averback 1987); give
  the core atom count so the number is checkable.
- ❌→self-prove 686-atom box: no sub-2000-atom cascade box exists in the
  literature → three self-proofs, all now part of the plan:
  (a) **PKA-vs-spike validation experiment** (same committee: one real
  200–300 eV PKA run vs the energy-equivalent spike; compare surviving FP) —
  this turns the weakest point into a figure;
  (b) finite-size convergence 5³/7³ (+4³) on surviving FP;
  (c) shock-reflection check: pressure/KE spatiotemporal decay plot (2.2 nm
  box, longitudinal round trip ~0.4 ps at 5.2 km/s — a REAL risk, must be
  shown harmless or the claim retreats to relative defect statistics).
- ✅ Quench window 50–200 ps: classical convention is 10–30 ps (Malerba JNM
  351, 28 (2006); Nordlund Nat. Commun. 9, 1084 (2018)); we are longer.
- ✅ N=10–20: SCK-CEN convention 5–20 per energy; Warrier et al.
  arXiv:2402.00359 (3500 cascades) sets the classical upper bar — add a
  statistical-power sentence; ours is the first DFT-grade statistics at all.
- ✅ NRT: Norgett, Robinson & Torrens, Nucl. Eng. Des. 33, 50 (1975);
  arc-dpa W fit: Nordlund et al., Nat. Commun. 9, 1084 (2018) (Fig. 3b) —
  note the injected-energy→damage-energy mapping must be stated.
- ✅ W anchors: a0 = 3.1648 Å, T_m = 3695 K (CRC/Lassner & Schubert);
  E_vf^exp = 3.6±0.2 eV; E_vf^PBE = 3.327 eV (Muzyk et al. PRB 84, 104115
  (2011)) — we report our own kjpaw values against these.
- 🟡→opportunity E_d: ASTM E521 average 90 eV; directional MD values exist
  (Banisalman et al. 2017) but **no DFT-resolved directional E_d for W
  exists** → add the "<100> E_d from DFT fallback" mini-experiment as a
  validation result (weakness → selling point).
- ✅ GAP-W bar: Byggmästar et al. PRB 100, **144105** (2019) (corrects the
  144135 typo); cascade DB: Warrier arXiv:2402.00359; latest: Byggmästar
  arXiv:2503.02710 (2 MeV, 10⁹ atoms). Compare TRENDS (FP vs injected-energy
  exponent, cluster-size distribution shape), not absolute FP counts.
- ✅ Wigner-Seitz analysis: method Nordlund et al. PRB 57, 7556 (1998);
  implementation Stukowski MSMSE 18, 015012 (2010) (OVITO).
- 🟡 W kjpaw precedent is thin (thesis-level only) → SI validation appendix:
  our kjpaw W a0 / bulk modulus / E_vf / phonon vs experiment, plus a few
  cross-check energies against VASP-PAW settings.
- 🔴 NEW calibration requirement (from the audit): the calibration window
  must explicitly include 4000–10000 K liquid/coexistence configurations and
  coverage must be reported on that subset; report DFT-fallback fraction vs
  spike temperature (if fallback ≈ 100% at 8000 K, the honest statement is
  that the surrogate contributes only outside the spike — know this number
  before writing).

## 3. DFT level (decided)

**PBE+D3** for all labels (QE: `vdw_corr='grimme-d3'`). Rationale: dispersion is required
for organic-liquid density; PBE+D3 is the interface-literature standard (e.g. Nat. Commun.
2025 methods: PBE+D3, 450 eV); consistent with Kundu et al.'s PBE-D3 label level.
Known caveat (state in the paper): PBE-D3 underestimates the EC ring-opening barrier by
>6 kcal/mol vs CCSD(T) (Debnath et al., JPCA 127, 9178 (2023)) → all kinetic claims stay
qualitative; optionally Δ-learn 3–5 cluster reaction profiles at M06-2X-D3 later.
Convergence tests before any campaign: ecutwfc/ecutrho and k-mesh on the Li slab + a bulk
electrolyte snapshot (Γ-only expected to suffice at 465 atoms; verify energy/force diffs).

### 3.1 SCF protocol for the 288-atom liquid box (found empirically, 2026-08-19/20)

The LP30 bulk box is pathological for naive SCF settings (four failed campaigns,
arrays 7614864/7614891/7615036 + probes 7615054/7615060/7615064 — see hpc.md
logbook). The converged protocol, now the campaign default:

- `nbnd = 536` (972 electrons → 486 occupied + 50 empty; zero empty bands leaves
  the HOMO unconstrained and the Davidson subspace fragile)
- `diago_david_ndim = 8` + `diago_full_acc = .true.` (else `c_bands: eigenvalues
  not converged` → eigenvector noise floors the density residual at ~1e-2 Ry)
- **`occupations='smearing'`, mv, `degauss=0.01` Ry** — the decisive fix: the
  stiff mode is the occupation response at the Fermi level, not mixing; with
  smearing the restarted SCF converged in **17 iterations / 4 min**
- `mixing_mode='local-TF'`, `mixing_beta=0.3`, `mixing_ndim=12`
- `conv_thr=1e-6` Ry (the accuracy norm is extensive: 1e-6 Ry on 288 atoms is
  per-atom tighter than 1e-8 Ry on the 2-atom Si validation cell)
- Restart from a saved density (`startingpot='file'`) whenever a previous
  partial run exists — an unconverged 200-step SCF is an asset, not waste.

These live in `QeConfig` (fields) + `make_ecut_series.py` (campaign values);
`adaptive_loop.py` exposes `--nbnd` / `--conv-thr` for the production box.

### 3.2 Production cutoffs (decided 2026-08-20, force-first rule)

Convergence gate for MLIP-label campaigns: per-atom force change between
successive cutoffs first (label-relevant), total-energy drift second.

- **Electrolyte (LP30 288, and the 465-atom interface by extension)**:
  ecutwfc=60, ecutrho=600. Evidence: ΔF RMS 0.0037 / max 0.045 eV/Å at
  50→60 Ry (ref RMS force 1.65 eV/Å); ΔE/atom 15.0 → 3.8 → 3.6 meV
  (40→50→60→70); 70-Ry SCF stalls at 5e-6 residual (energy stable to 1e-8
  Ry) — tight density closure is asymptotically expensive here and not
  needed for labels.
- **Tungsten (bcc 128)**: ecutwfc=70, ecutrho=700. Evidence: ΔF RMS
  0.0002 eV/Å at 70→80 Ry; ΔE/atom 34.2 → 15.8 → 2.2 → 1.6 → 2.2 meV
  (40→…→90) — absolute drift ~2 meV/atom is uniform and cancels in all
  defect energetics.

## 4. Labeling and compute plan (matches fleet reality)

- Labels from small cells only (the ~465-atom interface cell and the ~300-500 atom bulk
  box), QE on Neimeng A (`normal` 28-core nodes; `9242` 96-core/375G for the bigger cells);
  batch campaigns via job arrays; 4-node association cap respected.
- Committee fine-tuning: CPU nodes; production MD: distilled single model for long runs,
  committee for the switching signal (design decision D1, open).
- H20 GPU (when powered on): accelerates MACE inference ~10–100×; the ns-scale tier-1
  transport runs are the main GPU candidate. CPU fallback: few-ns/day on a 48-core node —
  feasible but slow; plan GPU time if available.

## 5. Risks and mitigations (from the literature scan)

- R1 surrogate blows up at reactive events → that IS the demo (conformal fallback);
  additionally seed training with short 400–450 K AIMD snapshots so reactive configs are
  in-domain early (Leung protocol).
- R2 PBE-D3 labels quantitatively wrong for kinetics → qualitative claims only; small
  Δ-benchmark vs M06-2X-D3/CCSD(T) cluster profiles.
- R3 charge-blind local MLIP misrepresents interfacial charge transfer (QMTP showed −2e
  changes decomposition order) → scope to neutral open-circuit interface and say so;
  ConstP/QEq is future work.
- R4 metallic-cell label cost & finite-size voltage artifacts → fixed cell size, Γ-point
  verified against 2×2×1 (Leung: barrier shift <0.03 eV), short diverse snapshots.
- R5 construction artifacts (Packmol overlaps, unrelaxed density, single-trajectory
  anecdotes) → bulk-NPT pre-equilibration to measured density, 3–5 independent
  trajectories, report distributions.
- R6 tier-1 transport needs ns × committee — too slow on CPU → distill committee to a
  single model for production; verify distillation preserves the switching signal.
- R7 (v2, W): kjpaw W pseudopotential convergence untested (semicore states may demand
  high ecutrho) → small ecut campaign before any production; 432-atom fallback box.
- R8 (v2, W): thermal-spike ≠ full PKA cascade — a reviewer may push. Mitigation:
  frame B as cascade-CORE physics (stated explicitly), cite the spike-quench
  literature precedent, and scope claims to core melt/defect formation.
- R9 (v2): two flagships dilute effort → A (electrolyte) remains primary; B runs on
  spare queue capacity and shares 100% of the platform code path.

## 6. Acceptance criteria for M3 (v2)

- Anchor A: density within 0.02 g/cm³ and D_Li/D_PF6 ordering correct vs experiment;
  measured coverage and DFT-call count reported.
- Anchor B: W vacancy formation energy and melting point within DFT-literature
  tolerance; same accounting.
- Act 1 (both): bare-committee control run degrades at the first event while the
  supervised run completes with measured α̂ ≤ α + 0.03.
- Act 2 A: initial SEI product set overlaps the QMTP/DP-QEq/XPS inventory; ≥1
  reactive event automatically caught by the conformal bound (bound-inflation plot).
- Act 2 B: surviving Frenkel-pair distribution over ≥10 spikes, shape consistent
  with NRT/arc-dpa expectation; defect-cluster census reported.
- Act 3 (both): engine fraction and α̂ vs ε_acc over ≥3 budgets.
- Everything seeded/restartable; decision logs + stores archived.

## 7. Open decisions before build

- D1: production-MD model = distilled single vs full committee (speed vs signal).
- D2: interface cell size — start with ~465 atoms (cheap) and scale to ~800 for the
  final figure, or build 800 directly.
- D3: electrolyte = EC:DMC 1:1 (LP30-class, maximal benchmark data) — confirm vs EC:EMC.
