# Research log — M3 system selection (2026-08-19)

Literature scan behind `docs/design-m3.md` (three parallel scans: validation/benchmarks,
MLIP landscape, interface-cell design). Citations marked [?] need verification before
manuscript use.

## Experimental benchmarks (bulk carbonate electrolyte)

- Landesfeind & Gasteiger, JES 166, A3079 (2019) — the gold-standard transport dataset
  (κ, D±, t⁺, thermodynamic factor vs c, T) for 1 M LiPF₆ EC:DMC / EC:EMC.
- Logan et al., JES (2018), doi:10.1149/2.0981803jes — density/viscosity/conductivity
  tables (Dahn group). 1 m LiPF₆ EC:DMC 30:70: κ = 11.14 mS/cm at 0 °C.
- Hayamizu, J. Chem. Eng. Data 57 (2012) — PFG-NMR species diffusivities; 1 M LiPF₆
  EC/DEC 4:6 at 303 K: D(EC) 3.5, D(DEC) 3.60, D(Li⁺) 1.70, D(PF₆⁻) 2.61 ×10⁻¹⁰ m²/s;
  D_Li < D_PF6 ordering is the force-field discriminator.
- Nyman et al., Electrochim. Acta 53 (2008) — EC:EMC transport; Valøen & Reimers, JES
  152, A882 (2005) [? per-point numbers unverified]; Zugmann et al., Electrochim. Acta 56,
  3926 (2011) — t⁺ method comparison.
- LP30 density 1.289 g/mL at 20 °C [? secondary compilation; verify against primary].
- Solvation: Li⁺–O(carbonyl) peak ~1.9 Å, CN ≈ 4 (Ong et al., JPCB 119, 1535 (2015)).
- Convergence references: Yeh & Hummer, JPCB 108, 15873 (2004) (finite-size D correction);
  Bullerjahn, von Bülow & Hummer, JCP 153, 024116 (2020); Maginn et al., LJCMS 1, 6324
  (2019) (transport best practices). ~50–100 ns for 10% D_Li error bars is our inference
  from estimator theory, not a quoted number.
- Databases: CALiSol-23 (de Blasio et al., Sci. Data 11, 750 (2024)); LiionDB (Wang et
  al., Prog. Energy 4, 032004 (2022)); Electrolyte Genome (Qu et al., CMS 103, 56 (2015)).

## MLIP landscape / motivation evidence

- Gong et al., BAMBOO, Nat. Mach. Intell. (2025), doi:10.1038/s42256-025-01009-7 —
  density MAE 0.01 g/cm³, viscosity dev 17%, conductivity dev 26%; bulk-only,
  non-reactive, needs experimental density alignment. No BAMBOO-2 found as of this scan.
- Wang et al. (Jun Cheng group), Nat. Commun. (2025), doi:10.1038/s41467-025-67982-0 —
  DeePMD universal electrolyte potential, ~2300 solvents/20 salts, committee-gated
  concurrent learning; bulk, non-reactive.
- Magdău et al., npj Comput. Mater. (2023), doi:10.1038/s41524-023-01100-w — bespoke
  GAP/DeePMD/MACE for EC/EMC; reports MACE-MP-0 zero-shot evaporates the liquid without
  dispersion, density ≈0.72 with D3 vs exp ≈1.2 g/cm³ [documented foundation failure].
- Guo et al., arXiv:2601.10938 (2026) — universal potentials on alkali battery kinetics:
  migration-barrier MAE 312 (MACE-MP-0b3) to 638 (M3GNet) meV; best Orb-v3 75–111 meV.
- Ho et al., arXiv:2510.00721 (2025) — conformal calibration on MACE-MP-0, offline; no
  online switching protocol.
- Niu et al., J. Energy Chem. 110, 356 (2025) — on-the-fly MLMD at Li-metal interfaces;
  **read in full before claiming novelty** [could not fetch full text].
- Kundu, Ye, Agarwal & Berkelbach, arXiv:2509.14067 (2025) — MACE-MP-0 fine-tuning for EC
  decomposition on Li; new bent-carbonyl pathway; PBE vs hybrid rates differ up to 9
  orders of magnitude.
- Kulichenko et al., Nat. Comput. Sci. 3, 230 (2023) (UDD-AL); Zaverkin et al., npj
  Comput. Mater. (2024) (uncertainty-biased MD) — method-side neighbors, not batteries.

## Interface / SEI references (tier 2 targets + recipes)

- Leung & Budzien, PCCP 12, 6583 (2010) — canonical AIMD SEI-onset protocol.
- Leung, Soto, Hankins, Balbuena & Harrison, JPCC (2016), arXiv:1605.07142 — Li(100)
  SEI energetics (LEDC 0.22 eV; Li₂CO₃ 1.19→0.66 eV with excess Li); V = Φ/|e| − 1.37 V.
- Camacho-Forero, Smith & Balbuena, JPCC 121, 182 (2017) — interfacial structure vs salt
  concentration (tier-2 comparison data).
- Camacho-Forero & Balbuena, PCCP 19, 30861 (2017) — excess-electron (constant-charge)
  decomposition protocol.
- Bertolini & Balbuena, JPCC 122, 10783 (2018) — ReaxFF Li(100)|electrolyte cell
  (55.2 × 27.4 × 164.8 Å); SEI clusters only after ~1.2 ns.
- Hu et al. (DP-QEq), Nat. Commun. (2025), doi:10.1038/s41467-025-62824-5 — constant-
  potential MLIP MD; dendrite **nucleation** only; test cell 834 atoms; products Li₂CO₃,
  Li₂O, LiF, C₂H₄, CO₂, CO; 200 ps ConstQ NPT volume-relaxation trick adopted here.
- Lai et al. (QMTP), npj Comput. Mater. 11, 121 (2025) — charged (−2e) interface flips
  bond-cleavage order: charge-blind MLIPs miss this (our R3).
- Lai et al. (HAML), npj Comput. Mater. 11, 245 (2025) — AIMD↔MLP hybrid, Maxvol γ-break;
  7.40 → 0.56 h/ps; 112-atom reactive interface cell precedent; documents naive-online
  collapse failure mode.
- Takenaka et al., npj Comput. Mater. (2026), doi:10.1038/s41524-026-02271-y — MLFF SEI
  formation product census matching experiment.
- Debnath et al., JPCA 127, 9178 (2023) — EC ring-opening: PBE-D3 MAD 6.83 kcal/mol vs
  CCSD(T); barrier underestimated >6 kcal/mol (our R2; M06-2X-D3 best practical).
- Builders: Packmol (Martinez et al., JCC 30, 2157 (2009)); ConstP codes for future work:
  MetalWalls (Marin-Laflèche et al., JOSS 2020 [?]), LAMMPS ELECTRODE, DP-QEq
  (github sxu39/DP-QEq).
