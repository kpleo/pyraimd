# Changelog

## 0.3.0

- Present Pyramid as **Python wrapped Ab initio Molecular Dynamics**, an
  extensible framework connecting reference engines, fast interatomic
  potentials, solvers and molecular dynamics.
- Add `pyraimd2.energetics` for directional residual response, empirical force
  forecasts, the directional residual-work coefficient, signed work and
  independent accepted-force checks.
- Add `EnergeticCalculator` and `EnergeticRunner` for reference-triggered
  anchoring and energetic MD decisions, with optional `on_label` adaptation.
- Add generic `AseEngine` and `AseSurrogate` adapters for configured ASE
  calculators, including force-consistent energy and optional stress settings.
- Add a harmonic energetic-loop example and an optional PySCF molecular example.
- Keep the base installation to NumPy and ASE. Provide `mace`, `pyscf`, `all`,
  `dev` and `builders` extras for optional dependencies; require Python 3.12 or
  later.
- Add the `pyramid` command alongside `pyraimd2`, retaining the `pyraimd2` Python
  import and existing switching workflows.
- Replace the overview with the energetic workflow and add architecture and
  method guides.
- Remove obsolete development-stage notes, missing design-document references
  and a historical analysis driver that depended on unpublished run files.
