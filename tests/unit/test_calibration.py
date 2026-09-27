"""calibrate-scale / fit_force_scale: exact closed form, refusal classes,
optimality on non-collinear data, and the CLI's NPZ/JSON protocol.

All analytic teaching data — no backend is ever constructed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from pyraimd2.cli import main as cli_main
from pyraimd2.surrogate import (
    CalibrationError,
    ScaledSurrogate,
    fit_force_scale,
)

FAST = np.array([[[0.10, 0.00, 0.00], [0.00, -0.20, 0.00]],
                 [[0.05, 0.05, 0.05], [0.30, 0.00, -0.10]],
                 [[-0.15, 0.20, 0.05], [0.00, 0.10, 0.25]]])  # (3, 2, 3)


def test_exact_closed_form_on_constructed_data() -> None:
    reference = 1.25 * FAST
    fit = fit_force_scale(reference, FAST)
    assert fit.schema_version == 1
    assert fit.status == "ok"
    assert fit.scale == pytest.approx(1.25, rel=1e-15)
    assert fit.denominator == pytest.approx(float((FAST**2).sum()))
    assert fit.numerator == pytest.approx(1.25 * float((FAST**2).sum()))
    assert (fit.n_frames, fit.n_atoms) == (3, 2)
    assert fit.force_unit == "eV/angstrom"
    # exact collinearity: the after-scaling training residual vanishes
    assert fit.force_residual_rms_after_eV_A == pytest.approx(0.0, abs=1e-15)
    expected_before = float(np.sqrt(((reference - FAST)**2).sum() / (3 * 2)))
    assert fit.force_residual_rms_before_eV_A == pytest.approx(
        expected_before)


def test_single_frame_single_atom_accepted() -> None:
    fit = fit_force_scale([[[1.0, 2.0, 3.0]]], [[[0.5, 1.0, 1.5]]])
    assert fit.scale == pytest.approx(2.0)
    assert (fit.n_frames, fit.n_atoms) == (1, 1)


def test_refusal_classes() -> None:
    # broadcastable-but-different shapes
    with pytest.raises(CalibrationError, match="shape"):
        fit_force_scale(np.ones((2, 2, 3)), np.ones((2, 3)))
    with pytest.raises(CalibrationError, match="one exact shape"):
        fit_force_scale(np.ones((2, 2, 3)), np.ones((3, 2, 3)))
    # complex dtype
    with pytest.raises(CalibrationError, match="complex"):
        fit_force_scale(np.ones((1, 1, 3)) + 1j, np.ones((1, 1, 3)))
    # non-numeric dtype
    with pytest.raises(CalibrationError, match="real numeric"):
        fit_force_scale(np.array([[["a", "b", "c"]]]), np.ones((1, 1, 3)))
    # wrong dimensionality
    with pytest.raises(CalibrationError, match="shape"):
        fit_force_scale(np.ones((2, 3)), np.ones((2, 3)))
    # empty axes
    with pytest.raises(CalibrationError, match="n_frames >= 1"):
        fit_force_scale(np.ones((0, 2, 3)), np.ones((0, 2, 3)))
    with pytest.raises(CalibrationError, match="n_atoms >= 1"):
        fit_force_scale(np.ones((1, 0, 3)), np.ones((1, 0, 3)))
    # non-finite values
    with pytest.raises(CalibrationError, match="finite"):
        fit_force_scale(np.full((1, 1, 3), np.nan), np.ones((1, 1, 3)))
    with pytest.raises(CalibrationError, match="finite"):
        fit_force_scale(np.ones((1, 1, 3)), np.full((1, 1, 3), np.inf))
    # vanishing denominator
    with pytest.raises(CalibrationError, match="denominator"):
        fit_force_scale(np.ones((2, 2, 3)), np.zeros((2, 2, 3)))
    # non-positive scale (anti-correlated forces)
    with pytest.raises(CalibrationError, match="positive"):
        fit_force_scale(-FAST, FAST)
    # overflowed products: 1e200^2 exceeds float64
    huge = np.full((1, 1, 3), 1e200)
    with pytest.raises(CalibrationError, match="overflow"):
        fit_force_scale(np.full((1, 1, 3), 1e200), huge)


def test_orthogonality_and_optimality_on_non_collinear_data() -> None:
    # F_ref is NOT a scalar multiple of F_fast (a deterministic in-plane
    # rotation on one frame), so the LS fit has a genuine residual
    reference = 1.25 * FAST.copy()
    reference[1, 0] = [-0.05, 0.05, 0.10]
    fit = fit_force_scale(reference, FAST)
    # the LS residual is exactly orthogonal to F_fast (normal equation)
    residual = reference - fit.scale * FAST
    assert float((residual * FAST).sum()) == pytest.approx(0.0, abs=1e-12)
    # and alpha minimizes the residual: slightly perturbed scales do worse
    for factor in (1 - 1e-6, 1 + 1e-6):
        perturbed = float(np.sqrt(((reference - fit.scale * factor * FAST)
                                   ** 2).sum() / (3 * 2)))
        assert perturbed > fit.force_residual_rms_after_eV_A
    # the fit genuinely improved on the unscaled model
    assert fit.force_residual_rms_after_eV_A < \
        fit.force_residual_rms_before_eV_A


def test_scaled_wrapper_reproduces_fitted_alpha() -> None:
    reference = 1.25 * FAST
    fit = fit_force_scale(reference, FAST)

    class ToyBase:
        def predict(self, atoms):
            from pyraimd2.surrogate.base import SurrogatePrediction
            return SurrogatePrediction(
                energy=0.5, forces=np.array(FAST[0]), stress=None,
                uncertainty=np.full(len(atoms), 0.1))

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
    scaled = ScaledSurrogate(ToyBase(), fit.scale)
    prediction = scaled.predict(atoms)
    np.testing.assert_allclose(prediction.forces, fit.scale * FAST[0])
    assert prediction.energy == pytest.approx(fit.scale * 0.5)


# ---------------------------------------------------------------------------
# CLI protocol


def _write_pairs(path: Path, reference, fast, *, frame_ids=None,
                 reference_id="toy-reference", fast_model_id="toy-fast",
                 force_unit="eV/angstrom") -> Path:
    n_frames = len(reference)
    if frame_ids is None:
        frame_ids = [f"frame-{index:03d}" for index in range(n_frames)]
    np.savez(path, reference_forces_eV_A=np.asarray(reference),
             fast_forces_eV_A=np.asarray(fast),
             frame_ids=np.asarray(frame_ids),
             reference_id=np.asarray(reference_id),
             fast_model_id=np.asarray(fast_model_id),
             force_unit=np.asarray(force_unit))
    return path


def test_cli_writes_report_and_refuses_overwrite(tmp_path, capsys) -> None:
    pairs = _write_pairs(tmp_path / "pairs.npz", 1.25 * FAST, FAST)
    output = tmp_path / "scale.json"
    code = cli_main(["calibrate-scale", "--pairs", str(pairs),
                     "--output", str(output)])
    out = capsys.readouterr().out
    assert code == 0
    assert "scale = 1.25" in out
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["status"] == "ok"
    assert report["scale"] == pytest.approx(1.25, rel=1e-15)
    assert report["n_frames"] == 3 and report["n_atoms"] == 2
    assert report["force_unit"] == "eV/angstrom"
    assert report["reference_id"] == "toy-reference"
    assert report["fast_model_id"] == "toy-fast"
    assert report["frame_ids"] == [f"frame-{i:03d}" for i in range(3)]
    # provenance: basename + content hash, never an absolute path
    assert report["input"]["file"] == "pairs.npz"
    assert "/" not in report["input"]["file"]
    assert report["input"]["sha256"] == hashlib.sha256(
        pairs.read_bytes()).hexdigest()
    assert str(tmp_path) not in output.read_text()

    # an existing output is kept; a refused overwrite never corrupts it
    original = output.read_bytes()
    code = cli_main(["calibrate-scale", "--pairs", str(pairs),
                     "--output", str(output)])
    assert code == 2
    assert "--force" in capsys.readouterr().err
    assert output.read_bytes() == original
    code = cli_main(["calibrate-scale", "--pairs", str(pairs),
                     "--output", str(output), "--force"])
    assert code == 0
    assert json.loads(output.read_text())["scale"] == pytest.approx(1.25)


def test_cli_protocol_refusals(tmp_path, capsys) -> None:
    valid = _write_pairs(tmp_path / "valid.npz", 1.25 * FAST, FAST)

    def expect_refusal(pairs: Path, needle: str) -> None:
        code = cli_main(["calibrate-scale", "--pairs", str(pairs),
                         "--output", str(tmp_path / "out.json")])
        assert code == 2
        assert needle in capsys.readouterr().err
        assert not (tmp_path / "out.json").exists()

    expect_refusal(tmp_path / "missing.npz", "cannot be read")
    (tmp_path / "junk.npz").write_bytes(b"not an archive")
    expect_refusal(tmp_path / "junk.npz", "not a readable")
    _write_pairs(tmp_path / "dup.npz", 1.25 * FAST, FAST,
                 frame_ids=["a", "a", "b"])
    expect_refusal(tmp_path / "dup.npz", "unique")
    _write_pairs(tmp_path / "short.npz", 1.25 * FAST, FAST,
                 frame_ids=["a", "b"])
    expect_refusal(tmp_path / "short.npz", "frames")
    _write_pairs(tmp_path / "unit.npz", 1.25 * FAST, FAST,
                 force_unit="kcal/mol")
    expect_refusal(tmp_path / "unit.npz", "eV/angstrom")
    _write_pairs(tmp_path / "empty-id.npz", 1.25 * FAST, FAST,
                 reference_id="")
    expect_refusal(tmp_path / "empty-id.npz", "non-empty")
    np.savez(tmp_path / "nokey.npz", reference_forces_eV_A=FAST)
    expect_refusal(tmp_path / "nokey.npz", "lacks the required")
    # a failed run never creates (or corrupts) the output
    expect_refusal(tmp_path / "dup.npz", "unique")
    assert not (tmp_path / "out.json").exists()
    assert valid.is_file()


def test_npy_container_refused_cleanly(tmp_path, capsys) -> None:
    # a valid NumPy file of the wrong container type (.npy, not .npz):
    # a clear input-format error, exit 2, no traceback, no output file
    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.zeros((1, 1, 3)))
    output = tmp_path / "out.json"
    code = cli_main(["calibrate-scale", "--pairs", str(wrong),
                     "--output", str(output)])
    captured = capsys.readouterr()
    assert code == 2
    assert "not an .npz archive" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_preexisting_tmp_file_never_touched(tmp_path, capsys) -> None:
    # the user's own scale.json.tmp survives a successful write, a --force
    # overwrite and a refused overwrite — byte-identical
    pairs = _write_pairs(tmp_path / "pairs.npz", 1.25 * FAST, FAST)
    user_tmp = tmp_path / "scale.json.tmp"
    user_tmp.write_bytes(b"the user's own file\n")
    user_tmp_hash = hashlib.sha256(user_tmp.read_bytes()).hexdigest()
    output = tmp_path / "scale.json"

    code = cli_main(["calibrate-scale", "--pairs", str(pairs),
                     "--output", str(output)])
    assert code == 0
    assert json.loads(output.read_text())["scale"] == pytest.approx(1.25)
    assert hashlib.sha256(user_tmp.read_bytes()).hexdigest() == user_tmp_hash

    code = cli_main(["calibrate-scale", "--pairs", str(pairs),
                     "--output", str(output), "--force"])
    assert code == 0
    assert hashlib.sha256(user_tmp.read_bytes()).hexdigest() == user_tmp_hash

    capsys.readouterr()
    code = cli_main(["calibrate-scale", "--pairs", str(pairs),
                     "--output", str(output)])   # exists, no --force
    assert code == 2
    assert "--force" in capsys.readouterr().err
    assert hashlib.sha256(user_tmp.read_bytes()).hexdigest() == user_tmp_hash
    assert {p.name for p in tmp_path.iterdir()} == {
        "pairs.npz", "scale.json", "scale.json.tmp"}  # no stray temps


def test_documented_example_end_to_end(tmp_path) -> None:
    example = Path(__file__).parents[2] / "examples" / "calibrate_scale"
    pairs = tmp_path / "pairs.npz"
    generated = subprocess.run(
        [sys.executable, str(example / "make_pairs.py"), "--output",
         str(pairs)], capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stderr[-500:]
    output = tmp_path / "scale.json"
    code = cli_main(["calibrate-scale", "--pairs", str(pairs),
                     "--output", str(output)])
    assert code == 0
    report = json.loads(output.read_text())
    assert report["scale"] == pytest.approx(1.25, rel=1e-12)
    assert report["force_residual_rms_after_eV_A"] == pytest.approx(
        0.0, abs=1e-12)
    assert report["n_frames"] >= 2 and report["n_atoms"] >= 2
