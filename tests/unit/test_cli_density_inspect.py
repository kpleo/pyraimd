"""``pyramid density inspect RUN_DIR``: read-only CLI over the
existing plan_density_reclaim decisions.

Covers the normal keep/reclaim-candidate mix, a corrupt generation
held, a publish-tmp leftover held, a genuinely never-attached
generation held, expected read failures at the command boundary
(symlinked registry, unreadable file, invalid UTF-8 — readable stderr,
non-zero exit, no traceback, nothing written), a missing run directory
(readable usage error, nothing created), and before/after byte-identity
proofs that inspections changed nothing on disk.  Fixture construction
reuses the same registry helpers as test_restart_registry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pyraimd2.cli import main as cli_main
from pyraimd2.runtime import restart as rr
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.runtime.restart import publish_density_generation


def _source(run_dir: Path, name="eval-000000") -> Path:
    attempt = run_dir / "calculations" / name
    save = attempt / "tmp" / "pyraimd2.save"
    save.mkdir(parents=True, exist_ok=True)
    (save / "charge-density.dat").write_bytes(b"density-v1")
    (save / "data-file-schema.xml").write_text("<xml/>")
    return attempt


def _prov(request_id: str) -> dict:
    return {"run_id": "run",
            "attempt": {"request_id": request_id, "attempt_id": "attempt-1"},
            "reference_fingerprint": "qe-pbe-d3:deadbeef",
            "nat": 2, "species": ["Si"], "disk_io": "nowf",
            "source": {"kind": "attempt",
                       "directory": "calculations/eval-000000"}}


def _files() -> list[tuple[str, str]]:
    return [("tmp/pyraimd2.save/charge-density.dat", "charge-density"),
            ("tmp/pyraimd2.save/data-file-schema.xml", "metadata")]


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root)).encode()
        if path.is_symlink():
            h.update(b"L" + rel + str(path.readlink()).encode())
        elif path.is_dir():
            h.update(b"D" + rel)
        else:
            h.update(b"F" + rel + path.read_bytes())
    return h.hexdigest()


def _write_real_checkpoints(run_dir: Path,
                            density_refs: dict[int, int | None]) -> None:
    manager = CheckpointManager(run_dir)
    for generation, density_generation in sorted(density_refs.items()):
        extra = ({} if density_generation is None else
                 {"density_generation": density_generation})
        manager.write(generation, {"step": generation},
                      {"positions": np.zeros((1, 3))}, extra)


def _run_inspect(run_dir: Path, capsys) -> tuple[int, dict]:
    code = cli_main(["density", "inspect", str(run_dir)])
    out = capsys.readouterr()
    assert code == 0, out.err
    return code, json.loads(out.out)


def test_density_inspect_keep_and_reclaim_candidate(
        tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2", "req-3"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    _write_real_checkpoints(run_dir, {1: 1, 2: 2})
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-4"))
    before = _tree_digest(run_dir)

    _, plan = _run_inspect(run_dir, capsys)

    assert plan["dry_run"] is True
    assert "not a deletion authorization" in plan["notice"]
    by_gen = {r["generation"]: r for r in plan["resources"]
              if r["kind"] == "generation"}
    assert by_gen[1]["decision"] == "keep"
    assert by_gen[2]["decision"] == "keep"
    assert by_gen[3]["decision"] == "reclaim_candidate"
    assert by_gen[4]["decision"] == "keep"
    assert by_gen[3]["reasons"] == ["once attached, now unreferenced"]
    assert plan["space"]["attached_unreferenced_seed_bytes"] > 0
    assert _tree_digest(run_dir) == before  # read-only proof


def test_density_inspect_corrupt_and_tmp_leftover_hold(
        tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))          # g1 attached
    # g2: published then made corrupt (manifest removed) -> hold
    publish_density_generation(run_dir, _source(run_dir, "eval-000001"),
                               files=_files(), provenance=_prov("req-2"))
    g2 = run_dir / "restart" / "density" / "g000002" / "manifest.json"
    g2.unlink()
    # g3: publish-interrupted style leftover tmp -> hold
    registry = run_dir / "restart" / "density"
    leftover = registry / ".tmp-g000099"
    leftover.mkdir()
    (leftover / "partial").write_bytes(b"half")
    before = _tree_digest(run_dir)

    _, plan = _run_inspect(run_dir, capsys)

    kinds = {r["name"]: r for r in plan["resources"]}
    assert kinds["g000002"]["kind"] == "corrupt_generation"
    assert kinds["g000002"]["decision"] == "hold"
    assert kinds[".tmp-g000099"]["decision"] == "hold"
    assert _tree_digest(run_dir) == before


def test_density_inspect_never_attached_hold(
        tmp_path: Path, capsys, monkeypatch) -> None:
    """A crash at the state-file commit boundary leaves a committed but
    never-attached generation: the plan holds it for diagnosis."""
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-0"))  # g1 attached
    real_replace = rr.os.replace

    def fail_state_replace(src, dst):
        if Path(dst).name == "state.json":
            raise OSError("crash")
        return real_replace(src, dst)

    monkeypatch.setattr(rr.os, "replace", fail_state_replace)
    try:
        publish_density_generation(run_dir, _source(run_dir, "eval-000001"),
                                   files=_files(), provenance=_prov("req-1"))
        raised = False
    except OSError:
        raised = True
    assert raised, "expected the injected state-commit crash"
    monkeypatch.undo()
    before = _tree_digest(run_dir)

    _, plan = _run_inspect(run_dir, capsys)

    g2 = next(r for r in plan["resources"]
              if r.get("generation") == 2)
    assert g2["decision"] == "hold"
    assert "never attached" in g2["reasons"][0]
    assert _tree_digest(run_dir) == before


def test_density_inspect_symlink_registry_readable_error(
        tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "restart").mkdir(parents=True)
    target = tmp_path / "elsewhere"
    target.mkdir()
    (run_dir / "restart" / "density").symlink_to(target, target_is_directory=True)
    before = _tree_digest(run_dir)

    code = cli_main(["density", "inspect", str(run_dir)])
    out = capsys.readouterr()

    assert code == 1
    assert "symlink" in out.err
    assert "Traceback" not in out.err
    assert _tree_digest(run_dir) == before


def test_density_inspect_unreadable_registry_readable_error(
        tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))
    registry = run_dir / "restart" / "density"
    before = _tree_digest(run_dir)
    registry.chmod(0)
    try:
        code = cli_main(["density", "inspect", str(run_dir)])
    finally:
        registry.chmod(0o755)
    out = capsys.readouterr()

    assert code == 1
    assert "density inspection failed" in out.err
    assert "Traceback" not in out.err
    assert _tree_digest(run_dir) == before


def test_density_inspect_invalid_utf8_readable_error(
        tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))
    manifest = run_dir / "restart" / "density" / "g000001" / "manifest.json"
    manifest.write_bytes(b"\xff\xfe\x00broken")
    before = _tree_digest(run_dir)

    code = cli_main(["density", "inspect", str(run_dir)])
    out = capsys.readouterr()

    assert code == 1
    assert "density inspection failed" in out.err
    assert "Traceback" not in out.err
    assert _tree_digest(run_dir) == before


def test_density_inspect_missing_dir_readable_error(
        tmp_path: Path, capsys) -> None:
    missing = tmp_path / "no-such-run"
    code = cli_main(["density", "inspect", str(missing)])
    out = capsys.readouterr()
    assert code == 2
    assert "does not exist" in out.err
    assert not missing.exists()  # nothing created


def test_density_inspect_fresh_empty_registry(
        tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _, plan = _run_inspect(run_dir, capsys)
    assert plan["resources"] == []
    assert plan["blocked"] == []
