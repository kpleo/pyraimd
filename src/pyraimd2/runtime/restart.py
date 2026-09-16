"""Run-owned density restart registry: publish, resume resolution, dry-run
reclaim planning (issue #7 base layer).

Layout under the run directory (never inside the managed scratch root)::

    <run_dir>/restart/density/
        state.json                  # ONE atomic registry state file:
                                    #   {"latest": int|null, "attached": [ints]}
        gNNNNNN/                    # one immutable published generation
            manifest.json           # versioned manifest (schema below)
            <declared files...>     # explicit relative paths from the caller
        .tmp-gNNNNNN/               # publish leftovers (classified, never
                                    # auto-deleted, never promoted)

Commit protocol (F1 fix): the pointer and the provable attach state are ONE
atomically replaced file, so they can never contradict each other.  Publish
has two observable windows: (i) after the complete generation commit
(``os.replace`` of ``gNNNNNN``) and before the state commit; (ii) inside the
atomic state write (a crash leaves the PREVIOUS state intact).  A generation
committed but never attached is classified ``published_unattached`` — held
for diagnosis, never promoted, never auto-reclaimed.  Retries are idempotent
by attempt identity: same attempt + same verified content reuses the
generation and attaches it ONLY when it was never attached and no newer
generation has since become latest (a stale retry never moves the pointer
backwards); same attempt + different content is refused — generations are
immutable.  Declared payload files are flushed and fsynced BEFORE the
generation commit; a durable publish never rests on metadata-only flushes.

Validity (F2/F3 fixes): registry reads distinguish "fresh" (no registry at
all), "ok" and "corrupt" (state file missing while generations exist,
unparseable, or structurally wrong).  A corrupt registry state makes resume
unresolvable and blocks reclaim — never reported as "never published".
Every generation is validated by one shared validator: the directory must be
confined (no symlink escape from the registry, directory name matching the
manifest generation), the manifest must bind the run identity, and the file
list must be nonempty with each declared relative path resolving to a
regular file inside the generation whose size and SHA-256 match.  The same
validator serves reuse, inspect, resume and the latest-pointer read.
Checkpoint references are read over the real CheckpointManager layout with
strict schema/structure checks (unknown schema or a mismatched generation
blocks reclaim); legacy checkpoints are exactly those with a supported
schema and no density field.  Scratch attempt records are validated with the
existing authoritative ``scratch._validate_record``; unknown schema/state
blocks reclaim rather than being silently skipped.

Managed-scratch sources (F3 fix): the persistent attempt archive holds only
pw.in/pw.out, so the density of a scratch attempt can only come from the
scratch tree itself.  ``publish_density_generation_from_attempt`` is the
bounded entry: it requires the attempt's authoritative record (validated by
``scratch._validate_record``), the record's run_root must be THIS run
directory, and the scratch directory must pass the existing
``scratch._verify_identity_path`` (path walk + independent ownership
marker).  Only the declared seed files are copied into the run-owned
registry; an unverified external path is still refused, and the caller never
acquires deletion rights over the source.

Reclaim: :func:`plan_density_reclaim` is the read-only dry-run (per-resource
keep/hold/reclaim-candidate reasons); :func:`execute_density_reclaim` is the
narrow execution entry — it holds the run's single-writer lock, re-reads
every reference under it, refuses a stale caller-supplied plan, persists a
durable tombstone in the state file BEFORE deleting each candidate, and
resumes interrupted deletions only when the tombstone, readable ownership
manifest and fresh unreferenced status agree. The registry accepts the
caller's file role list verbatim: registering ``charge-density.dat``/
``charge-density.hdf5`` proves the files were located and copied — it does
NOT claim the QE minimal seed is proven sufficient (that proof is the
delayed producer release on the engine side).  Nothing here scans the
filesystem to discover runs; every entry point takes the explicit run
directory.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from pyraimd2.runtime.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointManager
from pyraimd2.runtime.events import EVALUATION_COMMITTED, EventLog, EventLogError
from pyraimd2.runtime.scratch import (
    ScratchError,
    _atomic_write_json,
    _fsync_dir,
    _sha256,
    _validate_record,
    _verify_identity_path,
)

RESTART_SCHEMA = "density-restart-v1"
GENERATION_MANIFEST = "manifest.json"
STATE_FILE = "state.json"

# inspect classifications
CLASS_GENERATION = "generation"
CLASS_TMP_LEFTOVER = "publish_tmp_leftover"
CLASS_CORRUPT = "corrupt_generation"
CLASS_RECLAIMED = "reclaimed_generation"

# plan decisions (a plan is only a plan: nothing here deletes)
DECISION_KEEP = "keep"
DECISION_RECLAIM_CANDIDATE = "reclaim_candidate"
DECISION_HOLD = "hold"
DECISION_RECLAIMED = "reclaimed"

# resume resolution branches
BRANCH_OK = "ok"
BRANCH_EXTERNAL_INIT = "external_initialization_required"
BRANCH_UNRESOLVABLE = "unresolvable"

# sentinel for resolve_density_for_resume's explicit reference: the caller
# DID pass the resumed boundary's own recorded reference (None included —
# an explicit no-density declaration), overriding the checkpoint field
_UNSET = object()

# registry state branches
STATE_FRESH = "fresh"      # no registry at all: nothing was ever published
STATE_OK = "ok"
STATE_CORRUPT = "corrupt"

# scratch attempt-record states whose attempt is still in flight (its input
# density must survive until the attempt reaches a terminal state)
_IN_FLIGHT_SCRATCH_STATES = ("allocated", "archived", "archive_failed",
                             "cleanup_pending")


class RestartError(RuntimeError):
    """A registry operation cannot proceed; the message states the remedy."""


# ---------------------------------------------------------------------------
# path confinement


def _registry_dir(run_dir: Path) -> Path:
    return run_dir / "restart" / "density"


def _check_registry_confined(run_dir: Path) -> Path:
    """The registry target must be physically inside THIS run directory —
    a symlink at restart/, density/ or the registry itself (into another
    run) is refused before anything is created, written or read."""
    run_dir = run_dir.resolve()
    registry = _registry_dir(run_dir)
    probe = run_dir
    for component in ("restart", "density"):
        probe = probe / component
        if probe.is_symlink():
            raise RestartError(
                f"registry path component is a symlink (possible cross-run "
                f"escape): {probe}")
    if registry.exists():
        if registry.is_symlink():
            raise RestartError(f"registry directory is a symlink: {registry}")
        if not registry.resolve().is_relative_to(run_dir):
            raise RestartError(
                f"registry resolves outside the run directory: {registry}")
    return registry


def _check_generation_confined(registry: Path, directory: Path) -> None:
    if directory.is_symlink() or not directory.resolve().is_relative_to(registry):
        raise RestartError(
            f"generation directory escapes the registry: {directory}")


def _safe_relpath(rel: str) -> str:
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) \
            or rel.startswith("..") or "/../" in rel or rel.endswith("/..") \
            or "\\" in rel:
        raise RestartError(f"declared file escapes the source: {rel!r}")
    return rel


def _resolve_inside(base: Path, rel: str) -> Path:
    candidate = (base / rel).resolve()
    if not candidate.is_relative_to(base):
        raise RestartError(f"declared file resolves outside the source: {rel!r}")
    return candidate


# ---------------------------------------------------------------------------
# generation validation (ONE validator shared by reuse/inspect/resume/latest)


def _read_manifest(directory: Path) -> dict | None:
    path = directory / GENERATION_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema") != RESTART_SCHEMA:
        return None
    return manifest


def _ownership_problem(manifest: dict, *, directory_name: str,
                       run_dir: Path) -> str | None:
    """None when the manifest's identity/ownership fields are consistent —
    the generation matches the directory name, the recorded run root IS
    this run, and a carried run id does not contradict the run's own
    manifest — else the reason.  This is deliberately separate from
    payload completeness: an interrupted deletion may lack payload files
    but must never resume over an explicit ownership contradiction, and a
    manifest that is missing, unreadable or silent on ownership is never
    treated as positively owned."""
    expected = f"g{int(manifest.get('generation', -1)):06d}"
    if directory_name != expected:
        return (f"directory {directory_name} does not match manifest "
                f"generation {manifest.get('generation')!r}")
    # the manifest must bind THIS run: the resolved run root recorded at
    # publish, and the run id when both the run and the manifest carry one
    if manifest.get("run_root") != str(run_dir):
        return (f"generation manifest run_root {manifest.get('run_root')!r} "
                f"does not match this run {run_dir}")
    run_manifest = _read_json_quiet(run_dir / "manifest.json")
    if run_manifest is not None and manifest.get("run_id") is not None \
            and run_manifest.get("run_id") != manifest.get("run_id"):
        return (f"generation manifest run_id {manifest.get('run_id')!r} "
                f"does not match the run's {run_manifest.get('run_id')!r}")
    return None


def _validate_generation(directory: Path, manifest: dict,
                         *, registry: Path, run_dir: Path) -> str | None:
    """None when valid, else the reason.  Shared by reuse, inspect, resume
    and the latest read: directory name/identity, confinement, run binding,
    and a nonempty file list whose every entry is a confined regular file
    with matching size and SHA-256."""
    try:
        _check_generation_confined(registry, directory)
    except RestartError as error:
        return str(error)
    problem = _ownership_problem(manifest, directory_name=directory.name,
                                 run_dir=run_dir)
    if problem is not None:
        return problem
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return "manifest has no nonempty file list"
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) \
                or not isinstance(entry.get("role"), str) or not entry["role"]:
            return "manifest file entry without a usable path/role"
        try:
            path = _resolve_inside(directory, _safe_relpath(entry["path"]))
        except RestartError as error:
            return str(error)
        if path.is_symlink() or not path.is_file():
            return f"declared file missing or not a regular file: {entry['path']}"
        if not isinstance(entry.get("bytes"), int) \
                or path.stat().st_size != entry["bytes"]:
            return f"declared file size mismatch: {entry['path']}"
        if not isinstance(entry.get("sha256"), str) \
                or _sha256(path) != entry["sha256"]:
            return f"declared file content digest mismatch: {entry['path']}"
    return None


# ---------------------------------------------------------------------------
# registry state (one atomic file; the pointer and attach state never split)


def _read_state(registry: Path) -> tuple[str, dict | None, str | None]:
    """(branch, state, reason).  fresh: no registry at all (never
    published).  corrupt: generations exist without a state file, or the
    file is unparseable/structurally wrong.  The state invariant is checked
    semantically, never guessed: the ``latest`` KEY must be present (a
    missing key is not a legal null); with an empty ``attached`` the latest
    must be null; with a nonempty ``attached`` the latest must be its
    maximum; values are positive non-bool ints, strictly sorted and unique.
    ``reclaimed`` (absent in pre-reclaim state files, then empty) is the
    durable tombstone list of deliberately deleted generations: sorted
    unique positive ints, each attached, never the latest.  A contradictory
    state is corrupt — nothing rebuilds or resets it."""
    if not registry.is_dir():
        return STATE_FRESH, None, None
    path = registry / STATE_FILE
    state = _read_json_quiet(path)
    if state is None:
        has_generations = any(p.name.startswith("g") and p.is_dir()
                              and not p.name.startswith(".tmp-")
                              for p in registry.iterdir())
        if not path.exists() and not has_generations:
            return STATE_FRESH, None, None
        return STATE_CORRUPT, None, (
            "registry state file missing or unreadable while generations "
            "exist — the publish state cannot be determined")

    def _positive_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if "latest" not in state or "attached" not in state:
        return STATE_CORRUPT, None, ("registry state misses the latest/"
                                     "attached keys")
    latest = state["latest"]
    attached = state["attached"]
    if not (latest is None or _positive_int(latest)) \
            or not isinstance(attached, list) \
            or not all(_positive_int(g) for g in attached) \
            or len(set(attached)) != len(attached) \
            or attached != sorted(attached):
        return STATE_CORRUPT, None, ("registry state has invalid "
                                     "latest/attached values")
    if not attached and latest is not None:
        return STATE_CORRUPT, None, ("registry state contradicts itself: "
                                     "nothing attached but latest is set")
    if attached and latest != attached[-1]:
        return STATE_CORRUPT, None, ("registry state contradicts itself: "
                                     "latest is not the highest attached "
                                     "generation")
    reclaimed = state.get("reclaimed", [])
    if not isinstance(reclaimed, list) \
            or not all(_positive_int(g) for g in reclaimed) \
            or len(set(reclaimed)) != len(reclaimed) \
            or reclaimed != sorted(reclaimed):
        return STATE_CORRUPT, None, ("registry state has invalid reclaimed "
                                     "tombstone values")
    if any(g not in attached for g in reclaimed) or latest in reclaimed:
        return STATE_CORRUPT, None, (
            "registry state contradicts itself: reclaimed tombstones must "
            "name attached non-latest generations")
    return STATE_OK, {"latest": latest, "attached": attached,
                      "reclaimed": reclaimed}, None


def _ensure_or_read_state(registry: Path) -> dict:
    """A genuinely NEW registry persists an explicit empty state before any
    generation is written; an existing registry's missing/corrupt state
    REFUSES every publish path — a damaged history is never reset to
    fresh by writing over it."""
    branch, state, reason = _read_state(registry)
    if branch == STATE_CORRUPT:
        raise RestartError(
            f"registry state is corrupt: {reason}; refusing to publish over "
            "an undetermined history (diagnose the existing generations "
            "first — no state-or-default reset exists)")
    if branch == STATE_FRESH:
        state = {"latest": None, "attached": [], "reclaimed": []}
        _atomic_write_json(registry / STATE_FILE, state)
        _fsync_dir(registry)
        return state
    return state


def _commit_latest(registry: Path, generation: int, state: dict) -> None:
    """Atomically move the pointer and record the attach state in ONE file,
    computed from the already-verified state (never re-read loosely, never
    over a corrupt one).  Every generation that has ever been latest is in
    ``attached``; the superseded previous latest joins it here.  The
    reclaim tombstones ride along unchanged.  A crash inside the atomic
    write leaves the previous state intact."""
    attached = set(state["attached"])
    if state["latest"] is not None:
        attached.add(int(state["latest"]))
    attached.add(int(generation))
    _atomic_write_json(registry / STATE_FILE,
                       {"latest": int(generation),
                        "attached": sorted(attached),
                        "reclaimed": sorted(state.get("reclaimed", []))})


def _tombstone_reclaimed(registry: Path, generation: int) -> None:
    """Persist the durable reclaim tombstone BEFORE any byte is deleted:
    the generation joins the state's ``reclaimed`` list in one atomic
    state write.  Idempotent; only an attached, non-latest generation is
    tombstoned.  A crash after this point leaves a tombstoned generation
    on disk — inspection marks it and a later execution resumes the
    deletion; a fully deleted generation stays tombstoned forever, so a
    re-execution never mistakes it for corruption and its generation
    number is never reused."""
    branch, state, reason = _read_state(registry)
    if branch != STATE_OK:
        raise RestartError(f"registry state is not ok: {reason}")
    if generation in state["reclaimed"]:
        return
    if generation not in state["attached"] or generation == state["latest"]:
        raise RestartError(
            f"generation g{generation:06d} is not an attached non-latest "
            "generation — refusing the reclaim tombstone")
    _atomic_write_json(registry / STATE_FILE,
                       {"latest": state["latest"],
                        "attached": state["attached"],
                        "reclaimed": sorted([*state["reclaimed"], generation])})
    _fsync_dir(registry)


def _next_generation(registry: Path, state: dict) -> int:
    taken = []
    for path in registry.iterdir():
        name = path.name
        name = name.removeprefix(".tmp-")
        if name.startswith("g") and name[1:].isdigit():
            taken.append(int(name[1:]))
    # a reclaimed generation never frees its number: the attach history
    # and the tombstones join the floor, so an interrupted or completed
    # reclaim can never cause a generation number to be reused
    taken.extend(int(g) for g in state.get("attached", []))
    taken.extend(int(g) for g in state.get("reclaimed", []))
    if state.get("latest") is not None:
        taken.append(int(state["latest"]))
    return max(taken, default=0) + 1


def _generation_name(generation: int) -> str:
    return f"g{generation:06d}"


def _content_digest(files: list[dict], provenance: dict) -> str:
    import hashlib

    identity = {
        "files": [{"path": f["path"], "role": f["role"], "sha256": f["sha256"]}
                  for f in files],
        "reference_fingerprint": provenance.get("reference_fingerprint"),
        "nat": provenance.get("nat"),
        "species": provenance.get("species"),
        "disk_io": provenance.get("disk_io"),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _find_by_attempt(registry: Path, attempt: dict) -> tuple[Path, dict] | None:
    for path in sorted(registry.glob("g*")):
        if not path.is_dir() or path.name.startswith(".tmp-"):
            continue
        manifest = _read_manifest(path)
        if manifest is None:
            continue
        if manifest.get("attempt") == attempt:
            return path, manifest
    return None


def _copy_file_fsynced(source: Path, target: Path, *, tmp_root: Path) -> None:
    """Copy a declared payload file and fsync it BEFORE the generation
    commit — plus every newly created parent directory, leaf up to the
    generation root (a file fsync does not persist new directory entries)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    current = target.parent
    while current != tmp_root and tmp_root in current.parents:
        _fsync_dir(current)
        current = current.parent


def _run_binding(run_dir: Path, provenance: dict) -> dict:
    """The run identity bound into every generation manifest: the resolved
    run root (always) and the run id (when the run has one or the caller
    supplies it — both must agree with the run's own manifest)."""
    binding = {"run_root": str(run_dir)}
    run_manifest = _read_json_quiet(run_dir / "manifest.json")
    run_id = provenance.get("run_id")
    if run_manifest is not None:
        own = run_manifest.get("run_id")
        if run_id is not None and run_id != own:
            raise RestartError(
                f"provenance run_id {run_id!r} does not match the run's own "
                f"manifest {own!r}")
        run_id = own
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id:
            raise RestartError("provenance 'run_id' must be a nonempty string")
        binding["run_id"] = run_id
    return binding


def publish_density_generation(run_dir: str | Path, source_dir: str | Path, *,
                               files: list[tuple[str, str]],
                               provenance: dict,
                               _verified_scratch_source: bool = False) -> dict:
    """Publish one immutable density generation and point latest at it.

    ``files``: the caller's explicit role list ((relative path in the
    source, role) pairs).  ``provenance``: ``run_id``, ``attempt``
    ({"request_id", "attempt_id"}), ``reference_fingerprint``, ``nat``,
    ``species``, ``disk_io``, ``source`` (human-readable).  The source must
    be inside this run directory; the only alternative is the bounded
    managed-scratch entry below.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise RestartError(f"run directory not found: {run_dir}")
    registry = _check_registry_confined(run_dir)
    run_binding = _run_binding(run_dir, provenance)
    if not provenance.get("attempt") or not provenance.get("reference_fingerprint"):
        raise RestartError(
            "provenance must carry the attempt identity and the reference "
            "fingerprint — a generation without provenance is never published")
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise RestartError(f"density source directory not found: {source_dir}")
    if not _verified_scratch_source and not source.is_relative_to(run_dir):
        raise RestartError(
            f"density source {source_dir} is outside the run directory "
            f"{run_dir}: external sources are read-only and never become "
            "run-owned generations (the bounded entry is "
            "publish_density_generation_from_attempt over a verified "
            "managed-scratch record)")
    if not files:
        raise RestartError("no files declared: an empty seed is not a density")

    declared: list[dict] = []
    for rel, role in files:
        rel = _safe_relpath(rel)
        path = _resolve_inside(source, rel)
        if path.is_symlink() or not path.is_file():
            raise RestartError(f"declared file is not a regular file: {rel}")
        declared.append({"path": rel, "role": role,
                         "sha256": _sha256(path), "bytes": path.stat().st_size})
    digest = _content_digest(declared, provenance)

    registry.mkdir(parents=True, exist_ok=True)
    _fsync_dir(run_dir / "restart")
    # The publish state is verified BEFORE any generation write: a fresh
    # registry first persists an explicit empty state; a missing/corrupt
    # state on an existing registry refuses everything (never reset).
    state = _ensure_or_read_state(registry)
    existing = _find_by_attempt(registry, provenance["attempt"])
    if existing is not None:
        path, manifest = existing
        if manifest.get("content_digest") != digest:
            raise RestartError(
                f"attempt {provenance['attempt']} already published "
                f"{path.name} with different content: generations are "
                "immutable — a retry with different content is refused, "
                "never an overwrite")
        invalid = _validate_generation(path, manifest, registry=registry,
                                       run_dir=run_dir)
        if invalid is not None:
            raise RestartError(
                f"the existing generation {path.name} is no longer valid "
                f"({invalid}): refusing to claim a reuse of corrupt content")
        generation = int(manifest["generation"])
        latest = state["latest"]
        attached = set(state["attached"])
        if generation not in attached \
                and (latest is None or int(latest) < generation):
            # recovery of a never-attached generation with no newer progress
            _commit_latest(registry, generation, state)
        return {"generation": generation, "directory": str(path),
                "manifest": manifest, "reused": True}

    generation = _next_generation(registry, state)
    tmp = registry / f".tmp-{_generation_name(generation)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        for entry in declared:
            target = tmp / entry["path"]
            _copy_file_fsynced(source / entry["path"], target, tmp_root=tmp)
            if _sha256(target) != entry["sha256"]:
                raise RestartError(
                    f"copied file digest mismatch for {entry['path']}: the "
                    "extracted seed does not match the validated source")
        manifest = {
            "schema": RESTART_SCHEMA,
            "generation": generation,
            "content_digest": digest,
            "created_unix": time.time(),
            "files": declared,
            **provenance,
            **run_binding,  # the validated binding wins any caller copy
        }
        _atomic_write_json(tmp / GENERATION_MANIFEST, manifest)
        _fsync_dir(tmp)
        final = registry / _generation_name(generation)
        os.replace(tmp, final)      # the complete-generation commit point
        _fsync_dir(registry)
        _commit_latest(registry, generation, state)  # the state commit point
        return {"generation": generation, "directory": str(final),
                "manifest": manifest, "reused": False}
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def publish_density_generation_from_attempt(run_dir: str | Path, record: dict, *,
                                            files: list[tuple[str, str]],
                                            provenance: dict) -> dict:
    """The bounded managed-scratch entry: publish from an attempt's scratch
    directory whose authoritative record is verified.

    The record must pass the existing ``scratch._validate_record``; its
    run_root must be THIS run directory; the scratch directory must pass the
    existing ownership/marker walk (``scratch._verify_identity_path``); and
    the caller's attempt identity must match the record EXACTLY — a record
    for attempt A never publishes under attempt B's name.  Only the
    declared files are copied — never the whole .save tree, and the caller
    acquires no deletion rights over the source.
    """
    run_dir = Path(run_dir).resolve()
    try:
        _validate_record(record, path=Path("<caller-provided>"))
    except ScratchError as error:
        raise RestartError(
            f"the attempt record is not a valid authoritative record: "
            f"{error}") from error
    if Path(record["run_root"]).resolve() != run_dir:
        raise RestartError(
            f"the attempt record belongs to run root {record['run_root']!r}, "
            f"not {run_dir}: only this run's own attempts can publish here")
    attempt = provenance.get("attempt")
    if not isinstance(attempt, dict) \
            or not isinstance(attempt.get("request_id"), str) \
            or not attempt["request_id"] \
            or not isinstance(attempt.get("attempt_id"), str) \
            or not attempt["attempt_id"]:
        raise RestartError(
            "provenance must carry a usable attempt identity "
            "({'request_id', 'attempt_id'} as nonempty strings)")
    if (attempt["request_id"], attempt["attempt_id"]) != \
            (record["request_id"], record["attempt_id"]):
        raise RestartError(
            f"provenance attempt {attempt} does not match the verified "
            f"record's {record['request_id']!r}/{record['attempt_id']!r}: a "
            "verified attempt never publishes under another attempt's name")
    try:
        source = _verify_identity_path(record)
    except ScratchError as error:
        raise RestartError(
            f"the attempt scratch directory failed ownership verification: "
            f"{error}") from error
    return publish_density_generation(run_dir, source, files=files,
                                      provenance=provenance,
                                      _verified_scratch_source=True)


def _read_json_quiet(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# inspection / resolution


def inspect_density_registry(run_dir: str | Path) -> dict:
    """Classify every resource in the registry (read-only, confinement
    enforced).  A symlinked/escaping registry raises instead of reading
    another run's data.  Reclaim tombstones are reported honestly: a
    tombstoned generation still on disk (an interrupted deletion) is
    marked ``reclaim_tombstoned``; a fully reclaimed one is listed as
    ``reclaimed_generation`` — never mistaken for corruption or for a
    missing referenced generation.  A tombstoned CORRUPT tree (a
    partially deleted one) is resumable only while its identity stays
    verifiable — the directory name, the durable reclaimed list and a
    readable manifest that confirms ownership (run_root/run_id) must all
    agree — so the interrupted deletion resumes without re-validating
    already-deleted payload; a manifest that contradicts ownership is
    held forever, and a missing or unreadable one is never resumed."""
    run_dir = Path(run_dir).resolve()
    registry = _check_registry_confined(run_dir)
    branch, state, reason = _read_state(registry)
    reclaimed = [] if state is None else list(state["reclaimed"])
    resources: list[dict] = []
    if registry.is_dir():
        for path in sorted(registry.iterdir()):
            if path.name.startswith(".tmp-"):
                resources.append({"name": path.name, "kind": CLASS_TMP_LEFTOVER,
                                  "bytes": _tree_bytes(path)})
            elif path.is_dir() and path.name.startswith("g"):
                manifest = _read_manifest(path)
                invalid = None
                if manifest is None:
                    invalid = "missing/unreadable/unknown-version manifest"
                else:
                    invalid = _validate_generation(path, manifest,
                                                   registry=registry,
                                                   run_dir=run_dir)
                if invalid is not None:
                    resources.append({"name": path.name, "kind": CLASS_CORRUPT,
                                      "reason": invalid,
                                      "bytes": _tree_bytes(path)})
                else:
                    resources.append({"name": path.name, "kind": CLASS_GENERATION,
                                      "generation": int(manifest["generation"]),
                                      "manifest": manifest,
                                      "bytes": _tree_bytes(path)})
    on_disk: set[int] = set()
    for resource in resources:
        name = resource["name"]
        if name.startswith("g") and name[1:].isdigit():
            generation = int(name[1:])
            on_disk.add(generation)
            if generation not in reclaimed:
                continue
            if resource["kind"] == CLASS_GENERATION:
                resource["reclaim_tombstoned"] = True
            elif resource["kind"] == CLASS_CORRUPT:
                # An interrupted deletion (a partially deleted tree) is
                # resumable only when its ownership stays VERIFIABLE: the
                # directory name and the durable reclaimed list agree AND a
                # readable manifest confirms ownership (same run_root/run_id
                # binding as the normal validator).  Missing payload is the
                # normal state of a partial deletion; a manifest that
                # explicitly contradicts ownership — a foreign tree copied
                # into the slot — is held forever, and a missing/unreadable
                # manifest makes the identity unverifiable: never positively
                # owned, never resumed.
                manifest = _read_manifest(registry / name)
                if manifest is not None and _ownership_problem(
                        manifest, directory_name=name,
                        run_dir=run_dir) is None:
                    resource["generation"] = generation
                    resource["reclaim_tombstoned"] = True
    for generation in reclaimed:
        if generation not in on_disk:
            resources.append({"name": _generation_name(generation),
                              "kind": CLASS_RECLAIMED,
                              "generation": generation, "bytes": 0})
    return {"registry": str(registry),
            "state": branch,
            "state_reason": reason,
            "latest": (None if state is None else state["latest"]),
            "attach_history": ([] if state is None else state["attached"]),
            "reclaimed": reclaimed,
            "resources": resources}


def latest_density_generation(run_dir: str | Path) -> dict | None:
    """The latest generation's validated record, or None (corrupt state or
    a corrupt pointed generation both answer None — never a guess)."""
    view = inspect_density_registry(run_dir)
    if view["state"] != STATE_OK or view["latest"] is None:
        return None
    for resource in view["resources"]:
        if resource["kind"] == CLASS_GENERATION \
                and resource["generation"] == view["latest"]:
            return {"generation": resource["generation"],
                    "directory": str(Path(view["registry"]) / resource["name"]),
                    "manifest": resource["manifest"]}
    return None


def _density_reference_from(manifest: dict, *, what: str) -> tuple[int | None, str | None]:
    """The density_generation reference of an authoritative record/manifest.

    (value, None) when the reference is usable; (None, None) when the field
    is ABSENT (a legacy record, external init) or explicitly null (a
    declared "no density input" — a distinct, deliberate contract, never
    mistaken for a legacy missing field); (None, reason) when the field is
    present but invalid (not a positive non-bool int) — the caller must
    block rather than silently skip.
    """
    if "density_generation" not in manifest:
        return None, None  # legacy: no field
    value = manifest["density_generation"]
    if value is None:
        return None, None  # explicit "no density reference" declaration
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, (f"{what}: density_generation {value!r} is not a valid "
                      "positive integer reference")
    return int(value), None


def _read_checkpoint_manifest(run_dir: Path, generation: int) -> tuple[dict | None, str | None]:
    """Strict structure check over the real CheckpointManager layout.

    Returns (manifest, None) only for a supported checkpoint manifest with
    the authoritative state.json/arrays.npz digest keys and a valid (or
    legacy-absent / explicitly-null) density field; else (None, reason).
    """
    path = run_dir / "checkpoints" / str(generation) / "manifest.json"
    manifest = _read_json_quiet(path)
    if manifest is None:
        return None, f"checkpoint {generation}: manifest unreadable"
    if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return None, (f"checkpoint {generation}: unknown schema "
                      f"{manifest.get('checkpoint_schema_version')!r}")
    if manifest.get("generation") != generation:
        return None, (f"checkpoint {generation}: manifest generation "
                      f"{manifest.get('generation')!r} does not match")
    files = manifest.get("files")
    if not isinstance(files, dict) \
            or set(files) != {"state.json", "arrays.npz"} \
            or not all(isinstance(v, str) and len(v) == 64
                       for v in files.values()):
        return None, (f"checkpoint {generation}: missing or malformed "
                      "authoritative files block (state.json/arrays.npz "
                      "digests required)")
    _referenced, problem = _density_reference_from(
        manifest, what=f"checkpoint {generation}")
    if problem is not None:
        return None, problem
    return manifest, None


def resolve_density_for_resume(run_dir: str | Path, *,
                               checkpoint_generation: int | None = None,
                               density_reference=_UNSET) -> dict:
    """Resolve the density source for a continued/restored chain.

    An explicitly restored checkpoint's recorded ``density_generation``
    wins over the latest pointer and resolves EXACTLY that generation —
    never a later one.  A legacy checkpoint (supported schema, no field)
    gets the explicit external-initialization branch.  A corrupt registry
    state or an unreadable/unknown checkpoint manifest answers
    ``unresolvable`` with the reason — never "never published".

    ``density_reference`` (the resumed boundary's own committed record,
    e.g. the healed evaluation's commit payload) takes precedence over
    the checkpoint field whenever it is passed — including an explicit
    None ("the boundary declared no density reference" →
    external-initialization).  The value is validated with the same
    ``_density_reference_from`` semantics and error vocabulary as the
    checkpoint field: a present-but-invalid reference blocks the resume
    rather than being silently skipped.
    """
    run_dir = Path(run_dir).resolve()
    view = inspect_density_registry(run_dir)
    by_generation = {r["generation"]: r for r in view["resources"]
                     if r["kind"] == CLASS_GENERATION}

    def _resolve(generation: int) -> dict:
        resource = by_generation.get(generation)
        if resource is None:
            return {"branch": BRANCH_UNRESOLVABLE,
                    "reason": f"density generation g{generation:06d} is "
                              "referenced but missing/corrupt — it cannot be "
                              "substituted by a later generation"}
        return {"branch": BRANCH_OK, "generation": generation,
                "directory": str(Path(view["registry"]) / resource["name"]),
                "manifest": resource["manifest"]}

    if density_reference is not _UNSET:
        referenced, ref_problem = _density_reference_from(
            {"density_generation": density_reference},
            what="the resumed boundary's committed record")
        if ref_problem is not None:
            return {"branch": BRANCH_UNRESOLVABLE, "reason": ref_problem}
        if referenced is None:
            return {"branch": BRANCH_EXTERNAL_INIT,
                    "reason": "the resumed boundary declares no density "
                              "reference (explicit null): initialize from "
                              "the configured external source"}
        return _resolve(int(referenced))
    if checkpoint_generation is not None:
        manifest, problem = _read_checkpoint_manifest(run_dir, checkpoint_generation)
        if manifest is None:
            return {"branch": BRANCH_UNRESOLVABLE, "reason": problem}
        referenced, ref_problem = _density_reference_from(
            manifest, what=f"checkpoint {checkpoint_generation}")
        if ref_problem is not None:
            return {"branch": BRANCH_UNRESOLVABLE, "reason": ref_problem}
        if referenced is None:
            return {"branch": BRANCH_EXTERNAL_INIT,
                    "reason": "legacy checkpoint (supported schema, no "
                              "density_generation field) or an explicit "
                              "no-density declaration: initialize from the "
                              "configured external source"}
        return _resolve(int(referenced))
    if view["state"] == STATE_CORRUPT:
        return {"branch": BRANCH_UNRESOLVABLE, "reason": view["state_reason"]}
    latest = view["latest"]
    if latest is None:
        return {"branch": BRANCH_EXTERNAL_INIT,
                "reason": "no published density generation; initialize from "
                          "the configured external source"}
    return _resolve(int(latest))


# ---------------------------------------------------------------------------
# references and the dry-run plan


def _committed_boundary_reference(run_dir: Path) -> tuple[int | None, str | None]:
    """(reference, problem): the density reference of the actual
    recoverable committed boundary — the highest committed evaluation's
    own record (the resume boundary: normal tail, crash-window heal tail
    and checkpoint-fallback tail all resolve exactly it).  A torn final
    line is an uncommitted crash tail and is skipped exactly like the
    event-log reader skips it; any other unparseable line blocks the
    reference determination."""
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return None, None
    try:
        lines = [line for line in
                 events_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    except OSError as error:
        return None, f"events.jsonl is unreadable: {error}"
    boundary = None
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue  # torn crash tail: never committed, skipped on read
            return None, ("events.jsonl has an unparseable committed line; "
                          "the actual boundary reference cannot be "
                          "determined")
        if event.get("type") != EVALUATION_COMMITTED:
            continue
        context = event.get("context") or {}
        evaluation_id = context.get("evaluation_id")
        if isinstance(evaluation_id, bool) \
                or not isinstance(evaluation_id, int):
            return None, ("an evaluation commit has no usable evaluation_id; "
                          "the actual boundary reference cannot be "
                          "determined")
        if boundary is None or evaluation_id > boundary[0]:
            boundary = (evaluation_id, event)
    if boundary is None:
        return None, None
    return _density_reference_from(
        boundary[1], what=f"the committed boundary evaluation {boundary[0]}")


def compute_references(run_dir: str | Path, *,
                       extra_keep: tuple | list = ()) -> dict:
    """The reference union for reclaim planning (read-only).

    latest pointer (from the verified state file) + every retained
    checkpoint's recorded density_generation (strict structure checks) +
    the actual recoverable committed boundary's own record (the highest
    committed evaluation) + every in-flight attempt's recorded input
    generation (records validated with the existing authoritative scratch
    validator) + every UNCONSUMED producer seed (its attempt scratch still
    kept: no independent successful read has completed the delayed
    release) + bounded ``extra_keep``.  Anything whose reference cannot be
    determined BLOCKS reclaim with a reason; normal valid resources are
    still kept correctly — "hold everything forever" is not the
    implementation.
    """
    run_dir = Path(run_dir).resolve()
    view = inspect_density_registry(run_dir)
    references: dict[int, list[str]] = {}
    blocked: list[str] = []
    if view["state"] == STATE_CORRUPT:
        blocked.append(f"registry state corrupt: {view['state_reason']}")

    def add(generation: int, reason: str) -> None:
        references.setdefault(int(generation), []).append(reason)

    if view["latest"] is not None:
        add(int(view["latest"]), "latest pointer")

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        for path in sorted(checkpoints_dir.iterdir()):
            if not (path.is_dir() and path.name.isdigit()):
                continue
            manifest, problem = _read_checkpoint_manifest(run_dir, int(path.name))
            if manifest is None:
                blocked.append(problem)
                continue
            referenced, ref_problem = _density_reference_from(
                manifest, what=f"checkpoint {path.name}")
            if ref_problem is not None:
                blocked.append(ref_problem)
                continue
            if referenced is not None:
                add(int(referenced), f"retained checkpoint {path.name}")

    records_dir = run_dir / "scratch_records"
    validated_records: list[dict] = []
    if records_dir.is_dir():
        for record_path in sorted(records_dir.rglob("*.json")):
            record = _read_json_quiet(record_path)
            try:
                if record is None:
                    raise ScratchError("record is not parseable JSON")
                _validate_record(record, path=record_path)
            except ScratchError as error:
                blocked.append(f"scratch record {record_path.name}: {error}")
                continue
            validated_records.append(record)
            if record["state"] in _IN_FLIGHT_SCRATCH_STATES:
                referenced, ref_problem = _density_reference_from(
                    record, what=f"scratch record {record_path.name}")
                if ref_problem is not None:
                    blocked.append(ref_problem)
                    continue
                if referenced is not None:
                    add(int(referenced),
                        f"in-flight attempt input ({record_path.stem})")

    boundary_ref, boundary_problem = _committed_boundary_reference(run_dir)
    if boundary_problem is not None:
        blocked.append(boundary_problem)
    elif boundary_ref is not None:
        add(int(boundary_ref), "committed boundary evaluation")

    # A generation whose producer attempt scratch is still kept was never
    # independently consumed: the delayed-release proof is incomplete and
    # the published seed must survive.  A producer record in an abnormal
    # state keeps its generation too — the consumption state is then
    # undetermined, and uncertain dependencies retain data.  A producer
    # with no record at all (an unmanaged attempt, whose source tree is
    # permanent, or a lost record — whose scratch can never be reclaimed
    # without it) adds no reference from this channel: the fallback copy
    # is not deletable either way.  An invalid record already blocked the
    # whole reclaim above.
    by_attempt = {(record["request_id"], record["attempt_id"]): record
                  for record in validated_records}
    for resource in view["resources"]:
        if resource["kind"] != CLASS_GENERATION:
            continue
        generation = resource["generation"]
        attempt = resource["manifest"].get("attempt")
        if not isinstance(attempt, dict) \
                or not isinstance(attempt.get("request_id"), str) \
                or not attempt["request_id"] \
                or not isinstance(attempt.get("attempt_id"), str) \
                or not attempt["attempt_id"]:
            add(generation, "producer attempt identity missing from the "
                            "manifest — consumption state undetermined")
            continue
        producer = by_attempt.get((attempt["request_id"],
                                   attempt["attempt_id"]))
        if producer is None:
            continue
        if producer["state"] == "kept":
            add(generation, f"unconsumed producer seed "
                            f"({attempt['request_id']}/"
                            f"{attempt['attempt_id']})")
        elif producer["state"] not in ("archived", "cleanup_pending",
                                       "cleaned"):
            add(generation, f"producer attempt in unexpected state "
                            f"{producer['state']!r} — consumption state "
                            "undetermined")

    for generation in extra_keep:
        add(int(generation), "extra_keep")
    return {"references": references, "blocked": blocked,
            "attach_history": view["attach_history"],
            "reclaimed": view["reclaimed"], "latest": view["latest"]}


def plan_density_reclaim(run_dir: str | Path, *,
                         extra_keep: tuple | list = ()) -> dict:
    """Dry-run reclaim plan: per-resource keep/reclaim-candidate/hold with
    reasons, plus the space report.  NOTHING is deleted.

    Only a valid generation that is unreferenced AND once attached (in the
    state file's attached list) is a routine reclaim candidate.  A
    published-but-never-attached generation is held for diagnosis; a corrupt
    generation is held with its reason — unless it carries a reclaim
    tombstone, in which case it is an interrupted deletion and resumable;
    any blocked reference flips every candidate to hold — a plan over an
    incomplete reference set is not a deletion credential.  A fully
    reclaimed generation is reported as ``reclaimed`` (its durable
    tombstone): inspection distinguishes an intentional deletion from
    corruption, forever.
    """
    run_dir = Path(run_dir).resolve()
    view = inspect_density_registry(run_dir)
    refs = compute_references(run_dir, extra_keep=extra_keep)
    references = refs["references"]
    blocked = refs["blocked"]
    history = set(refs["attach_history"])

    resources: list[dict] = []
    space = {"referenced_seed_bytes": 0, "extra_kept_seed_bytes": 0,
             "attached_unreferenced_seed_bytes": 0, "unattached_seed_bytes": 0,
             "publish_tmp_leftover_bytes": 0, "corrupt_seed_bytes": 0,
             "failed_kept_scratch_bytes": 0, "lightweight_result_bytes": 0}
    for resource in view["resources"]:
        if resource["kind"] == CLASS_RECLAIMED:
            resources.append({**resource, "decision": DECISION_RECLAIMED,
                              "reasons": [("reclaimed earlier; the durable "
                                           "tombstone distinguishes this "
                                           "intentional deletion from "
                                           "corruption")]})
            continue
        if resource["kind"] == CLASS_TMP_LEFTOVER:
            space["publish_tmp_leftover_bytes"] += resource["bytes"]
            resources.append({**resource, "decision": DECISION_HOLD,
                              "reasons": [("publish interrupted before commit; "
                                           "classified, never auto-deleted")]})
            continue
        if resource["kind"] == CLASS_CORRUPT:
            space["corrupt_seed_bytes"] += resource["bytes"]
            reasons = [resource["reason"]]
            # A partially deleted tombstoned tree keeps its verifiable
            # generation identity (directory name + durable tombstone, no
            # manifest contradiction — attached by inspection); only then
            # may the interrupted deletion resume.  The tombstone, the
            # directory identity and the FRESH references decide together —
            # never a re-validation of payload that deletion already
            # removed.  A referenced-again tombstone stays suspended.
            generation = (resource.get("generation")
                          if resource.get("reclaim_tombstoned") else None)
            if generation is not None:
                if generation in references:
                    decision = DECISION_KEEP
                    reasons = (list(references[generation])
                               + [("a reclaim tombstone exists but the "
                                   "generation is referenced — the deletion "
                                   "stays suspended"), *reasons])
                elif blocked:
                    decision = DECISION_HOLD
                    reasons = ["reclaim blocked: " + "; ".join(blocked),
                               *reasons]
                else:
                    decision = DECISION_RECLAIM_CANDIDATE
                    reasons = [("the reclaim tombstone was persisted but the "
                                "deletion did not complete; resumable"),
                               *reasons]
            else:
                decision = DECISION_HOLD
            resources.append({**resource, "decision": decision,
                              "reasons": reasons})
            continue
        generation = resource["generation"]
        reasons = list(references.get(generation, []))
        if reasons:
            decision = DECISION_KEEP
            if resource.get("reclaim_tombstoned"):
                reasons.append("a reclaim tombstone exists but the "
                               "generation is referenced — the deletion "
                               "stays suspended")
            if set(references.get(generation, [])) == {"extra_keep"}:
                space["extra_kept_seed_bytes"] += resource["bytes"]
            else:
                space["referenced_seed_bytes"] += resource["bytes"]
        elif generation in history:
            if blocked:
                decision = DECISION_HOLD
                reasons = ["reclaim blocked: " + "; ".join(blocked)]
            else:
                decision = DECISION_RECLAIM_CANDIDATE
                if resource.get("reclaim_tombstoned"):
                    reasons = [("the reclaim tombstone was persisted but the "
                                "deletion did not complete; resumable")]
                else:
                    reasons = ["once attached, now unreferenced"]
            space["attached_unreferenced_seed_bytes"] += resource["bytes"]
        else:
            decision = DECISION_HOLD
            reasons = [("published but never attached; diagnose before any "
                        "reclaim (never auto-deleted, never auto-promoted)")]
            space["unattached_seed_bytes"] += resource["bytes"]
        resources.append({**resource, "decision": decision, "reasons": reasons})

    space["failed_kept_scratch_bytes"] = _failed_kept_scratch_bytes(run_dir)
    space["lightweight_result_bytes"] = _lightweight_result_bytes(run_dir)
    return {"run_dir": str(run_dir), "dry_run": True,
            "references": {str(g): r for g, r in sorted(references.items())},
            "blocked": blocked, "resources": resources, "space": space,
            "space_bound_note": (
                "normal serial progress with timely reclaim and a bounded "
                "extra_keep keeps at most (latest + retained-checkpoint refs "
                "+ the committed boundary + in-flight inputs + unconsumed "
                "producer seeds + extra_keep) seeds persistent; publish "
                "leftovers, failed_kept scratch and cleanup-failure retries "
                "are reported separately and are NOT part of that bound — no "
                "constant-whole-disk claim")}


def _plan_staleness(plan: dict, fresh: dict) -> str | None:
    """None when the caller-supplied dry-run plan still matches the fresh
    in-lock recomputation exactly (references, blocked reasons and the
    candidate set); else the difference — a stale plan is never a
    deletion credential."""
    if not isinstance(plan, dict) or plan.get("dry_run") is not True:
        return "the supplied plan is not a dry-run plan"
    if plan.get("references") != fresh["references"]:
        return "the reference set changed since the plan was computed"
    if plan.get("blocked") != fresh["blocked"]:
        return "the blocked reasons changed since the plan was computed"

    def _candidates(p: dict) -> list[int]:
        return sorted(int(r["generation"]) for r in p["resources"]
                      if r.get("decision") == DECISION_RECLAIM_CANDIDATE)

    if _candidates(plan) != _candidates(fresh):
        return "the reclaim candidate set changed since the plan was computed"
    return None


def _delete_generation_tree(registry: Path, name: str) -> int:
    """Delete one generation tree, anchored: the directory must still be a
    confined, non-symlink child of the registry at deletion time.  Returns
    the removed byte count."""
    directory = registry / name
    _check_generation_confined(registry, directory)
    removed = _tree_bytes(directory)
    shutil.rmtree(directory)
    _fsync_dir(registry)
    return removed


def execute_density_reclaim(run_dir: str | Path, *,
                            extra_keep: tuple | list = (),
                            plan: dict | None = None,
                            event_log=None) -> dict:
    """Actually reclaim the unreferenced owned old generations — the
    narrow execution entry over :func:`plan_density_reclaim`.

    The run's single-writer lock is held for the whole execution: pass the
    run's open ``event_log`` when the caller already holds it (the serial
    driver does), otherwise it is taken here — a live run (or a stale
    lock) refuses.  References are RE-READ under the lock and a caller
    supplied ``plan`` must match that fresh computation exactly — a stale
    plan is refused.  Per candidate, in order: the generation is
    re-validated (an interrupted deletion requires its tombstone, a
    readable matching ownership manifest and fresh unreferenced status;
    missing, unreadable or foreign ownership stays held), the durable
    tombstone is persisted, then the tree is
    deleted.  A per-generation failure is recorded in the receipt and
    never blocks the others; the tombstone keeps an interrupted deletion
    resumable and a completed one forever distinguishable from
    corruption.  Only generations this run fully owns — attached once,
    validated, unreferenced — are touched: publish leftovers, corrupt or
    unattached directories, external sources and anything outside the
    registry are never auto-deleted.

    Returns a receipt: ``ok`` (nothing failed; possibly nothing to do),
    ``incomplete`` (some candidates failed — inspectable, resumable) or
    ``refused`` (lock held or stale plan; nothing deleted).
    """
    run_dir = Path(run_dir).resolve()
    owns_log = event_log is None
    if owns_log:
        try:
            event_log = EventLog(run_dir)
        except EventLogError as error:
            return {"run_dir": str(run_dir), "dry_run": False,
                    "status": "refused",
                    "reason": f"cannot take the run lock: {error}"}
    try:
        fresh = plan_density_reclaim(run_dir, extra_keep=extra_keep)
        if plan is not None:
            stale = _plan_staleness(plan, fresh)
            if stale is not None:
                return {"run_dir": str(run_dir), "dry_run": False,
                        "status": "refused",
                        "reason": f"stale plan: {stale}; recompute the dry "
                                  "run against the live run"}
        registry = _check_registry_confined(run_dir)
        candidates = sorted(
            (r for r in fresh["resources"]
             if r.get("decision") == DECISION_RECLAIM_CANDIDATE),
            key=lambda r: int(r["generation"]))
        reclaimed: list[dict] = []
        failed: list[dict] = []
        for resource in candidates:
            generation = int(resource["generation"])
            name = resource["name"]
            try:
                if not resource.get("reclaim_tombstoned"):
                    directory = registry / name
                    manifest = _read_manifest(directory)
                    if manifest is None:
                        raise RestartError(
                            "the manifest is unreadable at deletion time")
                    invalid = _validate_generation(directory, manifest,
                                                   registry=registry,
                                                   run_dir=run_dir)
                    if invalid is not None:
                        raise RestartError(
                            f"the generation no longer validates: {invalid}")
                    _tombstone_reclaimed(registry, generation)
                removed = _delete_generation_tree(registry, name)
                reclaimed.append({"generation": generation, "bytes": removed})
            except (RestartError, OSError) as error:
                failed.append({"generation": generation,
                               "reason": f"{type(error).__name__}: {error}"})
        return {"run_dir": str(run_dir), "dry_run": False,
                "status": "ok" if not failed else "incomplete",
                "reclaimed": reclaimed, "failed": failed,
                "references": fresh["references"],
                "space": fresh["space"],
                "reason": (None if not failed else
                           "some candidates could not be deleted; their "
                           "tombstones persist and a later execution "
                           "resumes them")}
    finally:
        if owns_log:
            event_log.close()


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _failed_kept_scratch_bytes(run_dir: Path) -> int:
    records_dir = run_dir / "scratch_records"
    total = 0
    if not records_dir.is_dir():
        return 0
    for record_path in sorted(records_dir.rglob("*.json")):
        record = _read_json_quiet(record_path)
        if record is None:
            continue
        if record.get("state") == "failed_kept":
            scratch_dir = record.get("scratch_dir")
            if scratch_dir and Path(scratch_dir).is_dir():
                total += _tree_bytes(Path(scratch_dir))
    return total


def _lightweight_result_bytes(run_dir: Path) -> int:
    total = 0
    for name in ("trajectory.db", "events.jsonl", "summary.json",
                 "summary.csv", "resolved_config.json", "manifest.json",
                 "trajectory.extxyz"):
        path = run_dir / name
        if path.is_file():
            total += path.stat().st_size
    return total


def checkpoint_manifest_density_field(generation_record: dict) -> dict:
    """The ``manifest_extra`` a CheckpointManager.write caller adds so the
    checkpoint records which density generation its run state depends on.
    Pure helper — the checkpoint layout itself is unchanged."""
    return {"density_generation": int(generation_record["generation"])}


__all__ = [
    "RESTART_SCHEMA",
    "CheckpointManager",  # re-export for callers wiring the two layouts
    "RestartError",
    "checkpoint_manifest_density_field",
    "compute_references",
    "execute_density_reclaim",
    "inspect_density_registry",
    "latest_density_generation",
    "plan_density_reclaim",
    "publish_density_generation",
    "publish_density_generation_from_attempt",
    "resolve_density_for_resume",
]
