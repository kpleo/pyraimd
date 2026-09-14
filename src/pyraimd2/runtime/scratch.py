"""Managed scratch lifecycle: one unified temporary root, exclusive
attempt directories, durable-result archival and idempotent reclaim.

A run's *authoritative* results always live outside the scratch root —
solvers only borrow ``<root>/<run-uuid>/<backend-role>/<request-uuid>/
<attempt-id>/`` for their working files.  The durable result is archived
into the persistent run root with a content-verified copy (each declared
file compared against its validated source's length and SHA-256) before
the attempt's scratch subtree may be reclaimed; the shared root itself
is never recursively deleted (only empty per-run parent directories are
removed).  This module knows nothing about QE, wavefunctions or file
suffixes — adapters declare which small outputs make up the durable
result and which working directory the solver used.

State chain per attempt (recorded atomically OUTSIDE the scratch root):
``allocated`` → ``archived`` → ``cleaned``; failures stay
``failed_kept``; an archived attempt whose cleanup failed is
``cleanup_pending`` and safely retryable — never a DFT rerun.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

SCRATCH_ATTEMPT_RECORD = "scratch-attempt-v2"
SCRATCH_OWNER_RECORD = "scratch-owner-v2"
SCRATCH_RECORD_DIR = "scratch_records"
RETENTION_MODES = ("all", "results")
RECORD_STATES = ("allocated", "archived", "archive_failed", "failed_kept",
                 "kept", "cleanup_pending", "cleaned")
_WRITE_COUNTER = {"n": 0}

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

# anchored fd-relative deletion needs a no-follow open flag, dir_fd-aware
# syscalls and fd-aware scandir; where any is missing the safe answer is a
# refusal, never an unanchored delete
_SAFE_DELETE_CAPABLE = (
    hasattr(os, "O_NOFOLLOW")
    and os.scandir in os.supports_fd
    and os.stat in os.supports_follow_symlinks
    and {os.open, os.stat, os.unlink, os.rmdir} <= os.supports_dir_fd
)


class ScratchError(RuntimeError):
    """A scratch allocation, archival or cleanup step failed honestly."""


# ordinary maintenance failures that must never cost a delivered label
_MAINTENANCE_ERRORS = (OSError, ScratchError, ValueError, TypeError,
                       KeyError)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Durable small-record write: a uniquely named temp file (no shared
    scratch name two writers can clobber), flush+fsync, atomic replace in
    the destination's own filesystem, then the parent directory fsynced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _WRITE_COUNTER["n"] += 1
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}-{time.time_ns()}-{_WRITE_COUNTER['n']}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str, what: str) -> str:
    """A user-provided identity must be one confined path component —
    never an absolute path, a parent traversal or a separator."""
    if not value or value in (".", "..") or "/" in value or "\\" in value \
            or os.path.isabs(value):
        raise ScratchError(
            f"{what} must be a single confined path component, got {value!r}")
    return value


def _valid_declared_name(name: str) -> str:
    if not name or os.path.isabs(name) or name.startswith("..") \
            or "/../" in name or name.endswith("/..") or "\\" in name:
        raise ScratchError(
            f"declared durable result name escapes the archive: {name!r}")
    return name


def _validate_record(record: dict, *, path: Path) -> dict:
    """Strict authoritative-record validation: missing fields, wrong
    types, unknown schema, empty identity or an empty archive manifest
    are never silently treated as verified."""
    if record.get("record") != SCRATCH_ATTEMPT_RECORD:
        raise ScratchError(
            f"unsupported scratch record schema in {path}: "
            f"{record.get('record')!r}")
    for key in ("run_uuid", "backend_role", "request_id", "attempt_id",
                "run_root", "scratch_root", "scratch_dir", "archive_dir",
                "retention", "state"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ScratchError(
                f"scratch record {path} misses a usable {key!r}")
    if record["state"] not in RECORD_STATES:
        raise ScratchError(
            f"scratch record {path} has unknown state {record['state']!r}")
    if record["retention"] not in RETENTION_MODES:
        raise ScratchError(
            f"scratch record {path} has unknown retention "
            f"{record['retention']!r}")
    identity = record.get("identity")
    if not isinstance(identity, dict) \
            or not isinstance(identity.get("dev"), int) \
            or not isinstance(identity.get("ino"), int):
        raise ScratchError(
            f"scratch record {path} has no complete pinned identity")
    root = Path(record["scratch_root"])
    scratch_dir = Path(record["scratch_dir"])
    archive_dir = Path(record["archive_dir"])
    if not root.is_absolute() or not archive_dir.is_absolute():
        raise ScratchError(f"scratch record {path} has non-canonical paths")
    if root not in scratch_dir.parents:
        raise ScratchError(
            f"scratch record {path}: scratch_dir is outside its root")
    if scratch_dir != root / record["run_uuid"] / record["backend_role"] \
            / record["request_id"] / record["attempt_id"]:
        raise ScratchError(
            f"scratch record {path}: path does not match its identity")
    if archive_dir == scratch_dir or archive_dir in scratch_dir.parents \
            or scratch_dir in archive_dir.parents:
        raise ScratchError(
            f"scratch record {path}: archive overlaps the scratch subtree")
    if record["state"] in ("archived", "cleanup_pending", "cleaned"):
        archived = record.get("archived")
        if not isinstance(archived, list) or not archived:
            raise ScratchError(
                f"scratch record {path} has no verifiable archived manifest")
        for entry in archived:
            if not isinstance(entry, dict):
                raise ScratchError(
                    f"scratch record {path}: malformed archived entry")
            name = entry.get("file")
            if not isinstance(name, str) or not name:
                raise ScratchError(
                    f"scratch record {path}: archived entry without a name")
            _valid_declared_name(name)
            if not isinstance(entry.get("bytes"), int) \
                    or not isinstance(entry.get("sha256"), str) \
                    or len(entry["sha256"]) != 64:
                raise ScratchError(
                    f"scratch record {path}: archived entry without a "
                    "complete size/SHA-256")
    return record


@dataclass
class AttemptScratch:
    """One attempt's managed scratch handle (persistent record outside)."""

    run_uuid: str
    backend_role: str
    request_id: str
    attempt_id: str
    scratch_dir: Path
    archive_dir: Path
    record_path: Path

    def load_record(self) -> dict:
        record = _read_json(self.record_path)
        if record is None:
            raise ScratchError(
                f"unreadable scratch record at {self.record_path}; refusing "
                "to touch the attempt's scratch without its ownership record")
        return _validate_record(record, path=self.record_path)

    def update_record(self, **fields) -> dict:
        record = self.load_record()
        record.update(fields)
        _atomic_write_json(self.record_path, record)
        return record


def _checked_root(scratch_root: str | Path) -> Path:
    root = Path(scratch_root)
    if not root.is_absolute():
        raise ScratchError(
            f"scratch root must be absolute at engine use, got {scratch_root}")
    return root


def allocate(*, run_root: str | Path, scratch_root: str | Path,
             run_uuid: str, backend_role: str, request_id: str,
             attempt_id: str, archive_dir: str | Path,
             retention: str) -> AttemptScratch:
    """Atomically claim this attempt's exclusive scratch directory and
    its authoritative record (outside the scratch root).

    The canonical root, the run/request/attempt identity and the created
    directory's object identity are pinned at allocation — a later record
    alone can never move the target.  Identities from callers are confined
    to single path components.  Uniqueness comes from the generated
    identities, never from a user-chosen label; a collision is an error,
    never a silent merge.
    """
    run_root = Path(run_root).resolve()
    root = _checked_root(scratch_root).resolve()
    request_id = _safe_component(str(request_id), "request_id")
    attempt_id = _safe_component(str(attempt_id), "attempt_id")
    backend_role = _safe_component(str(backend_role), "backend_role")
    run_uuid = _safe_component(str(run_uuid), "run_uuid")
    archive_dir = Path(archive_dir).resolve()
    scratch_dir = root / run_uuid / backend_role / request_id / attempt_id
    if archive_dir == scratch_dir or archive_dir in scratch_dir.parents \
            or scratch_dir in archive_dir.parents:
        raise ScratchError(
            "archive_dir must be independent of the reclaimed scratch "
            f"subtree: archive {archive_dir} vs scratch {scratch_dir}")
    try:
        scratch_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ScratchError(
            f"scratch attempt directory already exists: {scratch_dir}"
        ) from error
    created = scratch_dir.stat()
    handle = AttemptScratch(
        run_uuid=run_uuid, backend_role=backend_role, request_id=request_id,
        attempt_id=attempt_id, scratch_dir=scratch_dir,
        archive_dir=archive_dir,
        record_path=(run_root / SCRATCH_RECORD_DIR / run_uuid
                     / f"{request_id}--{attempt_id}.json"))
    # the small disposable ownership marker inside scratch: enough to
    # find the authoritative run root, never the only cleanup evidence
    _atomic_write_json(root / run_uuid / "owner.json", {
        "record": SCRATCH_OWNER_RECORD,
        "run_uuid": run_uuid,
        "run_root": str(run_root),
        "created_unix": time.time(),
    })
    _atomic_write_json(handle.record_path, {
        "record": SCRATCH_ATTEMPT_RECORD,
        "run_uuid": run_uuid,
        "backend_role": backend_role,
        "request_id": request_id,
        "attempt_id": attempt_id,
        "run_root": str(run_root),
        "scratch_root": str(root),
        "scratch_dir": str(scratch_dir),
        "archive_dir": str(archive_dir),
        "identity": {"dev": created.st_dev, "ino": created.st_ino},
        "retention": retention,
        "state": "allocated",
        "created_unix": time.time(),
    })
    return handle


def archive(handle: AttemptScratch, files: list[str]) -> dict:
    """Durably copy the adapter-declared result files from the scratch
    directory into the persistent archive directory — each copy verified
    against its VALIDATED SOURCE's length and SHA-256 — and only then
    mark the attempt ``archived``.

    A copy that returns "success" with content that does not match the
    source is not an archive: the scratch is kept and no reclaimable
    state is published.  Declared names are confined relatives; the
    destination must land inside the archive directory (no parent
    traversal or link can drop the persistent output into the reclaimed
    subtree).  Any failure leaves the state un-archived and raises.
    """
    record = handle.load_record()
    if record["state"] not in ("allocated", "archive_failed"):
        raise ScratchError(
            f"cannot archive attempt in state {record['state']!r}")
    archived = []
    try:
        for raw_name in files:
            name = _valid_declared_name(str(raw_name))
            source = handle.scratch_dir / name
            if not source.is_file() or source.is_symlink():
                raise ScratchError(
                    f"declared durable result is not a regular file: {source}")
            # the expectation comes from the VALIDATED SOURCE, never from
            # re-hashing the destination alone
            expected_bytes = source.stat().st_size
            expected_sha256 = _sha256(source)
            destination = handle.archive_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if handle.archive_dir not in destination.resolve().parents:
                raise ScratchError(
                    f"archived destination escapes the archive directory: "
                    f"{destination}")
            tmp = destination.with_name(
                f".{destination.name}.{os.getpid()}-{time.time_ns()}.tmp")
            with source.open("rb") as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(tmp, destination)
            if destination.stat().st_size != expected_bytes \
                    or _sha256(destination) != expected_sha256:
                raise ScratchError(
                    f"archived copy does not match its source: {destination}")
            archived.append({"file": name, "bytes": expected_bytes,
                             "sha256": expected_sha256})
        _fsync_dir(handle.archive_dir)
    except Exception as error:
        handle.update_record(state="archive_failed", error=repr(error))
        raise
    return handle.update_record(state="archived", archived=archived,
                                archived_unix=time.time(), error=None)


def mark_failed(handle: AttemptScratch, error: str) -> dict:
    """A computation that never produced an archive is kept, with the
    reason — never silently cleaned, never auto-deleted by age.  An
    already-archived attempt keeps its archived state instead.  A record
    that cannot be updated does not mask the real error."""
    record = handle.load_record()
    if record["state"] == "archived":
        return record
    try:
        return handle.update_record(state="failed_kept", error=error,
                                    failed_unix=time.time())
    except _MAINTENANCE_ERRORS:
        return record


def _verify_identity_path(record: dict) -> Path:
    """Re-walk the pinned path level by level: no component may be a
    symlink, the target must still be exactly the allocated directory
    (same dev/ino pinned at allocation), and it must sit inside the
    pinned canonical root — a parent-symlink replacement, a sibling swap
    or an out-of-scope target is never the original attempt."""
    root = Path(record["scratch_root"])
    scratch_dir = Path(record["scratch_dir"])
    relative = scratch_dir.relative_to(root)
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if current.is_symlink():
            raise ScratchError(
                f"a parent of the attempt was replaced by a symlink: {current}")
    if scratch_dir.is_symlink():
        raise ScratchError(
            f"attempt scratch path was replaced by a symlink: {scratch_dir}")
    if not scratch_dir.is_dir():
        raise ScratchError(
            f"attempt scratch directory is missing: {scratch_dir}")
    pinned = record.get("identity") or {}
    actual = scratch_dir.stat()
    if int(pinned["dev"]) != actual.st_dev \
            or int(pinned["ino"]) != actual.st_ino:
        raise ScratchError(
            f"attempt directory identity changed since allocation "
            f"(pinned {pinned}, now dev={actual.st_dev} ino={actual.st_ino})")
    return scratch_dir


def _verify_archived_content(record: dict) -> list[str]:
    """Every file the archive committed must still exist in the archive
    as a regular file with its recorded SHA-256 — a missing, corrupted,
    same-size-tampered or link-replaced result keeps the scratch, with
    the reason named."""
    archive_dir = Path(record["archive_dir"])
    problems = []
    for entry in record["archived"]:
        path = archive_dir / entry["file"]
        if path.is_symlink() or not path.is_file():
            problems.append(f"archived result not a regular file: {path}")
            continue
        if path.stat().st_size != entry["bytes"]:
            problems.append(
                f"archived result size changed: {path} "
                f"({path.stat().st_size} != recorded {entry['bytes']})")
            continue
        if _sha256(path) != entry["sha256"]:
            problems.append(f"archived result content changed: {path}")
    return problems


def mark_kept(handle: AttemptScratch) -> dict:
    """``all`` retention: record the kept state under the same non-fatal
    maintenance boundary; the receipt says what happened."""
    try:
        handle.update_record(state="kept")
        return {"status": "kept"}
    except _MAINTENANCE_ERRORS as error:
        return {"status": "keep_persist_failed", "error": repr(error)}


class _AttemptLock:
    """A tiny per-attempt reclaimer mutex on a kernel-released lock
    (``fcntl.flock``): an active competitor is blocked, and the lock is
    released by the kernel when the holder exits or is killed — a stale
    file alone never blocks a later retry.  Never unlinked while held
    (that would create two locks on different inodes); no payload is
    written to the file, the kernel lock itself is the whole protocol.
    On platforms without flock, acquisition is a clear, safe refusal.
    Every fd is released exactly once; an unlock that reports an error
    is captured in ``exit_error`` (closing the fd still frees the kernel
    lock) instead of escaping into the caller's label path."""

    def __init__(self, handle: AttemptScratch):
        self.path = handle.record_path.with_suffix(".lock")
        self.fd = None
        self.exit_error = None

    def __enter__(self):
        if fcntl is None:
            raise ScratchError(
                "per-attempt reclaimer locking requires fcntl.flock on "
                "this platform; cleanup refused rather than run unguarded")
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o644)
        except OSError as error:
            raise ScratchError(
                f"cannot create the attempt lock {self.path}: {error}"
            ) from error
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._release_fd()
            if isinstance(error, BlockingIOError):
                raise ScratchError(
                    f"another reclaimer holds the attempt lock {self.path}"
                ) from error
            raise ScratchError(
                f"cannot take the attempt lock {self.path}: {error}"
            ) from error
        return self

    def _release_fd(self) -> None:
        """Close the held fd exactly once (closing releases the kernel
        lock even when an explicit LOCK_UN reported an error)."""
        if self.fd is None:
            return
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = None

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except OSError as error:
                self.exit_error = repr(error)
            self._release_fd()
        return False


def _open_child_fd(parent_fd: int, name: str,
                   expect: os.stat_result) -> tuple[int, os.stat_result]:
    """Open the directory ``name`` below the held parent fd without
    following links (a swapped-in symlink is refused by O_NOFOLLOW) and
    confirm the opened object is still the exact directory ``expect``
    statted a moment ago — an object replaced in that gap is never
    entered.  The fd is closed on every failure path."""
    child_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        actual = os.fstat(child_fd)
    except OSError:
        os.close(child_fd)
        raise
    if not stat.S_ISDIR(actual.st_mode):
        os.close(child_fd)
        raise ScratchError(
            f"attempt identity changed at the deletion boundary: {name!r} "
            "is no longer a directory")
    if actual.st_dev != expect.st_dev or actual.st_ino != expect.st_ino:
        os.close(child_fd)
        raise ScratchError(
            f"attempt identity changed at the deletion boundary: {name!r} "
            "was replaced between the check and the open")
    return child_fd, actual


def _step_down_fd(parent_fd: int, name: str) -> int:
    """Move the held directory fd one level down: the child must be a
    plain directory (never a link), opened relative to the held parent
    and identity-checked; the parent fd is closed as ownership moves."""
    listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(listed.st_mode):
        raise ScratchError(
            f"attempt identity changed at the deletion boundary: parent "
            f"{name!r} is no longer a plain directory")
    child_fd, _ = _open_child_fd(parent_fd, name, expect=listed)
    os.close(parent_fd)
    return child_fd


def _descend_and_empty(dir_fd: int) -> int:
    """Empty the held directory fd (the CPython fd-safe rmtree descent):
    each child is examined without following links; a directory child is
    opened relative to this fd and confirmed to be the same object
    before it is entered — a swap for a link to a live sibling is
    refused, never descended into; anything else is unlinked relative
    to this fd.  Returns the total size of the unlinked files."""
    removed = 0
    with os.scandir(dir_fd) as entries:
        for entry in entries:
            name = entry.name
            if name in (".", ".."):
                continue
            listed = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(listed.st_mode):
                child_fd, _ = _open_child_fd(dir_fd, name, expect=listed)
                try:
                    removed += _descend_and_empty(child_fd)
                finally:
                    os.close(child_fd)
                os.rmdir(name, dir_fd=dir_fd)
            else:
                os.unlink(name, dir_fd=dir_fd)
                removed += listed.st_size
    return removed


def _delete_tree_via_fd(record: dict) -> int:
    """Delete the pinned attempt subtree anchored on held directory fds:
    from the canonical root each level is opened relative to its held
    parent without following links and confirmed against the object
    statted just before; the attempt directory itself must match the
    dev/ino pinned at allocation; the final rmdir is issued relative to
    the held parent fd — the target is never re-located by its original
    absolute string.  Platforms without dir_fd/O_NOFOLLOW/fd-scandir
    support refuse safely instead of deleting unanchored.  Returns the
    total size of the removed files."""
    if not _SAFE_DELETE_CAPABLE:
        raise ScratchError(
            "anchored recursive deletion requires dir_fd, O_NOFOLLOW and "
            "fd-scandir support on this platform; cleanup refused rather "
            "than run unanchored")
    root = Path(record["scratch_root"])
    relative = Path(record["scratch_dir"]).relative_to(root)
    pinned = record["identity"]
    parent_fd = os.open(root, os.O_RDONLY)
    try:
        for component in relative.parts[:-1]:
            parent_fd = _step_down_fd(parent_fd, component)
        name = relative.parts[-1]
        listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(listed.st_mode):
            raise ScratchError(
                f"attempt identity changed at the deletion boundary: "
                f"{name!r} is no longer a plain directory")
        target_fd, target = _open_child_fd(parent_fd, name, expect=listed)
        if target.st_dev != pinned["dev"] or target.st_ino != pinned["ino"]:
            os.close(target_fd)
            raise ScratchError(
                f"attempt directory identity changed at the deletion "
                f"boundary (pinned dev={pinned['dev']} "
                f"ino={pinned['ino']}, now dev={target.st_dev} "
                f"ino={target.st_ino})")
        try:
            removed = _descend_and_empty(target_fd)
        finally:
            os.close(target_fd)
        os.rmdir(name, dir_fd=parent_fd)
        return removed
    finally:
        os.close(parent_fd)


def cleanup(handle: AttemptScratch) -> dict:
    """Reclaim an ARCHIVED attempt's scratch subtree (idempotent).

    Returns a complete receipt dict — ``cleaned``, ``already_cleaned``,
    ``failed`` (retryable, recorded ``cleanup_pending``), or ``refused``
    with the reason (unarchived, unsafe, corrupted archive, lock busy).
    A missing or already-reclaimed directory is a correct end state,
    never a computation failure.  Nothing is deleted when the ownership
    cannot be verified, when the archived result no longer matches its
    recorded content, or when the pre-delete state cannot be durably
    persisted.  One maintenance boundary covers the whole public entry —
    the record read, the filesystem state queries (an EIO from
    ``exists`` is a query failure with a reason, never misclassified as
    "directory absent" and marked ``already_cleaned``), entering the
    lock, the in-lock maintenance, exiting the lock and the terminal
    state return: any ordinary failure there becomes a receipt with the
    reason, never a lost label; a reclaim that completed keeps its
    ``cleaned`` fact with the maintenance or persist error attached
    (``maintenance_error``/``persist_error``).  Interrupts
    (KeyboardInterrupt/SystemExit) always propagate.
    """
    try:
        return _cleanup_impl(handle)
    except _MAINTENANCE_ERRORS as error:
        return {"status": "failed", "error": str(error)}


def _cleanup_impl(handle: AttemptScratch) -> dict:
    """The entry flow behind the public maintenance boundary: state
    chain, absent-directory finish and the locked reclaim lifecycle.
    Ordinary failures that escape its local catches are turned into a
    receipt by the public wrapper."""
    try:
        record = handle.load_record()
    except _MAINTENANCE_ERRORS as error:
        return {"status": "refused", "reason": str(error)}
    state = record["state"]
    scratch_dir = Path(record["scratch_dir"])
    if state == "cleaned":
        return {"status": "already_cleaned", "state": state}
    if state not in ("archived", "cleanup_pending"):
        return {"status": "refused", "reason": f"state {state!r} is not "
                    "archived — nothing safe to reclaim"}
    if not scratch_dir.exists():
        try:
            handle.update_record(state="cleaned",
                                 cleaned_note="directory already absent")
        except _MAINTENANCE_ERRORS as error:
            return {"status": "already_cleaned",
                    "state": "directory already absent",
                    "persist_error": repr(error)}
        return {"status": "already_cleaned",
                "state": "directory already absent"}
    lock = _AttemptLock(handle)
    try:
        with lock:
            receipt = _cleanup_under_lock(handle, record)
    except _MAINTENANCE_ERRORS as error:
        return {"status": "failed", "error": str(error)}
    if lock.exit_error is not None:
        # the maintenance work finished; only the lock release reported
        # an error (the closed fd still freed the kernel lock) — the
        # receipt keeps the actual outcome plus the maintenance error
        receipt["maintenance_error"] = lock.exit_error
    return receipt


def _cleanup_under_lock(handle: AttemptScratch, record: dict) -> dict:
    """The locked reclaim critical section: the pre-delete state is made
    durable before any removal, identity and archived content are
    re-verified inside the lock, the anchored deletion runs, then the
    terminal state is persisted.  Ordinary failures become the distinct
    receipts above; nothing here raises into the label path."""
    try:
        # the pre-delete state must be durable before any removal
        handle.update_record(state="cleanup_pending",
                             cleanup_started_unix=time.time())
    except _MAINTENANCE_ERRORS as error:
        return {"status": "failed",
                "error": f"cannot persist the pre-cleanup state: "
                         f"{error!r}"}
    try:
        _verify_identity_path(record)
    except _MAINTENANCE_ERRORS as error:
        _try_mark_cleanup_failed(handle, repr(error))
        return {"status": "refused", "reason": str(error)}
    try:
        problems = _verify_archived_content(record)
    except _MAINTENANCE_ERRORS as error:
        _try_mark_cleanup_failed(handle, repr(error))
        return {"status": "failed",
                "error": f"archived content could not be "
                         f"re-verified: {error!r}"}
    if problems:
        reason = ("archived result no longer verifies — scratch "
                  "kept: " + "; ".join(problems))
        _try_mark_cleanup_failed(handle, reason)
        return {"status": "refused", "reason": reason}
    try:
        removed_bytes = _delete_tree_via_fd(record)
    except _MAINTENANCE_ERRORS as error:
        _try_mark_cleanup_failed(handle, repr(error))
        return {"status": "failed", "error": repr(error)}
    # best-effort empty-parent finish: rmdir only ever removes an EMPTY
    # directory, so this can do less but can never delete content
    parent = Path(record["scratch_dir"]).parent
    run_dir = Path(record["scratch_root"]) / record["run_uuid"]
    while parent != run_dir and parent != parent.parent:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    try:
        handle.update_record(state="cleaned",
                             cleaned_unix=time.time(),
                             removed_bytes=removed_bytes)
    except _MAINTENANCE_ERRORS as error:
        return {"status": "cleaned",
                "removed_bytes": removed_bytes,
                "persist_error": repr(error)}
    return {"status": "cleaned", "removed_bytes": removed_bytes}


def _try_mark_cleanup_failed(handle: AttemptScratch, error: str) -> None:
    """Best-effort cleanup_pending marking; a failed write never raises
    into the label path."""
    try:
        handle.update_record(state="cleanup_pending", error=error,
                             cleanup_failed_unix=time.time())
    except _MAINTENANCE_ERRORS:
        pass


def _record_view(record_path: Path) -> dict:
    """One record for the inspect view: valid records report their
    state; anything else is an unreadable entry with a reason — never a
    crash and never a guard bypass."""
    record = _read_json(record_path)
    if record is None:
        return {"state": "unreadable", "record": record_path.name,
                "request_id": "?", "attempt_id": "?", "scratch_dir": "",
                "retention": None, "present_bytes": 0,
                "error": "not a JSON object"}
    try:
        record = _validate_record(record, path=record_path)
    except _MAINTENANCE_ERRORS as error:
        return {"state": "unreadable", "record": record_path.name,
                "request_id": record.get("request_id", "?"),
                "attempt_id": record.get("attempt_id", "?"),
                "scratch_dir": str(record.get("scratch_dir", "")),
                "retention": record.get("retention"), "present_bytes": 0,
                "error": str(error)}
    scratch_dir = Path(record["scratch_dir"])
    size = (sum(p.stat().st_size for p in scratch_dir.rglob("*")
                if p.is_file() and not p.is_symlink())
            if scratch_dir.is_dir() else 0)
    return {
        "request_id": record["request_id"],
        "attempt_id": record["attempt_id"],
        "state": record["state"],
        "retention": record["retention"],
        "scratch_dir": str(scratch_dir),
        "present_bytes": size,
        "error": record.get("error")}


def inspect_root(scratch_root: str | Path) -> dict:
    """One read-only view of a scratch root: managed runs with their
    attempts' states, retention reasons and reclaimable sizes, plus
    unknown entries (never touched by a manager, always kept).  A single
    corrupt record is listed as unreadable — it never crashes the view
    and never bypasses a guard."""
    root = _checked_root(scratch_root).resolve()
    entries: list[dict] = []
    unknown: list[str] = []
    if not root.is_dir():
        return {"root": str(root), "runs": entries, "unknown": unknown,
                "note": "root does not exist"}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.is_symlink():
            unknown.append(run_dir.name)
            continue
        owner = _read_json(run_dir / "owner.json")
        if owner is None or owner.get("record") != SCRATCH_OWNER_RECORD \
                or not isinstance(owner.get("run_uuid"), str) \
                or not isinstance(owner.get("run_root"), str):
            unknown.append(run_dir.name)
            continue
        run_root = Path(str(owner["run_root"]))
        attempts = []
        records_dir = run_root / SCRATCH_RECORD_DIR / str(owner["run_uuid"])
        for record_path in sorted(records_dir.glob("*.json")) \
                if records_dir.is_dir() else []:
            attempts.append(_record_view(record_path))
        entries.append({"run_uuid": owner["run_uuid"],
                        "run_root": str(run_root),
                        "attempts": attempts})
    return {"root": str(root), "runs": entries, "unknown": unknown}


def _reclaimable_state(record: dict) -> bool:
    """The archived→cleanup window: already ``cleanup_pending``, or
    ``archived`` whose recorded retention is ``results`` (an ``all``
    retention is never auto-reclaimed)."""
    if record["state"] == "cleanup_pending":
        return True
    return record["state"] == "archived" \
        and record["retention"] == "results"


def _reclaim_refusal_reason(record: dict) -> str | None:
    """The side-effect-free reclaimability judgment shared by the dry
    run and the real reclaim: the same strict identity and
    archived-content verification the locked reclaim re-runs (the locked
    re-check stays authoritative — a preview is never an authorization).
    A reason string means the attempt is kept; ``None`` means a real
    reclaim would proceed (an already-absent directory is only
    finalized, never an obstacle).  A query error (an EIO from
    ``exists`` is not "absent") is a reason, never an exception that
    ends the whole batch."""
    try:
        if not Path(record["scratch_dir"]).exists():
            return None
    except _MAINTENANCE_ERRORS as error:
        return f"scratch state could not be queried: {error!r}"
    try:
        _verify_identity_path(record)
    except _MAINTENANCE_ERRORS as error:
        return str(error)
    try:
        problems = _verify_archived_content(record)
    except _MAINTENANCE_ERRORS as error:
        return f"archived content could not be re-verified: {error!r}"
    if problems:
        return ("archived result no longer verifies — scratch kept: "
                + "; ".join(problems))
    return None


def clean_pending(scratch_root: str | Path, *, dry_run: bool = True) -> dict:
    """Idempotent retry of reclaimable attempts under one root.

    Only attempts whose record strictly validates, proves the archived
    state and the ``results`` retention, and whose ownership still
    verifies are reclaimed; unknown, failed, active or corrupt entries
    are listed with their reason in ``kept`` — the dry run applies the
    same judgment, never claiming a reclaimable set the real run would
    refuse.  With ``dry_run=True`` nothing is touched.
    """
    root = _checked_root(scratch_root).resolve()
    reclaimable, kept, receipts = [], [], []
    for run in inspect_root(root)["runs"]:
        run_root = Path(run["run_root"])
        for attempt in run["attempts"]:
            key = {"run_uuid": run["run_uuid"],
                   "request_id": attempt["request_id"],
                   "attempt_id": attempt["attempt_id"],
                   "scratch_dir": attempt["scratch_dir"]}
            if attempt["state"] == "unreadable":
                kept.append({**key,
                             "reason": f"record unreadable "
                                       f"({attempt['error']})"})
                continue
            record_path = (run_root / SCRATCH_RECORD_DIR
                           / run["run_uuid"]
                           / f"{attempt['request_id']}--"
                             f"{attempt['attempt_id']}.json")
            try:
                record = _validate_record(
                    _read_json(record_path) or {}, path=record_path)
            except _MAINTENANCE_ERRORS as error:
                kept.append({**key, "reason": f"record unreadable "
                                              f"({error})"})
                continue
            if not _reclaimable_state(record):
                kept.append({**key,
                             "reason": f"state {attempt['state']!r}"})
                continue
            # ownership verification before any deletion: the record's
            # own paths must still point inside this root and run root
            if not Path(record["archive_dir"]).resolve().is_relative_to(
                    run_root) or \
                    not Path(record["scratch_dir"]).resolve().is_relative_to(
                        root):
                kept.append({**key,
                             "reason": "ownership paths no longer verify"})
                continue
            # the dry run and the real run share one eligibility check
            # (strict identity + archived content); execution re-verifies
            # it inside the lock before anything is removed
            reason = _reclaim_refusal_reason(record)
            if reason is not None:
                kept.append({**key, "reason": reason})
                continue
            handle = AttemptScratch(
                run_uuid=run["run_uuid"],
                backend_role=record["backend_role"],
                request_id=record["request_id"],
                attempt_id=record["attempt_id"],
                scratch_dir=Path(record["scratch_dir"]),
                archive_dir=Path(record["archive_dir"]),
                record_path=record_path)
            reclaimable.append(key)
            if not dry_run:
                receipts.append({**key, **cleanup(handle)})
    return {"root": str(root), "dry_run": dry_run,
            "reclaimable": reclaimable, "kept": kept,
            "unknown": inspect_root(root)["unknown"], "receipts": receipts}
