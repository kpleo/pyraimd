"""Shared successful-density publication bridge for the QE adapters.

Opt-in per engine: pass ``density_registry_run_dir`` naming the top-level
run directory that owns ``restart/density``.  Without it every behavior is
exactly as before — no registry is created, attempted or even looked at.

This is the *save-only* integration stage of the run-owned density chain:
a successful attempt that actually produced a new charge density publishes
the candidate seed resource pack (the producing build's
``charge-density.dat``/``charge-density.hdf5`` plus the
``data-file-schema.xml`` next to it, and — when the attempt wrote one —
the PAW ``paw.txt`` becsum a ``startpot='file'`` restart reads) as one
immutable generation through
:mod:`pyraimd2.runtime.restart`, and the attempt record notes the outcome
under ``density_publish``.  The pack is a *candidate* seed: name-level
recognition proves the files exist; sufficiency for a warm start is
build- and format-specific and is claimed only where actually verified
(QE 7.5, HDF5, PAW, non-spin-polarized SCF restart — the A4 evidence).
The bridge also carries the DELAYED producer release: an attempt's
scratch is released only after a later ordinary calculation has
independently read that exact published seed and succeeded
(:func:`release_consumed_scratch`, shared verbatim by both adapters).
Nothing here rewires the next SCF's density source, touches checkpoints,
or changes the default (retention/results) behavior.

Identity: the publication is bound to the real attempt identity
(``request_id``/``attempt_id`` of the verified scratch record in managed
mode).  When the caller gave no ``request_id``, the engines compose a
persistent one from the actually allocated, uniquely numbered compute
directory (relative to the owner), so a rebuilt engine, a second
``compute`` or a deeper adapter layout never collides with a density
already published.  An explicit caller ``request_id`` keeps its existing
meaning.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from pyraimd2.runtime import restart as restart_mod
from pyraimd2.runtime import scratch as scratch_mod

# The schema file QE writes next to the charge density (PW/src: the XML
# data-file-schema document of the .save tree).  Required in the candidate
# seed pack; its absence means "not published", never a guessed seed.
SEED_METADATA_NAME = "data-file-schema.xml"

# QE 7.5 PAW builds write the augmentation occupations (becsum) next to the
# density; a startpot='file' restart needs it (read_scf).  Included in the
# seed pack when present — non-PAW runs never write it.
PAW_BECSUM_NAME = "paw.txt"


def resolve_registry_owner(run_root: Path, density_registry_run_dir, *,
                           scratch_root, retention: str,
                           engine_name: str) -> Path | None:
    """Validate the opt-in explicit run-owner context at construction.

    ``None`` keeps the bridge off (default behavior untouched).  Otherwise
    the owner must be the top-level run directory whose subtree contains
    this engine's ``run_root`` — a shared parent directory is NOT enough —
    and the registry location must stay clear of the scratch root.  The
    bridge currently composes only with ``retention="all"``: it is
    save-only and no reclaim is wired yet, so a reclaiming mode is refused
    up front (the existing ``results``/density-chain refusals stay as they
    are).
    """
    if density_registry_run_dir is None:
        return None
    if retention != "all":
        raise ValueError(
            "density_registry_run_dir currently composes only with "
            f"retention='all' (got retention={retention!r}): publication is "
            f"save-only and no scratch reclaim is wired yet: {engine_name}")
    owner = Path(density_registry_run_dir).resolve()
    root = Path(run_root).resolve()
    if root != owner and owner not in root.parents:
        raise ValueError(
            f"density_registry_run_dir {density_registry_run_dir!r} resolves "
            f"to {owner}, which does not own this engine's run_root {root}: "
            "pass the top-level run directory whose calculations subtree "
            f"the engine writes into: {engine_name}")
    if scratch_root is not None:
        scratch = Path(scratch_root).resolve()
        registry = owner / "restart" / "density"
        if registry == scratch or registry.is_relative_to(scratch) \
                or scratch.is_relative_to(registry):
            raise ValueError(
                f"the density registry ({registry}) and the scratch root "
                f"({scratch}) overlap: published generations must never "
                f"live inside reclaimable scratch: {engine_name}")
    return owner


def directory_request_id(owner: Path, directory: Path) -> str:
    """The persistent request identity composed from an actually allocated
    compute directory: a fixed short purpose prefix plus the SHA-256 of the
    JSON serialization of its owner-relative component array.  The digest
    alone decides identity, so different depths or names containing the
    display separator never collide, and the same directory always yields
    the same id (a retry of the same request stays idempotent).  No mtime
    or randomness is mixed in.

    The readable path is NOT spliced into the id: an earlier readable form
    pushed the scratch record's file name (the id, the attempt suffix and
    the atomic temp suffix) past the filesystem component limit for long
    but legal directory names, failing the launch before any SCF.  The
    short id leaves ample room for those suffixes; the full path stays in
    the durable source records — the attempt receipt and the published
    generation's provenance — never in a file name."""
    relative = Path(directory).resolve().relative_to(Path(owner).resolve())
    parts = list(relative.parts)
    digest = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
    return f"dir#{digest[:16]}"


def publish_attempt_density(*, owner_dir: str | Path, source_root: Path,
                            density_file: Path | None,
                            scratch_handle=None,
                            request_id: str, attempt_id: str,
                            reference_fingerprint: str, nat: int,
                            species: list[str], disk_io: str | None,
                            source_desc: dict) -> dict:
    """Publish one successful attempt's candidate density seed, save-only.

    ``source_root``/``density_file`` describe the verified product of the
    attempt (the caller already applied the shared disk_io + product-file
    rule).  ``scratch_handle`` is the attempt's managed-scratch handle when
    one exists — its authoritative record is read INSIDE the failure
    boundary below and verified by the bounded, ownership-checking entry —
    and ``None`` for an unmanaged in-run source, which must sit inside the
    owner run directory (enforced by the run-owned entry itself).

    Never raises into the engine's success path: the label is already
    delivered.  Returns a status record — ``{"status": "published",
    "generation": int, ...}``, ``{"status": "not_published", "reason":
    ...}`` (the seed pack is incomplete; the source and the previous valid
    generation stay as they are) or ``{"status": "failed", "reason":
    ...}`` (the record read failed, the registry refused or the copy
    failed; no SCF is rerun, no generation is claimed, and the failure is
    reported in memory only — nothing claims it was persisted).
    """
    owner = Path(owner_dir).resolve()
    source_root = Path(source_root)
    if density_file is None:
        return {"status": "not_published",
                "reason": "this attempt left no charge-density file"}
    density_file = Path(density_file)
    try:
        # the whole persistent post-processing sits inside this boundary:
        # the source artifact queries (an EIO from a stat is a publication
        # failure, never an exception escaping into the delivered success
        # path), the parameter evaluation, the authoritative record read,
        # the refusal checks and the copy
        try:
            rel_density = density_file.relative_to(source_root)
        except ValueError:
            return {"status": "not_published",
                    "reason": f"the density file {density_file} is outside the "
                              f"verified attempt source {source_root}"}
        metadata = density_file.parent / SEED_METADATA_NAME
        if not metadata.is_file() or metadata.is_symlink():
            return {"status": "not_published",
                    "reason": f"the candidate seed pack is incomplete: "
                              f"{SEED_METADATA_NAME} missing next to the "
                              f"density file"}
        try:
            rel_metadata = metadata.relative_to(source_root)
        except ValueError:
            return {"status": "not_published",
                    "reason": f"the seed metadata {metadata} is outside the "
                              f"verified attempt source {source_root}"}
        files = [(rel_density.as_posix(), "charge-density"),
                 (rel_metadata.as_posix(), "metadata")]
        # QE 7.5 PAW builds write the augmentation occupations (becsum) as
        # paw.txt next to the density, and a startpot='file' restart reads
        # it in read_scf — a PAW seed pack without it fails with "Reading
        # PAW becsum" (real-QE A4 finding).  Include the file when the
        # attempt actually wrote one; non-PAW runs never produce it, and a
        # pack member is never claimed without the verified file present.
        paw = density_file.parent / PAW_BECSUM_NAME
        if paw.is_file() and not paw.is_symlink():
            try:
                rel_paw = paw.relative_to(source_root)
            except ValueError:
                return {"status": "not_published",
                        "reason": f"the PAW becsum file {paw} is outside "
                                  f"the verified attempt source {source_root}"}
            files.append((rel_paw.as_posix(), "paw-becsum"))
        provenance = {
            "attempt": {"request_id": request_id, "attempt_id": attempt_id},
            "reference_fingerprint": reference_fingerprint,
            "nat": int(nat),
            "species": sorted(species),
            "disk_io": disk_io,
            "source": source_desc,
        }
        record = (None if scratch_handle is None
                  else scratch_handle.load_record())
        if record is not None:
            outcome = restart_mod.publish_density_generation_from_attempt(
                owner, record, files=files, provenance=provenance)
        else:
            outcome = restart_mod.publish_density_generation(
                owner, source_root, files=files, provenance=provenance)
    except Exception as error:  # noqa: BLE001 — a delivered label never breaks on publication
        return {"status": "failed",
                "reason": f"{type(error).__name__}: {error}"}
    return {"status": "published",
            "generation": int(outcome["generation"]),
            "reused": bool(outcome["reused"]),
            "directory": outcome["directory"]}


def load_published_density(owner_dir: str | Path, *,
                           generation: int | None,
                           reference_fingerprint: str, nat: int,
                           species: list[str]
                           ) -> tuple[tuple[Path, Path, dict] | None, str,
                                      int | None]:
    """Load a verified persistent density generation for reuse.

    The shared read side of the run-owned chain, used identically by both
    QE adapters.  ``generation=None`` resolves the registry's latest
    pointer; an explicit ``generation`` resolves exactly that generation —
    never a later one, and never a guess from mtimes or directory names.

    Verification chain (all through the existing registry machinery):
    the registry state must read ``ok``; the target must classify as a
    valid generation (``restart``'s shared validator already checked the
    directory confinement, the run binding, and every declared file's
    size and SHA-256 — a damaged generation never classifies); and the
    manifest's provenance must match this engine's reference fingerprint
    and the current structure's atom count and species.  Any mismatch
    answers ``(None, reason, None)`` — the caller decides the fallback,
    this bridge never substitutes another generation.

    On success returns ``((generation_dir, save_dir, manifest), "",
    generation)`` where ``save_dir`` is the parent directory of the
    manifest's ``charge-density`` role entry (the registry records real
    relative paths; nothing here hardcodes a ``.save`` layout).  The
    ``DensitySource`` wrap stays on the engine side — this module must
    not import the engines (circular).
    """
    view = restart_mod.inspect_density_registry(owner_dir)
    if view["state"] != restart_mod.STATE_OK:
        if view["state"] == restart_mod.STATE_FRESH:
            return None, "no published density generation", None
        return None, (f"the density registry state is not readable "
                      f"({view['state_reason']})"), None
    target = generation if generation is not None else view["latest"]
    if target is None:
        return None, "no published density generation", None
    resource = next(
        (r for r in view["resources"]
         if r["kind"] == restart_mod.CLASS_GENERATION
         and r["generation"] == int(target)), None)
    if resource is None:
        return None, (f"density generation g{int(target):06d} is missing or "
                      "fails validation — it is never substituted by "
                      "another generation"), None
    manifest = resource["manifest"]
    if manifest.get("reference_fingerprint") != reference_fingerprint:
        return None, "reference settings differ from the generation's", None
    if manifest.get("nat") != int(nat):
        return None, (f"atom count differs ({manifest.get('nat')} != "
                      f"{int(nat)})"), None
    if manifest.get("species") != sorted(species):
        return None, "species differ", None
    density_entries = [entry for entry in manifest.get("files", [])
                       if entry.get("role") == "charge-density"]
    if not density_entries:
        return None, "the generation declares no charge-density file", None
    generation_dir = Path(view["registry"]) / resource["name"]
    save_dir = (generation_dir / density_entries[0]["path"]).parent
    return (generation_dir, save_dir, manifest), "", int(target)


# ---------------------------------------------------------------------------
# delayed producer release (A2): a published seed is a deletion credential
# only after an independent successful calculation consumed that exact seed


# release-receipt proof states carried by the producer attempt's
# authoritative record under ``released_evidence``
PROOF_PENDING = "pending_independent_consumption"
PROOF_CONSUMED = "consumed"


def pending_release_receipt(*, evaluation_id: int,
                            input_generation: int | None,
                            output_generation: int, manifest: dict,
                            reference_fingerprint: str) -> dict:
    """The release receipt recorded on a producer attempt's authoritative
    record at its own evaluation's commit: the attempt's result is
    committed and its seed is published and verified, but the scratch
    stays kept until a later ordinary calculation independently reads
    that exact seed and succeeds.  Binds the seed generation, its content
    digest and the verified-compatible reference settings."""
    return {
        "evaluation_id": int(evaluation_id),
        "density_input_generation": input_generation,
        "density_output_generation": int(output_generation),
        "seed_content_digest": manifest["content_digest"],
        "reference_fingerprint": reference_fingerprint,
        "proof": PROOF_PENDING,
        "recorded_unix": time.time(),
    }


def find_attempt_record(owner_dir: str | Path,
                        attempt: dict) -> tuple[Path, dict] | None:
    """Locate and validate the authoritative scratch record of the attempt
    a published generation names as its producer (provenance identity).

    Returns ``(record_path, validated_record)`` or ``None`` when no
    usable record exists — the consumption state of the seed is then
    undeterminable and the caller must preserve, never delete."""
    records_dir = Path(owner_dir) / scratch_mod.SCRATCH_RECORD_DIR
    if not records_dir.is_dir():
        return None
    for path in sorted(records_dir.rglob("*.json")):
        record = restart_mod._read_json_quiet(path)
        if record is None or record.get("record") != scratch_mod.SCRATCH_ATTEMPT_RECORD:
            continue
        if (record.get("request_id"), record.get("attempt_id")) != \
                (attempt["request_id"], attempt["attempt_id"]):
            continue
        try:
            scratch_mod._validate_record(record, path=path)
        except scratch_mod.ScratchError:
            return None
        return path, record
    return None


def _handle_for_record(record: dict, record_path: Path):
    """Rebuild the managed-scratch handle of an attempt from its validated
    authoritative record (the record pins every identity component)."""
    return scratch_mod.AttemptScratch(
        run_uuid=record["run_uuid"], backend_role=record["backend_role"],
        request_id=record["request_id"], attempt_id=record["attempt_id"],
        scratch_dir=Path(record["scratch_dir"]),
        archive_dir=Path(record["archive_dir"]), record_path=record_path)


def _complete_consumption_release(engine, consumption: dict, *,
                                  evaluation_id: int,
                                  on_reclaimed) -> dict:
    """Release the PRODUCER scratch of the seed this evaluation's compute
    independently consumed — the actual deletion gate of the delayed
    release.

    The proof object names the consumed generation, the consuming attempt
    identity, the pinned seed content digest/settings and the staged-copy
    facts (the actual-read evidence).  The producer is found through the
    generation manifest's bound attempt identity and its authoritative
    record; release happens only when that record is still ``kept``,
    carries the pending release receipt of exactly this generation, and
    the seed identity agrees across all three records — the consuming
    attempt's pin, the persisted receipt and the live manifest.  Anything
    else — a missing/unreadable record, an unexpected state, a record
    written before pending receipts existed, a digest or settings
    contradiction, or incomplete evidence — keeps the scratch with the
    reason; a damaged receipt is never repaired to the current value to
    approve a deletion.  The completed receipt merges the pending one
    (producing evaluation, seed generation, content digest, compatible
    settings) with the consumption proof (consuming evaluation and
    attempt, staged source and bytes, read evidence), then hands the
    attempt to the existing cleanup machinery; a refusal/failure there is
    reported, never retried or forced.
    """
    registry_dir = engine._density_registry_dir
    generation = int(consumption["generation"])
    view = restart_mod.inspect_density_registry(registry_dir)
    resource = next((r for r in view["resources"]
                     if r["kind"] == restart_mod.CLASS_GENERATION
                     and r["generation"] == generation), None)
    if resource is None:
        return {"status": "kept", "generation": generation,
                "reason": f"the consumed generation g{generation:06d} no "
                          "longer verifies in the registry; the producer "
                          "scratch is preserved for diagnosis"}
    attempt = resource["manifest"].get("attempt")
    if not isinstance(attempt, dict) \
            or not isinstance(attempt.get("request_id"), str) \
            or not isinstance(attempt.get("attempt_id"), str):
        return {"status": "kept", "generation": generation,
                "reason": "the consumed generation's manifest has no usable "
                          "producer attempt identity; the producer scratch "
                          "is preserved"}
    current = engine._last_scratch_handle
    if current is not None \
            and (attempt["request_id"], attempt["attempt_id"]) == \
            (current.request_id, current.attempt_id):
        # an attempt never consumes the seed it produced itself
        return {"status": "kept", "generation": generation,
                "reason": "the consuming attempt is the producer itself; "
                          "nothing to release"}
    found = find_attempt_record(registry_dir, attempt)
    if found is None:
        return {"status": "kept", "generation": generation,
                "reason": f"the producer attempt record for g{generation:06d} "
                          "is missing or invalid; the consumption state "
                          "cannot be determined and the scratch is preserved"}
    record_path, producer = found
    state = producer["state"]
    if state == "cleaned":
        return {"status": "already_cleaned", "generation": generation}
    if state != "kept":
        return {"status": "kept", "generation": generation,
                "reason": f"the producer attempt is {state!r}, not kept; "
                          "the source is preserved"}
    pending = producer.get("released_evidence")
    if not (isinstance(pending, dict)
            and pending.get("proof") == PROOF_PENDING
            and pending.get("density_output_generation") == generation
            and isinstance(pending.get("evaluation_id"), int)):
        return {"status": "kept", "generation": generation,
                "reason": f"the producer record carries no pending release "
                          f"receipt for g{generation:06d} (written before "
                          "pending receipts existed, or never recorded); "
                          "publication alone is not a deletion credential — "
                          "the scratch is preserved"}
    # The seed identity must agree across all three records: the consuming
    # attempt's pinned digest (selection/staging time), the producer's
    # persisted pending receipt, and the generation's live manifest.  Any
    # explicit contradiction — or a missing digest on either side — keeps
    # the producer with the reason; a damaged receipt is never "repaired"
    # to the current value to approve the deletion.  The reference
    # settings identity is cross-checked the same way.
    live_manifest = resource["manifest"]
    live_digest = live_manifest.get("content_digest")
    identity_pairs = (
        ("the consuming attempt pinned", consumption.get("seed_content_digest")),
        ("the producer's pending receipt recorded",
         pending.get("seed_content_digest")),
    )
    for who, digest in identity_pairs:
        if not isinstance(digest, str) or not digest:
            return {"status": "kept", "generation": generation,
                    "reason": f"{who} no usable seed content digest — the "
                              "consumption evidence is incomplete and the "
                              "producer scratch is preserved"}
        if not isinstance(live_digest, str) or live_digest != digest:
            return {"status": "kept", "generation": generation,
                    "reason": f"{who} seed content digest {digest[:16]}… but "
                              f"the live manifest of g{generation:06d} "
                              f"records {str(live_digest)[:16]}… — the "
                              "records contradict each other and the "
                              "producer scratch is preserved"}
    live_fingerprint = live_manifest.get("reference_fingerprint")
    for who, fingerprint in (
            ("the consuming attempt pinned",
             consumption.get("reference_fingerprint")),
            ("the producer's pending receipt recorded",
             pending.get("reference_fingerprint"))):
        if not isinstance(fingerprint, str) or not fingerprint \
                or fingerprint != live_fingerprint:
            return {"status": "kept", "generation": generation,
                    "reason": f"{who} reference settings "
                              f"{str(fingerprint)[:40]!r}, inconsistent with "
                              f"the live manifest's "
                              f"{str(live_fingerprint)[:40]!r} — the "
                              "producer scratch is preserved"}
    evidence = {**pending, "proof": PROOF_CONSUMED,
                "consumed_unix": time.time(),
                "consumed_by": {
                    "evaluation_id": int(evaluation_id),
                    "request_id": consumption["consumer_attempt"]["request_id"],
                    "attempt_id": consumption["consumer_attempt"]["attempt_id"],
                    "staged_from": consumption.get("staged_from"),
                    "staged_bytes": consumption.get("staged_bytes"),
                    "staged_copy_s": consumption.get("staged_copy_s"),
                    # the actual-read observation, parsed from the
                    # consuming attempt's own raw output and bound to its
                    # launch input and output digests
                    "read_evidence": consumption.get("read_evidence")}}
    handle = _handle_for_record(producer, record_path)
    try:
        release_record = scratch_mod.release_consumed(handle,
                                                      evidence=evidence)
    except Exception as error:  # noqa: BLE001 — a receipt, never an exception
        return {"status": "kept", "generation": generation,
                "reason": f"release refused: {error}"}
    cleanup_receipt = scratch_mod.cleanup(handle)
    receipt = {"status": cleanup_receipt.get("status"),
               "generation": generation,
               "release": {"state": release_record["state"],
                           "evidence": dict(evidence)},
               "cleanup": cleanup_receipt}
    if cleanup_receipt.get("status") in ("cleaned", "already_cleaned") \
            and on_reclaimed is not None:
        on_reclaimed(handle.archive_dir)
    return receipt


def release_consumed_scratch(engine, *, evaluation_id: int,
                             on_reclaimed=None) -> dict:
    """Release managed scratch under the delayed-release contract — a
    receipt, never an exception into the run loop.  Shared by both QE
    adapters so the two entries behave identically.

    Phase 1 — the consumption proof carried by THIS evaluation's compute
    (``engine._pending_consumption``, set only by a successful attempt
    that actually staged a registry generation as its density start AND
    whose raw output proves the solver really read it: never an atomic
    fallback, never a cache hit, never a borrow from the producer's own
    tree, never a silent/no-marker success) releases the PRODUCER of that
    seed.  A staged seed without actual-read evidence
    (``engine._consumption_unproven``) keeps the producer with the parsed
    reason — staging plus success is not consumption, and an unknown or
    silent output format preserves the source rather than assuming a
    build's behavior from a version string.  Phase 2 — this attempt's own
    scratch: a verified publication only earns the pending release receipt
    (persisted on the authoritative record, idempotent); the scratch stays
    ``kept`` until a later independent consumption completes it.  A
    failed/incomplete/unverifiable publication keeps the scratch (it holds
    the only verified copy).  An attempt with no density product releases
    directly: nothing persistent depends on its save tree.

    ``not_applicable`` is reported honestly: registry disabled, no managed
    scratch, the last compute unsuccessful, or the attempt not in the kept
    state.  ``on_reclaimed`` (engine hook) is called with the archive dir
    of each attempt whose scratch was actually reclaimed.
    """
    registry_dir = engine._density_registry_dir
    if registry_dir is None:
        return {"status": "not_applicable",
                "reason": "the density registry is not enabled"}
    handle = engine._last_scratch_handle
    if handle is None:
        return {"status": "not_applicable",
                "reason": "the last compute used no managed scratch"}
    if not engine.last_attempt_records \
            or engine.last_attempt_records[-1].get("status") != "success":
        return {"status": "not_applicable",
                "reason": "the last compute did not succeed"}
    record = engine.last_attempt_records[-1]
    try:
        state = handle.load_record()["state"]
    except Exception as error:  # noqa: BLE001 — a receipt, never an exception
        return {"status": "failed",
                "reason": f"the attempt record is unreadable: {error}"}

    consumption = getattr(engine, "_pending_consumption", None)
    consumption_receipt = None
    if consumption is not None:
        engine._pending_consumption = None
        consumption_receipt = _complete_consumption_release(
            engine, consumption, evaluation_id=evaluation_id,
            on_reclaimed=on_reclaimed)
    unproven = getattr(engine, "_consumption_unproven", None)
    if unproven is not None:
        engine._consumption_unproven = None
        generation = int(unproven["generation"])
        consumption_receipt = {
            "status": "kept",
            "generation": generation,
            "reason": (f"the successful attempt staged g{generation:06d} "
                       f"but its raw output carries no actual density-read "
                       f"evidence ({unproven['reason']}); staging plus "
                       "success is not consumption — the producer scratch "
                       "is preserved")}

    if state != "kept":
        return {"status": "not_applicable",
                "reason": f"the attempt scratch is {state!r}, not kept "
                          "(already released or never kept)",
                "consumption": consumption_receipt}
    publish = record.get("density_publish")
    if publish is not None and publish.get("status") == "published":
        generation = int(publish["generation"])
        # verify the publication really landed before relying on it: the
        # generation must still classify as a valid generation of THIS
        # run's registry (content digests included)
        view = restart_mod.inspect_density_registry(registry_dir)
        resource = next((r for r in view["resources"]
                         if r["kind"] == restart_mod.CLASS_GENERATION
                         and r["generation"] == generation), None)
        if not (view["state"] == "ok" and resource is not None):
            receipt = {"status": "kept",
                       "reason": f"generation g{generation:06d} does not "
                                 "verify in the registry; the scratch "
                                 "source is preserved",
                       "consumption": consumption_receipt}
            record["scratch_cleanup"] = receipt
            return receipt
        try:
            durable = handle.load_record()
            existing = durable.get("released_evidence")
            if not (isinstance(existing, dict)
                    and existing.get("proof") == PROOF_PENDING
                    and existing.get("density_output_generation")
                    == generation):
                existing = pending_release_receipt(
                    evaluation_id=evaluation_id,
                    input_generation=record.get("density_input_generation"),
                    output_generation=generation,
                    manifest=resource["manifest"],
                    reference_fingerprint=engine.fingerprint)
                handle.update_record(released_evidence=existing)
        except Exception as error:  # noqa: BLE001 — a receipt, never an exception
            receipt = {"status": "kept",
                       "reason": f"the pending release receipt could not be "
                                 f"persisted ({error}); the scratch stays "
                                 "kept",
                       "consumption": consumption_receipt}
            record["scratch_cleanup"] = receipt
            return receipt
        receipt = {"status": "kept",
                   "reason": f"generation g{generation:06d} is published and "
                             "verified; the scratch stays kept until an "
                             "independent successful calculation consumes "
                             "that exact seed",
                   "pending_release": existing,
                   "consumption": consumption_receipt}
        record["scratch_cleanup"] = receipt
        return receipt
    if publish is not None:
        receipt = {"status": "kept",
                   "reason": f"the density publication did not complete "
                             f"({publish.get('status')}: "
                             f"{publish.get('reason')}); the scratch "
                             "holds the only verified copy",
                   "consumption": consumption_receipt}
        record["scratch_cleanup"] = receipt
        return receipt
    # else: the attempt produced no density — nothing persistent depends
    # on its save tree, release directly
    evidence = {"evaluation_id": int(evaluation_id),
                "density_input_generation":
                    record.get("density_input_generation"),
                "density_output_generation": None,
                "released_unix": time.time()}
    try:
        release_record = scratch_mod.release_consumed(handle,
                                                      evidence=evidence)
    except Exception as error:  # noqa: BLE001 — a receipt, never an exception
        receipt = {"status": "kept",
                   "reason": f"release refused: {error}",
                   "consumption": consumption_receipt}
        record["scratch_cleanup"] = receipt
        return receipt
    cleanup_receipt = scratch_mod.cleanup(handle)
    receipt = {"status": cleanup_receipt.get("status"),
               "release": {"state": release_record["state"],
                           "evidence": dict(evidence)},
               "cleanup": cleanup_receipt,
               "consumption": consumption_receipt}
    record["scratch_cleanup"] = receipt
    if cleanup_receipt.get("status") in ("cleaned", "already_cleaned") \
            and on_reclaimed is not None:
        on_reclaimed(handle.archive_dir)
    return receipt
