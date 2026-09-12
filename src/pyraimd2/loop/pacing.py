"""Cost-aware calibration pacing (0.6 prototype — opt-in, default off).

When enabled, the adaptive controller defers the probe investment of a
post-refusal recalibration while calibrations keep failing to produce an
accept: after ``failure_streak_limit`` consecutive calibrations each
proven sterile by a refused next forecast, the next calibration
opportunities are skipped and the steps drive reference-direct — exactly
the existing refusal path; the refused step's reference force propagates
either way.  A retained stale anchor keeps its prefix CLOSED (a failed
step stops the accepted prefix, unchanged semantics) and serves only the
diagnostic residual signal.  The wait is bounded: after ``wait`` skipped
opportunities the next refusal calibrates once (the bounded retry), and
failure rounds grow the wait ``wait_initial, 2*wait_initial, ...`` capped
at ``wait_max`` — an upper-bounded re-probe interval, never a claim that
every short beneficial window is caught.  Any accept resets the streak and
the backoff.  During a wait, each refused step's ALREADY-RECORDED
corrected-surrogate error (the retained anchor's correction against the
step's own reference label, computed anyway) exceeding the run's force
budget cancels the wait immediately and recalibrates — the reference
drive during the wait was already valid; this only re-enables model
recalibration sooner.

The rule is a pure state machine: all transitions live here as small
functions over a plain state, so the MD path only records decisions and
applies their outcomes.  Pacing consumes no random stream — verification
draws and bath streams are untouched.  Nothing here relaxes any gate,
budget, cap, or check probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Durable-record version for pacing fields on decision events, commit
# metadata and checkpoints.  Old records lack the fields entirely, which
# means the feature was off — they are never reinterpreted as on.
PACING_RECORD = "calibration_pacing_v1"

DECISIONS = ("calibrate", "defer", "unavailable")


@dataclass(frozen=True)
class CalibrationPacing:
    """Validated ``[policy.calibration_pacing]`` settings (enabled)."""

    failure_streak_limit: int
    wait_initial: int
    wait_max: int


@dataclass
class PacingState:
    """The durable pacing state.

    ``sterile``: consecutive completed calibrations that never produced an
    accept.  ``pending``: a calibration has completed but its first
    forecast has not been seen (only a completed calibration with a usable
    direction opens a pending segment).  ``wait``: remaining calibration
    opportunities to skip.  ``wait_len``: current backoff length.
    ``test_due``: the wait is complete — the next refusal must recalibrate
    (the bounded retry).  The three counters are the explainability record
    (deferred opportunities, forced retries, safety exits).
    """

    sterile: int = 0
    pending: bool = False
    wait: int = 0
    wait_len: int = 1
    test_due: bool = False
    n_deferred: int = 0
    n_retried: int = 0
    n_safety_exits: int = 0

    def as_dict(self) -> dict:
        return {"record": PACING_RECORD, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict) -> PacingState:
        if payload.get("record") != PACING_RECORD:
            raise ValueError(
                f"pacing state claims record {payload.get('record')!r}, "
                f"expected {PACING_RECORD!r}; the run directory is "
                "inconsistent")
        return cls(**{key: payload[key] for key in
                      ("sterile", "pending", "wait", "wait_len", "test_due",
                       "n_deferred", "n_retried", "n_safety_exits")})


def initial_state(settings: CalibrationPacing) -> PacingState:
    return PacingState(wait_len=settings.wait_initial)


def on_accept(state: PacingState,
              settings: CalibrationPacing) -> PacingState:
    """A committed accept: the serving calibration proved fertile — reset
    the streak and the backoff.  The valid anchor is retained by the
    caller for consecutive accepts; an accept is never an anchor
    invalidation.  (Waits are always complete by the time a forecast can
    accept, so ``wait`` is already zero.)"""
    return PacingState(sterile=0, pending=False, wait=0,
                       wait_len=settings.wait_initial, test_due=False,
                       n_deferred=state.n_deferred,
                       n_retried=state.n_retried,
                       n_safety_exits=state.n_safety_exits)


def decide_on_refusal(state: PacingState, settings: CalibrationPacing, *,
                      observed_error: float | None,
                      force_budget: float) -> tuple[str, str, PacingState]:
    """Pure decision at a refused evaluation whose serving anchor exists.

    Returns ``(decision, reason, state_after)``.  The sterility
    bookkeeping comes first: if the serving segment's first forecast just
    refused, that calibration is proven sterile exactly once (a segment
    that already accepted is fertile forever; waiting steps are not new
    failures and never increment the streak).  The safety exit — this
    step's own reference label against the retained anchor's correction
    over the existing force budget — cancels an active wait and
    recalibrates now.  Then in order: an active wait skips (decrementing
    to zero arms the forced retry); a completed wait forces one test
    calibration; a streak at the limit opens a new wait whose FIRST skip
    is this refusal (``wait_initial = 1`` skips only this opportunity);
    otherwise the ordinary recalibration.  ``state_after`` applies only
    when the evaluation commits — never on a failed attempt.
    """
    sterile = state.sterile + (1 if state.pending else 0)

    def after(*, wait=0, wait_len=None, test_due=False,
              n_deferred=None, n_retried=None, n_safety_exits=None):
        return PacingState(
            sterile=sterile, pending=False, wait=wait,
            wait_len=state.wait_len if wait_len is None else wait_len,
            test_due=test_due,
            n_deferred=state.n_deferred if n_deferred is None else n_deferred,
            n_retried=state.n_retried if n_retried is None else n_retried,
            n_safety_exits=(state.n_safety_exits if n_safety_exits is None
                            else n_safety_exits))

    # The safety exit: this step's own reference label against the retained
    # anchor's correction exceeds the existing budget — cancel the wait and
    # recalibrate now (the reference drive during the wait was valid; this
    # only re-enables model recalibration sooner).  This calibration IS the
    # retry: if it proves sterile too, the streak opens a fresh wait at the
    # next refusal with the grown backoff length.
    if state.wait and observed_error is not None \
            and observed_error > force_budget:
        return ("calibrate", "safety_exit",
                after(wait=0, test_due=False,
                      n_safety_exits=state.n_safety_exits + 1))
    # An active wait skips this opportunity; decrementing to zero arms the
    # forced retry at the next calibratable refusal.
    if state.wait:
        wait = state.wait - 1
        return ("defer", "waiting",
                after(wait=wait, test_due=wait == 0,
                      n_deferred=state.n_deferred + 1))
    # A completed wait forces exactly one test calibration.
    if state.test_due:
        return ("calibrate", "forced_retry",
                after(test_due=False, n_retried=state.n_retried + 1))
    # The streak limit is reached at THIS refusal: the first skip is this
    # opportunity (wait_initial = 1 skips only this one), then the backoff
    # grows wait_initial, 2*wait_initial, ... capped at wait_max.
    if sterile >= settings.failure_streak_limit:
        wait = state.wait_len - 1
        wait_len = min(2 * state.wait_len, settings.wait_max)
        return ("defer", "sterile_streak",
                after(wait=wait, wait_len=wait_len, test_due=wait == 0,
                      n_deferred=state.n_deferred + 1))
    return "calibrate", "sterile_below_limit", after()


def on_calibration(state: PacingState) -> PacingState:
    """A completed calibration with a usable direction opens its pending
    segment.  A calibration that found no valid direction (unavailable) is
    not a completed calibration and never calls this."""
    return PacingState(sterile=state.sterile, pending=True,
                       wait=state.wait, wait_len=state.wait_len,
                       test_due=state.test_due,
                       n_deferred=state.n_deferred,
                       n_retried=state.n_retried,
                       n_safety_exits=state.n_safety_exits)
