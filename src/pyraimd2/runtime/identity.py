"""Declared identities of models and reference settings.

A *fingerprint* names the settings/lineage of a backend (model file and
options, or functional/basis/pseudopotential recipe); a *generation* counts
model updates inside a run.  Together they form the ``model_id`` carried by
each :class:`~pyraimd2.runtime.context.EvaluationContext`.

The energetic decision layer keys its caches on this identity: a model change
invalidates pending proposals, anchors and cached results for the same
geometry (WP01).  This is the *decision* cache only — the numeric label cache
keyed by geometry plus backend settings is WP02 and is deliberately separate.
"""

from __future__ import annotations


def fingerprint_of(obj: object) -> str | None:
    """Return the object's declared fingerprint, or None when it declares none.

    Backends may expose ``fingerprint`` as a string attribute or a zero-arg
    method.  The fingerprint is never guessed: an object without a declared
    one has no identity opinion, and callers must not invent one.
    """
    value = getattr(obj, "fingerprint", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    return str(value)


def model_id_for(model: object, generation: int) -> str:
    """``<fingerprint-or-class>#g<generation>``: the decision-cache identity."""
    base = fingerprint_of(model) or type(model).__qualname__
    return f"{base}#g{int(generation)}"
