"""Frozen high-velocity forward comparison in the existing private cloud runtime.

Development uses two previously labelled paths only. New paths are generated only
after selection is written to disk. One worker per arm, without cross-arm labels.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import forward_h2o_20260905 as base
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'analysis/execution_20260905/h2o'
OUT = ROOT / 'analysis/execution_20260905/h2o_hot_cloud'
SEEDS = [2026090521, 2026090522]
STARTS = [0, 151]
STATES = 40
MODEL_SHA = '2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736'
CHECKPOINT_SHA = '4ccae88097a2a76f7037893a480354df678034ff7628c9bf5fa0798172bdb8bb'
MODEL_ORIGIN = '/Users/pengkang/.cache/mace/20231210mace128L0_energy_epoch249model'


class TimedModel:
    def __init__(self, model):
        self.model = model
        self.seconds = 0.
        self.calls = 0

    def predict(self, atoms):
        t = time.perf_counter()
        result = self.model.predict(atoms)
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return result


def configure():
    import torch
    from pyscf import lib
    torch.set_num_threads(1)
    lib.num_threads(1)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s reference timeout')))
    base.MODEL = Path(os.environ['H2O_MODEL_PATH']).resolve()
    base.OUT = OUT


def restored():
    import torch
    state = torch.load(OLD / 'common_checkpoint.pt', map_location='cpu', weights_only=False)
    if state['model_specs'] != [MODEL_ORIGIN] * 4 or state['n_members'] != 4:
        raise RuntimeError('Unexpected checkpoint identity; refuse relocation')
    # Only relocate the verified backing-file path in memory; retain all tensors
    # and original checkpoint bytes. Production load_state_dict stays strict.
    state = dict(state, model_specs=[str(base.MODEL)] * 4)
    m = base.model()
    m.load_state_dict(state)
    return TimedModel(m)


def replay(paths, calibration, kind, setting):
    rows = []
    for path in paths:
        pairs = calibration.copy()
        rng = np.random.default_rng([path['seed'], 919])
        n = v = requests = 0
        for step, (s, e) in enumerate(path['pairs']):
            accept = step > 0 and (step % setting != 0 if kind == 'periodic'
                                   else base.qhat(pairs) * (s + .001) * setting <= base.EPS)
            audit = bool(rng.random() < .1) if accept else False
            revealed = not accept or audit
            n += int(accept)
            v += int(accept and e > base.EPS)
            requests += int(revealed)
            if revealed:
                pairs.append((s, e))
        rows.append({'seed': path['seed'], 'accepted': n, 'violations': v,
                     'requests': requests, 'risk': v / n if n else None,
                     'audit_adjusted_requests': len(path['pairs']) - .9 * n})
    return {'kind': kind, 'setting': setting, 'paths': rows,
            'eligible': all(r['accepted'] == 0 or r['risk'] <= .05 for r in rows),
            'usable': all(r['accepted'] >= 4 for r in rows),
            'worst_path_risk': max((r['risk'] for r in rows if r['risk'] is not None), default=None),
            'audit_adjusted_requests': sum(r['audit_adjusted_requests'] for r in rows),
            'requests': sum(r['requests'] for r in rows),
            'violations': sum(r['violations'] for r in rows)}


def prepare():
    configure()
    OUT.mkdir(parents=True, exist_ok=False)
    original = json.loads((OLD / 'locked_protocol.json').read_text())
    assert base.sha(base.MODEL) == MODEL_SHA
    assert base.sha(OLD / 'common_checkpoint.pt') == CHECKPOINT_SHA
    spec = {'purpose': 'Forward error/reference-cost comparison at doubled initial speed',
            'seeds': SEEDS, 'start_indices': STARTS, 'velocity_factor': 2.,
            'initial_kinetic_temperature_K': 1200, 'pilot_evaluation_states': STATES,
            'epsilon_ev_A': base.EPS, 'dt_fs': base.DT_FS, 'audit_p': base.AUDIT_P,
            'update_model': False, 'growth_gamma': 2., 'growth_status': 'empirical, not certified',
            'period_candidates': [1, 2, 4, 8, 16],
            'q_multiplier_candidates': [1., 1.5, 2., 3., 4., 8., 16., 1e6],
            'selection': 'Usable candidates accept >=4/20 states on each old hot path. Among those with <=5% observed accepted violations on each path, minimize audit-adjusted requests, then violations, then more conservative setting. If none pass, select usable candidate minimizing worst-path observed risk then adjusted requests and report failed development risk. If no calibrated candidate is usable, stop before new forward paths. K=1 is covered by the reference arm; do not duplicate it.',
            'calibration_pairs': original['calibration_pairs'],
            'development_paths': [f'velocity_{s}_2_steps.json' for s in [2026090501, 2026090502]],
            'new_reference_attempt_limit': 321, 'per_arm_attempt_limit': 40,
            'wall_limit_seconds': 14400, 'per_worker_threads': 1, 'workers': 4,
            'model_sha256': MODEL_SHA, 'checkpoint_sha256': CHECKPOINT_SHA,
            'relocation': {'original_model_path': MODEL_ORIGIN, 'runtime_model_path': str(base.MODEL),
                           'scope': 'in-memory model_specs path only'},
            'compatibility_tolerances': {'model_force_component_ev_A': 1e-6,
                                          'reference_force_component_ev_A': 1e-6,
                                          'reference_energy_ev': 1e-6},
            'source_sha256': {p.name: base.sha(p) for p in [Path(__file__), Path(base.__file__)]},
            'development_path_sha256': {f'velocity_{s}_2_steps.json': base.sha(OLD / f'velocity_{s}_2_steps.json') for s in [2026090501, 2026090502]},
            'created_unix': time.time(),
            'limits': 'New velocity seeds at known geometries, two short trajectories per arm; not a new-material test, equilibrium validation, or a simultaneous risk guarantee.'}
    base.write(OUT / 'development_spec.json', spec)
    m = restored()
    old_frames = base.archived()
    paths = []
    delta = 0.
    for seed, start in zip([2026090501, 2026090502], STARTS):
        data = json.loads((OLD / f'velocity_{seed}_2_steps.json').read_text())
        pairs = []
        for row in data:
            a = old_frames[start][0].copy()
            a.positions = np.array(row['positions'])
            p = m.predict(a)
            delta = max(delta, float(np.max(np.abs(p.forces - np.array(row['surrogate_forces'])))))
            pairs.append((float(p.uncertainty.max()), base.force_norm(p.forces - np.array(row['reference_forces']))))
        paths.append({'seed': seed, 'pairs': pairs})
    if delta > spec['compatibility_tolerances']['model_force_component_ev_A']:
        base.write(OUT / 'compatibility_failure.json', {'model_max_difference': delta})
        raise RuntimeError('Cloud model differs from recorded frozen predictions')
    ledger = base.ReferenceLedger('compatibility', 1, 600)
    a, old_truth = old_frames[0]
    ledger.cache.pop(base.key(a), None)
    truth, _ = ledger.evaluate(a, 'compatibility:anchor0')
    df = float(np.max(np.abs(truth.forces - old_truth.forces)))
    de = abs(truth.energy - old_truth.energy)
    comp = {'model_max_component_difference_ev_A': delta, 'reference_max_component_difference_ev_A': df,
            'reference_energy_difference_ev': de, 'new_reference_attempts': ledger.n,
            'reference_seconds': ledger.reference_seconds, 'prediction_seconds': m.seconds,
            'prediction_calls': m.calls,
            'packages': {p: importlib.metadata.version(p) for p in ['torch', 'pyscf', 'mace-torch', 'e3nn', 'numpy', 'scipy', 'ase']}}
    comp['passed'] = df <= 1e-6 and de <= 1e-6
    base.write(OUT / 'compatibility.json', comp)
    if not comp['passed']:
        raise RuntimeError('Reference compatibility check failed')
    calibration = [tuple(x) for x in spec['calibration_pairs']]
    candidates = [replay(paths, calibration, kind, s) for kind, settings in
                  [('periodic', spec['period_candidates']), ('calibrated', spec['q_multiplier_candidates'])]
                  for s in settings]
    selected, selection_status = {}, {}
    for kind in ['periodic', 'calibrated']:
        usable = [r for r in candidates if r['kind'] == kind and r['usable']]
        if not usable:
            base.write(OUT / 'development_replay.json', {'paths': paths, 'candidates': candidates})
            base.write(OUT / 'development_stop.json', {'kind': kind, 'reason': 'No usable candidate; do not spend forward labels on degenerate comparison'})
            raise RuntimeError(f'No usable {kind} candidate; scientific stop')
        eligible = [r for r in usable if r['eligible']]
        conservative = lambda r, kind=kind: r['setting'] if kind == 'periodic' else -r['setting']
        if eligible:
            choice = min(eligible, key=lambda r: (r['audit_adjusted_requests'], r['violations'], conservative(r)))
            selection_status[kind] = 'passed_observed_development_risk'
        else:
            choice = min(usable, key=lambda r: (r['worst_path_risk'], r['audit_adjusted_requests'], conservative(r)))
            selection_status[kind] = 'nontrivial_fallback_failed_development_risk'
        selected[kind] = choice['setting']
    base.write(OUT / 'development_replay.json', {'paths': paths, 'candidates': candidates})
    spec.update({'selected': selected, 'selection_status': selection_status, 'locked_unix': time.time(), 'compatibility_passed': True,
                 'development_spec_sha256': base.sha(OUT / 'development_spec.json')})
    base.write(OUT / 'locked_protocol.json', spec)
    print(json.dumps({'compatibility': comp, 'selected': selected}), flush=True)


def worker(arm, seed, start):
    configure()
    protocol = json.loads((OUT / 'locked_protocol.json').read_text())
    instances = []

    def tracked_restore():
        m = restored()
        instances.append(m)
        return m

    base.restored = tracked_restore
    ledger = base.ReferenceLedger(f'pilot_{arm}_{seed}', STATES, 14000)
    try:
        result = base.trajectory(arm, seed, start, ledger, protocol, n_states=STATES,
                                 update_model=False, velocity_factor=2., save_final_model=False)
        result.update({'model_predict_calls': sum(m.calls for m in instances),
                       'model_predict_seconds': sum(m.seconds for m in instances),
                       'reference_seconds': ledger.reference_seconds})
        base.write(OUT / f'{arm}_{seed}_summary.json', result)
        return result
    except Exception as exc:
        base.write(OUT / f'{arm}_{seed}_failure.json', {'error': repr(exc), 'attempts': ledger.n})
        raise


def run():
    prepare()
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, arm, seed, start): [arm, seed] for seed, start in zip(SEEDS, STARTS) for arm in base.ARMS}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 -- preserve all arms, then fail the batch below
                failures.append({'task': futures[future], 'error': repr(exc)})
            base.write(OUT / 'pilot_summary.json', {'status': 'running', 'arms': results, 'failures': failures})
    attempt = 0
    # Preserve worker ledgers and build the legacy analysis input with globally
    # unique IDs. Identical cross-worker calls are still charged separately.
    with (OUT / 'pilot_reference_attempts.jsonl').open('x') as combined:
        for path in sorted(OUT.glob('pilot_*_reference_attempts.jsonl')):
            ids = {}
            for line in path.read_text().splitlines():
                row = json.loads(line)
                local_id = row['attempt']
                if local_id not in ids:
                    attempt += 1
                    ids[local_id] = attempt
                row.update({'worker_ledger': path.name, 'worker_attempt': local_id, 'attempt': ids[local_id]})
                combined.write(json.dumps(row) + '\n')
    base.write(OUT / 'pilot_summary.json', {'status': 'failed' if failures else 'complete',
               'arms': results, 'failures': failures, 'forward_reference_attempts': attempt,
               'compatibility_attempts': 1, 'total_reference_attempts': attempt + 1})
    if failures:
        raise RuntimeError(f'{len(failures)} arm(s) failed; completed outputs retained')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
