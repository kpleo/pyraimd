"""Representative framework validation using the single adapted frozen model.

Four new velocity seeds are paired across three kinetic-energy levels. Every
policy generates its own path; only revealed labels update controller history.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cloud_hot_forward_20260905 as hot

base = hot.base
ROOT = hot.ROOT
OUT = ROOT / 'analysis/execution_20260905/h2o_feasibility'
ADAPT = ROOT / 'analysis/execution_20260905/h2o_hot_adapt'
CHECKPOINT_SHA = '12d92bcb1e0aa5c0161ebb464f7a83d460ebf932fa1889bf6d5bceb1c6296b36'
SEEDS = [2026090541, 2026090542, 2026090543, 2026090544]
STARTS = [0, 151, 0, 151]
LEVELS = [('kinetic_300', 1.), ('kinetic_600', math.sqrt(2.)), ('kinetic_1200', 2.)]
STATES = 50
WORKERS = 12


def restored():
    import torch
    state = torch.load(ADAPT / 'adapted_checkpoint.pt', weights_only=False, map_location='cpu')
    if state['model_specs'] != [str(base.MODEL)] * 4 or state['n_members'] != 4:
        raise RuntimeError('Unexpected adapted checkpoint identity')
    model = base.model()
    model.load_state_dict(state)
    return hot.TimedModel(model)


def prepare():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run in an allocated compute job')
    hot.configure()
    OUT.mkdir(parents=True, exist_ok=False)
    if base.sha(ADAPT / 'adapted_checkpoint.pt') != CHECKPOINT_SHA or base.sha(base.MODEL) != hot.MODEL_SHA:
        raise RuntimeError('Model identity changed')
    adapted = json.loads((ADAPT / 'summary.json').read_text())
    if adapted['status'] != 'complete' or adapted['new_dft_calls'] != 0:
        raise RuntimeError('Incomplete preceding adaptation')
    predictions = json.loads((ADAPT / 'validation_post.json').read_text())
    calibration = [(r['spread_max_ev_A'], r['error_max_atom_ev_A']) for r in predictions['cold_calibration']]
    paths = [{'seed': seed, 'pairs': [(r['spread_max_ev_A'], r['error_max_atom_ev_A'])
                                     for r in predictions[f'old_hot_{seed}']]}
             for seed in [2026090501, 2026090502]]
    candidates = [hot.replay(paths, calibration, 'periodic', k) for k in [1, 2, 4, 8, 16]]
    usable = [r for r in candidates if r['usable']]
    eligible = [r for r in usable if r['eligible']]
    if eligible:
        choice = min(eligible, key=lambda r: (r['audit_adjusted_requests'], r['violations'], r['setting']))
        status = 'passed_observed_development_risk'
    else:
        choice = min(usable, key=lambda r: (r['worst_path_risk'], r['audit_adjusted_requests'], r['setting']))
        status = 'nontrivial_fallback_failed_development_risk'
    q1 = hot.replay(paths, calibration, 'calibrated', 1.)
    if not q1['usable']:
        raise RuntimeError('Fixed q1 calibration is not usable on existing development paths')
    source_files = [Path(__file__), Path(__file__).with_suffix('.sbatch'), Path(base.__file__),
                    Path(hot.__file__), ROOT / 'experiments/analyze_forward_pilot_20260905.py',
                    ROOT / 'src/pyraimd2/surrogate/committee.py']
    input_files = [ADAPT / n for n in ['locked_protocol.json', 'summary.json', 'validation_post.json']]
    protocol = {
        'locked_unix': time.time(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'purpose': 'Representative molecular realization of the general physical-horizon and verification framework, with nontrivial propagation and transparent reference cost',
        'seeds': SEEDS, 'start_indices': STARTS, 'arms': base.ARMS,
        'velocity_factors': dict(LEVELS), 'pilot_evaluation_states': STATES,
        'dt_fs': base.DT_FS, 'epsilon_ev_A': base.EPS, 'audit_p': base.AUDIT_P,
        'update_model': False, 'growth_gamma': 2., 'h_env_fs': 8.,
        'calibration_pairs': calibration, 'selected': {'periodic': choice['setting'], 'calibrated': 1.},
        'periodic_development_status': status, 'periodic_candidates': candidates,
        'calibrated_q1_development': q1,
        'model_sha256': hot.MODEL_SHA, 'checkpoint_sha256': CHECKPOINT_SHA,
        'source_sha256': {str(p.relative_to(ROOT)): base.sha(p) for p in source_files},
        'input_sha256': {str(p.relative_to(ROOT)): base.sha(p) for p in input_files},
        'reference_attempt_limit': 2400, 'per_arm_attempt_limit': 50,
        'prior_P1_reference_attempts': 795, 'P1_ceiling_including_this_batch': 3195,
        'prior_velocity_reference_attempts_separately_counted': 122,
        'worker_count': WORKERS, 'threads_per_worker': 1, 'job_wall_limit_seconds': 10800,
        'primary_reporting': 'Per seed and kinetic level: accepted count, accepted violations/max error, policy reference requests, reference-Hamiltonian drift and paired bond/angle differences. Display every path.',
        'practical_feasibility_criterion': 'At a tested kinetic level, all four seeds have >=20% accepted states, <=5% observed accepted-force violations, and <=90% policy reference requests. These thresholds define this screening decision, not statistical confidence or a new physical constant.',
        'comparative_criterion': 'Claim an error/reference-cost advantage only if actual paired outcomes support it; all-reference fallback and zero accepted points never count as successful surrogate propagation.',
        'independence': 'Four new velocity directions at two known geometries. Same seeds paired across levels and policies; 48 paths are not 48 independent initial conditions.',
        'scope': 'Framework generality follows from its material/model-independent definitions and stated mathematical assumptions. This molecular experiment samples representative instantiations; it does not define the framework applicability domain. Kinetic normalizations are not equilibrated ensembles; empirical horizons do not certify DFT curvature or guarantee every implementation is efficient.',
        'amendment': 'User authorized a larger scientifically targeted compute investment and distinguished general framework design from finite validation coverage. This is a new frozen-model validation protocol, not completion of the interrupted online-training pilot.',
    }
    base.write(OUT / 'locked_protocol.json', protocol)
    for p in source_files:
        (OUT / f'source_{p.name}').write_bytes(p.read_bytes())
    for name, factor in LEVELS:
        folder = OUT / name
        folder.mkdir()
        base.write(folder / 'locked_protocol.json', dict(protocol, velocity_factor=factor,
                   kinetic_energy_normalization_K=300*factor*factor, parent_protocol_sha256=base.sha(OUT / 'locked_protocol.json')))
    print(json.dumps({'locked': True, 'selected_period': choice['setting'], 'period_status': status,
                      'q1_development_eligible': q1['eligible'], 'tasks': 48, 'attempt_limit': 2400}), flush=True)


def worker(name, factor, arm, seed, start):
    hot.configure()
    base.OUT = OUT / name
    instances = []

    def tracked():
        model = restored()
        instances.append(model)
        return model

    base.restored = tracked
    protocol = json.loads((base.OUT / 'locked_protocol.json').read_text())
    ledger = base.ReferenceLedger(f'pilot_{arm}_{seed}', STATES, 10000)
    try:
        result = base.trajectory(arm, seed, start, ledger, protocol, n_states=STATES,
                                 update_model=False, velocity_factor=factor, save_final_model=False)
        result.update({'level': name, 'velocity_factor': factor,
                       'model_predict_calls': sum(m.calls for m in instances),
                       'model_predict_seconds': sum(m.seconds for m in instances),
                       'reference_seconds': ledger.reference_seconds})
        base.write(base.OUT / f'{arm}_{seed}_summary.json', result)
        return result
    except Exception as exc:
        base.write(base.OUT / f'{arm}_{seed}_failure.json', {'error': repr(exc), 'attempts': ledger.n})
        raise


def run():
    prepare()
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, name, factor, arm, seed, start): [name, arm, seed]
                   for name, factor in LEVELS for seed, start in zip(SEEDS, STARTS) for arm in base.ARMS}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 -- retain other arms and fail the completed batch below
                failures.append({'task': futures[future], 'error': repr(exc)})
            base.write(OUT / 'summary.json', {'status': 'running', 'arms': results, 'failures': failures})
    total = 0
    for name, _ in LEVELS:
        folder = OUT / name
        attempt = 0
        with (folder / 'pilot_reference_attempts.jsonl').open('x') as stream:
            for path in sorted(folder.glob('pilot_*_reference_attempts.jsonl')):
                ids = {}
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    local_id = row['attempt']
                    if local_id not in ids:
                        attempt += 1
                        ids[local_id] = attempt
                    row.update({'worker_ledger': path.name, 'worker_attempt': local_id, 'attempt': ids[local_id]})
                    stream.write(json.dumps(row) + '\n')
        total += attempt
    base.write(OUT / 'summary.json', {'status': 'failed' if failures else 'complete', 'arms': results,
                                     'failures': failures, 'total_reference_attempts': total})
    if failures:
        raise RuntimeError(f'{len(failures)} task(s) failed; records retained')


if __name__ == '__main__':
    run()
