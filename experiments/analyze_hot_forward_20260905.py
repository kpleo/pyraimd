"""Diagnose recorded hot-path rejection without generating labels or predictions."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/execution_20260905/h2o_hot_cloud'


def readl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    result = {'source': 'saved records only; post-hoc descriptive diagnosis', 'paths': []}
    for seed in [2026090521, 2026090522]:
        r = readl(OUT / f'horizon_{seed}_steps.jsonl')
        anchors = readl(OUT / f'horizon_{seed}_anchors.jsonl')
        by_step = {a['step']: a for a in anchors}
        counter = {'seed': seed, 'states': len(r), 'post_initial_states': len(r)-1,
                   'accepted': sum(x['accepted'] for x in r),
                   'model_within_budget_on_reference_path': sum(x['error_ev_A'] <= .1 for x in r),
                   'anchor_model_exceeds_budget': sum(a['e0'] > .1 for a in anchors),
                   'zero_announced_horizon': sum(a['horizon_fs'] == 0 for a in anchors),
                   'zero_horizon_despite_valid_anchor': sum(a['horizon_fs'] == 0 and a['e0'] <= .1 for a in anchors),
                   'spread_blocks': sum(x['bound'] > .1 for x in r[1:]),
                   'time_blocks': sum(x['anchor_age_fs'] > x['horizon_fs'] + 1e-10 for x in r[1:]),
                   'motion_blocks': sum(x['motion_bound'] is None or x['motion_bound'] > .1 for x in r[1:]),
                   'safe_next_point_but_zero_previous_horizon': 0,
                   'safe_next_point_and_valid_anchor_but_zero_horizon': 0,
                   'max_first_forecast_distance_discrepancy_A': 0.,
                   'first_forecast_bound_violation_count': 0}
        for j, nxt in enumerate(r[1:]):
            a = by_step[j]
            safe = nxt['error_ev_A'] <= .1
            counter['safe_next_point_but_zero_previous_horizon'] += int(safe and a['horizon_fs'] == 0)
            counter['safe_next_point_and_valid_anchor_but_zero_horizon'] += int(safe and a['e0'] <= .1 and a['horizon_fs'] == 0)
            if a['forecast']:
                distance = np.linalg.norm(np.array(nxt['positions']) - np.array(r[j]['positions']))
                counter['max_first_forecast_distance_discrepancy_A'] = max(counter['max_first_forecast_distance_discrepancy_A'], abs(float(distance) - a['forecast'][0][1]))
                counter['first_forecast_bound_violation_count'] += int(nxt['error_ev_A'] > a['forecast'][0][2] + 1e-12)
        assert counter['max_first_forecast_distance_discrepancy_A'] < 1e-12
        result['paths'].append(counter)
    result['interpretation'] = 'Model error above budget and conservative motion/spread admission both matter. Zero accepted points gives undefined accepted risk, not a zero-risk efficiency result. Saved one-step comparisons are hindsight, not a certified alternative controller.'
    (OUT / 'rejection_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
