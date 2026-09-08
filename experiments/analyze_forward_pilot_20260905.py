"""Independent arithmetic/kinematic verification of recorded prospective runs.
No dynamics, fitting, reference evaluations or parameter selection occurs here.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np
from ase import units
from pyraimd2.store import Store

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/execution_20260905/h2o'
DT = .5 * units.fs
EPS = .10


def readl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=OUT)
    OUT=parser.parse_args().directory
    protocol=json.loads((OUT/'locked_protocol.json').read_text())
    target_states=protocol['pilot_evaluation_states']
    update_model=protocol.get('update_model',True)
    old = list(Store(ROOT/'analysis/h2o_streak/coverage_h2o.db').iter_labels('collect-h2o-nve-300K'))
    masses = old[0][0].get_masses()[:, None]
    symbols = old[0][0].get_chemical_symbols()
    oxygen = symbols.index('O')
    hs = [i for i, s in enumerate(symbols) if s == 'H']
    references = {}
    for file in OUT.glob('reference_*_steps.jsonl'):
        references[int(file.stem.split('_')[1])] = readl(file)
    ledger = readl(OUT/'pilot_reference_attempts.jsonl')
    started = [r for r in ledger if r['status']=='started']
    completed = [r for r in ledger if r['status']=='complete']
    seconds = {r['tag']: r['seconds'] for r in completed}
    rows, checks = [], []
    for file in sorted(OUT.glob('*_steps.jsonl')):
        arm, seed = file.stem.split('_')[:2]
        seed = int(seed)
        records = readl(file)
        proposals = readl(OUT/f'{arm}_{seed}_proposals.jsonl')
        assert len(proposals) >= len(records)
        x = np.array([r['positions'] for r in records])
        p = np.array([r['momenta_full'] for r in records])
        f = np.array([r['driving_forces'] for r in records])
        fref = np.array([r['reference_forces'] for r in records])
        N = sum(r['accepted'] for r in records)
        V = sum(r['violation'] for r in records)
        D = sum(r['violation'] and r['audit'] for r in records)
        rng = np.random.default_rng([seed,919])
        nvisible = 0
        ferr = proposal_error = 0.
        for j,(r,q) in enumerate(zip(records,proposals)):
            assert r['step']==q['step']==j and r['accepted']==q['accepted']
            audit = bool(rng.random()<.1) if r['accepted'] else False
            assert audit == r['audit']
            assert r['revealed'] == (not r['accepted'] or audit)
            assert r['model_updates_before'] == (nvisible//8 if arm!='reference' and update_model else 0)
            if arm!='reference':
                fs = np.array(r['surrogate_forces'])
                e = np.linalg.norm(fs-fref[j],axis=1).max()
                ferr = max(ferr,abs(e-r['error_ev_A']))
                proposal_error = max(proposal_error,float(np.abs(fs-np.array(q['surrogate_forces'])).max()))
                assert r['violation']==(r['accepted'] and e>EPS)
                np.testing.assert_allclose(f[j],fs if r['accepted'] else fref[j],atol=1e-12,rtol=0)
            else:
                np.testing.assert_allclose(f[j],fref[j],atol=1e-12,rtol=0)
            nvisible += int(r['revealed'])
        dx = float(np.abs(x[1:]-x[:-1]-DT*(p[:-1]+.5*DT*f[:-1])/masses).max()) if len(x)>1 else 0.
        dp = float(np.abs(p[1:]-p[:-1]-.5*DT*(f[1:]+f[:-1])).max()) if len(x)>1 else 0.
        assert dx<1e-12 and dp<1e-12 and ferr<1e-12 and proposal_error<1e-12
        ke = .5*np.sum(p*p/masses,axis=(1,2))
        np.testing.assert_allclose(ke,[r['kinetic_energy_ev'] for r in records],atol=1e-12,rtol=0)
        href = ke + np.array([r['reference_energy_ev'] for r in records])
        delta = href-href[0]
        power = np.sum((p/masses)*(f-fref),axis=(1,2))
        work = np.r_[0,np.cumsum(.5*DT*(power[1:]+power[:-1]))]
        other = np.array([r['positions'] for r in references[seed][:len(x)]])
        ncompare = min(len(x),len(other))
        def obs(xx):
            d = xx[:,hs,:]-xx[:,[oxygen],:]
            bonds = np.linalg.norm(d,axis=2)
            cosine = np.sum(d[:,0]*d[:,1],axis=1)/(bonds[:,0]*bonds[:,1])
            return bonds,np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
        bonds,angle = obs(x[:ncompare]); bref,aref = obs(other)
        rsec = sum(seconds.get(f'{arm}:{seed}:{r["step"]}',0.) for r in records)
        online_sec = sum(seconds.get(f'{arm}:{seed}:{r["step"]}',0.) for r in records if r['revealed'])
        row={'arm':arm,'seed':seed,'states':len(records),'complete':len(records)==target_states,
             'accepted':N,'violations':V,'risk':V/N if N else None,'audit_detections':D,
             'online_requests':sum(r['revealed'] for r in records),'audit_requests':sum(r['audit'] for r in records),
             'new_reference_attempts_completed':sum(f'{arm}:{seed}:{r["step"]}' in seconds for r in records),
             'all_reference_seconds':rsec,'revealed_new_reference_seconds':online_sec,
             'sequential_upper':min(1.,(math.log(2)*D+math.log(20))/(N*(-math.log(.95)))) if N else None,
             'max_Href_drift_meV':float(abs(delta).max()*1000),
             'end_Href_drift_meV':float(delta[-1]*1000),
             'max_work_balance_residual_meV':float(abs(delta-work).max()*1000),
             'paired_geometry_states':ncompare,
             'OH_bond_RMSE_A':float(np.sqrt(np.mean((bonds-bref)**2))),
             'HOH_angle_RMSE_degrees':float(np.sqrt(np.mean((angle-aref)**2))),
             'max_OH_bond_difference_A':float(np.max(np.abs(bonds-bref))),
             'max_HOH_angle_difference_degrees':float(np.max(np.abs(angle-aref)))}
        summary = OUT/f'{arm}_{seed}_summary.json'
        if summary.exists():
            s = json.loads(summary.read_text())
            for k in ['accepted','violations','audit_detections']:
                assert row[k]==s[k]
            row['training_updates']=len(s['training_updates'])
            row['training_seconds']=sum(r['wall_time_s'] for r in s['training_updates'])
            row['recorded_arm_wall_seconds']=s['wall_seconds']
        checks.append({'arm':arm,'seed':seed,'max_Verlet_position_error_A':dx,
                       'max_Verlet_momentum_error_ASE':dp,'max_force_norm_error':ferr,
                       'max_frozen_proposal_force_difference':proposal_error,'audit_stream_and_reveal_consistent':True})
        rows.append(row)
    aggregate=[]
    for arm in ['reference','periodic','calibrated','horizon']:
        subset=[r for r in rows if r['arm']==arm and r['complete']]
        if not subset: continue
        n=sum(r['accepted'] for r in subset); v=sum(r['violations'] for r in subset)
        aggregate.append({'arm':arm,'complete_trajectories':len(subset),'states':sum(r['states'] for r in subset),
                          'accepted':n,'violations':v,'pooled_descriptive_risk':v/n if n else None,
                          'online_requests':sum(r['online_requests'] for r in subset),
                          'mean_bond_RMSE_A':float(np.mean([r['OH_bond_RMSE_A'] for r in subset])),
                          'mean_angle_RMSE_degrees':float(np.mean([r['HOH_angle_RMSE_degrees'] for r in subset]))})
    expected_arms=len(protocol['seeds'])*len(protocol.get('arms',['reference','periodic','calibrated','horizon']))
    result={'status':'complete' if len(rows)==expected_arms and all(r['complete'] for r in rows) else 'partial',
            'source':'recorded trajectories only; no additional reference or model evaluation',
            'epsilon_ev_A':EPS,'dt_fs':.5,'independent_initial_condition_seeds':len(protocol['seeds']),'target_states':target_states,'update_model':update_model,
            'arms':rows,'aggregate':aggregate,'checks':checks,
            'reference_attempts':len(started),'completed_attempts':len(completed),
            'reference_seconds':sum(r['seconds'] for r in completed),
            'unfinished_or_failed_attempt_ids':sorted(set(r['attempt'] for r in started)-set(r['attempt'] for r in completed)),
            'limits':'Forward initial states use known archived geometries and new velocities. Paired observables compare separate trajectories over (target_states-1)*0.5 fs, not equilibrium distributions. Work uses trapezoidal on-step residual power and is not an exact continuous-time integral. Sequential bounds are per arm and seed; no simultaneous pooling claim. Measured revealed-reference seconds exclude historical cache costs, training and other runtime; they are not a speedup.'}
    common=[]
    for seed in protocol['seeds']:
        paths={arm:OUT/f'{arm}_{seed}_steps.jsonl' for arm in ['reference','periodic','calibrated','horizon']}
        if not all(p.exists() for p in paths.values()):
            continue
        groups={arm:readl(path) for arm,path in paths.items()}
        count=min(map(len,groups.values()))
        if not count:
            continue
        prefix={'seed':seed,'matched_states':count,'duration_fs':(count-1)*.5,
                'definition':'Largest complete common prefix, limited by external runtime stop. Descriptive comparison only.', 'arms':[]}
        xr=np.array([r['positions'] for r in groups['reference'][:count]])
        br,ar=obs(xr)
        for arm,rr in groups.items():
            subset=rr[:count]
            energies=np.array([r['reference_energy_ev']+r['kinetic_energy_ev'] for r in subset])
            b,a=obs(np.array([r['positions'] for r in subset]))
            prefix['arms'].append({'arm':arm,'accepted':sum(r['accepted'] for r in subset),
                'violations':sum(r['violation'] for r in subset),
                'reference_requests':sum(r['revealed'] for r in subset),
                'audit_requests':sum(r['audit'] for r in subset),
                'max_energy_drift_meV':float(abs(energies-energies[0]).max()*1000),
                'OH_bond_RMSE_A':float(np.sqrt(np.mean((b-br)**2))),
                'HOH_angle_RMSE_degrees':float(np.sqrt(np.mean((a-ar)**2)))})
        common.append(prefix)
    result['matched_prefixes']=common
    if len(common)==1:
        write(OUT/'matched_prefix.json',common[0])
    write(OUT/'pilot_independent_analysis.json',result)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'pilot_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':result['status'],'attempts':result['reference_attempts'],'aggregate':aggregate}))


if __name__=='__main__': main()
