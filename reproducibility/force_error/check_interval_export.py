"""Reconstruct a formal interval export using only NumPy and the standard library.

Default acceptance requires all eight 80-fs branches (1200 segments).
--test-prefix explicitly tests a shorter export without accepting formal completion.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np

def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))
def close(a,b):
    np.testing.assert_allclose(np.asarray(a,float),np.asarray(b,float),rtol=2e-9,atol=1e-10)
def check(directory,test_prefix=False):
    report=json.loads((directory/'INTERVAL_RESULT.json').read_text())
    duration=float(report['duration_fs'])
    assert (0<duration<80 and not report['complete_time_coverage']) if test_prefix else (duration==80 and report['complete_time_coverage'])
    assert report['DFT_calls']==0 and report['new_branches']==6 and report['reused_baseline_paths']==2
    metrics=rows(directory/'interval_metrics.csv')
    strips=rows(directory/'interval_formal_segments.csv')
    assert len(metrics)==8
    keys={(int(m['configuration']),float(m['tau_fs'])) for m in metrics}
    assert keys=={(r,t) for r in [1,2] for t in [.25,.5,1.,2.]}
    sensitivity=report['sensitivity']
    dk=float(sensitivity['delta_kappa_star_eV_A2']);dx=float(sensitivity['delta_x_star_A'])
    diagnostics=report['diagnostics']
    close(dk,max([.0018101175133137248]+[d['delta_kappa_eV_A2'] for d in diagnostics if 'delta_kappa_eV_A2' in d]))
    close(dx,max([9.358712126481334e-8]+[d['delta_x_A'] for d in diagnostics if 'delta_x_A' in d]))
    starts={};reconstructed={};count=0
    for m in metrics:
        r=int(m['configuration']);tau=float(m['tau_fs'])
        z=np.load(directory/f'configuration_{r}_tau{tau:g}.npz',allow_pickle=False)
        n=int(round(duration/tau));assert n*tau==duration
        assert len(z['W_eV'])==n==int(m['n_segments'])
        ss=[s for s in strips if int(s['configuration'])==r and float(s['tau_fs'])==tau]
        assert len(ss)==n
        assert [int(s['segment']) for s in ss]==list(range(n))
        close(z['start_time_fs'],np.arange(n)*tau)
        close(z['end_time_fs'],np.arange(1,n+1)*tau)
        close(z['duration_fs'],tau)
        close(z['center_positions_A'][1:],z['end_positions_A'][:-1])
        close(z['center_momenta_ASE'][1:],z['end_momenta_ASE'][:-1])
        close(z['center_base_energy_eV'][1:],z['end_base_energy_eV'][:-1])
        close(z['center_reference_energy_eV'][1:],z['end_reference_energy_eV'][:-1])
        start=(z['center_positions_A'][0],z['center_momenta_ASE'][0],
               z['center_base_energy_eV'][0],z['center_reference_energy_eV'][0])
        if r in starts:
            for a,b in zip(start,starts[r]):close(a,b)
        else:starts[r]=start
        displacement=z['end_positions_A']-z['center_positions_A']
        correction_work=np.einsum('nij,nij->n',z['c_eV_A'],displacement)
        eb=z['end_base_energy_eV']-z['center_base_energy_eV']
        er=z['end_reference_energy_eV']-z['center_reference_energy_eV']
        kinetic_start=np.sum(z['center_momenta_ASE']**2,axis=(1,2))/(2*28.085)
        kinetic_end=np.sum(z['end_momenta_ASE']**2,axis=(1,2))/(2*28.085)
        work=er-eb+correction_work
        defect=eb+kinetic_end-kinetic_start-correction_work
        close(work,z['W_eV']);close(defect,z['D_eV'])
        kappa=np.einsum('nij,nij->n',z['u'],z['H_delta_u'])
        close(np.sum(z['u']**2,axis=(1,2)),1)
        momenta_norm=np.linalg.norm(z['center_momenta_ASE'].reshape(n,-1),axis=1)
        close(z['u'],z['center_momenta_ASE']/momenta_norm[:,None,None])
        # ASE time unit in fs; Si masses are fixed at 28.085 amu.
        speed=momenta_norm/28.085*.09822694788464063
        close(speed,z['velocity_norm_A_per_fs']);close(kappa,z['kappa_eV_A2'])
        predicted=.5*speed**2*kappa*tau**2
        close(predicted,z['W_pred_eV'])
        residual=z['end_base_forces_eV_A']+z['c_eV_A']-z['end_reference_forces_eV_A']
        residual_norm=np.linalg.norm(residual.reshape(n,-1),axis=1)
        close(residual_norm,z['endpoint_residual_norm_eV_A'])
        eta=1e-10+abs(defect)+.5*speed**2*tau**2*dk+residual_norm*dx
        close(eta,z['eta_empirical_eV'])
        error=predicted-work;norm=abs(work).sum()
        direct_h=float(er.sum()+kinetic_end[-1]-kinetic_start[0])
        close(direct_h,np.sum(work+defect))
        result=dict(sum_W_eV=float(work.sum()),delta_H_ref_eV=direct_h,
                    rate_W_meV_atom_ps=work.sum()*1e6/64/duration,
                    rate_H_meV_atom_ps=direct_h*1e6/64/duration,
                    predicted_rate_meV_atom_ps=predicted.sum()*1e6/64/duration,
                    mean_v2kappa_eV_fs2=np.mean(speed**2*kappa),
                    E_seg_percent=100*abs(error).sum()/norm,
                    E_cum_percent=100*max(abs(np.cumsum(error)))/norm,
                    max_segment_error_percent=100*max(abs(error/work)),
                    U_over_S_percent=100*eta.sum()/norm,
                    sum_D_eV=defect.sum(),sum_abs_D_over_S_percent=100*abs(defect).sum()/norm,
                    n_positive=int((work>0).sum()),n_negative=int((work<0).sum()),
                    n_weak=int((abs(work)<=5*eta).sum()),endpoint_bias_percent=100*error.sum()/norm)
        jr=next(q for q in report['metrics'] if q['configuration']==r and q['tau_fs']==tau)
        for name,value in result.items():close(value,m[name]);close(value,jr[name])
        p0=z['center_momenta_ASE'][0].ravel()
        anchor_p=np.concatenate([z['center_momenta_ASE'].reshape(n,-1),z['end_momenta_ASE'][-1].reshape(1,-1)])
        angle=np.degrees(np.arccos(np.clip(anchor_p@p0/(np.linalg.norm(anchor_p,axis=1)*np.linalg.norm(p0)),-1,1)))
        close(angle[-1],m['final_velocity_rotation_degrees'])
        close(angle.max(),m['maximum_anchor_velocity_rotation_degrees'])
        assert jr['dense_quadrature_available']==(tau==1)
        if tau!=1:assert jr['max_dense_quadrature_difference_eV'] is None and m['max_dense_quadrature_difference_eV']==''
        for name,values in [('W_eV',work),('W_pred_eV',predicted),('D_eV',defect),
                            ('kappa_eV_A2',kappa),('speed_A_fs',speed),('eta_eV',eta),
                            ('start_fs',z['start_time_fs']),('end_fs',z['end_time_fs']),
                            ('W_cumulative_meV_atom',np.cumsum(work)*1000/64),
                            ('predicted_cumulative_meV_atom',np.cumsum(predicted)*1000/64)]:
            close([s[name] for s in ss],values)
        if not test_prefix:
            times=[d['start_time_fs'] for d in diagnostics if d['configuration']==r and d['tau_fs']==tau]
            for t in ([0.,35.,65.] if tau==1 else [0.,32.,64.]):assert t in times,(r,tau,times)
        reconstructed[(r,tau)]=result
        count+=n
    assert len(strips)==count
    for m in metrics:
        r=int(m['configuration']);tau=float(m['tau_fs'])
        a=reconstructed[(r,tau)];b=reconstructed[(r,1.)]
        close(m['work_ratio_to_tau1'],a['sum_W_eV']/b['sum_W_eV'])
        close(m['work_ratio_divided_by_tau_ratio'],a['sum_W_eV']/b['sum_W_eV']/tau)
        close(m['mean_factor_ratio_to_tau1'],a['mean_v2kappa_eV_fs2']/b['mean_v2kappa_eV_fs2'])
    for r in [1,2]:
        taus=[.25,.5,1.,2.]
        slope=np.polyfit(np.log(taus),np.log([reconstructed[(r,t)]['rate_W_meV_atom_ps'] for t in taus]),1)[0]
        close(slope,report['descriptive_log_slopes'][str(r)])
    if not test_prefix:assert count==1200
    return dict(passed=True,formal_completion=not test_prefix,duration_fs=duration,branches=8,segments=count,
                independent_endpoint_work_and_integration_defect=True,
                momentum_velocity_and_curvature=True,same_initial_states=True,
                all_metrics_and_plot_rows=True,new_DFT_calls=0,no_model_calls=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--test-prefix',action='store_true',help='QA only: accept an explicitly incomplete export.')
    args=ap.parse_args()
    print(json.dumps(check(args.directory,args.test_prefix),indent=2))
if __name__=='__main__':main()
