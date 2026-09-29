"""Reconstruct all formal work predictions from the compact numerical archive."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parent

def close(a,b,atol=1e-10):np.testing.assert_allclose(a,b,atol=atol,rtol=2e-10)

def main():
    curves=json.loads((ROOT/'curves.json').read_text())['trajectories']
    metrics=list(csv.DictReader((ROOT/'summary.csv').open()))
    verified=[]
    for d in curves:
        ref=d['reference'];r=d['replica'];n=500 if ref=='MACE' else 80
        with np.load(ROOT/f'{ref.lower()}_configuration_{r+1}.npz',allow_pickle=False) as z:
            assert len(z['W_eV'])==n
            mass=z['masses_amu'].reshape(64,1)
            assert np.all(mass==28.085)
            assert np.all(z['duration_fs']==1) and np.all(z['h_A']==.02)
            v=z['start_momenta_ASE']/mass*z['ase_units_fs'];speed=np.linalg.norm(v,axis=(-2,-1))
            u=v/speed[:,None,None];close(u,z['u']);close(speed,z['velocity_norm_A_per_fs'])
            for role,sign in [('plus',1),('minus',-1)]:
                close(z[role+'_positions_A']-z['center_positions_A'],sign*z['h_A'][:,None,None]*u)
            c=z['center_reference_forces_eV_A']-z['center_base_forces_eV_A'];close(c,z['c_eV_A'])
            minus=z['minus_reference_forces_eV_A']-z['minus_base_forces_eV_A']
            plus=z['plus_reference_forces_eV_A']-z['plus_base_forces_eV_A']
            response=(minus-plus)/(2*z['h_A'][:,None,None]);close(response,z['H_delta_u'])
            k=np.einsum('nij,nij->n',u,response);close(k,z['kappa_eV_A2'])
            pred=.5*speed**2*k*z['duration_fs']**2;close(pred,z['W_pred_eV'])
            dx=z['end_positions_A']-z['center_positions_A'];linear=np.einsum('nij,nij->n',c,dx)
            db=z['end_base_energy_eV']-z['center_base_energy_eV']
            dr=z['end_reference_energy_eV']-z['center_reference_energy_eV']
            work=dr-db+linear;close(work,z['W_eV'])
            dk=np.sum((z['end_momenta_ASE']**2-z['start_momenta_ASE']**2)/(2*mass),axis=(-2,-1))
            defect=dk+db-linear;close(defect,z['D_eV']);close(dk+dr,z['delta_H_ref_eV'])
            e=np.linalg.norm(z['end_base_forces_eV_A']+c-z['end_reference_forces_eV_A'],axis=-1).max(axis=-1)
            close(e,z['epsilon_actual_eV_A'])
            close(z['end_positions_A'][:-1],z['center_positions_A'][1:])
            close(z['end_momenta_ASE'][:-1],z['start_momenta_ASE'][1:])
            close(z['start_momenta_ASE'].sum(axis=1),0)
            assert np.all(z['plus_completed_unix']<=z['frozen_unix'])
            assert np.all(z['minus_completed_unix']<=z['frozen_unix'])
            assert np.all(z['frozen_unix']<z['motion_started_unix'])
            assert np.all(z['motion_started_unix']<z['end_completed_unix'])
            S=np.abs(work).sum();error=pred-work;Eseg=100*np.abs(error).sum()/S;Ecum=100*np.abs(np.cumsum(error)).max()/S
            row=next(m for m in metrics if m['reference']==ref and int(m['replica'])==r)
            close(Eseg,float(row['E_seg_percent']));close(Ecum,float(row['E_cum_percent']))
            close(d['segment_W_eV'],work);close(d['segment_W_pred_eV'],pred);close(d['formal_S_eV'],S)
            curve=d['formal_curve'];close(curve['time_fs'],np.arange(n+1))
            for key,arr in [('W_cumulative_eV',work),('W_pred_cumulative_eV',pred),('D_cumulative_eV',defect),
                    ('delta_H_ref_direct_eV',dk+dr),('W_pred_plus_D_eV',pred+defect),('abs_segment_W_cumulative_eV',np.abs(work))]:
                close(curve[key],np.r_[0,np.cumsum(arr)])
            full=d['full_curve']
            for key in curve:
                if key=='time_fs':close(np.array(full[key])[20:]-20,curve[key])
                else:close(np.array(full[key])[20:]-full[key][20],curve[key])
            U=np.sum(d['eta_empirical_eV']);close(100*U/S,float(row['empirical_U_over_S_percent']))
            assert np.all(work>0) and np.all(np.abs(work)>5*np.array(d['eta_empirical_eV']))
            verified.append(dict(reference=ref,configuration=r+1,segments=n,E_seg_percent=Eseg,E_cum_percent=Ecum))
    assert sum(r['segments'] for r in verified)==1160
    print(json.dumps({'passed':True,'reconstructed':verified},indent=2))

if __name__=='__main__':main()
