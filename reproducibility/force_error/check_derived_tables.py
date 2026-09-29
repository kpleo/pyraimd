"""Reconstruct plotted quantities from the accompanying numerical records."""
from pathlib import Path
import math
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
P=ROOT
D=ROOT/'data'
def rows(path):return list(csv.DictReader(path.open()))
def close(a,b):np.testing.assert_allclose(np.asarray(a,float),np.asarray(b,float),rtol=2e-9,atol=1e-10)
def rank(x):
 x=np.asarray(x);order=np.argsort(x,kind='stable');out=np.empty(len(x),float)
 i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  out[order[i:j]]=(i+j-1)/2+1;i=j
 return out
def rho(x,y):return float(np.corrcoef(rank(x),rank(y))[0,1])
def block_integral(n,order=768):
 nodes,weights=np.polynomial.legendre.leggauss(order);t=60*(nodes+1)
 density=np.sqrt(t/(2*np.pi))*np.exp(-t/2)
 F=np.array([math.erf(math.sqrt(v/2)) for v in t])-2*density
 F=np.maximum(F,np.finfo(float).tiny)
 truncated_mean=3*np.maximum(F-2*t*density/3,0)/F
 return np.sum(60*weights*n*density*np.exp((n-1)*np.log(F))*(1+(n-1)*truncated_mean/t))
def main():
 atoms=rows(D/'atomic_work.csv');original=rows(P/'atomic_work/figure_a_atoms.csv')
 assert len(atoms)==len(original)==474
 for a,b in zip(atoms,original):
  assert int(a['atom'])==int(b['atom_index_zero_based'])
  close(a['Q_meV'],b['integrated_Q_meV'])
  close(a['residual_eV_A'],np.linalg.norm([float(b['residual_'+x+'_eV_per_A']) for x in 'xyz']))
 assert int(max(atoms,key=lambda a:float(a['residual_eV_A']))['atom'])==157
 events=rows(D/'normalized_events.csv');old=rows(P/'tables/events.csv')
 directions={(r['anchor'],int(r['index'])):r for r in rows(P/'tables/directions.csv')}
 assert len(events)==len(old)==256
 eventmap={(r['anchor'],int(r['direction']),int(r['budget'])):r for r in events}
 assert len(eventmap)==256
 for r in events:
  d=directions[(r['anchor'],int(r['direction']))]
  k=float(d['kappa_ev_A2']);g=float(d['g_ev_A2'])
  close(r['C_pred_A2_eV'],k/g**2);close(r['n_infty'],(float(d['q_norm_ev_A2'])/g)**2)
 for r in rows(P/'tables/fig2_all_256_events.csv'):
  a=eventmap[(r['anchor'],int(r['direction_index']),int(r['budget_index']))]
  close(a['C_measured_A2_eV'],2*float(r['observed_W_meV'])/1000/float(r['epsilon_ev_A'])**2)
 ec=rows(D/'ecdf.csv')
 for anchor in ['A','B']:
  for kind,b in [('predicted',0),('measured',0),('measured',1)]:
   selected=[r for r in events if r['anchor']==anchor and int(r['budget'])==b]
   x=np.sort([float(r['C_pred_A2_eV'] if kind=='predicted' else r['C_measured_A2_eV']) for r in selected])
   plotted=[r for r in ec if r['anchor']==anchor and r['kind']==kind and int(r['budget'])==b]
   assert len(plotted)==len(x)==64
   close([r['normalized_work_A2_eV'] for r in plotted],x)
   close([r['F'] for r in plotted],np.arange(1,65)/64)
 for r in rows(D/'size_mechanism.csv'):
  z=np.load(P/'silicon/size_series'/('Si'+r['N']+'_'+r['geometry'])/'directions.npz',allow_pickle=False)
  h=z['H_delta_u'];u=z['u'];s2=np.sum(h*h,axis=(1,2));g2=np.max(np.sum(h*h,axis=2),axis=1)
  k=np.sum(u*h,axis=(1,2));chi=k/s2;c=k/g2
  close(r['chi_mean'],chi.mean());close(r['C_mean'],c.mean())
  close(r['n_infty_mean'],np.mean(s2/g2));close(r['n2_mean'],np.mean(s2*s2/np.sum(np.sum(h*h,axis=2)**2,axis=1)))
  close(r['C_CV'],c.std(ddof=1)/abs(c.mean()));close(r['chi_CV'],chi.std(ddof=1)/abs(chi.mean()))
 for r in rows(P/'tables/silicon_size_implications.csv'):
  if r['geometry']=='ideal':
   t=next(t for t in rows(D/'block_curve.csv') if t['N']==r['N'])
   close(t['m_N'],r['block_mean_n_infty'])
 data=rows(D/'formal_strips.csv');metrics=rows(D/'formal_metrics.csv');curves=json.loads((P/'longtime/curves.json').read_text())['trajectories']
 assert len(data)==1160
 for reference in ['MACE','PBE']:
  for r in [1,2]:
   z=np.load(P/'longtime'/f'{reference.lower()}_configuration_{r}.npz',allow_pickle=False)
   ss=[s for s in data if s['reference']==reference and int(s['configuration'])==r];n=len(ss)
   assert n==(500 if reference=='MACE' else 80)
   assert [int(s['segment']) for s in ss]==list(range(n))
   d=z['end_positions_A']-z['center_positions_A']
   w=z['end_reference_energy_eV']-z['center_reference_energy_eV']-z['end_base_energy_eV']+z['center_base_energy_eV']+np.sum(z['c_eV_A']*d,axis=(1,2))
   pred=.5*z['velocity_norm_A_per_fs']**2*z['kappa_eV_A2']*z['duration_fs']**2
   S=np.abs(w).sum();cs=np.cumsum(pred-w)
   close([s['W_meV'] for s in ss],1000*w);close([s['W_pred_meV'] for s in ss],1000*pred)
   close([s['kappa_eV_A2'] for s in ss],z['kappa_eV_A2'])
   close([s['speed_A_fs'] for s in ss],z['velocity_norm_A_per_fs'])
   delta=100*(np.array([float(s['W_pred_meV']) for s in ss])/1000-w)/np.abs(w)
   close(delta,100*(pred-w)/np.abs(w))
   speed2=np.array([float(s['speed_A_fs'])**2 for s in ss]);kappa=np.array([float(s['kappa_eV_A2']) for s in ss])
   close((speed2/speed2.mean())*(kappa/kappa.mean()),2*pred/(z['duration_fs']**2*speed2.mean()*kappa.mean()))
   close([s['signed_cumulative_error_percent'] for s in ss],100*cs/S)
   curve=next(c for c in curves if c['reference']==reference and c['replica']==r-1)
   close([s['empirical_scale_percent'] for s in ss],100*np.cumsum(curve['eta_empirical_eV'])/S)
   m=next(m for m in metrics if m['reference']==reference and int(m['configuration'])==r)
   close(m['Eseg_percent'],100*np.abs(pred-w).sum()/S)
   close(m['Ecum_percent'],100*np.max(np.abs(cs))/S)
   close(m['max_segment_error_percent'],100*np.max(np.abs(pred-w)/np.abs(w)))
 # Original Fig. 1 structure, distinct from the B1 atomic-work endpoint.
 anchor=json.loads((D/'anchor_A.json').read_text())
 z=np.load(P/'member_ensemble/A/anchor.npz',allow_pickle=False)
 close(anchor['positions_angstrom'],z['positions']);close(anchor['cell_angstrom'],z['cell'])
 numbers={'H':1,'Li':3,'C':6,'O':8,'F':9,'P':15}
 assert [numbers[s] for s in anchor['symbols']]==list(z['numbers'])
 assert all(0<=i<474 and 0<=j<474 and i!=j for i,j in anchor['bonds'])
 # Correlations count each of the 128 directions once, irrespective of budget.
 correlations=json.loads((D/'rank_correlations.json').read_text())['results']
 for r in correlations:
  subset=[d for (a,i),d in directions.items() if a==r['anchor'] or r['anchor']=='pooled']
  assert len(subset)==r['n_directions']
  k=np.array([float(d['kappa_ev_A2']) for d in subset]);g=np.array([float(d['g_ev_A2']) for d in subset]);s2=np.array([float(d['q_norm_ev_A2'])**2 for d in subset])
  close(r['rho_absC_n_infty'],rho(abs(k/g**2),s2/g**2))
  close(r['rho_absC_abschi'],rho(abs(k/g**2),abs(k/s2)))
 curve=rows(D/'block_curve.csv')
 assert [int(c['N']) for c in curve]==sorted(set(int(c['N']) for c in curve))
 assert int(curve[-1]['N'])==1250
 for n in [64,216,512,1000,1250]:
  t=next(c for c in curve if int(c['N'])==n)
  close(t['m_N'],block_integral(n))
 # Velocity curves preserve the original numerical rows.
 assert rows(D/'velocity_rotation.csv')==rows(P/'longtime/dynamics.csv')
 for name in ['a_atoms','a_bonds','a_cell','a_projection','acd_frozen_coefficients','c_frozen_curves','c_pbe_endpoints','d_work_errors']:
  assert rows(D/('silicon_'+name+'.csv'))==rows(P/'tables'/('silicon_figure_'+name+'.csv'))
 assert rows(D/'atomic_groups.csv')==rows(P/'atomic_work/figure_b_groups.csv')
 # Direct factor-plane verification: 128 unique interface directions and 32 Si directions.
 factors=rows(D/'factor_plane.csv')
 assert len(factors)==160
 for r in factors:
  j=int(r['direction'])
  if r['system']=='interface':
   v=directions[(r['anchor'],j)]
   k=float(v['kappa_ev_A2']);s2=float(v['q_norm_ev_A2'])**2;g2=float(v['g_ev_A2'])**2
  else:
   assert r['system']=='silicon1000' and r['anchor']=='ideal'
   v=np.load(P/'silicon/size_series/Si1000_ideal/directions.npz',allow_pickle=False)
   h=v['H_delta_u'][j];u=v['u'][j]
   k=np.sum(u*h);s2=np.sum(h*h);g2=np.max(np.sum(h*h,axis=1))
  close(r['kappa_eV_A2'],k);close(r['chi_A2_eV'],k/s2);close(r['n_infty'],s2/g2);close(r['C_A2_eV'],k/g2)
 positives=[r for r in factors if float(r['C_A2_eV'])>0]
 assert len(positives)==1 and positives[0]['anchor']=='A' and int(positives[0]['direction'])==58
 interval_status={'present':False,'formal_completion':False}
 if (ROOT/'formal_interval/INTERVAL_RESULT.json').exists():
  from check_interval_export import check
  interval_status=check(ROOT/'formal_interval')
  for name in ['interval_metrics.csv','interval_formal_segments.csv']:
   assert rows(D/name)==rows(ROOT/'formal_interval'/name),'Plotted interval data differ from accepted export'
 print(json.dumps(dict(factor_plane_directions=160,passed=True,atomic_values=474,paired_budget_events=256,ecdf_values=384,size_conditions=8,
   formal_segments=1160,rank_correlation_samples=[64,64,128],original_structure_atoms=474,finite_N_extent=1250,speed_curvature_and_segment_errors=True,no_model_calls=True,formal_interval_series=interval_status),indent=2))
if __name__=='__main__':main()
