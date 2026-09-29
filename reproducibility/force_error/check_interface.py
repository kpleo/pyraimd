"""Reconstruct every saved interface quadrature from force labels."""
from pathlib import Path
import csv,json
import numpy as np

ROOT=Path(__file__).resolve().parent

def close(a,b):
    np.testing.assert_allclose(a,b,rtol=2e-10,atol=1e-10)

def main():
    data=ROOT/'interface'
    paths={i:dict(np.load(data/f'labels_{i}.npz',allow_pickle=False)) for i in range(4)}
    q=dict(np.load(data/'derived/quadrature.npz',allow_pickle=False))
    for j,i in enumerate(q['path_index']):
        z=paths[int(i)]; end=float(q['end_fs'][j]);dt=float(q['spacing_fs'][j])
        times=np.arange(round(end/dt)+1)*dt
        ix=[]
        for t in times:
            hit=np.flatnonzero(np.isclose(z['time_fs'],t,rtol=0,atol=1e-10))
            assert len(hit)==1
            ix.append(int(hit[0]))
        x=z['positions_A'][ix];c=z['reference_forces_eV_A'][0]-z['base_forces_eV_A'][0]
        residual=z['base_forces_eV_A'][ix]+c-z['reference_forces_eV_A'][ix]
        atomic=(.5*(residual[1:]+residual[:-1])*np.diff(x,axis=0)).sum(axis=(0,2))
        work=z['reference_energy_eV'][ix[-1]]-z['reference_energy_eV'][0]-(z['base_energy_eV'][ix[-1]]-z['base_energy_eV'][0])+np.sum(c*(x[-1]-x[0]))
        close(atomic,q['atomic_force_integral_eV'][j]);close(atomic.sum(),q['force_integral_eV'][j]);close(work,q['endpoint_work_eV'][j])
    # B1, 1 fs, 0.125-fs quadrature: every atomic contribution is retained.
    ix=np.flatnonzero((q['path_index']==2)&np.isclose(q['end_fs'],1)&np.isclose(q['spacing_fs'],.125))
    assert len(ix)==1
    rows=list(csv.DictReader((ROOT/'atomic_work/figure_a_atoms.csv').open()))
    close(1000*q['atomic_force_integral_eV'][ix[0]],[float(r['integrated_Q_meV']) for r in rows])
    print(json.dumps({'passed':True,'paths':4,'reconstructed_quadratures':len(q['path_index']),'atomic_contributions':474}))

if __name__=='__main__':main()
