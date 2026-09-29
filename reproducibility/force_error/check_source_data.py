"""Reconstruct physical quantities from the accompanying saved numerical labels."""
from pathlib import Path
import csv
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent

def rows(name):
    with (ROOT/'tables'/name).open() as f:
        return list(csv.DictReader(f))

def maxnorm(x):
    return np.linalg.norm(x, axis=-1).max()

def check_atomic_work():
    def read(name):
        with (ROOT/'atomic_work'/name).open() as stream:
            return list(csv.DictReader(stream))
    totals = {row['endpoint']: row for row in read('figure_totals.csv')}
    for endpoint, filename in [('B1', 'figure_a_atoms.csv'), ('A2', 'figure_c_atomic_work.csv')]:
        atoms = read(filename)
        assert len(atoms) == 474
        assert sorted(int(row['atom_index_zero_based']) for row in atoms) == list(range(474))
        q = np.array([float(row['integrated_Q_meV']) for row in atoms])
        residual = np.array([[float(row[f'residual_{axis}_eV_per_A']) for axis in 'xyz'] for row in atoms])
        assert np.allclose(np.linalg.norm(residual, axis=1),
                           [float(row['residual_norm_eV_per_A']) for row in atoms], rtol=0, atol=1e-12)
        total = totals[endpoint]
        for field, value in [('net_Q_meV', q.sum()), ('positive_Q_meV', q[q>0].sum()), ('negative_Q_meV', q[q<0].sum())]:
            assert abs(value-float(total[field])) < 1e-10
        if endpoint == 'B1':
            assert int(np.argmax(np.linalg.norm(residual, axis=1))) == 157
            assert int(np.count_nonzero(q>q[157]))+1 == 17
            for group in read('figure_b_groups.csv'):
                values = q[[row['initial_group'] == group['initial_group'] for row in atoms]]
                assert len(values) == int(group['atom_count'])
                for field, value in [('net_Q_meV', values.sum()), ('positive_Q_meV', values[values>0].sum()), ('negative_Q_meV', values[values<0].sum())]:
                    assert abs(value-float(group[field])) < 1e-10
        else:
            assert q[111] < 0 < q.sum()
        print(f'{endpoint}: {len(atoms)} atoms; net path integral={q.sum():.9f} meV')
    print('PASS: atomic residuals, signed work sums, group contributions and reported ranks')

def main():
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    for row in rows('material_endpoints.csv'):
        with np.load(ROOT/'material_endpoints'/f"{row['path']}.npz", allow_pickle=False) as z:
            c=z['correction']; d=z['positions']-z['anchor_positions']
            e=maxnorm(z['base_forces']+c-z['F_reference_ev_A'])
            W=(z['E_reference_ev']-z['E_reference_anchor_ev'])-(z['E_base_ev']-z['E_base_anchor_ev'])+np.sum(c*d)
            assert abs(W-float(row['W_ev']))<1e-9
            assert abs(e-float(row['e_ev_A']))<1e-12
    directions=rows('directions.csv'); events=rows('events.csv')
    assert len(directions)==128 and len(events)==256
    coeff={}
    for r in directions:
        key=(r['anchor'],int(r['index']))
        path=ROOT/'member_ensemble'/key[0]/f'd{key[1]:03d}'
        with np.load(path/'direction.npz') as v, np.load(path/'coefficients.npz') as h:
            q=h['q']; kappa=np.sum(v['u']*q); g=maxnorm(q); C=kappa/g**2
            assert abs(C-float(r['C_A2_per_ev']))<1e-10
            coeff[key]=C
    widths={}
    for r in events:
        key=(r['anchor'],int(r['index'])); b=int(r['budget_index'])
        path=ROOT/'member_ensemble'/key[0]
        with np.load(path/'anchor.npz') as a, np.load(path/f'd{key[1]:03d}'/f'budget{b}.npz') as z:
            c=a['F_ref']-a['F_base']; d=z['positions'][0]-a['positions']
            e=maxnorm(z['F_base'][0]+c-z['F_ref'][0])
            W=(z['E_ref'][0]-a['E_ref'])-(z['E_base'][0]-a['E_base'])+np.sum(c*d)
            assert abs(W-float(r['W_ev']))<1e-10
            assert abs(e-float(r['e_ev_A']))<1e-12
        pred=float(r['epsilon_ev_A'])**2*coeff[key]/2
        assert abs(pred-float(r['W_pred_target_ev']))<1e-12
        widths.setdefault((key[0],b),[]).append((float(W),pred))
    for (anchor,b), values in sorted(widths.items()):
        data=np.array(values)*1000
        width=np.diff(np.quantile(data,[.1,.9],axis=0),axis=0)[0]
        print(f'{anchor}, budget {b}: measured width={width[0]:.6f} meV, predicted={width[1]:.6f} meV')
    maximum=0.
    for path in sorted((ROOT/'derivative_labels').glob('*.npz')):
        with np.load(path) as z:
            hs=z['h_values']; offsets=z['displaced_positions']-z['anchor_positions']
            calculated=[]
            for j,h in enumerate(hs):
                # Identify signs from the displacement, without assuming storage order.
                projection=np.einsum('sij,ij->s',offsets[j],z['u'])
                minus,plus=np.argsort(projection)
                assert abs(projection[plus]-h)<1e-10 and abs(projection[minus]+h)<1e-10
                residual=z['F_base'][j]-z['F_reference'][j]
                calculated.append((residual[plus]-residual[minus])/(2*h))
            fine=int(np.argmin(hs)); error=maxnorm(calculated[fine]-z['q_hvp_saved'])
            maximum=max(maximum,float(error)); tolerance=max(1e-6,.01*maxnorm(z['q_hvp_saved']))
            assert error<=tolerance
            assert maxnorm(calculated[0]-calculated[1])<=tolerance
    print(f'PASS: 4 DFT endpoints, 128 directions, 256 events, 16 derivative checks; max fine FD-HVP difference={maximum:.9g} eV/angstrom^2')
    check_atomic_work()

if __name__=='__main__':
    main()
