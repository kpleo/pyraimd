import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
spec=importlib.util.spec_from_file_location('review',str(Path(__file__).with_name('pyramid_core_review.py')))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
if len(sys.argv)>1:
    r,_,_=b.resume(Path(sys.argv[1]));r.run(int(sys.argv[2]))
    state={'positions':r.atoms.positions.tolist(),'momenta':r.atoms.get_momenta().tolist(),'draws':b.draws(Path(sys.argv[1]))}
    r.close();print(json.dumps(state));raise SystemExit

def child(path,steps):
    result=subprocess.run(['uv','run','--no-sync','python',__file__,str(path),str(steps)],check=True,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    return json.loads(result.stdout)
# Baseline: complete-step resume + unchanged cache reads.
r,m,e,u,p=b.world('clean');r.run(2);r._write_checkpoint()
before=r.calc.n_evaluations
for _ in range(3):r.atoms.get_potential_energy();r.atoms.get_forces()
assert before==r.calc.n_evaluations
r.close();restored=child(p,2)
c,*_,cp=b.world('clean-control');c.run(4)
assert np.array_equal(restored['positions'],c.atoms.positions)
assert np.array_equal(restored['momenta'],c.atoms.get_momenta())
assert restored['draws']==b.draws(cp)
row=b.Store(cp/'trajectory.db')._row_at_step('r',0)
assert row.data['metadata']['checked']
assert not np.array_equal(row.data['driving']['forces'],row.data['engine']['forces'])
print(json.dumps({'case':'cross_process_complete_boundary','position_momentum_bitwise_equal':True,'draws_equal':True,'repeated_properties_no_new_evaluation':True,'checked_driving_not_replaced':True}))
c.close()
# Uncommitted proposal window, in a genuinely separate Python process.
r,m,e,u,p=b.world('pending');r.run(0);e.fail_next=True
try:r.run(1)
except b.EngineError:pass
r.close();restored=child(p,3)
print(json.dumps({'case':'cross_process_pending_rng','draws':restored['draws'],'path':str(p)}))
assert restored['draws'][:2]==[.2616121342493164]*2
