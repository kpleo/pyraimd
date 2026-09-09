import json,tempfile
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize.optimize import OptimizableAtoms
from pyraimd2.config import load_config
from pyraimd2.workflows import run_workflow,resume_workflow
from pyraimd2.store import Store
from pyraimd2.runtime.checkpoint import CheckpointManager
R=Path(tempfile.mkdtemp(prefix='pyramid-review-workflows-',dir='/tmp'))
def out(case,**kw): print(json.dumps({'case':case,**kw},ensure_ascii=False,default=str),flush=True)
def config(name,kind='md',mode='reference',pos=None,fixed=None):
    d=R/name;d.mkdir()
    a=Atoms('H2',positions=pos if pos is not None else [[.3,0,0],[.5,0,0]])
    a.set_momenta(np.zeros((2,3)))
    write(d/'structure.extxyz',a)
    s=f'''schema_version = 1
[run]
id = "review"
directory = "run"
[task]
kind = "{kind}"
mode = "{mode}"
[structure]
file = "structure.extxyz"
[dynamics]
steps = 2
timestep_fs = 0.5
temperature_K = 0.0
[{mode}]
backend = "harmonic-{mode}"
k = 1.0
r0 = 0.0
'''+('bias = 0.0\n' if mode=='surrogate' else '')+'''
[relax]
optimizer = "fire"
fmax_eV_A = 0.05
steps = 2
[checkpoint]
interval_steps = 1
'''
    if fixed is not None:s+='\n[constraints]\nfix_atoms_indices = '+repr(fixed)+'\n'
    p=d/'run.toml';p.write_text(s);return load_config(p)
def events(c):return [json.loads(x) for x in (c.run.directory/'events.jsonl').read_text().splitlines()]
out('environment',artifacts=R)
for kind in ['singlepoint','relax']:
    c=config(kind,kind=kind,pos=[[1,0,0],[.02,0,0]],fixed=[0])
    w=run_workflow(c,verbose=False,handle_sigint=False)
    rows=list(Store(w.run_dir/'trajectory.db')._db.select(run_id='review'))
    row=rows[-1];forces=np.asarray(row.data['driving']['forces']); projected=forces.copy();projected[0]=0
    out('fixed_'+kind,stopped_early=w.stopped_early,steps=w.steps_completed,stored_driving_fixed_force=forces[0].tolist(),max_raw=np.linalg.norm(forces,axis=1).max(),max_projected=np.linalg.norm(projected,axis=1).max(),constraint_metadata=row.data['metadata'].get('constraint'),summary=[x for x in events(c) if x['type']=='run_summary'][-1],rows=len(rows))
a=Atoms('H');opt=OptimizableAtoms(a);g=np.array([.04,.04,0.])
out('ase_convergence_metric',maximum_component=float(abs(g).max()),maximum_atom_norm=float(opt.gradient_norm(g)),converged_at_005=bool(opt.converged(g,.05)))
for mode in ['reference','surrogate']:
    c=config('stationary-'+mode,mode=mode,pos=[[0,0,0],[0,0,0]])
    run_workflow(c,verbose=False,handle_sigint=False)
    try:
        w=resume_workflow(c.run.directory,1,verbose=False,handle_sigint=False)
        out('stationary_resume',mode=mode,status='success',steps=w.steps_completed)
    except Exception as x: out('stationary_resume',mode=mode,status='FAIL',error_type=type(x).__name__,error=str(x))
# Only tmp files are altered. The original run's configuration/backend physics
# changes; the checkpoint must reject this regardless of reference/surrogate.
for mode in ['reference','surrogate']:
    c=config('identity-'+mode,mode=mode)
    run_workflow(c,verbose=False,handle_sigint=False)
    checkpoint=CheckpointManager(c.run.directory).read_latest_valid()
    p=c.run.directory/'resolved_config.json';data=json.loads(p.read_text());data[mode]['options']['k']=2.0;p.write_text(json.dumps(data))
    try:
        w=resume_workflow(c.run.directory,1,verbose=False,handle_sigint=False)
        rows=list(Store(w.run_dir/'trajectory.db')._db.select(run_id='review'));last=rows[-1]
        out('changed_backend_resume',mode=mode,status='ACCEPTED',steps=w.steps_completed,checkpoint_model=checkpoint.state['model_id'],new_model=last.data['metadata']['context']['model_id'])
    except Exception as x:out('changed_backend_resume',mode=mode,status='REJECTED',error_type=type(x).__name__,error=str(x))
