"""Tiny offline checks of the 0.4.0 complete-step export integration."""
import json, sys, tempfile, contextlib, io, re
from pathlib import Path
import numpy as np
from ase.io import read
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'tests/unit'))
from test_review_r1 import world, resume_world, events
from pyraimd2.runtime.events import EVALUATION_COMMITTED, STEP_COMPLETED
from pyraimd2.runtime.inspect import inspect_run
from pyraimd2.store import Store
from pyraimd2.workflows import export_run, run_workflow
from pyraimd2.workflows.export import frames_from_store, completed_step_ids
from pyraimd2.config import load_config
from pyraimd2.workflows.templates import HARMONIC_CONFIG,HARMONIC_STRUCTURE
D=Path(tempfile.mkdtemp(prefix='pyramid-040-export-'))
results={}
# Initial evaluation is a full initial state, not a half-step force evaluation.
r,*_=world(D/'initial');initial=r.atoms.get_momenta().copy();r.run(2);r.close()
s=Store(D/'initial/trajectory.db')
frames=frames_from_store(s,'run',force_source='driving',complete_steps=completed_step_ids(D/'initial'))
f=frames[0];raw=s._row_at_step('run',-1).toatoms()
results['initial_frame']={'initial_p':initial.tolist(),'raw_initial_p':raw.get_momenta().tolist(),'exported_initial_p':f.get_momenta().tolist(),'max_error':float(abs(initial-f.get_momenta()).max()),'phase':f.info.get('integration_phase'),'momenta_source':f.info.get('momenta_source')}
# Singlepoint and relaxation have no MD STEP_COMPLETED event.
for kind in ('singlepoint','relax'):
 d=D/kind;d.mkdir();(d/'structure.extxyz').write_text(HARMONIC_STRUCTURE)
 t=HARMONIC_CONFIG.replace('kind = "md"',f'kind = "{kind}"').replace('mode = "adaptive"','mode = "reference"').replace('steps = 20','steps = 2')
 t=re.sub(r'\[surrogate\][\s\S]*?(?=\[checkpoint\])','',t)
 (d/'run.toml').write_text(t)
 with contextlib.redirect_stdout(io.StringIO()): w=run_workflow(load_config(d/'run.toml'),verbose=False,handle_sigint=False)
 st=Store(w.run_dir/'trajectory.db');rows=list(st._db.select(run_id='harmonic-demo'))
 try: out=export_run(w.run_dir);result={'exported':out['frames']}
 except Exception as exc:result={'error':type(exc).__name__+': '+str(exc)}
 result.update({'stored_rows':len(rows),'steps':[int(x.key_value_pairs['step']) for x in rows],'run_dir':str(w.run_dir)})
 results[kind]=result
# An orphan and committed replacement may share a completed step number.
r,*_=world(D/'orphan',fingerprinted=False);r.run(0);log=r.calc._event_log;orig=log.append_once
def fail_commit(key,typ,payload):
 if typ==EVALUATION_COMMITTED and payload['context']['evaluation_id']==1:raise RuntimeError('injected DB-before-event failure')
 return orig(key,typ,payload)
log.append_once=fail_commit
try:r.run(1)
except RuntimeError:pass
r.close();rr,*_=resume_world(D/'orphan',fingerprinted=False);rr.run(1);rr.close()
s=Store(D/'orphan/trajectory.db');fs=frames_from_store(s,'run',force_source='reference',complete_steps=completed_step_ids(D/'orphan'))
cs=[e for e in events(D/'orphan') if e['type']==EVALUATION_COMMITTED]
results['orphan_export']={'committed_row_ids':[e.get('row_id') for e in cs],'committed_evaluations':len(cs),'exported_frames':len(fs),'exported_evaluation_ids':[f.info.get('evaluation_id') for f in fs],'exported_label_ids':[f.info.get('reference_label_id',f.info.get('label_id')) for f in fs]}
# A finished evaluation with no committed complete step must not supply a
# misleading current trajectory time/temperature.
r,*_=world(D/'incomplete');r.run(0);orig=r.calc._emit_once
def fail_boundary(key,typ,**payload):
 if typ==STEP_COMPLETED:raise RuntimeError('injected before complete-step commit')
 return orig(key,typ,**payload)
r.calc._emit_once=fail_boundary
try:r.run(1)
except RuntimeError:pass
r.close();info=inspect_run(D/'incomplete');out=export_run(D/'incomplete');last=read(out['output'],index=-1)
results['incomplete_inspect']={'complete_steps':info['n_complete_steps'],'inspect_time_fs':info['physical_time_fs'],'inspect_last_step':info['trajectory']['last_step'],'inspect_temperature':info['trajectory']['last_temperature_K'],'export_last_step':last.info.get('step_id'),'export_time_fs':last.info.get('physical_time_fs'),'export_temperature':float(last.get_temperature())}
print(json.dumps({'head':'9c7dc4b','tempdir':str(D),'results':results},indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))
