"""Read-only 0.4.0 core rereview. All generated runs live under /tmp."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from ase import Atoms, units
from pyraimd2.engines.base import EngineError, EngineResult
from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.runtime import ResumeError
from pyraimd2.runtime.events import EventLog
from pyraimd2.runtime.models import (ModelRegistry, content_array_sink, content_array_source,
                                    dump_state_arrays, load_state_arrays)
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction, TrainReport
from pyraimd2.switch.base import LabelObservation
ROOT=Path(tempfile.mkdtemp(prefix='pyramid-040-core-',dir='/tmp'))
REPORT=TrainReport(1,1,1.,.5,(.5,),0.)
class Model:
    fingerprint='review-harmonic-v1'
    def __init__(self,k=.8): self.k=k; self.calls=0
    def predict(self,a):
        return SurrogatePrediction(.5*self.k*float((a.positions**2).sum()),-self.k*a.positions,None,np.full(len(a),np.nan))
    def finetune(self,labels): list(labels); self.k+=.01; self.calls+=1; return REPORT
    def state_dict(self): return {'k':self.k,'calls':self.calls}
    def load_state_dict(self,s): self.k=s['k']; self.calls=s['calls']
class Ref:
    def __init__(self,known=True,first_label_delta=0.):
        self.fingerprint='review-ref-v1' if known else None
        self.fail_next=False;self.delta=first_label_delta
    def compute(self,a):
        if self.fail_next: self.fail_next=False; raise EngineError('injected check failure')
        # Tiny different successful result on a re-execution, fault-injected
        # to distinguish orphan and commit-bound payloads, not a new workflow setting.
        k=1.2+self.delta;self.delta=0.
        return EngineResult(.5*k*float((a.positions**2).sum()),-k*a.positions,None,0.)
def direction(a):
    d=np.zeros((len(a),3));d[:,0]=1.;return d

def make(name,*,update_n=None,stationary=False,p=.5,known=True,cap=2.,model=None):
    path=ROOT/name;path.mkdir();a=Atoms('H',positions=[[0. if stationary else .2,0,0]])
    a.set_velocities([[0. if stationary else .1,0,0]])
    m=Model() if model is None else model;e=Ref(known)
    u=None if update_n is None else GuardedUpdater(m,UpdatePolicy(n_label=update_n,guard_size=1))
    r=EnergeticRunner(a,m,e,Store(path/'trajectory.db'),'r',on_label=u,run_dir=path,
        event_log=EventLog(path),checkpoint_interval_steps=100,direction=direction,
        force_budget=.08,timestep_fs=.1,time_cap_fs=cap,check_probability=p,check_seed=2)
    return r,m,e,u,path

def restore(path,n=None,known=True,delta=0.,model=None):
    m=Model() if model is None else model
    u=None if n is None else GuardedUpdater(m,UpdatePolicy(n_label=n,guard_size=1))
    r=EnergeticRunner.resume(path,m,Ref(known,delta),updater=u,direction=direction)
    return r,m,u

def events(path):return [json.loads(l) for l in (path/'events.jsonl').read_text().splitlines()]
def draws(path):return [e['check_draw'] for e in events(path) if e['type']=='evaluation_proposed' and e['accepted']]
def out(case,**kw):print(json.dumps({'case':case,**kw},sort_keys=True))
def fail_commit(r,eval_id):
    orig=r.calc._event_log.append_once
    def hook(key,typ,payload):
        if typ=='evaluation_committed' and payload['context']['evaluation_id']==eval_id:
            raise RuntimeError('injected DB/event window')
        return orig(key,typ,payload)
    r.calc._event_log.append_once=hook

def passes():
    r,m,e,u,p=make('rng');r.run(0);e.fail_next=True
    try:r.run(1)
    except EngineError:pass
    r.close();rr,_,_=restore(p);rr.run(3)
    c,*_,cp=make('rng-control');c.run(3)
    assert draws(p)==draws(cp)
    assert np.array_equal(rr.atoms.positions,c.atoms.positions)
    rr.close();c.close();out('PASS_F01',draws=draws(p))
    for window in (0,2):
        r,m,e,u,p=make('cache'+str(window),update_n=2,stationary=True,p=1.)
        r.run(0);r.run(window);r.close()
        rr,mm,uu=restore(p,2);rr.run(1)
        assert uu.n_consumed==1 and uu.n_updates==0 and mm.k==.8
        rr.close();out('PASS_F02_F07',window=window,consumed=uu.n_consumed,updates=uu.n_updates)
    r,m,e,u,p=make('artifact',update_n=2,p=1.);r.run(1);r.close()
    up=next(e for e in events(p) if e['type']=='model_update')
    f=p/'models'/up['model_id']/'state.json';data=json.loads(f.read_text());data['updater_state']['surrogate']['k']=9.9
    f.write_text(json.dumps(data))
    try:restore(p,2)
    except ResumeError as ex:out('PASS_F04',error=str(ex))
    else:raise AssertionError('tampered artifact accepted')
    class Structure(Model):
        def __init__(self):super().__init__();self.seen=[]
        def predict(self,a):
            self.seen.append((a.cell.volume,a.pbc.tolist(),a.get_initial_charges().tolist(),a.get_initial_magnetic_moments().tolist()))
            return super().predict(a)
        def finetune(self,labels):
            items=list(labels)
            for a,l in items:self.predict(a)
            return super().finetune(items)
    m=Structure();u=GuardedUpdater(m,UpdatePolicy(n_label=1,guard_size=1));a=Atoms('H',positions=[[.2,0,0]],cell=[3,3,3],pbc=True)
    a.set_initial_charges([.2]);a.set_initial_magnetic_moments([1.])
    assert u(LabelObservation(0,a,m.predict(a),Ref().compute(a),label_id='L'))
    assert all(np.isclose(s[0],27.) and s[1:]==([True]*3,[.2],[1.]) for s in m.seen)
    out('PASS_F05',guard_and_training_preserve_structure=True)
    class Pair(Model):
        def __init__(self):super().__init__();self.k=1.
        def predict(self,a):
            d=a.positions[1]-a.positions[0]
            return SurrogatePrediction(.5*float(d@d),np.array([d,-d])*self.k,None,np.full(2,np.nan))
        def finetune(self,labels):list(labels);self.k=1.1;return REPORT
    m=Pair();u=GuardedUpdater(m,UpdatePolicy(n_label=1,guard_size=1));a=Atoms('H2',positions=[[0,0,0],[1,0,0]])
    ref=EngineResult(.6,np.array([[1.2,0,0],[-1.2,0,0]]),None,0.)
    assert not u(LabelObservation(0,a,m.predict(a),ref,label_id='L'))
    assert m.k==1. and u.rejections[-1]['reason']=='energy_force_inconsistent'
    out('PASS_F06',rejection=u.rejections[-1]['reason'])
    class Broken(Model):
        def predict(self,a):
            if self.calls:raise RuntimeError('candidate inference failed')
            return super().predict(a)
    m=Broken();u=GuardedUpdater(m,UpdatePolicy(n_label=1,guard_size=1));a=Atoms('H',positions=[[.2,0,0]])
    assert not u(LabelObservation(0,a,m.predict(a),Ref().compute(a),label_id='L'))
    assert m.k==.8 and u.rejections[-1]['reason']=='validation_failed'
    r,m,e,u,p=make('save-failure',update_n=2,p=1.);r.run(0)
    def fail(*a):raise OSError('injected publisher failure')
    r.calc._model_publisher=fail
    try:r.run(1)
    except OSError:pass
    assert m.k==.8 and u.n_updates==0 and r.calc.model_generation==0
    r.close();out('PASS_F08_NARROW',validation_and_publisher_failure_rollback=True)
    reg=ModelRegistry(ROOT/'array-artifact');original={'weights':np.arange(6.).reshape(2,3)}
    reg.publish('m',{'updater_state':original});loaded=reg.read('m',resolve=True)['updater_state']
    assert np.array_equal(loaded['weights'],original['weights'])
    out('PASS_F09_NARROW',ordinary_array_artifact_roundtrip=True)

def orphan_force():
    r,m,e,u,p=make('orphan-force',known=False,p=0.,cap=.001);r.run(0);fail_commit(r,1)
    try:r.run(1)
    except RuntimeError:pass
    r.close()
    rr,_,_=restore(p,known=False,delta=.001);rr.run(1)
    bound=rr.calc.store.committed_row(rr.calc._event_log,'r',1)
    old=rr.calc.store._row_at_step('r',0)
    actual=rr.atoms.get_momenta().copy();force=np.array(bound.data['driving']['forces'])
    assert not np.array_equal(force,old.data['driving']['forces'])
    rr._write_checkpoint();rr.close()
    again,_,_=restore(p,known=False)
    error=float(np.max(np.abs(again.atoms.get_momenta()-actual)))
    assert error>0 and np.array_equal(again.calc.results['forces'],old.data['driving']['forces'])
    out('RESIDUAL_COMMIT_FORCE',committed_row_id=bound.id,orphan_row_id=old.id,committed_force=force.tolist(),restored_force=again.calc.results['forces'].tolist(),full_momentum_error=error,path=str(p))
    again.close()

def tensor_identity():
    data={'matrix':np.zeros((2,2)),'vector':np.zeros(4),'float':np.array([1.],dtype=np.float64),'integer':np.array([4607182418800017408],dtype=np.uint64)}
    path=ROOT/'array-collision';encoded=dump_state_arrays(data,content_array_sink(path))
    loaded=load_state_arrays(encoded,content_array_source(path))
    assert loaded['vector'].shape!=(4,) and loaded['integer'].dtype!=data['integer'].dtype
    reg=ModelRegistry(ROOT/'scalar-artifact');reg.publish('scalar',{'updater_state':{'scalar':np.array(2.)}})
    scalar=reg.read('scalar',resolve=True)['updater_state']['scalar']
    assert scalar.shape!=()
    out('RESIDUAL_ARRAY_IDENTITY',vector_original_shape=list(data['vector'].shape),vector_loaded_shape=list(loaded['vector'].shape),integer_original_dtype=str(data['integer'].dtype),integer_loaded_dtype=str(loaded['integer'].dtype),integer_loaded_value=loaded['integer'].tolist(),scalar_original_shape=[],scalar_loaded_shape=list(scalar.shape),path=str(path))

def post_training_log():
    r,m,e,u,p=make('post-training-log',update_n=2,p=1.);r.run(0)
    orig=r.calc._event_log.append
    def hook(typ,payload):
        if typ=='task' and payload.get('operation')=='training' and payload.get('status')=='success':
            raise OSError('injected post-training log write failure')
        return orig(typ,payload)
    r.calc._event_log.append=hook
    try:r.run(1)
    except OSError:pass
    assert m.k==.81 and u.n_updates==1 and r.calc.model_generation==0
    out('RESIDUAL_POST_TRAINING_LOG',model_k=m.k,updates=u.n_updates,generation=r.calc.model_generation,model_update_events=sum(e['type']=='model_update' for e in events(p)),path=str(p))
    r.close()

def tail_calibration():
    r,m,e,u,p=make('tail-calibration',update_n=1,p=1.);r.run(1)
    orig=r.calc._finish
    def hook(pending):
        if pending.index==2:raise RuntimeError('injected after calibration and proposal')
        return orig(pending)
    r.calc._finish=hook
    try:r.run(1)
    except RuntimeError:pass
    r.close();rr,mm,uu=restore(p,1);rr.run(2)
    c,cm,ce,cu,cp=make('tail-control',update_n=1,p=1.);c.run(3)
    assert np.array_equal(rr.atoms.positions,c.atoms.positions)
    assert rr.calc.n_calibrations!=c.calc.n_calibrations
    out('RESIDUAL_TAIL_CALIBRATION',restored_calibrations=rr.calc.n_calibrations,control_calibrations=c.calc.n_calibrations,restored_segment=rr.calc._segment,control_segment=c.calc._segment,restored_generation=rr.calc.model_generation,control_generation=c.calc.model_generation,physical_positions_equal=True,path=str(p))
    rr.close();c.close()

if __name__=='__main__':
    cases={'passes':passes,'orphan':orphan_force,'arrays':tensor_identity,'log':post_training_log,'tail':tail_calibration}
    for name in sys.argv[1:] or cases:cases[name]()
    print('ARTIFACT_ROOT',ROOT)
