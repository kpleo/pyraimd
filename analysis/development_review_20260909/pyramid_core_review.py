import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from ase import Atoms
from pyraimd2.engines.base import EngineResult, EngineError
from pyraimd2.surrogate.base import SurrogatePrediction, TrainReport
from pyraimd2.switch.base import LabelObservation
from pyraimd2.loop import EnergeticRunner, GuardedUpdater, UpdatePolicy
from pyraimd2.runtime.events import EventLog
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.store import Store

ROOT = Path(tempfile.mkdtemp(prefix='pyramid-core-review-', dir='/tmp'))
class Model:
    fingerprint = 'review-harmonic-v1'
    def __init__(self, k=.8):
        self.k = k
        self.calls = 0
    def predict(self, a):
        return SurrogatePrediction(.5*self.k*float((a.positions**2).sum()), -self.k*a.positions, None, np.full(len(a),np.nan))
    def finetune(self, labels):
        self.calls += 1
        items=list(labels)
        self.k += .01
        return TrainReport(len(items),1,1.,.5,(.5,),0.)
    def state_dict(self):
        return {'k':self.k,'calls':self.calls}
    def load_state_dict(self, s):
        self.k=s['k']; self.calls=s['calls']
class Reference:
    fingerprint = 'review-ref-v1'
    def __init__(self):
        self.fail_next=False
    def compute(self,a):
        if self.fail_next:
            self.fail_next=False
            raise EngineError('injected reference failure')
        return EngineResult(.6*float((a.positions**2).sum()), -1.2*a.positions,None,0.)
def direction(a):
    d=np.zeros((len(a),3));d[:,0]=1.;return d

def world(name, *, update_n=None, stationary=False, p=.5):
    path=ROOT/name;path.mkdir()
    a=Atoms('H',positions=[[0. if stationary else .2,0,0]])
    a.set_velocities([[0. if stationary else .1,0,0]])
    m=Model();e=Reference()
    u=None if update_n is None else GuardedUpdater(m,UpdatePolicy(n_label=update_n,guard_size=1))
    r=EnergeticRunner(a,m,e,Store(path/'trajectory.db'),'r',on_label=u,
        run_dir=path,event_log=EventLog(path),checkpoint_interval_steps=100,
        direction=direction,force_budget=.08,timestep_fs=.1,
        time_cap_fs=2.,check_probability=p,check_seed=2)
    return r,m,e,u,path

def resume(path, update_n=None, model=None):
    m=Model() if model is None else model
    u=None if update_n is None else GuardedUpdater(m,UpdatePolicy(n_label=update_n,guard_size=1))
    r=EnergeticRunner.resume(path,m,Reference(),updater=u,direction=direction)
    return r,m,u

def events(path):
    return [json.loads(l) for l in (path/'events.jsonl').read_text().splitlines()]
def draws(path):
    return [e['check_draw'] for e in events(path) if e['type']=='evaluation_proposed' and e['accepted']]
def out(name, **kw):
    print(json.dumps({'case':name,**kw},sort_keys=True))

def rng_case():
    r,m,e,u,p=world('rng-crash');r.run(0);e.fail_next=True
    try:r.run(1)
    except EngineError:pass
    r.close()
    tail=events(p)[-2:]
    rr,_,_=resume(p);rr.run(3);rr.close()
    c,*_,cp=world('rng-control');c.run(3);c.close()
    assert draws(p)[0]==draws(cp)[0] and draws(p)[1]!=draws(cp)[1]
    out('pending_rng',restored=draws(p),continuous=draws(cp),path=str(p))

def cache_case():
    r,m,e,u,p=world('cache-crash',update_n=100,stationary=True,p=1.)
    r.run(0);r.run(2)
    records=events(p);r.close()
    hits=[(e['evaluation_id'],e['label_id']) for e in records if e['type']=='task' and e.get('cache_hit')]
    try:rr,_,_=resume(p,100)
    except Exception as exc:out('cache_consumed',hits=hits,consumed=u.n_consumed,error=str(exc),path=str(p))
    else:rr.close();raise AssertionError('unexpected resume success')

def row_case():
    r,m,e,u,p=world('row-crash');r.run(0)
    orig=r.calc._event_log.append_once
    def hook(key,typ,payload):
        if typ=='evaluation_committed' and payload['context']['evaluation_id']==1:
            raise RuntimeError('crash after SQLite commit before event commit')
        return orig(key,typ,payload)
    r.calc._event_log.append_once=hook
    try:r.run(1)
    except RuntimeError:pass
    r.close()
    rr,_,_=resume(p);rr.run(1);rr.close()
    rows=list(Store(p/'trajectory.db')._db.select(run_id='r',step=0))
    commits=[e for e in events(p) if e['type']=='evaluation_committed' and e['context']['evaluation_id']==1]
    assert len(rows)==2 and len(commits)==1
    out('db_event_window',rows=[{'db_id':r.id,'label_id':r.data.get('engine_label_id')} for r in rows],event_label=commits[0]['label_id'],read_by_step_label=Store(p/'trajectory.db')._row_at_step('r',0).data['engine_label_id'],path=str(p))

def publish_case():
    r,m,e,u,p=world('publish-failure',update_n=2,p=1.);r.run(0)
    parent=m.state_dict();old_id=r.calc.model_id
    def fail(*args):raise OSError('injected model artifact write failure')
    r.calc._model_publisher=fail
    try:r.run(1)
    except OSError:pass
    data={'parent':parent,'after':m.state_dict(),'old_id':old_id,'after_id':r.calc.model_id,'updates':u.n_updates,'artifact_count':len(list((p/'models').glob('*/state.json'))),'model_update_events':len([e for e in events(p) if e['type']=='model_update'])}
    assert m.state_dict()!=parent and data['model_update_events']==0
    r.close()
    out('publish_failure',**data,path=str(p))

def validation_case():
    class Broken(Model):
        def predict(self,a):
            if self.calls:raise RuntimeError('candidate cannot infer')
            return super().predict(a)
    m=Broken();u=GuardedUpdater(m,UpdatePolicy(n_label=1,guard_size=1))
    a=Atoms('H',positions=[[.2,0,0]])
    obs=LabelObservation(0,a,m.predict(a),Reference().compute(a),label_id='L1')
    parent=m.state_dict()
    try:u(obs)
    except RuntimeError as exc:err=str(exc)
    else:raise AssertionError('expected failure')
    assert m.state_dict()!=parent
    out('validation_failure',parent=parent,after=m.state_dict(),error=err,n_rejected=u.n_rejected)

def cell_case():
    class Inspect(Model):
        def __init__(self):super().__init__();self.seen=[]
        def finetune(self, labels):
            labels=list(labels)
            self.seen=[{'cell':a.cell.array.tolist(),'pbc':a.pbc.tolist(),'charges':a.get_initial_charges().tolist(),'magmoms':a.get_initial_magnetic_moments().tolist()} for a,_ in labels]
            return super().finetune(labels)
    m=Inspect();u=GuardedUpdater(m,UpdatePolicy(n_label=1,guard_size=1))
    a=Atoms('H',positions=[[.2,0,0]],cell=[3,3,3],pbc=True)
    a.set_initial_charges([.2]);a.set_initial_magnetic_moments([1.])
    obs=LabelObservation(0,a,m.predict(a),Reference().compute(a),label_id='L1')
    accepted=u(obs)
    assert m.seen[0]['pbc']==[False]*3
    out('training_structure',published=accepted,seen=m.seen)

cases={'rng':rng_case,'cache':cache_case,'row':row_case,'publish':publish_case,'validation':validation_case,'cell':cell_case}
if __name__ == '__main__':
    for name in sys.argv[1:] or cases:
        cases[name]()
    print('ARTIFACT_ROOT',ROOT)
