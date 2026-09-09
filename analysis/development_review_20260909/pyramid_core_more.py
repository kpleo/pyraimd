from pathlib import Path
import json
import importlib.util
import sys
spec=importlib.util.spec_from_file_location('review',str(Path(__file__).with_name('pyramid_core_review.py')))
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

def artifact_case():
    r,m,e,u,p=b.world('artifact-corrupt',update_n=2,p=1.)
    r.run(0);r.run(1);r.close()
    update=[e for e in b.events(p) if e['type']=='model_update'][0]
    artifact=p/'models'/update['model_id']/'state.json'
    data=json.loads(artifact.read_text())
    before=data['updater_state']['surrogate']['k']
    data['updater_state']['surrogate']['k']=9.9
    artifact.write_text(json.dumps(data))
    rr,mm,uu=b.resume(p,2)
    assert mm.k==9.9
    b.out('artifact_corrupt',recorded=before,loaded=mm.k,model_id=rr.calc.model_id,event_k=update['updater_state']['surrogate']['k'],path=str(p))
    rr.close()

def recalibration_case():
    r,m,e,u,p=b.world('tail-recalibration',update_n=2)
    r.run(1)
    original=r.calc._finish
    def crash(pending):
        if pending.index==2:raise RuntimeError('crash after new-model calibration and tail proposal')
        return original(pending)
    r.calc._finish=crash
    try:r.run(1)
    except RuntimeError:pass
    r.close()
    rr,mm,uu=b.resume(p,2)
    rr.run(1)
    after_two={'n_calibrations':rr.calc.n_calibrations,'segment':rr.calc._segment,'deferred_origin':None if rr.calc._deferred_origin is None else rr.calc._deferred_origin.index}
    rr.run(1)
    probes_resumed=len([ev for ev in b.events(p) if ev['type']=='task' and ev.get('operation')=='reference' and ev.get('purpose')=='probe'])
    r2,m2,e2,u2,p2=b.world('tail-control',update_n=2);r2.run(3)
    probes_control=len([ev for ev in b.events(p2) if ev['type']=='task' and ev.get('operation')=='reference' and ev.get('purpose')=='probe'])
    b.out('tail_recalibration',after_two=after_two,probe_executions_resumed=probes_resumed,probe_executions_control=probes_control,final_calibrations_resumed=rr.calc.n_calibrations,final_calibrations_control=r2.calc.n_calibrations,path=str(p))
    rr.close();r2.close()

def truncated_event_case():
    r,m,e,u,p=b.world('event-tail');r.run(1);r.close()
    with (p/'events.jsonl').open('ab') as f:f.write(b'{"seq": 30, "type": "task", "task_id":')
    try:b.resume(p)
    except Exception as exc:b.out('torn_event_tail',error=str(exc),has_valid_checkpoint=b.CheckpointManager(p).read_latest_valid() is not None,path=str(p))
    else:raise AssertionError('expected error')

def resumed_cache_training_case():
    r,m,e,u,p=b.world('cache-training',update_n=2,stationary=True,p=1.)
    r.run(0);r.close()
    rr,mm,uu=b.resume(p,2);rr.run(1)
    control,cm,ce,cu,cp=b.world('cache-training-control',update_n=2,stationary=True,p=1.);control.run(1)
    b.out('cache_lost_retraining',resumed_k=mm.k,continuous_k=cm.k,resumed_updates=uu.n_updates,continuous_updates=cu.n_updates,resumed_consumed=uu.n_consumed,continuous_consumed=cu.n_consumed,path=str(p))
    assert uu.n_updates==1 and cu.n_updates==0
    rr.close();control.close()

for name, fn in [('artifact',artifact_case),('recalibration',recalibration_case),('torn',truncated_event_case),('cachetrain',resumed_cache_training_case)]:
    if len(sys.argv)==1 or name in sys.argv[1:]:fn()
