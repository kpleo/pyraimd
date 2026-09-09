"""Offline R5/R6 rereview probes. All outputs in /tmp; no source edits."""
import contextlib, io, json, os, sys, tempfile
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.mixing import SumCalculator
from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.engines.ase_qe import AseQeEngine
from pyraimd2.engines.qe_engine import QeConfig, QeEngine
from pyraimd2.runtime.events import EventLog
from pyraimd2.runtime.costs import summarize_tasks

FIXTURE = Path(__file__).resolve().parents[2] / 'tests/data/qe_si_scf.out'
FAKE = r'''import json, pathlib, re, sys
mode, counter, fixture = sys.argv[1:4]
p=pathlib.Path(counter);n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))
s=pathlib.Path(sys.argv[sys.argv.index('-in')+1]).read_text()
text=pathlib.Path(fixture).read_text()
if mode=='flaky' and n==1: print('transient launcher failure');sys.exit(139)
if mode=='nonconv': print('convergence NOT achieved after 200 iterations');sys.exit(1)
if mode=='missing_done': text=text.replace('JOB DONE.','')
if mode=='stress_missing': text=text[:text.index('total   stress  (Ry/bohr**3)')]+'\nJOB DONE.\n'
if mode=='stress_bad': text=re.sub(r'(total   stress[^\n]*\n\s*)\S+',r'\g<1>**********',text,count=1)
if mode=='duplicate': text=re.sub(r'(atom\s+)2(\s+type)',r'\g<1>1\2',text)
if mode=='manifestdir': pathlib.Path('density_manifest.json').mkdir()
if mode in ('save','densityfail'):
    if mode=='densityfail' and re.search(r"startingpot\s*=\s*'file'",s):
        print('Error in routine read_rho: injected bad density');sys.exit(2)
    d=pathlib.Path('tmp/pyraimd2.save');d.mkdir(parents=True,exist_ok=True);(d/'charge-density.dat').write_text('fake-density')
sys.stdout.write(text)
'''
def atom():
    a=Atoms('Si2',positions=[[0,0,0],[1.36,1.36,1.36]],cell=[5.43]*3,pbc=True)
    a.set_momenta(np.zeros((2,3)));return a
def events(d): return [json.loads(s) for s in (d/'events.jsonl').read_text().splitlines()]
def command(d, mode):
    d.mkdir(parents=True,exist_ok=True);p=d/'fake.py';p.write_text(FAKE)
    return (sys.executable,str(p),mode,str(d/'calls'),str(FIXTURE))
def count(d):return int((d/'calls').read_text()) if (d/'calls').exists() else 0

def main():
    R=Path(tempfile.mkdtemp(prefix='pyramid-040-backends-',dir='/tmp'));results=[]
    def add(case,**data):results.append(dict(case=case,**data))
    ar=Atoms('Ar2',positions=[[0,0,0],[1.5,0,0]])
    for name,factory in [('plain_lj',lambda e:LennardJones(epsilon=e)),('ase_sum',lambda e:SumCalculator([LennardJones(epsilon=e)]))]:
        ee=[AseEngine(factory(e)) for e in [1,2]]
        add('fingerprint_'+name,fingerprints=[e.fingerprint for e in ee],energies=[e.compute(ar).energy for e in ee])
    for K in [QeEngine,AseQeEngine]:
        for mode in ['good','missing_done','nonconv','stress_missing','stress_bad','duplicate','manifestdir']:
            d=R/(K.__name__+'-'+mode);cmd=command(d,mode);log=EventLog(d/'log')
            e=K(QeConfig(pseudo_dir=str(d),pw_cmd=cmd,max_retries=3 if mode=='nonconv' else 0),d/'calcs',event_log=log)
            try:
                r=e.compute(atom(),request_id='probe');out={'status':'returned','finite':bool(np.isfinite(r.forces).all()),'force_consistent':r.force_consistent}
            except Exception as x:out={'status':'rejected','error':type(x).__name__+': '+str(x).splitlines()[0]}
            log.close();add(mode,backend=K.__name__,calls=count(d),attempt_events=len([x for x in events(d/'log') if x['type']=='attempt']),records=e.last_attempt_records,**out)
        # New objects continue directories; per-evaluation magnetic/charge state.
        d=R/(K.__name__+'-state');cmd=command(d,'good');cfg=QeConfig(pseudo_dir=str(d),pw_cmd=cmd,max_retries=0)
        a=atom();a.set_initial_magnetic_moments([1.,-1.]);a.set_initial_charges([.2,.3])
        e=K(cfg,d/'calcs');e.compute(a);a.set_initial_magnetic_moments([0.,0.]);a.set_initial_charges([0.,0.]);e.compute(a);K(cfg,d/'calcs').compute(a)
        files=sorted((d/'calcs').rglob('pw.in' if K is QeEngine else 'espresso.pwi'))
        add('electronic_state_and_directory',backend=K.__name__,calls=count(d),inputs={str(p.relative_to(d)): [s.strip() for s in p.read_text().splitlines() if any(k in s for k in ['nspin','starting_magnetization','tot_charge','ntyp'])] for p in files})
        # Genuine staged warm start, density fallback, and its actual time intervals.
        for mode in ['save','densityfail']:
            d=R/(K.__name__+'-'+mode);cmd=command(d,mode);log=EventLog(d/'log');e=K(QeConfig(pseudo_dir=str(d),pw_cmd=cmd,startpot_file=True),d/'calcs',event_log=log)
            e.compute(atom(),request_id='first');e.compute(atom(),request_id='second');log.close();ev=events(d/'log');ats=[x for x in ev if x['type']=='attempt'];ios=[x for x in ev if x.get('record')=='physical_io']
            add(mode,backend=K.__name__,calls=count(d),records=e.last_attempt_records,decision=e.last_density_decision,io_events=ios,attempt_events=ats,total_elapsed=summarize_tasks(ev)['total_elapsed_s'])
        # Pre-exec failure: nonexistent executable, hence zero process starts.
        d=R/(K.__name__+'-noexe');d.mkdir();log=EventLog(d/'log');e=K(QeConfig(pseudo_dir=str(d),pw_cmd=(str(d/'does-not-exist'),),max_retries=0),d/'calcs',event_log=log)
        try:e.compute(atom(),request_id='noexe')
        except Exception as x:err=type(x).__name__+': '+str(x).splitlines()[0]
        log.close();add('no_executable',backend=K.__name__,error=err,events=events(d/'log'),records=e.last_attempt_records)
    output={'root':str(R),'source':__import__('pyraimd2').__file__,'results':results}
    Path('/tmp/pyramid_040_backend_tiny.json').write_text(json.dumps(output,indent=2,default=str));print(json.dumps(output,default=str))
if __name__=='__main__': main()
