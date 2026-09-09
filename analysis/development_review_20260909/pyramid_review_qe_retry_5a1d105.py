import sys, os, json, re, tempfile
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
from pyraimd2.engines.qe_engine import QeEngine,QeConfig,pseudo_identities
from pyraimd2.config import load_config
from pyraimd2.workflows import run_workflow
from pyraimd2.runtime.costs import summarize_tasks
R=Path(tempfile.mkdtemp(prefix='pyramid-review-retry-',dir='/tmp'))
FIXTURE=Path(__file__).resolve().parents[2] / 'tests/data/qe_si_scf.out'
def out(case,**kw):print(json.dumps({'case':case,**kw},default=str),flush=True)
a=Atoms('Si2',positions=[[0,0,0],[1.36,1.36,1.36]],cell=[5.43]*3,pbc=True)
out('environment',artifacts=R)
# A deterministic SCF failure with a nonzero exit code.
nc=R/'nonconvergence.py';nc.write_text("print('convergence NOT achieved after 200 iterations')\nraise SystemExit(1)\n")
e=QeEngine(QeConfig(pseudo_dir=str(R),pw_cmd=(sys.executable,str(nc)),max_retries=3),R/'nc')
try:e.compute(a)
except Exception as x:out('nonconvergence_nonzero_exit',attempts=len(e.last_attempt_records),retryable=[x['retryable'] for x in e.last_attempt_records],error=type(x).__name__)
# One transient process failure then success, executed through the real workflow.
flaky=R/'flaky.py';flaky.write_text("import pathlib,sys\nif pathlib.Path.cwd().name=='attempt-1':\n    print('transient launcher failure');sys.exit(139)\nsys.stdout.write(pathlib.Path("+repr(str(FIXTURE))+").read_text())\n")
write(R/'structure.extxyz',a)
(R/'Si.UPF').write_text('offline dummy for path and hashing tests only')
cfgtext=f'''schema_version=1
[run]
id="retry"
directory="run"
[task]
kind="singlepoint"
mode="reference"
[structure]
file="structure.extxyz"
[dynamics]
steps=1
timestep_fs=0.5
[reference]
backend="qe"
pseudo_dir={json.dumps(str(R))}
pw_cmd={json.dumps([sys.executable,str(flaky)])}
max_retries=1
[reference.pseudos]
Si="Si.UPF"
'''
(R/'run.toml').write_text(cfgtext)
c=load_config(R/'run.toml');run_workflow(c,verbose=False,handle_sigint=False)
events=[json.loads(x) for x in (c.run.directory/'events.jsonl').read_text().splitlines()]
records=[x for x in events if x.get('type')=='task']
out('inner_attempts_outer_ledger',attempt_directories=len(list((c.run.directory/'calculations').glob('eval-*/attempt-*'))),task_events=[{k:x.get(k) for k in ('operation','attempt','status')} for x in records],ledger=summarize_tasks(events)['reference'])
# Python API: pseudo identity is hashed in caller cwd, but relative pseudo_dir
# is written verbatim and read in the different attempt cwd.
pseudodir=R/'relative'/'pseudos';pseudodir.mkdir(parents=True);(pseudodir/'Si.UPF').write_text('fake UPF')
reader=R/'check_pseudo.py';reader.write_text('import pathlib,sys,re\np=pathlib.Path(sys.argv[sys.argv.index("-in")+1])\ns=re.search(r"pseudo_dir\\s*=\\s*\'([^\']+)\'",p.read_text()).group(1)\nif not (pathlib.Path(s)/"Si.UPF").is_file():\n    print("Error in routine read_pseudo: missing pseudo in subprocess cwd");sys.exit(1)\nsys.stdout.write(pathlib.Path('+repr(str(FIXTURE))+').read_text())\n')
old=Path.cwd()
try:
    os.chdir(R/'relative')
    cfg=QeConfig(pseudo_dir='pseudos',pseudos={'Si':'Si.UPF'},pw_cmd=(sys.executable,str(reader)),max_retries=0)
    identities=pseudo_identities(cfg)
    e=QeEngine(cfg,'runs')
    try:e.compute(a);out('relative_pseudo_dir',status='success')
    except Exception as x:out('relative_pseudo_dir',hash_exists=identities['Si']['sha256'] is not None,status='FAIL',error_type=type(x).__name__,error=str(x))
finally:os.chdir(old)
