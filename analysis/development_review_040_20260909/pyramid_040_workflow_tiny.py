"""R6 tiny config/workflow probes; imports the fake helper in /tmp."""
import contextlib,io,json,tempfile
from pathlib import Path
from ase.io import write
from pyraimd2.cli import main as cli_main
from pyraimd2.config import load_config
from pyraimd2.workflows import run_workflow
from pyraimd2.runtime.costs import summarize_tasks
from pyramid_040_backend_tiny import atom,command,count,events

def main():
    R=Path(tempfile.mkdtemp(prefix='pyramid-040-workflow-',dir='/tmp'));result=[]
    for backend in ['qe','qe-ase']:
        cases=[('singlepoint','flaky',False),('relax','flaky',False),('md','flaky',False),('md','good',True),('singlepoint','noexe',False),('relax','nonconv',False),('md','nonconv',False),('singlepoint','manifestdir',False)]
        for kind,mode,stationary in cases:
            d=R/(backend+'-'+kind+'-'+mode);d.mkdir();cmd=command(d,mode)
            if mode=='noexe':cmd=(str(d/'nonexistent-executable'),)
            a=atom()
            if not stationary:a.set_momenta([[.01,0,0],[.01,0,0]])
            write(d/'atoms.extxyz',a);(d/'Si.UPF').write_text('offline fake UPF')
            p=d/'run.toml';p.write_text(f'''schema_version=1
[run]
id="r6"
directory="run"
[task]
kind="{kind}"
mode="reference"
[structure]
file="atoms.extxyz"
[dynamics]
steps=2
timestep_fs=0.1
[reference]
backend="{backend}"
pseudo_dir={json.dumps(str(d))}
pw_cmd={json.dumps(list(cmd))}
max_retries={1 if mode=='flaky' else 0}
[reference.pseudos]
Si="Si.UPF"
[relax]
steps=1
fmax_eV_A=0.05
[checkpoint]
interval_steps=1
''')
            capture=io.StringIO();entry='cli' if kind=='singlepoint' and mode=='flaky' else 'run_workflow'
            try:
                with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
                    if entry=='cli':status='exit '+str(cli_main(['run',str(p)]))
                    else:run_workflow(load_config(p),verbose=False,handle_sigint=False);status='success'
            except Exception as x:status=type(x).__name__+': '+str(x).splitlines()[0]
            (d/'stdout.log').write_text(capture.getvalue());ev=events(d/'run');ledger=summarize_tasks(ev);ats=[e for e in ev if e['type']=='attempt'];tasks=[e for e in ev if e['type']=='task'];cached=[e for e in tasks if e['status']=='cache_hit']
            result.append(dict(backend=backend,kind=kind,mode=mode,entry=entry,status=status,calls=count(d),ledger=ledger,attempt_count=len(ats),task_count=len(tasks),attempt_sum=sum(e['elapsed_s'] for e in ats),cached_elapsed=[e['elapsed_s'] for e in cached],unmatched_attempt_parents=sorted({e['request_id'] for e in ats}-{e['task_id'] for e in tasks}),root=str(d)))
    output=dict(root=str(R),results=result);Path('/tmp/pyramid_040_workflow_tiny.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))
if __name__=='__main__':main()
