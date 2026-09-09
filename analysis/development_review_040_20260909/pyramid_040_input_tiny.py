"""Small remaining R5 path/fingerprint/precheck controls; no real SCF."""
import json,os,tempfile
from pathlib import Path
from pyraimd2.engines.qe_engine import QeConfig,QeEngine
from pyraimd2.engines.ase_qe import AseQeEngine
from pyraimd2.runtime.events import EventLog
from pyramid_040_backend_tiny import atom,command,count,events
R=Path(tempfile.mkdtemp(prefix='pyramid-040-inputs-',dir='/tmp'));rows=[]
for K in [QeEngine,AseQeEngine]:
    d=R/K.__name__;cmd=command(d,'good');(d/'pseudos').mkdir();(d/'pseudos'/'Si.UPF').write_text('UPF-one')
    # Real fake executable verifies pseudo_dir as resolved inside its own cwd.
    script=d/'fake.py';t=script.read_text().replace('sys.stdout.write(text)',"pseudodir=re.search(r\"pseudo_dir\\s*=\\s*'([^']+)'\",s).group(1)\nassert (pathlib.Path(pseudodir)/'Si.UPF').read_text()=='UPF-one'\nsys.stdout.write(text)");script.write_text(t)
    old=Path.cwd();os.chdir(d)
    try:
        e=K(QeConfig(pseudo_dir='pseudos',pseudos={'Si':'Si.UPF'},pw_cmd=cmd,max_retries=0),d/'calcs')
    finally:os.chdir(old)
    fp=e.fingerprint;e.compute(atom());(d/'pseudos'/'Si.UPF').write_text('UPF-two-new')
    rows.append(dict(case='relative_pseudo_and_file_hash',backend=K.__name__,calls=count(d),resolved=e.config.pseudo_dir,fingerprint_changed=fp!=e.fingerprint))
    a=atom();a.set_initial_magnetic_moments([[1,0,0],[0,1,0]]);log=EventLog(d/'precheck');e=K(e.config,d/'precheck-calcs',event_log=log)
    before=count(d)
    try:e.compute(a)
    except Exception as x:error=type(x).__name__+': '+str(x)
    log.close();rows.append(dict(case='noncollinear',backend=K.__name__,additional_calls=count(d)-before,error=error,attempts=len(events(d/'precheck'))))
try:AseQeEngine(QeConfig(pseudo_dir=str(R),timeout_s=1),R/'timeout')
except Exception as x:rows.append(dict(case='ase_timeout_rejected',error=type(x).__name__+': '+str(x)))
output=dict(root=str(R),results=rows);Path('/tmp/pyramid_040_input_tiny.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))
