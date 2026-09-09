import sys, os, json, re, dataclasses, tempfile, time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.surrogate.ase_surrogate import AseSurrogate
from pyraimd2.engines.ase_qe import AseQeEngine, make_espresso_calculator
from pyraimd2.engines.qe_engine import QeEngine,QeConfig,write_qe_input,parse_qe_output
R=Path(tempfile.mkdtemp(prefix='pyramid-review-backends-',dir='/tmp'))
FIXTURE=Path(__file__).resolve().parents[2] / 'tests/data/qe_si_scf.out'
original=FIXTURE.read_text()
def out(tag, **kw): print(json.dumps({'case':tag,**kw},ensure_ascii=False,default=str),flush=True)
def fake(name, text, logic=''):
    p=R/(name+'.py')
    p.write_text('import sys, pathlib, re\n'+logic+'\nsys.stdout.write('+repr(text)+')\n')
    return (sys.executable,str(p))
def si(): return Atoms('Si2',positions=[[0,0,0],[1.36,1.36,1.36]],cell=[5.43]*3,pbc=True)
def cfg(cmd, **kw): return QeConfig(pseudo_dir=str(R), pw_cmd=cmd,max_retries=0,**kw)
out('environment', artifacts=R)
a=Atoms('Ar2',positions=[[0,0,0],[1.5,0,0]])
e1,e2=[AseEngine(LennardJones(epsilon=x)) for x in (1,2)]
s1,s2=[AseSurrogate(LennardJones(epsilon=x)) for x in (1,2)]
out('ase_fingerprint_collision',fingerprints=[e1.fingerprint,e2.fingerprint],energies=[e1.compute(a).energy,e2.compute(a).energy],surrogate_same=s1.fingerprint==s2.fingerprint)
cmd=fake('good',original)
e=AseQeEngine(cfg(cmd),R/'ase-restart')
r=e.compute(si())
out('ase_qe_metadata',caps_force_consistent=e.capabilities.force_consistent,result_force_consistent=r.force_consistent,energy_kind=r.energy_kind)
try: AseQeEngine(cfg(cmd),R/'ase-restart').compute(si())
except Exception as x: out('ase_qe_fresh_instance',error_type=type(x).__name__,error=str(x))
# A valid result body followed by premature termination, or a nonconvergence marker.
for name,text in [('missing_done',original.replace('JOB DONE.','')),('not_converged',original.replace('JOB DONE.','convergence NOT achieved after 200 iterations\nJOB DONE.'))]:
    command=fake(name,text)
    for kind,K in [('ase',AseQeEngine),('handwritten',QeEngine)]:
        try:
            label=K(cfg(command),R/(name+'-'+kind)).compute(si())
            out(name,backend=kind,status='ACCEPTED',energy=label.energy)
        except Exception as x: out(name,backend=kind,status='REJECTED',error_type=type(x).__name__,error=str(x).splitlines()[0])
# No density exists. A real input reader fails if startingpot=file was written.
command=fake('needs_density',original,"p=pathlib.Path(sys.argv[sys.argv.index('-in')+1])\nif re.search(r\"startingpot\\s*=\\s*'file'\",p.read_text()) and not pathlib.Path('tmp/pyraimd2.save').is_dir():\n    print('Error in routine read_rho: no density file');sys.exit(2)\n")
for kind,K in [('ase',AseQeEngine),('handwritten',QeEngine)]:
    try:
        K(cfg(command,startpot_file=True),R/('warm-'+kind)).compute(si())
        out('warm_start_no_density',backend=kind,status='success')
    except Exception as x: out('warm_start_no_density',backend=kind,error_type=type(x).__name__,error=str(x))
# A nonzero spin is a supported ASE Atoms input, not a backend config option.
a=si();a.set_initial_magnetic_moments([1,-1])
c=cfg(cmd)
p=R/'spin-handwritten.in';write_qe_input(p,a,c)
calc=make_espresso_calculator(c,directory=R/'spin-ase');calc.directory.mkdir();calc.write_inputfiles(a,['energy'])
ase_text=(calc.directory/'espresso.pwi').read_text()
out('spin_input_mapping',handwritten=[x.strip() for x in p.read_text().splitlines() if any(y in x for y in ('nspin','ntyp','starting_magnetization'))],ase=[x.strip() for x in ase_text.splitlines() if any(y in x for y in ('nspin','ntyp','starting_magnetization'))])
# Malformed/truncated stress and repeated force indices.
bad_stress,n_changed=re.subn(r'(total   stress[^\n]*\n\s*)\S+',r'\g<1>**********',original,count=1)
assert n_changed==1 and bad_stress != original
for name,text in [('missing_stress',original[:original.index('total   stress  (Ry/bohr**3)')]+'\nJOB DONE.\n'),('bad_stress',bad_stress),('duplicate_atom',re.sub(r'(atom\s+)2(\s+type)',r'\g<1>1\2',original))]:
    try:
        e=QeEngine(cfg(fake(name,text)),R/name);r=e.compute(si())
        out(name,status='ACCEPTED',stress=r.stress,forces=r.forces.tolist(),stress_capability=e.capabilities.stress_available)
    except Exception as x: out(name,status='REJECTED',error_type=type(x).__name__,error=str(x).splitlines()[0])
