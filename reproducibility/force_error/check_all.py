"""Validate a standalone copy using only NumPy and the standard library."""
from pathlib import Path,PurePosixPath
import csv,hashlib,json,subprocess,sys,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent

def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
    if not __debug__:raise RuntimeError('Run without Python -O: numerical assertions must remain enabled.')
    listed=set()
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);rel=PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts and name not in listed
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        listed.add(name)
    present={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'}
    present.add('silicon/MANIFEST.sha256')
    assert listed==present,(listed-present,present-listed)
    schema=json.loads((ROOT/'schema.json').read_text())
    for name,record in schema.items():
        p=ROOT/name
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                assert set(z.files)==set(record['arrays'])
                for key,a in record['arrays'].items():
                    v=z[key]
                    assert list(v.shape)==a['shape'] and str(v.dtype)==a['dtype']
                    assert v.dtype.kind in 'biuf' and np.isfinite(v).all(),(name,key)
        else:
            rr=rows(p)
            assert len(rr)==record['rows'] and list(rr[0])==record['columns'],name
    report={'passed':True,'version':(ROOT/'VERSION').read_text().strip(),'manifest_files':len(listed),'numerical_files':len(schema),'checks':{}}
    for script in ['check_source_data.py','check_interface.py','silicon/check_silicon_source_data.py','longtime/check_longtime_source_data.py','check_derived_tables.py']:
        result=subprocess.run([sys.executable,'-B',str(ROOT/script)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(script+'\n'+result.stdout+'\n'+result.stderr)
        report['checks'][script]=result.stdout.strip()
    with tempfile.TemporaryDirectory() as td:
        output=Path(td)/'baselines.csv'
        subprocess.run([sys.executable,'-B',str(ROOT/'compute_baselines.py'),str(ROOT/'data/formal_strips.csv'),str(output)],check=True,capture_output=True,text=True)
        actual,expected=rows(output),rows(ROOT/'data/free_predictor_baselines.csv')
        assert len(actual)==len(expected)==16
        for a,b in zip(actual,expected):
            assert set(a)==set(b)
            for k in a:
                if k in ['reference','configuration','predictor']:assert a[k]==b[k]
                else:np.testing.assert_allclose(float(a[k]),float(b[k]),rtol=2e-10,atol=1e-10)
    report['checks']['free_predictor_baselines']={'rows':16,'all_predictors_same_segment_subset':True}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
