import json
import numpy as np
from ase import Atoms
from pyraimd2.loop import GuardedUpdater, UpdatePolicy
from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate.base import SurrogatePrediction, TrainReport
from pyraimd2.switch.base import LabelObservation
class PairModel:
    def __init__(self): self.factor=1.
    def predict(self,atoms):
        d=atoms.positions[1]-atoms.positions[0]
        return SurrogatePrediction(.5*float(d@d),np.array([d,-d])*self.factor,None,np.full(2,np.nan))
    def state_dict(self):return {'factor':self.factor}
    def load_state_dict(self,s):self.factor=s['factor']
    def finetune(self,labels):
        n=len(list(labels));self.factor=1.1
        return TrainReport(n,1,1.,.1,(.1,),0.)
a=Atoms('H2',positions=[[0.,0.,0.],[1.,0.,0.]])
m=PairModel();u=GuardedUpdater(m,UpdatePolicy(n_label=1,guard_size=1))
ref=EngineResult(.6,np.array([[1.2,0.,0.],[-1.2,0.,0.]]),None,0.)
accepted=u(LabelObservation(0,a,m.predict(a),ref,label_id='L1'))
h=1e-4;p=a.copy();n=a.copy();p.positions[1,0]+=h;n.positions[1,0]-=h
fd=(m.predict(p).energy-m.predict(n).energy)/(2*h)
error=abs(fd+m.predict(a).forces[1,0])
assert accepted and error>.09
print(json.dumps({'published':accepted,'factor':m.factor,'internal_coordinate_consistency_error':error,'n_updates':u.n_updates}))
