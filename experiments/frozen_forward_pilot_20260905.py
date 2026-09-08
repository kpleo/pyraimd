"""Budget-driven frozen-model diagnostic with fresh velocity seeds.
This is a separately locked experiment, not completion of the interrupted
online-training pilot. It shares the original total attempt and wall budgets.
"""
import json
import os
from pathlib import Path
import signal
import time
import torch
from pyscf import lib
import forward_h2o_20260905 as base


def main():
    torch.set_num_threads(1); lib.num_threads(1)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Reference time budget')))
    previous=base.OUT
    stop=json.loads((previous/'runtime_stop.json').read_text())
    protocol=json.loads((previous/'locked_protocol.json').read_text())
    current=previous.parent/'h2o_frozen'
    current.mkdir(exist_ok=False)
    base.OUT=current
    seeds=[2026090511,2026090512]
    protocol.update({'experiment':'frozen-model forward diagnostic after runtime-limited online pilot',
        'seeds':seeds,'pilot_evaluation_states':80,'update_model':False,
        'online_updates':'none; calibrated scores and reference anchors can update, weights stay frozen',
        'reason_for_amendment':'Cost of online refitting; no admission parameters, force thresholds or model weights retuned on forward labels',
        'original_online_attempts':stop['attempts'],'combined_P1_attempt_limit':900,
        'remaining_attempt_limit':900-stop['attempts'],'absolute_deadline_unix':stop['deadline_unix'],
        'new_reference_attempt_limits':{'frozen_diagnostic':900-stop['attempts']},
        'source_script_sha256':base.sha(base.__file__),'driver_sha256':base.sha(__file__),
        'locked_unix':time.time(),'original_online_study_complete':False})
    base.write(current/'locked_protocol.json',protocol)
    os.link(previous/'common_checkpoint.pt',current/'common_checkpoint.pt')
    base.write(current/'budget_deadline.json',{'deadline_unix':stop['deadline_unix'],'remaining_seconds':stop['deadline_unix']-time.time()})
    ledger=base.ReferenceLedger('pilot',900-stop['attempts'],stop['deadline_unix']-time.time())
    summaries=[]
    try:
        for seed,start in zip(seeds,base.START_INDICES):
            for arm in base.ARMS:
                summaries.append(base.trajectory(arm,seed,start,ledger,protocol,n_states=80,update_model=False))
                base.write(current/'pilot_summary.json',{'status':'partial','arms':summaries,'attempts':ledger.n,
                    'combined_P1_attempts':ledger.n+stop['attempts'],'reference_seconds':ledger.reference_seconds})
        base.write(current/'pilot_summary.json',{'status':'complete','arms':summaries,'attempts':ledger.n,
            'combined_P1_attempts':ledger.n+stop['attempts'],'reference_seconds':ledger.reference_seconds})
    except BaseException as exc:
        base.write(current/'interrupted.json',{'status':'incomplete','reason':type(exc).__name__+': '+str(exc),
            'completed_arms':len(summaries),'attempts':ledger.n,'combined_P1_attempts':ledger.n+stop['attempts'],'time_unix':time.time()})
        raise


if __name__=='__main__': main()
