"""CPU contract/counter and reference GO/STOP tests; not native readiness."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace as N
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'training/source/f1_beyondmimic')]
import torch
from forward_stop_curriculum import SPEC,configure,audit_live
from forward_stop_resume_train import validate_request
from f1_beyondmimic.utils.velocity_gait_reference import VelocityGaitReference
from f1_beyondmimic.utils.checkpoint import load_checkpoint

def main():
    e=ROOT/'.oma/experiments/exp-20261119-004';q=json.loads((e/'input/resume_request.json').read_text(encoding='utf-8'))
    assert validate_request(q)==(15001,2000)
    for delta in ({'additional_iterations':2001},{'spend_cap_milli_credit':50001},{'target_total_iterations':30001}):
        bad=copy.deepcopy(q);bad['limits'].update(delta)
        try:validate_request(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid registered bounds accepted')
    bad=copy.deepcopy(q);bad['training_delta']['yaw_rate_range']=[-.2,.2]
    try:validate_request(bad)
    except ValueError:pass
    else:raise AssertionError('Turn curriculum accepted in forward-only scope')
    saved=load_checkpoint(Path(q['parent']['path']));runtime=saved['infos']['f1_runtime'];nominal=float(runtime['motion']['nominal_reference_speed_mps'][0])
    cfg=N(commands=N(motion=N(go_duration_range=(4.,12.),stop_duration_range=(2.,4.))),untouched=object())
    untouched=cfg.untouched;configure(cfg,SPEC);assert cfg.untouched is untouched
    reference=VelocityGaitReference(runtime['motion'],2,((0.,nominal),(0.,0.),(0.,0.)),acceleration=runtime['phase_acceleration'])
    command=N(cfg=cfg.commands.motion,num_envs=2,reference=reference,command_limits=((0.,nominal),(0.,0.),(0.,0.)))
    audit_live(command,SPEC)
    ids=torch.tensor([0,1]);reference.reset(ids,ids,torch.zeros(2,2),torch.zeros(2))
    reference.request(torch.tensor([[nominal,0,0]]*2),torch.ones(2,dtype=torch.bool))
    for _ in range(400):reference.step(runtime['policy_dt'])
    before_phase=reference.clock.phase.clone();before_cycles=reference.clock.cycles.clone();before_root=reference.root.position.clone()
    reference.request(torch.zeros(2,3),torch.zeros(2,dtype=torch.bool))
    assert torch.equal(reference.clock.phase,before_phase) and torch.equal(reference.clock.cycles,before_cycles) and torch.equal(reference.root.position,before_root)
    for _ in range(300):reference.step(runtime['policy_dt'])
    assert torch.equal(reference.clock.rate,torch.zeros(2))
    assert all(float(x) in reference.clock.support.tolist() for x in reference.clock.phase)
    stop_root=reference.root.position.clone();stop_cycles=reference.clock.cycles.clone()
    reference.request(torch.tensor([[nominal,0,0]]*2),torch.ones(2,dtype=torch.bool))
    assert torch.equal(reference.root.position,stop_root) and torch.equal(reference.clock.cycles,stop_cycles)
    for _ in range(400):reference.step(runtime['policy_dt'])
    assert (reference.root.position[:,0]>stop_root[:,0]).all() and torch.equal(reference.root.yaw,torch.zeros(2))
    report={'scope':'CPU parent load/bounds/config and original real reference transitions; not native rollout/training/readiness',
            'parent_iteration':int(saved['iter']),'parent_loaded':True,'parent_actor_finite':all(torch.isfinite(x).all().item() for x in saved['model_state_dict'].values()),
            'bounds_negative_checks':'passed','fixed_forward_and_zero_yaw_config':'passed','all_supports_go_stop_restart_no_reset':'passed','native_curriculum_audit':'pending_actual_startup'}
    (e/'raw/cpu-candidate-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
