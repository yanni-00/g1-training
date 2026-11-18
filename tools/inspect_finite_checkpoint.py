"""Download/load an exact new checkpoint; never count a relay parent as progress."""
import argparse,hashlib,json,sys,urllib.request
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--exp',required=True);p.add_argument('--iteration',required=True,type=int);a=p.parse_args()
    import re
    if not re.fullmatch(r'exp-\d{8}-\d{3}',a.exp):raise ValueError('Invalid experiment')
    e=ROOT/'.oma/experiments'/a.exp;c=read(e/'config.json');request=read(e/'input/resume_request.json')
    first=request['parent']['iteration']+1
    if not first<=a.iteration<=c['checkpoint']['expectedIteration']:raise ValueError('Not a new registered child')
    row=next(x for x in read(e/'raw/task-models.json')['data']['rows'] if int(x['checkpoint'])==a.iteration)
    out=e/'output/checkpoints';out.mkdir(parents=True,exist_ok=True);path=out/f'model_{a.iteration}.pt'
    with urllib.request.urlopen(row['policUrlDown'],timeout=120) as src,path.open('wb') as dst:
        import shutil
        shutil.copyfileobj(src,dst)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    parent=torch.load(Path(request['parent']['path']),map_location='cpu',weights_only=True)
    if int(saved['iter'])!=a.iteration or sha(path)==request['parent']['sha256']:raise ValueError('Wrong or unchanged parent')
    if not all(torch.isfinite(v).all().item() for v in saved['model_state_dict'].values()):raise ValueError('Non-finite child state')
    runtime=saved['infos']['f1_runtime'];baseline=parent['infos']['f1_runtime']
    for key in ('binding','urdf_sha256','actor_observation_contract','reference_sha256','joint_names','policy_dt','native_contact_audit','action_processing'):
        if runtime[key]!=baseline[key]:raise ValueError('Runtime drift: '+key)
    provenance=saved['infos']['f1_continuation_execution']
    if provenance['request_sha256']!=sha(e/'input/resume_request.json') or provenance['entry_sha256']!=sha(ROOT/'tools/finite_resume_train.py') or provenance['exp_id']!=a.exp or provenance['parent']['sha256']!=request['parent']['sha256']:raise ValueError('New execution binding mismatch')
    steps={int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values() if 'step' in v}
    if len(steps)!=1 or next(iter(steps))<=request['parent']['optimizer_step']:raise ValueError('Optimizer did not advance')
    count=int(saved['model_state_dict']['actor_obs_normalizer.count'])
    if count<=request['parent']['normalizer_count']:raise ValueError('Normalizer did not advance')
    sys.path.insert(0,str(ROOT/'training/source/f1_beyondmimic'))
    from f1_beyondmimic.utils.runtime_policy import RuntimeActor
    actor=RuntimeActor(saved).eval()
    with torch.inference_mode():values=actor(actor.mean.expand(64,-1))
    if values.shape!=(64,29) or not torch.isfinite(values).all():raise ValueError('Portable actor load failed')
    report={'taskId':c['gmTaskId'],'path':str(path.relative_to(e)),'sha256':sha(path),'iteration':a.iteration,'loaded':True,'finite':True,'parentSha256':request['parent']['sha256'],'optimizerStep':next(iter(steps)),'normalizerCount':count,
        'scope':'Downloaded actual new checkpoint; restored state advanced, immutable parent binding and separate new execution identity; CPU portable actor loaded. No behavior or deployment acceptance.',
        'actorObservationDimension':actor.observation_dim,'execution':provenance,'nativeParityAtResume':provenance['parent_native_parity']}
    target=e/f'output/checkpoint_inspection_{a.iteration}.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='execution'},ensure_ascii=False))
if __name__=='__main__':main()
