"""SDK entry: validate a new finite run permit, restore parent, replace interpreter."""
import argparse,hashlib,json,os,shutil,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tasks=['Tracking-VelocityReference-F1-Proprio-v0']
try:os.dup2(1,2)
except OSError:pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True,type=Path)
    p.add_argument('--permit',required=True,type=Path);p.add_argument('--checkpoint-url',required=True)
    args=p.parse_args()
    request=ROOT/args.request;permit=ROOT/args.permit
    cfg=json.loads(request.read_text());receipt=json.loads(permit.read_text())
    digest=hashlib.sha256(request.read_bytes()).hexdigest()
    if receipt['request_sha256']!=digest or receipt['exp_id']!=cfg['exp_id'] or receipt['canTrain'] is not True:
        raise ValueError('Actual per-experiment continuation permit mismatch')
    from support_timing_resume_train import validate_request,sha
    validate_request(cfg)
    for rel,value in cfg['frozen_sources'].items():
        if sha(ROOT/rel)!=value:raise ValueError('Frozen continuation source changed: '+rel)
    sys.path.insert(0,str(ROOT/'training/source/f1_beyondmimic'))
    from f1_beyondmimic.assets_archive import ensure_assets
    ensure_assets(ROOT/'training')
    print('[F1 resume] original pinned asset archive restored and verified',flush=True)
    logs=ROOT/'training/logs/gm_resume';logs.mkdir(parents=True,exist_ok=True)
    checkpoint=logs/f"model_{cfg['parent']['iteration']}.pt"
    with urllib.request.urlopen(args.checkpoint_url,timeout=120) as src,checkpoint.open('wb') as dst:
        shutil.copyfileobj(src,dst)
    if sha(checkpoint)!=cfg['parent']['sha256']:raise ValueError('Downloaded parent hash mismatch')
    os.environ.update(F1_BM_RESUME_CHECKPOINT=str(checkpoint),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    os.chdir(ROOT/'training')
    argv=[sys.executable,str(ROOT/'tools/support_timing_resume_train.py'),'--resume-request',str(request),
          '--checkpoint',str(checkpoint),'--headless','--device','cuda:0']
    print('[F1 resume] verified finite request '+json.dumps({'exp_id':cfg['exp_id'],'limits':cfg['limits'],'parent_sha256':cfg['parent']['sha256']}),flush=True)
    os.execv(sys.executable,argv)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback,time
        traceback.print_exc(file=sys.stdout);sys.stdout.flush();time.sleep(15);raise
