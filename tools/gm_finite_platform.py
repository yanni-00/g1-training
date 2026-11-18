"""Finite continuation adapter; isolated account and existing OMA resource gates.

The owner's direct-formal exception applies only to the exact missing separate
minimal-run reason. All other native gate failures, source drift, budget and
remote-commit mismatch reject create/run. No training or global login mutation.
"""
import argparse,datetime as dt,hashlib,json,os,shlex,subprocess
from decimal import Decimal
from pathlib import Path
import gm_platform as gm
LOOP=Path(__file__).resolve().parents[1];PROJECT=LOOP.parents[1]
EXP=LOOP/'.oma/experiments/exp-20261118-011'
PUBLISH=LOOP/'.oma/experiments/exp-20261116-001/input/publish'
PERMIT=EXP/'input/submission_preflight.json'
read,write,sha,canonical=gm.read,gm.write,gm.sha,gm.canonical
gm.EXP=EXP;gm.PUBLISH=PUBLISH;gm.ACCOUNT_INDEX=22
git=gm.git
def request_sha256(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()
def flux(args,name,expected=0):
    c=read(EXP/'config.json');a=c['execution']['account'];index=c['execution']['account_index']
    if index!=22:raise ValueError('Unexpected account')
    owner=read(PROJECT/'.trae/skills/flux-cli/api_key.json')['accounts'][index]
    if owner['email']!=a['id']:raise ValueError('Account identity mismatch')
    env={**os.environ,a['keyEnv']:owner['api_key'],'GM_ACCOUNT_BINDING':json.dumps(a)}
    bridge="""const cp=require('child_process'),r=require(process.argv[1]);const env=r.platformEnvironment(JSON.parse(process.env.GM_ACCOUNT_BINDING));const p=cp.spawnSync(process.execPath,[process.argv[2],...process.argv.slice(3)],{env,encoding:'utf8'});process.stdout.write(p.stdout||'');process.stderr.write(p.stderr||'');process.exit(p.status===null?1:p.status);"""
    p=subprocess.run(['node','-e',bridge,str(PROJECT/'oh-my-algorithm/src/utils/resources.js'),gm.CLI,'--yes','--timeout','25s','--retry','0',*args],cwd=LOOP,env=env,capture_output=True,encoding='utf-8',timeout=45)
    try:data=json.loads(p.stdout)
    except ValueError:raise RuntimeError('GM returned no JSON; private request withheld')
    write(EXP/'raw'/name,data)
    if p.returncode!=expected or not data.get('success'):raise RuntimeError('GM failure; inspect private '+name)
    return data
def account_observation(config):
    index = config['execution']['account_index']
    observation = read(EXP/f'raw/account-{index}.json')
    observed = dt.datetime.fromisoformat(observation['observed_at'])
    age = (dt.datetime.now(dt.timezone.utc)-observed).total_seconds()
    if not 0 <= age < 300:
        raise ValueError('Refresh bound-account observation before platform admission')
    balance = observation['balance']['response']['data']
    amount = (Decimal(str(balance['balance']))+Decimal(str(balance['giftBalance'])))*1000
    # The owner cap is 50 credit for this experiment in total. Earlier aborted
    # attempts already consumed part of it, so the run may use what is left; the
    # cap itself is never enlarged.
    if not 0 < amount <= 50000 or observation['active_tasks']['response']['data']['total'] != 0:
        raise ValueError('Dedicated account lacks a positive balance within the owner cap or has an active task')
    if amount < 50000:
        initial=read(EXP/'raw/attempts/TASK_20261118_325/platform-admission.json')['account']
        if initial['index']!=index or initial['id']!=observation['account'] or initial['balance_milli_credit']!=50000:
            raise ValueError('Remaining balance has no original full-budget admission')
    if observation['account'] != config['execution']['account']['id']:
        raise ValueError('Bound account differs from observed identity')
    return index, observation, int(amount)


def platform():
    config = read(EXP/'config.json')
    if config.get('gmTaskId'):
        raise ValueError('Task already exists; do not rebuild platform admission')
    receipt = read(PERMIT)
    resume = EXP/'input/resume_request.json'
    from finite_resume_train import validate_request
    if receipt.get('format')!='f1_finite_resume_package_permit_v1' or receipt.get('canTrain') is not True:
        raise ValueError('Wrong continuation package permit')
    if receipt['request_sha256']!=sha(resume) or validate_request(read(resume))!=(15001,2000):
        raise ValueError('Finite request mismatch')
    for rel,digest in read(resume)['frozen_sources'].items():
        if sha(PUBLISH/rel)!=digest:raise ValueError('Published source drift: '+rel)
    request = config['execution']['plan']['request']
    commit = request['taskCodeInfo']['commitId']
    if len(commit or '') != 40:
        raise ValueError('Bind the actual remote commit before platform admission')
    head = git('rev-parse', 'HEAD').stdout.strip()
    if head != commit:
        raise ValueError(f'Publish HEAD {head[:12]} differs from the requested commit {commit[:12]}')
    dirty = git('status', '--porcelain').stdout.strip()
    if dirty:
        raise ValueError('Publish working tree is not frozen: '+dirty.replace('\n', ' | '))
    branch = git('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
    remote = git('ls-remote', 'origin', f'refs/heads/{branch}').stdout.split()
    if not remote or remote[0] != commit:
        raise ValueError('Remote branch does not point at the requested commit')
    index, observation, amount = account_observation(config)
    dry_run = read(EXP/'raw/train-dry-run.json')
    if not dry_run['success'] or dry_run['data']['body'] != request:
        raise ValueError('Actual platform dry-run is not bound to this exact request')
    budget_path = EXP/'input/budget-policy.json'
    write(budget_path, {'kind': 'owner_allocation_cap_not_cost_prediction',
        'authorization': 'User 2026-11-16 explicitly confirmed 50 credit per task',
        'available_milli_credit': amount, 'allocated_milli_credit': 50000,
        'unit': 'milli-GM-credit', 'account_index': index,
        'source_fields': ['balance.response.data.balance', 'balance.response.data.giftBalance',
                          'active_tasks.response.data.total'],
        'account_observation': 'raw/account-%d.json' % index,
        'no_topup_or_overdraft_authorized': True, 'cost_prediction': None,
        'failure_action': ('Preserve actual checkpoint/evidence and report budget exhaustion separately; '
                           'never recharge or enlarge this task allocation'),
        'freshness_rule_seconds': 300})
    pin = lambda path: {'path': Path(path).relative_to(EXP).as_posix(), 'sha256': sha(path)}
    candidate = {'account': config['execution']['account'], 'balance': amount, 'reserved': 0,
        'unit': 'milli-GM-credit', 'observed_at': observation['observed_at'],
        'valid_until': (dt.datetime.fromisoformat(observation['observed_at'])
                        + dt.timedelta(seconds=300)).isoformat(), 'plan_sha256': None,
        'request_check': 'passed',
        'evidence': [pin(EXP/f'raw/account-{index}.json'), pin(EXP/'raw/train-dry-run.json'),
                     pin(EXP/'raw/project-created.json')]}
    config['execution']['resource'] = {'mode': 'metered', 'plan_sha256': None,
        'required': {'amount': amount, 'unit': 'milli-GM-credit',
                     'evidence': [pin(budget_path), pin(EXP/f'raw/account-{index}.json')]},
        'allocation_cap_milli_credit': 50000,
        'diagnostic_spend_milli_credit': 50000-amount,
        'candidates': [candidate]}
    write(EXP/'config.json', config)
    resource_js = """const fs=require('fs'),r=require(process.argv[1]),a=require(process.argv[2]),c=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
      const h=r.executionPlanDigest(c,a.contractDigest);c.execution.resource.plan_sha256=h;c.execution.resource.candidates[0].plan_sha256=h;
      fs.writeFileSync(process.argv[3],JSON.stringify(c,null,2)+'\\n');
      const path=require('path'),crypto=require('crypto');const p=r.resourceAdmission(c,{digest:a.contractDigest,pin:x=>{
       const actual=crypto.createHash('sha256').update(fs.readFileSync(path.resolve(process.argv[4],x.path))).digest('hex');if(actual!==x.sha256)throw Error('Resource evidence changed');}});
      process.stdout.write(JSON.stringify(p));"""
    result = subprocess.run(['node', '-e', resource_js, str(PROJECT/'oh-my-algorithm/src/utils/resources.js'),
        str(PROJECT/'oh-my-algorithm/src/utils/admission.js'), str(EXP/'config.json'), str(EXP)],
        capture_output=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr)
    resource_admission = json.loads(result.stdout)
    write(EXP/'raw/resource-admission.json', resource_admission)
    standard = subprocess.run(['node', str(PROJECT/'oh-my-algorithm/bin/oma.js'), 'run', 'preflight',
        '--exp-ref', EXP.name, '--cwd', str(LOOP), '--json'], capture_output=True, encoding='utf-8')
    try:
        standard_report = json.loads(standard.stdout)
    except ValueError:
        raise RuntimeError('Cannot inspect the standard OMA preflight report')
    write(EXP/'raw/oma-preflight.json', standard_report)
    reasons = standard_report.get('reasons', [])
    waiver_reason = 'Formal execution needs results.assessment.readiness from an actual minimal run'
    if not standard_report.get('canTrain') and (not reasons or
            not all(reason == waiver_reason for reason in reasons)):
        raise ValueError('Unwaived standard preflight failure: '+str(reasons))
    now = dt.datetime.now(dt.timezone.utc)
    admission = {
        'format': 'f1_beyondmimic_platform_admission_v1',
        'exp_id': EXP.name,
        'project_id': request['taskBaseInfo']['projectId'],
        'branch': branch,
        'commit_id': commit,
        'request_sha256': request_sha256(request),
        'permit_sha256': sha(PERMIT),
        'account': {'index': index, 'id': observation['account'],
                    'observation_sha256': sha(EXP/f'raw/account-{index}.json'),
                    'balance_milli_credit': amount, 'active_tasks': 0},
        'budget_policy_sha256': sha(budget_path),
        'resource_admission': {'path': 'raw/resource-admission.json',
                               'sha256': sha(EXP/'raw/resource-admission.json'),
                               'plan_sha256': resource_admission['planSha256']},
        'standard_oma_preflight': {'path': 'raw/oma-preflight.json',
                                   'sha256': sha(EXP/'raw/oma-preflight.json'),
                                   'canTrain': standard_report.get('canTrain'),
                                   'reasons': reasons,
                                   'waived_scope': ('only prior probe/minimal-closure workflow; '
                                                    'exact reason: '+waiver_reason)},
        'dry_run': {'path': 'raw/train-dry-run.json', 'sha256': sha(EXP/'raw/train-dry-run.json'),
                    'success': True},
        'created_at': now.isoformat(),
        'valid_until': (now+dt.timedelta(seconds=1800)).isoformat(),
    }
    write(EXP/'raw/platform-admission.json', admission)
    print(json.dumps({'platform_admission': str(EXP/'raw/platform-admission.json'),
                      'commit_id': commit, 'branch': branch,
                      'request_sha256': admission['request_sha256'],
                      'permit_sha256': admission['permit_sha256'],
                      'standard_oma_preflight_canTrain': standard_report.get('canTrain'),
                      'valid_until': admission['valid_until']}, ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('setup','bind','preview','admit','create','run'));a=p.parse_args()
    c=read(EXP/'config.json')
    if a.action=='setup':
        for args in (['--help'],['task','--help'],['project','--help']):
            r=subprocess.run(['node',gm.CLI,*args],capture_output=True,timeout=30)
            if r.returncode:raise ValueError('CLI capability check failed')
        existing=EXP/'raw/project-created.json'
        if not existing.exists():
            payload=EXP/'work/create-project.json';write(payload,{'projectName':'F1 BeyondMimic finite continuation','projectDescription':'Owner-authorized 5000 further updates from model15000; isolated budget; server MuJoCo evaluation'})
            args=['project','create','--file',str(payload.relative_to(LOOP))]
            flux(['--dry-run',*args],'project-dry-run.json',10);flux(args,'project-created.json')
        project=read(existing)['data']
        if not isinstance(project,str):project=project['projectId']
        c['execution']['plan']['request']['taskBaseInfo']['projectId']=project;write(EXP/'config.json',c)
        flux(['project','info','--project-id',project],'project-access.json')
        flux(['task','resource','list','--goods-back-category','3','--limit','20'],'resources.json')
        flux(['task','image','official'],'images.json')
        flux(['task','image','versions','--image-id','BJX00000170'],'image-versions.json')
        print(json.dumps({'project_id':project,'account_index':22,'resource_observations_saved':True}));return
    if a.action=='bind':
        if git('status','--porcelain').stdout.strip():raise ValueError('Publish tree dirty')
        commit=git('rev-parse','HEAD').stdout.strip()
        remote=git('ls-remote','origin','refs/heads/codex/beyondmimic-f1-continue').stdout.split()
        if not remote or remote[0]!=commit:raise ValueError('Remote mismatch')
        models=read(LOOP/'.oma/experiments/exp-20261116-001/raw/task-models.json')['data']['rows']
        model=next(x for x in models if str(x['checkpoint'])=='15000')
        url=model['policUrlDown']
        req=c['execution']['plan']['request'];code=req['taskCodeInfo'];code['commitId']=commit
        code['startScript']='gm-run g1-training/tools/gm_finite_resume_entry.py --request .oma/experiments/'+EXP.name+'/input/resume_request.json --permit .oma/experiments/'+EXP.name+'/input/submission_preflight.json --checkpoint-url '+shlex.quote(url)
        write(EXP/'config.json',c);print(json.dumps({'bound_commit':commit,'private_parent_url_bound':True}));return
    if a.action=='admit':platform();return
    request=c['execution']['plan']['request'];payload=EXP/'work/create-train.json';write(payload,request)
    if a.action=='preview':
        flux(['--dry-run','task','create','--file',str(payload.relative_to(LOOP))],'train-dry-run.json',10)
    else:
        gm.verify_admission(c,a.action)
        if a.action=='create':
            if c.get('gmTaskId'):raise ValueError('Existing task must be recovered')
            account_observation(c)
            active=flux(['task','list','--status','1,2,3,4','--limit','50'],'active-before-create.json')
            if active['data']['total']:raise ValueError('Account no longer idle')
            data=flux(['task','create','--file',str(payload.relative_to(LOOP))],'task-created.json')['data']
            c['gmTaskId']=data if isinstance(data,str) else data['taskId'];write(EXP/'config.json',c)
        else:
            if not c.get('gmTaskId'):raise ValueError('No task to run')
            flux(['task','run','--task-id',c['gmTaskId']],'task-run.json')
            c['started_at']=dt.datetime.now(dt.timezone.utc).isoformat();write(EXP/'config.json',c)
    print(json.dumps({'action':a.action,'task_id':c.get('gmTaskId'),'additional_iterations':2000,'final_iteration':17000}))
if __name__=='__main__':main()
