"""Preserve one terminal startup attempt and rebind the same finite package.

Does not create/run a cloud task, increase limits, or change frozen training.
"""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'.oma/experiments/exp-20261118-011'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    c=read(EXP/'config.json');task=c['gmTaskId'];terminal=c['execution']['terminal']
    if terminal['taskId']!=task or terminal['status'] not in ('5','6','7') or not terminal.get('notifiedAt'):raise ValueError('Terminal attempt must be observed and notified')
    models=read(EXP/'raw/task-models.json')['data']['rows']
    if any(int(x['checkpoint'])>c['checkpoint']['parent']['iteration'] for x in models):raise ValueError('New updates exist; explicit checkpoint recovery required')
    destination=EXP/'raw/attempts'/task;destination.mkdir(parents=True,exist_ok=True)
    names=('task-created.json','task-run.json','task-info.json','task-logs.json','task-models.json','platform-admission.json','oma-preflight.json','resource-admission.json','train-dry-run.json','watch.log')
    for name in names:
        src=EXP/'raw'/name
        if src.exists():shutil.copyfile(src,destination/name)
    for rel in ('config.json','input/resume_request.json','input/submission_preflight.json'):shutil.copyfile(EXP/rel,destination/Path(rel).name)
    design=ROOT/'.oma/designs/delta-f1-finite-continuation-20261118.md'
    with design.open('a',encoding='utf-8') as f:f.write('\n'+task+'原生恢复审计在PPO之前结束，仅有父15000；平台finished不等于训练完成。当前异常日志确认父恢复与原生配置/动作parity检查已过，计数源码审计未识别循环。改为符号解析本地起点/终点别名，仍要求起点等于恢复计数、终点仅加注册更新量、保存编号等于循环编号；打印实际native源码后再判定，保留所有边界拒绝与预算。\n')
    request=read(EXP/'input/resume_request.json')
    for rel in request['frozen_sources']:
        if rel.startswith('tools/') or rel.startswith('.oma/designs/'):request['frozen_sources'][rel]=sha(ROOT/rel)
    from finite_resume_train import validate_request
    if validate_request(request)!=(15001,2000):raise ValueError('Owner latest finite limit changed')
    write(EXP/'input/resume_request.json',request)
    permit=read(EXP/'input/submission_preflight.json');permit['request_sha256']=sha(EXP/'input/resume_request.json');write(EXP/'input/submission_preflight.json',permit)
    c['previous_tasks'].append({'taskId':task,'outcome':'native_resume_audit_unobserved','terminal':terminal,'source_commit':c['execution']['plan']['request']['taskCodeInfo']['commitId'],'evidence':'raw/attempts/'+task,'new_checkpoint_observed':False})
    c['gmTaskId']=None;c.pop('started_at',None);c['execution'].pop('terminal',None)
    c['deployment']['basis']['sha256']=sha(design)
    for source in c['execution']['pipeline']['sources']:source['sha256']=sha(Path(source['path']))
    write(EXP/'config.json',c)
    fields={'note_append':'TASK327平台终态5但无实际更新，仅父15000。当前结果属接入/恢复审计未闭合，不算算法成功/失败。先补恢复分派内部异常打印（在SimulationApp finally退出前），保持所有恢复/源码/原生/计数门；总新增仍2000，到17000，同一预算账号；原记录保全于raw/attempts/'+task}
    p=subprocess.run(['node',str(ROOT.parents[1]/'oh-my-algorithm/bin/oma.js'),'decision','backfill','--card','F1_BM_WALK','--exp-ref',EXP.name,'--outcome','待定','--file','-'],cwd=ROOT,input=json.dumps(fields,ensure_ascii=False),capture_output=True,encoding='utf-8')
    print(p.stdout)
    if p.returncode:raise RuntimeError(p.stderr)
    print(json.dumps({'preserved_attempt':task,'additional_iterations':2000,'expected_final_iteration':17000}))
if __name__=='__main__':main()
