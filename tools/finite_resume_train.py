"""Finite resume of immutable F1 training source; no environment/reward/control delta.

The original training snapshot and its package permit remain byte-identical to
the parent. This external, separately admitted entry changes only resume counter
accounting and checkpoint relay. New checkpoints include this entry's identity.
"""
import argparse,ast,hashlib,inspect,json,os,shutil,sys,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRAINING=ROOT/'training'


def sha(path):
    with Path(path).open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def validate_request(request):
    if request.get('kind')!='finite_original_training_continuation':raise ValueError('Wrong run kind')
    limits=request['limits'];parent=request['parent']
    if limits['spend_cap_milli_credit']!=50000 or limits['original_total_iterations']!=30000 or not 0<limits['target_total_iterations']<=limits['original_total_iterations']:
        raise ValueError('Owner cap/total target mismatch')
    first=parent['iteration']+1
    if not 0<first<limits['target_total_iterations'] or limits['additional_iterations']!=limits['target_total_iterations']-first:
        raise ValueError('Remaining PPO iteration count mismatch')
    if request['training_delta']!='none':raise ValueError('Undeclared environment/reward/control delta')
    return first,limits['additional_iterations']


def audit_runner_source(runner):
    source=textwrap.dedent(inspect.getsource(super(type(runner),runner).learn))
    print('[F1 resume] actual native learn source '+json.dumps(source),flush=True)
    return audit_learn_source(source)


def audit_learn_source(source):
    tree=ast.parse(source)
    assignments={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):assignments.setdefault(target.id,[]).append(node.value)
    def resolve(node,seen=()):
        if isinstance(node,ast.Name) and node.id not in seen and len(assignments.get(node.id,[]))==1:
            return resolve(assignments[node.id][0],(*seen,node.id))
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return ('add',resolve(node.left,seen),resolve(node.right,seen))
        return ast.unparse(node)
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
           and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and len(n.iter.args)==2
           and resolve(n.iter.args[0])=='self.current_learning_iteration']
    if len(loops)!=1:raise ValueError('Unknown RSL-RL loop semantics; no updates allowed')
    loopvar=ast.unparse(loops[0].target)
    if resolve(loops[0].iter.args[1]) not in (
        ('add','self.current_learning_iteration','num_learning_iterations'),
        ('add','num_learning_iterations','self.current_learning_iteration')):
        raise ValueError('Unknown RSL-RL upper bound')
    counters=[n for n in ast.walk(loops[0]) if isinstance(n,ast.Assign) and any(ast.unparse(t)=='self.current_learning_iteration' for t in n.targets)]
    if not any(ast.unparse(n.value)==loopvar for n in counters):raise ValueError('Unknown RSL-RL checkpoint counter')
    return hashlib.sha256(source.encode()).hexdigest()


def resume_learn(runner,original_requested_iterations,request,request_path):
    print('[F1 resume] audit 01-dispatch-entered',flush=True)
    import torch
    from f1_beyondmimic.utils.checkpoint import collect_runtime,load_checkpoint
    from f1_beyondmimic.utils.runtime_policy import verify_runtime_checkpoint
    first,remaining=validate_request(request)
    checkpoint=Path(os.environ['F1_BM_RESUME_CHECKPOINT'])
    saved=load_checkpoint(checkpoint)
    print('[F1 resume] audit 02-parent-loaded',flush=True)
    if sha(checkpoint)!=request['parent']['sha256'] or int(saved['iter'])!=request['parent']['iteration']:
        raise ValueError('Wrong parent checkpoint')
    if runner.current_learning_iteration!=int(saved['iter']) or original_requested_iterations!=request['limits']['original_total_iterations']:
        raise ValueError('Parent or original training limit mismatch')
    steps={int(v['step'].item()) if hasattr(v['step'],'item') else int(v['step']) for v in runner.alg.optimizer.state.values() if 'step' in v}
    if steps!={request['parent']['optimizer_step']}:raise ValueError('Optimizer was not restored exactly')
    count=int(runner.alg.policy.actor_obs_normalizer.count.item())
    if count!=request['parent']['normalizer_count']:raise ValueError('Actor normalizer was not restored')
    print('[F1 resume] audit 03-restored-counters',flush=True)
    current=collect_runtime(runner.env,runner.alg.policy)
    print('[F1 resume] audit 04-native-runtime-collected',flush=True)
    parent=saved['infos']['f1_runtime']
    for key in ('binding','urdf_sha256','actor_observation_contract','actor_obs_normalization','normalizer_epsilon',
                'joint_names','body_names','tracked_body_names','anchor_body_name','policy_dt','phase_acceleration',
                'command_limits','reference_sha256','action_processing','native_contact_audit'):
        if current[key]!=parent[key]:raise ValueError('Unexpected native continuation difference: '+key)
    for key,value in parent['control'].items():torch.testing.assert_close(current['control'][key],value,rtol=0,atol=0)
    for key,value in parent['motion'].items():
        actual=current['motion'][key]
        if isinstance(value,torch.Tensor):torch.testing.assert_close(actual,value,rtol=0,atol=0)
        elif actual!=value:raise ValueError('Motion changed: '+key)
    parity=verify_runtime_checkpoint(runner.alg.policy,checkpoint)
    print('[F1 resume] audit 05-parent-native-parity',flush=True)
    runner_source=audit_runner_source(runner)
    provenance={'exp_id':request['exp_id'],'request_sha256':sha(request_path),'entry_sha256':sha(Path(__file__)),
        'parent':request['parent'],'first_iteration':first,'additional_iterations':remaining,
        'expected_final_iteration':request['limits']['target_total_iterations']-1,
        'runner_learn_source_sha256':runner_source,'parent_native_parity':parity,
        'restored_optimizer_step':next(iter(steps)),'restored_normalizer_count':count,
        'environment_and_rewards_delta':'none','simulation_state_and_rng_restored':False,
        'replay_backend':'self_managed_server_mujoco'}
    runner._f1_runtime=current
    original_save=runner.save
    relay=checkpoint.parent
    def save(path,infos=None):
        info=dict(infos or {});info['f1_continuation_execution']=provenance
        original_save(path,info)
        destination=relay/Path(path).name
        if Path(path).resolve()!=destination.resolve():shutil.copy2(path,destination)
        print('[F1 resume] checkpoint relay '+json.dumps({'iteration':runner.current_learning_iteration,'file':destination.name}),flush=True)
    runner.save=save
    original_log=runner.log
    delivered_first=False
    def log(*args,**kwargs):
        nonlocal delivered_first
        original_log(*args,**kwargs)
        locs=args[0] if args else kwargs.get('locs',{})
        if not delivered_first and int(locs.get('it',-1))==first:
            runner.save(str(Path(runner.log_dir)/f'model_{first}.pt'))
            delivered_first=True
    runner.log=log
    runner.current_learning_iteration=first
    log=Path(runner.log_dir);log.mkdir(parents=True,exist_ok=True)
    (log/'finite_continuation_audit.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('[F1 resume] actual native closure '+json.dumps(provenance),flush=True)
    runner.learn(num_learning_iterations=remaining,init_at_random_ep_len=True)
    if runner.current_learning_iteration!=request['limits']['target_total_iterations']-1:
        raise ValueError('Unexpected actual final iteration')
    final=log/f'model_{runner.current_learning_iteration}.pt'
    runner.save(str(final))
    print('[F1 resume] finite continuation complete '+json.dumps(provenance),flush=True)
    import time
    time.sleep(request['delivery_upload_window_s'])


def dispatch(runner,n,request,request_path):
    # SimulationApp.close() in the original trainer's finally can terminate
    # Python before its outer exception handler prints. Flush inside this frame.
    try:return resume_learn(runner,n,request,request_path)
    except BaseException:
        import traceback,time
        traceback.print_exc(file=sys.stdout);sys.stdout.flush()
        time.sleep(15)
        raise


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--resume-request',type=Path,required=True)
    args,rest=p.parse_known_args()
    request=json.loads(args.resume_request.read_text())
    validate_request(request)
    for relative,digest in request['frozen_sources'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Frozen source changed: '+relative)
    source=TRAINING/'scripts/train.py'
    if sha(source)!=request['original_train_entry_sha256']:raise ValueError('Wrong original trainer')
    text=source.read_text()
    needle='runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)'
    if text.count(needle)!=1:raise ValueError('Unknown trainer call site')
    text=text.replace(needle,'_finite_resume_learn(runner, agent.max_iterations)')
    sys.path[:0]=[str(TRAINING/'scripts'),str(TRAINING/'source/f1_beyondmimic')]
    sys.argv=[str(source),*rest]
    scope={'__name__':'__main__','__file__':str(source),
        '_finite_resume_learn':lambda runner,n:dispatch(runner,n,request,args.resume_request)}
    # AppLauncher remains before any torch/NumPy/native imports, exactly as in
    # the immutable training entry. Only the finite learn-call dispatch differs.
    exec(compile(text,str(source),'exec'),scope)

if __name__=='__main__':main()
