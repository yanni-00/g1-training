"""Compare existing recorded states to the frozen native joint velocity limits."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
contract_path=ROOT/'training/configs/robot_contract.json'
cfg=json.loads(contract_path.read_text(encoding='utf-8'))
cases={'original_5ms':ROOT/'.oma/experiments/exp-20261118-002/output/states.csv',
       'refined_1p25ms':ROOT/'.oma/experiments/exp-20261118-004/output/states.csv',
       'discrete_5ms':ROOT/'.oma/experiments/exp-20261118-006/output/states.csv'}
result={'format':'f1_replay_velocity_constraint_audit_v1',
        'contract':str(contract_path), 'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        'native_application':'training/source/f1_beyondmimic/f1_beyondmimic/robots/f1.py#ImplicitActuatorCfg.velocity_limit_sim',
        'scope':'Recorded policy-tick pre-step samples only; peaks between logged samples unknown. Configured PhysX constraint is not proof of native measured enforcement. No state clamping applied.',
        'cases':{}}
for case,path in cases.items():
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
    values=[]
    for name in cfg['model_joint_names']:
        row=max(rows,key=lambda r:abs(float(r['dq:'+name])))
        peak=abs(float(row['dq:'+name]));limit=cfg['joint_properties'][name]['limit']['velocity']
        values.append({'joint':name,'configured_limit_rad_s':limit,'peak_abs_rad_s':peak,
            'peak_time_s':float(row['time_s']),'exceeds_configured_limit':peak>limit})
    values.sort(key=lambda r:r['peak_abs_rad_s']/r['configured_limit_rad_s'],reverse=True)
    result['cases'][case]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'samples':len(rows),'exceeding_joints':sum(v['exceeds_configured_limit'] for v in values),'joints':values}
out=ROOT/'.oma/experiments/exp-20261118-006/output/velocity_limit_audit.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:{'samples':v['samples'],'exceeding_joints':v['exceeding_joints'],'largest_ratio':v['joints'][0]} for k,v in result['cases'].items()}))
