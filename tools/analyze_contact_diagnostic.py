"""Describe recorded contact data and require unchanged original trajectory."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--exp',default='exp-20261118-010');a=p.parse_args()
E=ROOT/'.oma/experiments'/a.exp
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8',newline='')))
old=rows(ROOT/'.oma/experiments/exp-20261118-008/output/states.csv');new=rows(E/'output/states.csv')
if len(old)!=len(new):raise ValueError('Telemetry changed episode length')
max_error=0.
for x,y in zip(old,new):
    for key,value in x.items():
        try:error=abs(float(value)-float(y[key]))
        except ValueError:
            if value!=y[key]:raise ValueError('Changed original field '+key)
            continue
        if not math.isfinite(error):raise ValueError('Nonfinite parity result')
        max_error=max(max_error,error)
if max_error!=0:raise ValueError('Telemetry changed original trajectory: '+str(max_error))
sub=rows(E/'output/contact_substeps.csv')
report={'format':'f1_contact_quality_diagnostic_v1','original_columns_max_abs_difference':max_error,
    'original_policy_steps':len(old),'physics_samples':len(sub),'source':'output/contact_substeps.csv',
    'force_threshold_N':10.,'force_threshold_source':'model15000#infos.f1_runtime.native_contact_audit.force_threshold',
    'scope':'Descriptive only in existing MuJoCo drive diagnostic; no new naturalness or deployment gate. No independent training seeds.',
    'stages':{}}
for stage in dict.fromkeys(r['stage'] for r in sub):
    group=[r for r in sub if r['stage']==stage];stats={}
    for side in ('left','right'):
        contact=np.array([float(r[side+':contact_point_normal_force_N'])>report['force_threshold_N'] for r in group])
        speeds=np.array([float(r[side+':contact_point_tangent_rms_mps']) for r in group])
        refspeed=np.array([math.hypot(float(r[side+':reference_body_vx_mps']),float(r[side+':reference_body_vy_mps'])) for r in group])
        starts=np.flatnonzero(contact&~np.r_[False,contact[:-1]])
        events=[]
        for start in starts:
            if start==0:continue
            events.append({'time_s':float(group[start]['time_s']),'phase':float(group[start]['phase']),
                'contact_tangent_rms_mps':float(speeds[start]),
                'actual_collision_lowest_z_m':float(group[start][side+':collision_lowest_z_m']),
                'reference_collision_lowest_z_m':float(group[start][side+':reference_collision_lowest_z_m'])})
        stats[side]={'contact_fraction':float(contact.mean()),'contact_samples':int(contact.sum()),
            'contact_tangent_mean_mps':float(speeds[contact].mean()) if contact.any() else None,
            'contact_tangent_rms_mps':float(np.sqrt((speeds[contact]**2).mean())) if contact.any() else None,
            'reference_foot_body_planar_speed_during_actual_contact_mean_mps':float(refspeed[contact].mean()) if contact.any() else None,
            'touchdown_candidates':events}
    stats['double_support_fraction']=float(np.mean([(float(r['left:contact_point_normal_force_N'])>10) and (float(r['right:contact_point_normal_force_N'])>10) for r in group]))
    report['stages'][stage]=stats
write=E/'output/contact_quality_analysis.json';write.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({**{k:v for k,v in report.items() if k!='stages'},'straight_sustained':{k:{n:v for n,v in s.items() if n!='touchdown_candidates'} if isinstance(s,dict) else s for k,s in report['stages']['straight_sustained'].items()}},indent=2))
