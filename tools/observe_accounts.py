"""Read-only isolated GM account observation. Never prints or stores credentials."""
import concurrent.futures
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

LOOP = Path(__file__).resolve().parents[1]
PROJECT = LOOP.parents[1]
BASE = 'https://internal.limxdynamics.com/dev-api'
CLI = 'E:/nodejs/node_modules/@limxdynamics/flux-cli/bin/flux.js'
EXP = LOOP/'.oma/experiments/exp-20261116-001'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--exp', default=EXP.name, help='Actual experiment receiving private observations')
    args = parser.parse_args()
    pool = json.loads((PROJECT/'.trae/skills/flux-cli/api_key.json').read_text(encoding='utf-8-sig'))
    import re
    if not re.fullmatch(r'exp-\d{8}-\d{3}', args.exp):
        raise ValueError('Invalid experiment identity')
    output = LOOP/'.oma/experiments'/args.exp/'raw'
    output.mkdir(parents=True, exist_ok=True)
    # Account 15 belongs to the already-running walking route; do not select it.
    indices = args.indices or [i for i in range(len(pool['accounts'])) if i != 15]
    if any(i == 15 or not 0 <= i < len(pool['accounts']) for i in indices):
        raise ValueError('Do not select the existing walking account or an invalid index')
    balance_js = """const https=require('https');const q=https.get(process.env.GM_BAL_URL,
    {headers:{'X-API-Key':process.env.FLUX_API_KEY},timeout:15000},r=>{let b='';r.on('data',x=>b+=x);
    r.on('end',()=>{process.stdout.write(b);if(r.statusCode!==200)process.exitCode=1;});});
    q.on('timeout',()=>q.destroy(new Error('timeout')));q.on('error',()=>{process.exitCode=1;});"""

    def observe(index):
        account = pool['accounts'][index]
        env = {**os.environ, 'FLUX_API_KEY': account['api_key'], 'FLUX_BASE_URL': BASE,
               'FLUX_PROFILE': 'beyondmimic-f1-'+str(index), 'GM_BAL_URL': BASE+'/api/user/getUserBalance'}
        value = {'index': index, 'account': account['email'], 'service': BASE,
                 'observed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        try:
            result = subprocess.run(['node', '-e', balance_js], env=env, capture_output=True,
                encoding='utf-8', timeout=22)
            if result.returncode:
                return {'index': index, 'available': None, 'error': 'balance query failed'}
            value['balance'] = {'source': env['GM_BAL_URL'], 'response': json.loads(result.stdout)}
            balance = value['balance']['response'].get('data', {})
            from decimal import Decimal
            amount = (Decimal(str(balance['balance']))+Decimal(str(balance['giftBalance'])))*1000
            if not 0 < amount <= 50000 or amount != int(amount):
                return {'index': index, 'available_milli_credit': str(amount), 'eligible_under_cap': False}
            for name, args in {'identity': ['auth', 'whoami'],
                    'active_tasks': ['task', 'list', '--status', '1,2,3,4', '--limit', '50'],
                    'projects': ['project', 'list', '--limit', '50']}.items():
                result = subprocess.run(['node', CLI, '--yes', '--timeout', '15s', '--retry', '0', *args],
                    env=env, capture_output=True, encoding='utf-8', timeout=25)
                value[name] = {'exit': result.returncode, 'response': json.loads(result.stdout)}
            path = output/f'account-{index}.json'
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            active = value['active_tasks']['response'].get('data', {}).get('total')
            return {'index': index, 'available_milli_credit': int(amount), 'active_tasks': active,
                'eligible_under_cap': active == 0 and all(value[k]['exit'] == 0 for k in ('identity', 'active_tasks', 'projects'))}
        except Exception as error:
            return {'index': index, 'error': type(error).__name__+' (details withheld)'}

    observations = []
    # Read-only queries may be parallel; selection/submission remains sequential.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(observe, indices):
            observations.append(result)
            print(json.dumps(result), flush=True)
    (output/'account-selection.json').write_text(json.dumps({'observations': observations,
        'scope': 'Read-only observation; no task created and no global login modified'}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
