"""Comprobación de los entregables técnicos de la campaña oficial."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import mlflow
from mlflow import MlflowClient
from .config import SETTINGS as S, REPORT_DIR
from .modeling.advanced import setup, read_json, write_json


def main():
    setup();client=MlflowClient();questions={};model_checks=[]
    for q in ['p1','p2','p3']:
        rows=read_json(REPORT_DIR/f'results_{q}.json')
        if sum(r['role']=='candidate' for r in rows)<3:
            raise ValueError(f'{q} needs three candidates')
        for row in rows:
            run=client.get_run(row['run_id'])
            if run.info.status!='FINISHED':raise ValueError('Unfinished model run')
            if run.data.tags.get('git_dirty')!='false':raise ValueError('Dirty training code')
            if not run.data.tags.get('mlflow.parentRunId'):raise ValueError('Missing nested relation')
            if not run.inputs.dataset_inputs:raise ValueError('Missing training dataset lineage')
            folders={f.path for f in client.list_artifacts(row['run_id'])}
            if not {'plots','config','data','evaluation'}<=folders:raise ValueError('Missing artifacts')
            info=mlflow.models.get_model_info(row['model_uri'])
            if not info.signature or not info.signature.inputs or not info.signature.outputs:
                raise ValueError('Missing model signature')
            key={'p1':'mae','p2':'f2','p3':'silhouette'}[q]
            if not {f'val_{key}',f'test_{key}'}<=run.data.metrics.keys():raise ValueError('Missing metrics')
            model_checks.append({'question':q,'run_id':row['run_id'],'status':run.info.status,
                'parent_run_id':run.data.tags['mlflow.parentRunId'],
                'signature':True,'dataset_input':True,'artifact_folders':sorted(folders)})
        questions[q]={'models':len(rows),'candidates':sum(r['role']=='candidate' for r in rows)}
    registry=read_json(REPORT_DIR/'registry_verification.json')
    if len(registry)!=3 or not all(r['predictions_match'] for r in registry):raise ValueError('Registry verification failed')
    suites=ET.parse(S.root/'data/test_results.xml').getroot().iter('testsuite')
    tests=0
    for suite in suites:
        if int(suite.get('failures',0))+int(suite.get('errors',0)):raise ValueError('Test suite failed')
        tests+=int(suite.get('tests',0))-int(suite.get('skipped',0))
    check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,check=True)
    result={'questions':questions,'model_checks':model_checks,'registry_models_verified':3,
            'tests_passed':tests,'pip_check':check.stdout.strip(),
            'baseline_rebuilt':True,'previous_database_available':False}
    write_json(REPORT_DIR/'validation.json',result)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':main()
