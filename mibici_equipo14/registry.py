"""Registro, validación por inferencia y promoción por alias."""
import argparse
import json
from pathlib import Path
import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc
import numpy as np
import pandas as pd
from .config import SETTINGS as S, REGISTRY_NAMES, REPORT_DIR
from .modeling.advanced import setup, project_run, read_json, write_json

DESCRIPTIONS={
 'p1':'Demanda por estación para la siguiente hora. Apoya al operador en la planeación de flota. Cohorte histórica fija; no cubre estaciones sin historia.',
 'p2':'Clasificación de presión aproximada sobre la capacidad para las siguientes tres horas. Incluye umbral de validación. Inventario inicial supuesto al 50 por ciento y capacidad de septiembre de 2026 aplicada a viajes de 2025; no predice faltantes observados.',
 'p3':'Grupos de estaciones por perfiles históricos de uso. Incluye StandardScaler ajustado solo en entrenamiento. Coordenadas utilizadas para interpretación posterior; los grupos no representan zonas oficiales.',
}


def fixture_check(model_uri, run_id):
    path=mlflow.artifacts.download_artifacts(artifact_uri=f'runs:/{run_id}/evaluation/loading_fixture.json')
    fixture=read_json(path)
    X=pd.DataFrame(fixture['input']).astype(fixture['dtypes'])
    model=mlflow.pyfunc.load_model(model_uri)
    predicted=model.predict(X)
    expected=pd.DataFrame(fixture['output']).to_numpy() if isinstance(fixture['output'][0],dict) else np.asarray(fixture['output'])
    values=predicted.to_numpy() if isinstance(predicted,pd.DataFrame) else np.asarray(predicted)
    if values.shape!=expected.shape or not np.allclose(values,expected,atol=1e-8,rtol=0):
        raise ValueError('Loaded model differs from the evaluated model')
    return {'rows':len(X),'columns':list(X.columns),'predictions_match':True,
            'absolute_tolerance':1e-8,'output':predicted.to_dict('records') if isinstance(predicted,pd.DataFrame) else values.tolist()}


def promote(question='all', local=False):
    with project_run('eq14_registry_promote'):
        client=MlflowClient();selected=read_json(REPORT_DIR/'selection.json')
        result=[]
        for q in REGISTRY_NAMES if question=='all' else [question]:
            record=read_json(S.root/'data/interim'/f'reproduced_{q}.json') if local else selected[q]
            name=REGISTRY_NAMES[q]
            versions=client.search_model_versions(f"name = '{name}'")
            existing=next((v for v in versions if v.run_id==record['run_id']),None)
            mv=existing or mlflow.register_model(record['model_uri'],name)
            version=str(mv.version)
            check=fixture_check(f'models:/{name}/{version}',record['run_id'])
            client.update_registered_model(name,description=DESCRIPTIONS[q])
            client.update_model_version(name,version,description=f"Validado por carga e inferencia. Origen {record['run_id']}. {DESCRIPTIONS[q]}")
            for key,value in {'git_commit':record['git_commit'],'data_version':record['data_version'],
                'question':q,'delivery':'2','equipo':'14','validation_status':'passed',
                'deployment_scope':'academic_local','lifecycle_scheme':'alias',
                'source_experiment':S.experiment}.items():client.set_model_version_tag(name,version,key,str(value))
            try:previous=client.get_model_version_by_alias(name,'champion').version
            except mlflow.exceptions.MlflowException:previous=None
            client.set_registered_model_alias(name,'champion',version)
            result.append({'question':q,'name':name,'version':version,'alias':'champion',
                'previous_version':previous,'source_run_id':record['run_id'],
                'git_commit':record['git_commit'],'data_version':record['data_version'],**check})
        mlflow.log_dict(result,'registry/promotion.json')
        output=(S.root/'data/interim/local_registry_promotion.json') if local else REPORT_DIR/'registry_promotion.json'
        write_json(output,result)
        return result


def verify(question='all', local=False):
    with project_run('eq14_registry_verification'):
        client=MlflowClient();result=[]
        selection=read_json(REPORT_DIR/'selection.json')
        for q in REGISTRY_NAMES if question=='all' else [question]:
            name=REGISTRY_NAMES[q];version=client.get_model_version_by_alias(name,'champion')
            record=read_json(S.root/'data/interim'/f'reproduced_{q}.json') if local else selection[q]
            if version.run_id!=record['run_id']:raise ValueError(f'{name} has an unexpected source run')
            for key in ['git_commit','data_version']:
                if version.tags.get(key)!=record[key]:raise ValueError(f'{name} lost {key} lineage')
            check=fixture_check(f'models:/{name}@champion',version.run_id)
            result.append({'question':q,'name':name,'version':str(version.version),'alias':'champion',
                'source_run_id':version.run_id,'git_commit':record['git_commit'],
                'data_version':record['data_version'],**check})
        mlflow.log_dict(result,'registry/verification.json')
        write_json(S.root/'data/interim/local_registry_verification.json' if local else REPORT_DIR/'registry_verification.json',result)
        print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=['promote','verify'],required=True)
    parser.add_argument('--question',choices=['all','p1','p2','p3'],default='all')
    parser.add_argument('--local',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    (promote if args.action=='promote' else verify)(args.question,bool(args.local))
