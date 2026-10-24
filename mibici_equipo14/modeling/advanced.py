"""Campaña parametrizada con selección en validación y modelos completos."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import cloudpickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    average_precision_score, fbeta_score, recall_score, precision_score,
    balanced_accuracy_score, confusion_matrix, precision_recall_curve,
    adjusted_rand_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..config import SETTINGS as S, CAMPAIGN, MODEL_DEFAULTS, DEFAULT_MODELS, REPORT_DIR
from ..dataset import sha256
from ..features import (hourly_counts, p1_hourly_counts, temporal_features,
                        capacity_labels, split_mask, station_profiles)
from ..tracking import git_state
from .train import FEATURES, cluster_metrics
from .wrappers import SeasonalNaive, HistoricalMean, RiskDecision

P2_COLUMNS = FEATURES + ['capacity']
WORK = REPORT_DIR / 'work'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(values):
    return {k: float(v) for k, v in values.items() if np.isfinite(v)}


def setup():
    threadpool_limits(limits=2)
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
    mlflow.set_registry_uri(os.environ.get('MLFLOW_REGISTRY_URI', mlflow.get_tracking_uri()))
    mlflow.set_experiment(S.experiment)
    WORK.mkdir(parents=True, exist_ok=True)


@contextmanager
def project_run(name):
    setup()
    parent = os.environ.pop('MLFLOW_RUN_ID', None)
    with (mlflow.start_run(run_id=parent) if parent else mlflow.start_run(run_name=name)) as run:
        mlflow.set_tags({'mlflow.runName': name, 'delivery': '2', 'equipo':'14'})
        yield run


def manifest():
    reference = read_json(S.root/'references/data_manifest.json')
    local_path = S.root/'data/processed/ingestion_manifest.json'
    if not local_path.exists():
        raise ValueError('Run the ingest entry point before training')
    local = read_json(local_path)
    if local['data_version'] != reference['data_version']:
        raise ValueError('Data version differs from the first delivery')
    for item in local['processed_files']:
        if sha256(S.root/item['path']) != item['sha256']:
            raise ValueError(f"Processed data changed: {item['path']}")
    return local


def prepare(question):
    """Mismas filas y transformaciones que la primera entrega; caché local."""
    source = manifest()
    cache = S.root/'data/interim'/source['data_version']/question
    cache.mkdir(parents=True, exist_ok=True)
    paths = {s:cache/f'{s}.parquet' for s in ['train','val','test']}
    meta_path = cache/'metadata.json'
    # Una huella del código de features evita reutilizar una caché incompatible.
    feature_version = sha256(S.root/'mibici_equipo14/features.py')
    if all(p.exists() for p in paths.values()) and meta_path.exists():
        meta = read_json(meta_path)
        if meta['feature_version'] == feature_version:
            return {s:pd.read_parquet(p) for s,p in paths.items()}, meta, source
    print(f'Preparing {question} features', flush=True)
    trips = pd.read_parquet(S.root/'data/processed/trips.parquet')
    stations = pd.read_parquet(S.root/'data/processed/stations.parquet')
    meta = {'feature_version':feature_version, 'data_version':source['data_version']}
    if question in ['p1','p2']:
        hourly = p1_hourly_counts(trips,stations) if question=='p1' else hourly_counts(trips,stations)
        frame = temporal_features(hourly)
        columns = FEATURES if question=='p1' else P2_COLUMNS
        if question=='p2':
            frame = frame.merge(stations[['station_id','capacity']],on='station_id',how='left')
            frame['risk'], direction = capacity_labels(trips,hourly,stations,return_direction=True)
            sensitivity = {}
            for fraction in [.25,.5,.75]:
                y = frame.risk.to_numpy() if fraction==.5 else capacity_labels(trips,hourly,stations,fraction)
                valid = pd.notna(y) & frame.risk.notna().to_numpy()
                sensitivity[str(fraction)] = {
                    'prevalence':float(np.nanmean(y)),
                    'agreement_with_main':float(np.mean(y[valid]==frame.risk.to_numpy()[valid])),
                }
            meta['initial_inventory_sensitivity'] = sensitivity
            meta['direction_counts'] = pd.Series(direction).value_counts().to_dict()
        frame = frame.dropna(subset=columns+(['risk'] if question=='p2' else []))
        parts = {s:frame[split_mask(frame.time,s,1 if question=='p1' else S.horizon_hours)].copy()
                 for s in ['train','val','test']}
        meta['cohort_stations'] = int(frame.station_id.nunique())
    else:
        hourly = hourly_counts(trips,stations)
        training = hourly[split_mask(hourly.time,'train')].copy()
        training['date'] = training.time.dt.date
        active = training[training.departures+training.arrivals>0].groupby('station_id').date.nunique()
        ids = active[active>=28].index
        bounds={'train':(S.start,S.train_end),'val':(S.train_end,S.val_end),'test':(S.val_end,S.end)}
        parts={s:station_profiles(hourly,lo,hi,ids) for s,(lo,hi) in bounds.items()}
        ids=parts['train'].index.intersection(parts['val'].index).intersection(parts['test'].index)
        parts={s:f.loc[ids].copy() for s,f in parts.items()}
        meta['eligible_stations']=len(ids)
    for s,f in parts.items():
        f.to_parquet(paths[s],index=question=='p3')
    meta['rows']={s:len(f) for s,f in parts.items()}
    write_json(meta_path,meta)
    return parts,meta,source


def resolve(question, model=None, **overrides):
    if question not in DEFAULT_MODELS:
        raise ValueError('question must be p1, p2 or p3')
    model = model or DEFAULT_MODELS[question]
    allowed={'p1':{'seasonal_naive','historical_mean','hgb'}, 'p2':{'dummy','rf'}, 'p3':{'kmeans'}}
    if model not in allowed[question]:raise ValueError('Model does not belong to question')
    values={'model':model, **MODEL_DEFAULTS[model]}
    for key,value in overrides.items():
        if value not in [None,0]:
            if key not in MODEL_DEFAULTS[model]:raise ValueError(f'{key} does not apply to {model}')
            if value<=0:raise ValueError(f'{key} must be positive')
            values[key]=value
    if model=='kmeans' and values['k']<2:raise ValueError('k must be at least 2')
    return values


def inputs(question, spec, frame):
    if question=='p3':return frame
    if spec['model']=='historical_mean':return frame[HistoricalMean.keys]
    if spec['model']=='seasonal_naive':return frame[['departures_lag168']]
    return frame[P2_COLUMNS if question=='p2' else FEATURES]


def build_model(question, spec):
    name=spec['model']; params={k:v for k,v in spec.items() if k not in ['model','role']}
    if name=='seasonal_naive':return SeasonalNaive()
    if name=='historical_mean':return HistoricalMean()
    if name=='hgb':return HistGradientBoostingRegressor(loss='poisson',random_state=S.seed,**params)
    if name=='dummy':return DummyClassifier(strategy='prior')
    if name=='rf':return RandomForestClassifier(class_weight='balanced',random_state=S.seed,n_jobs=1,**params)
    return make_pipeline(StandardScaler(),KMeans(n_init=20,random_state=S.seed,n_clusters=params['k']))


def task_metrics(question, frame, X, prediction, raw_model=None):
    if question=='p1':
        y=frame.departures.to_numpy(); pred=np.asarray(prediction)
        return finite({'mae':mean_absolute_error(y,pred), 'rmse':mean_squared_error(y,pred)**.5,
                       'wape':np.abs(y-pred).sum()/y.sum() if y.sum() else np.nan})
    if question=='p2':
        y=frame.risk.astype(int); hat=np.asarray(prediction['risk_class']); prob=np.asarray(prediction['risk_probability'])
        values={'f2':fbeta_score(y,hat,beta=2,zero_division=0),
                'recall':recall_score(y,hat,zero_division=0),
                'precision':precision_score(y,hat,zero_division=0),
                'balanced_accuracy':balanced_accuracy_score(y,hat),'prevalence':y.mean()}
        if y.nunique()==2:values['average_precision']=average_precision_score(y,prob)
        return finite(values)
    return finite(cluster_metrics(raw_model[0].transform(X),np.asarray(prediction)))


def figure(path, name, draw):
    fig,ax=plt.subplots(figsize=(7,4.5))
    draw(ax);fig.tight_layout();dest=path/f'{name}.png'
    fig.savefig(dest,dpi=150);plt.close(fig);mlflow.log_artifact(str(dest),'plots')


def plots(question, frame, X, prediction, path, split, spec):
    if question=='p1':
        y=frame.departures.to_numpy();pred=np.asarray(prediction)
        take=np.random.default_rng(S.seed).choice(len(y),min(3000,len(y)),replace=False)
        figure(path,f'{split}_actual_predicted',lambda ax:(ax.scatter(y[take],pred[take],s=7,alpha=.25),ax.set(xlabel='Viajes observados',ylabel='Viajes estimados',title=f'P1 {split} muestra reproducible')))
        figure(path,f'{split}_residuals',lambda ax:(ax.hist((y-pred)[take],bins=40),ax.set(xlabel='Real menos estimado',ylabel='Frecuencia',title=f'P1 {split} residuales')))
        groups=frame[['hour','station_id']].copy();groups['absolute_error']=np.abs(y-pred)
        by_hour=groups.groupby('hour').absolute_error.mean()
        figure(path,f'{split}_mae_by_hour',lambda ax:(ax.bar(by_hour.index,by_hour.values),ax.set(xlabel='Hora',ylabel='MAE',title=f'P1 {split} error por hora')))
        mlflow.log_dict({'by_hour':{str(k):v for k,v in by_hour.items()},'by_station':groups.groupby('station_id').absolute_error.mean().to_dict()},f'evaluation/{split}_errors.json')
    elif question=='p2':
        y=frame.risk.astype(int);hat=prediction.risk_class;prob=prediction.risk_probability
        cm=confusion_matrix(y,hat,labels=[0,1])
        def matrix(ax):
            ax.imshow(cm,cmap='Blues')
            ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicción',ylabel='Real',title=f'P2 {split} riesgo proxy clase positiva 1')
            for (i,j),v in np.ndenumerate(cm):ax.text(j,i,f'{v:,}',ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
        figure(path,f'{split}_confusion_matrix',matrix)
        precision,recall,_=precision_recall_curve(y,prob)
        figure(path,f'{split}_precision_recall',lambda ax:(ax.plot(recall,precision),ax.set(xlabel='Recall',ylabel='Precisión',title=f'P2 {split} curva precisión recall')))
        figure(path,f'{split}_scores',lambda ax:(ax.hist(prob,bins=30),ax.axvline(spec['threshold'],color='red'),ax.set(xlabel='Probabilidad del riesgo proxy',ylabel='Frecuencia',title=f'P2 {split} umbral {spec["threshold"]:.2f}')))
        mlflow.log_dict({'labels':[0,1],'matrix':cm.tolist()},f'evaluation/{split}_confusion_matrix.json')
    else:
        labels=np.asarray(prediction).astype(int)
        stations=pd.read_parquet(S.root/'data/processed/stations.parquet')
        assigned=pd.DataFrame({'station_id':frame.index,'cluster':labels}).merge(stations[['station_id','latitude','longitude']],on='station_id',how='left')
        figure(path,f'{split}_station_map',lambda ax:(ax.scatter(assigned.longitude,assigned.latitude,c=assigned.cluster,cmap='tab10',s=18),ax.set(xlabel='Longitud',ylabel='Latitud',title=f'P3 {split} grupos por uso')))
        sizes=pd.Series(labels).value_counts().sort_index()
        figure(path,f'{split}_group_sizes',lambda ax:(ax.bar(sizes.index,sizes.values),ax.set(xlabel='Grupo',ylabel='Estaciones',title=f'P3 {split} tamaño de grupos')))
        profiles=frame.copy();profiles['cluster']=labels
        averages=profiles.groupby('cluster').mean()
        cols=[f'departures_{i}' for i in range(24)]
        def profile_plot(ax):
            for cluster,row in averages.iterrows():ax.plot(range(24),row[cols],label=str(cluster))
            ax.legend(title='Grupo');ax.set(xlabel='Hora laborable',ylabel='Proporción del perfil',title=f'P3 {split} perfiles medios')
        figure(path,f'{split}_profiles',profile_plot)
        mlflow.log_dict(assigned.to_dict('records'),f'evaluation/{split}_station_clusters.json')
        mlflow.log_dict(averages.reset_index().to_dict('records'),f'evaluation/{split}_profiles.json')


def fit_and_log(question, spec, parts, meta, source):
    spec=dict(spec);role=spec.pop('role','candidate')
    token=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:7]
    name=f'eq14_{question}_{spec["model"]}_{token}'
    with mlflow.start_run(run_name=name,nested=True) as run:
        print(f'Training {name}',flush=True)
        sha,dirty=git_state()
        if dirty:raise ValueError('Commit tracked code changes before the official run')
        mlflow.set_tags({'question':question,'pregunta':question.upper(),'model_family':spec['model'],
            'stage_of_work':'baseline' if role=='baseline' else 'candidate','delivery':'2','equipo':'14',
            'previous_champion':str(role=='previous_champion').lower(),'git_commit':sha,'git_dirty':'false',
            'data_version':source['data_version'],'reference_kind':'baseline_replay' if role in ['baseline','previous_champion'] else 'new_candidate'})
        config={**spec,'seed':S.seed,'start':S.start,'train_end':S.train_end,'val_end':S.val_end,'end':S.end,
                'data_version':source['data_version'],'feature_version':meta['feature_version'],
                'horizon_hours':1 if question=='p1' else S.horizon_hours if question=='p2' else 0}
        train=parts['train'];fit=train
        if question=='p2':fit=train.sample(n=min(100000,len(train)),random_state=S.seed).sort_values('time')
        X=inputs(question,spec,fit)
        y=fit.departures if question=='p1' else fit.risk.astype(int) if question=='p2' else None
        model=build_model(question,spec);clock=time.perf_counter();model.fit(X,y);seconds=time.perf_counter()-clock
        val=parts['val'];valX=inputs(question,spec,val)
        decision=model
        if question=='p2':
            if list(model.classes_) != [0,1]:raise ValueError('Both classes required for risk evaluation')
            prob=model.predict_proba(valX)[:,1]
            threshold=max(np.arange(.1,1,.1),key=lambda t:(fbeta_score(val.risk,prob>=t,beta=2,zero_division=0),t))
            config.update(threshold=float(threshold),initial_fraction=S.initial_fraction,
                          low_fraction=S.low_fraction,high_fraction=S.high_fraction,
                          class_weight='balanced' if spec['model']=='rf' else 'none',sampling_rows=len(fit))
            decision=RiskDecision(model,threshold,X.columns)
        def predict(x):return decision.predict(None,x) if question=='p2' else decision.predict(x)
        pred=predict(valX)
        values={f'val_{k}':v for k,v in task_metrics(question,val,valX,pred,model).items()}
        if question=='p3':
            train_labels=model.predict(parts['train'])
            values['val_temporal_ari']=adjusted_rand_score(train_labels,pred)
            values['val_min_cluster_size']=int(np.bincount(pred,minlength=spec['k']).min())
        elif question in ['p1','p2']:
            for month,f in val.groupby(val.time.dt.month):
                xm=inputs(question,spec,f);pm=predict(xm)
                values.update({f'val_month_{month}_{k}':v for k,v in task_metrics(question,f,xm,pm,model).items()})
        example=X.head(3).copy()
        example_y=predict(example)
        signature=infer_signature(example,example_y)
        # Código propio incluido en el paquete para cargarlo fuera del clon.
        common=dict(name='model',signature=signature,input_example=example,
                    code_paths=[str(S.root/'mibici_equipo14')],pip_requirements=str(S.root/'requirements.lock.txt'))
        if question=='p2':
            info=mlflow.pyfunc.log_model(python_model=decision,**common)
        else:info=mlflow.sklearn.log_model(sk_model=model,**common)
        path=WORK/run.info.run_id;path.mkdir(parents=True,exist_ok=True)
        config.update(columns=list(X.columns),training_rows_available=len(train),training_rows_used=len(fit))
        mlflow.log_params({k:v for k,v in config.items() if k!='columns'})
        mlflow.log_dict(config,'config/resolved_config.json')
        mlflow.log_artifact(str(S.root/'mibici_equipo14/config.py'),'config')
        mlflow.log_artifact(str(S.root/'requirements.lock.txt'),'config')
        mlflow.log_dict(source,'data/data_manifest.json')
        mlflow.log_dict(meta,'data/feature_metadata.json')
        training=X.copy()
        if y is not None:training['target']=np.asarray(y)
        dataset=mlflow.data.from_pandas(training,source='data/processed/trips.parquet',
                    name=f'mibici_{question}_training',targets='target' if y is not None else None)
        mlflow.log_input(dataset,context='training')
        batch=valX.head(min(1000,len(valX)));predict(batch)
        durations=[]
        for _ in range(5):
            start=time.perf_counter();predict(batch);durations.append((time.perf_counter()-start)*1000*1000/len(batch))
        values.update(fit_seconds=seconds,predict_ms_per_1000_rows=float(np.median(durations)),
                      serialized_model_bytes=len(cloudpickle.dumps(decision)))
        mlflow.log_metrics(finite(values))
        plots(question,val,valX,pred,path,'val',config)
        mlflow.log_dict({'input':example.to_dict('records'),'dtypes':{c:str(t) for c,t in example.dtypes.items()},'output':example_y.to_dict('records') if isinstance(example_y,pd.DataFrame) else np.asarray(example_y).tolist()},'evaluation/loading_fixture.json')
        result={'question':question,'role':role,'run_id':run.info.run_id,'run_name':name,
                'model_uri':info.model_uri,'git_commit':sha,'data_version':source['data_version'],
                'spec':spec,'config':config,'metrics':finite(values),'rows':meta['rows']}
        mlflow.log_dict(result,'evaluation/validation.json')
        return result


def choose(question, results):
    if question=='p1':
        best=min(r['metrics']['val_mae'] for r in results)
        eligible=[r for r in results if r['metrics']['val_mae']<=best+.01]
        priority={'seasonal_naive':0,'historical_mean':1,'hgb':2}
        return min(eligible,key=lambda r:(priority[r['spec']['model']],r['spec'].get('max_iter',0),r['spec'].get('max_leaf_nodes',0)))
    if question=='p2':
        return min(results,key=lambda r:(-r['metrics']['val_f2'],r['role']!='baseline',r['metrics']['predict_ms_per_1000_rows']))
    eligible=[r for r in results if r['metrics'].get('val_min_cluster_size',0)>=5 and 'val_silhouette' in r['metrics']]
    if not eligible:raise ValueError('No admissible clustering: do not promote an invalid grouping')
    best=max(r['metrics']['val_silhouette'] for r in eligible)
    return min((r for r in eligible if r['metrics']['val_silhouette']>=best-.01),key=lambda r:r['spec']['k'])


def evaluate_test(result, parts):
    q=result['question'];spec=result['spec'];f=parts['test'];X=inputs(q,spec,f)
    model=mlflow.pyfunc.load_model(result['model_uri']);prediction=model.predict(X)
    raw=mlflow.sklearn.load_model(result['model_uri']) if q=='p3' else None
    metrics={f'test_{k}':v for k,v in task_metrics(q,f,X,prediction,raw).items()}
    if q=='p3':metrics['test_temporal_ari']=adjusted_rand_score(raw.predict(parts['train']),prediction)
    with mlflow.start_run(run_id=result['run_id'],nested=True):
        mlflow.log_metrics(finite(metrics))
        plots(q,f,X,prediction,WORK/result['run_id'],'test',result['config'])
        if q in ['p1','p2']:
            for month,monthly in f.groupby(f.time.dt.month):
                xm=inputs(q,spec,monthly)
                values=task_metrics(q,monthly,xm,model.predict(xm))
                metrics.update({f'test_month_{month}_{k}':v for k,v in values.items()})
            mlflow.log_metrics(finite(metrics))
        result['metrics'].update(finite(metrics))
        mlflow.log_dict(result,'evaluation/results.json')
    return result


def add_improvements(question, results):
    baseline=next(r for r in results if r['role']=='baseline')
    previous=next(r for r in results if r['role']=='previous_champion')
    key={'p1':'mae','p2':'f2','p3':'silhouette'}[question]
    direction=-1 if question=='p1' else 1
    for r in results:
        for split in ['val','test']:
            metric=f'{split}_{key}';value=r['metrics'].get(metric)
            if value is None:continue
            for label,ref in [('baseline',baseline),('previous',previous)]:
                base=ref['metrics'].get(metric)
                if base is None:continue
                r['metrics'][f'{split}_gain_vs_{label}']=direction*(value-base)
                if question=='p1' and base>0:r['metrics'][f'{split}_gain_percent_vs_{label}']=100*(base-value)/base


def search(question='all'):
    questions=list(CAMPAIGN) if question=='all' else [question]
    with project_run(f'eq14_{question}_campaign') as parent:
        campaign_id=parent.info.run_id;all_results={};selected={}
        for q in questions:
            parts,meta,source=prepare(q)
            with mlflow.start_run(run_name=f'eq14_{q}_search',nested=True):
                mlflow.set_tags({'question':q,'stage_of_work':'search','delivery':'2','equipo':'14'})
                results=[fit_and_log(q,spec,parts,meta,source) for spec in CAMPAIGN[q]]
                winner=dict(choose(q,results))
                winner['selection_basis']='validation_only_before_test'
                selected[q]=winner
                selection_path=REPORT_DIR/'selection.json'
                previous=read_json(selection_path) if selection_path.exists() else {}
                previous.update(selected);write_json(selection_path,previous)
                mlflow.log_dict(winner,'selection/validation_choice.json')
                for result in results:evaluate_test(result,parts)
                add_improvements(q,results)
                all_results[q]=results
                write_json(REPORT_DIR/f'results_{q}.json',results)
                flat=[{'run_id':r['run_id'],'run_name':r['run_name'],'role':r['role'],**r['spec'],**r['metrics']} for r in results]
                pd.DataFrame(flat).to_csv(REPORT_DIR/f'comparison_{q}.csv',index=False)
                mlflow.log_dict(results,f'comparison/{q}.json')
            print(f'Selected {q}: {winner["spec"]}',flush=True)
        summary={'campaign_run_id':campaign_id,'questions':questions,'candidate_count':sum(r['role']=='candidate' for rows in all_results.values() for r in rows),'baseline_reconstructed':True,'results':all_results}
        write_json(REPORT_DIR/'campaign_summary.json',summary)
        mlflow.log_dict(summary,'summary/campaign.json')
        return summary


def single(question, spec, role='candidate', reproduction=False):
    with project_run(f'eq14_{question}_reproduction' if reproduction else f'eq14_{question}_single'):
        parts,meta,source=prepare(question)
        result=fit_and_log(question,{**spec,'role':role},parts,meta,source)
        evaluate_test(result,parts)
        if reproduction:
            path=S.root/'data/interim'/f'reproduced_{question}.json'
            write_json(path,result)
            reference=read_json(REPORT_DIR/f'results_{question}.json')
            reference=next(r for r in reference if r['run_id']==read_json(REPORT_DIR/'selection.json')[question]['run_id'])
            diffs={key:abs(value-reference['metrics'][key]) for key,value in result['metrics'].items()
                   if key.startswith(('val_','test_')) and key in reference['metrics']}
            if any(v>1e-8 for v in diffs.values()):raise ValueError(f'Reproduction differs: {diffs}')
            write_json(S.root/'references/validaciones'/f'local_{question}.json',{'question':question,'passed':True,'run_id':result['run_id'],'reference_run_id':reference['run_id'],'absolute_differences':diffs,'data_version':source['data_version']})
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--question',choices=list(DEFAULT_MODELS),required=True)
    parser.add_argument('--model',default='')
    for key in ['max_iter','max_leaf_nodes','n_estimators','max_depth','min_samples_leaf','k']:
        parser.add_argument('--'+key.replace('_','-'),type=int,default=0)
    parser.add_argument('--learning-rate',type=float,default=0)
    args=vars(parser.parse_args());question=args.pop('question');model=args.pop('model')
    single(question,resolve(question,model,**args))


def reproduce(question):
    setup()
    choices=read_json(REPORT_DIR/'selection.json')
    for q in list(choices) if question=='all' else [question]:
        single(q,choices[q]['spec'],role='reproduction',reproduction=True)
