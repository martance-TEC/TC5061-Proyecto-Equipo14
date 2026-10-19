"""Experimentos P1 P2 P3 con selección en validación y test reservado."""
import json
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,average_precision_score,fbeta_score,recall_score,precision_score,balanced_accuracy_score,confusion_matrix,silhouette_score,davies_bouldin_score,calinski_harabasz_score,adjusted_rand_score
from ..config import SETTINGS as S
from ..features import split_mask,station_profiles
FEATURES=[f"{c}_{suffix}" for c in ["departures","arrivals"] for suffix in ["lag1","lag24","lag168","mean24","mean168"]]+["hour","weekday","month"]


def historical_mean_predictions(training, evaluation):
    keys = ['station_id', 'weekday', 'hour']
    means = training.groupby(keys).departures.mean()
    index = pd.MultiIndex.from_frame(evaluation[keys])
    return means.reindex(index).fillna(float(training.departures.mean())).to_numpy()


def select_regression_model(results):
    best = min(values['val_mae'] for values in results.values())
    return next(name for name, values in results.items()
                if values['val_mae'] <= best + .01)


class RegressionExperiment:
    def run(self,frame,tracker,max_iter):
        x=frame.dropna(subset=FEATURES).copy();m={s:split_mask(x.time,s,1) for s in ["train","val","test"]}
        train=x[m["train"]];model=HistGradientBoostingRegressor(loss="poisson",max_iter=max_iter,learning_rate=.1,max_leaf_nodes=31,random_state=S.seed)
        model.fit(train[FEATURES],train.departures)
        results={}
        definitions={
            'seasonal_naive':'Observed departures at the same hour 168 hours earlier',
            'station_weekday_hour_mean':'Mean departures by station,weekday,hour fitted on training only; global training mean fallback',
            'hist_gradient_boosting':'Poisson HistGradientBoosting fitted on training features',
        }
        for name in definitions:
            metrics={}
            with tracker.start("P1",name,{"horizon_hours":1,"max_iter":max_iter if name=="hist_gradient_boosting" else 0,"learning_rate":.1,"max_leaf_nodes":31,"training_rows":len(train),"cohort":"training departures intersect catalogue"}):
                for split in ["val","test"]:
                    f=x[m[split]];y=f.departures.to_numpy()
                    if name=='seasonal_naive':
                        p=f.departures_lag168.to_numpy()
                    elif name=='station_weekday_hour_mean':
                        p=historical_mean_predictions(train,f)
                    else:
                        p=model.predict(f[FEATURES])
                    metrics[f"{split}_mae"]=mean_absolute_error(y,p);metrics[f"{split}_rmse"]=mean_squared_error(y,p)**.5
                    if y.sum():metrics[f"{split}_wape"]=abs(y-p).sum()/y.sum()
                    errors=f[["station_id","hour"]].copy();errors["absolute_error"]=abs(y-p)
                    tracker.artifact({"by_station":errors.groupby("station_id").absolute_error.mean().to_dict(),"by_hour":{str(k):v for k,v in errors.groupby("hour").absolute_error.mean().items()}},f"evaluation/{split}_mae_groups.json")
                tracker.metrics(metrics)
                tracker.artifact({"feature_columns":FEATURES,"evaluation_rows":{s:int(m[s].sum()) for s in m}},"evaluation/definition.json")
                if name=="hist_gradient_boosting":mlflow.sklearn.log_model(model,name="model",input_example=train[FEATURES].head(3))
                if name=='station_weekday_hour_mean':
                    means=train.groupby(['station_id','weekday','hour']).departures.mean().rename('mean_departures')
                    tracker.artifact({'means':means.reset_index().to_dict('records'),'fallback':float(train.departures.mean())},'model/historical_means.json')
                tracker.artifact({"definition":definitions[name]},"model_definition.json")
                results[name]=metrics
        winner=select_regression_model(results)
        return {"models":results,"selected":winner,"selection":"validation MAE; tolerance 0.01; tie priority seasonal_naive,station_weekday_hour_mean,hist_gradient_boosting"}
class ClassificationExperiment:
    def run(self,frame,tracker):
        features=FEATURES+["capacity"]
        x=frame.dropna(subset=features+["risk"]).copy();m={s:split_mask(x.time,s,3) for s in ["train","val","test"]};train=x[m["train"]]
        # Bounded laptop experiment: deterministic sample only from training.
        fit=train.sample(n=min(100000,len(train)),random_state=S.seed).sort_values("time")
        results={}
        for name,model in [("dummy_prior",DummyClassifier(strategy="prior")),("random_forest",RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=10,class_weight="balanced",random_state=S.seed,n_jobs=1))]:
            model.fit(fit[features],fit.risk.astype(int));val=x[m["val"]];prob=model.predict_proba(val[features])[:,list(model.classes_).index(1)]
            grid=np.arange(.1,1,.1);threshold=max(grid,key=lambda t:(fbeta_score(val.risk,prob>=t,beta=2,zero_division=0),t))
            metrics={};matrices={}
            with tracker.start("P2",name,{"horizon_hours":3,"initial_fraction":S.initial_fraction,"low_fraction":S.low_fraction,"high_fraction":S.high_fraction,"threshold":float(threshold),"training_rows_available":len(train),"training_rows_used":len(fit),"sampling":"uniform train only seed 42, max 100000","n_estimators":200 if name=="random_forest" else 0,"max_depth":12,"min_samples_leaf":10,"class_weight":"balanced"}):
                for split in ["val","test"]:
                    f=x[m[split]];y=f.risk.astype(int);p=model.predict_proba(f[features])[:,list(model.classes_).index(1)];hat=p>=threshold
                    metrics.update({f"{split}_f2":fbeta_score(y,hat,beta=2,zero_division=0),f"{split}_recall":recall_score(y,hat,zero_division=0),f"{split}_precision":precision_score(y,hat,zero_division=0),f"{split}_balanced_accuracy":balanced_accuracy_score(y,hat),f"{split}_prevalence":y.mean()})
                    if y.nunique()==2:metrics[f"{split}_average_precision"]=average_precision_score(y,p)
                    matrices[split]=confusion_matrix(y,hat,labels=[0,1]).tolist()
                tracker.metrics(metrics);tracker.artifact(matrices,"evaluation/confusion_matrices.json")
                tracker.artifact({"warning":"Hypothetical initial inventory with real station capacity snapshot September 2026. Labels describe retrospective flow pressure, not observed 2025 empty/full states.","features":features},"evaluation/target_definition.json")
                mlflow.sklearn.log_model(model,name="model",input_example=fit[features].head(3))
                results[name]=metrics
        winner="random_forest" if results["random_forest"]["val_f2"]>results["dummy_prior"]["val_f2"] else "dummy_prior"
        return {"models":results,"selected":winner,"selection":"maximum validation F2; ties favor baseline"}
def cluster_metrics(x,labels):
    if not 2<=len(set(labels))<len(x):return {}
    return {"silhouette":silhouette_score(x,labels),"davies_bouldin":davies_bouldin_score(x,labels),"calinski_harabasz":calinski_harabasz_score(x,labels)}
class ClusteringExperiment:
    def run(self,hourly,stations,tracker):
        train=hourly[split_mask(hourly.time,"train")].copy();train["date"]=train.time.dt.date
        active=train[train.departures+train.arrivals>0].groupby("station_id").date.nunique();ids=active[active>=28].index
        frames={s:station_profiles(hourly,lo,hi,ids) for s,(lo,hi) in {"train":(S.start,S.train_end),"val":(S.train_end,S.val_end),"test":(S.val_end,S.end)}.items()}
        ids=frames["train"].index.intersection(frames["val"].index).intersection(frames["test"].index);frames={s:f.loc[ids] for s,f in frames.items()}
        results={};valid=[]
        for k in range(2,7):
            if k>=len(ids):continue
            model=make_pipeline(StandardScaler(),KMeans(n_clusters=k,n_init=20,random_state=S.seed));model.fit(frames["train"])
            metrics={};labels={}
            with tracker.start("P3",f"kmeans_k{k}",{"k":k,"n_init":20,"eligible_stations":len(ids),"min_active_train_days":28,"scaler_fit":"train only","geography_in_fit":False}):
                for split,f in frames.items():
                    z=model[0].transform(f);labels[split]=model.predict(f)
                    metrics.update({f"{split}_{key}":v for key,v in cluster_metrics(z,labels[split]).items()})
                metrics["val_temporal_ari"]=adjusted_rand_score(labels["train"],labels["val"]);metrics["test_temporal_ari"]=adjusted_rand_score(labels["train"],labels["test"])
                tracker.metrics(metrics);mlflow.sklearn.log_model(model,name="model",input_example=frames["train"].head(3))
                assignments=pd.DataFrame({"station_id":ids,"cluster":labels["train"].astype(int)}).merge(stations[["station_id","name","latitude","longitude"]],on="station_id",how="left")
                tracker.artifact(assignments.to_dict("records"),"interpretation/station_clusters.json")
                tracker.artifact(frames["train"].reset_index().to_dict("records"),"interpretation/train_station_profiles.json")
                import matplotlib;matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(figsize=(8,6));ax.scatter(assignments.longitude,assignments.latitude,c=assignments.cluster,cmap="tab10",s=15);ax.set(xlabel="Longitud",ylabel="Latitud",title=f"MiBici grupos por uso k={k}");fig.tight_layout();path=S.root/"reports/figures"/f"clusters_k{k}.png";fig.savefig(path,dpi=160);plt.close(fig);mlflow.log_artifact(str(path),"interpretation")
                counts=np.bincount(labels["val"],minlength=k)
                if counts.min()>=5 and "val_silhouette" in metrics:valid.append(k)
                results[str(k)]=metrics
        winner=2
        for k in valid:
            if results[str(k)].get("val_silhouette",-1)>results[str(winner)].get("val_silhouette",-1)+.01:winner=k
        return {"models":results,"selected_k":winner,"selection":"validation silhouette, min 5 stations per validation group; tolerance 0.01 favors smaller k","eligible_stations":len(ids)}
