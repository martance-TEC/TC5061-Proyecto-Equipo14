"""Punto único de ejecución por pregunta; trazabilidad y resumen verificable."""
import argparse,json,os
import pandas as pd
import mlflow
from .config import SETTINGS as S
from .features import hourly_counts,temporal_features,capacity_labels
from .tracking import ExperimentTracker
from .modeling.train import RegressionExperiment,ClassificationExperiment,ClusteringExperiment
def main(question="all",integrante="local",max_iter=0):
    manifest=json.loads((S.root/"references/data_manifest.json").read_text())
    tracker=ExperimentTracker(integrante,manifest)
    trips=pd.read_parquet(S.root/"data/processed/trips.parquet");stations=pd.read_parquet(S.root/"data/processed/stations.parquet")
    print("Aggregating closed hourly events",flush=True);hourly=hourly_counts(trips,stations);frame=None;results={}
    # Consume the Projects run identifier once so child runs cannot resume it.
    parent_id=os.environ.pop("MLFLOW_RUN_ID",None)
    with (mlflow.start_run(run_id=parent_id) if parent_id else mlflow.start_run(run_name=f"eq14_all_{integrante}")) as parent:
        mlflow.set_tag("mlflow.runName",f"eq14_{question}_orquestador_{integrante}")
        if question in ["all","P1","P2"]:frame=temporal_features(hourly)
        if question in ["all","P1"]:
            print("Running P1",flush=True);results["P1"]=RegressionExperiment().run(frame,tracker,max_iter or S.max_iter)
        if question in ["all","P2"]:
            print("Creating P2 minute prefix labels",flush=True);frame=frame.merge(stations[["station_id","capacity"]],on="station_id",how="left")
            frame["risk"],frame["risk_direction"]=capacity_labels(trips,hourly,stations,return_direction=True)
            sensitivity={}
            for f in [.25,.5,.75]:
                labels=frame.risk.to_numpy() if f==.5 else capacity_labels(trips,hourly,stations,f)
                valid=pd.notna(labels)&frame.risk.notna().to_numpy()
                sensitivity[str(f)]={"prevalence":float(pd.Series(labels).mean()),"agreement_with_main":float((labels[valid]==frame.risk.to_numpy()[valid]).mean())}
            results["P2"]=ClassificationExperiment().run(frame,tracker);results["P2"]["initial_inventory_sensitivity"]=sensitivity;results["P2"]["direction_counts"]=frame.risk_direction.value_counts().to_dict()
        if question in ["all","P3"]:
            print("Running P3",flush=True);results["P3"]=ClusteringExperiment().run(hourly,stations,tracker)
        result={"git_commit":tracker.sha,"git_dirty":tracker.dirty,"data_version":manifest["data_version"],"parent_run_id":parent.info.run_id,"results":results}
        mlflow.log_dict(result,"summary/results.json")
        path=S.root/"reports"/f"reproduccion_{integrante}.json";path.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--question",choices=["all","P1","P2","P3"],default="all");p.add_argument("--integrante",default="local");p.add_argument("--max-iter",type=int,default=0);main(**vars(p.parse_args()))
