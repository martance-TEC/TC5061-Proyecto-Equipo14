"""Tracking explícito del commit, fuentes, métricas y modelos."""
import json,os,shutil
import subprocess  # nosec B404: consulta Git local con argumentos fijos y sin shell.
import mlflow
import numpy as np
from .config import SETTINGS as S
def git_state():
    git = shutil.which('git')
    if git is None:
        raise RuntimeError('Git no está disponible')
    sha=subprocess.check_output([git,"rev-parse","HEAD"],cwd=S.root,text=True).strip()  # nosec B603: consulta fija de Git.
    dirty=bool(subprocess.check_output([git,"status","--porcelain","--untracked-files=no"],cwd=S.root,text=True).strip())  # nosec B603: consulta fija de Git.
    return sha,dirty
class ExperimentTracker:
    def __init__(self,integrante,manifest):
        self.integrante=integrante;self.manifest=manifest;self.sha,self.dirty=git_state()
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))
        mlflow.set_experiment(S.experiment)
    def start(self,question,model,params):
        run=mlflow.start_run(run_name=f"eq14_{question}_{model}",nested=True)
        mlflow.set_tags({"equipo":"14","pregunta":question,"git_commit":self.sha,"git_dirty":str(self.dirty).lower(),"data_version":self.manifest["data_version"],"capacity_interpretation":"retrospective capacity pressure proxy"})
        mlflow.log_params({"seed":S.seed,"start":S.start,"train_end":S.train_end,"val_end":S.val_end,"end":S.end,"data_version":self.manifest["data_version"],"model":model,**params})
        mlflow.log_dict(self.manifest,"provenance/data_manifest.json")
        return run
    def metrics(self,values):
        mlflow.log_metrics({k:float(v) for k,v in values.items() if np.isfinite(v)})
    def artifact(self,value,name):mlflow.log_dict(value,name)
