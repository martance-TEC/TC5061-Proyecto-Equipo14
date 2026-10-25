"""Controles de selección, empaquetado y configuración de la segunda entrega."""
import numpy as np
import pandas as pd
import pytest
import mlflow.pyfunc
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from mibici_equipo14.modeling.advanced import (resolve, choose, reproduction_tolerance,reproduction_passes)
from mibici_equipo14.modeling.wrappers import HistoricalMean, RiskDecision


def test_invalid_model_parameters_are_rejected():
    with pytest.raises(ValueError):resolve('p1','rf')
    with pytest.raises(ValueError):resolve('p3','kmeans',max_iter=100)
    with pytest.raises(ValueError):resolve('p2','rf',n_estimators=-1)
    assert resolve('p1','hgb',max_iter=0)['max_iter']==100


def test_reproduction_tolerance_is_question_specific():
    assert reproduction_tolerance('p1') == 1e-8
    assert reproduction_tolerance('p2') == 2e-3
    assert reproduction_tolerance('p3') == 1e-8


def test_p2_accepts_small_cross_environment_variation():
    assert reproduction_passes('p2', {'test_f2': .0011})
    assert not reproduction_passes('p2', {'test_f2': .0021})


def test_p3_keeps_strict_reproduction_tolerance():
    assert reproduction_passes('p3', {'test_silhouette': 1e-9})
    assert not reproduction_passes('p3', {'test_silhouette': .0011})


def test_historical_mean_falls_back_without_using_evaluation_targets():
    X=pd.DataFrame({'station_id':['a','a'],'weekday':[0,0],'hour':[8,8]})
    model=HistoricalMean().fit(X,[2,4])
    future=pd.DataFrame({'station_id':['a','b'],'weekday':[0,0],'hour':[8,8]})
    assert model.predict(future).tolist()==[3.,3.]


def test_clustering_selector_excludes_inadmissible_baseline():
    def r(k,score,size):return {'spec':{'k':k},'metrics':{'val_silhouette':score,'val_min_cluster_size':size}}
    assert choose('p3',[r(2,.9,1),r(3,.2,10),r(4,.205,10)])['spec']['k']==3
    with pytest.raises(ValueError):choose('p3',[r(2,.9,1)])


def test_regression_tolerance_keeps_the_simple_model():
    def r(model,mae):return {'spec':{'model':model},'metrics':{'val_mae':mae}}
    assert choose('p1',[r('historical_mean',.86),r('hgb',.855)])['spec']['model']=='historical_mean'


def test_risk_wrapper_retains_threshold_after_loading(tmp_path):
    X=pd.DataFrame({'x':np.arange(20,dtype=float)})
    y=np.array([0,1]*10)
    forest=RandomForestClassifier(n_estimators=5,max_depth=2,random_state=42).fit(X,y)
    wrapper=RiskDecision(forest,.25,X.columns)
    expected=wrapper.predict(None,X)
    path=tmp_path/'risk'
    mlflow.pyfunc.save_model(path=str(path),python_model=wrapper,
        signature=infer_signature(X,expected),input_example=X.head(3),pip_requirements=[])
    actual=mlflow.pyfunc.load_model(str(path)).predict(X)
    np.testing.assert_allclose(actual.to_numpy(),expected.to_numpy())
    assert np.array_equal(actual.risk_class,actual.risk_probability>=.25)

