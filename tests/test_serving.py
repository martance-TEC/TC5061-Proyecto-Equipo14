"""Contrato público de inferencia académica de P1"""
from types import SimpleNamespace
from unittest.mock import patch

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.models import Model, ModelSignature, infer_signature
from mlflow.types.schema import ColSpec, Schema

from mibici_equipo14.modeling.wrappers import HistoricalMean
from mibici_equipo14.serving import create_app


def test_predictions_preserve_order_fractions_and_loaded_version(tmp_path):
    training = pd.DataFrame({
        'station_id': ['010', '010', '20'],
        'weekday': [2, 2, 6],
        'hour': [0, 0, 23],
    }).astype({'weekday': 'int32', 'hour': 'int32'})
    model = HistoricalMean().fit(training, np.array([1., 2., 6.]))
    path = tmp_path / 'historical_mean'
    mlflow.sklearn.save_model(
        sk_model=model, path=str(path),
        signature=infer_signature(training, model.predict(training)),
        input_example=training, pip_requirements=[],
    )
    loaded = mlflow.pyfunc.load_model(str(path))
    with patch('mibici_equipo14.serving.MlflowClient') as registry, patch(
        'mibici_equipo14.serving.mlflow.pyfunc.load_model', side_effect=[loaded],
    ):
        registry.return_value.get_model_version_by_alias.return_value = SimpleNamespace(version='3')
        with TestClient(create_app()) as client:
            payload = {'instances': [
                {'station_id': '20', 'weekday': 6, 'hour': 23},
                {'station_id': '010', 'weekday': 2, 'hour': 0},
                {'station_id': 'new', 'weekday': 0, 'hour': 0},
                {'station_id': '010', 'weekday': 2, 'hour': 0},
            ]}
            response = client.post('/predict', json=payload)
            assert response.status_code == 200
            assert response.json() == {'predictions': [6., 1.5, 3., 1.5]}
            registry.return_value.get_model_version_by_alias.return_value = SimpleNamespace(version='4')
            assert client.get('/health').json()['model_version'] == '3'
            assert client.post('/predict', json=payload).json() == response.json()


def test_invalid_requests_are_rejected_before_prediction():
    metadata = Model(signature=ModelSignature(inputs=Schema([
        ColSpec('string', 'station_id'),
        ColSpec('integer', 'weekday'),
        ColSpec('integer', 'hour'),
    ])))
    valid = {'station_id': '10', 'weekday': 2, 'hour': 0}
    with patch(
        'mibici_equipo14.serving.mlflow.pyfunc.load_model',
        return_value=SimpleNamespace(metadata=metadata),
    ), TestClient(create_app(model_version=3)) as client:
        for payload in [
            {}, {'instances': []}, {'instances': None},
            {'instances': [valid], 'extra': True},
            {'instances': [{**valid, 'extra': True}]},
            {'instances': [{'station_id': '10', 'weekday': 2}]},
        ]:
            assert client.post('/predict', json=payload).status_code == 422, payload
        for field, value in [
            ('station_id', ''), ('station_id', 10), ('station_id', None),
            ('weekday', -1), ('weekday', 7), ('weekday', '2'),
            ('weekday', 2.0), ('weekday', True),
            ('hour', -1), ('hour', 24), ('hour', '0'),
            ('hour', 0.0), ('hour', False),
        ]:
            payload = {'instances': [{**valid, field: value}]}
            assert client.post('/predict', json=payload).status_code == 422, payload


def test_incompatible_four_column_signature_stops_startup():
    metadata = Model(signature=ModelSignature(inputs=Schema([
        ColSpec('string', 'station_id'),
        ColSpec('integer', 'weekday'),
        ColSpec('integer', 'hour'),
        ColSpec('double', 'departures_lag168'),
    ])))
    with patch(
        'mibici_equipo14.serving.mlflow.pyfunc.load_model',
        return_value=SimpleNamespace(metadata=metadata),
    ), pytest.raises(RuntimeError, match='firma P1'):
        with TestClient(create_app(model_version=3)):
            pass
