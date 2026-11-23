"""Límite de lote del endpoint de predicción de demanda P1."""

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.models import Model, ModelSignature
from mlflow.types.schema import ColSpec, Schema

from mibici_equipo14.modeling.wrappers import HistoricalMean
from mibici_equipo14.serving import create_app


@pytest.fixture
def prediction_client(monkeypatch):
    training = pd.DataFrame([{'station_id': '10', 'weekday': 2, 'hour': 0}])
    model = HistoricalMean().fit(training, [2.0])
    metadata = Model(signature=ModelSignature(inputs=Schema([
        ColSpec('string', 'station_id'),
        ColSpec('integer', 'weekday'),
        ColSpec('integer', 'hour'),
    ])))
    monkeypatch.setattr(
        'mibici_equipo14.serving.mlflow.pyfunc.load_model',
        lambda _: SimpleNamespace(metadata=metadata, predict=model.predict),
    )
    with TestClient(create_app(model_version=1)) as client:
        yield client


def test_predict_accepts_1000_valid_instances(prediction_client):
    instance = {'station_id': '10', 'weekday': 2, 'hour': 0}
    response = prediction_client.post('/predict', json={'instances': [instance] * 1000})

    assert response.status_code == 200
    assert response.json() == {'predictions': [2.0] * 1000}


def test_predict_rejects_1001_valid_instances(prediction_client):
    instance = {'station_id': '10', 'weekday': 2, 'hour': 0}
    response = prediction_client.post('/predict', json={'instances': [instance] * 1001})

    assert response.status_code == 422
    assert any(
        error['loc'] == ['body', 'instances'] and error['type'] == 'too_long'
        for error in response.json()['detail']
    )
