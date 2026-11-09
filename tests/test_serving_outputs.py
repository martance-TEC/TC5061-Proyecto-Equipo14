"""Casos límite de entrada y protección de la respuesta de inferencia."""
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from mlflow.models import Model, ModelSignature
from mlflow.types.schema import ColSpec, Schema
import numpy as np
import pytest

from mibici_equipo14.serving import create_app


@pytest.fixture
def loaded_model():
    signature = ModelSignature(inputs=Schema([
        ColSpec('string', 'station_id'),
        ColSpec('integer', 'weekday'),
        ColSpec('integer', 'hour'),
    ]))
    return SimpleNamespace(metadata=Model(signature=signature))


@pytest.mark.parametrize('values', [[float('nan')], [float('inf')], [-1.], [], [1., 2.]])
def test_invalid_model_outputs_are_not_returned(loaded_model, values):
    loaded_model.predict = lambda frame: np.array(values)
    with patch('mibici_equipo14.serving.mlflow.pyfunc.load_model', return_value=loaded_model):
        with TestClient(create_app(model_version=1)) as client:
            result = client.post('/predict', json={
                'instances': [{'station_id': '10', 'weekday': 2, 'hour': 0}],
            })
            assert result.status_code == 500
            assert 'detail' in result.json()


@pytest.mark.parametrize('instances', [
    [{'station_id': '   ', 'weekday': 0, 'hour': 0}],
    [{'station_id': '10', 'weekday': 0, 'hour': 0}] * 1001,
])
def test_invalid_station_or_oversized_request_is_rejected(loaded_model, instances):
    with patch('mibici_equipo14.serving.mlflow.pyfunc.load_model', return_value=loaded_model):
        with TestClient(create_app(model_version=1)) as client:
            assert client.post('/predict', json={'instances': instances}).status_code == 422


def test_model_version_must_be_positive():
    with pytest.raises(ValueError, match='positiva'):
        create_app(model_version=0)
