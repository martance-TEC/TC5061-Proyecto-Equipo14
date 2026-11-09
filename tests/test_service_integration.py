"""Validacion HTTP del servicio P1 desplegado en contenedor."""

import os

import mlflow
import numpy as np
import pandas as pd
import pytest
import requests


SERVICE_URL = os.getenv("MIBICI_SERVICE_URL", "http://127.0.0.1:8000")
MODEL_URI = os.getenv(
    "MIBICI_MODEL_URI",
    "models:/mibici-equipo14-p1-demanda/1",
)


def test_container_health_reports_expected_model():
    response = requests.get(f"{SERVICE_URL}/health", timeout=10)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_name": "mibici-equipo14-p1-demanda",
        "model_version": "1",
    }


def test_container_predictions_match_registry_and_preserve_order():
    instances = [
        {"station_id": "10", "weekday": 2, "hour": 0},
        {"station_id": "20", "weekday": 0, "hour": 8},
        {"station_id": "10", "weekday": 2, "hour": 0},
    ]

    response = requests.post(
        f"{SERVICE_URL}/predict",
        json={"instances": instances},
        timeout=10,
    )

    assert response.status_code == 200

    api_predictions = np.asarray(
        response.json()["predictions"],
        dtype=float,
    )

    frame = pd.DataFrame(instances).astype(
        {"weekday": "int32", "hour": "int32"}
    )

    model = mlflow.pyfunc.load_model(MODEL_URI)
    registry_predictions = np.asarray(
        model.predict(frame),
        dtype=float,
    )

    assert len(api_predictions) == len(instances)
    assert api_predictions[0] == api_predictions[2]

    np.testing.assert_allclose(
        api_predictions,
        registry_predictions,
        rtol=0,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "instance",
    [
        {"station_id": "10", "weekday": 2},
        {"station_id": "10", "weekday": 7, "hour": 0},
        {"station_id": "10", "weekday": 2, "hour": 24},
    ],
)
def test_container_rejects_invalid_requests(instance):
    response = requests.post(
        f"{SERVICE_URL}/predict",
        json={"instances": [instance]},
        timeout=10,
    )

    assert response.status_code == 422
