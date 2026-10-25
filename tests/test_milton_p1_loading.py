"""Prueba de carga del promedio histórico de P1"""
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from mibici_equipo14.modeling.wrappers import HistoricalMean


def test_historical_mean_roundtrip_preserves_groups_fallback_and_row_order(tmp_path):
    training = pd.DataFrame({
        'station_id': ['a', 'a', 'b'],
        'weekday': [0, 0, 1],
        'hour': [8, 8, 9],
    })
    model = HistoricalMean().fit(training, np.array([2., 6., 16.]))
    evaluation = pd.DataFrame({
        'station_id': ['a', 'b', 'a', 'new', 'a'],
        'weekday': [0, 1, 2, 0, 0],
        'hour': [8, 9, 8, 8, 8],
    }, index=[17, 3, 17, 1, 5])
    # Claves nuevas: promedio global
    expected = np.array([4., 16., 8., 8., 4.])
    path = tmp_path / 'historical_mean'
    mlflow.sklearn.save_model(
        sk_model=model, path=str(path),
        signature=infer_signature(training, model.predict(training)),
        input_example=training.head(3), pip_requirements=[],
    )
    loaded = mlflow.pyfunc.load_model(str(path))
    np.testing.assert_array_equal(loaded.predict(evaluation), expected)
    order = [3, 0, 4, 2, 1]
    np.testing.assert_array_equal(
        loaded.predict(evaluation.iloc[order]), expected[order],
    )
    # Conserva el resultado en llamadas sucesivas
    np.testing.assert_array_equal(loaded.predict(evaluation), expected)


def test_historical_mean_keeps_predictions_after_training_data_changes(tmp_path):
    training = pd.DataFrame({
        'station_id': ['a', 'a', 'b'],
        'weekday': [0, 0, 1],
        'hour': [8, 8, 9],
    })
    targets = np.array([2., 6., 16.])
    fixed_inputs = pd.DataFrame({
        'station_id': ['a', 'b', 'new'],
        'weekday': [0, 1, 0],
        'hour': [8, 9, 8],
    })
    model = HistoricalMean().fit(training, targets)
    expected = np.array([4., 16., 8.])
    np.testing.assert_array_equal(model.predict(fixed_inputs), expected)
    path = tmp_path / 'fitted_mean'
    mlflow.sklearn.save_model(
        sk_model=model, path=str(path),
        signature=infer_signature(training, model.predict(training)),
        input_example=training.head(3), pip_requirements=[],
    )

    # Cambia las variables y los objetivos originales después del ajuste
    training['station_id'] = ['changed'] * len(training)
    training['weekday'] = 6
    training['hour'] = 23
    targets[:] = 1000.

    # El estimador ajustado conserva sus predicciones sobre entradas fijas
    np.testing.assert_array_equal(model.predict(fixed_inputs), expected)
    loaded = mlflow.pyfunc.load_model(str(path))
    np.testing.assert_array_equal(loaded.predict(fixed_inputs), expected)
