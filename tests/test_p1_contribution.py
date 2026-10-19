from dataclasses import replace

import numpy as np
import pandas as pd

from mibici_equipo14 import features
from mibici_equipo14.modeling import train


def test_future_only_station_cannot_change_p1_training_population(monkeypatch):
    monkeypatch.setattr(features, 'S', replace(features.S, start='2025-01-01',
        train_end='2025-01-03', val_end='2025-01-04', end='2025-01-05'))
    base = pd.DataFrame({'origin_id': ['2'], 'destination_id': ['2'],
        'started_at': pd.to_datetime(['2025-01-01 10:00']),
        'ended_at': pd.to_datetime(['2025-01-01 10:30'])})
    future = pd.DataFrame({'origin_id': ['3'], 'destination_id': ['3'],
        'started_at': pd.to_datetime(['2025-01-04 10:00']),
        'ended_at': pd.to_datetime(['2025-01-04 10:30'])})
    stations = pd.DataFrame({'station_id': ['2', '3']})
    a = features.p1_hourly_counts(base, stations)
    b = features.p1_hourly_counts(pd.concat([base, future]), stations)
    assert set(b.station_id) == {'2'}
    pd.testing.assert_frame_equal(a[a.time < '2025-01-03'], b[b.time < '2025-01-03'])
    # The existing shared panel remains available for the team's P2/P3 definitions.
    assert set(features.hourly_counts(pd.concat([base, future]), stations).station_id) == {'2', '3'}


def test_historical_mean_uses_training_targets_only():
    training = pd.DataFrame({'station_id': ['2', '2', '3'],
        'weekday': [0, 0, 1], 'hour': [10, 10, 11], 'departures': [2., 6., 10.]})
    evaluation = pd.DataFrame({'station_id': ['2', '99'],
        'weekday': [0, 6], 'hour': [10, 23], 'departures': [999., 999.]})
    before = train.historical_mean_predictions(training, evaluation)
    evaluation['departures'] = 0.
    after = train.historical_mean_predictions(training, evaluation)
    np.testing.assert_array_equal(before, [4., 6.])
    np.testing.assert_array_equal(before, after)


def test_p1_selection_uses_validation_and_favors_simpler_model_within_tolerance():
    results = {'seasonal_naive': {'val_mae': 1., 'test_mae': .1},
        'station_weekday_hour_mean': {'val_mae': .9, 'test_mae': 100.},
        'hist_gradient_boosting': {'val_mae': .895, 'test_mae': .01}}
    assert train.select_regression_model(results) == 'station_weekday_hour_mean'
    results['hist_gradient_boosting']['test_mae'] = 10000.
    assert train.select_regression_model(results) == 'station_weekday_hour_mean'


def test_p1_tolerance_is_measured_against_the_best_validation_result():
    results = {'seasonal_naive': {'val_mae': 1.},
        'station_weekday_hour_mean': {'val_mae': .995},
        'hist_gradient_boosting': {'val_mae': .989}}
    assert train.select_regression_model(results) == 'station_weekday_hour_mean'
