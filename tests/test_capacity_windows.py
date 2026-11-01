"""El riesgo se construye con extremos dentro de la ventana, no solo con el saldo final."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mibici_equipo14 import features


@pytest.fixture
def short_window(monkeypatch):
    settings = replace(features.S, start='2025-01-01', end='2025-01-01 04:00:00')
    monkeypatch.setattr(features, 'S', settings)
    return pd.DataFrame({'station_id': ['a'] * 4,
                         'time': pd.date_range(settings.start, periods=4, freq='h')})


def test_intermediate_shortage_remains_a_positive_label(short_window):
    trips = pd.DataFrame({'origin_id': ['a'] * 9, 'destination_id': ['a'] * 9,
                          'started_at': pd.to_datetime(['2025-01-01 00:10'] * 9),
                          'ended_at': pd.to_datetime(['2025-01-01 00:20'] * 9)})
    stations = pd.DataFrame({'station_id': ['a'], 'capacity': [20]})
    labels, directions = features.capacity_labels(trips, short_window, stations,
                                                 return_direction=True)
    assert labels.tolist() == [1, 0, 0, 0]
    assert directions.tolist() == ['bicycles', 'none', 'none', 'none']


def test_no_events_with_valid_capacity_is_not_risk(short_window):
    trips = pd.DataFrame({'origin_id': ['b'], 'destination_id': ['b'],
                          'started_at': pd.to_datetime(['2025-01-01 00:10']),
                          'ended_at': pd.to_datetime(['2025-01-01 00:20'])})
    stations = pd.DataFrame({'station_id': ['a'], 'capacity': [20]})
    labels = features.capacity_labels(trips, short_window, stations)
    assert labels.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize('capacity', [0, -1, np.nan])
def test_invalid_catalogue_capacity_does_not_create_a_false_target(short_window, capacity):
    trips = pd.DataFrame({'origin_id': ['a'], 'destination_id': ['a'],
                          'started_at': pd.to_datetime(['2025-01-01 00:10']),
                          'ended_at': pd.to_datetime(['2025-01-01 00:20'])})
    stations = pd.DataFrame({'station_id': ['a'], 'capacity': [capacity]})
    labels, directions = features.capacity_labels(trips, short_window, stations,
                                                 return_direction=True)
    assert np.isnan(labels).all()
    assert set(directions) == {'capacity_missing'}
