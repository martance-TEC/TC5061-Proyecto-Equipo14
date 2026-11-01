"""Controles de la validación con ejemplos pequeños y errores conocidos."""
import numpy as np
import pandas as pd
import pytest

from mibici_equipo14.quality_data import (
    drift_check, integrity_suite, result_status, table_issues,
)
from mibici_equipo14.modeling.advanced import FEATURES


@pytest.fixture
def hourly_frame():
    frame = pd.DataFrame({c: [1., 2., 3.] for c in FEATURES})
    frame['station_id'] = ['a', 'a', 'a']
    frame['time'] = pd.date_range('2025-01-08', periods=3, freq='h')
    frame['departures'] = [0, 1, 2]
    frame['arrivals'] = [1, 0, 2]
    return frame


def test_valid_hourly_data_and_zero_counts(hourly_frame):
    assert table_issues(hourly_frame, 'p1', 'train') == []
    assert result_status(integrity_suite(hourly_frame, 'p1'))['passed']


@pytest.mark.parametrize('case,expected', [
    ('missing', 'Faltan columnas'), ('null', 'faltantes'),
    ('duplicate', 'duplicadas'), ('infinite', 'no finitos'),
    ('negative', 'negativos'), ('type', 'numérica'), ('future', 'temporal'),
])
def test_invalid_hourly_data_is_rejected(hourly_frame, case, expected):
    frame = hourly_frame.copy()
    if case == 'missing':
        frame = frame.drop(columns=['departures'])
    elif case == 'null':
        frame.loc[0, 'departures_lag1'] = np.nan
    elif case == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif case == 'infinite':
        frame.loc[0, 'departures_lag1'] = np.inf
    elif case == 'negative':
        frame.loc[0, 'departures'] = -1
    elif case == 'type':
        frame['departures'] = 'invalid'
    else:
        frame.loc[0, 'time'] = pd.Timestamp('2025-09-01')
    assert any(expected in issue for issue in table_issues(frame, 'p1', 'train'))


def test_deepchecks_detects_missing_values(hourly_frame):
    hourly_frame.loc[0, 'departures_lag1'] = np.nan
    status = result_status(integrity_suite(hourly_frame, 'p1'))
    assert not status['passed']
    assert not status['execution_errors']


def test_deepchecks_detects_duplicate_keys(hourly_frame):
    duplicate = pd.concat([hourly_frame, hourly_frame.iloc[:1]], ignore_index=True)
    assert not result_status(integrity_suite(duplicate, 'p1'))['passed']


def test_deepchecks_detects_a_known_distribution_shift():
    train = pd.DataFrame({'departures_lag1': np.linspace(0, 1, 200)})
    test = pd.DataFrame({'departures_lag1': np.linspace(10, 11, 200)})
    result = drift_check(train, test, 'p1')
    status = result_status(result)
    assert not status['passed']
    assert not status['execution_errors']


def test_empty_partition_is_rejected(hourly_frame):
    assert table_issues(hourly_frame.iloc[:0], 'p1', 'train')
