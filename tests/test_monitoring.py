"""Pruebas de periodos, calidad y cálculo real de Evidently."""
import json

import numpy as np
import pandas as pd
import pytest

from mibici_equipo14.monitoring import (
    COLUMNS, HISTORY, INPUTS, prepare_periods, quality_summary,
    compare_segments, run_evidently, sample_period, summarize_metrics,
)


def example_frame(size=200):
    time = pd.date_range('2025-08-01', periods=size, freq='h')
    frame = pd.DataFrame({'station_id': ['1'] * size, 'time': time,
                          'weekday': time.dayofweek, 'hour': time.hour,
                          'month': time.month, 'departures': np.arange(size) % 7})
    for column in HISTORY:
        frame[column] = np.arange(size) % 7
    return frame


def test_periods_preserve_cohort_and_exclude_holdout():
    trips = pd.DataFrame({
        'origin_id': ['1', '1', '2', '1'],
        'destination_id': ['1'] * 4,
        'started_at': pd.to_datetime(['2025-01-01', '2025-09-01', '2025-10-01', '2025-11-01']),
        'ended_at': pd.to_datetime(['2025-01-01 00:10', '2025-09-01 00:10',
                                     '2025-10-01 00:10', '2025-11-01 00:10']),
    })
    reference, current, warmup = prepare_periods(trips, pd.DataFrame({'station_id': ['1', '2']}))
    assert warmup == 168
    assert reference.time.min() == pd.Timestamp('2025-01-08')
    assert reference.time.max() < pd.Timestamp('2025-09-01')
    assert current.time.min() == pd.Timestamp('2025-09-01')
    assert current.time.max() == pd.Timestamp('2025-10-31 23:00')
    assert set(reference.station_id) == set(current.station_id) == {'1'}
    assert current.departures.sum() == 1
    assert not reference[HISTORY].isna().any().any()


def test_lags_use_only_previous_hours():
    trips = pd.DataFrame({'origin_id': ['1', '1'], 'destination_id': ['1', '1'],
                          'started_at': pd.to_datetime(['2025-01-01 00:00', '2025-09-01 01:00']),
                          'ended_at': pd.to_datetime(['2025-01-01 00:10', '2025-09-01 01:10'])})
    _, current, _ = prepare_periods(trips, pd.DataFrame({'station_id': ['1']}))
    at_event = current[current.time == '2025-09-01 01:00'].iloc[0]
    after = current[current.time == '2025-09-01 02:00'].iloc[0]
    assert at_event.departures == 1 and at_event.departures_lag1 == 0
    assert after.departures_lag1 == 1


def test_sampling_is_repeatable_and_without_replacement():
    frame = example_frame()
    first = sample_period(frame, 100, 42)
    pd.testing.assert_frame_equal(first, sample_period(frame, 100, 42))
    assert first.index.is_unique and len(first) == 100
    assert len(sample_period(frame, 300, 42)) == len(frame)
    with pytest.raises(ValueError):
        sample_period(frame, 99, 42)


def test_quality_distinguishes_repeated_values_from_duplicate_keys():
    frame = example_frame()
    assert quality_summary(frame)['duplicate_station_hours'] == 0
    broken = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    broken.loc[1, 'hour'] = 24
    broken.loc[2, 'departures'] = -1
    broken.loc[3, 'departures_lag1'] = np.nan
    result = quality_summary(broken)
    assert result['duplicate_station_hours'] == 1
    assert result['invalid_values']['hour'] == 1
    assert result['invalid_values']['departures'] == 1
    assert result['missing']['departures_lag1'] == 1


def test_evidently_detects_shift_and_writes_report(tmp_path):
    pytest.importorskip('evidently')
    reference = example_frame()
    current = reference.copy()
    current['departures'] += 30
    raw = run_evidently(reference, current, tmp_path)
    summary = {}
    summarize_metrics(raw, summary)
    results = {row['column']: row for row in summary['variables']}
    assert results['departures']['drift_detected']
    assert results['departures']['score'] > .1
    assert not summary['model_inputs']['dataset_drift']
    assert set(results) == set(COLUMNS)
    assert set(INPUTS) <= set(results)
    assert (tmp_path / 'evidently_drift.html').stat().st_size > 1000
    assert json.loads((tmp_path / 'evidently_metrics.json').read_text())['metrics']


def test_summary_rejects_incomplete_report():
    with pytest.raises(ValueError):
        summarize_metrics({'metrics': []}, {})


def test_segment_comparison_normalizes_different_period_lengths():
    reference = example_frame()
    current = pd.concat([reference, reference], ignore_index=True)
    rows = compare_segments(reference, current)
    assert all(row['difference'] == 0 for row in rows)
    current['departures'] += 1
    rows = compare_segments(reference, current)
    assert all(row['difference'] == pytest.approx(1) for row in rows)
