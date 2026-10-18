import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mibici_equipo14 import features, pipeline


def test_all_keeps_p2_p3_population_while_p1_uses_training_cohort(tmp_path, monkeypatch):
    settings = replace(features.S, root=tmp_path, start='2025-01-01',
        train_end='2025-01-10', val_end='2025-01-11', end='2025-01-12')
    monkeypatch.setattr(features, 'S', settings)
    monkeypatch.setattr(pipeline, 'S', settings)
    (tmp_path / 'references').mkdir()
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'references/data_manifest.json').write_text(json.dumps({'data_version': 'fixture'}))
    trips = pd.DataFrame({'origin_id': ['2', '3'], 'destination_id': ['2', '3'],
        'started_at': pd.to_datetime(['2025-01-01 10:00', '2025-01-11 10:00']),
        'ended_at': pd.to_datetime(['2025-01-01 10:30', '2025-01-11 10:30'])})
    stations = pd.DataFrame({'station_id': ['2', '3'], 'capacity': [20, 20]})
    monkeypatch.setattr(pipeline.pd, 'read_parquet',
        lambda p: trips if p.name == 'trips.parquet' else stations)
    seen = {}

    class Tracker:
        def __init__(self, *args):
            self.sha, self.dirty = 'unit-test', False

    class Regression:
        def run(self, frame, *args):
            seen['P1'] = set(frame.station_id)
            return {}

    class Classification:
        def run(self, frame, *args):
            seen['P2'] = set(frame.station_id)
            return {}

    class Clustering:
        def run(self, frame, *args):
            seen['P3'] = set(frame.station_id)
            return {}

    @contextmanager
    def run(**kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id='unit-test'))

    def labels(trips, hourly, stations, *args, **kwargs):
        zeros = np.zeros(len(hourly))
        return (zeros, np.repeat('none', len(hourly))) if kwargs.get('return_direction') else zeros

    for name, value in [('ExperimentTracker', Tracker), ('RegressionExperiment', Regression),
        ('ClassificationExperiment', Classification), ('ClusteringExperiment', Clustering),
        ('capacity_labels', labels)]:
        monkeypatch.setattr(pipeline, name, value)
    monkeypatch.setattr(pipeline.mlflow, 'start_run', run)
    monkeypatch.setattr(pipeline.mlflow, 'set_tag', lambda *a, **k: None)
    monkeypatch.setattr(pipeline.mlflow, 'log_dict', lambda *a, **k: None)
    pipeline.main('all', 'milton')
    assert seen == {'P1': {'2'}, 'P2': {'2', '3'}, 'P3': {'2', '3'}}
