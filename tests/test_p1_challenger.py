"""Contrato del challenger P1 y protección de los alias existentes."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mibici_equipo14.modeling import challenger
from mibici_equipo14.modeling.wrappers import HistoricalMean


def parts():
    def frame(times, values):
        time = pd.to_datetime(times)
        return pd.DataFrame({'time': time, 'station_id': ['10'] * len(time),
                             'weekday': time.dayofweek, 'hour': time.hour,
                             'departures': values})
    return {'train': frame(['2025-01-08', '2025-08-27'], [2, 4]),
            'val': frame(['2025-09-03', '2025-10-29'], [6, 8]),
            'test': object()}


def test_fit_uses_train_and_val_without_accessing_holdout():
    fit, detail = challenger.fit_frame(parts())
    model = HistoricalMean().fit(challenger.model_inputs(fit), fit.departures)
    assert len(fit) == 4
    assert model.predict(challenger.model_inputs(fit)).tolist() == [5.] * 4
    assert detail['source_rows'] == {'train': 2, 'val': 2}
    assert detail['fit_max_timestamp'] == '2025-10-29T00:00:00'
    assert detail['cohort_station_ids'] == ['10']


@pytest.mark.parametrize('split,date', [('train', '2025-09-01'), ('val', '2025-11-01'), ('val', '2025-08-31')])
def test_rejects_rows_outside_original_partitions(split, date):
    data = parts()
    data[split].loc[0, 'time'] = pd.Timestamp(date)
    with pytest.raises(ValueError, match='periodo'):
        challenger.fit_frame(data)


def test_rejects_new_stations_and_duplicates():
    data = parts()
    data['val'].loc[0, 'station_id'] = '999'
    with pytest.raises(ValueError, match='cohorte'):
        challenger.fit_frame(data)
    data = parts()
    data['val'] = pd.concat([data['val'], data['val']])
    with pytest.raises(ValueError, match='duplicadas'):
        challenger.fit_frame(data)


def test_signature_matches_serving():
    X = challenger.model_inputs(parts()['train'])
    assert X.columns.tolist() == HistoricalMean.keys
    assert X.dtypes.astype(str).tolist() == ['object', 'int32', 'int32']
    assert challenger.input_signature(X).inputs.to_dict() == [
        {'type': 'string', 'name': 'station_id', 'required': True},
        {'type': 'integer', 'name': 'weekday', 'required': True},
        {'type': 'integer', 'name': 'hour', 'required': True},
    ]


class Client:
    def __init__(self, aliases):
        self.aliases = aliases
        self.writes = []
    def get_registered_model(self, name):
        return SimpleNamespace(aliases=self.aliases.copy())
    def set_registered_model_alias(self, name, alias, version):
        self.writes.append((alias, version))
        self.aliases[alias] = version


def test_existing_challenger_is_preserved():
    client = Client({'champion': '1', 'challenger': '2'})
    with pytest.raises(ValueError, match='challenger'):
        challenger.available_aliases(client)
    assert client.writes == []


def test_publish_only_changes_challenger():
    client = Client({'champion': '1'})
    before = challenger.available_aliases(client)
    after = challenger.assign_challenger(client, '2', before)
    assert after == {'champion': '1', 'challenger': '2'}
    assert client.writes == [('challenger', '2')]


def test_concurrent_alias_change_stops_publication():
    client = Client({'champion': '1'})
    before = challenger.available_aliases(client)
    client.aliases['challenger'] = '3'
    with pytest.raises(ValueError, match='cambiaron'):
        challenger.assign_challenger(client, '2', before)
    assert client.writes == []


def test_requires_existing_champion():
    with pytest.raises(ValueError, match='champion'):
        challenger.available_aliases(Client({}))


def test_invalid_predictions_rejected():
    for predicted in ([np.nan], [-1.], [1., 2.]):
        with pytest.raises(ValueError, match='predicciones'):
            challenger.check_predictions(predicted, np.array([1.]))


def test_registry_round_trip_preserves_champion(tmp_path, monkeypatch):
    import mlflow
    from mlflow import MlflowClient
    import mlflow.sklearn

    uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        client = MlflowClient()
        data = parts()
        X = challenger.model_inputs(data['train'])
        champion = HistoricalMean().fit(X, data['train'].departures)
        model_path = tmp_path / 'champion'
        mlflow.sklearn.save_model(champion, str(model_path),
                                  signature=challenger.input_signature(X),
                                  pip_requirements=[])
        client.create_registered_model(challenger.NAME)
        version = client.create_model_version(challenger.NAME, model_path.as_uri())
        client.set_registered_model_alias(challenger.NAME, 'champion', version.version)
        monkeypatch.setattr(challenger, 'prepare_training_parts', lambda: (
            data, {'feature_version': 'fixture', 'fit_partition_sha256': {}}, {'data_version': 'fixture'}))
        monkeypatch.setattr(challenger, 'git_state', lambda: ('test-commit', False))
        result = challenger.run(uri, tmp_path / 'artifacts', tmp_path / 'evidence.json')
        assert result['aliases_after'] == {'champion': '1', 'challenger': '2'}
        assert result['prediction_check']['passed'] is True
        assert result['prediction_check']['fallback_passed'] is True
        assert client.get_run(result['run_id']).data.metrics == {}
        assert mlflow.pyfunc.load_model(f'models:/{challenger.NAME}@champion').predict(X).tolist() == [3., 3.]
        assert mlflow.pyfunc.load_model(f'models:/{challenger.NAME}@challenger').predict(X).tolist() == [5., 5.]
        with pytest.raises(ValueError, match='challenger'):
            challenger.run(uri, tmp_path / 'artifacts', tmp_path / 'second.json')
        assert len(client.search_model_versions(f"name='{challenger.NAME}'")) == 2
    finally:
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)


def test_training_loader_reads_only_train_and_val(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    monkeypatch.setattr(challenger, 'S', replace(challenger.S, root=tmp_path))
    monkeypatch.setattr(challenger, 'manifest', lambda: {'data_version': 'fixture'})
    feature = tmp_path / 'mibici_equipo14/features.py'
    feature.parent.mkdir()
    feature.write_text('# fixture')
    cache = tmp_path / 'data/interim/fixture/p1'
    cache.mkdir(parents=True)
    (cache / 'metadata.json').write_text(json.dumps({
        'data_version': 'fixture', 'feature_version': challenger.sha256(feature)}))
    data = parts()
    for split in ['train', 'val']:
        data[split].to_parquet(cache / f'{split}.parquet')
    holdout = cache / 'test.parquet'
    holdout.write_bytes(b'holdout must not be read as parquet')
    before = holdout.read_bytes()
    loaded, metadata, source = challenger.prepare_training_parts()
    assert set(loaded) == {'train', 'val'}
    assert len(challenger.fit_frame(loaded)[0]) == 4
    assert holdout.read_bytes() == before
    assert set(metadata['fit_partition_sha256']) == {'train', 'val'}
