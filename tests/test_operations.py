"""Controles del modelo exportado y del arranque repetible del Registry."""
import json
from unittest.mock import Mock, patch
import pytest
from mibici_equipo14.operations import prepare, bundle_hash, bootstrap


def test_export_detects_model_changes_and_does_not_retrain(tmp_path):
    first = prepare(tmp_path, smoke=True)
    assert first['synthetic_ci'] is True
    assert first['predictions'] == [1.5, 4., 1.5]
    assert prepare(tmp_path, smoke=True) == first
    (tmp_path / 'model' / 'extra.txt').write_text('changed')
    with pytest.raises(ValueError, match='Cambió'):
        prepare(tmp_path, smoke=True)


def test_bootstrap_reuses_matching_version_and_rejects_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv('BOOTSTRAP_SMOKE', '0')
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model' / 'MLmodel').write_text('test')
    digest = bundle_hash(tmp_path / 'model')
    (tmp_path / 'origin.json').write_text(json.dumps({
        'bundle_sha256': digest, 'synthetic_ci': False,
    }))
    client = Mock()
    version = Mock(version='1', tags={'bundle_sha256': digest})
    client.search_model_versions.return_value = [version]
    with patch('mibici_equipo14.operations.MlflowClient', return_value=client):
        bootstrap(tmp_path)
        client.create_run.assert_not_called()
        version.tags = {'bundle_sha256': 'different'}
        with pytest.raises(ValueError, match='otro modelo'):
            bootstrap(tmp_path)


def test_real_bootstrap_rejects_synthetic_export(tmp_path, monkeypatch):
    prepare(tmp_path, smoke=True)
    monkeypatch.setenv('BOOTSTRAP_SMOKE', '0')
    with pytest.raises(ValueError, match='tipo de datos'):
        bootstrap(tmp_path)
