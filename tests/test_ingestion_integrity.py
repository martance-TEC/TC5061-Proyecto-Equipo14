import json
from dataclasses import replace

import pandas as pd
import pytest

from mibici_equipo14 import dataset


def ingestion_fixture(tmp_path, monkeypatch, *, variant_headers=False):
    monkeypatch.setattr(dataset, 'S', replace(dataset.S, root=tmp_path))
    raw = tmp_path / 'data/raw'
    refs = tmp_path / 'references'
    raw.mkdir(parents=True)
    refs.mkdir()
    trips = pd.DataFrame({
        'Viaje_Id': ['001', '001', '002'],
        'Inicio_del_viaje': ['2025-01-01 10:00:00'] * 3,
        'Fin_del_viaje': ['2025-01-01 10:15:00', '2025-01-01 10:15:00', 'bad'],
        'Origen_Id': ['02'] * 3,
        'Destino_Id': ['03'] * 3,
        'Usuario_Id': ['private'] * 3,
        'Género': ['M'] * 3,
    })
    if variant_headers:
        trips = trips.rename(columns={
            'Viaje_Id': 'Víaje Id', 'Inicio_del_viaje': ' Inicio del viaje ',
            'Fin_del_viaje': 'FIN_DEL_VIAJE', 'Origen_Id': 'Origen Id',
            'Destino_Id': 'Destino-Id',
        })
    path = raw / 'datos_abiertos_2025_01.csv'
    trips.to_csv(path, index=False, encoding='cp1252')
    pd.DataFrame({'id': ['02', '03'], 'name': ['a', 'b'],
                  'latitude': [20., 20.1], 'longitude': [-103., -103.1],
                  'dpcapacity': [10, 10]}).to_csv(raw / 'estaciones.csv', index=False)
    sources = [{'name': path.name, 'url': 'https://example.invalid/trips.csv'},
               {'name': 'estaciones.csv', 'url': 'https://example.invalid/stations.csv'}]
    (refs / 'sources.json').write_text(json.dumps(sources))
    return path


def test_variant_headers_encoding_privacy_and_identifiers(tmp_path, monkeypatch):
    ingestion_fixture(tmp_path, monkeypatch, variant_headers=True)
    manifest = dataset.ingest(verify=True)
    trips = pd.read_parquet(tmp_path / 'data/processed/trips.parquet')
    assert list(trips.columns) == ['trip_id', 'started_at', 'ended_at', 'origin_id', 'destination_id']
    assert trips.trip_id.tolist() == ['001']
    assert trips.origin_id.tolist() == ['02']
    assert manifest['duplicates_removed'] == 1
    assert manifest['invalid_rows'] == 1


def test_manifest_verifies_processed_bytes_and_rows(tmp_path, monkeypatch):
    ingestion_fixture(tmp_path, monkeypatch)
    manifest = dataset.ingest(verify=True)
    outputs = {entry['path']: entry for entry in manifest['processed_files']}
    output = tmp_path / 'data/processed/trips.parquet'
    entry = outputs['data/processed/trips.parquet']
    assert entry['sha256'] == dataset.sha256(output)
    assert entry['bytes'] == output.stat().st_size
    assert entry['rows'] == 1


def test_changed_source_is_rejected_before_processed_files_are_replaced(tmp_path, monkeypatch):
    source = ingestion_fixture(tmp_path, monkeypatch)
    dataset.ingest(verify=True)
    output = tmp_path / 'data/processed/trips.parquet'
    previous_hash = dataset.sha256(output)
    with source.open('a', encoding='cp1252') as handle:
        handle.write('003,2025-01-01 11:00:00,2025-01-01 11:20:00,02,03,private,M\n')
    with pytest.raises((ValueError, AssertionError), match='version|source|Source|mismatch'):
        dataset.ingest(verify=True)
    assert dataset.sha256(output) == previous_hash
