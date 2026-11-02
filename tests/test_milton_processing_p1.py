"""Casos de ingesta y respaldo del promedio histórico de P1"""
import numpy as np
import pandas as pd
import pytest

from mibici_equipo14.dataset import canonical_column, trip_chunks
from mibici_equipo14.modeling.wrappers import HistoricalMean


@pytest.mark.parametrize(('original', 'expected'), [
    ('\ufeff VÍaje - ID ', 'viaje_id'),
    ('Inicio del viaje', 'inicio_del_viaje'),
    ('DESTINO...ID', 'destino_id'),
])
def test_column_names_are_normalized(original, expected):
    assert canonical_column(original) == expected


def test_trip_chunks_preserve_identifier_strings_and_leading_zeroes(tmp_path):
    path = tmp_path / 'viajes.csv'
    path.write_text(
        'Víaje ID,Inicio del viaje,Fin del viaje,Origen ID,Destino ID,Usuario ID\n'
        '00012,2025-01-01 08:00:00,2025-01-01 08:10:00,007,0009,privado\n',
        encoding='utf-8',
    )
    frame = pd.concat(list(trip_chunks(path, 'utf-8')), ignore_index=True)
    assert list(frame.columns) == [
        'trip_id', 'started_at', 'ended_at', 'origin_id', 'destination_id',
    ]
    assert frame.loc[0, ['trip_id', 'origin_id', 'destination_id']].tolist() == [
        '00012', '007', '0009',
    ]


@pytest.mark.parametrize('missing', [
    'viaje_id', 'inicio_del_viaje', 'fin_del_viaje', 'origen_id', 'destino_id',
])
def test_missing_operational_column_is_reported(tmp_path, missing):
    path = tmp_path / 'incompleto.csv'
    required = ['viaje_id', 'inicio_del_viaje', 'fin_del_viaje', 'origen_id', 'destino_id']
    columns = [name for name in required if name != missing]
    path.write_text(','.join(columns) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Missing operational columns') as error:
        list(trip_chunks(path, 'utf-8'))
    assert missing in str(error.value)
    assert path.name in str(error.value)


def test_p1_station_without_history_uses_global_training_mean():
    training = pd.DataFrame({
        'station_id': ['a', 'a', 'b'],
        'weekday': [0, 0, 0],
        'hour': [8, 8, 8],
    })
    model = HistoricalMean().fit(training, np.array([2., 4., 12.]))
    evaluation = pd.DataFrame({
        'station_id': ['a', 'b', 'nueva'],
        'weekday': [0, 0, 0],
        'hour': [8, 8, 8],
    })
    np.testing.assert_array_equal(model.predict(evaluation), [3., 12., 6.])
