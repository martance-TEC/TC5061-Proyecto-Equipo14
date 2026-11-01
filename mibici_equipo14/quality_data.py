"""Integridad de las particiones y cambios de distribución, sin reentrenar."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from .config import SETTINGS as S

os.environ.setdefault('DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY', '1')

SAMPLE_ROWS = 10000
DRIFT_THRESHOLD = 0.2


def table_issues(frame, question, split):
    """Revisa todas las filas. Los ceros válidos no se consideran faltantes."""
    from .modeling.advanced import FEATURES, P2_COLUMNS

    if question == 'p3':
        required = [f'{kind}_{i}' for kind in ('departures', 'arrivals') for i in range(48)]
        required += ['log_volume', 'weekend_share', 'balance']
        keys = None
    else:
        required = ['station_id', 'time', 'departures', 'arrivals']
        required += list(P2_COLUMNS if question == 'p2' else FEATURES)
        if question == 'p2':
            required += ['risk']
        keys = ['station_id', 'time']
    required = list(dict.fromkeys(required))
    issues = []
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        return [f'Faltan columnas: {", ".join(missing)}']
    if frame.empty:
        return ['La partición está vacía']
    if frame[required].isna().any().any():
        issues.append('Hay valores faltantes en columnas obligatorias')
    if keys:
        if frame.duplicated(keys).any():
            issues.append('Hay claves estación-hora duplicadas')
        if not is_datetime64_any_dtype(frame.time):
            issues.append('time debe tener tipo datetime')
        else:
            bounds = {'train': (S.start, S.train_end), 'val': (S.train_end, S.val_end),
                      'test': (S.val_end, S.end)}
            lo, hi = map(pd.Timestamp, bounds[split])
            horizon = S.horizon_hours if question == 'p2' else 1
            invalid = (frame.time < lo) | (frame.time + pd.Timedelta(hours=horizon) > hi)
            if invalid.any():
                issues.append('El horizonte cruza el límite de su partición temporal')
        if frame.station_id.astype('string').str.strip().eq('').any():
            issues.append('Hay identificadores de estación vacíos')
    elif frame.index.has_duplicates or frame.index.hasnans:
        issues.append('El índice de estaciones debe ser único y no nulo')
    numeric = [c for c in required if c not in ('station_id', 'time')]
    for column in numeric:
        if not is_numeric_dtype(frame[column]):
            issues.append(f'{column} debe ser numérica')
        elif not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            issues.append(f'{column} contiene valores no finitos')
    if issues:
        return issues
    nonnegative = [c for c in numeric if c != 'balance']
    if (frame[nonnegative] < 0).any().any():
        issues.append('Hay valores negativos en conteos o variables no negativas')
    if question == 'p2':
        if (frame.capacity <= 0).any():
            issues.append('La capacidad debe ser positiva')
        if not frame.risk.isin([0, 1]).all():
            issues.append('risk debe contener solamente 0 y 1')
    if question == 'p3':
        if not frame.weekend_share.between(0, 1).all():
            issues.append('weekend_share debe estar entre 0 y 1')
        if not frame.balance.between(-1, 1).all():
            issues.append('balance debe estar entre -1 y 1')
    return issues


def integrity_suite(frame, question):
    """Deepchecks comprueba faltantes y claves; los perfiles repetidos son válidos."""
    from deepchecks.tabular import Dataset, Suite
    from deepchecks.tabular.checks import DataDuplicates, PercentOfNulls

    sample = frame.sample(n=min(len(frame), SAMPLE_ROWS), random_state=S.seed)
    if question == 'p3':
        sample = sample.reset_index()
        keys = [sample.columns[0]]
    else:
        keys = ['station_id', 'time']
    suite = Suite('Integridad',
                  PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0),
                  DataDuplicates(columns=keys).add_condition_ratio_less_or_equal(0))
    categorical = [c for c in ('station_id', 'hour', 'weekday', 'month', 'risk') if c in sample]
    return suite.run(Dataset(sample, cat_features=categorical,
                             datetime_name='time' if 'time' in sample else None))


def drift_check(train, test, question):
    """KS para numéricas y Cramér V para calendario; alerta a partir de 0.2."""
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import FeatureDrift

    # Mes e identificadores no describen drift de una variable de uso.
    columns = [c for c in train.select_dtypes(include='number')
               if c not in ('month', 'risk') and c in test.columns]
    categorical = [c for c in ('hour', 'weekday') if c in columns]
    datasets = []
    for frame in (train, test):
        sample = frame[columns].sample(n=min(len(frame), SAMPLE_ROWS), random_state=S.seed)
        datasets.append(Dataset(sample, cat_features=categorical))
    check = FeatureDrift(random_state=S.seed)
    check.add_condition_drift_score_less_than(max_allowed_numeric_score=DRIFT_THRESHOLD,
                                              max_allowed_categorical_score=DRIFT_THRESHOLD)
    return check.run(*datasets)


def result_status(result):
    """Un fallo de ejecución nunca cuenta como una condición aprobada."""
    from deepchecks.core import CheckFailure

    results = result.results if hasattr(result, 'results') else [result]
    errors = [str(r) for r in results if isinstance(r, CheckFailure)]
    failed = [r.header for r in results if not isinstance(r, CheckFailure)
              and not r.passed_conditions()]
    return {'execution_errors': errors, 'failed_conditions': failed,
            'passed': not errors and not failed}


def run(output):
    from .dataset import sha256
    from .modeling.advanced import prepare

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = S.root / 'data/processed/ingestion_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    reference = json.loads((S.root / 'references/data_manifest.json').read_text(encoding='utf-8'))
    if manifest['data_version'] != reference['data_version']:
        raise ValueError('La versión local de datos no coincide con la referencia')
    for entry in manifest['processed_files']:
        path = S.root / entry['path']
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError(f'Archivo procesado ausente o modificado: {entry["path"]}')
    summary = {'data_version': manifest['data_version'], 'seed': S.seed,
               'sample_limit': SAMPLE_ROWS, 'drift_threshold': DRIFT_THRESHOLD,
               'integrity_scope': 'Todas las filas con reglas de dominio; Deepchecks con muestra fija',
               'questions': {}, 'passed': True}
    for question in ('p1', 'p2', 'p3'):
        print(f'Validando {question.upper()}', flush=True)
        parts, meta, _ = prepare(question)
        item = {'rows': meta['rows'], 'partitions': {}}
        for split, frame in parts.items():
            issues = table_issues(frame, question, split)
            record = {'issues': issues, 'checked_rows': len(frame)}
            if not issues:
                result = integrity_suite(frame, question)
                result.save_as_html(str(output / f'{question}_{split}_integrity.html'), as_widget=False)
                record['deepchecks'] = result_status(result)
            item['partitions'][split] = record
            if issues or not record.get('deepchecks', {}).get('passed', False):
                summary['passed'] = False
        if all(not p['issues'] for p in item['partitions'].values()):
            result = drift_check(parts['train'], parts['test'], question)
            result.save_as_html(str(output / f'{question}_drift.html'), as_widget=False)
            item['drift'] = result_status(result)
            item['drift']['scores'] = result.value
            if item['drift']['execution_errors']:
                summary['passed'] = False
        summary['questions'][question] = item
        del parts
    (output / 'data_checks.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print('Integridad:', 'APROBADA' if summary['passed'] else 'CON ERRORES', flush=True)
    print('Las alertas de drift se conservan para revisión; no se cambian las particiones.')
    return summary['passed']
