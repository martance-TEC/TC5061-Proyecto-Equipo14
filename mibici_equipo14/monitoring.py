"""Compara la referencia de P1 con un periodo posterior, sin tocar el holdout."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
# Solo se consulta Git local con argumentos fijos y sin shell.
import subprocess  # nosec B404

os.environ.setdefault('DO_NOT_TRACK', '1')

import numpy as np
import pandas as pd

from .config import SETTINGS as S
from .dataset import sha256
from .features import p1_hourly_counts, temporal_features

INPUTS = ['station_id', 'weekday', 'hour']
HISTORY = [f'{kind}_{suffix}' for kind in ['departures', 'arrivals']
           for suffix in ['lag1', 'lag24', 'lag168', 'mean24', 'mean168']]
NUMERICAL = ['departures'] + HISTORY
CATEGORICAL = INPUTS + ['month']
COLUMNS = INPUTS + NUMERICAL + ['month']
OUTPUT = S.root / 'reports' / 'monitoreo'


def prepare_periods(trips, stations):
    """Mantiene la cohorte original y los rezagos; excluye noviembre-diciembre.

    Las primeras 168 horas no tienen toda la historia y se excluyen igual que
    al entrenar P1. Los ceros de la malla horaria se conservan. La referencia
    termina en train_end y el periodo actual termina en val_end.
    """
    trips = trips[(trips.started_at >= S.start) & (trips.started_at < S.val_end)]
    hourly = p1_hourly_counts(trips, stations)
    hourly = hourly[hourly.time < S.val_end]
    frame = temporal_features(hourly)
    warmup = frame[HISTORY].isna().any(axis=1)
    eligible = frame.loc[~warmup].copy()
    reference = eligible[eligible.time < S.train_end].copy()
    current = eligible[eligible.time >= S.train_end].copy()
    if reference.empty or current.empty:
        raise ValueError('Los dos periodos deben contener observaciones')
    return reference, current, int(warmup.sum())


def sample_period(frame, size, seed):
    """Muestra aleatoria sin reemplazo, repetible y ordenada por estación-hora."""
    if size < 100:
        raise ValueError('La muestra debe tener al menos 100 observaciones')
    return frame.sample(n=min(size, len(frame)), random_state=seed).sort_index()


def quality_summary(frame):
    """Revisa toda la malla, usando estación-hora como clave de duplicados."""
    invalid = {
        'station_id': int(frame.station_id.isna().sum() +
                          frame.station_id.astype(str).str.strip().eq('').sum()),
        'weekday': int((~frame.weekday.between(0, 6)).sum()),
        'hour': int((~frame.hour.between(0, 23)).sum()),
        'departures': int(((frame.departures < 0) |
                           (frame.departures % 1 != 0)).sum()),
    }
    return {
        'rows': len(frame), 'stations': int(frame.station_id.nunique()),
        'first_hour': str(frame.time.min()), 'last_hour': str(frame.time.max()),
        'missing': {c: int(frame[c].isna().sum()) for c in COLUMNS},
        'non_finite_numeric': int((~np.isfinite(frame[NUMERICAL])).sum().sum()),
        'duplicate_station_hours': int(frame.duplicated(['station_id', 'time']).sum()),
        'invalid_values': invalid,
        'departures_total': int(frame.departures.sum()),
        'departures_mean': float(frame.departures.mean()),
        'zero_departures_share': float(frame.departures.eq(0).mean()),
    }


def compare_segments(reference, current):
    """Compara promedios, no totales de periodos de distinta duración."""
    rows = []
    for column in ['station_id', 'hour', 'weekday']:
        means = pd.concat([reference.groupby(column).departures.mean().rename('reference_mean'),
                           current.groupby(column).departures.mean().rename('current_mean')], axis=1)
        means['difference'] = means.current_mean - means.reference_mean
        for key, row in means.iterrows():
            rows.append({'segment': column, 'value': str(key),
                         'reference_mean': float(row.reference_mean),
                         'current_mean': float(row.current_mean),
                         'difference': float(row.difference)})
    return rows


def run_evidently(reference, current, output):
    """Usa la API vigente; las distancias son tamaños de cambio, no p-valores."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    schema = DataDefinition(numerical_columns=NUMERICAL,
                            categorical_columns=CATEGORICAL)
    ref = Dataset.from_pandas(reference[COLUMNS], data_definition=schema)
    cur = Dataset.from_pandas(current[COLUMNS], data_definition=schema)
    report = Report([
        DataDriftPreset(columns=COLUMNS, num_method='wasserstein',
                        cat_method='jensenshannon', threshold=0.1),
        DataSummaryPreset(columns=COLUMNS),
    ])
    result = report.run(cur, ref)
    result.save_html(str(output / 'evidently_drift.html'))
    raw = json.loads(result.json())
    write_json(output / 'evidently_metrics.json', raw)
    return raw


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding='utf-8')


def verify_inputs():
    """Valida las huellas de la ingesta antes de comparar periodos."""
    expected = json.loads((S.root / 'references/data_manifest.json').read_text())
    local = json.loads((S.root / 'data/processed/ingestion_manifest.json').read_text())
    if expected['data_version'] != local['data_version']:
        raise ValueError('La versión local de los datos no coincide con el manifiesto')
    for name in ['trips.parquet', 'stations.parquet']:
        item = next(x for x in local['processed_files'] if Path(x['path']).name == name)
        if sha256(S.root / item['path']) != item['sha256']:
            raise ValueError(f'Los datos procesados cambiaron: {name}')
    return local['data_version']


def source_commit():
    """Identifica el código ejecutado sin incluir rutas locales en el reporte."""
    git = shutil.which('git')
    if git is None:
        raise RuntimeError('Git no está disponible')
    # La ruta ejecutable se resuelve localmente; los argumentos son constantes.
    return subprocess.check_output([git, 'rev-parse', 'HEAD'], cwd=S.root, text=True).strip()  # nosec B603


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size', type=int, default=50000)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    data_version = verify_inputs()
    trips = pd.read_parquet(S.root / 'data/processed/trips.parquet',
                            filters=[('started_at', '<', pd.Timestamp(S.val_end))])
    stations = pd.read_parquet(S.root / 'data/processed/stations.parquet')
    reference, current, warmup = prepare_periods(trips, stations)
    args.output.mkdir(parents=True, exist_ok=True)
    sampled = [sample_period(f, args.sample_size, S.seed) for f in [reference, current]]
    print('Ejecutando Evidently sobre muestras de ambos periodos...', flush=True)
    raw = run_evidently(*sampled, args.output)
    summary = {
        'question': 'p1', 'model_name': 'mibici-equipo14-p1-demanda',
        'data_version': data_version,
        'reference_period': [S.start, S.train_end],
        'current_period': [S.train_end, S.val_end],
        'holdout_excluded': [S.val_end, S.end],
        'period_convention': 'inicio inclusivo, fin exclusivo',
        'warmup_rows_excluded': warmup,
        'quality': {'reference': quality_summary(reference),
                    'current': quality_summary(current)},
        'sampling': {'method': 'uniform_without_replacement', 'seed': S.seed,
                     'requested_rows_per_period': args.sample_size,
                     'reference_rows': len(sampled[0]), 'current_rows': len(sampled[1])},
        'methods': {'numerical': 'wasserstein', 'categorical': 'jensenshannon',
                    'threshold': 0.1, 'dataset_drift_share': 0.5},
        'input_columns': INPUTS, 'context_columns': NUMERICAL + ['month'],
        'feature_sha256': sha256(S.root / 'mibici_equipo14/features.py'),
        'monitoring_sha256': sha256(Path(__file__)),
        'source_commit': source_commit(),
    }
    observed = trips[(trips.started_at >= S.train_end) & (trips.started_at < S.val_end)]
    outside = ~observed.origin_id.isin(reference.station_id.unique())
    summary['coverage'] = {
        'current_trips_all_stations': len(observed),
        'current_trips_outside_training_cohort': int(outside.sum()),
        'current_origin_stations_outside_cohort': int(observed.loc[outside, 'origin_id'].nunique()),
        'scope': 'malla de la cohorte original; no equivale a disponibilidad real de bicicletas',
    }
    summary['segments'] = compare_segments(reference, current)
    from importlib.metadata import version
    summary['environment'] = {'python': platform.python_version(), 'evidently': version('evidently'),
                               'dependencies': 'requirements-monitoring.lock.txt'}
    summarize_metrics(raw, summary)
    write_json(args.output / 'resumen_drift.json', summary)
    make_plots(reference, current, summary, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize_metrics(raw, summary):
    """Extrae los resultados por variable del reporte de Evidently."""
    results = []
    for metric in raw['metrics']:
        config = metric.get('config', {})
        if config.get('type', '').endswith('ValueDrift'):
            column = config['column']
            score = float(metric['value'])
            results.append({'column': column, 'score': score,
                            'method': 'jensenshannon' if column in CATEGORICAL else 'wasserstein',
                            'threshold': 0.1, 'drift_detected': score >= 0.1,
                            'role': 'model_input' if column in INPUTS else 'context'})
    if len(results) != len(COLUMNS):
        raise ValueError('El reporte no contiene todas las variables de monitoreo')
    summary['variables'] = results
    for label, cols in [('model_inputs', INPUTS), ('context_without_month', NUMERICAL)]:
        selected = [r for r in results if r['column'] in cols]
        count = sum(r['drift_detected'] for r in selected)
        summary[label] = {'drifted': count, 'total': len(selected),
                          'share': count / len(selected),
                          'dataset_drift': count / len(selected) >= 0.5}
    count = sum(r['drift_detected'] for r in results)
    summary['all_monitored_columns'] = {'drifted': count, 'total': len(results),
                                        'share': count / len(results),
                                        'dataset_drift': count / len(results) >= 0.5}


def make_plots(reference, current, summary, output):
    """Gráficas de resultados calculados; no representan capturas de pantalla."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = summary['variables']
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([x['column'] for x in values], [x['score'] for x in values],
            color=['#b94a48' if x['drift_detected'] else '#277c8e' for x in values])
    ax.axvline(.1, color='black', linestyle='--', label='Umbral de alerta: 0.10')
    ax.invert_yaxis()
    ax.set(xlabel='Distancia por variable (métodos distintos según tipo)',
           title='P1: cambios entre enero-agosto y septiembre-octubre de 2025')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output / '01_drift_variables.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for frame, label in [(reference, 'Enero-agosto'), (current, 'Septiembre-octubre')]:
        axes[0].plot(frame.groupby('hour').departures.mean(), label=label)
        daily = frame.groupby(frame.time.dt.date).departures.mean()
        axes[1].plot(pd.to_datetime(daily.index), daily.values, label=label)
    axes[0].set(xlabel='Hora', ylabel='Salidas por estación-hora', title='Perfil horario observado')
    axes[1].set(xlabel='Fecha', ylabel='Salidas por estación-hora', title='Promedio diario de demanda')
    axes[1].tick_params(axis='x', rotation=25)
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / '02_demanda_periodos.png', dpi=180)
    plt.close(fig)
    stations = sorted([r for r in summary['segments'] if r['segment'] == 'station_id'],
                       key=lambda r: r['difference'])
    selected = stations[:5] + stations[-5:]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([r['value'] for r in selected], [r['difference'] for r in selected], color='#277c8e')
    ax.axvline(0, color='black', linewidth=.8)
    ax.set(xlabel='Cambio en salidas medias por estación-hora', ylabel='Estación',
           title='Cinco mayores descensos y cinco mayores aumentos')
    fig.tight_layout()
    fig.savefig(output / '03_cambios_estaciones.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
