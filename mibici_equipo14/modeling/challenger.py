"""Entrena HistoricalMean P1 con train + val y registra el alias challenger."""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd

from ..config import SETTINGS as S, REGISTRY_NAMES
from ..dataset import sha256
from ..tracking import git_state
from .advanced import manifest, read_json, write_json
from .wrappers import HistoricalMean

NAME = REGISTRY_NAMES['p1']
EXPERIMENT = 'mibici-equipo14-p1-challenger'


def prepare_training_parts():
    """Carga train y val de la preparacion oficial de P1."""
    source = manifest()
    cache = S.root / 'data/interim' / source['data_version'] / 'p1'
    metadata_path = cache / 'metadata.json'
    if not metadata_path.exists():
        raise ValueError("Ejecuta prepare('p1') antes de entrenar el challenger")
    metadata = read_json(metadata_path)
    if (metadata.get('data_version') != source['data_version'] or
            metadata.get('feature_version') != sha256(S.root / 'mibici_equipo14/features.py')):
        raise ValueError("Actualiza las particiones con prepare('p1')")
    paths = {split: cache / f'{split}.parquet' for split in ['train', 'val']}
    metadata = {**metadata, 'fit_partition_sha256': {s: sha256(p) for s, p in paths.items()}}
    return {s: pd.read_parquet(p) for s, p in paths.items()}, metadata, source


def model_inputs(frame):
    """Mantiene los tipos y el orden usados por la API P1."""
    return frame[HistoricalMean.keys].astype(
        {'station_id': str, 'weekday': 'int32', 'hour': 'int32'})


def input_signature(example):
    return infer_signature(example, np.zeros(len(example), dtype='float64'))


def fit_frame(parts):
    """Combina las particiones originales conservando la cohorte de train."""
    bounds = {'train': (S.start, S.train_end), 'val': (S.train_end, S.val_end)}
    selected = []
    for split, (start, end) in bounds.items():
        frame = parts[split]
        if frame.empty or not ((frame.time >= start) &
                               (frame.time + pd.Timedelta(hours=1) <= end)).all():
            raise ValueError(f'{split}: filas fuera del periodo original')
        selected.append(frame)
    cohort = set(parts['train'].station_id)
    if set(parts['val'].station_id) != cohort:
        raise ValueError('La cohorte de val debe coincidir con train')
    fit = pd.concat(selected, ignore_index=True).sort_values(['station_id', 'time'])
    if fit.duplicated(['station_id', 'time']).any():
        raise ValueError('Hay filas estación/hora duplicadas')
    if not np.isfinite(fit.departures).all() or (fit.departures < 0).any():
        raise ValueError('Las salidas deben ser finitas y no negativas')
    detail = {
        'fit_start_inclusive': S.start, 'fit_end_exclusive': S.val_end,
        'fit_min_timestamp': fit.time.min().isoformat(),
        'fit_max_timestamp': fit.time.max().isoformat(),
        'source_rows': {s: len(parts[s]) for s in bounds}, 'fit_rows': len(fit),
        'cohort_station_ids': sorted(cohort), 'cohort_stations': len(cohort),
        'cohort_selection_end_exclusive': S.train_end,
        'holdout_start_inclusive': S.val_end, 'holdout_end_exclusive': S.end,
    }
    return fit, detail


def read_aliases(client):
    return {key: str(value) for key, value in client.get_registered_model(NAME).aliases.items()}


def available_aliases(client):
    aliases = read_aliases(client)
    if 'champion' not in aliases:
        raise ValueError('Se requiere un champion existente en este Registry')
    if 'challenger' in aliases:
        raise ValueError('Ya existe challenger. Revisa esa versión antes de continuar')
    return aliases


def assign_challenger(client, version, before):
    if read_aliases(client) != before:
        raise ValueError('Los alias cambiaron durante el entrenamiento')
    client.set_registered_model_alias(NAME, 'challenger', version)
    after = read_aliases(client)
    if after != {**before, 'challenger': str(version)}:
        raise RuntimeError('Verifica los alias del Registry antes de continuar')
    return after


def check_predictions(predicted, expected):
    values = np.asarray(predicted)
    if (values.shape != expected.shape or not np.isfinite(values).all()
            or (values < 0).any()
            or not np.allclose(values, expected, atol=1e-8, rtol=0)):
        raise ValueError('Las predicciones cargadas difieren del modelo entrenado')
    return values.tolist()


def run(tracking_uri, artifact_root, output):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    before = available_aliases(client)
    parts, metadata, source = prepare_training_parts()
    fit, detail = fit_frame(parts)
    X = model_inputs(fit)
    model = HistoricalMean().fit(X, fit.departures)
    # Muestra de ambos periodos y una estacion nueva para comprobar el fallback
    example = pd.concat([model_inputs(parts['train']).head(3),
                         model_inputs(parts['val']).tail(3)], ignore_index=True)
    unknown = example.iloc[[0]].copy()
    unknown['station_id'] = '__unknown_station__'
    example = pd.concat([example, unknown], ignore_index=True)
    expected = model.predict(example)
    signature = input_signature(example)
    champion = mlflow.pyfunc.load_model(f'models:/{NAME}/{before["champion"]}')
    if champion.metadata.signature.to_dict() != signature.to_dict():
        raise ValueError('La firma del challenger debe coincidir con champion')
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (experiment.experiment_id if experiment else
                     client.create_experiment(EXPERIMENT,
                         artifact_location=Path(artifact_root).resolve().as_uri()))
    sha, dirty = git_state()
    code_files = ['mibici_equipo14/modeling/challenger.py',
                  'mibici_equipo14/modeling/wrappers.py',
                  'mibici_equipo14/modeling/advanced.py',
                  'mibici_equipo14/features.py', 'mibici_equipo14/config.py',
                  'requirements.lock.txt']
    evidence = {
        'model_name': NAME, 'model_class': 'HistoricalMean', 'alias': 'challenger',
        'git_commit': sha, 'git_dirty': dirty,
        'code_sha256': {p: sha256(S.root / p) for p in code_files},
        'data_version': source['data_version'], 'feature_version': metadata['feature_version'],
        'fit_partition_sha256': metadata['fit_partition_sha256'],
        'signature': signature.to_dict(), 'fit': detail,
        'evaluation_scope': 'serialization_and_prediction_contract',
        'tracking_registry_same_backend': True, 'aliases_before': before,
    }
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name='eq14_p1_historical_mean_challenger') as active:
        evidence['run_id'] = active.info.run_id
        mlflow.set_tags({'equipo': '14', 'pregunta': 'P1', 'integrante': 'Milton',
                         'model_family': 'historical_mean', 'role': 'challenger',
                         'git_commit': sha, 'git_dirty': str(dirty).lower(),
                         'data_version': source['data_version']})
        mlflow.log_params({'fit_start': S.start, 'fit_end_exclusive': S.val_end,
                           'original_train_end': S.train_end, 'fit_rows': len(fit),
                           'cohort_stations': detail['cohort_stations'], 'seed': S.seed})
        info = mlflow.sklearn.log_model(
            sk_model=model, name='model', signature=signature, input_example=example,
            code_paths=[str(S.root / 'mibici_equipo14')],
            pip_requirements=str(S.root / 'requirements.lock.txt'))
        evidence['model_uri'] = info.model_uri
        loaded = mlflow.sklearn.load_model(info.model_uri)
        if not isinstance(loaded, HistoricalMean):
            raise TypeError('El artefacto debe contener HistoricalMean')
        check_predictions(loaded.predict(example), expected)
        # El alias se escribe despues de comprobar la version registrada
        if read_aliases(client) != before:
            raise ValueError('Los alias cambiaron durante el entrenamiento')
        version = mlflow.register_model(info.model_uri, NAME)
        evidence['version'] = str(version.version)
        uri = f'models:/{NAME}/{version.version}'
        registered = mlflow.pyfunc.load_model(uri)
        if registered.metadata.signature.to_dict() != signature.to_dict():
            raise ValueError('La versión registrada perdió su firma')
        predictions = check_predictions(registered.predict(example), expected)
        evidence['prediction_check'] = {
            'input': example.to_dict('records'), 'output': predictions,
            'absolute_tolerance': 1e-8, 'passed': True,
            'fallback_passed': bool(np.isclose(predictions[-1], model.fallback_)),
        }
        for key, value in {'git_commit': sha, 'data_version': source['data_version'],
                           'model_family': 'historical_mean', 'fit_end_exclusive': S.val_end,
                           'integrante': 'Milton', 'validation_status': 'contract_passed'}.items():
            client.set_model_version_tag(NAME, version.version, key, value)
        evidence['aliases_after'] = assign_challenger(client, str(version.version), before)
        mlflow.log_dict(evidence, 'challenger/evidence.json')
        write_json(output, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking-uri', required=True)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--output', default='reports/challenger_p1/evidence.json')
    args = parser.parse_args()
    result = run(args.tracking_uri, args.artifact_root, args.output)
    print(json.dumps({k: result[k] for k in ['run_id', 'version', 'aliases_after']}, indent=2))


if __name__ == '__main__':
    main()
