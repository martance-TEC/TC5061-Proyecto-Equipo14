"""Preparación del modelo y arranque del entorno local de operación."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
# Ejecutable resuelto y argumentos separados, sin shell.
import subprocess  # nosec B404
import tempfile
import time

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd

from .config import REGISTRY_NAMES, SETTINGS
from .modeling.wrappers import HistoricalMean

MODEL_NAME = REGISTRY_NAMES['p1']
EXAMPLE = pd.DataFrame({
    'station_id': ['10', '20', '10'], 'weekday': [2, 0, 2], 'hour': [0, 8, 0],
}).astype({'weekday': 'int32', 'hour': 'int32'})


def bundle_hash(path):
    """Huella de todos los archivos del modelo, con nombres relativos ordenados."""
    digest = hashlib.sha256()
    for item in sorted(Path(path).rglob('*')):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def prepare(output, source_uri=None, tracking_uri=None, smoke=False):
    """Exporta P1 del Registry o reconstruye su entrenamiento original."""
    output = Path(output)
    bundle = output / 'model'
    output.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        record = json.loads((output / 'origin.json').read_text())
        if record['bundle_sha256'] != bundle_hash(bundle):
            raise ValueError('Cambió el contenido del modelo exportado')
        if bool(record['synthetic_ci']) != smoke:
            raise ValueError('No se mezclan el modelo real y el ejemplo de CI')
        return record
    if source_uri:
        if not tracking_uri:
            raise ValueError('Falta la URI del Registry de origen')
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        metadata = mlflow.models.get_model_info(source_uri)
        downloaded = mlflow.artifacts.download_artifacts(artifact_uri=source_uri)
        shutil.copytree(downloaded, bundle)
        run = MlflowClient().get_run(metadata.run_id)
        origin = {'source_uri': source_uri, 'source_run_id': metadata.run_id,
                  'data_version': run.data.params.get('data_version') or run.data.tags.get('data_version'),
                  'training_commit': run.data.tags.get('mlflow.source.git.commit') or run.data.tags.get('git_commit')}
    else:
        if smoke:
            frame = EXAMPLE
            target = np.array([1., 4., 2.])
            origin = {'data_version': 'synthetic-ci-only'}
        else:
            from .dataset import ingest
            from .modeling.advanced import prepare as prepare_data
            from .tracking import git_state
            if not (SETTINGS.root / 'data/processed/ingestion_manifest.json').exists():
                ingest()
            parts, _, manifest = prepare_data('p1')
            frame = parts['train'][HistoricalMean.keys].astype(
                {'weekday': 'int32', 'hour': 'int32'})
            target = parts['train'].departures.to_numpy()
            origin = {'data_version': manifest['data_version'],
                      'training_commit': git_state()[0],
                      'train_start': str(parts['train'].time.min()),
                      'train_end_exclusive': SETTINGS.train_end,
                      'training_rows': len(frame)}
        model = HistoricalMean().fit(frame, target)
        mlflow.sklearn.save_model(
            model, str(bundle), signature=infer_signature(EXAMPLE, model.predict(EXAMPLE)),
            input_example=EXAMPLE, pip_requirements=['scikit-learn==1.5.2', 'pandas==2.2.3', 'numpy==1.26.4'],
        )
        origin['source_uri'] = 'reconstructed-original-p1' if not smoke else 'synthetic-ci-only'
    model = mlflow.pyfunc.load_model(str(bundle))
    result = {**origin, 'synthetic_ci': smoke, 'model_name': MODEL_NAME,
              'bundle_sha256': bundle_hash(bundle),
              'example': EXAMPLE.to_dict('records'),
              'predictions': model.predict(EXAMPLE).tolist()}
    (output / 'origin.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def define_operating_version(client, version, record):
    """Identifica la versión operativa local sin sustituir otra ya aprobada."""
    if record['synthetic_ci']:
        return
    aliases = client.get_registered_model(MODEL_NAME).aliases
    current = aliases.get('champion')
    if current is not None and str(current) != str(version):
        raise ValueError('champion apunta a otra versión; se requiere revisar la promoción')
    client.set_model_version_tag(MODEL_NAME, str(version), 'deployment_scope', 'academic_local')
    client.set_model_version_tag(MODEL_NAME, str(version), 'lifecycle_scheme', 'alias')
    if current is None:
        client.set_registered_model_alias(MODEL_NAME, 'champion', str(version))


def bootstrap(directory='/bootstrap'):
    """Importa una copia trazable al Registry del entorno, sin alterar el origen."""
    directory = Path(directory)
    smoke = os.getenv('BOOTSTRAP_SMOKE', '0') == '1'
    if smoke:
        directory = Path(tempfile.mkdtemp(prefix='mibici-smoke-'))
        prepare(directory, smoke=True)
    record = json.loads((directory / 'origin.json').read_text())
    if record['bundle_sha256'] != bundle_hash(directory / 'model'):
        raise ValueError('La huella del modelo no coincide')
    if bool(record['synthetic_ci']) != smoke:
        raise ValueError('El tipo de datos no coincide con el modo de despliegue')
    client = MlflowClient()
    for attempt in range(60):
        try:
            client.search_experiments(max_results=1)
            break
        except mlflow.exceptions.MlflowException:
            if attempt == 59:
                raise
            time.sleep(2)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        if len(versions) != 1 or versions[0].tags.get('bundle_sha256') != record['bundle_sha256']:
            raise ValueError('El Registry de operación ya contiene otro modelo. Conservar el volumen e investigar.')
        define_operating_version(client, versions[0].version, record)
        print(json.dumps({'model_version': versions[0].version, 'reused': True}), flush=True)
        return
    mlflow.set_experiment('mibici-operacion-ci' if smoke else SETTINGS.experiment)
    with mlflow.start_run(run_name='p1_importacion_operacion') as run:
        mlflow.set_tags({'equipo': '14', 'stage_of_work': 'operation',
                         'data_version': record['data_version'],
                         'synthetic_ci': str(smoke).lower(),
                         'source_uri': record['source_uri'],
                         'source_run_id': record.get('source_run_id', 'reconstructed'),
                         'source_commit': record.get('training_commit') or os.getenv('MIBICI_COMMIT', 'not_recorded')})
        mlflow.log_dict(record, 'config/origin.json')
        mlflow.log_artifacts(str(directory / 'model'), 'model')
        version = mlflow.register_model(f'runs:/{run.info.run_id}/model', MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, version.version, 'bundle_sha256', record['bundle_sha256'])
        client.set_model_version_tag(MODEL_NAME, version.version, 'synthetic_ci', str(smoke).lower())
        define_operating_version(client, version.version, record)
    print(json.dumps({'model_version': version.version, 'run_id': run.info.run_id,
                      'synthetic_ci': smoke}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'bootstrap', 'up'])
    parser.add_argument('--output', type=Path, default=Path('data/operacion'))
    parser.add_argument('--source-uri')
    parser.add_argument('--tracking-uri')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.action == 'bootstrap':
        bootstrap()
    else:
        prepare(args.output, args.source_uri, args.tracking_uri, args.smoke)
        if args.action == 'up':
            command = shutil.which('podman-compose')
            if command is None:
                raise RuntimeError('Instalar requirements-operation.txt y activar su entorno')
            # command se resuelve con which y los argumentos son fijos.
            subprocess.run([command, '-f', 'docker-compose.yml', 'up', '-d', '--build'], check=True)  # nosec B603


if __name__ == '__main__':
    main()
