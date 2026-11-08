"""Comprueba el modelo P1 del Registry antes de desplegar el servicio."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import mlflow.pyfunc
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from mlflow.types.schema import ColSpec, Schema

from .config import REGISTRY_NAMES


def inspect_model(version: int) -> dict:
    if version < 1:
        raise ValueError('La versión debe ser positiva')
    name = REGISTRY_NAMES['p1']
    client = MlflowClient()
    registered = client.get_model_version(name, str(version))
    model = mlflow.pyfunc.load_model(f'models:/{name}/{version}')
    schema = Schema([
        ColSpec('string', 'station_id'),
        ColSpec('integer', 'weekday'),
        ColSpec('integer', 'hour'),
    ])
    if model.metadata.get_input_schema() != schema:
        raise ValueError('La firma no coincide con el contrato público de P1')
    inputs = pd.DataFrame({
        'station_id': ['10', '20', '10'],
        'weekday': [2, 0, 2],
        'hour': [0, 8, 0],
    }).astype({'weekday': 'int32', 'hour': 'int32'})
    predictions = np.asarray(model.predict(inputs), dtype=float)
    if predictions.shape != (len(inputs),) or not np.isfinite(predictions).all():
        raise ValueError('El modelo no devuelve una predicción finita por fila')
    if (predictions < 0).any():
        raise ValueError('La demanda no puede ser negativa')
    if predictions[0] != predictions[2]:
        raise ValueError('La misma entrada produce resultados distintos')
    run = client.get_run(registered.run_id)
    return {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': name,
        'model_version': str(registered.version),
        'model_uri': f'models:/{name}/{version}',
        'run_id': registered.run_id,
        'experiment_id': run.info.experiment_id,
        'training_commit': run.data.tags.get('git_commit'),
        'data_version': run.data.tags.get('data_version'),
        'inputs': inputs.to_dict(orient='records'),
        'predictions': predictions.tolist(),
        'signature_verified': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-version', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('data/servicio/registry.json'))
    args = parser.parse_args()
    result = inspect_model(args.model_version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
