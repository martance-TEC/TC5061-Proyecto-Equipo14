"""Comprueba la API desplegada contra el modelo servido por el Registry."""
import argparse
import json
import mlflow
import mlflow.pyfunc
import numpy as np
import requests
from .operations import EXAMPLE, MODEL_NAME


def check(url, tracking_uri):
    health = requests.get(f'{url}/health', timeout=10)
    health.raise_for_status()
    info = health.json()
    if info != {'status': 'ready', 'model_name': MODEL_NAME, 'model_version': '1'}:
        raise ValueError('El servicio cargó otro modelo')
    response = requests.post(f'{url}/predict', json={'instances': EXAMPLE.to_dict('records')}, timeout=10)
    response.raise_for_status()
    actual = np.asarray(response.json()['predictions'], dtype=float)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    expected = mlflow.pyfunc.load_model(f'models:/{MODEL_NAME}/1').predict(EXAMPLE)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
    codes = []
    for row in [{'station_id': '10', 'hour': 0},
                {'station_id': '10', 'weekday': 7, 'hour': 0},
                {'station_id': '10', 'weekday': 0, 'hour': 24}]:
        code = requests.post(f'{url}/predict', json={'instances': [row]}, timeout=10).status_code
        if code != 422:
            raise ValueError(f'Entrada inválida recibió HTTP {code}')
        codes.append(code)
    return {'health': info, 'predictions': actual.tolist(),
            'max_absolute_difference': float(np.max(np.abs(actual - expected))),
            'invalid_http_codes': codes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8014')
    parser.add_argument('--tracking-uri', default='http://127.0.0.1:5015')
    args = parser.parse_args()
    print(json.dumps(check(args.url, args.tracking_uri), indent=2))
