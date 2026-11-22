"""Ejecuta Locust y muestrea CPU y memoria de los servicios durante cada carga."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(output, url, engine, connection, users, seconds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(engine)
    if executable is None:
        raise RuntimeError(f'No se encontró {engine}')
    command = [executable] + (['--connection', connection] if connection else [])
    runs = []
    for count in users:
        prefix = output / f'locust_{count}'
        started = time.time()
        timestamps = []
        with (output / f'locust_{count}.log').open('w') as log:
            process = subprocess.Popen([
                sys.executable, '-m', 'locust', '-f', 'locustfile.py', '--headless',
                '--host', url, '--users', str(count), '--spawn-rate', str(max(1, count // 5)),
                '--run-time', f'{seconds}s', '--stop-timeout', '10',
                '--csv', str(prefix), '--csv-full-history', '--html', f'{prefix}.html',
            ], stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    sample = subprocess.run(command + [
                        'stats', '--no-stream', '--format=json',
                    ], capture_output=True, text=True, check=True, timeout=20)
                    rows = json.loads(sample.stdout)
                    selected = [{k: v for k, v in row.items()
                                 if k.lower().replace('_', '') in {
                                     'name', 'cpu', 'cpupercent', 'cputime', 'memusage',
                                     'memperc', 'cpup', 'mempercent'}}
                                for row in rows if any(
                                    name in row.get('name', '').lower().replace('_', '-')
                                    for name in ['mibici-operacion-api', 'mibici-operacion-mlflow'])]
                    if not selected:
                        raise RuntimeError('No se encontraron los contenedores de operación en stats')
                    timestamps.append({'timestamp': time.time(), 'containers': selected})
                    time.sleep(3)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=20)
        if process.returncode != 0:
            raise RuntimeError(f'Locust terminó con código {process.returncode}; revisar {prefix}.log')
        (output / f'resources_{count}.json').write_text(json.dumps(timestamps, indent=2))
        runs.append({'users': count, 'configured_seconds': seconds,
                     'elapsed_seconds': time.time() - started, 'samples': len(timestamps)})
        (output / 'execution.json').write_text(json.dumps({
            'utc': datetime.now(timezone.utc).isoformat(), 'engine': engine,
            'one_inference_per_request': True, 'ramp_included': True,
            'wait_time_seconds': [0.2, 0.8], 'runs': runs,
        }, indent=2))
    return runs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/operacion')
    parser.add_argument('--url', default='http://127.0.0.1:8014')
    parser.add_argument('--engine', choices=['podman', 'docker'], default='podman')
    parser.add_argument('--connection')
    parser.add_argument('--users', type=int, nargs='+', default=[5, 20, 50])
    parser.add_argument('--seconds', type=int, default=90)
    args = parser.parse_args()
    if min(args.users) <= 0 or args.seconds < 10:
        parser.error('Usuarios positivos y duración mínima de 10 segundos')
    print(json.dumps(run(args.output, args.url, args.engine, args.connection, args.users, args.seconds)))
