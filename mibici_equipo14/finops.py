"""Resumen reproducible de Locust, recursos y costo estimado por inferencia."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRICE_SOURCE = 'https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-bundles.html'


def estimate_cost(seconds, successes, monthly_usd=24., planning_hours=730.):
    """Asigna el costo de toda la instancia al intervalo y a respuestas exitosas."""
    if seconds <= 0 or successes <= 0 or monthly_usd <= 0 or planning_hours <= 0:
        raise ValueError('Duración, éxitos, tarifa y horas deben ser positivos')
    window = monthly_usd / planning_hours * seconds / 3600
    return {'window_usd': window, 'per_inference_usd': window / successes,
            'per_1000_usd': window / successes * 1000}


def memory_mib(value):
    amount, unit = re.fullmatch(r'([\d.]+)\s*([A-Za-z]+)', value.split('/')[0].strip()).groups()
    factors = {'B': 1, 'kB': 1000, 'KB': 1000, 'MB': 1e6, 'GB': 1e9,
               'KiB': 1024, 'MiB': 1024 ** 2, 'GiB': 1024 ** 3}
    return float(amount) * factors[unit] / 1024 ** 2


def cpu_seconds(value):
    factors = {'h': 3600, 'm': 60, 's': 1, 'ms': .001, 'µs': 1e-6, 'ns': 1e-9}
    return sum(float(amount) * factors[unit]
               for amount, unit in re.findall(r'([\d.]+)(ms|µs|ns|h|m|s)', value))


def resource_frame(samples):
    rows = []
    for sample in samples:
        for item in sample['containers']:
            rows.append({'timestamp': sample['timestamp'], 'service': 'api' if 'api' in item['name'] else 'mlflow',
                         'memory_mib': memory_mib(item['mem_usage']),
                         'cpu_seconds': cpu_seconds(item['cpu_time'])})
    frame = pd.DataFrame(rows).sort_values(['service', 'timestamp'])
    frame['cpu_percent'] = frame.groupby('service').cpu_seconds.diff() / frame.groupby('service').timestamp.diff() * 100
    return frame


def summarize(directory, monthly_usd=24.):
    directory = Path(directory)
    execution = json.loads((directory / 'execution.json').read_text())
    results = []
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    fig_resources, resource_axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for run in execution['runs']:
        users = run['users']
        stats = pd.read_csv(directory / f'locust_{users}_stats.csv')
        row = stats.loc[stats.Name == 'Aggregated'].iloc[0]
        attempts, failures = int(row['Request Count']), int(row['Failure Count'])
        duration = attempts / float(row['Requests/s'])
        resources = resource_frame(json.loads((directory / f'resources_{users}.json').read_text()))
        by_service = {}
        for name, group in resources.groupby('service'):
            total_time = group.timestamp.max() - group.timestamp.min()
            mean_cpu = (group.cpu_seconds.iloc[-1] - group.cpu_seconds.iloc[0]) / total_time * 100
            by_service[name] = {'mean_cpu_percent_one_core': mean_cpu,
                                'peak_memory_mib': float(group.memory_mib.max())}
        combined = resources.groupby('timestamp').memory_mib.sum()
        result = {'users': users, 'requests': attempts, 'failures': failures,
                  'successes': attempts - failures, 'error_percent': failures / attempts * 100,
                  'requests_per_second': float(row['Requests/s']), 'locust_seconds': duration,
                  'mean_ms': float(row['Average Response Time']), 'p50_ms': float(row['50%']),
                  'p95_ms': float(row['95%']), 'p99_ms': float(row['99%']),
                  'resources': by_service, 'peak_combined_memory_mib': float(combined.max()),
                  'estimated_cost': estimate_cost(duration, attempts - failures, monthly_usd)}
        results.append(result)
        history = pd.read_csv(directory / f'locust_{users}_stats_history.csv')
        history = history.loc[history.Name == 'Aggregated']
        elapsed = history.Timestamp - history.Timestamp.min()
        axes[0].plot(elapsed, history['Requests/s'], label=f'{users} usuarios')
        axes[1].plot(elapsed, pd.to_numeric(history['95%'], errors='coerce'), label=f'{users} usuarios')
        axes[2].plot(elapsed, history['Failures/s'], label=f'{users} usuarios')
        api = resources.loc[resources.service == 'api']
        elapsed_resources = api.timestamp - resources.timestamp.min()
        resource_axes[0].plot(elapsed_resources, api.cpu_percent, label=f'{users} usuarios')
        total = resources.groupby('timestamp').memory_mib.sum()
        resource_axes[1].plot(total.index - total.index.min(), total, label=f'{users} usuarios')
    for ax, label in zip(axes, ['Peticiones por segundo', 'Latencia p95 (ms)', 'Fallos por segundo']):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(loc='upper left')
    axes[2].set_xlabel('Segundos desde el inicio de cada escenario (incluye rampa)')
    fig.suptitle('Locust: solicitudes válidas a POST /predict')
    fig.savefig(directory / '01_locust.png', dpi=170)
    plt.close(fig)
    for ax, label in zip(resource_axes, ['CPU API (% de un núcleo)', 'Memoria API + MLflow (MiB)']):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(loc='upper left')
    resource_axes[-1].set_xlabel('Segundos desde el primer muestreo')
    fig_resources.suptitle('Recursos del servicio durante la carga')
    fig_resources.savefig(directory / '02_recursos.png', dpi=170)
    plt.close(fig_resources)
    output = {'execution_utc': execution['utc'], 'model': 'mibici-equipo14-p1-demanda/1',
              'price': {'monthly_usd': monthly_usd, 'planning_hours_per_month': 730,
                        'reference': 'Lightsail Linux/Unix IPv4, 2 vCPU, 4 GB, 80 GB SSD',
                        'source': PRICE_SOURCE, 'consulted': '2026-11-21'},
              'method': 'Carga local ARM64. Proyección económica, no benchmark cloud ni factura.',
              'cpu_method': 'Diferencia del tiempo acumulado de CPU / intervalo de muestreo; 100% equivale a un núcleo.',
              'runs': results,
              'monthly_demand_scenarios': [
                  {'inferences_per_month': count, 'monthly_usd': monthly_usd,
                   'per_1000_usd': monthly_usd / count * 1000}
                  for count in [10000, 100000, 1000000]],
              'price_sensitivity': [
                  {'monthly_usd': rate,
                   'per_1000_usd_at_largest_local_load': estimate_cost(
                       results[-1]['locust_seconds'], results[-1]['successes'], rate)['per_1000_usd']}
                  for rate in [12, 24, 44]],
              'exclusions': ['impuestos', 'egreso adicional al paquete', 'respaldos y retención extra',
                             'balanceador/TLS', 'CI/CD', 'operación humana', 'entrenamientos futuros'],
              'limitations': ['Una repetición breve por carga, sin prueba de saturación ni duración prolongada.',
                              'Los contenedores comparten la VM. Locust corre en el anfitrión.',
                              'La rampa está incluida. No se afirma capacidad equivalente en AWS.',
                              'El arranque e importación del modelo se midieron aparte de la carga.']}
    (directory / 'summary.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='reports/operacion')
    parser.add_argument('--monthly-usd', type=float, default=24.)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.monthly_usd), indent=2, ensure_ascii=False))
