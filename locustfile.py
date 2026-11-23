"""Carga de predicciones P1: una inferencia por solicitud válida."""
import math
from locust import HttpUser, between, task


class PredictionUser(HttpUser):
    wait_time = between(0.2, 0.8)

    def on_start(self):
        self.sequence = 0

    @task
    def predict(self):
        row = {'station_id': ['10', '20', '30'][self.sequence % 3],
               'weekday': self.sequence % 7, 'hour': self.sequence % 24}
        self.sequence += 1
        with self.client.post('/predict', json={'instances': [row]},
                              name='/predict', catch_response=True, timeout=10) as response:
            if response.status_code != 200:
                response.failure(f'HTTP {response.status_code}')
                return
            try:
                values = response.json()['predictions']
                if (len(values) != 1 or isinstance(values[0], bool)
                        or not math.isfinite(values[0]) or values[0] < 0):
                    response.failure('Predicción inválida')
            except (ValueError, KeyError, TypeError):
                response.failure('Contrato de respuesta inválido')
