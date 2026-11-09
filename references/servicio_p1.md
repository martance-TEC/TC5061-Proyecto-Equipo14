# API de predicción P1

El servicio calcula la demanda por estación, día de la semana y hora con el
modelo P1 registrado en MLflow. FastAPI recibe las solicitudes y Pydantic
valida los datos de entrada.

## Entrada y respuesta

`POST /predict` recibe un objeto con `instances`, una lista de 1 a 1,000
filas. Cada fila tiene tres campos obligatorios:

- `station_id`: texto con al menos un carácter distinto de espacio. Conserva sus ceros iniciales.
- `weekday`: entero de 0 a 6. El lunes corresponde a 0.
- `hour`: entero de 0 a 23.

La validación es estricta. Los campos enteros deben llegar como enteros y cada
fila debe contener únicamente los tres campos del contrato. Las entradas
inválidas reciben HTTP 422.

La respuesta contiene una lista `predictions` con un valor por fila, en el
mismo orden de la solicitud. Los promedios pueden tener decimales. La API valida que haya un valor finito y no negativo por fila; una salida inválida produce HTTP 500.

`GET /health` devuelve `status`, `model_name` y `model_version` para consultar
el estado del servicio y el modelo cargado.

## Configuración y ejecución

El servicio usa Python 3.11 y las versiones de `requirements.lock.txt`.
FastAPI, Pydantic, Uvicorn, MLflow, pandas y scikit-learn están incluidos en
ese archivo. El README del proyecto describe el entorno y la configuración
de tracking y Registry.

El modelo debe estar disponible en Registry con esta firma, en este orden:
`station_id:string`, `weekday:integer` y `hour:integer`. Los dos campos
enteros se convierten a int32 antes de la predicción.

Desde la raíz del repositorio, con el entorno activo:

```bash
python -m mibici_equipo14.serving
```

La configuración inicial usa `127.0.0.1:8000`, el modelo
`mibici-equipo14-p1-demanda` definido en `REGISTRY_NAMES['p1']` y el alias
`champion`.

Las opciones de ejecución son:

- `--model-name`: nombre del modelo registrado.
- `--model-version` o `--model-alias`: selección del modelo por versión o alias.
- `--tracking-uri` y `--registry-uri`: direcciones de los servicios de MLflow.
- `--host` y `--port`: dirección y puerto de la API.

Las URI también pueden configurarse con `MLFLOW_TRACKING_URI` y
`MLFLOW_REGISTRY_URI`.

## Petición

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"station_id":"10","weekday":2,"hour":0}]}'
```

Esta petición solicita la demanda de la estación 10 para el miércoles a las
00:00. La respuesta incluye un valor en la lista `predictions`.

## Pruebas

```bash
MLFLOW_DISABLE_TELEMETRY=true python -m pytest -q tests/test_serving.py
```

Las tres pruebas comprueban el orden y los valores de las predicciones, la
validación de entradas y la firma del modelo. Usan `HistoricalMean` guardado
y cargado con MLflow, con la consulta al Registry simulada.

## Referencias

- FastAPI. (s. f.). *Request body*. https://fastapi.tiangolo.com/tutorial/body/
- MLflow. (s. f.). *MLflow Model Registry*. https://mlflow.org/docs/latest/ml/model-registry/
- Pydantic. (s. f.). *Strict mode*. https://docs.pydantic.dev/latest/concepts/strict_mode/
