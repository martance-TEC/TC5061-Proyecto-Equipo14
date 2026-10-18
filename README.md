# MiBici Equipo 14

Proyecto de Operaciones de aprendizaje automático. Martha Alicia Villalobos Facundo, Milton Vera de la Torre y Víctor Villa Herrera. Repositorio: https://github.com/martance-TEC/TC5061-Proyecto-Equipo14

# Descripción del proyecto

Este proyecto utiliza los datos abiertos de MiBici para analizar el comportamiento de los viajes y de las estaciones del sistema.

Se trabajan tres preguntas:

P1. Predicción de viajes: estimar cuántos viajes comenzarán en cada estación durante la siguiente hora.
P2. Riesgo de presión en estaciones: identificar si una estación podría acercarse a una situación de pocas bicicletas o pocos espacios disponibles durante las siguientes tres horas.
P3. Agrupación de estaciones: encontrar estaciones con patrones de uso similares.

El proyecto utiliza una estructura reproducible. Esto permite relacionar los datos utilizados, el código, los modelos y los resultados registrados en MLflow.


## Datos y partición
Doce CSV mensuales de 2025, más catálogo de estaciones septiembre de 2026 con dpcapacity. Ver references/sources.json, data_dictionary.md y data_manifest.json. Entrenamiento enero-agosto, validación septiembre-octubre, prueba noviembre-diciembre; Cuando una predicción necesita información de las horas siguientes, se evita que ese periodo pase de entrenamiento a validación o de validación a prueba. Los campos personales se excluyen. Ningún viaje se publica en Git. La ingesta deja 14 archivos procesados en Parquet con huella, tamaño y filas registrados. Con `--verify`, contrasta data_version con el manifiesto antes de reemplazar los existentes. P1 utiliza un grupo de 369 estaciones que tenían salidas registradas durante entrenamiento y que también se encuentran en el catálogo de estaciones. Se utiliza un calendario horario completo, donde un valor de cero significa que no se registraron viajes durante esa hora. Tras excluir las primeras 168 horas, P1 tiene 2,090,016 filas estación hora en entrenamiento, 540,216 en validación y 540,216 en prueba. P2 es un escenario retrospectivo con capacidad fija e inventario inicial hipotético; no predice faltantes reales observados.


## Estructura del proyecto
El repositorio sigue una estructura basada en Cookiecutter Data Science. La información original y procesada se guarda en data/, la lógica del proyecto se encuentra en mibici_equipo14/, las definiciones y referencias en references/ y los resultados en reports/.
config.py concentra la configuración, dataset.py prepara los datos, features.py construye las variables utilizadas por los modelos, train.py contiene los experimentos y pipeline.py coordina la ejecución.

## Instalación en Windows
El proyecto fue reproducido en Windows utilizando Python 3.11.9 de 64 bits.
Desde PowerShell, en la carpeta raíz del proyecto:

py -3.11-64 --version
py -3.11-64 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.0.1
.\.venv\Scripts\python.exe -m pip install setuptools==78.1.0 wheel==0.45.1
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
python -m pip check
python -m pytest -q


La descarga tarda según la red. Conserva las fuentes locales con sus hashes. Un error HTTP o una fuente modificada debe investigarse; no reemplazar datos silenciosamente. Para verificar copias ya descargadas: `python -m mibici_equipo14.dataset --verify`.

## Ejecución de experimentos con MLflow
En otra consola ubicada en la misma raíz, iniciar un servidor limitado a la laptop:

```powershell
$artifactPath = ([System.Uri](Join-Path $PWD 'mlartifacts')).AbsoluteUri
.\.venv\Scripts\mlflow.exe server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db --default-artifact-root $artifactPath
```

En la consola del proyecto:

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
$env:PYTHONIOENCODING = 'utf-8'
mlflow run . --env-manager local --experiment-name mibici-equipo14 -P question=all -P integrante=martha
```

MLproject permite question=all, P1, P2 o P3; integrante identifica a quien realmente ejecuta. max_iter=0 usa la configuración central 100 para P1; por ejemplo `-P question=P1 -P max_iter=150` registra la variación sin editar celdas. python_env.yaml y requirements.lock.txt documentan un entorno reconstruible; esta ejecución local usa el entorno previamente instalado y verificado. No requiere Conda. El pipeline crea una ejecución principal y ejecuciones separadas para cada modelo, lo que permite comparar sus parámetros y resultados en MLflow.

## Modelos y métricas
P1. Se comparan tres alternativas: un modelo semanal, un promedio histórico por estación, día y hora, e HistGradientBoosting. La selección se realiza con el MAE de validación. Cuando dos modelos tienen una diferencia menor o igual a 0,01, se prefiere el modelo más sencillo.

P2. Se compara un modelo base con Random Forest. Para el entrenamiento se utilizan como máximo 100.000 observaciones, manteniendo completos los conjuntos de validación y prueba. La métrica principal es F2, ya que interesa dar mayor importancia a la detección de posibles casos de riesgo.

P3. Se prueban agrupaciones con KMeans desde k=2 hasta k=6. La selección utiliza principalmente el silhouette de validación, además de revisar la estabilidad de los grupos a través del tiempo.

## Resultados y reproducción
Los resultados de cada integrante se mantienen separados para conservar la trazabilidad de las reproducciones. Martha registró las ejecuciones iniciales de P1, P2 y P3; Milton reprodujo la ingesta y P1; Víctor reprodujo P2 y P3 y posteriormente ejecutó P1 como parte de la revisión cruzada.

Las reproducciones realizadas utilizaron la misma data_version: 97b8d9d898ca5f71ffaffc6ff86b100fbe371abfd5d67837d989e5f81bf9934f

La ejecución de Víctor quedó vinculada al commit b0a6534. Las ejecuciones finales se realizaron con git_dirty=false, permitiendo relacionar los resultados de MLflow con una versión concreta del código.

En P1, la reproducción de Víctor confirmó los resultados de Milton: el promedio por estación, día de la semana y hora obtuvo val_mae=0.8639 y test_mae=0.8776, mientras que HistGradientBoosting obtuvo val_mae=0.8582. La diferencia en validación se encuentra dentro de la tolerancia de 0.01 definida previamente, por lo que se selecciona el modelo promedio.

En P2, Random Forest fue seleccionado frente al baseline, con test_f2=0.5627, test_recall=0.8098 y test_precision=0.2534. Estos resultados corresponden al proxy de presión definido por el proyecto y no representan observaciones reales de estaciones vacías o llenas.

En P3 se evaluaron valores de k entre 2 y 6. El procedimiento seleccionó k=3 sobre 369 estaciones elegibles, con val_silhouette=0.0735 y test_temporal_ari=0.8061.

Estos resultados corresponden a evaluaciones realizadas con los datos del proyecto y no representan directamente el impacto que tendrían los modelos en la operación real de MiBici

## Git y colaboración
El desarrollo se realizó mediante ramas individuales y Pull Requests para conservar la autoría y facilitar la revisión del código.

Las principales ramas de trabajo son:

feat/mavf-config-project: desarrollo base.
feat/milton-ingesta-p1: ingesta y reproducción de P1.
feat/victor-p2-p3: reproducción de P2/P3 y mejoras de reproducibilidad y documentación.

Los Pull Requests son revisados por otro integrante antes de integrarse. Cada integrante conserva la autoría de sus commits y no aprueba su propio Pull Request.

El registro adicional de colaboración se encuentra en references/collaboration.md.

## Limitaciones y mejora
El análisis tiene algunas limitaciones importantes:

La capacidad disponible corresponde al catálogo de 2026 y se utiliza para analizar viajes de 2025.
No se cuenta con el inventario histórico real de bicicletas por estación.
No se observan directamente las operaciones de rebalanceo.
No se mide la demanda que no pudo ser atendida.
No se incluyen variables meteorológicas.
P3 excluye estaciones con menos de 28 días activos durante entrenamiento.
Las primeras 168 horas no cuentan con todos los rezagos históricos necesarios.

Por estas razones, P2 debe interpretarse como una aproximación de presión sobre la capacidad y no como evidencia de que una estación estuvo realmente vacía o llena.

## Reproducción local en macOS

La ejecución de Milton usa MLflow local con base SQLite y artefactos locales, sobre el entorno `.venv` ya existente con Python 3.11 y los paquetes fijados en `requirements.lock.txt`. El lock incluye pywin32 con marcador de Windows, por lo que pip no lo instala en macOS.

Para reproducir en el clon actual, desde la raíz del repositorio:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m mibici_equipo14.dataset --verify
env MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow_milton.db" .venv/bin/python -m mibici_equipo14.pipeline --question P1 --integrante milton
```

pytest reporta 16 pruebas aprobadas. MLflow guarda los runs en `mlflow_milton.db` y los artefactos locales en `mlartifacts`. El experimento actual ya se creó con esa ruta de artefactos. En un clon con `.venv` recreado, el experimento debe crearse con Python antes de ejecutar, para fijar `artifact_location` en `mlartifacts`. El bloque lo crea solo si no existe, de modo que se puede repetir:

```bash
env MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow_milton.db" .venv/bin/python - <<'PY'
import pathlib
import mlflow

nombre = "mibici-equipo14"
if mlflow.get_experiment_by_name(nombre) is None:
    mlflow.create_experiment(
        nombre,
        artifact_location=pathlib.Path("mlartifacts").resolve().as_uri(),
    )
PY
```

El `.venv` actual se reconstruyó con uv, que crea el entorno sin pip, y esa es la reconstrucción validada. Para recrear el entorno con herramientas estándar:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

Los datos crudos y procesados quedan locales. En un clon remoto limpio, las fuentes se reconstruyen primero con `dataset`, sin `--verify` la primera vez, y después se verifican:

```bash
.venv/bin/python -m mibici_equipo14.dataset
.venv/bin/python -m mibici_equipo14.dataset --verify
```

## Referencias
MLflow. (s. f.). MLflow Projects. https://mlflow.org/docs/latest/ml/projects/
MLflow. (s. f.). MLflow example. https://github.com/mlflow/mlflow-example
DrivenData. (s. f.). Cookiecutter Data Science. https://cookiecutter-data-science.drivendata.org/
MiBici. (s. f.). Datos abiertos. https://mibici.net/es/datos-abiertos/
Python Software Foundation. (2024). Python 3.11.9. https://www.python.org/downloads/release/python-3119/
Real Python. (s. f.). Object oriented programming in Python. https://realpython.com/python3-object-oriented-programming/
Las lecturas institucionales y el podcast se encuentran en la consigna, pero no se citan como consultados íntegramente sin acceso verificable.


## Ejecución de Milton
Milton ejecutó en su equipo local la ingesta y P1 desde la rama feat/milton-ingesta-p1. La ingesta procesó 4,532,032 viajes de 13 fuentes y generó 14 archivos Parquet procesados. P1 evaluó tres modelos sobre una cohorte de 369 estaciones y seleccionó el promedio por estación, día de la semana y hora, con la regla de validación y tolerancia. Las 16 pruebas pasaron. Los resultados y su trazabilidad están en reports/ingesta_p1_milton.json y reports/reproduccion_milton.json, y los comandos de reproducción en macOS están en la sección Reproducción local en macOS. 

## Ejecución de Víctor
Víctor reprodujo localmente la ingesta y verificó 4,532,032 viajes con la misma data_version utilizada por el equipo. Durante la reproducción desde un clon limpio se identificó un error HTTP 406 en la descarga automática de las fuentes y se ajustó la identificación de la solicitud utilizada por dataset.py.

Posteriormente reprodujo P2 y P3 mediante MLflow con trazabilidad al commit b0a6534 y git_dirty=false. P2 seleccionó Random Forest con test_f2=0.5627 y test_recall=0.8098. P3 seleccionó KMeans con k=3 sobre 369 estaciones elegibles.

Como parte de la revisión cruzada, Víctor reprodujo también P1 y confirmó la selección del promedio histórico por estación, día de la semana y hora bajo la regla de tolerancia definida sobre MAE de validación. Las 16 pruebas automatizadas fueron ejecutadas correctamente.



## Resumen de resultados
P1	Promedio por estación, día y hora	Test MAE = 0,8776
P2	Random Forest	Test F2 = 0,5627; Recall = 0,8098
P3	KMeans (k=3)	369 estaciones; Test ARI = 0,8061


