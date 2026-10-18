# MiBici Equipo 14

Proyecto de Operaciones de aprendizaje automático. Martha Alicia Villalobos Facundo, Milton Vera de la Torre y Víctor Villa Herrera. Repositorio: https://github.com/martance-TEC/TC5061-Proyecto-Equipo14

## Estado de la entrega
Los PR remotos 1, 2 y 3 son de Martha. La implementación base se encuentra en el PR 1, pendiente de revisión humana: https://github.com/martance-TEC/TC5061-Proyecto-Equipo14/pull/1 . Hasta que se apruebe e integre, main conserva solamente el inicio de la plantilla. Para obtener el proyecto ejecutable de Martha usa `git clone --branch feat/mavf-config-project https://github.com/martance-TEC/TC5061-Proyecto-Equipo14.git`. La rama docs/mavf-entrega agrega los resultados y evidencias posteriores a su ejecución.

La contribución local de Milton, con ingesta y P1, está en la rama feat/milton-ingesta-p1, creada sobre docs/mavf-revision-ingesta-p1. Sus cambios entran al proyecto mediante un PR revisado por otra persona del equipo, y quien lo abre no lo aprueba.

Dos ejecuciones locales completas de Martha procesaron 4,532,032 viajes y produjeron las mismas 91 métricas con tolerancia 1e-12. El commit, la versión de datos y el run padre de esas ejecuciones están en reports/reproduction_audit.json y reports/reproduccion_martha.json, en la rama documental. En esos reportes, P1 seleccionó HistGradientBoosting con test_mae 0.8051 frente a 0.9904 del baseline. P2 seleccionó RandomForest con test_f2 0.5627 y test_average_precision 0.3924, bajo la etiqueta proxy definida. P3 seleccionó k=3 sobre 369 estaciones elegibles.

La ejecución local de Milton evalúa P1 con tres modelos (semanal, promedio por estación, día de la semana y hora, y HistGradientBoosting Poisson) sobre una cohorte de 369 estaciones con salidas en entrenamiento y entrada en el catálogo. La selección por validación con tolerancia de 0.01 eligió el promedio, con val_mae 0.8639 y test_mae 0.8776. La ingesta generó 14 archivos Parquet procesados y pasaron 16 pruebas. La trazabilidad de esta ejecución está en reports/reproduccion_milton.json.

En el cuadro actual de colaboración, la cuenta verificada es MiltonVeraTorre y su contribución local es la ingesta y P1. El PR de esa contribución requiere la revisión de otro integrante del equipo antes de integrarse.

## Problema y usuarios
P1: regresión del número de viajes iniciados por estación durante la siguiente hora, para el operador del sistema. P2: clasificación de presión aproximada sobre la capacidad en las siguientes tres horas, para rebalanceo. P3: clustering de perfiles históricos por estación y su localización, para planeación urbana y expansión. Las decisiones completas, baselines y métricas están en references/problem_definition.md y el ML Canvas en references/ml_canvas.md.

## Datos y partición
Doce CSV mensuales de 2025, más catálogo de estaciones septiembre de 2026 con dpcapacity. Ver references/sources.json, data_dictionary.md y data_manifest.json. Entrenamiento enero-agosto, validación septiembre-octubre, prueba noviembre-diciembre; se purgan horizontes que cruzan fronteras. Los campos personales se excluyen. Ningún viaje se publica en Git. La ingesta deja 14 archivos procesados en Parquet con huella, tamaño y filas registrados. Con `--verify`, contrasta data_version con el manifiesto antes de reemplazar los existentes. P1 usa una cohorte de 369 estaciones con salidas en entrenamiento y entrada en el catálogo, sobre un calendario horario completo donde un cero significa cero viajes registrados en los archivos. Tras excluir las primeras 168 horas, P1 tiene 2,090,016 filas estación hora en entrenamiento, 540,216 en validación y 540,216 en prueba. P2 es un escenario retrospectivo con capacidad fija e inventario inicial hipotético; no predice faltantes reales observados.

## Estructura CCDS y OOP
Generado con ccds 2.3.0; procedencia en references/template_provenance.json. data/ contiene fuentes y Parquet ignorados; notebooks/ solo exploración; mibici_equipo14/ la lógica final; references/ definiciones y arquitectura; reports/ resúmenes y figuras; models/ ignorado. Settings centraliza configuración. ExperimentTracker gestiona trazabilidad; RegressionExperiment, ClassificationExperiment y ClusteringExperiment encapsulan sus experimentos sin jerarquías innecesarias. Las funciones de features.py son reutilizables y comprobables.

## Instalación Windows con Python 3.11.9 de 64 bits
Ejecutar desde la raíz del clon, en PowerShell:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.0.1
.\.venv\Scripts\python.exe -m pip install setuptools==78.1.0 wheel==0.45.1
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
python -m pip check
python -m pytest -q
python -m mibici_equipo14.dataset
```

La descarga tarda según la red. Conserva las fuentes locales con sus hashes. Un error HTTP o una fuente modificada debe investigarse; no reemplazar datos silenciosamente. Para verificar copias ya descargadas: `python -m mibici_equipo14.dataset --verify`.

## MLflow y ejecución de un comando
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

MLproject permite question=all, P1, P2 o P3; integrante identifica a quien realmente ejecuta. max_iter=0 usa la configuración central 100 para P1; por ejemplo `-P question=P1 -P max_iter=150` registra la variación sin editar celdas. python_env.yaml y requirements.lock.txt documentan un entorno reconstruible; esta ejecución local usa el entorno previamente instalado y verificado. No requiere Conda. El orquestador se reincorpora al run padre de Projects y crea runs descriptivos para cada modelo.

## Modelos y métricas
P1 compara el modelo semanal, el promedio por estación, día de la semana y hora, y HistGradientBoosting Poisson. La selección usa val_mae: entre los modelos a 0.01 viajes o menos del mejor, gana el de menor complejidad con prioridad semanal, promedio y boosting. Se reportan MAE, RMSE y WAPE en prueba. P2 prior contra RandomForest de 200 árboles y profundidad 12; ajuste con máximo 100000 ejemplos de train, semilla 42, para la laptop; validación y prueba completas. Umbral por val F2, se registran AP, recall, precision, balanced accuracy, prevalencia y matrices. P3 KMeans k2 baseline y candidatos k3-k6 con StandardScaler ajustado en train; selección val silhouette y mínimo cinco estaciones por grupo, métricas internas y ARI temporal.

## Resultados y reproducción
Los resultados previos de Martha se guardan en reports/reproduccion_martha.json y se identifican con commit, data_version y run padre. Los resultados locales de Milton para ingesta y P1 se guardan en reports/ingesta_p1_milton.json y reports/reproduccion_milton.json, con la misma trazabilidad. Ambas series se mantienen separadas. No confundir estas métricas offline con impacto operacional. Para reproducir clona el mismo commit, instala el lock y usa las mismas fuentes verificadas con el manifiesto; repite el comando. Víctor debe ejecutar personalmente su parte en su equipo y aportar su resumen auténtico. Un clon limpio ejecutado por Martha acredita repetibilidad local, no colaboración independiente.

En la ejecución de Milton, P1 comparó tres modelos con los mismos datos de evaluación: 2,090,016 filas de entrenamiento, 540,216 de validación y 540,216 de prueba. El val_mae fue 1.0333 para el semanal, 0.8639 para el promedio y 0.8582 para HistGradientBoosting. La diferencia de 0.0057 entre el promedio y boosting quedó dentro de la tolerancia de 0.01, y la regla seleccionó el promedio, con test_mae 0.8776. Boosting tuvo test_mae 0.8077 como candidato, y ese valor no modifica la selección. MLflow registró un run padre y tres hijos FINISHED con 18 métricas de modelos, y pasaron 16 pruebas.

## Git y colaboración
Ramas feat/mavf-config-project y docs/mavf-entrega, commits con autoría real. La contribución local de Milton está en feat/milton-ingesta-p1, con base docs/mavf-revision-ingesta-p1. Todo cambio posterior al arranque debe entrar mediante PR revisado por otra persona del equipo. No aprobar el propio PR. Registro previo de colaboración en references/collaboration.md. Los nombres asignados a roles no prueban trabajo realizado. Quedan pendientes la revisión de los PR y la consulta docente sobre cuatro integrantes para este equipo de tres.

## Limitaciones y mejora
Capacidades de 2026 aplicadas retrospectivamente; sin inventario ni rebalanceo observado, demanda no atendida, meteorología ni calendario de altas/bajas históricas de estaciones. P3 excluye estaciones con menos de 28 días activos de entrenamiento. Primera semana excluida por rezagos. Validar telemetría histórica antes de uso operativo, evaluar cobertura por fecha de apertura y monitorizar cambios de distribución. No se acreditó lectura institucional restringida ni escucha de un episodio concreto del podcast.

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

## Comprobación previa de Martha del rol Persona2_Milton
La revisión de ingesta y P1 se repitió en la carpeta Persona2_Milton, en un clon separado y un entorno reconstruido desde el lock. La ejecución técnica fue solicitada por Martha y se conserva su cuenta y autoría reales. Ver references/revision_ingesta_p1_persona2.md, references/lecturas_persona2.md, reports/data_summary.json y los resúmenes Persona2_Milton. Las ocho pruebas pasaron; la ejecución completa produjo nueve hijos finalizados y 103 valores numéricos coincidentes con la reproducción anterior dentro de 1e-12. Esto no acredita participación independiente de Milton. Las invitaciones a MiltonVeraTorre y VictorVillaHe están pendientes de aceptación al verificar. Las capturas de esta comprobación se conservan en el informe Word local y no se agregan a este PR.

## Ejecución de Milton
Milton ejecutó en su equipo local la ingesta y P1 desde la rama feat/milton-ingesta-p1. La ingesta procesó 4,532,032 viajes de 13 fuentes y generó 14 archivos Parquet procesados. P1 evaluó tres modelos sobre una cohorte de 369 estaciones y seleccionó el promedio por estación, día de la semana y hora, con la regla de validación y tolerancia. Las 16 pruebas pasaron. Los resultados y su trazabilidad están en reports/ingesta_p1_milton.json y reports/reproduccion_milton.json, y los comandos de reproducción en macOS están en la sección Reproducción local en macOS. P2 y P3 conservan los componentes del equipo y sus resultados, registrados en los reportes de Martha.
