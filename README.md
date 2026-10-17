# MiBici Equipo 14

Proyecto de Operaciones de aprendizaje automático. Martha Alicia Villalobos Facundo, Milton Vera de la Torre y Víctor Villa Herrera. Repositorio: https://github.com/martance-TEC/TC5061-Proyecto-Equipo14

## Estado de la entrega
La implementación se encuentra en el PR 1, pendiente de revisión humana: https://github.com/martance-TEC/TC5061-Proyecto-Equipo14/pull/1 . Hasta que se apruebe e integre, main conserva solamente el inicio de la plantilla. Para obtener el proyecto ejecutable usa `git clone --branch feat/mavf-config-project https://github.com/martance-TEC/TC5061-Proyecto-Equipo14.git`. La rama docs/mavf-entrega agrega los resultados y evidencias posteriores a la ejecución.

Dos ejecuciones locales completas de Martha procesaron 4,532,032 viajes y produjeron las mismas 91 métricas con tolerancia 1e-12. La ejecución desde clon limpio usa el commit 393a22abcac18008db68455abb9069f05ccf2278 y la versión de datos 97b8d9d898ca5f71ffaffc6ff86b100fbe371abfd5d67837d989e5f81bf9934f. Ver reports/reproduction_audit.json y reports/reproduccion_martha.json en la rama documental. P1 seleccionó HistGradientBoosting con test_mae 0.8051 frente a 0.9904 del baseline. P2 seleccionó RandomForest con test_f2 0.5627 y test_average_precision 0.3924, bajo la etiqueta proxy definida. P3 seleccionó k=3 sobre 369 estaciones elegibles.

La invitación que Martha envió a VictorVillaHe está pendiente de aceptación al verificar. Todavía falta el usuario de GitHub de Milton. No se acreditan PR, revisiones ni ejecuciones independientes de ellos. La reproducción de Martha no sustituye su participación real. No se ha enviado la entrega a la plataforma académica.

## Problema y usuarios
P1: regresión del número de viajes iniciados por estación durante la siguiente hora, para el operador del sistema. P2: clasificación de presión aproximada sobre la capacidad en las siguientes tres horas, para rebalanceo. P3: clustering de perfiles históricos por estación y su localización, para planeación urbana y expansión. Las decisiones completas, baselines y métricas están en references/problem_definition.md y el ML Canvas en references/ml_canvas.md.

## Datos y partición
Doce CSV oficiales de 2025, más catálogo de estaciones septiembre de 2026 con dpcapacity. Ver references/sources.json, data_dictionary.md y data_manifest.json. Entrenamiento enero-agosto, validación septiembre-octubre, prueba noviembre-diciembre; se purgan horizontes que cruzan fronteras. Los campos personales se excluyen. Ningún viaje se publica en Git. P2 es un escenario retrospectivo con capacidad fija e inventario inicial hipotético; no predice faltantes reales observados.

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
P1 seasonal naive contra HistGradientBoosting Poisson; selección val_mae, test MAE RMSE WAPE. P2 prior contra RandomForest de 200 árboles y profundidad 12; ajuste con máximo 100000 ejemplos de train, semilla 42, para la laptop; validación y prueba completas. Umbral por val F2, se registran AP, recall, precision, balanced accuracy, prevalencia y matrices. P3 KMeans k2 baseline y candidatos k3-k6 con StandardScaler ajustado en train; selección val silhouette y mínimo cinco estaciones por grupo, métricas internas y ARI temporal.

## Resultados y reproducción
Los resultados reales se guardan en reports/reproduccion_martha.json; se identifican con commit, data_version y run padre. No confundir estas métricas offline con impacto operacional. Para reproducir clona el mismo commit, instala el lock y usa las mismas fuentes SHA256; repite el comando. Milton y Víctor deben ejecutar personalmente en sus equipos y aportar sus resúmenes auténticos. Un clon limpio ejecutado por Martha acredita repetibilidad local, no colaboración independiente.

## Git y colaboración
Ramas feat/mavf-config-project y docs/mavf-entrega, commits con autoría real. Todo cambio posterior al arranque debe entrar mediante PR revisado. No aprobar el propio PR. Estado de colaboración en references/collaboration.md. Los nombres asignados a roles no prueban trabajo realizado. Permanecen pendientes los handles y revisiones auténticas de los demás integrantes, así como aclaración docente sobre cuatro usuarios para este equipo de tres.

## Limitaciones y mejora
Capacidades de 2026 aplicadas retrospectivamente; sin inventario ni rebalanceo observado, demanda no atendida, meteorología ni calendario de altas/bajas históricas de estaciones. P3 excluye estaciones con menos de 28 días activos de entrenamiento. Primera semana excluida por rezagos. Validar telemetría histórica antes de uso operativo, evaluar cobertura por fecha de apertura y monitorizar cambios de distribución. No se acreditó lectura institucional restringida ni escucha de un episodio concreto del podcast.

## Referencias
MLflow. (s. f.). MLflow Projects. https://mlflow.org/docs/latest/ml/projects/
MLflow. (s. f.). MLflow example. https://github.com/mlflow/mlflow-example
DrivenData. (s. f.). Cookiecutter Data Science. https://cookiecutter-data-science.drivendata.org/
MiBici. (s. f.). Datos abiertos. https://mibici.net/es/datos-abiertos/
Python Software Foundation. (2024). Python 3.11.9. https://www.python.org/downloads/release/python-3119/
Real Python. (s. f.). Object oriented programming in Python. https://realpython.com/python3-object-oriented-programming/
Las lecturas institucionales y el podcast se encuentran en la consigna, pero no se citan como consultados íntegramente sin acceso verificable.

## Comprobación del rol Persona 2 Milton
La revisión de ingesta y P1 se repitió en la carpeta Persona2_Milton, en un clon separado y un entorno reconstruido desde el lock. La ejecución técnica fue solicitada por Martha y se conserva su cuenta y autoría reales. Ver references/revision_ingesta_p1_persona2.md, references/lecturas_persona2.md, reports/data_summary.json y los resúmenes Persona2_Milton. Las ocho pruebas pasaron; la ejecución completa produjo nueve hijos finalizados y 103 valores numéricos coincidentes con la reproducción anterior dentro de 1e-12. Esto no acredita participación independiente de Milton. Las invitaciones a MiltonVeraTorre y VictorVillaHe están pendientes de aceptación al verificar. Las capturas de esta comprobación se conservan en el informe Word local y no se agregan a este PR.