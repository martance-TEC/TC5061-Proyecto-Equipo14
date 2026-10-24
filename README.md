# MiBici

## Segunda entrega de MLflow

El [protocolo](references/segunda_entrega.md) amplía el proyecto con entrenamiento parametrizado, tres candidatos por pregunta, búsquedas anidadas y Model Registry. La base de la primera computadora no se conservó: los baselines se reconstruyen con los mismos datos y definiciones y quedan identificados como reproducciones. No se reutilizan los run_id históricos.

El [reporte editable](reports/TC5061_Reporte_MLFlow_Equipo14.docx) reúne las comparaciones, la selección, el linaje y las capturas de la interfaz. La [verificación desde un clon limpio](reports/segunda_entrega/clean_clone_verification.json) registró diferencias de 0.0 en todas las métricas comparadas de los tres modelos elegidos. Las [comprobaciones](reports/segunda_entrega/validation.json) incluyen 23 pruebas aprobadas y la auditoría de quince corridas y tres modelos cargados por alias.

### Preparación en macOS

Se utiliza Python 3.11.9 y MLflow 3.4.0. Desde la raíz del clon:

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0.1 setuptools==78.1.0 wheel==0.45.1
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m pytest -q
```

El intérprete debe indicar 3.11.9. Si falta, instalar esa versión por una vía autorizada. No hace falta Docker ni Conda. `--env-manager local` usa este entorno verificado; no instala las dependencias por sí solo.

### Preparación en Windows

Desde PowerShell, con Python 3.11.9 de 64 bits instalado:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.0.1 setuptools==78.1.0 wheel==0.45.1
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
python -m pip check
python -m pytest -q
```

No se necesita cambiar la política de ejecución de PowerShell: los comandos usan los ejecutables del entorno o agregan su carpeta a PATH.

### Servidor con SQLite

En una terminal con el entorno activo, desde la raíz. El puerto 5014 evita el servicio de macOS que puede ocupar el 5000:

```text
mlflow server --host 127.0.0.1 --port 5014 --workers 1 --backend-store-uri sqlite:///mlflow.db --serve-artifacts --artifacts-destination ./mlartifacts
```

Mantener esa terminal abierta. En otra terminal del proyecto, activar el mismo entorno y configurar:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5014
export MLFLOW_REGISTRY_URI=http://127.0.0.1:5014
```

Equivalente en PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5014'
$env:MLFLOW_REGISTRY_URI = $env:MLFLOW_TRACKING_URI
```

Abrir http://127.0.0.1:5014. La base y `mlartifacts/` se conservan localmente y están excluidas de Git. No cambiar la ubicación del almacén sin verificar sus referencias. Un clon no contiene el historial de otra computadora.

### Ejecutar la campaña completa

```text
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo14
mlflow run . -e verify --env-manager local --experiment-name mibici-equipo14
mlflow run . -e campaign --env-manager local --experiment-name mibici-equipo14
mlflow run . -e promote --env-manager local --experiment-name mibici-equipo14
mlflow run . -e verify_registry --env-manager local --experiment-name mibici-equipo14
```

El código de entrenamiento debe estar comprometido y limpio antes de entrenar. Los resúmenes se generan en `reports/segunda_entrega/`; se publican después en un commit documental. La fuente debe coincidir exactamente con el SHA256 versionado: una descarga diferente detiene la ingesta antes de sobrescribir procesados.

`search -P question=p1` ejecuta solo esa búsqueda. `train` ejecuta una configuración. Ejemplo:

```text
mlflow run . -e train --env-manager local --experiment-name mibici-equipo14 -P question=p1 -P model=hgb -P max_iter=150 -P learning_rate=0.1 -P max_leaf_nodes=15
```

Los valores opcionales 0 o modelo vacío delegan a `config.py`; no son hiperparámetros reales. El parser rechaza parámetros que no correspondan al modelo. Los entry points `main` y sus parámetros anteriores se conservan como compatibilidad de la primera entrega.

### Modelos elegidos y reproducción independiente

La configuración seleccionada queda en `reports/segunda_entrega/selection.json`. `reproduce` la utiliza sin decidir de nuevo y comprueba diferencias de métricas contra la campaña, con tolerancia absoluta 1e-8. El lock, los datos y el orden de las filas se mantienen. Si no coincide, se investiga; no se declara coincidencia por aproximación visual.

```text
mlflow run . -e reproduce --env-manager local --experiment-name mibici-equipo14 -P question=p1
mlflow run . -e reproduce --env-manager local --experiment-name mibici-equipo14 -P question=p2
mlflow run . -e reproduce --env-manager local --experiment-name mibici-equipo14 -P question=p3
```

Cada compañero usa su servidor local; no se confunden esos run_id con los oficiales. Para registrar la reproducción propia: `-e promote -P question=p2 -P local=1`, y para cargarla: `-e verify_registry -P question=p2 -P local=1` con las demás opciones habituales. Solo se usa `local=1` después de reproducir esa pregunta.

### Ciclo de vida

Se usan los aliases vigentes en lugar de etapas clásicas obsoletas. Los nombres son `mibici-equipo14-p1-demanda`, `mibici-equipo14-p2-riesgo` y `mibici-equipo14-p3-grupos`. Se verifica por versión numérica antes de asignar `champion`, y luego por nombre, por ejemplo `mlflow.pyfunc.load_model('models:/mibici-equipo14-p1-demanda@champion')`. Se requiere una entrada con la firma del modelo elegido. El código de verificación descarga su ejemplo, conserva los tipos y llama predict.

El alias designa el modelo elegido para el consumo previsto en este proyecto académico. No equivale a un servicio real desplegado en MiBici. La versión registra origen, commit, data_version, descripción y validación. P2 conserva dentro del paquete el umbral elegido; P3 incluye el scaler; el promedio de P1 incluye su respaldo global.

### Colaboración independiente

[Guía de Milton](references/GUIA_MILTON.md) y [guía de Víctor](references/GUIA_VICTOR.md): pueden ejecutarse desde la misma versión estable sin depender entre sí. Cada integrante debe hacer sus commits, abrir su PR y revisar otro PR en esta entrega. Las revisiones anteriores no sustituyen las actuales. Todo se integra mediante PR revisado y con la protección de main activa.

---

## Primera entrega y antecedentes

Proyecto de Operaciones de aprendizaje automático para estudiar demanda de viajes, presión sobre la capacidad y patrones de uso de estaciones. [Repositorio](https://github.com/martance-TEC/TC5061-Proyecto-Equipo14).

## Problema y usuarios

| Pregunta | Tarea y alcance | Usuario y decisión |
|---|---|---|
| P1 ¿Cuántos viajes comenzarán? | Regresión por estación para la siguiente hora | Operador del sistema: planear la flota |
| P2 ¿Hay riesgo de pocas bicicletas o anclajes? | Clasificación de presión aproximada por estación para las siguientes tres horas | Equipo de rebalanceo: priorizar estaciones |
| P3 ¿Qué estaciones comparten patrones de uso y dónde están? | Clustering de perfiles históricos por estación, con mapa posterior | Planeación urbana y expansión: estudiar patrones y necesidades de servicio |

La formulación, baselines, variables y reglas de selección están en [problem_definition.md](references/problem_definition.md). El [ML Canvas](references/ml_canvas.md) y la [arquitectura](references/architecture.md) explican la solución y las herramientas.

## Datos y partición

Se utilizan los doce meses de 2025: 4,532,032 viajes de 13 fuentes que suman 393,104,868 bytes, incluido el catálogo de septiembre de 2026. El catálogo contiene 484 estaciones, 480 con capacidad positiva. Los enlaces exactos, tamaños y SHA256 están en [sources.json](references/sources.json) y [data_manifest.json](references/data_manifest.json). El [diccionario](references/data_dictionary.md) documenta variables, uso de datos y privacidad.

- Entrenamiento: enero–agosto de 2025.
- Validación: septiembre–octubre de 2025, para seleccionar modelo, umbral y número de grupos.
- Prueba: noviembre–diciembre de 2025, reservada para evaluación final.

Cada objetivo termina dentro de su partición: una hora en P1 y tres en P2. Los rezagos y promedios móviles usan horas anteriores cerradas. Las transformaciones se ajustan solo con entrenamiento. P1 congela una cohorte de 369 estaciones observadas en entrenamiento; registra por separado los viajes fuera de ella. Quedan 2,090,016 filas de entrenamiento y 540,216 tanto en validación como en prueba, después de excluir las primeras 168 horas sin historia suficiente.

P2 utiliza la capacidad oficial de la estación. El inventario inicial de 0.5 × capacidad es un supuesto. La etiqueta se activa si el saldo de llegadas menos salidas cruza 0.1 × capacidad o 0.9 × capacidad en cualquier minuto de las tres horas. Se evalúan también inventarios iniciales de 25 % y 75 %. La capacidad de 2026 aplicada a viajes de 2025 define un escenario retrospectivo; no demuestra faltantes históricos reales.

La ingesta normaliza columnas, codificación y fechas, excluye información personal, controla duplicados y escribe 14 Parquet locales. El manifiesto versiona las fuentes y registra filas, tamaño y huellas de los procesados. `--verify` comprueba las fuentes antes de reemplazar los Parquet. Los datos no se suben a Git; los hashes detectan cambios, pero no recuperan una fuente reemplazada.

## Estructura y refactorización

La estructura se generó con CCDS y `include_code_scaffold=Yes`; su procedencia está en [template_provenance.json](references/template_provenance.json).

```text
mibici_equipo14/
  config.py          # rutas, fechas, semilla y configuración
  dataset.py         # descarga, normalización y versionado
  features.py        # variables reutilizables y objetivos
  modeling/train.py  # regresión, clasificación y clustering
  tracking.py        # parámetros, métricas, etiquetas y artefactos
  pipeline.py        # coordinación y resumen
notebooks/           # exploración
references/          # Canvas, arquitectura, fuentes y definiciones
reports/             # resultados agregados y figuras
tests/              # controles de integridad y fuga temporal
```

`data/`, `models/` y los almacenes de MLflow permanecen locales. `Settings` centraliza configuración; `ExperimentTracker` concentra trazabilidad; cada clase de experimento encapsula una tarea. Las funciones documentadas de `features.py` se comparten sin copiar lógica en notebooks. Pandas/PyArrow permiten procesamiento tabular y Parquet; scikit-learn aporta modelos y transformaciones; MLflow Projects define el comando reproducible y Tracking conserva evidencias.

## Instalación en Windows

Desde PowerShell en la raíz del clon, con Python 3.11 de 64 bits. `python_env.yaml` fija Python 3.11.9 y `requirements.lock.txt` fija los paquetes.

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.0.1 setuptools==78.1.0 wheel==0.45.1
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
python -m pip check
python -m pytest -q
python -m mibici_equipo14.dataset
python -m mibici_equipo14.dataset --verify
```

La primera ingesta descarga las fuentes. Las posteriores pueden verificarse sin volver a descargarlas. Una fuente cambiada requiere investigar la diferencia, no sustituir silenciosamente el manifiesto.

En macOS/Linux se usan `python3.11 -m venv .venv`, `source .venv/bin/activate` y los mismos comandos `python -m pip`, `pytest` y `dataset`. `pywin32` tiene un marcador de Windows en el lock.

## MLflow y ejecución con un comando

En una segunda consola PowerShell, desde la raíz:

```powershell
$artifactPath = ([System.Uri](Join-Path $PWD 'mlartifacts')).AbsoluteUri
.\.venv\Scripts\mlflow.exe server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db --default-artifact-root $artifactPath
```

En la consola con el entorno activado:

```powershell
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
$env:PYTHONIOENCODING = 'utf-8'
mlflow run . --env-manager local --experiment-name mibici-equipo14 -P question=all
```

Abrir http://127.0.0.1:5000. El experimento es `mibici-equipo14`. La ejecución completa genera un run padre y diez hijos: tres modelos P1, dos P2 y cinco P3. `question` acepta `all`, `P1`, `P2` o `P3`. Para variar P1 sin editar código: `mlflow run . --env-manager local --experiment-name mibici-equipo14 -P question=P1 -P max_iter=150`. `max_iter=0` usa el valor central 100. El gestor local utiliza el entorno instalado con el lock; no necesita Conda. También existen los entry points `ingest` y `verify`.

Cada run registra fechas, semilla, modelo e hiperparámetros, versión de datos, commit y estado del código, pregunta, métricas con prefijo `val_` o `test_` y artefactos. Los resultados de una nueva ejecución se guardan en `reports/reproduccion_local.json` y en el artefacto `summary/results.json` del padre.

## Modelos y resultados

| Pregunta | Comparación y selección | Resultado de prueba |
|---|---|---|
| P1 | Semanal, promedio estación/día/hora y HistGradientBoosting Poisson. Selección por MAE de validación; tolerancia 0.01 a favor del más sencillo | Promedio seleccionado: MAE 0.877629, RMSE 1.594179, WAPE 66.28 % |
| P2 | Dummy prior y Random Forest, 200 árboles, profundidad 12. Umbral por F2 de validación | Random Forest: F2 0.562956, recall 0.810477, precisión 0.253400, AP 0.392378 |
| P3 | StandardScaler y KMeans k=2…6; ajuste en train, selección por silhouette de validación y tamaño mínimo de grupo | k=3, 369 estaciones; silhouette 0.056384 y ARI temporal 0.806066 |

P1: el promedio obtiene `val_mae=0.863884` y el candidato `val_mae=0.858212`. La diferencia 0.005673 es menor que la tolerancia fijada; por eso se elige el promedio. El candidato alcanza `test_mae=0.807710`, pero la prueba no cambia la regla de selección. Los valores completos de la cohorte corregida están en [reproduccion_milton.json](reports/reproduccion_milton.json).

P2: se ajusta con hasta 100,000 ejemplos de entrenamiento, semilla 42; validación y prueba se evalúan completas. El recall elevado se acompaña de baja precisión, por lo que muchas alertas requerirían revisión. P3 presenta separación débil pese a la estabilidad temporal; no se asignan automáticamente nombres de zonas a los grupos. Los resultados integrados de las tres preguntas están en [reproduccion_local.json](reports/reproduccion_local.json), vinculados al commit `658e428` con código limpio. La [validación integrada](reports/validacion_integrada.json) registra 18 pruebas aprobadas, los once runs terminados y la comparación con la referencia anterior. P2 conserva el modelo seleccionado, con pequeñas diferencias numéricas registradas; no se declara coincidencia exacta. Los reportes anteriores de P1 conservan sus commits de origen y no sustituyen la evaluación de la cohorte corregida.

## Evidencias

La [propuesta en PDF](reports/TC5061_Propuesta_Proyecto_Equipo14.pdf) reúne la formulación, ML Canvas, metodología, resultados, arquitectura, referencias y capturas de la interfaz.

[Guía de evidencia](references/evidencias.md): lista de runs, detalle con parámetros/métricas/etiquetas, artefacto abierto dentro de MLflow y comparación de modelos. Las capturas deben corresponder a runs identificables y a las métricas informadas. El mapa independiente muestra el resultado, pero la captura de su registro debe mostrar la pestaña Artifacts.

## Flujo de Git

`main` requiere PR revisado y aprobación de otra cuenta, también para administradores. No se usa force push ni se elimina la protección. Ramas `feat/descripcion`, `fix/descripcion` o `docs/descripcion`; commits `feat:`, `fix:`, `docs:` o `test:`. Cada PR describe el cambio, sus comprobaciones y el resultado. Cada integrante abre al menos un PR y revisa otro; las aprobaciones y autorías se verifican en GitHub, no se sustituyen por tablas declarativas.

Las responsabilidades técnicas se separan en configuración/integración, ingesta/regresión y variables/clasificación/clustering. El historial conserva las aportaciones originales al integrar ramas. No se publican datos de viajes, modelos, almacenes de MLflow ni credenciales.

## Limitaciones

Faltan inventarios históricos y rebalanceos observados; el catálogo de capacidad es posterior a los viajes. Los ceros del panel no prueban que una estación ya operara. P1 no cubre nuevas estaciones sin historia; P3 exige al menos 28 días activos en entrenamiento. No se modelan clima ni demanda no atendida. Las métricas describen evaluación histórica, no mejoras operativas demostradas.

## Referencias

- DrivenData. (s. f.). *Cookiecutter Data Science*. https://cookiecutter-data-science.drivendata.org/
- MiBici. (s. f.). *Datos abiertos*. https://mibici.net/es/datos-abiertos/
- MLflow. (s. f.-a). *MLflow example* [Repositorio de código]. https://github.com/mlflow/mlflow-example
- MLflow. (s. f.-b). *MLflow Projects*. https://mlflow.org/docs/latest/ml/projects/
- MLflow. (s. f.-c). *MLflow Tracking*. https://mlflow.org/docs/latest/ml/tracking/
- Real Python. (s. f.). *Object-oriented programming in Python*. https://realpython.com/python3-object-oriented-programming/

Las notas técnicas y sus fuentes están en [lecturas_ingesta.md](references/lecturas_ingesta.md) y [lecturas_variables.md](references/lecturas_variables.md).
