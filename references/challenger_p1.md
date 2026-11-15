# Challenger P1 con HistoricalMean

El challenger calcula la demanda media por estación, día de la semana y hora con las particiones originales de enero a octubre de 2025. Usa las 369 estaciones de la cohorte definida con las salidas de enero a agosto. Para una combinación sin historia, devuelve la media del conjunto de ajuste.

La preparación oficial conserva las fechas y las variables del proyecto. El entrenamiento reúne `train` y `val`, con 2,630,232 filas. Las primeras filas disponibles empiezan el 8 de enero por los rezagos de la preparación original. Noviembre y diciembre forman el holdout para la comparación A/B.

El modelo se registra en `mibici-equipo14-p1-demanda` con el alias `challenger`. El comando verifica el alias `champion` existente y requiere que `challenger` esté libre. La firma de entrada coincide con la API: `station_id` como texto, `weekday` y `hour` como enteros de 32 bits. La salida es un vector de números de doble precisión.

## Preparar los datos

Desde la raíz del repositorio, con Python 3.11 y las dependencias de `requirements.lock.txt`:

```bash
python -m mibici_equipo14.dataset
python -c "from mibici_equipo14.modeling.advanced import prepare; prepare('p1')"
```

La ingesta descarga las fuentes oficiales que falten y comprueba su hash contra el manifiesto. Si ya están descargadas, también se puede usar `python -m mibici_equipo14.dataset --verify`. `prepare('p1')` crea las particiones originales. El comando del challenger carga únicamente los archivos `train.parquet` y `val.parquet`.

## Registrar el candidato

El URI debe apuntar al backend que ya contiene el `champion` local. Tracking y Registry usan ese mismo URI durante la ejecución. Usa una carpeta local persistente para los artefactos y ejecuta el registro cuando nadie más esté cambiando los alias de ese backend.

```bash
python -m mibici_equipo14.modeling.challenger \
  --tracking-uri 'sqlite:////ruta/al/runtime/mlflow.db' \
  --artifact-root '/ruta/al/runtime/challenger_artifacts' \
  --output reports/challenger_p1/evidence.json
```

El comando entrena `HistoricalMean`, comprueba su clase al cargarlo y compara las predicciones antes y después del registro. La muestra incluye filas de ambos periodos de ajuste y una estación nueva para verificar el fallback. El alias se asigna después de comprobar la versión registrada. Si ya existe un challenger, el comando se detiene para revisar esa versión.

`evidence.json` contiene el run, la versión, los alias antes y después, las fechas y filas de ajuste, la cohorte, el commit, los hashes del código y de las particiones, la versión de datos, la firma y las predicciones verificadas. Estas comprobaciones corresponden al contrato y la carga del modelo. El desempeño se compara con el holdout de noviembre y diciembre.

## Repetir en Windows

En PowerShell, abre el repositorio y activa su entorno Python 3.11. Usa el backend donde ya está registrado el `champion` de esa computadora. Para un servidor local de MLflow en el puerto 5014:

```powershell
$TrackingUri = "http://127.0.0.1:5014"
$Artifacts = Join-Path $PWD "mlartifacts/challenger_p1"
python -c "import sys; from mlflow import MlflowClient; c=MlflowClient(tracking_uri=sys.argv[1], registry_uri=sys.argv[1]); print(c.get_registered_model('mibici-equipo14-p1-demanda').aliases)" $TrackingUri
python -m mibici_equipo14.dataset
python -c "from mibici_equipo14.modeling.advanced import prepare; prepare('p1')"
python -m mibici_equipo14.modeling.challenger --tracking-uri $TrackingUri --artifact-root $Artifacts --output reports/challenger_p1/evidence_victor.json
```

Para un backend SQLite local, cambia `$TrackingUri` por su ruta, por ejemplo `sqlite:///C:/MiBici/runtime/mlflow.db`. Cada Registry asigna sus propios números de versión y run IDs. Para la comparación A/B, usa los valores de la evidencia generada en esa computadora.

## Pruebas

```bash
python -m pytest tests/test_p1_challenger.py -q
```

Las pruebas cubren las fechas, la cohorte, los duplicados, la firma y la validación de predicciones. Una prueba registra dos versiones en un SQLite temporal y verifica que `champion` conserva sus predicciones, que `challenger` incorpora septiembre y octubre y que una segunda ejecución respeta el alias existente. Otra prueba usa un archivo holdout que no se puede leer como Parquet para comprobar que el cargador solo abre train y val.
