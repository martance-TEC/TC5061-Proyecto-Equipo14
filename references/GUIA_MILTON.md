# Reproducción independiente de P1

Usar la versión estable de la segunda entrega indicada en el PR de Martha. No hace falta esperar al trabajo de Víctor. Consultar primero la instalación macOS o Windows y el servidor local del README. Se necesitan Python 3.11.9 y las dependencias del lock. No se necesita Docker.

## Ejecución

1. Clonar el repositorio en una carpeta nueva. Cambiar a la rama estable indicada en el PR si aún no se integró a main. Crear el entorno y ejecutar `python -m pip check` y `python -m pytest -q`.
2. Iniciar un servidor propio en localhost:5014 con una base nueva. Configurar `MLFLOW_TRACKING_URI` y `MLFLOW_REGISTRY_URI` para ese servidor. El historial oficial permanece en la computadora de la campaña; no copiar su base o sus modelos.
3. Ejecutar los comandos siguientes desde la raíz, con el entorno activo:

```text
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo14
mlflow run . -e verify --env-manager local --experiment-name mibici-equipo14
mlflow run . -e reproduce --env-manager local --experiment-name mibici-equipo14 -P question=p1
```

`reproduce` lee la configuración elegida y compara las métricas con la referencia del repositorio. Escribe el resultado local en `references/validaciones/local_p1.json`. Ese archivo se excluye por defecto de Git para evitar publicar resultados sin revisar. Si el resultado falla, conservar el mensaje y no marcarlo aprobado.

4. Abrir la interfaz local, comprobar los parámetros de P1 y el manifiesto, y guardar una captura legible para la validación. No repetir toda la búsqueda.
5. Añadir una prueba en `tests/test_reproduccion_p1.py` que compruebe una propiedad real de P1: por ejemplo, que un cambio posterior al corte no modifique el promedio entrenado ni los rezagos anteriores. No copiar sin comprobar una prueba ya existente.
6. Crear `references/validaciones/p1.md` con commit utilizado, comando, data_version, run local y comparación real de métricas. No incluir datos de viajes ni modelos.
7. Crear una rama `test/p1-reproduccion`, hacer commit de la prueba y la validación, publicar y abrir PR con base en la rama de integración indicada en el PR de Martha. Cuando la infraestructura esté en main, la base final será main. Martha revisa ese PR.
8. Revisar el PR de Víctor desde la cuenta propia en Files changed. Enviar una revisión formal con Approve o Request changes según corresponda.

Esto deja una contribución técnica, un PR y una revisión verificables de esta entrega. No se necesita editar el código común de entrenamiento ni el reporte general.
