# Proyecto final MiBici

El sistema permite consultar la demanda esperada por estación y hora mediante
una API. El trabajo incluye las preguntas P1, P2 y P3, pero el despliegue y el
monitoreo final se concentran en P1. Es una demostración académica local con datos
históricos, no un servicio oficial ni una operación conectada a MiBici.

## Recorrido del sistema

| Fase | Implementación | Evidencia y decisión |
|---|---|---|
| Estructura y datos | Cookiecutter, `dataset.py`, `config.py` y `features.py` | [Procedencia de la plantilla](template_provenance.json), [fuentes](sources.json) y [manifiesto](data_manifest.json); los viajes permanecen fuera de Git |
| Experimentación | `MLproject`, entorno fijado y módulos de entrenamiento | [Campaña](../reports/segunda_entrega/campaign_summary.json); runs anidados, modelos con firma, configuración y métricas |
| Registry | `registry.py` y copia operativa en `operations.py` | [Verificación histórica](../reports/segunda_entrega/registry_verification.json) y [origen del modelo servido](../reports/operacion/lineage.json) |
| Servicio | FastAPI, Pydantic, Containerfile y Compose | [Validación HTTP publicada](../reports/operacion/validation.json); la API consulta el modelo registrado al arrancar |
| Calidad y entrega | PyTest, Deepchecks, Bandit y GitHub Actions | [Hallazgos y alcance de seguridad](../reports/calidad/hallazgos.md); CI prueba y construye antes de desplegar temporalmente en el runner |
| Monitoreo | Evidently, comparación temporal y A/B offline | [Interpretación de drift](../reports/monitoreo/interpretacion.md) y [resultado pareado](../reports/ab_testing/p1_ab_victor.json) |
| Operación y costo | Locust, Podman stats y `finops.py` | [Resumen reproducible](../reports/operacion/summary.json); costo unitario y sensibilidad a la utilización |
| Gobernanza | Model Card, linaje y revisión de cambios | [Model Card](../MODEL_CARD.md); revisión humana antes de modificar la versión vigente |

## Qué estima P1

HistoricalMean aprende el promedio de salidas para cada estación, día de la
semana y hora. La entrada contiene `station_id` como texto, `weekday` entre 0 y 6
y `hour` entre 0 y 23. La salida es un número esperado de viajes, posiblemente
fraccionario. No es el inventario de bicicletas ni una medición de demanda no
atendida. Un identificador sin historia usa la media global; esa respuesta tiene
menor respaldo local y no valida cobertura de estaciones nuevas.

La cohorte se fija con las 369 estaciones del entrenamiento. Enero-agosto de
2025 es la referencia original; septiembre-octubre es validación y periodo
actual para drift; noviembre-diciembre es el holdout. Los rezagos usan horas
cerradas y las transformaciones se ajustan sin utilizar el futuro.

## Modelo vigente y alternativa

La selección inicial de P1 se hizo con MAE de validación y una tolerancia de
0.01 a favor del modelo sencillo. El promedio histórico quedó como v1.
La operación publicada mantiene esa versión numérica para conservar las
mediciones y evitar que un cambio de alias cambie silenciosamente el servicio.

La comparación posterior reentrena la misma familia con información hasta
octubre. En las mismas 540,216 observaciones de noviembre-diciembre, v2 obtuvo
MAE 0.860265 frente a 0.877629 de v1, una reducción aproximada de 1.98%.
El bootstrap por días también favorece a v2. Esa es una recomendación de
promoción, no evidencia de una promoción ya ejecutada ni de tráfico A/B online.
La API y la carga medidas siguen siendo v1. Antes de anunciar v2 como vigente
hay que aprobar su promoción, comprobar firma, linaje y comportamiento HTTP,
y repetir las mediciones que se quieran atribuir a esa versión.

El Registry de operación es una copia separada. Su v1 conserva el origen en
`config/origin.json` y se identifica como `champion` de alcance `academic_local`.
El bootstrap no cambia los aliases del Registry original ni permite reemplazar
un champion local diferente. Las evidencias A/B de otra reconstrucción tienen
otros run IDs; se interpretan dentro de su propio Registry.

## Decisiones de operación

Podman ejecuta los contenedores en macOS y Compose coordina MLflow, la importación
inicial del modelo y FastAPI. Postman o curl sirven para enviar solicitudes;
no sustituyen al motor de contenedores. MLflow y la API comparten una red interna;
SQLite y los artefactos se guardan en un volumen persistente. Los healthchecks
ordenan el arranque. Los puertos publicados solo escuchan en localhost.

La inferencia es síncrona: el modelo permanece cargado en memoria y no se
reentrena al recibir una petición. Se omiten Redis y RQ porque no se midió una
necesidad de trabajo asíncrono. CI construye la misma imagen y realiza un
despliegue temporal con datos sintéticos identificados. Eso comprueba integración,
no sustituye las pruebas locales del modelo real ni un despliegue permanente.

Los tres escenarios locales de Locust registraron 12,987 solicitudes sin fallos.
Con 50 usuarios se observaron 96.57 solicitudes/s y p95 de 10 ms. Son pruebas
breves con una repetición; no prueban capacidad máxima ni rendimiento en AWS.
La estimación de USD 0.00000009468 por inferencia usa USD 24/mes, 730 horas de
planeación y 8,681 respuestas exitosas en 90 segundos. Con poco tráfico mensual
el costo unitario aumenta: 10,000 inferencias/mes implican USD 2.40 por mil.

La CPU media de la API de 20.89% de un núcleo y el pico conjunto de 502.87 MiB
justifican evaluar un tamaño menor, no asegurar de antemano que funcionará igual.
Hay que medir arranque, duración prolongada y margen de recursos. Mantener un
solo modelo cargado, evitar servicios innecesarios y usar entornos temporales
de pruebas son optimizaciones coherentes con lo observado. El cálculo excluye
impuestos, costos extra de red, respaldos, TLS, trabajo humano y reentrenamiento.

## Monitoreo y uso responsable

Evidently detectó una alerta en 1 de 15 variables: `month`, esperable al comparar
meses distintos. No hubo drift global con el criterio de 50%, ni alerta en las
tres entradas directas del modelo. Esto no demuestra ausencia de degradación o
concept drift. La política propone revisión mensual y evaluación de errores,
calidad y segmentos antes de entrenar o promover una alternativa.

No se afirma equidad por zona ni beneficio operativo real: no se dispone de
inventario observado, demanda no atendida, clima o rebalanceos completos.
Las dependencias históricas conservan avisos de seguridad documentados; el
servicio no está aprobado para exposición pública. El Model Card reúne estos
límites y los usos previstos. Continuous Training automático es una ampliación
opcional, no una función implementada que deba mostrarse en la demo.

## Comprobación de la demo

La preparación e ingesta se hacen antes de grabar. Desde la raíz, con el entorno
de operación instalado y la máquina de Podman activa:

```bash
python -m mibici_equipo14.operations up
podman-compose -f docker-compose.yml ps
python -m mibici_equipo14.operation_check
curl -sS http://127.0.0.1:8014/health
curl -sS -X POST http://127.0.0.1:8014/predict -H 'Content-Type: application/json' --data-binary @references/demo_request.json
curl -sS -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8014/predict -H 'Content-Type: application/json' --data-binary @references/demo_invalid.json
```

La petición válida devuelve tres predicciones en el mismo orden; la primera y
la tercera deben coincidir. La segunda petición debe devolver 422. En Swagger,
`http://127.0.0.1:8014/docs` permite ejecutar las mismas peticiones y mostrar el
JSON sin depender de una terminal. MLflow se abre en `http://127.0.0.1:5015`.
Para finalizar, `podman-compose -f docker-compose.yml down` conserva el volumen;
no se añade `-v`.

La [verificación de cierre](../reports/cierre/validacion.json) conserva el commit
de código comprobado, sus huellas, los conteos de pruebas y las respuestas HTTP.
La reconstrucción de contenedores se verificó sin borrar el volumen del Registry.
El estado `healthy` es una condición previa a enviar la petición de la demo.

Las capturas históricas respaldan las mediciones publicadas. La demo de cierre
debe mostrar una petición real durante la grabación y no una captura presentada
como ejecución nueva. El video dura entre dos y cinco minutos, es continuo y
complementa la presentación y el enlace a este repositorio.

## Referencias

Gift, N., & Deza, A. (2021). *Practical MLOps: Operationalizing machine learning models*. O'Reilly Media.

Huyen, C. (2022). *Designing machine learning systems: An iterative process for production-ready applications*. O'Reilly Media.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220–229. https://doi.org/10.1145/3287560.3287596

MLflow. (s. f.). *MLflow Model Registry*. https://mlflow.org/docs/latest/ml/model-registry/

Hugging Face. (s. f.). *Model cards*. https://huggingface.co/docs/hub/model-cards

Podman. (s. f.). *podman-machine*. https://docs.podman.io/en/latest/markdown/podman-machine.1.html
