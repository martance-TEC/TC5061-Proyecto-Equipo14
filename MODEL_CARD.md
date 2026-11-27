# Model Card: P1 Demanda por Estación

## 1. Información general

**Nombre registrado:** `mibici-equipo14-p1-demanda`

**Pregunta del proyecto:** P1, demanda por estación para la siguiente hora.

**Tipo de problema:** regresión.

**Familia del modelo:** `HistoricalMean`.

**Equipo:** Equipo 14, TC5061.

Este Model Card documenta el modelo P1 utilizado como servicio en el proyecto integrador, incluyendo su propósito, datos, desempeño, linaje, limitaciones, monitoreo y condiciones de uso. Complementa el README del repositorio, que contiene las instrucciones técnicas de reproducción y operación del sistema.

## 2. Descripción del modelo

P1 estima el número esperado de salidas de bicicletas por estación para la siguiente hora.

El modelo `HistoricalMean` aprende el promedio histórico de salidas para cada combinación de:

- `station_id`: identificador de la estación;
- `weekday`: día de la semana;
- `hour`: hora del día.

Cuando una combinación no tiene historia disponible, el modelo utiliza una media global de respaldo.

La predicción representa una estimación de demanda histórica esperada. No representa el inventario físico de bicicletas, la disponibilidad de anclajes ni la demanda no atendida.

### Entradas

| Variable | Descripción |
|---|---|
| `station_id` | Identificador de la estación |
| `weekday` | Día de la semana |
| `hour` | Hora del día |

### Salida

`departures`: número esperado de salidas durante la siguiente hora.

## 3. Uso previsto

El modelo fue desarrollado para el proyecto académico MiBici y para demostrar un ciclo completo de Machine Learning con prácticas de MLOps.

Los usos previstos incluyen:

- estimar patrones históricos de demanda por estación y hora;
- apoyar análisis de planeación de demanda;
- comparar versiones del modelo bajo las mismas condiciones;
- proporcionar inferencias mediante una API reproducible;
- demostrar seguimiento de experimentos, versionado, despliegue, monitoreo y evaluación operacional.

El modelo no constituye un sistema oficial de MiBici ni demuestra beneficios operativos reales en producción.

## 4. Usos no recomendados

P1 no debe utilizarse por sí solo para:

- determinar la disponibilidad exacta de bicicletas;
- determinar la disponibilidad de anclajes;
- medir demanda no atendida;
- ejecutar automáticamente decisiones de rebalanceo de flota;
- extrapolar con confianza a estaciones nuevas sin historia;
- atribuir cambios de demanda a clima, eventos, infraestructura u otras causas no incorporadas al modelo;
- tomar decisiones operativas críticas sin validación humana y datos adicionales.

Las métricas del proyecto corresponden a evaluaciones históricas. Una mejora estadística no implica automáticamente una mejora operacional.

## 5. Datos

La fuente principal corresponde a los datos abiertos de MiBici utilizados por el proyecto para el año 2025.

**Versión de datos:**

`97b8d9d898ca5f71ffaffc6ff86b100fbe371abfd5d67837d989e5f81bf9934f`

La cohorte principal de P1 contiene 369 estaciones con historia suficiente para la preparación temporal definida por el proyecto.

El pipeline conserva trazabilidad mediante manifiestos, hashes criptográficos y archivos Parquet procesados. La preparación construye una malla estación-hora que permite incluir horas con cero salidas registradas.

Un valor igual a cero significa ausencia de viajes registrados durante esa estación-hora. No demuestra que existieran bicicletas o anclajes disponibles.

### Periodos utilizados

| Uso | Periodo |
|---|---|
| Entrenamiento original de P1 v1 | enero-agosto de 2025 |
| Información disponible para challenger v2 | enero-octubre de 2025 |
| Holdout final | noviembre-diciembre de 2025 |

Noviembre y diciembre permanecieron fuera del entrenamiento del challenger para conservar un periodo posterior de evaluación.

## 6. Feature engineering

El proyecto transforma los datos de viajes en observaciones temporales por estación y hora.

La preparación general incluye variables como:

- identificador de estación;
- hora;
- día de la semana;
- mes;
- rezagos de salidas;
- rezagos de llegadas;
- medias móviles de actividad.

P1 utiliza directamente tres entradas:

- `station_id`;
- `weekday`;
- `hour`.

Otras características temporales son utilizadas por el pipeline general, las demás preguntas del proyecto y el monitoreo.

Los rezagos y medias móviles se construyen utilizando únicamente información anterior al objetivo, con el propósito de evitar leakage temporal.

La implementación del feature engineering se encuentra principalmente en `mibici_equipo14/features.py`.

## 7. Experimentación con MLflow

El proyecto utiliza MLflow para registrar y reproducir el proceso de experimentación.

El repositorio incluye `MLproject` y `python_env.yaml`, además de entry points para diferentes etapas del flujo.

Los runs registran elementos como:

- configuración del modelo;
- semilla;
- periodos temporales;
- hiperparámetros cuando corresponden;
- métricas de validación y prueba;
- versión de datos;
- commit de Git;
- estado del código;
- artefactos de evaluación;
- firma del modelo.

Esta información permite vincular una versión del modelo con el código y los datos utilizados para producirla.

## 8. Model Registry y linaje

El modelo P1 está registrado en MLflow Model Registry con el nombre:

`mibici-equipo14-p1-demanda`

El proyecto conserva trazabilidad entre el modelo, la ejecución de MLflow, los datos y el código utilizado durante el entrenamiento.

La evidencia registrada incluye:

- versión del modelo;
- run de origen;
- commit de Git;
- versión de datos;
- firma de entrada;
- validación de carga e inferencia.

### Versión operativa demostrada

La API utilizada en la operación final carga explícitamente:

**P1 versión 1**

La evidencia reproducible del servicio registra:

- modelo: `mibici-equipo14-p1-demanda`;
- versión: `1`;
- URI: `models:/mibici-equipo14-p1-demanda/1`;
- run de origen: `48bc3dd1f2414f4ba31f9f2020226764`;
- commit de entrenamiento: `5373f1e79bb469c3c4e0da2481e59ea275b6789b`;
- versión de datos: `97b8d9d898ca5f71ffaffc6ff86b100fbe371abfd5d67837d989e5f81bf9934f`;
- firma del modelo verificada.

La versión numérica se fija en el servicio para evitar que un cambio posterior de alias modifique silenciosamente el modelo utilizado por la API.

### Ciclo champion/challenger

En la etapa de monitoreo se registró una segunda versión de P1 como `challenger`.

El estado evaluado fue:

- P1 v1: `champion`;
- P1 v2: `challenger`.

P1 v2 incorpora información disponible hasta octubre de 2025 y fue evaluado posteriormente con noviembre-diciembre como holdout.

La evaluación estadística recomienda considerar la promoción de v2. Sin embargo, la operación demostrada en la entrega final continúa utilizando P1 v1.

Por lo tanto, este Model Card distingue entre:

- **modelo operativo demostrado:** P1 v1;
- **modelo challenger recomendado para promoción:** P1 v2.

No se declara que P1 v2 haya sido promovido o desplegado cuando dicha acción no forma parte de la evidencia operativa final.

## 9. Evaluación champion vs challenger

Champion v1 y challenger v2 se compararon sobre exactamente las mismas 540,216 observaciones del holdout noviembre-diciembre de 2025.

| Métrica | Champion v1 | Challenger v2 | Cambio |
|---|---:|---:|---:|
| MAE | 0.877629 | 0.860265 | -1.98% |
| RMSE | 1.594179 | 1.549914 | -2.78% |
| WAPE | 0.662798 | 0.649684 | -1.98% |

El challenger presentó menor error en las tres métricas principales.

En la comparación pareada por observación:

- challenger mejor: 244,964 observaciones;
- champion mejor: 193,832 observaciones;
- empates: 101,420 observaciones.

El delta MAE se definió como:

`error challenger - error champion`

Por lo tanto, valores negativos favorecen al challenger.

Con 1,000 iteraciones de bootstrap pareado y semilla 42 se obtuvo un IC 95% de delta MAE de:

`[-0.017799, -0.016933]`

El intervalo se encuentra completamente por debajo de cero y favorece a P1 v2.

## 10. Validación temporal

Las observaciones horarias consecutivas no necesariamente son independientes. Por ello se complementó el bootstrap por filas con un bootstrap por bloques diarios.

Resultados:

- días evaluados: 61;
- challenger mejor: 57 días, 93.44%;
- champion mejor: 4 días, 6.56%;
- empates: 0;
- IC 95% diario de delta MAE: `[-0.020158, -0.014708]`.

El intervalo temporal también permanece completamente por debajo de cero.

La comparación es retrospectiva y offline. No corresponde a un experimento A/B online con usuarios asignados aleatoriamente a cada versión.

## 11. Monitoreo y drift

El proyecto incorpora monitoreo de calidad y drift con Evidently.

Los periodos definidos fueron:

| Periodo | Uso |
|---|---|
| enero-agosto de 2025 | referencia |
| septiembre-octubre de 2025 | periodo actual |
| noviembre-diciembre de 2025 | holdout excluido del monitoreo |

Se monitorearon 15 variables utilizando criterios previamente establecidos.

Resultado:

- variables monitoreadas: 15;
- variables con alerta: 1;
- proporción con alerta: 6.67%;
- variable con alerta: `month`;
- criterio global: 50%;
- conclusión: no se detectó drift global.

La señal de `month` es consistente con la comparación entre periodos que contienen meses distintos.

`HistoricalMean` no utiliza `month` como entrada directa. Por lo tanto, esta señal por sí sola no justifica sustituir el modelo.

La ausencia de drift global tampoco demuestra que el desempeño permanezca constante. Por esta razón el monitoreo de datos se complementó con evaluación de errores y comparación de versiones.

## 12. Limitaciones

Las principales limitaciones identificadas para P1 son:

1. El modelo depende de historia previa y no cubre adecuadamente estaciones nuevas sin historial suficiente.
2. No se dispone de inventarios históricos completos de bicicletas y anclajes para reconstruir la disponibilidad real.
3. No se modelan explícitamente los rebalanceos operativos realizados por MiBici.
4. Los ceros de la malla estación-hora representan ausencia de viajes registrados, no disponibilidad física confirmada.
5. No se incorporan variables externas como clima, eventos, cierres viales o condiciones extraordinarias.
6. No se modela explícitamente la demanda no atendida.
7. Las métricas corresponden a evaluación histórica y retrospectiva.
8. Una mejora estadística no demuestra automáticamente una mejora operacional.
9. Los patrones históricos pueden perder representatividad si cambia de forma persistente el comportamiento del sistema.
10. Las métricas globales pueden ocultar cambios importantes en estaciones o franjas horarias específicas.

Estas limitaciones deben considerarse antes de utilizar la predicción para decisiones operativas reales.

## 13. Consideraciones de uso responsable

P1 utiliza datos operativos abiertos y no requiere datos personales para realizar inferencias.

No obstante, las predicciones representan patrones aprendidos a partir de información histórica y deben interpretarse con precaución.

Se recomienda:

- mantener revisión humana para decisiones relevantes;
- investigar problemas de calidad antes de reentrenar;
- conservar trazabilidad de datos, versiones y código;
- evitar atribuir causalidad a variables que el modelo no observa;
- revisar estaciones y horarios donde las métricas agregadas puedan ocultar deterioros locales;
- validar nuevamente el modelo antes de utilizarlo fuera del contexto estudiado;
- conservar evidencia reproducible antes de cambiar la versión utilizada por el servicio.

El modelo forma parte de un proyecto académico y no constituye un sistema oficial de MiBici.

## 14. Despliegue y operación

El modelo P1 se expone mediante una API desarrollada con FastAPI.

La arquitectura operativa integra:

- FastAPI para inferencia;
- MLflow para el modelo registrado y su linaje;
- contenedor definido mediante `Containerfile`;
- `docker-compose.yml` para orquestar los servicios;
- almacenamiento persistente;
- red interna entre servicios;
- healthchecks;
- límites de recursos;
- pruebas automatizadas;
- GitHub Actions para CI/CD.

La API acepta peticiones de predicción con la firma de P1 y mantiene un límite de lote de hasta 1,000 observaciones por petición.

El pipeline de GitHub Actions ejecuta pruebas, análisis de seguridad, construcción de la imagen y comprobaciones de despliegue temporal.

La operación demostrada conserva P1 v1 de manera explícita y reproducible.

## 15. Desempeño bajo carga

El sistema fue evaluado con Locust utilizando escenarios de 5, 20 y 50 usuarios.

En el escenario de mayor carga se registraron:

- usuarios concurrentes: 50;
- solicitudes: 8,681;
- respuestas exitosas: 8,681;
- fallos: 0;
- solicitudes por segundo: 96.57;
- latencia media: aproximadamente 4.64 ms;
- latencia p95: 10 ms;
- latencia p99: 19 ms;
- CPU media de la API: aproximadamente 20.89% de un núcleo;
- pico de memoria de la API: aproximadamente 179.58 MiB;
- pico combinado de memoria de API y MLflow: aproximadamente 502.87 MiB.

Los tres escenarios de Locust acumularon 12,987 solicitudes sin fallos.

Estos datos corresponden a una prueba local de corta duración. No representan una prueba de saturación, una prueba prolongada ni un benchmark equivalente en infraestructura cloud.

## 16. FinOps

El proyecto incorpora una estimación reproducible del costo por inferencia.

La referencia económica utiliza:

- costo mensual de referencia: USD 24;
- horas de planeación mensual: 730;
- referencia de infraestructura: Amazon Lightsail Linux/Unix, 2 vCPU, 4 GB de memoria y 80 GB SSD.

Para el escenario de 50 usuarios:

- duración: 90 segundos;
- respuestas exitosas: 8,681;
- costo estimado por respuesta: aproximadamente USD 0.00000009468;
- costo estimado por 1,000 respuestas: aproximadamente USD 0.00009468.

El cálculo distribuye proporcionalmente el costo de la instancia durante la ventana observada.

Este valor no representa una factura real ni un benchmark ejecutado en AWS. La estimación excluye elementos como impuestos, egreso adicional, respaldos y retención adicional, balanceadores, TLS, CI/CD, operación humana y futuros entrenamientos.

El costo unitario depende de la utilización. Mantener infraestructura encendida con poco tráfico puede producir un costo por inferencia considerablemente mayor.

## 17. Gobernanza y ciclo de vida

El proyecto utiliza un ciclo de vida controlado para los modelos:

1. preparación y validación de datos;
2. feature engineering reproducible;
3. entrenamiento de candidatos;
4. seguimiento de experimentos con MLflow;
5. registro y versionado en Model Registry;
6. verificación de firma y linaje;
7. selección de una versión para consumo;
8. monitoreo de calidad y drift;
9. creación de challengers cuando existe evidencia suficiente;
10. evaluación sobre datos posteriores al entrenamiento;
11. comparación champion/challenger;
12. revisión humana antes de cualquier promoción.

No se recomienda reentrenar automáticamente ante cualquier alerta de drift.

La estrategia propuesta de Continuous Training considera iniciar una revisión cuando exista evidencia como:

- drift persistente durante dos ventanas completas;
- deterioro de MAE superior a 10% frente a una referencia comparable;
- cambios persistentes de cobertura;
- cambios relevantes por estación o franja horaria.

Una nueva versión debe mantener un periodo posterior fuera del entrenamiento y demostrar una mejora útil antes de sustituir al modelo utilizado por el servicio.

## 18. Reproducibilidad y evidencias

El repositorio conserva código, reportes y pruebas asociados al ciclo de vida del sistema.

Artefactos principales:

- `MLproject`
- `python_env.yaml`
- `mibici_equipo14/features.py`
- `mibici_equipo14/registry.py`
- `mibici_equipo14/serving.py`
- `mibici_equipo14/monitoring.py`
- `mibici_equipo14/ab_testing.py`
- `mibici_equipo14/ab_registry.py`
- `mibici_equipo14/finops.py`
- `Containerfile`
- `docker-compose.yml`
- `locustfile.py`
- `.github/workflows/operacion.yml`
- `reports/segunda_entrega/registry_verification.json`
- `reports/servicio/registry.json`
- `reports/monitoreo/`
- `reports/ab_testing/`
- `reports/operacion/`

El repositorio incluye pruebas automatizadas para datos, feature engineering, modelos, Registry, API, A/B testing, monitoreo, operación, límites HTTP y FinOps.

La integración de cambios se realizó mediante ramas, Pull Requests y revisiones de código, conservando las contribuciones individuales en el historial de Git.

## 19. Referencias

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). *Model Cards for Model Reporting*. Proceedings of the Conference on Fairness, Accountability, and Transparency.
- Hugging Face. *Model Cards*. Hugging Face Hub documentation.
- MiBici. *Datos abiertos*.
- MLflow. *MLflow Tracking and Model Registry*.
- Evidently AI. *Data and ML checks*.
- Amazon Web Services. *Amazon Lightsail bundles*.
- Evidencias reproducibles incluidas en el repositorio del Equipo 14.

---

## Estado documentado

**Modelo servido y demostrado operacionalmente:** P1 v1.

**Challenger evaluado:** P1 v2.

**Resultado de la evaluación A/B:** favorable a P1 v2.

**Recomendación:** considerar la promoción de P1 v2 después de la revisión correspondiente.

**Promoción de P1 v2 demostrada en la operación final:** no.

**Naturaleza del sistema:** proyecto académico de MLOps; no constituye un servicio oficial de MiBici.
