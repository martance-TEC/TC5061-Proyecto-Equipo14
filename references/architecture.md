# Arquitectura del proyecto MiBici

## Sistema integrado

```mermaid
flowchart TD
 A[CSV abiertos de MiBici] --> B[Ingesta y manifiesto SHA256]
 B --> C[Parquet local y variables temporales]
 C --> D[MLflow Projects y entrenamiento P1 P2 P3]
 D --> E[Tracking y Model Registry original]
 E --> F[Exportación trazable del modelo P1 v1]
 subgraph O[Compose en Podman]
  G[MLflow operativo y volumen persistente] --> H[Bootstrap y verificación de huella]
  H --> I[FastAPI con modelo cargado en memoria]
 end
 F --> H
 J[Cliente HTTP] --> I
 I --> K[Predicción JSON]
 C --> L[Evidently y calidad entre periodos]
 E --> M[A/B retrospectivo v1 frente a v2]
 L --> N[Revisión humana y política de actualización]
 M --> N
 I --> P[Locust y medición de recursos]
 P --> Q[FinOps y costo por inferencia]
 R[GitHub Actions] --> S[Pruebas y Bandit]
 S --> T[Construcción y despliegue temporal de Compose]
 U[Model Card] -. documenta .-> E
 U -. límites y uso .-> I
```

El Registry original y la copia operativa son distintos. La API usa P1 v1;
v2 es un challenger recomendado por la evaluación histórica, sin promoción
automática. El modelo se recupera por HTTP al arrancar y permanece en memoria.
SQLite y los artefactos operativos viven en el volumen `registry-data`, fuera
de Git. API y MLflow se conectan por la red interna de Compose. El bootstrap
es una tarea de inicialización, no un tercer servicio de inferencia.

La orquestación y el despliegue demostrados son locales. CI realiza una prueba
temporal con un ejemplo sintético marcado; las mediciones de carga publicadas
usan el P1 real. Evidently y A/B se ejecutan por lotes sobre periodos históricos;
no son una plataforma de monitoreo de tráfico continuo. No hay rebalanceo
automático ni infraestructura cloud contratada.

## Experimento inicial

```mermaid
flowchart TD
 A[Portal oficial MiBici CSV mensual y catálogo con capacidad] --> B[dataset.py descarga validación SHA256]
 B --> C[Parquet local ignorado por Git]
 B --> D[Manifiesto versionado en references]
 C --> E[features.py eventos horarios y etiquetas por minuto]
 E --> F[P1 regresión demanda a una hora]
 E --> G[P2 clasificación presión de capacidad a tres horas]
 E --> H[P3 perfiles de estaciones y KMeans]
 I[MLproject parámetros y entorno fijado] --> J[pipeline.py]
 J --> F
 J --> G
 J --> H
 F --> K[MLflow SQLite local y artefactos locales]
 G --> K
 H --> K
 K --> L[Resultados y evidencias de la propuesta]
 M[Git y PR con revisión] --> I
```

Python permite funciones comprobables y clases por responsabilidad. Pandas y PyArrow procesan datos tabulares sin incorporar viajes al historial. Scikit-learn compone transformaciones ajustadas solo en entrenamiento y modelos. MLflow Projects define un punto de entrada parametrizado; Tracking conserva procedencia, métricas y artefactos. SQLite y archivos locales permiten trabajar en la laptop sin infraestructura de nube. GitHub conserva historial y revisiones. CCDS separa datos, código, exploración y resultados.

El diagrama describe el experimento offline. No existe servicio productivo, rebalanceo automatizado ni validación con inventario observado.
