# Arquitectura del experimento MiBici

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
 M[Git y PR con revisión humana pendiente] --> I
```

Python permite funciones comprobables y clases por responsabilidad. Pandas y PyArrow procesan datos tabulares sin incorporar viajes al historial. Scikit-learn compone transformaciones ajustadas solo en entrenamiento y modelos. MLflow Projects define un punto de entrada parametrizado; Tracking conserva procedencia, métricas y artefactos. SQLite y archivos locales permiten trabajar en la laptop sin infraestructura de nube. GitHub conserva historial y revisiones. CCDS separa datos, código, exploración y resultados.

El diagrama describe el experimento offline. No existe servicio productivo, rebalanceo automatizado ni validación con inventario observado.
