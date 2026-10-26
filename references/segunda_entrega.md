# Segunda entrega

La campaña conserva las fuentes, las particiones y las métricas de la primera entrega. El historial local anterior no se conservó al cambiar de computadora. Por eso los baselines y los controles de ganadores anteriores se reconstruyen con `reference_kind=baseline_replay`, sin reutilizar los antiguos identificadores de corrida. Las métricas anteriores siguen disponibles en los reportes históricos.

## Protocolo fijado antes de ejecutar

- P1: baseline semanal; control de promedio histórico; HGB Poisson (150, 0.10, 15 hojas), (250, 0.05, 31 hojas), (350, 0.03, 31 hojas). Selección por MAE de validación y tolerancia 0.01 a favor del modelo sencillo.
- P2: Dummy prior; control RF 200 árboles, profundidad 12, hoja mínima 10; candidatos RF (100, 8, 20), (300, 12, 10), (300, 16, 20). Misma muestra de hasta 100000 filas de entrenamiento, semilla 42. Umbral por F2 de validación entre 0.10 y 0.90; empate favorece el umbral mayor. Modelo elegido por F2 de validación.
- P3: baseline k=2; control k=3; candidatos k=4, 5 y 6. StandardScaler más KMeans con n_init=20. Grupos de al menos cinco estaciones en validación; entre opciones a 0.01 del mejor silhouette admisible, se elige menor k.

Cada búsqueda tiene padre e hijas reales. La selección se guarda antes de calcular las métricas de prueba de los candidatos. Noviembre y diciembre ya fueron analizados anteriormente: siguen siendo un benchmark histórico comparable, no una prueba externa nueva.

En P2 se conserva la etiqueta proxy basada en capacidad oficial y ocupación inicial hipotética. No se interpreta como desabasto observado. El objeto pyfunc incluye la probabilidad y la decisión con el umbral congelado. En P3 el scaler queda dentro del modelo. El promedio histórico de P1 se guarda como estimador cargable con respaldo global.

## Artefactos y linaje

Cada run guarda firma, ejemplo, modelo, manifiesto en `data/`, configuración y lock en `config/`, gráficas en `plots/` y evaluación en `evaluation/`. `mlflow.log_input` registra el conjunto efectivo de entrenamiento, sin columnas personales. La versión de fuentes se distingue del digest calculado por MLflow.

Los modelos se registran como `mibici-equipo14-p1-demanda`, `mibici-equipo14-p2-riesgo` y `mibici-equipo14-p3-grupos`. Se carga la versión numérica, se comparan sus predicciones y después se asigna `champion`. Cada versión conserva commit, versión de datos y run de origen. El alcance es académico y local: el alias no demuestra un despliegue real en MiBici.

## Reproducción entre entornos

La comprobación independiente conserva las fuentes, particiones y configuraciones de la campaña. P1 coincide en sus 18 métricas comparadas. P3 presenta una diferencia máxima de 4.440892098500626e-16. En P2 se observó una diferencia máxima de 0.001007361487795455 y se adoptó un límite absoluto de 0.002 para la reproducción, frente a 1e-8 para P1 y P3. Es una comprobación con tolerancia, no una coincidencia exacta. La repetibilidad local no permite atribuir por sí sola la diferencia a una causa específica del entorno.

El cambio afecta únicamente la aceptación de reproducciones y no modifica las métricas, el entrenamiento ni la selección de la campaña. La verificación de carga por nombre y alias conserva su tolerancia de 1e-8. Las evidencias de comprobación guardan el código, los datos y los identificadores de sus corridas locales.
