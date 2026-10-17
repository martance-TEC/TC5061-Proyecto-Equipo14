# Variables y arquitectura: revisión del rol Persona 3

Responsable asignado: Víctor Villa Herrera. Ejecución técnica asistida solicitada
por Martha; cuenta y autoría reales martance-TEC. No acredita participación
independiente de Víctor ni sustituye su revisión personal de un PR ajeno.

## Decisiones verificadas

| Pregunta y usuario | Variables y objetivo | Prevención de fuga |
|---|---|---|
| P1: regresión, operador de flota | Salidas por estación en la próxima hora; historia de salidas/ingresos y calendario | Rezagos y medias desplazadas; horizonte íntegro dentro del split |
| P2: clasificación, equipo de rebalanceo | Riesgo aproximado en las próximas tres horas con capacidad real C del catálogo; inventario inicial supuesto 0.5 C | Eventos futuros solo para la etiqueta; capacidad y predictores históricos para el modelo; purga de tres horas |
| P3: clustering, planeación urbana y expansión | 99 variables de perfiles de uso por estación; mapa posterior con coordenadas | Train/val/test separados; StandardScaler y KMeans ajustados en train; coordenadas fuera del ajuste |

P2 usa salidas con signo negativo e ingresos al finalizar el viaje con signo
positivo. Se declara riesgo cuando el mínimo del balance es <= 0.1 C o el
máximo es >= 0.9 C en la ventana; incluye extremos intermedios. El inventario
inicial es hipotético, no medido. Se evalúa sensibilidad a 0.25 C, 0.5 C y
0.75 C. Capacidad no positiva o desconocida excluye la estación de P2.
El catálogo de septiembre de 2026 no demuestra la capacidad histórica de 2025.

P2 compara DummyClassifier prior y RandomForest; selección por F2 de validación
para priorizar recall, con umbral elegido solo en validación. Se informan además
precision, recall, average precision, balanced accuracy y matriz de confusión.
P3 compara KMeans k=2 a k=6; selección por silhouette de validación, grupos de
al menos cinco estaciones y tolerancia 0.01 que favorece menor k. No se usa
accuracy para un problema sin etiquetas de grupos reales.

## Refactorización y OOP

features.py contiene funciones reutilizables y ahora documenta entradas,
supuestos, finalidad y límites. Se verificó que agregar docstrings no cambia
la estructura del código ejecutable. Dos nuevas pruebas comprueban que los
eventos futuros y las coordenadas no alteran los perfiles de entrenamiento.
Las pruebas existentes cubren capacidad, extremos intermedios, rezagos y purga.

Settings concentra configuración; ExperimentTracker concentra trazabilidad;
RegressionExperiment, ClassificationExperiment y ClusteringExperiment agrupan
la ejecución de cada tarea. La composición con pipelines de scikit-learn evita
ajustar transformaciones con validación o prueba. No se agrega herencia sin uso.
Arquitectura editable: references/architecture.md; representación: architecture.svg.
El sistema es un experimento offline, no una aplicación operativa desplegada.

## Pendientes de colaboración

Víctor debe aceptar su invitación, revisar realmente un PR de otro integrante,
abrir su propio PR desde su cuenta y registrar una aportación auténtica.
Los nombres de carpeta y etiquetas de rol no prueban autoría personal.
