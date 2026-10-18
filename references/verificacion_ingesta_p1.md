# Verificación de ingesta y P1

## Alcance y ejecución

Código ejecutado: `0e2e31579ea2d3611c2a55c8ac79c611ee421ccb`. Versión de datos: `97b8d9d898ca5f71ffaffc6ff86b100fbe371abfd5d67837d989e5f81bf9934f`. Los cambios de esta rama documentan la comprobación posterior al entrenamiento. La ingesta reconstruyó los Parquet a partir de copias locales de las fuentes oficiales; no se afirma una descarga nueva desde Internet.

## Fuentes y condiciones de uso
Consulta del 17 de octubre de 2026. Portal: https://mibici.net/es/datos-abiertos/ . Las trece URL exactas constan en references/sources.json. Se consultaron https://mibici.net/es/aviso-de-privacidad/ y https://mibici.net/es/termino-y-condiciones/ . El aviso describe tratamiento de datos personales y los términos corresponden al servicio de bicicletas. No se encontró allí una licencia específica que permita afirmar CC0, MIT o dominio público para los CSV. No se redistribuyen viajes individuales.

## Tamaño y variables
Se verificaron 12 archivos mensuales de 2025 y un catálogo de estaciones, 393,104,868 bytes en total. Entrada y viajes utilizables: 4,532,032 y 4,532,032. Duplicados eliminados: 0; conflictos: 0; inválidos: 0. Catálogo: 484 estaciones, 480 con capacidad positiva; cuatro entradas sin capacidad positiva no aportan etiqueta de presión válida. Son conteos del catálogo, no cuatro estaciones necesariamente presentes en todas las particiones.

Se conservan identificador del viaje, inicio, fin, estación de origen y destino; no se importan los campos demográficos del CSV. Los identificadores se leen como texto y las fechas con formato explícito. El manifiesto contiene tamaño, codificación y SHA256 de cada fuente. Los tamaños evaluables de P1 son {"train": 2101344, "val": 543144, "test": 543144} y se obtuvieron del artefacto evaluation/definition.json.

## Validaciones y temporalidad
La ingesta terminó con código de salida cero, reconstruyó el conjunto combinado y los archivos mensuales y confirmó la versión de datos. Se ejecutaron ocho pruebas: variación del hash al cambiar una fuente, estabilidad de la versión canónica, dependencia del riesgo respecto de la capacidad, detección de extremos intermedios, capacidad positiva obligatoria, rezagos sin eventos futuros, purga del horizonte y llegada contabilizada por hora de fin.

Entrenamiento: enero a agosto; validación: septiembre y octubre; prueba: noviembre y diciembre de 2025. Las fronteras se implementan con límites temporales explícitos y purga del horizonte. Los rezagos de 1, 24 y 168 horas y las medias móviles desplazadas utilizan pasado. La selección del modelo se hace con validación y no con prueba.

## P1 y resultados
Tarea: regresión del número de viajes iniciados por estación durante la siguiente hora. Usuario: operador del sistema, para planear la flota. Baseline: misma hora de la semana anterior. Candidato: HistGradientBoostingRegressor, pérdida Poisson, 100 iteraciones, tasa 0.1, 31 hojas y semilla 42. Se selecciona por val_mae; la tolerancia 0.01 favorece al baseline.

Baseline val_mae=1.02801835, test_mae=0.99035983. Candidato val_mae=0.85398633, test_mae=0.80511835, test_rmse=1.43027496, test_wape=0.61096252. Reducción de MAE de prueba frente al baseline: 18.70 %. Esta diferencia es una evaluación offline y no mide impacto operativo. Modelo seleccionado: hist_gradient_boosting.

Run padre de P1: `8c84139031d74630b7ea76eecda4e238`. Run padre completo: `8505a490438d4bd6983175e431e63e2e`. El ensayo P1 produjo dos hijos y el completo produjo nueve hijos: dos de P1, dos de P2 y cinco de P3. Todos finalizaron correctamente. Se compararon 103 valores numéricos con la reproducción anterior y todos coinciden dentro de 1e-12.

## P2 y P3 y límites
P2 conserva clasificación para el equipo de rebalanceo. Usa capacidad real del catálogo y un inventario inicial hipotético de 50 %; detecta presión extrema por flujos en tres horas. El catálogo es de septiembre de 2026 aplicado retrospectivamente a 2025. No hay inventario ni rebalanceos históricos observados, por lo que no equivale a predecir faltantes reales. P3 conserva clustering de perfiles para planeación urbana y expansión; seleccionó k=3 sobre 369 estaciones elegibles.
