# Monitoreo de demanda P1

## Resultado principal

Evidently no detectó drift global con los métodos y umbrales establecidos.
Solo `month` superó el umbral: 1 de 15 variables, equivalente a 6.67%.
Las tres entradas del modelo y las once variables de demanda e historia
quedaron por debajo de 0.10. Esto describe estabilidad de las distribuciones
marginales analizadas; no demuestra que la precisión del modelo siga igual.

## Periodos y preparación

Se comparó enero-agosto de 2025 con septiembre-octubre de 2025. El primer
periodo corresponde al entrenamiento original de P1 y el segundo representa
un periodo posterior de operación. Noviembre-diciembre se excluyó del análisis
y se conserva para la comparación final de versiones.

Se mantuvieron las 369 estaciones de la cohorte original y la malla de una
observación por estación y hora, incluyendo las horas con cero viajes. La
referencia tiene 2,090,016 observaciones y el periodo actual 540,216. Las
primeras 168 horas por estación se excluyeron por falta de historia:
61,992 filas en total. Por ello, la referencia efectiva empieza el 8 de enero.
Los rezagos y las medias móviles solo usan horas anteriores al objetivo.

Se usó la versión de datos
`97b8d9d898ca5f71ffaffc6ff86b100fbe371abfd5d67837d989e5f81bf9934f`.
El programa verifica las huellas del manifiesto y los Parquet antes de generar
el reporte. No usa datos personales ni modifica el modelo del Registry.

## Método y calidad

Evidently 0.7.20 comparó muestras aleatorias de 50,000 filas por periodo, sin
reemplazo y con semilla 42. Se fijó Jensen-Shannon para las variables
categóricas y Wasserstein normalizada para las numéricas. La alerta por
variable es una distancia mayor o igual a 0.10. Son medidas de cambio,
no pruebas de significancia basadas en p-valores. El criterio global es
al menos 50% de variables con alerta. `station_id`, `weekday`, `hour` y
`month` se declararon como categóricas para evitar interpretar sus códigos
como magnitudes continuas.

La calidad se revisó también sobre todas las observaciones. No se encontraron
faltantes en las variables analizadas, valores numéricos no finitos,
duplicados de estación-hora ni incumplimientos de los rangos de hora, día
de semana o conteos de salida. Esto no descarta todos los posibles errores
de origen. Un cero representa ausencia de viajes registrados, no confirma
que hubiera bicicletas o anclajes disponibles.

En septiembre-octubre hubo 796,853 viajes en todas las estaciones. De ellos,
122 salieron de una estación fuera de la cohorte de entrenamiento. Se
documentaron por separado; la comparación principal conserva la cohorte
original y no representa toda la expansión del sistema.

## Variables e interpretación

| Variable | Distancia | Método | Alerta |
|---|---:|---|---|
| station_id | 0.041286 | Jensen-Shannon | No |
| weekday | 0.020787 | Jensen-Shannon | No |
| hour | 0.010694 | Jensen-Shannon | No |
| departures | 0.041007 | Wasserstein normalizada | No |
| month | 0.832555 | Jensen-Shannon | Sí |

Los diez rezagos y medias móviles tuvieron distancias entre 0.019779 y
0.040310, sin alerta. La única variable afectada fue `month`, porque la
referencia contiene enero-agosto y el periodo actual septiembre-octubre.
Este cambio de calendario es esperado. HistoricalMean no usa el mes como
entrada, de modo que esta señal no justifica por sí sola sustituir el modelo.

La media de salidas pasó de 1.4142 a 1.4748 por estación-hora, un aumento de
4.29%. La proporción de horas sin salidas bajó de 52.66% a 51.58%.
Se comparan promedios y proporciones, ya que los periodos tienen distinta
duración; comparar sus totales directamente sería engañoso.

Los cambios por estación merecen atención aunque no haya alerta global.
La estación 34 pasó de 3.4303 a 1.5437 salidas por hora; la 194 pasó de
4.4688 a 6.2561. Un promedio histórico podría sobreestimar la primera y
subestimar la segunda si esos patrones continúan. Los promedios por estación
no controlan la mezcla de días y horas, por lo que son señales descriptivas,
no prueba de concept drift ni de una causa concreta. No se atribuyen estos
cambios a lluvia, vacaciones o infraestructura sin datos que lo comprueben.

La malla fija hace que las distribuciones de estación y hora sean muy
estables por construcción. Por eso, la ausencia de drift en estas columnas
no sustituye la evaluación de errores por estación y franja horaria.
Además, el muestreo y las comparaciones univariadas pueden pasar por alto
eventos raros, cambios conjuntos o diferencias locales. En la muestra de
Evidently aparecen máximos de 207 salidas en el periodo actual y 44 en la
referencia; son máximos muestrales, no máximos poblacionales. Conviene revisar
estos casos por su posible efecto sobre RMSE antes de excluirlos o corregirlos.

## Recomendación de seguimiento

No se recomienda cambiar automáticamente el alias `champion` a partir de
este reporte. La decisión de conservar o promover una versión debe apoyarse
en el error de ambas sobre exactamente las mismas observaciones de
noviembre-diciembre, conservadas fuera del entrenamiento del challenger.
El ejercicio es una comparación histórica pareada; no se presenta como
un experimento con tráfico real asignado al azar.

Para operación se propone revisar mensualmente los datos, la cobertura de
estaciones y los errores cuando estén disponibles los valores reales. Los
problemas de calidad se investigan antes de reentrenar. Una alerta persistente
durante dos ventanas completas, o deterioro de MAE superior a 10% frente a
un nivel de referencia previamente fijado para el mismo servicio, activaría
una revisión y entrenamiento candidato. Estos son umbrales operativos
propuestos, todavía no calibrados con operación real.

El candidato debe conservar su linaje, pasar las pruebas de calidad y
reproducibilidad y compararse en un periodo posterior sin usarlo para
ajustar el modelo. La promoción necesita mejora útil en MAE, revisión de
RMSE y WAPE y ausencia de deterioro inaceptable en estaciones relevantes.
Para estimar incertidumbre conviene remuestrear bloques temporales, por
ejemplo días completos, por la dependencia entre observaciones horarias.
El holdout final de esta entrega no debe reutilizarse repetidamente para
optimizar versiones futuras.

El aprendizaje principal fue separar una alerta de datos de una decisión
sobre el modelo. El marco metodológico permitió mantener los periodos,
las transformaciones y los criterios de comparación sin cambiar las reglas
después de observar el resultado. También mostró que un reporte global debe
acompañarse de revisión por segmentos para resultar útil en la operación.

## Reproducción y evidencias

El README contiene instalación y comandos para macOS, Linux y Windows.
`python -m mibici_equipo14.monitoring` genera el HTML, las métricas JSON,
el resumen de toda la población y las tres gráficas de apoyo. Las capturas
`04_evidently_resumen.jpg` y `05_evidently_demanda.jpg` proceden del HTML
generado por Evidently. Las tres imágenes PNG son gráficas calculadas con
los resultados; no se presentan como capturas de la herramienta.

## Referencias

Evidently AI. (s. f.). *Data and ML checks*. Recuperado el 15 de noviembre de
2026, de https://docs.evidentlyai.com/quickstart_ml

Evidently AI. (s. f.). *Data drift*. Recuperado el 15 de noviembre de 2026,
de https://docs.evidentlyai.com/metrics/explainer_drift

Huyen, C. (2022). *Designing machine learning systems: An iterative process
for production-ready applications*. O'Reilly Media.

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J.,
Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps:
How to scale AI in the enterprise*. O'Reilly Media.
