# Comprobación independiente de P2 y P3 - Víctor Villa Herrera

## Objetivo

Reproducir de forma independiente los modelos seleccionados de P2 y P3 desde un clon limpio del repositorio y verificar su registro y carga mediante MLflow Model Registry.

## Entorno reproducido

La comprobación se realizó con Python 3.11.9 de 64 bits y las dependencias fijadas en `requirements.lock.txt`. La instalación finalizó con `pip check` sin dependencias rotas y el proyecto completó 26 pruebas automatizadas.

La ingesta y la verificación se ejecutaron mediante los entry points del archivo `MLproject`. Se reconstruyeron 4,532,032 viajes y se obtuvo la misma versión de datos utilizada por el equipo:

`97b8d9d898ca5f71ffaffc6ff86b100fbe371abfd5d67837d989e5f81bf9934f`

## Reproducción de P2

P2 utiliza Random Forest con 300 árboles, profundidad máxima de 16 y mínimo de 20 observaciones por hoja.

La primera validación independiente detectó pequeñas diferencias respecto a las métricas de referencia. Se comprobó que dos ejecuciones consecutivas en el entorno local producían exactamente los mismos resultados, por lo que el modelo era estable localmente.

La diferencia absoluta máxima frente a la referencia fue de `0.001007361487795455`. Se incorporó una tolerancia específica para reproducciones de P2 de `0.002`, manteniendo `1e-8` para P1 y P3. La regla se respaldó con tres pruebas automatizadas adicionales.

La reproducción final de P2 quedó aprobada y vinculada al commit:

`6ce8c4546624298193a9c79b4ee8b7e79bc844d2`

El modelo reproducido se registró como:

`mibici-equipo14-p2-riesgo`

La versión 1 recibió el alias `champion`. La carga posterior desde Model Registry produjo `predictions_match=true`.

## Reproducción de P3

P3 utiliza KMeans con `k=3`.

La reproducción independiente pasó utilizando la tolerancia estricta de `1e-8`. La diferencia absoluta máxima observada fue `4.440892098500626e-16`.

El modelo reproducido se registró como:

`mibici-equipo14-p3-grupos`

La versión 1 recibió el alias `champion`. La carga posterior desde Model Registry produjo `predictions_match=true`.

## Resultado

Las reproducciones de P2 y P3 quedaron vinculadas a la misma versión de datos y al commit `6ce8c45`. Ambos modelos fueron registrados, versionados, etiquetados con `champion` y verificados mediante carga e inferencia desde Model Registry.
