# Calidad y seguridad

La ejecución conjunta comprobó 86 pruebas, sin fallos, errores ni casos omitidos,
tras integrar las pruebas de ingesta, P1 y validación de perfiles de P2 y P3.
La comprobación anterior desde un clon separado incluyó 71 pruebas y utilizó
el mismo entorno local; no equivale a una instalación nueva en Windows o Linux.
El commit, las versiones y las cifras completas están en [resultados.json](resultados.json).

## Pruebas y reproducibilidad

Se conservaron las pruebas anteriores y se añadieron casos de descarga segura,
integridad, ventanas de capacidad y repetibilidad. La prueba de entrenamiento
construye y ajusta dos modelos nuevos con los mismos datos y semilla: HGB para P1,
Random Forest para P2 y StandardScaler con KMeans para P3. Las predicciones
coinciden con tolerancia absoluta de 1e-12 dentro del entorno comprobado.
En P2 también se comparan las probabilidades; en P3 se verifica que el escalador
se ajuste solamente con entrenamiento.

Las pruebas de datos alterados comprueban que se detecten faltantes, claves
duplicadas, valores no finitos, tipos incorrectos, conteos negativos y fechas
fuera de su partición. Una distribución artificialmente desplazada activa la
condición de drift. De esta manera se comprueba la capacidad de detectar errores,
además del comportamiento con datos válidos.

La cobertura global es cercana al 40 %. El módulo de variables alcanza cerca del
98 %. La diferencia se debe a que el cálculo global también incluye orquestación,
gráficos, servidores y operaciones de Registry no ejecutadas por las pruebas
unitarias. No se afirma cobertura completa del sistema ni validación operativa
en producción.

## Datos reales y drift

La ingesta verificada conservó 4,532,032 viajes, 484 estaciones en catálogo y
480 con capacidad positiva. Las reglas de dominio revisaron todas las filas de
las particiones preparadas. Deepchecks utilizó muestras fijas de hasta 10,000
filas por partición para P1 y P2, y las 369 estaciones de P3.

Pasaron las condiciones de integridad. En las muestras de P1 y P2 no se activó
la alerta de drift con umbral 0.2; esto no demuestra ausencia de cambios en toda
la población. En P3, nueve de las 99 variables superaron ese umbral. El mayor
valor KS fue aproximadamente 0.3089 para `arrivals_25`.

La alerta señala cambios entre los perfiles de entrenamiento y prueba. Puede
estar relacionada con patrones temporales de uso, pero el chequeo no identifica
su causa ni demuestra por sí solo una degradación del modelo. Se conservó el
hallazgo, sin modificar fechas, datos ni umbral. Antes de utilizar los grupos
para decisiones, es necesario revisar sus perfiles y su desempeño en periodos
adicionales. Esto se suma a la separación débil de grupos observada en la entrega
anterior.

## Código, archivos y dependencias

Bandit terminó con cero hallazgos y cero errores de análisis. La descarga quedó
restringida al portal oficial mediante HTTPS, sin redirecciones. Las respuestas
incompletas no reemplazan archivos válidos. La validación de unicidad del catálogo
ya no depende de `assert`. Las llamadas locales sin shell conservan una justificación
específica, y el lector XML usa defusedxml.

La revisión de texto de los archivos actuales y del historial local no encontró
los patrones de credenciales buscados ni archivos de datos o modelos excluidos.
Este resultado tiene un alcance definido: no analiza el contenido comprimido
de PDF, Word o imágenes y no garantiza que no exista ninguna forma de secreto.

pip-audit encontró avisos en 11 paquetes. El resultado agregado conserva 54
identificadores distintos por paquete; algunas entradas del servicio se repiten,
por lo que no se sumaron dos veces. Entre los paquetes afectados se encuentran
MLflow, cryptography, requests y PyArrow. La lista y las versiones sugeridas por
el servicio están en el reporte de resultados. No se aplicaron exclusiones de
avisos para obtener una auditoría limpia.

El entorno histórico mantiene sus versiones de entrenamiento para conservar la
referencia de resultados. pip se actualizó a 26.2.2 al resolver las herramientas
de calidad. Los avisos restantes requieren una actualización controlada y pruebas
de compatibilidad, carga de modelos y reproducción antes de sustituir el entorno.
Hasta completar esa revisión, no se considera adecuado exponerlo como servicio
público. Mantener los servidores en localhost reduce la exposición, pero no
elimina las vulnerabilidades.

`quality all` comprueba pruebas, datos, Bandit y archivos locales. La auditoría de
dependencias se incluye expresamente con `--audit-dependencies`; mientras existan
avisos, ese comando devuelve código 1. Un Bandit limpio no debe interpretarse como
una auditoría de dependencias aprobada.

## Evidencias

- [Resumen de la ejecución](../figures/calidad/01_resumen_ejecucion.jpg).
- [Reporte original de integridad de Deepchecks](../figures/calidad/02_integridad_deepchecks.jpg).
- [Alerta de drift de P3](../figures/calidad/03_alerta_drift_p3.jpg).

Las capturas provienen del navegador: el resumen presenta los resultados
guardados de las herramientas y las otras dos muestran los HTML originales de
Deepchecks. Los reportes completos permanecen en `data/quality/`, fuera de Git.

La actividad permitió distinguir entre una función que produce un resultado y
una función cuyo comportamiento se conoce ante errores. También mostró que una
ejecución sin fallos no elimina las limitaciones de los datos ni los riesgos de
las dependencias. Conservar esos hallazgos facilita revisar el proyecto y decidir
qué debe corregirse antes de avanzar.
