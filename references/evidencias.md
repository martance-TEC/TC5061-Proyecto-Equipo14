# Evidencias del experimento en MLflow

Las capturas del punto 7 deben proceder de la interfaz real de MLflow. Los gráficos exportados son resultados complementarios; no sustituyen la vista de su registro dentro de un run.

1. **Lista de runs:** abrir el experimento `mibici-equipo14`, expandir el padre y mostrar los modelos de P1, P2 y P3. Los nombres deben permitir distinguir pregunta y modelo. Mostrar el estado de los runs.
2. **Detalle:** abrir un hijo, por ejemplo P2 Random Forest. Mostrar parámetros, métricas y etiquetas técnicas: pregunta, commit y versión de datos. Si no caben, usar dos capturas consecutivas del mismo run. Conservar el identificador del run para relacionarlas.
3. **Artefacto:** desde Artifacts de ese run, abrir `evaluation/confusion_matrices.json`; también puede usarse el mapa `interpretation/clusters_k3.png` de P3. Mostrar árbol, nombre y contenido del artefacto dentro de MLflow.
4. **Comparación:** seleccionar modelos de la misma pregunta, por ejemplo P2 Dummy prior y Random Forest, y abrir Compare. Mostrar encabezados que identifiquen los dos modelos y sus métricas de prueba. Las métricas de tareas distintas no se comparan directamente.

P1 tiene tres modelos en su definición actual; la referencia semanal y el promedio histórico son competidores distintos. Los runs de versiones anteriores se conservan con sus commits, pero las capturas no deben presentarse como si pertenecieran a la misma definición de P1.

Los parámetros, métricas, commits y versiones de datos se mantienen sin alteraciones. Se pueden ocultar columnas de rutas personales para mejorar la presentación, sin cambiar resultados ni atribuir una ejecución a otra cuenta.

## Capturas de la ejecución integrada

| Evidencia | Archivo |
|---|---|
| Lista de los once runs terminados | [evidencia_01_lista_runs.jpg](../reports/figures/evidencia_01_lista_runs.jpg) |
| Parámetros de capacidad, métricas y etiquetas de P2 | [evidencia_02_detalle_run.jpg](../reports/figures/evidencia_02_detalle_run.jpg) |
| Matrices de confusión abiertas en Artifacts | [evidencia_03_artefacto.jpg](../reports/figures/evidencia_03_artefacto.jpg) |
| Comparación de Random Forest y Dummy prior | [evidencia_04_comparacion.jpg](../reports/figures/evidencia_04_comparacion.jpg) |

Código ejecutado: `658e428a83101363e66dd4727675b51daac8b662`, con `git_dirty=false`. Run padre: `10a63b549af148d3862f707678cf2105`. Detalle y artefacto: run P2 `a212dbfe379b420b83fbb03c300832fd`. La comparación incluye el baseline `f49c0b91410a48009cd6a4c7d43849a3`. Las capturas muestran la interfaz real y no modifican métricas ni etiquetas.
