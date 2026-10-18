# ML Canvas MiBici

| Bloque | Contenido definido |
| Propuesta de valor | Apoyar decisiones de flota, rebalanceo y planeación con demanda pronosticada, presión aproximada y perfiles de uso |
| Usuarios y decisiones | Operador para P1; equipo de rebalanceo para P2; planeación urbana y expansión para P3 |
| Fuentes de datos | Viajes mensuales de MiBici de 2025 y catálogo oficial de estaciones 2026_09 con capacidad y coordenadas verificadas |
| Recolección | Descarga automática desde enlaces oficiales, normalización, validación, particiones Parquet y manifiesto con SHA256 |
| Tarea predictiva | P1 regresión por estación y hora; P2 clasificación binaria de riesgo proxy a tres horas; P3 agrupamiento por perfiles históricos |
| Características | Calendario, rezagos y ventanas cerradas para P1 y P2; perfiles de uso y volumen para P3 |
| Predicciones | Conteo esperado; probabilidad y alerta proxy con umbral; identificador de grupo y perfil explicativo |
| Evaluación fuera de línea | MAE para P1; F2 y average precision para P2; silhouette y estabilidad para P3; comparación temporal con baselines |
| Uso de resultados | Tabla horaria de apoyo para P1 y P2; análisis histórico de grupos y mapa para P3; revisión humana de decisiones |
| Monitoreo | Calidad de ingesta, cambios de capacidad, cobertura de estaciones, distribución de variables y métricas en periodos posteriores |
| Criterios de éxito | Reproducción del dataset y de resultados; comparación justa en validación; evidencia por pregunta; utilidad operativa por validar |
No confundir una meta de éxito con un resultado. Las mejoras porcentuales se calcularán después de entrenar.
