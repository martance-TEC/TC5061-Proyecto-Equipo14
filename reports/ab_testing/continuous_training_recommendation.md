# Decisión A/B y estrategia de Continuous Training para P1

## Resumen ejecutivo

El monitoreo de P1 no detectó drift global entre el periodo de referencia enero-agosto de 2025 y el periodo septiembre-octubre de 2025. Solo 1 de las 15 variables monitoreadas, `month`, superó el umbral de drift de 0.10, equivalente al 6.67% de las variables y por debajo del criterio global de 50%.

La ausencia de drift global no implica que el modelo deba mantenerse indefinidamente. Por ello se evaluó un challenger `HistoricalMean` actualizado con información de enero-octubre frente al champion original ajustado con enero-agosto.

Ambos modelos se evaluaron sobre exactamente las mismas 540,216 observaciones de noviembre-diciembre de 2025, periodo excluido del entrenamiento de ambos.

## Resultado A/B

| Métrica | Champion v1 | Challenger v2 | Cambio |
|---|---:|---:|---:|
| MAE | 0.877629 | 0.860265 | -1.98% |
| RMSE | 1.594179 | 1.549914 | -2.78% |
| WAPE | 0.662798 | 0.649684 | -1.98% |

El challenger mejora simultáneamente las tres métricas principales.

En la comparación pareada por observación:

- challenger mejor: 244,964 observaciones (45.35%);
- champion mejor: 193,832 observaciones (35.88%);
- empates: 101,420 observaciones (18.77%);
- delta MAE, definido como error challenger menos error champion: -0.017365.

El bootstrap pareado con 1,000 iteraciones y semilla 42 produjo un intervalo de confianza del 95% para delta MAE de:

[-0.017799, -0.016933]

El intervalo completo se encuentra por debajo de cero y favorece al challenger.

## Validación temporal

Para considerar la dependencia entre observaciones horarias se realizó adicionalmente un bootstrap por días completos.

Resultados:

- días evaluados: 61;
- challenger mejor: 57 días (93.44%);
- champion mejor: 4 días (6.56%);
- empates: 0;
- delta MAE promedio diario: -0.017365;
- IC 95%: [-0.020158, -0.014708].

El intervalo temporal también se encuentra completamente por debajo de cero.

Por lo tanto, la ventaja observada no depende únicamente de tratar cada observación horaria como independiente.

## Decisión

Se recomienda promover P1 v2 de `challenger` a `champion`.

La recomendación no se fundamenta en la existencia de drift global, ya que Evidently no detectó drift global.

La promoción se fundamenta en que:

1. el challenger incorpora información más reciente hasta octubre sin utilizar el holdout final;
2. MAE, RMSE y WAPE mejoran simultáneamente;
3. la mejora de MAE es aproximadamente 1.98%;
4. el análisis pareado favorece al challenger;
5. el IC 95% del bootstrap por observaciones está completamente debajo de cero;
6. el challenger supera al champion en 57 de los 61 días del holdout;
7. el IC 95% del bootstrap temporal permanece completamente debajo de cero;
8. el linaje y la versión de datos fueron verificados antes de realizar la comparación.

La recomendación corresponde a una evaluación histórica offline y no debe interpretarse como un experimento A/B con asignación aleatoria de tráfico en producción.

## Continuous Training

No se recomienda implementar reentrenamiento automático ante cualquier alerta de drift.

Se propone un esquema de Continuous Training condicionado:

1. Ejecutar monitoreo mensualmente sobre calidad, cobertura y drift.
2. Investigar problemas de calidad antes de iniciar reentrenamiento.
3. Activar una revisión cuando exista alguna de las siguientes condiciones:
   - drift persistente durante dos ventanas completas;
   - deterioro de MAE superior a 10% frente al nivel de referencia definido para el mismo servicio;
   - incremento relevante y persistente de estaciones fuera de la cohorte;
   - cambios locales relevantes por estación o franja horaria.
4. Cuando se active una revisión, generar una nueva versión candidata sin modificar el champion.
5. Conservar un periodo temporal posterior fuera del entrenamiento.
6. Comparar champion y challenger sobre exactamente las mismas observaciones.
7. Revisar como mínimo MAE, RMSE, WAPE y comportamiento por segmentos.
8. Utilizar comparación pareada y un método de incertidumbre que respete la dependencia temporal.
9. Promover únicamente después de validación reproducible y revisión humana.

## Criterio de promoción

Una alerta de drift por sí sola no debe provocar una promoción.

La promoción requiere simultáneamente:

- datos y linaje válidos;
- ausencia de leakage;
- mejora útil fuera de muestra;
- ausencia de deterioro material en métricas secundarias;
- estabilidad temporal de la mejora;
- revisión de segmentos relevantes;
- aprobación antes de modificar el alias `champion`.

## Estado final de esta evaluación

Drift global: No.

Variables con alerta: 1 de 15.

Champion evaluado: P1 v1.

Challenger evaluado: P1 v2.

Holdout: noviembre-diciembre de 2025.

Observaciones: 540,216.

Días: 61.

Resultado A/B: challenger.

Recomendación: promover v2 a champion.

Continuous Training: recomendado de forma condicionada, con monitoreo mensual, triggers explícitos, evaluación champion/challenger y promoción controlada.

## Evidencias

- `reports/monitoreo/resumen_drift.json`
- `reports/monitoreo/evidently_metrics.json`
- `reports/monitoreo/evidently_drift.html`
- `reports/monitoreo/interpretacion.md`
- `reports/challenger_p1/evidence_victor.json`
- `reports/ab_testing/p1_ab_victor.json`

La promoción del alias no se ejecuta automáticamente como parte de este análisis. Se conserva `champion=v1` y `challenger=v2` hasta completar la revisión e integración final del equipo.
