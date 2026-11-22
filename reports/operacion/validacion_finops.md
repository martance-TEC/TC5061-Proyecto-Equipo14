# Validación independiente de FinOps

Se contrastaron los totales finales del HTML de Locust con summary.json. Para 8,681 respuestas exitosas en 90 segundos, la asignación proporcional de USD 24 mensuales y 730 horas de planeación es (24/730) × (90/3600) / 8681: USD 0.00000009468 por respuesta y USD 0.00009468 por mil. La equivalencia con inferencias individuales requiere una observación por petición.

Este costo bajo carga no representa el costo mensual con baja utilización: con 10,000 inferencias mensuales, distribuir USD 24 completos implica USD 2.40 por mil. La CPU media de la API fue 20.89% de un núcleo y el pico conjunto de memoria, 502.87 MiB. Conviene evaluar dimensionamiento menor mediante nuevas pruebas. Cincuenta usuarios no demuestran capacidad máxima: faltan saturación, repeticiones y carga prolongada. La ejecución local ARM64 no acredita desempeño equivalente en AWS; tampoco incluye impuestos, operación ni costos adicionales.

Fuente oficial: [Paquetes de Amazon Lightsail](https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-bundles.html).