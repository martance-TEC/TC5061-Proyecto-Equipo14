# Diccionario y tratamiento de fuentes

Fuentes: https://mibici.net/es/datos-abiertos/ y URLs exactas en sources.json. Doce archivos mensuales de 2025 y nomenclatura septiembre de 2026. La sección pública permite descargar CSV; no se verificó una licencia abierta específica, por lo que no se declara CC0 ni se redistribuyen los datos.

Viaje_Id pasa a trip_id string; Inicio_del_viaje pasa a started_at; Fin_del_viaje pasa a ended_at; Origen_Id pasa a origin_id; Destino_Id pasa a destination_id. Usuario_Id, Genero y Año_de_nacimiento se excluyen. Tiempos locales de Guadalajara, America/Mexico_City, sin inferir otra zona.

Catálogo: id pasa a station_id string; name es nombre de estación; latitude y longitude son coordenadas públicas; dpcapacity pasa a capacity. Solo capacidad positiva observada es admisible para P2. Su vigencia 2026-09 no demuestra infraestructura histórica de 2025.

Entrenamiento [2025-01-01,2025-09-01); validación [2025-09-01,2025-11-01); prueba [2025-11-01,2026-01-01). Cada etiqueta debe terminar dentro de su ventana: una hora para P1 y tres para P2. Las variables de cada corte solo usan eventos de horas cerradas. Los ceros se crean dentro del calendario cubierto por los doce archivos; no acreditan operación de una estación antes de su apertura.

Normalización con decodificación estricta UTF-8-sig o cp1252 según archivo; fechas explícitas, tipos estables, validación de fin posterior a inicio. Identificadores duplicados idénticos se eliminan globalmente; si un ID tiene eventos contradictorios se detiene la ingesta. data_manifest.json registra tamaños reales, conteos y SHA256 de cada fuente. data_version es SHA256 del JSON canónico de las fuentes, sin rutas absolutas ni fecha de ejecución. Las copias originales se conservan localmente y no se publican; los hashes detectan cambios pero no recuperan archivos reemplazados.
