# Reproducción independiente de P2 y P3

Usar la misma versión estable de la segunda entrega indicada en el PR de Martha. No depender de la rama ni del PR de Milton. Consultar la instalación macOS o Windows del README. No se necesita Docker.

## Ejecución

1. Clonar en una carpeta nueva. Si el PR de Martha sigue abierto, ejecutar `git switch --track origin/feat/mlflow-registry`; si ya se integró, usar main actualizado. Preparar Python 3.11.9 y el lock; ejecutar `python -m pip check` y `python -m pytest -q`.
2. Iniciar el servidor propio con SQLite, configurar ambos URI y ejecutar:

```text
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo14
mlflow run . -e verify --env-manager local --experiment-name mibici-equipo14
mlflow run . -e reproduce --env-manager local --experiment-name mibici-equipo14 -P question=p2
mlflow run . -e reproduce --env-manager local --experiment-name mibici-equipo14 -P question=p3
mlflow run . -e promote --env-manager local --experiment-name mibici-equipo14 -P question=p2 -P local=1
mlflow run . -e promote --env-manager local --experiment-name mibici-equipo14 -P question=p3 -P local=1
mlflow run . -e verify_registry --env-manager local --experiment-name mibici-equipo14 -P question=p2 -P local=1
mlflow run . -e verify_registry --env-manager local --experiment-name mibici-equipo14 -P question=p3 -P local=1
```

La opción `local=1` registra la reproducción propia, no los URI del servidor oficial. Los resultados locales se conservan en `references/validaciones/local_p2.json`, `local_p3.json` y `data/interim/`. No subir bases, modelos ni datos.

3. En la interfaz, verificar el umbral incluido en P2, el scaler de P3, sus ejemplos y el alias champion local. La comparación de predicciones debe pasar; no basta que el modelo aparezca en la lista.
4. Añadir una prueba propia en `tests/test_reproduccion_p2_p3.py`, por ejemplo que el wrapper cargado conserve un umbral distinto de 0.5 o que la selección no acepte un grupo demasiado pequeño. Revisar las pruebas existentes y cubrir un caso adicional.
5. Crear `references/validaciones/p2_p3.md` con commit, comandos, versión de datos, métricas reales, versiones registradas y resultado de carga por alias. Añadir capturas si aportan evidencia legible.
6. Desde la versión estable ejecutar `git switch -c test/p2-p3-reproduccion`. Hacer commit de sus archivos y publicar con `git push -u origin test/p2-p3-reproduccion`. Abrir PR con base main. Si la infraestructura de Martha aún no está integrada, indicar esa dependencia; al integrarse desaparecerá del diff compartido. Milton revisa ese PR.
7. Revisar el PR de Martha desde la cuenta propia. Comprobar cambios y evidencia; enviar Approve si son correctos o Request changes si falta algo. La aprobación no se sustituye por un comentario general.

No es necesario volver a ejecutar las nueve combinaciones, editar P1 ni coordinar cambios de archivos con Milton. La contribución mínima incluye ejecución real, prueba técnica, PR propio y revisión formal.
