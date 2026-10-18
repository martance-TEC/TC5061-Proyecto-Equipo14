# Material de apoyo para Persona 3

Consulta técnica asistida, no lectura personal acreditada de Víctor.

- MLflow. (s. f.). MLflow Projects. https://mlflow.org/docs/latest/ml/projects/
  Un MLproject describe parámetros y comandos. Se comprobó el comando local
  parametrizado y la relación entre el run padre de Projects y sus hijos.
- scikit-learn developers. (s. f.). fbeta_score.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
  Beta mayor que uno pondera más el recall; F2 orienta la comparación de P2.
  Un buen resultado contra el proxy no demuestra detección de faltantes reales.
- scikit-learn developers. (s. f.). KMeans.
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  Se fijan semilla y n_init explícitamente, y se comparan varios k.
  El entorno ejecutado conserva scikit-learn 1.5.2 del lock, no la versión latest.
- MLflow. (s. f.). MLflow example. https://github.com/mlflow/mlflow-example
  Referencia ya revisada para el entry point; no se trasladó su partición
  aleatoria a una evaluación temporal de viajes.
- Real Python. (s. f.). Object oriented programming in Python.
  https://realpython.com/python3-object-oriented-programming/
  Material revisado previamente: clases por responsabilidad y funciones puras
  en features.py. No se agrega herencia que no resuelva una necesidad concreta.
- Senthil, E. (2023). MLOps Refactoring Python Code.
  https://medium.com/analytics-vidhya/mlops-refactoring-python-code-bdeeddd1112e
  Material revisado previamente para modularización y nombres claros; en esta
  comprobación se ampliaron docstrings y pruebas de las funciones de variables.

La presentación local y la transcripción previa del video se usaron como
contexto de Tracking, parámetros, métricas y artefactos. No se afirma que el
video se haya vuelto a reproducir íntegro durante esta comprobación.

Pendiente de completar personalmente por Víctor: Thompson sección 7.4,
https://learning.oreilly.com/library/view/managing-machine-learning/9781633439023/OEBPS/Text/07.html#heading_id_13
y los capítulos de Lauchande y Wilson cuyo acceso completo no fue verificado.
También falta la escucha completa del episodio Data Skeptic ML Ops Best
Practices: https://dataskeptic.com/blog/episodes/2022/ml-ops-best-practices
La ficha y las notas públicas no sustituyen el audio.
