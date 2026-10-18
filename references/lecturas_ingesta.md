# Referencias técnicas de ingesta y P1

## Material consultado
- MLflow Projects: https://mlflow.org/docs/latest/ml/projects/ . El manifiesto del proyecto declara parámetros y comandos; las dependencias están fijadas en el lock. El cambio de pregunta se pasa como parámetro, sin editar celdas.
- Ejemplo oficial: https://github.com/mlflow/mlflow-example . Se revisaron MLproject y train.py: entry point con alpha y l1_ratio, entrenamiento ElasticNet y registro de parámetros, métricas y modelo. Se utiliza como patrón de empaquetado. Su partición aleatoria de vino no se trasladó a la serie temporal MiBici.
- Real Python: https://realpython.com/python3-object-oriented-programming/ . La revisión se centró en clases, objetos, atributos y métodos. En el proyecto Settings agrupa configuración; RegressionExperiment agrupa la lógica de P1 y ExperimentTracker centraliza el registro.
- Senthil: https://medium.com/analytics-vidhya/mlops-refactoring-python-code-bdeeddd1112e . La revisión se centró en nombres claros, valores configurables, funciones reutilizables y documentación. Se contrastó con config.py, features.py y dataset.py; no se copió un preprocesamiento que ajustara transformaciones sobre toda la muestra.
- Presentación local 2.2_MLFlow_tracking_registry-1.pdf y transcripción previa del video proporcionado: se contrastaron tracking, parámetros, métricas, artefactos, experimentos, runs y registro de modelos. Registrar un modelo en un run no acredita por sí solo publicación en Model Registry ni despliegue. El video no se volvió a reproducir íntegramente en esta sesión.

## Acceso o escucha pendiente
- Wilson, capítulo 6: https://learning.oreilly.com/library/view/machine-learning-engineering/9781617298714/OEBPS/Text/ch06_Wilson.htm . Se pudo comprobar el título e índice, no el capítulo completo. No se atribuyen conclusiones específicas al texto no accesible.
- Lauchande, capítulo 5: https://learning.oreilly.com/library/view/machine-learning-engineering/9781800560796/B16783_05_Final_SB_epub.xhtml#_idParaDest-81 . Acceso completo no verificado.
- Thompson, sección 7.4: https://learning.oreilly.com/library/view/managing-machine-learning/9781633439023/OEBPS/Text/07.html#heading_id_13 . Acceso completo no verificado.
- Data Skeptic, ML Ops Best Practices, 12 de agosto de 2022: https://dataskeptic.com/blog/episodes/2022/ml-ops-best-practices . Se consultó la ficha con Piotr Niedźwiedź, relacionada con seguimiento de experimentos. La escucha completa queda pendiente; la ficha no sustituye el audio.
