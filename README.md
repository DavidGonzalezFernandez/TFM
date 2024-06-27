# Evaluación de la Librería TINTOlib

## Descripción general
Este repositorio contiene el código y los recursos para replicar los experimentos realizados en mi Trabajo de Fin de Máster.

## Ficheros y carpetas
Ficheros:
- `utils.py`: contiene funciones para cargar los datasets; hacer los splits de entrenamiento, valicación y test; obtener información sobre los datasets; crear las ramas de los modelos
- `dataset_information.ipynb`: notebook para probar a cargar los dataset y obtener información sobre la cantidad y tipo de datos.
- `evaluate_models.ipynb`: notebook para evaluar los modelos entrenados
- `train_classic.ipynb`: notebook para entrenar descriptores clásicos con los datos de un dataset.
- `train_cnn.ipynb`: notebook para entrenar CNNs a partir de imágenes sintéticas.
- `train_cnn_mlp.ipynb`: notebook para entrenar modelos CNN+MLP a partir de las imágenes sintéticas y los datos de un dataset. El entrenamiento se realiza con las imágenes sintéticas (en la rama CNN) y los datos del dataset (en la rama FFNN).
- `train_cnn_ML.ipynb`: notebook para entrenar modelos CNN+ML a partir de las imágenes sintéticas y los datos de un dataset. El entrenamiento se realiza con las imágenes sintéticas (en la rama CNN) y las predicciones que de un modelo RandomForest devuelve (concatenados a la predicción de la rama CNN).

Carpetas:
- `datasets`: contiene los datasets en bruto
