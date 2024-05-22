# TFM

## Descripción general
Este repositorio contiene el código, los recursos y los resultados de los experimentos realizados en mi Trabajo de Fin de Máster.

## Ficheros y carpetas
Ficheros:
- `dataset_information.ipynb`: notebook para probar a cargar los dataset y obtener información sobre la cantidad y tipo de datos.
- `datasets_to_images.py`: script para transformar los datasets en imágenes utilizando los métodos disponibles en TINTOlib. 
- `train_classic.ipynb`: notebook para entrenar descriptores clásicos con el conjunto de entrenamiento para los datasets disponibles.
- `train_classic_with_split1.ipynb`: notebook para entrenar descriptores clásicos con parte del conjunto de entrenamiento para los datasets disponibles.
- `train_cnn.ipynb`: notebook para entrenar CNNs a partir de las imágenes.
- `train_cnn_fnn.ipynb`: notebook para entrenar modelos CNN+FFNN a partir de las imágenes y los datos del dataset. El entrenamiento se realiza con las imágenes generadas con `datasets_to_images.py` (en la rama CNN) y los datos del dataset (en la rama FFNN).
- `train_cnn_classic.ipynb`: notebook para entrenar modelos CNN+descriptores a partir de las imágenes y los datos del dataset. El entrenamiento se realiza con las imágenes generadas con `datasets_to_images.py` (en la rama CNN) y las predicciones que los modelos de `train_classic_with_split1.ipynb` devuelven (concatenados a la predicción de la rama CNN).
- `utils.py`: contiene funciones para cargar los datasets; hacer los splits de entrenamiento, valicación y test; y obtener información sobre el tipo de dataset.

Carpetas:
- `datasets`: contiene los datasets en bruto
- `images`: contiene los datasets de `utils` convertidos a imágenes.
- `results`: contiene los resultados obtenidos en los scripts y notebooks.
