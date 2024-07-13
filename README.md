# Sistema de recomendación para grupos

**Autor: David Mora Garrido**

**Sistemas de recomendación, Máster Universitario en Investigación en Inteligencia Artificial (UIMP)**

## Sobre la implementación

Hemos llevado a cabo las dos estrategias descritas en el libro **_Group Recommender Systems: An Introduction_** para realizar recomendaciones para grupos: _Aggregated Models_ y _Aggregated Predictions_. El _dataset_ utilizado es la versión 'pequeña' de [MovieLens (100k puntuaciones)](https://grouplens.org/datasets/movielens/) pues, dado que hemos utilizado la librería `surprise`, y ésta nos muestra en el apartado [_Benchmarks_](https://surpriselib.com/) de su página web métricas obtenidas con diferentes algoritmos sobre el mismo _dataset_ (aunque la recomendación es para usuarios, no para grupos), podíamos tener así una referencia.

## Instalación entorno virtual

El desarrollo de esta práctica se ha realizado utilizando un entorno virtual de Python para poder instalar de forma aislada las dependencias necesarias. En nuestro caso se ha utilizado la versión 3.12 de Python en Windows y verificado con Linux basado en Debian con versión 3.10.

En Windows:

```
python -m venv venv
```

En Linux, en primer lugar será necesario instalar `python3-dev` (en el siguiente ejemplo se utiliza `apt-get`, utilice el _package manager_ correspondiente a la distro en cuestión):

```
sudo apt-get update && sudo apt-get install python3-dev
python3 -m venv venv
```

Activación en Windows (PowerShell):

```
.\venv\Scripts\Activate.ps1
```

Activación en Linux:

```
source ./venv/bin/activate
```

Además, en Windows será necesario tener instalado la versión 14.0 o mayor de Microsoft Visual C++ (Microsoft C++ Build Tools) más información [aquí](https://stackoverflow.com/a/50210015). En Linux, es necesario tener instalado `gcc`.

```
python -m pip install --upgrade pip setuptools wheel
pip install -r ./requirements.txt
```

## Script de entrenamiento

El archivo `train.py` contiene todo el código necesario para entrenar un modelo de predicción que nos permita realizar recomendaciones para grupos, así como la evaluación del mismo. Para ver qué parametros hay disponibles, se puede ejecutar el siguiente comando de ayuda:

```
python ./train.py -h
```

Salida:

```
usage: train.py [-h] [-m {SVD,KNNBaseline}] -a {agg-models,agg-predictions} [-t TRAIN_SIZE] [-g GROUP_SIZE]
                [-k NUM_RECOMMENDED_ITEMS] [-r RELEVANCE_THRESHOLD] [-s SEED] [-n]

options:
  -h, --help            show this help message and exit
  -m {SVD,KNNBaseline}, --model {SVD,KNNBaseline}
                        Prediction algorithm defined in the surprise library to be trained.
  -a {agg-models,agg-predictions}, --agg-method {agg-models,agg-predictions}
                        Which aggregation strategy to use (Aggregated Models or Aggregated Predictions).
  -t TRAIN_SIZE, --train-size TRAIN_SIZE
                        Fraction of the dataset that will be used for training.
  -g GROUP_SIZE, --group-size GROUP_SIZE
                        Size of the groups to be created.
  -k NUM_RECOMMENDED_ITEMS, --num-recommended-items NUM_RECOMMENDED_ITEMS
                        Number of items that will be recommended to each group.
  -r RELEVANCE_THRESHOLD, --relevance-threshold RELEVANCE_THRESHOLD
                        Value in the range [0.5, 5.0], so that those items with a true rating equal to or greater
                        than this one will be considered relevant.
  -s SEED, --seed SEED  Seed for the pseudo-random number generator. It allow us to create the same data partitions
                        and be able to reproduce the same training for the given arguments.
  -n, --skip-training   If specified, training is skipped and we try to load a saved model given the value of other
                        arguments. This way, we can just evaluate the loaded model on the testset of the same holdout
                        partition used for it's training, potentially with other argument values (for example, other
                        relevance threshold, or other group size in the case of having specified the agg-predictions 
                        strategy).
```

No es necesario especificar todos los argumentos en la línea de comandos, pues algunos tienen un valor por defecto. Por ejemplo, en nuestro caso, todas las ejecuciones las hemos realizado con el mismo valor (por defecto) de `-t` (`train_size`) y `-s` (`seed`) para poder realizar la misma partición de los datos en un holdout train-test (y también para obtener la misma partición dentro del conjunto de entrenamiento al realizar una búsqueda en grid de hiperparámetros).

## Ejecución notebook

Hemos creado un notebook para poder guardar la salida de todas las ejecuciones/entrenamientos realizados, así como para mostrar ejemplos de cómo utilizar los modelos entrenados y evaluados para recomendar items a un grupo en concreto.

Para ejecutarlo, primero será necesario instalar `notebook` en el entorno virtual:

```
pip install notebook
```

Una vez hecho esto, podemos lanzar el servidor y abrir `executions_and_inference.ipynb`:

```
jupyter notebook executions_and_inference.ipynb
```

Además, también se ha exportado la ejecución de dicho _notebook_ a una página web (archivo `executions_and_inference.html`)