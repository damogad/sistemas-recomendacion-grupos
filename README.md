# Sistema de recomendación para grupos

El desarrollo de esta práctica se ha realizado utilizando un entorno virtual de Python para poder instalar de forma aislada las dependencias necesarias.

```
python -m venv venv
```

Windows (PowerShell):

```
.\venv\Scripts\Activate.ps1
```

Linux:

```
source ./venv/bin/activate
```

Además, en Windows será necesario tener instalado la versión 14.0 o mayor de Microsoft Visual C++ (Microsoft C++ Build Tools) más información [aquí](https://stackoverflow.com/a/50210015). En Linux, simplemente es necesario instalar `gcc`.

```
python -m pip install --upgrade pip setuptools wheel
pip install -r ./requirements.txt
```