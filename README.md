# Sistema de recomendación para grupos

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