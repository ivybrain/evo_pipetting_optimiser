# Evo_Pipetting_Optimiser

## Overview and usage

### Installation

This package can be installed by:

1. `cd evo_pipetting_optimiser`
2. `pip install .`

### Deinstallation

You can just uninstall the package with:
 ```
 pip uninstall evo_pipetting_optimiser
 ```

### Usage

The file  [`example_using_pkg.py`](example_using_pkg.py) shows how we can import and use functions from the package.

## What makes a package?

`__init__.py` is the magic sauce to change a directory of Python files into a package. In this repo, the package is the directory [`evo_pipetting_optimiser`](evo_pipetting_optimiser), take a second to look in the folder and the files' contents.  We have 3 files containing functions and then the good ole [`__init__.py`](evo_pipetting_optimiser/__init__.py).

In the [`__init__.py`](evo_pipetting_optimiser/__init__.py) we can specify how users can reach our functionality.  For example, core functionality might be listed out here so users can access easier.  In this package the area functions might be considered more core because they're listed in [`__init__.py`](my_pkg/__init__.py).  Note in [`example_using_pkg.py`](example_using_pkg.py) how this effects how we interact with the area vs perimeter functions.

## [`setup.py`](setup.py)

This file is what leads our package to be easily installed with `pip`.  It contains some info on the package as well as dependency information. Add the required packages that your package needs inside `setup.py` at `install_requires`.

## Unit tests

Python makes it very easy to write and execute unit tests. Just put the tests you want to write into the folder tests and make sure that the file has the form test_*.py. Look into the tests folder for an example. To execute the tests locally, just run 
```
pytest
```
on the commandline from the root directory. If you want to measure code coverage run

```
pytest --cov=evo_pipetting_optimiser tests/
```

## CI Pipeline

Since we use Azure Devops for organization of the agile process we can leverage it's CI pipeline. In Azure Devops UI you can just connect the repository to a pipeline with a few clicks, see [here](https://learn.microsoft.com/en-us/azure/devops/pipelines/create-first-pipeline?view=azure-devops&tabs=python%2Ctfs-2018-2%2Cbrowser) for additional information. Here, the `azure-pipelines.yml` excutes the unit tests with `pytest` with code coverage getting displayed afterwards.

## .vscode

In the settings.json you can specify some rules for working in visual studio code. Here, we have the following settings:
1. File gets autoformatted during save adhering to the official Python style-guide `PEP8` using `black`. See [here](https://pep8.org/) and [here](https://pypi.org/project/black/) for further information.
2. pylint is used to check for errors, give hints for coding standard, look for code smells, and make suggestions about how the code could be refactored.

## Package versioning

To increment the version of the package, manually adjust the version number in the setup.py file.