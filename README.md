MerlinTools package
===========

The merlintools package provides code for manipulating data collected using
a Medipix detector on a scanning transmission electron microscope (STEM) in an
so-called 4D-STEM experiment. Specifically, the package is designed to process data
collected on a specfic microscope configuration at the National Institute for Standards
and Technology (NIST).  The functionality is intended to prepare data and automate downstream processing tasks.


Installation
------------

  Anaconda (Preferred):
  ---------------------
  * Installation should be done into a pre-configured Anaconda   environment that has been configured for FPD and py4DSTEM.  Details regarding this can be found at the following URL:

    https://github.com/py4dstem/py4DSTEM#installation
    
    In brief, first create a new Anaconda environment named merlin. Then, activate
    the new environment and use pip to install py4DSTEM.
    ```bash
    $ conda create -n merlin python==3.8
    $ conda activate merlin
    $ conda install pip
    $ pip install py4dstem

    ```
  * Next, install FPD from the GitLab repo (https://gitlab.com/fpdpy/fpd) via pip:
    ```bash
    $ pip install fpd
    ```
  * Once the Anaconda environment is ready, MerlinTools can be installed from Github:
    ```bash
    pip install git+https://github.com/andrewherzing/merlintools.git
    ```

Removal
-------
The package can be removed with:

```bash
pip uninstall merlintools
```


Usage
-----
In python or ipython:

```python
import merlintools as merlin
data = merlin.io.create_dataset('4DSTEM.hdf5')
```

Documentation is very limited at this point


Documentation
-------------
Release: https://github.com/andrewherzing/merlintools

Further documentation, notebooks and examples will be made available over time.


Related projects
----------------
https://github.com/py4dstem/py4DSTEM

https://fpdpy.gitlab.io/fpd/
