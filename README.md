MerlinTools package
===========

The merlintools package provides code for manipulating data collected using
a Medipix detector on a scanning transmission electron microscope (STEM) in an
so-called 4D-STEM experiment. Specifically, the package is designed to process data
collected on a specfic microscope configuration at the National Institute for Standards
and Technology (NIST).  The functionality is intended to prepare data for processing
with the pyXem diffraction analysis package.


Installation
------------

  Anaconda (Preferred):
  ---------------------
  * Installation should be done into a pre-configured Anaconda environment
    that has been configured for pyXem.  Details regarding this can be found
    at the following URL:

    https://pyxem.github.io/pyxem-website/getting_started.html
    
    In brief, first create a new Anaconda environment named pyxem. Then, activate
    the new environment and using conda to install pyXem.
    ```bash
    $ conda create -n pyxem
    $ conda activate pyxem
    $ conda install -c conda-forge pyxem

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
import tomotools as merlin
data = merlin.io.get_merlin_data(mibfiles, hdrfile, dmfile)
```

Documentation is very limited at this point


Documentation
-------------
Release: https://github.com/andrewherzing/merlintools

Further documentation, notebooks and examples will be made available over time.


Related projects
----------------
http://hyperspy.org/

https://pyxem.github.io/pyxem-website/

https://fpdpy.gitlab.io/fpd/
