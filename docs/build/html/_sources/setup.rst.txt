Setup Instructions
==================

Prerequisites
-------------
PyFragment requires `numpy <http://www.numpy.org>`_, `scipy
<http://www.scipy.org>`_, and `MPI4Py <http://www.mpi4py.scipy.org>`_. These
modules should be readily importable, i.e.::
    
    import numpy
    import scipy
    import mpi4py
should execute without any errors in your Python interpreter.

Your PYTHONPATH environment variable should include the directory containing
'pyfrag'.  This will allow you to import the PyFrag modules from anywhere. 

PyFragment requires at least quantum chemistry backend to perform the fragment
calculations. Currently supported packages are `Psi4
<http://www.psicode.org>`_, `NWChem
<http://www.nwchem-sw.org>`_, and `Gaussian09 <http://www.gaussian.com>`_. As
such, at least one of the executables **psi4**, **nwchem**, or **g09** must be
in the PATH.

Gellmann or personal computer setup
-----------------------------------
One of the most convenient ways to manage your environment is with the
`anaconda <https://www.continuum.io/downloads>`_ platform.  It serves as a
package manager and virtual environment manager, with which you can configure
an environment with libaries stored in your home directory.

Once conda is installed, download the modern versions of prerequisite
packages ::
    conda install numpy
    conda install scipy
    conda install mpi4py
and make sure that your PATH is correctly set to run the conda-managed python
interpreter.

Blue Waters setup
-----------------
Blue Waters comes equipped with high-performance builds of the necessary
python `modules <https://bluewaters.ncsa.illinois.edu/python>`_.  Your job submission scripts can set up the environment with the two lines ::
    module load bwpy
    module load bwpy-mpi

Running as executable
---------------------
To run as an executable on 16 cores, invoke :: 
    mpirun -n 16 python /directory/to/pyfrag <input-file>
from the command line. This causes Python to run the __main__.py module
located in pyfrag directory.
