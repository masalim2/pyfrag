Setup Instructions
==================

Prerequisites
-------------
PyFragment requires Python version 2.7. It uses the modules `numpy
<http://www.numpy.org>`_, `scipy <http://www.scipy.org>`_, `h5py
<http://www.h5py.org>`_, and `MPI4Py <http://www.mpi4py.scipy.org>`_. These must
be readily importable, i.e.::
    
    import numpy
    import scipy
    import mpi4py
    import h5py
    import argparse

should execute without any errors in your Python interpreter.

The :data:`PYTHONPATH` environment variable must include the path to the top level 
pyfrag directory. This is necessary for the subpackages to find each other. 

PyFragment requires at least one quantum chemistry backend to perform the
fragment calculations. Currently supported packages are `Psi4
<http://www.psicode.org>`_ and `NWChem <http://www.nwchem-sw.org>`_. The
executables  :data:`psi4` and :data:`nwchem.x` must therefore be in the system
:data:`PATH`. The :data:`backend` subpackage can easily be extended for
PyFragment to interface with other quantum chemistry backends.

Gellmann setup tips
--------------------
To replace the default system Python 2.6 with Python 2.7, the **opt-python**
module must be loaded.
A convenient way to set up the requisite Python environment is with virtualenv 
and the pip package manager. Virtualenv allows users to manage a Python
installation and install packages in a loadable environment stored in the home directory. 
This does not require administrator priveleges and the environment can easily be loaded/unloaded
as necessary.  In a subdirectory of your home directory, perhaps
called my-env: run the following commands ::

        module load opt-python # replace default Python 2.6 with 2.7
        wget https://pypi.python.org/packages/source/p/pip/pip-1.1.tar.gz --no-check-certificate # download PIP
        virtualenv --system-site-packages your_environment_name # setup a new environment with PIP 
        source your_environment_name/bin/activate # load the environment

Notice that the terminal now indicates your virtual environment is loaded. You
are now able to install packages to this environment using **pip**. Invoke ::

        pip install mpi4py
        pip install scipy

In future interactive sessions or any job submissions that require Python 2.7,
mpi4py, or scipy, be sure that **opt-python** is loaded and 
the **source** line is invoked.

Blue Waters setup tips
------------------------
Blue Waters comes equipped with high-performance builds of the necessary
python `modules <https://bluewaters.ncsa.illinois.edu/python>`_.  Your job submission scripts can set up the environment with the two lines ::

    module load bwpy
    module load bwpy-mpi

Installing Psi4 on Blue Waters
******************************
Checkout the Psi4 repository and prepare the environment for setup::

    git clone https://github.com/psi4/psi4.git
    module load bwpy
    export CRAYPE_LINK_TYPE=dynamic
    export CRAY_ADD_RPATH=yes
    mkdir ~/libsci

Now, query the currently (default) loaded Cray LibSci module to find the
directory containg the LibSci libaries.  Make symlinks in ~/libsci 
to all the necessary versions, naming the links corresponding to libsci.* 
as liblapack.*. This is necessary for CMake to recognize the math libraries.
Finally, export the MATH_ROOT environment variable::

    export MATH_ROOT=~/libsci

You will need a newer version of cmake than is provided on Blue Waters. A
convenient way to quickly get the binary is with Miniconda (conda install
cmake). This binary will work fine for the build process.  ::

    cd psi4
    cmake -H. -Bobjdir
    cd objdir
    nohup make >& make.log &

The compilation is rather lengthy; using nohup will allow you to launch the
build and then log off without interrupting the process.
    

Running PyFragment
---------------------
The directory containing PyFragment should be included in the :data:`PYTHONPATH`
environment variable.
    
To run as an executable on 16 cores, invoke :: 

    mpirun -n 16 python /directory/to/pyfrag <input-file>

from the command line. This causes Python to run the __main__.py module
located in the pyfrag directory. Alternatively, the program can be invoked 
using ::

    mpirun -n 16 python -m pyfrag <input-file>
