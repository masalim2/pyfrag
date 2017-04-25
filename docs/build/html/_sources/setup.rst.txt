Setup Instructions
==================

Prerequisites
-------------
PyFragment requires Python 2.7 or later, `numpy <http://www.numpy.org>`_, `scipy
<http://www.scipy.org>`_, and `MPI4Py <http://www.mpi4py.scipy.org>`_. These
modules should be readily importable, i.e.::
    
    import numpy
    import scipy
    import mpi4py
    import argparse
should execute without any errors in your Python interpreter.

Your PYTHONPATH environment variable should include the directory containing
'pyfrag'.  This will allow you to import the PyFrag modules from anywhere. 

PyFragment requires at least quantum chemistry backend to perform the fragment
calculations. Currently supported packages are `Psi4
<http://www.psicode.org>`_, `NWChem
<http://www.nwchem-sw.org>`_, and `Gaussian09 <http://www.gaussian.com>`_. As
such, at least one of the executables **psi4**, **nwchem.x**, or **g09** must be
in the PATH.

Gellmann setup
--------------
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

Blue Waters setup
-----------------
Blue Waters comes equipped with high-performance builds of the necessary
python `modules <https://bluewaters.ncsa.illinois.edu/python>`_.  Your job submission scripts can set up the environment with the two lines ::
    module load bwpy
    module load bwpy-mpi

Installing Psi4
***************
::
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
    

Running as executable
---------------------
To run as an executable on 16 cores, invoke :: 
    mpirun -n 16 python /directory/to/pyfrag <input-file>
from the command line. This causes Python to run the __main__.py module
located in pyfrag directory.
