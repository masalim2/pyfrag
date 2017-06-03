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

should execute without any errors in your Python interpreter. To obtain the
source, clone the git repository::

    git clone https://github.com/masalim2/pyfrag.git

The :data:`PYTHONPATH` environment variable must include the path to the top
level pyfrag directory. This is necessary for the subpackages inside pyfrag to
find each other. 

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
python `modules <https://bluewaters.ncsa.illinois.edu/python>`_.  Your job submission scripts should set up the environment with 
the following four lines ::

    module load bwpy
    module load bwpy-mpi
    export PMI_NO_FORK=1
    export PMI_NO_PREINITIALIZE=1

Note that Psi4 is unable to run correctly on the compute nodes unless the latter two
environment variables are set.

Installing Psi4 on Blue Waters
******************************
You need to be using the GNU Programming environment.  If :data:`module list`
shows :data:`PrgEnv-cray` instead, be sure to first invoke :data:`module swap
PrgEnv-cray PrgEnv-gnu`.
Next, checkout the Psi4 repository and prepare the environment for setup. ::

    git clone https://github.com/psi4/psi4.git
    module load bwpy
    export CRAYPE_LINK_TYPE=dynamic
    export CRAY_ADD_RPATH=yes

Now, query the currently (default) loaded Cray LibSci module to find the
directory containg the LibSci libaries::

    module display cray-libsci

Search for the line containing :data:`prepend-path  CRAY_LD_LIBRARY_PATH`: it should
point to a directory like :data:`/opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib`. 
In this directory, identify all the library files that do not contain the string
"mp" or "mpi"; there should be four files similar to :data:`libsci_gnu_49.a`,
:data:`libsci_gnu_49.so`, :data:`libsci_gnu_49.so.5`, and
:data:`libsci_gnu_49.so.5.0`. 

Create a folder named :data:`libsci` in your home directory. In this folder, create two symlinks
for each of the identified library files. The two links corresponding to :data:`libsci.*`
should be named :data:`liblapack.*` and :data:`libblas.*`. For instance, ::
    
    cd ~/libsci
    ln -s /opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib/libsci_gnu_49.a      liblapack.a
    ln -s /opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib/libsci_gnu_49.so     liblapack.so
    ln -s /opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib/libsci_gnu_49.so.5   liblapack.so.5
    ln -s /opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib/libsci_gnu_49.so.5.0 liblapack.so.5.0

    ln -s /opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib/libsci_gnu_49.a      libblas.a
    ln -s /opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib/libsci_gnu_49.so     libblas.so
    ln -s /opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib/libsci_gnu_49.so.5   libblas.so.5
    ln -s /opt/cray/libsci/16.11.1/GNU/4.9/x86_64/lib/libsci_gnu_49.so.5.0 libblas.so.5.0

    export MATH_ROOT=$(pwd)

Finally, exporting the MATH_ROOT environment variable will allow CMake to recognize 
and use these preferred math libraries. CMake should be version 3.3 or higher;
if this is not the case, see the section below on getting a newer CMake binary
through conda. If :data:`cmake --version` returns 3.3 or higher, you are fine.
With the above steps done, building should be straightforward: ::
    cd psi4
    cmake -H. -Bobjdir
    cd objdir
    nohup make >& make.log &

The compilation is rather lengthy; using nohup will allow you to launch the
build and then log off without interrupting the process.

After installation, the :data:`psi4/objdir/stage/.../bin` directory (which
contains the executable psi4) should be added to your :data:`PATH`, while the
:data:`psi4/objdir/stage/.../lib` directory (which contains the importable
module) should be added to your :data:`PYTHONPATH`.  Alternatively, you can use
make install to copy the relevant code into a user-specified location.

Getting Miniconda and CMake on Blue Waters
*********************************************
You *might* need a newer version of cmake than is provided on Blue Waters. 
If cmake --version returns 3.3 or higher (after loading module bwpy), then you
can safely ignore this section.  

Otherwise, a
convenient way to quickly get the binary is with `Miniconda
<https://conda.io/miniconda.html`_. Install Miniconda after downloading the
installer directly to Blue Waters without allowing it to override your :data:`PATH` or 
:data:`PYTHONPATH` environment. (Letting conda alter these environment variables can clobber
the environment which is supposed to be managed entirely by the Blue Waters
module system. This may cause other things to break.) ::
    
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    bash Miniconda2-latest-Linux-x86_64.sh
    # installation process...don't let it mess with your .bashrc
    # Once you have conda, get cmake 
    cd ~/path/to/miniconda2/bin
    ./conda install cmake 

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
