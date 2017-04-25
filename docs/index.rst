.. pyfragment documentation master file, created by
   sphinx-quickstart on Tue Feb 28 11:17:08 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyFragment Documentation
========================
`PyFragment <https://www.github.com/masalim2/pyfrag>`_ is a collection of Python
modules that facilitate the **setup**, **parallel execution**, and **analysis**
of *embedded-fragment* calculations on molecular clusters, liquids, and solids.
It currently interfaces with the quantum chemistry software packages `Psi4
<http://www.psicode.org>`_ and `NWChem <http://www.nwchem-sw.org>`_.

The binary interaction method (**BIM**) module performs a variety of
calculations:

* Total energy (or unit cell energy) evaluation at HF, MP2, or beyond
* Molecular clusters or systems with 1- to 3-dimensional periodic boundary conditions
* Nuclear gradients and stress tensor, even for nonorthogonal lattice vectors
* Nuclear hessian and vibrational analysis tools
* Parallel execution using MPI (through the mpi4py bindings)

The experimental valence bond charge-transfer (**VBCT**) module is part of a new
method development effort. The intent is to extend molecular fragment
calculations to systems with significant charge-resonance, where integer
electron counts cannot be assigned to individual fragments.

Modules can be invoked from the command line with a freeform `input file
<inputfile.html>`_.  They may also be imported to user-written Python programs to
create new functionality or automate tedious tasks. Several such **driver** scripts
are already included, which use the BIM and VBCT modules to: 

* Integrate a **molecular dynamics** trajectory, with optional Nose-Hoover or Berendsen
  thermostats/barostats
* Perform a **PES scan** along a user-defined coordinate
* **Optimize** crystal structure with the BFGS algorithm

Pyfragment also includes a suite of **tools** to process the output of the above
calculations and facilitate data analysis:

* Phonon dispersion/density of states calculations from solid Hessian data
* Molecular dynamics trajectory analysis tool

User guide
----------
.. toctree::
   :maxdepth: 2

   setup
   quickstart
   inputfile

Subpackage documentation
---------------------------
.. toctree::
   :maxdepth: 2

   bim
   drivers
   tools
   vbct
   backend
   test
   globals


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
