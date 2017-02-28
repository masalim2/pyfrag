.. pyfragment documentation master file, created by
   sphinx-quickstart on Tue Feb 28 11:17:08 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyFragment Documentation
========================
PyFragment is a collection of Python modules that facilitate the setup and
parallel execution of *embedded-fragment* calculations on molecular clusters,
liquids, and solids. It interfaces with the quantum chemistry software
packages `Psi4
<http://www.psicode.org>`_, `NWChem
<http://www.nwchem-sw.org>`_, and `Gaussian09 <http://www.gaussian.com>`_. 

The **binary interaction method (BIM)** modules support 
    * Molecular clusters or systems with 1, 2, or 3-dimensional periodic
      boundary conditions
    * Total energy (or unit cell energy) evaluation (HF, MP2, and beyond)
    * Nuclear gradients and hessians of the total energy

The experimental **valence bond charge-transfer (VBCT)** module is a research
branch of the software. The intent is to extend molecular fragment
calculations to systems with significant charge-resonance, where integer
electron counts cannot be assigned to individual fragments.

Where applicable, modules can be run as standalone executables with a
input file supplied on the command line. They may also be imported into
user-written Python programs to create new functionality or automate some
tedious task. Several **driver** scripts are already included, which use the
**BIM** and **VBCT** modules to:
    * Perform a PES scan along any user-defined coordinate
    * Find a local minimum in the PES using L-BFGS optimization
    * Integrate a molecular dynamics trajectory

Several tools to process the output of the above calculations are
included:
    * Phonon dispersion calculation from solid Hessian data
    * *to be continued...*

Contents
========
.. toctree::
   :maxdepth: 2

   setup
   background
   globals
   samples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
