Quick Start Guide
==================
Follow the `setup instructions <setup.html>`_ and start a Python shell.  Try the
command ::

    from pyfrag import bim

If any errors occur, it is likely that a prerequisite Python package is missing
or the directory containing :data:`pyfrag` is missing from the
:data:`PYTHONPATH` environment variable.  Be sure that :data:`nwchem.x` and
:data:`psi4` are callable from the shell.

A minimal input file
--------------------
The following is an input file to run a BIM calculation of the water
trimer Hartree-Fock/sto-3g energy: :: 

   geometry {
    O    -0.167787    1.645761    0.108747
    H    0.613411    1.10262    0.113724
    H    -0.093821    2.20972    -0.643619
    O    1.517569    -0.667424    -0.080674
    H    1.989645    -1.098799    0.612047
    H    0.668397    -1.091798    -0.139744
    O    -1.350388    -0.964879    -0.092208
    H    -1.908991    -1.211298    0.626207
    H    -1.263787    -0.018107    -0.055536
   }
   correlation = False
   basis = sto-3g
   task = bim_e
 

Save the file to :data:`minimal.inp` and invoke::
   
 mpirun -n 3 python -m pyfrag minimal.inp -v


If all is working correctly, your output should end with the lines ::

    Computing Fragment sums
    Task bim_e done in 00m:07s

      E(monomer)    -224.91589306
        E(dimer)       0.01182801
      E(coulomb)       0.00000000
    -----------------------------
        E(total)    -224.90406505
      E(total)/N     -74.96802168

Setting up a molecular dynamics run
-----------------------------------
The following input file is more involved and overrides various defaults in the
code. It loads a liquid water geometry from an external .xyz file and requests a
new NPT molecular dynamics trajectory.

.. literalinclude:: md_example.inp

All of the trajectory information will be logged to :data:`md_example.hdf5`
which is a convenient storage format for later trajectory analyses with Python.
It also serves as the restart file, which may be requested by the
:data:`md_restart_file` parameter in a later job to resume MD integration from
the last time step.
