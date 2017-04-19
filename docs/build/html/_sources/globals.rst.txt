The **Globals** Module
======================

The **Globals** module contains essential shared data and functionality 
for *all* types of fragment calculations. 

Globals.geom
------------
Defines the fundamental Atom class to conveniently load and print geometry
information.  Contains functions for loading geometry and performing
*fragmentation*, that is, assigning which atoms belong to which fragments.

.. autofunction:: Globals.geom.load_geometry
