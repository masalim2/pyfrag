===================
Input to PyFragment
===================

Modular usage
===============
The modules of PyFragment can be imported into other Python programs or
interactive sessions. Then, the relevant calculation input can be set
programatically by interfacing with the **Globals** modules. The following
code snippet shows an example of the syntax: ::

    from pyfrag.Globals import params, geom
    from pyfrag.bim import bim
    params.options['basis'] = 'cc-pvtz'
    params.options['task'] = 'bim_grad'
    params.set_defaults()
    geomtxt = '''He 0 0 0
                 He 1 0 0
                 He 2 0 0'''
    geom.load_geometry(geomtxt) # build the geometry object
    geom.perform_fragmentation() # auto-fragment
    params.quiet = True # don't print anything
    result = bim.kernel() # get dictionary of results
    grad = result['gradient']
    print grad

All imports from PyFragment should be in the form of ::

    from pyfrag.Globals import logger, params
    from pyfrag.backend import nw

.. warning::
    **NEVER** import shared data directly from modules, as in:: 
    
        from pyfrag.Globals.params import options
     
    This will produce local objects that do not change in the scope of other
    modules when updated. This will result in very difficult bugs to track. By 
    importing the modules themselves and referencing their attributes, 
    data is correctly shared between the program modules.


Standalone execution
====================
If PyFragment is invoked from the command line, input must come in 
the form of an input file argument. The input format is somewhat flexible:

    * case-insensitive
    * ignores whitespace
    * ignores comments starting with '#' character

The parser recognizes two types of entries in the input file.
    
    1) **One line** entries use an = (equals sign) ::

        geometry = geom1.xyz

    2) **Multi-line** entries are enclosed in curly braces ::

        geometry {   
        2.0 0.0 0.0 90.0 90.0 0.0 0
        H 0 0 0
        F 1 0 0
        }

Input File Structure
********************
Here is a sample input file with comments explaining the meaning of the
parameters. The order of input does not matter and parameters irrelvant to the
calculation can be omitted.


.. literalinclude:: example.inp
