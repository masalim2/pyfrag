===================
Input to PyFragment
===================

Modular Usage
===============
The modules of PyFragment can be imported into other Python programs or
interactive sessions. Then, the relevant calculation input can be set
programatically by interfacing with the **globals.params** module. The following
code snippet shows an example of the syntax ::

    from pyfragment.globals import params, geom
    # ... other code here ...
    params.basis = 'cc-pvtz'
    params.fragmentation = 'auto'
    params.r_qm = 10.3
    params.geometry = '''He 0 0 0
                         He 1 0 0
                         He 2 0 0'''.split('\n')
    geom.load_geometry(params.geometry) # build the geometry object
    geom.set_frag_auto() # perform the fragmentation
    # ... more code here ...

Standalone Execution
====================
If PyFragment is invoked from the command line, input must come in 
the form of an input file argument. The input format is somewhat flexible:

    * **case-insensitive**
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

Sample File
***********
Here is a sample input file with comments explaining 
the meaning of the parameters 

.. literalinclude:: example.inp
