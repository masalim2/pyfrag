Extending PyFragment
====================

When importing modules, all imports should be in the form of ::

    from pyfrag.Globals import logger, params
    from pyfrag.backend import nw

And then module attributes should be accessed with the notation::

    params.verbose

.. warning::
    Importing data directly from modules will yield variables that do not
    correctly update when the corresponding module-level variable is changed.
    This can result in very difficult bugs to track. NEVER import variables
    from modules if they may change during program execution!
