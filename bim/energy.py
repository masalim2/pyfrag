from pyfrag.Globals import logger, MPI, params
from pyfrag.Globals import geom, lattice
from pyfrag.Globals import neighbor, coulomb

def kernel():
    VERB = params.verbose
    if VERB and MPI.rank == 0: logger.print_geometry()
    
    # Perform fragmentation
    if params.fragmentation == 'auto':
        geom.set_frag_auto()
    elif params.fragmentation == "full_system":
        geom.set_frag_full_system()
    else:
        geom.set_frag_manual()
    if VERB and MPI.rank == 0: logger.print_fragment()

    # Get neighbor lists
    neighbor.build_lists()
    if VERB and MPI.rank == 0: logger.print_neighbors()

    return {'energy' : 0.0}
