import numpy as np

from pyfrag.Globals import MPI
from pyfrag.Globals import params
from pyfrag.Globals import geom, neighbor
from pyfrag.backend import backend

def monomerSCF(comm=None):
    '''Cycle embedded monomer calculations until ESP charges converge.

    BIM version: include all monomers, take charges from input geometry,
    bq_lists from Globals.neighbor, and embedding option from input file. PBC is
    implicitly handled No need to do anything if embedding option is off.

    Args:
        comm: specify a sub-communicator for parallel execution.
            Default None: use the top-level communicator in Globals.MPI
    Returns:
        espcharges: a list of esp-fit atom-centered charges
    '''
    if comm is None:
        comm = MPI.comm

    options = params.options

    RMSD_TOL = 0.001
    MAXITER  = 10
    RMSD = RMSD_TOL + 1
    itr = 0

    espcharges = [0.0 for at in geom.geometry]
    if not options['embedding']: return espcharges
    natm = float(len(geom.geometry))

    while RMSD > RMSD_TOL and itr < MAXITER:

        espcharges0 = espcharges[:] # copy
        myfrags = MPI.scatter(comm, range(len(geom.fragments)), master=0)
        mycharges = []

        for m in myfrags:
            fragment = [(m,0,0,0)]
            net_chg = geom.charge(m)
            bqlist = neighbor.bq_lists[m]
            result = backend.run('esp', fragment, net_chg, bqlist, espcharges)
            mycharges.append(result['esp_charges'])

        espcharges = MPI.allgather(comm, mycharges)
        espcharges = [chg for m in espcharges for chg in m]
        residual = np.array(espcharges) - np.array(espcharges0)
        RMSD = np.linalg.norm(residual) / natm**0.5
        itr += 1

    if RMSD > RMSD_TOL:
        raise RuntimeError("Monomer SCF did not converge")
    else:
        return espcharges
