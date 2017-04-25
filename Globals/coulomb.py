'''Evaluate classical point charge interactions between fragments'''
import numpy as np
from pyfrag.Globals import geom, params, MPI
from pyfrag.Globals import lattice as lat
from pyfrag.Globals.neighbor import BFS_lattice_traversal

energy_coulomb = 0.0
gradient_coulomb = np.zeros((len(geom.geometry), 3))
virial_coulomb = np.zeros((3,3))
com = []

def accumulate_pair(idx1, idx2, cell, scale, charges):
    '''Coluomb E, grad, virial for a specific dimer.

    The coulomb interactions are accumulated into module-level variables
    energy_coulomb, gradient_coulomb, and virial_coulomb.

    Args:
        idx1, idx2: indices of monomers
        cell: lattice vectors for molecule at idx2
        scale: scale factor for energy contribution
        charges: list of charges for the whole system
    Returns None
    '''

    global energy_coulomb, gradient_coulomb, virial_coulomb
    lat_vecs = lat.lat_vecs
    a,b,c = cell
    shift = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]

    frag1 = geom.fragments[idx1]
    com1  = geom.com(frag1)
    frag2 = geom.fragments[idx2]
    com2  = geom.com(frag2) + shift

    for i in frag1:
        ri = geom.geometry[i].pos
        qi = charges[i]
        for j in frag2:
            rj = geom.geometry[j].pos + shift
            qj = charges[j]
            rij = (rj - ri) * geom.ANG2BOHR #points i to j
            rij_norm = np.linalg.norm(rij)
            energy_coulomb += scale*qi*qj / rij_norm

            d_ri = scale * rij * qi * qj / rij_norm**3

            gradient_coulomb[i,:] += d_ri # same sign as r_ij
            gradient_coulomb[j,:] -= d_ri # opposite sign

            for p in range(3):
                for q in range(3):
                    virial_coulomb[p,q] -= com1[q]*geom.ANG2BOHR*d_ri[p]
                    virial_coulomb[p,q] += com2[q]*geom.ANG2BOHR*d_ri[p]

def coulomb_accumulator(cell, charges):
    '''Accumulate all the dimer interactions between cell 0 and a given cell.

    This is called by a generic breadth-first search method which traverses all
    unit cells outward from cell 0 (3D flood-fill).  The pair interactions are
    computed in parallel and must be accumulated later.

    Args:
        cell: a,b,c lattice indices of cell interacting with cell 0
        charges: list of esp charges on each fragment
    Returns:
        num_pairs: the number of interactions counted with the given cell
    '''
    options = params.options
    a, b, c = cell
    lat_vecs = lat.lat_vecs
    shift = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]
    num_pairs = 0
    num_fragments = len(geom.fragments)
    rQM2 = options['r_qm']**2
    rBQ2 = options['r_bq']**2
    rLR2 = options['r_lr']**2

    # add i to bqlist[j] && j to bqlist[i]
    # only count QM pair if i<j OR i==j and cell>0
    for i in range(MPI.rank, num_fragments, MPI.nproc):
        ri = com[i]
        for j in range(num_fragments):
            rj = com[j] + shift

            if cell == (0,0,0) and i == j:
                continue

            rij = rj - ri
            rij2 = np.dot(rij, rij)

            if rij2 < rQM2:
                num_pairs += 1
            elif rij2 < rBQ2:
                # correct overcount
                num_pairs += 1
                accumulate_pair(i, j, cell, -0.5, charges)
            elif rij2 < rLR2:
                # add LR interaction
                num_pairs += 1
                accumulate_pair(i, j, cell, 0.5, charges)
            elif num_pairs > 0 and rij2 > rLR2:
                # Complete the cell
                accumulate_pair(i, j, cell, 0.5, charges)

    return num_pairs

def evaluate_coulomb(espfield):
    '''Evaluate all coulomb interactions via BFS over cells'''
    global energy_coulomb, gradient_coulomb, virial_coulomb
    global com
    energy_coulomb = 0.0
    gradient_coulomb = np.zeros((len(geom.geometry), 3))
    virial_coulomb = np.zeros((3,3))
    com = [geom.com(frag) for frag in geom.fragments]
    BFS_lattice_traversal(coulomb_accumulator, charges=espfield)

    energy_coulomb = MPI.allreduce(MPI.comm, energy_coulomb)
    gradient_coulomb = MPI.allreduce(MPI.comm, gradient_coulomb)
    virial_coulomb = MPI.allreduce(MPI.comm, virial_coulomb)
