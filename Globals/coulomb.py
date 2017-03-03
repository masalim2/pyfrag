import numpy as np
from pyfrag.Globals import geom
from pyfrag.Globals import lattice as lat
from pyfrag.Globals.neighbor import BFS_lattice_traversal

energy_coulomb = 0.0
gradient_coulomb = np.zeros((len(geom.geometry), 3))
virial_coulomb = np.zeros((3,3))
com = []

def accumulate_pair(idx1, idx2, cell, scale, charges):
    global energy_coulomb, gradient_coulomb, virial_coulomb
    lat_vecs = lat.lat_vecs
    a,b,c = cell
    shift = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]

    frag1 = geom.fragments[idx1]
    com1  = geom.com(frag1)
    frag2 = geom.fragments[idx2]
    com2  = geom.com(frag2) + shift
    
    for i in frag1:
        ri = geom.geometry[i]
        qi = charges[i]
        for j in frag2:
            rj = geom.geometry[j] + shift
            qj = charges[j]
            rij = (rj - ri) * geom.ANG2BOHR #points i to j
            rij_norm = np.linalg.norm(rij)
            energy_coulomb += scale*qi*qj / rij_norm

            d_ri = scale * rij * qi * qj / rij_norm**3

            gradient_coulomb[i,:] += d_ri # same sign as r_ij
            gradient_coulomb[j,:] -= d_ri # opposite sign

            for p in range(3):
                for q in range(3):
                    virial_coulomb[p,q] -= com1[q]*geom.ang2bohr*d_ri[p]
                    virial_coulomb[p,q] += com2[q]*geom.ang2bohr*d_ri[p]

def coulomb_accumulator(cell, charges):
    '''count up dimers'''
    a, b, c = cell
    lat_vecs = lat.lat_vecs
    shift = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]
    num_pairs = 0
    num_fragments = len(geom.fragments)
    rQM2 = params.options['r_qm']**2
    rBQ2 = params.options['r_bq']**2
    rLR2 = params.options['r_lr']**2

    # add i to bqlist[j] && j to bqlist[i]
    # only count QM pair if i<j OR i==j and cell>0
    for i in range(num_fragments):
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
                accumulate_pair(i, j, cell, -0.5, charges) # BUG different from qcbim
            elif rij2 < rLR2:
                # add LR interaction
                num_pairs += 1
                accumulate_pair(i, j, cell, 0.5, charges)
    return num_pairs

def evaluate_coulomb(espfield):
    global energy_coulomb, gradient_coulomb, virial_coulomb
    global com
    energy_coulomb = 0.0
    gradient_coulomb = np.zeros((len(geom.geometry), 3))
    virial_coulomb = np.zeros((3,3))
    com = [geom.com(frag) for frag in geom.fragments]

    BFS_lattice_traversal(coulomb_accumulator, charges=espfield)
