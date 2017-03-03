from pyfrag.Globals import lattice as lat
from pyfrag.Globals import geom
from pyfrag.Globals import params
from itertools import combinations
import numpy as np

# globally shared neighbor lists
dimer_lists = []
bq_lists = [ [] for i in range(len(geom.fragments)) ]

def pair_dist(pair_tup):
    i,j,a,b,c = pair_tup
    lat_vecs = lat.lat_vecs
    shift = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]
    ri = geom.com(geom.fragments[i])
    rj = geom.com(geom.fragments[j]) + shift
    return np.linalg.norm(ri - rj)

def pairlist_accumulator(cell, com):
    '''count up dimers'''
    a, b, c = cell
    lat_vecs = lat.lat_vecs
    shift = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]
    num_pairs = 0
    num_fragments = len(geom.fragments)
    rQM2 = params.options['r_qm']**2
    rBQ2 = params.options['r_bq']**2

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
                bq_lists[i].append( (j,a,b,c) )
                if i < j:
                    dimer_lists.append( [(i,0,0,0),(j,a,b,c)] )
                elif i == j and cell > (0,0,0):
                    dimer_lists.append( [(i,0,0,0),(j,a,b,c)] )
            elif rij2 < rBQ2:
                num_pairs += 1
                bq_lists[i].append( (j,a,b,c) )
    return num_pairs

def BFS_lattice_traversal(pair_accumulate_fxn, **args):
    '''Generic floodfill algorithm to visit cells & build neighbor lists.

    This function performs a Breadth-First Search (BFS) starting from the
    central unit cell (0,0,0) and moving outwards in all periodic dimensions.
    It requires an accumulation function as its first argument, which is used
    to count up all the relevant interactions between origin and a given cell
    (a,b,c). Once no more interactions are counted, the outward fill ends.

    Args
        pair_accumulate_fxn: function of the form f(cell, **args) 
            which builds neighbor lists between the given cell and origin
            cell.  It must return the number of pairs within range. 
        **args: additional arguments to pair_accumulate_fxn
    Returns
        None
    '''
    cells   = [(0,0,0)]
    visited = [(0,0,0)]

    while cells:
        cell0 = cells.pop(0)
        a,b,c = cell0
        num_pairs = pair_accumulate_fxn(cell0, **args)
        if num_pairs > 0:
            for dim in range(3):
                if lat.PBC_flag[dim]:
                    newcell = list(cell0)
                    newcell[dim] += 1
                    newcell = tuple(newcell)
                    if newcell not in visited:
                        cells.append(newcell)
                        visited.append(newcell)
                    
                    newcell = list(cell0)
                    newcell[dim] -= 1
                    newcell = tuple(newcell)
                    if newcell not in visited:
                        cells.append(newcell)
                        visited.append(newcell)

def build_lists():
    global dimer_lists, bq_lists
    mass_centers = [geom.com(frag) for frag in geom.fragments]

    # clear out neighbor lists
    dimer_lists[:] = []
    bq_lists[:] = [ [] for i in range(len(geom.fragments)) ]

    # build pair lists
    BFS_lattice_traversal(pairlist_accumulator, com=mass_centers)
