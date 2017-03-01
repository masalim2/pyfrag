import lattice as lat
import geom

def qm_accumulate_fxn(cell):
    '''count up dimers'''
    pass

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

def build_dimer_lists():
    mass_centers = [geom.com(frag) for frag in geom.fragments]
    for 
    pass

def build_bq_lists():
    pass

def eval_lr():
    pass
