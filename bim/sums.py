'''BIM summation functions to compute full-system properties.

Each sum function takes the same arguments:
    specifiers: a list of specifier tuples (as defined in bim.kernel)

    calcs: a dict (with keys matching specifiers.keys()) of dicts containing the
        results from each fragment calculation.

and each returns a dictionary of calculation results.
'''
import numpy as np

from pyfrag.Globals import params
from pyfrag.Globals import geom, lattice
from pyfrag.Globals import coulomb

def energy_sum(specifiers, calcs):
    '''Determine total energy'''
    E1 = 0.0
    E2 = 0.0
    nfrag = len(geom.fragments)

    for m in specifiers[0:nfrag]:
        E1 += calcs[m]['E_tot']

    for i in range(nfrag, len(specifiers), 3):
        cij, ci, cj = specifiers[i:i+3]
        E2 += calcs[cij]['E_tot'] - calcs[ci]['E_tot'] - calcs[cj]['E_tot']

    E_coulomb = coulomb.energy_coulomb

    return {'E':E1+E2+E_coulomb, 'E1': E1, 'E2' : E2, 'Ec' : E_coulomb}

def gradient_sum(specifiers, calcs):
    '''compute gradient of total energy'''
    lat_vecs = lattice.lat_vecs
    natm  = len(geom.geometry)
    nfrag = len(geom.fragments)
    E1 = 0.0
    E2 = 0.0
    E_coulomb = coulomb.energy_coulomb
    grad1 = np.zeros((natm, 3))
    grad2 = np.zeros((natm, 3))
    vir1  = np.zeros((3,3))
    vir2  = np.zeros((3,3))

    # monomer sums
    for mon in specifiers[0:nfrag]:
        i, = mon
        atm_i   = geom.fragments[i]
        com     = geom.com(atm_i)

        E1 += calcs[mon]['E_tot']

        # QM gradient/virial
        grad0 = calcs[mon]['gradient']
        grad1[atm_i] += grad0
        for p in range(3):
            for q in range(3):
                vir1[p,q] -= (com[p]*geom.ANG2BOHR*grad0[:,q]).sum()

        # BQ gradient/virial
        bqlist = calcs[mon]['bq_list']
        bqgrad = calcs[mon]['bq_gradient']
        n = 0
        for (j,a,b,c) in bqlist:
            at_j   = geom.fragments[j]
            com_j  = geom.com(at_j)
            com_j += a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]

            grad0  = bqgrad[n:n+len(at_j)]
            grad1[at_j] += grad0
            for p in range(3):
                for q in range(3):
                    vir1[p,q] -= (com_j[p]*geom.ANG2BOHR*grad0[:,q]).sum()
            n += len(at_j)

    # Dimer sums
    for idim in range(nfrag, len(specifiers), 3):
        cij, ci, cj = specifiers[idim:idim+3]
        i,j,a,b,c   = cij

        atm_i  = geom.fragments[i]
        atm_j  = geom.fragments[j]
        atm_ij = atm_i + atm_j
        natm_i = len(atm_i)
        natm_j = len(atm_j)
        com_i  = geom.com(atm_i)
        com_j  = geom.com(atm_j)
        com_j += a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]

        E2 += calcs[cij]['E_tot'] - calcs[ci]['E_tot'] - calcs[cj]['E_tot']

        # QM gradient/virial
        grad_ij = calcs[cij]['gradient']
        grad_i  = np.vstack((calcs[ci]['gradient'], np.zeros((natm_j, 3))))
        grad_j  = np.vstack((np.zeros((natm_i, 3)),calcs[cj]['gradient']))
        grad0   = grad_ij - grad_i - grad_j
        grad2[atm_ij] += grad0

        for p in range(3):
            for q in range(3):
                vir2[p,q] -= (com_i[p]*geom.ANG2BOHR*grad0[0:natm_i,q]).sum()
                vir2[p,q] -= (com_j[p]*geom.ANG2BOHR
                              *grad0[natm_i:natm_i+natm_j,q]).sum()

        # BQ gradient/virial
        bqgrad_ij = calcs[cij]['bq_gradient']
        bqlist_ij = calcs[cij]['bq_list']
        bqgrad_i  =  calcs[ci]['bq_gradient']
        bqlist_i  =  calcs[ci]['bq_list']
        bqgrad_j  =  calcs[cj]['bq_gradient']
        bqlist_j  =  calcs[cj]['bq_list']
        grad0 = np.zeros(grad2.shape)
        vir0  = np.zeros((3,3))

        n = 0
        for (k,ka,kb,kc) in bqlist_ij:
            at_k   = geom.fragments[k]
            natm_k = len(at_k)
            com_k  = geom.com(at_k)
            com_k += ka*lat_vecs[:,0] + kb*lat_vecs[:,1] + kc*lat_vecs[:,2]

            g0 = bqgrad_ij[n:n+natm_k]
            grad0[at_k] += g0
            for p in range(3):
                for q in range(3):
                    vir0[p,q] -= (com_k[p]*geom.ANG2BOHR*g0[:,q]).sum()
            n += natm_k

        n = 0
        for (k,ka,kb,kc) in bqlist_i:
            at_k   = geom.fragments[k]
            natm_k = len(at_k)
            com_k  = geom.com(at_k)
            com_k += ka*lat_vecs[:,0] + kb*lat_vecs[:,1] + kc*lat_vecs[:,2]

            g0 = bqgrad_i[n:n+natm_k]
            grad0[at_k] -= g0
            for p in range(3):
                for q in range(3):
                    vir0[p,q] += (com_k[p]*geom.ANG2BOHR*g0[:,q]).sum()
            n += natm_k

        n = 0
        for (k,ka,kb,kc) in bqlist_j:
            at_k   = geom.fragments[k]
            natm_k = len(at_k)
            com_k  = geom.com(at_k)
            com_k += ka*lat_vecs[:,0] + kb*lat_vecs[:,1] + kc*lat_vecs[:,2]

            g0 = bqgrad_j[n:n+natm_k]
            grad0[at_k] -= g0
            for p in range(3):
                for q in range(3):
                    vir0[p,q] += (com_k[p]*geom.ANG2BOHR*g0[:,q]).sum()
            n += natm_k

        grad2 += grad0
        vir2  += vir0

    gcoul = coulomb.gradient_coulomb
    vircoul  = coulomb.virial_coulomb
    totalgradient = grad1+grad2+gcoul
    totalvirial   = vir1+vir2+vircoul

    E_result = {'E':E1+E2+E_coulomb, 'E1': E1, 'E2' : E2, 'Ec' : E_coulomb}
    g_result = {'gradient' : totalgradient, 'virial' : totalvirial, 'g1' : grad1,
            'g2' : grad2, 'gc' : gcoul, 'vir1' : vir1, 'vir2' : vir2, 'virc' :
            vircoul}
    g_result.update(E_result)
    return g_result

def hessian_sum(specifiers, calcs):
    '''compute interaction force constants (hessian)'''
    from itertools import product
    cell_ranges = params.options['interaction_cells']
    cell_ranges = map(int, cell_ranges)
    for i, pbc_flag in enumerate(lattice.PBC_flag):
        if not pbc_flag:
            cell_ranges[i] = 0
    Na,Nb,Nc = cell_ranges

    ncells = (2*Na+1)*(2*Nb+1)*(2*Nc+1)
    cells = list(enumerate(product(range(-Na,Na+1), range(-Nb,Nb+1),
        range(-Nc,Nc+1))))
    assert len(cells) == ncells
    cellmap = {(a,b,c) : i for (i, (a,b,c)) in cells}
    cell_list = np.array([ [a,b,c] for (i, (a,b,c)) in cells],dtype=np.int)

    natm  = len(geom.geometry)
    nfrag = len(geom.fragments)

    hess1 = np.zeros((3*natm,3*natm))
    hess2 = np.zeros((ncells,3*natm,3*natm))
    hess_result = {}

    # monomer sums
    for mon in specifiers[0:nfrag]:
        i, = mon
        hess0 = calcs[mon]['hess_tri']
        iatoms   = geom.fragments[i]
        natomi = len(iatoms)
        row0 = 3*iatoms[0]
        n = 0
        for row in range(row0, row0+3*natomi):
            for col in range(row0, row+1):
                hess1[row,col] += hess0[n]
                print row, col, hess0[n], i
                if col != row:
                    hess1[col,row] += hess0[n]
                    print col, row, hess0[n], i
                n += 1


    for idim in range(nfrag, len(specifiers), 3):
        cij, ci, cj = specifiers[idim:idim+3]
        i,j,a,b,c   = cij

        atm_i  = geom.fragments[i]
        atm_j  = geom.fragments[j]
        natm_i = len(atm_i)
        natm_j = len(atm_j)

        hess0 = calcs[cij]['hess_tri']
        n = 0
        for row00 in range(3*(natm_i+natm_j)):
            for col00 in range(row00+1):
                if row00 < 3*natm_i:
                    row0 = 3*atm_i[0]
                    col0 = row0
                    n0 = (0,0,0) # i(0)i(0)
                else:
                    assert natm_i == 3
                    row0 = 3*atm_j[0] - 3*natm_i
                    if col00 < 3*natm_i:
                        col0 = 3*atm_i[0]
                        n0 = (-a,-b,-c) # j(0)i(-n)
                    else:
                        col0 = row0
                        n0 = (0,0,0) # j(0)j(0)

                row = row0 + row00
                col = col0 + col00

                if abs(n0[0]) > Na or abs(n0[1]) > Nb or abs(n0[2]) > Nc:
                    n += 1
                    continue

                icell = cellmap[n0]
                hess2[icell, row, col] += hess0[n]
                print row, col, hess0[n], i, j

                if row != col or n0 != (0,0,0):
                    icell = cellmap[(-n0[0], -n0[1], -n0[2])]
                    hess2[icell, col, row] += hess0[n]
                    print col, row, hess0[n], i, j

                n += 1

        icell0 = cellmap[(0,0,0)]
        hess0 = calcs[ci]['hess_tri']
        n = 0
        row0 = 3*atm_i[0]
        for row in range(row0, row0+3*natm_i):
            for col in range(row0, row+1):
                hess2[icell0,row,col] -= hess0[n]
                print row, col, -hess0[n], i
                if col != row:
                    hess2[icell0,col,row] -= hess0[n]
                    print col, row, -hess0[n], i
                n += 1

        hess0 = calcs[cj]['hess_tri']
        n = 0
        row0 = 3*atm_j[0]
        for row in range(row0, row0+3*natm_j):
            for col in range(row0, row+1):
                hess2[icell0,row,col] -= hess0[n]
                print row, col, -hess0[n], j
                if col != row:
                    hess2[icell0,col,row] -= hess0[n]
                    print col, row, -hess0[n], j
                n += 1

    hess_total = hess2.copy()
    hess_total[icell0,:,:] += hess1
    hess_result = {'hess1' : hess1, 'hess2' : hess2, 'hess' : hess_total}
    hess_result['cell_list'] = cell_list
    return hess_result
