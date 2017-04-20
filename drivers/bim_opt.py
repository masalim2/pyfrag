'''Geometry optimization (fixed unit cell)'''
import numpy as np
from scipy.optimize import minimize

from pyfrag.Globals import lattice as lat
from pyfrag.Globals import geom, params, MPI
from pyfrag.bim import bim

def to_flat():
    '''return 1D ndarray input for objective fxn'''
    return geom.pos_array().flatten()*geom.ANG2BOHR

def from_flat(x):
    '''reconstruct geom from flat ndarray'''
    ii = 0
    for iat in range(len(geom.geometry)):
        for mu in range(3):
            geom.geometry[iat].pos[mu] = x[ii]/geom.ANG2BOHR
            ii += 1

def objective_gopt(x):
    '''energy opt wrt geometry (using E and gradient)

    This objective function packs/unpacks the geometry from
    a 1D ndarray, updates geom.geometry, and invokes bim.kernel
    to update the energy/gradient.
    '''
    from_flat(x)
    geom.geometry = MPI.bcast(geom.geometry, master=0)
    results = bim.kernel()
    E = results['E']
    grad = results['gradient']
    if MPI.rank == 0:
        print "%18.8f %18.8f" % (E, np.abs(grad).mean())
    return E, grad.flatten()

def gopt_callback(xk):
    '''Report geometry by appending to gopt.xyz'''
    if MPI.rank != 0: return
    with open('gopt.xyz', 'a') as fp:
        fp.write('%d\n' % len(geom.geometry))
        fp.write( (6*'%.6f ' + ' %d\n')%tuple(lat.lattice))
        for at in geom.geometry:
            fp.write(str(at) + '\n')

def kernel():
    '''Invoke scipy optimizer'''
    options = params.options
    params.quiet = True
    maxiter = options.get('maxiter', 50)
    at_gmax   = options.get('atom_gmax', 0.0015)

    minopt = {'maxiter' : maxiter, 'gtol': at_gmax}
    geom0 = to_flat()

    if MPI.rank == 0:
        print "%18s %18s" % ("Energy/au", "MAGrad/au/bohr")
    res = minimize(objective_gopt, geom0, method='BFGS', jac=True,
            options=minopt, callback=gopt_callback)

    new_geom = res.x
    from_flat(new_geom)
    if MPI.rank == 0 and res.success:
        print "Optimization converged"
    return {}
