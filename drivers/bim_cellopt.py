import numpy as np
from scipy.optimize import minimize

from pyfrag.Globals import lattice as lat
from pyfrag.Globals import geom, params, MPI, logger

def kernel():
    options = params.options
    params.quiet = True
    maxiter = options.get('maxiter', 50)
    at_gmax   = options.get('atom_gmax', 0.0015)
    press = options.get('pressure_bar', 0) / geom.AU2BAR
    
    cell0 = to_flat()
    res = minimize(objective_cellopt, cell0, args=(press,), method='Nelder-Mead',
            options=minopt)
    cellopt = res.x
    from_flat(cellopt)

    if MPI.rank == 0 and res.success:
        print "Optimization converged"
        logger.print_geometry()
    return {}

def objective_cellopt(lattice, press):
    '''enthalpy opt with respect to lattice parameters'''
    from pyfrag.drivers import bim_opt
    from_flat(lattice)
    lat.lattice  = MPI.bcast(lat.lattice, master=0)
    lat.lat_vecs = MPI.bcast(lat.lat_vecs, master=0)
    
    geom0 = bim_opt.to_flat()
    res = minimize(objective_gopt, geom0, method='BFGS', jac=True)
    new_geom = res.x
    from_flat(new_geom)
    geom.geometry = MPI.bcast(geom.geometry, master=0)
    
    E = res.fun
    enthalpy = E + press*lat.volume()*geom.ANG2BOHR**3
    enthalpy = MPI.bcast(enthalpy, master=0)
    return enthalpy
