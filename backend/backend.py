import numpy as np
import sys
import os

from pyfrag.Globals import geom, lattice
from pyfrag.Globals import logger, params
from pyfrag.backend import nw, psi4

def build_atoms(frags, bq_list, bq_charges):
    '''Make the input geometry/embedding for a QM calculation.

    Args
        frags: a list of 4-tuples (i,a,b,c) where i is the fragment index,
        abc indicate the lattice cell.
        bq_list: a list of (4-tuples) indicating the molecules to be placed in
        the embedding field.
        bq_charges: a list of point charges for each atom in the geometry.
    Returns
        atoms: a list of Atoms for easy printing to the QM "geometry" input.
        bq_field: a list of numpy arrays(length-4) in format (x,y,z,q)
    '''

    lat_vecs = lattice.lat_vecs
    atoms = []
    for (i,a,b,c) in frags:
        vec = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]
        atoms.extend(geom.geometry[at].shift(vec) for at in geom.fragments[i])

    bq_field = []
    for (i,a,b,c) in bq_list:
        vec = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]
        for at in geom.fragments[i]:
            bq_pos = geom.geometry[at].shift(vec).pos
            bq_field.append(np.append(bq_pos, bq_charges[at]))

    return atoms, bq_field

def run(calc, frags, charge, bq_list, bq_charges, 
        noscf=False, guess=None, save=False):
    '''QM backend dispatcher: invoke a calculation.

    Currently, for ESP calculation the NWChem package is always dispatched
    for its higher performance.  Otherwise, the backend is determined by the
    params backend option.

    Args
        calc: one of esp, energy, gradient, hessian
        frags: list of 4-tuples (i,a,b,c) (See build_atoms documentation)
        charge: the net charge of the fragments in QM calculation
        bq_list: the embedding fragments, list of 4-tuples
        bq_charges: the embedding charges for each atom in the geometry
        noscf (default False): if True, only build Fock matrix from initial
          guess and diagonalize once.
        guess (default None): if supplied, provide prior MO for initial
          density.
        save (default False): if True, return a handle to the MO resulting
          from calculation.  For NW & Gaussian, this is a file path & MO file
          will be copied to a shared directory.  For Psi4 and PySCF, the MO
          vectors are directly returned as a 2D numpy array.
    Returns
        results: a dictionary packaging all of the results together. The
        contents of 'dictionary' depend on the type of calculation invoked and
        other arguments.
    '''
    options = params.options
    calc = calc.lower().strip()
    assert calc in 'esp energy gradient hessian'.split()
    assert isinstance(frags, list)
    assert all(isinstance(frag, tuple) and 
               len(frag)==4 for frag in frags)
    
    atoms, bq_field = build_atoms(frags, bq_list, bq_charges)
    if noscf and guess is None:
        raise RuntimeError("No SCF useless without input guess")
    
    if calc == 'esp':
        backend = nw
    else:
        backend = getattr(sys.modules[__name__], options['backend'])

    cwd = os.getcwd()
    os.chdir(params.options['scrdir'])
    inp = backend.inp(calc, atoms, bq_field, charge, noscf, guess, save)
    output = backend.calculate(inp, calc, save)
    os.chdir(cwd)
    if params.qm_logfile:
        logger.log_input(inp)
        logger.log_output(output)
    results = backend.parse(output, calc, inp, atoms, bq_field, save)
    if 'bq_gradient' in results:
        results['bq_list'] = bq_list
    else:
        results['bq_list'] = []
        results['bq_gradient'] = []
    return results
