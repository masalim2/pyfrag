'''Backend for Psi4 -- importing Psi4 module directly'''
import numpy as np
import psi4

from pyfrag.Globals import params, geom

def calculate(inp, calc, save):
    '''Run nwchem on input, return data in results dict

    Args
        inp: Psi4 input object (geometry, bq_pos, bq_charges)
        calc: calculation type
        save: save calculation results
    Returns
        results: dictionary
    '''
    options = params.options
    mol, bqfield, bqs = inp
    psi4.core.set_global_option_python('EXTERN', bqfield.extern)
    psi4.core.set_global_option('BASIS', options['basis'])
    psi4.set_memory(int(options['mem_mb']*1e6))

    if options['correlation'] and calc not in ['esp', 'energy_hf']:
        theory = options['correlation']
        psi4.core.set_global_option('freeze_core', 'True')
    else:
        theory = 'scf'
    method_str = theory + '/' + options['basis']

    results = {}
    if calc == 'energy' or calc == 'energy_hf':
        results['E_tot'] = psi4.energy(method_str)
    elif calc == 'esp':
        raise RuntimeError("Not implemented")
    elif calc == 'gradient':
        grad, wfn = psi4.gradient(method_str, return_wfn=True)
        E = psi4.core.get_variable('CURRENT ENERGY')
        results['E'] = E
        results['gradient'] = psi4.p4util.mat2arr(grad)
        with open('grid.dat', 'w') as fp:
            for bq in bqs:
                fp.write('%16.10f%16.10f%16.10f\n' % (bq[0], bq[1], bq[2]))
        psi4.oeprop(wfn, 'GRID_FIELD')
        bq_field = np.loadtxt('grid_field.dat')
        assert bq_field.shape == (len(bqs), 3)
        bqgrad = []
        for chg, field_vec in zip(bqs, bq_field):
            bqgrad.append(-1.0 * chg[3] * field_vec)
        results['bq_gradient'] = np.array(bqgrad)
    elif calc == 'hessian':
        hess = psi4.hessian(theory)
        hessarr = np.array(psi4.p4util.mat2arr(hess))
        results['hess'] = hessarr
    return results

def inp(calc, atoms, bqs, charge, noscf=False, guess=None, save=False):
    '''Prepare Psi4 module by setting geometry and BQ field'''

    geom_str = '\n'.join(map(str, atoms))
    geom_str += "\nsymmetry c1\nno_reorient\nno_com\n"
    mol = psi4.geometry(geom_str)
    mol.update_geometry()

    nelec = sum(geom.z_map[at.sym] for at in atoms) - charge
    mult = nelec % 2
    mol.set_multiplicity(mult)
    mol.set_molecular_charge(charge)

    bqfield = psi4.QMMM()
    for bq in bqs:
        bqfield.extern.addCharge(bq[3], bq[0], bq[1], bq[2])

    return mol, bqfield, bqs

def parse(data, calc, inp, atoms, bqs, save):
    '''Return data; for compatibilility with generic backend'''
    return data
