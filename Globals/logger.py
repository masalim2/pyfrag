'''logger module
  0) different logs can go to different files
  1) automatically opens a new file with useful/unique name when invoked for the first time
  2) keeps track of open file handles, closes when application is done
  3) takes input, formats it nicely, writes it to the appropriate file
    log_input(inp) -- the input file or input data for a QM calc
    log_output(output) -- the results and/or errors of a QM calc
    log_gopt
    log_cellopt
    log_MD
    log_PES
  -if superlog: 
     logs calculation input and output (for debugging)
  -if invoked by other drivers (PES or MD or OPTIMIZER), writes pretty logs out to disk

  log_calcs: each calc input and output is appended to a big log file
'''
from pyfrag.Globals import geom, lattice, neighbor, params
import numpy as np
import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

def log_input(inp):
    pass

def log_output(output):
    pass

def prettyprint_atoms(atoms, chgs=None):
    if chgs is None:
        for at in atoms:
            print "%2s %10.2f %10.2f %10.2f" % (at.sym.capitalize(), 
                    at.pos[0], at.pos[1], at.pos[2])
    else:
        for at,chg in zip(atoms, chgs):
            print "%2s %10.2f %10.2f %10.2f %10.2f" % (at.sym.capitalize(), 
                    at.pos[0], at.pos[1], at.pos[2], chg)

def print_geometry():
    print "Input Geometry / Angstroms"
    print "--------------------------"
    prettyprint_atoms(geom.geometry)
    print ""
    ndim = sum(lattice.PBC_flag)
    if ndim == 0:
        print "PBC OFF"
    else:
        print "%dD PBC Lattice Vectors" % ndim
        print "----------------------"
        with printoptions(precision=3, suppress=True):
            print lattice.lat_vecs
    print ""

def print_fragment(fragments=None, net_charges=None, esp_charges=None, charges_only=False):

    if fragments is None:
        fragments = geom.fragments
    if net_charges is None:
        net_charges = [geom.charge(frag) for frag in fragments]
    chgs = None
    
    for n, frag in enumerate(fragments):
        atoms = [geom.geometry[i] for i in frag]
        if esp_charges: chgs = [esp_charges[i] for i in frag]
        
        print "(Fragment %d Charge %+d)" %  (n, net_charges[n])
        if charges_only:
            for at, chg in zip(atoms, chgs):
                print "%2s  %6.2f" % (at.sym.capitalize(), chg)
        else:
            prettyprint_atoms(atoms, chgs)
        print "---------------------------------"
    print ""

def print_neighbors():
    print "%d unique dimers" % len(neighbor.dimer_lists)
    for d in neighbor.dimer_lists:
        (i,n0,n1,n2), (j,a,b,c) = d
        pair_str = ("%d--%d(%d,%d,%d)" % (i,j,a,b,c)).ljust(15)
        dist_str = ("  [%.2f Angstrom]" % 
                neighbor.pair_dist((i,j,a,b,c))).rjust(12)
        print "   ", pair_str + dist_str
    print "Embedding fields"
    for i, bqlist in enumerate(neighbor.bq_lists):
        print "   Fragment %d: %d molecules in bq field" % (i, len(bqlist))
    print ""
                
def print_parameters():
    options = params.options
    display = '''scrdir backend mem_mb basis hftype
              correlation embedding r_qm r_bq r_lr
              task'''.split()

    if 'vbct' in options['task']:
        display.extend('''diagonal relax_neutral_dimers
        corr_neutral_dimers coupling charge_states'''.split())

    if 'opt' in options['task'] or 'md' in options['task']:
        display.extend('''pressure freeze_cell atom_gmax lat_gmax
        opt_maxiter'''.split())

    if 'md' in options['task']:
        display.extend('''md_restart_file num_steps save_intval d_time
        temperature'''.split())

    if 'hess' in options['task']:
        display.append('interaction_cells')

    print "Input Parameters"
    print "----------------"
    width = max(len(s) for s in display)
    for k in display:
        if k in options:
            v = options[k]
        else:
            v = "None"
        print k.rjust(width), "    ", str(v)
    print ""

def print_bim_e_results(results):
    print "E", results['energy']

def print_vbct_e_results(results):
    print "VBCT Energy Results (Energy/Eigenvector)"
    E_nuclear = results['E_nuclear']
    eigvals = results['eigvals']
    eigvecs = results['eigvecs']
    for val, vec in zip(eigvals, eigvecs.T):
        print ("%12.8f" + len(vec)*"%9.4f") %((val+E_nuclear,)+tuple(vec))
