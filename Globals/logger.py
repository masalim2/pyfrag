''' Convenience functions for pretty printing/file IO
'''
from pyfrag.Globals import geom, lattice, neighbor, params
import numpy as np
import os
import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

def log_input(inp):
    homedir = params.options['home_dir']
    with open(os.path.join(homedir, params.qm_logfile), 'a') as fp:
        fp.write(open(inp).read()+'\n')

def log_output(output):
    if type(output) is list:
        output = '\n'.join(output)
    if type(output) is not str:
        return
    homedir = params.options['home_dir']
    with open(os.path.join(homedir, params.qm_logfile), 'a') as fp:
        fp.write(output)

def pretty_matrix(mat, precision=2, name=''):
    rows, cols = mat.shape
    float_fmt = '.' + str(precision) +'f'
    s_max = ("%"+float_fmt) % mat.max()
    s_min = ("%"+float_fmt) % mat.min()
    width = max(len(s_max), len(s_min)) + 1
    format = "%" + str(width) + float_fmt
    if name:
        print name
        print '-'*len(name)
    for r in range(rows):
        for c in range(cols):
            print format % mat[r,c],
        print ""
    print ""
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

        print "(Frag %d Chg %+d)" %  (n, net_charges[n])
        if charges_only:
            for at, chg in zip(atoms, chgs):
                print "%2s  %6.2f" % (at.sym.capitalize(), chg)
        else:
            prettyprint_atoms(atoms, chgs)
        print "----------"
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
        display.extend(['vbct_scheme'])

    if 'opt' in options['task']:
        display.extend('''pressure_bar freeze_cell atom_gmax lat_gmax
        opt_maxiter'''.split())

    if 'md' in options['task']:
        display.extend('''pressure_bar md_restart_file num_steps save_intval dt_fs
        temperature t_bath p_bath'''.split())

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
    print "%12s %16.8f" % ('E(monomer)', results['E1'])
    print "%12s %16.8f" % ('E(dimer)', results['E2'])
    print "%12s %16.8f" % ('E(coulomb)', results['Ec'])
    print "-----------------------------"
    print "%12s %16.8f" % ('E(total)',   results['E'])
    print "%12s %16.8f" % ('E(total)/N', results['E']/len(geom.fragments))

def print_bim_grad_results(results):
    print_bim_e_results(results)
    print ""
    print "BIM ENERGY GRADIENTS / a.u."
    print "--------------------"
    with printoptions(precision=4, suppress=True):
        print results['gradient']
    print ""
    print "VIRIAL STRESS TENSOR / bar"
    print "--------------------------"
    with printoptions(precision=3, suppress=True):
        stress = results['virial'] / (lattice.volume()*geom.ANG2BOHR**3)
        print stress*geom.AU2BAR
    print ""

def print_bim_hess_results(results):
    opts = params.options
    fname = params.args.input_file
    head, tail = os.path.split(fname)
    base, ext = os.path.splitext(tail)
    fname = os.path.join(opts['home_dir'], base+'.hess')
    n = 0
    while os.path.exists(fname):
        fname = '%s%d.hess' % (base, n)
        fname = os.path.join(opts['home_dir'], fname)
        n += 1
    print "Writing hessian data out to %s" % fname
    np.savez(fname, hess=results['hess'], cell_list=results['cell_list'])

def print_vbct_e_results(results):
    print "VBCT Energy Results (Energy/Eigenvector)"
    E_nuclear = results['E_nuclear']
    eigvals = results['eigvals']
    eigvecs = results['eigvecs']
    for val, vec in zip(eigvals, eigvecs.T):
        print ("%12.8f" + len(vec)*"%9.4f") %((val+E_nuclear,)+tuple(vec))

def print_bim_md_results(results):
    pass

def print_bim_opt_results(results):
    pass


def print_vbct_state(idx):
    nfrag = len(geom.fragments)
    frag_labels = [fraglabel(i,i==idx) for i in range(nfrag)]
    print "State: " + ''.join(frag_labels)

def print_vbct_init(states):
    '''Print header for calculation'''
    print_parameters()
    print_geometry()
    print "VBCT Basis"
    print "----------"
    for idx in states:
        print_vbct_state(idx)
    print "\nMonomer SCF & Diagonal Element Calculation"
    print "------------------------------------------"

def fraglabel(frag, chg):
    '''Generate chemical formula string for a fragment'''
    atoms = [geom.geometry[i].sym for i in geom.fragments[frag]]
    atom_counts = { sym : atoms.count(sym) for sym in set(atoms) }
    label = '('
    for sym, count in sorted(atom_counts.items()):
        label += sym.capitalize()
        if count > 1: label += str(count)
    label += ')'
    if chg != 0: label += "%+d" % chg
    return label

def print_diagonal_calc_details(diag_results):
    if params.options['vbct_scheme'] == 'chglocal':
        print_chglocal_diag(diag_results)
    elif params.options['vbct_scheme'] == 'monoip':
        print_monoip_diag(diag_results)

def print_chglocal_diag(diag_results):
    from itertools import combinations
    dimer_idxs = list(combinations(range(len(diag_results)), 2))
    for i, res in enumerate(diag_results):
        print_vbct_state(i)
        print_fragment(net_charges=res['net_charges'],
                              esp_charges=res['esp'], charges_only=True)
        header = " E = %.6f" % (res['E1'] + res['E2'])
        print header
        print "-"*len(header)
        print " "*(len(header)/3) + "  1-BODY SUM (%.6f)" % res['E1']
        print "    %10s %12s %12s %12s" % ("Fragment", "E_HF", "E_corr", "E_total")
        for j, mon in enumerate(res['monomers']):
            print "    %10s %12.6f %12.6f %12.6f" % (fraglabel(j,int(j==i)),
                    mon['E_hf'], mon.get('E_corr', 0.0),
                    mon['E_tot'])
        print ""
        print " " * (len(header)/3) + "2-BODY SUM (%.6f)" % res['E2']
        for dimidx, dim in zip(dimer_idxs, res['dimers']):
            print "    %10s %12.6f %12.6f %12.6f" % (str(dimidx),
                    dim['E_hf'], dim.get('E_corr', 0.0),
                    dim['E_tot'])
        print "\n"
    print "Coupling and Overlap Matrix Elements"
    print "------------------------------------"

def print_monoip_diag(diag_results):
    from itertools import combinations
    dimer_idxs = list(combinations(range(len(diag_results)), 2))
    for i, res in enumerate(diag_results):
        print_vbct_state(i)
        print_fragment(net_charges=res['net_charges'],
                              esp_charges=res['esp'], charges_only=True)
        header = " E = %.6f" % (res['E1'] + res['E2'])
        print header
        print "-"*len(header)
        print " "*(len(header)/3) + "  1-BODY SUM (%.6f)" % res['E1']
        print "    %10s %12s %12s %12s" % ("Fragment", "E_HF", "E_corr", "E_total")

        for j, mon in enumerate(res['monomers']):
            print "    %10s %12.6f %12.6f %12.6f" % (fraglabel(j,0),
                    mon['E_hf'], mon.get('E_corr', 0.0),
                    mon['E_tot'])

        mon = res['monomer_cation']
        print "    %10s %12.6f %12.6f %12.6f" % (fraglabel(i,1),
                mon['E_hf'], mon.get('E_corr', 0.0),
                mon['E_tot'])
        print ""

        print " " * (len(header)/3) + "2-BODY SUM (%.6f)" % res['E2']
        for dimidx, dim in zip(dimer_idxs, res['dimers']):
            print "    %10s %12.6f %12.6f %12.6f" % (str(dimidx),
                    dim['E_hf'], dim.get('E_corr', 0.0),
                    dim['E_tot'])
        print "\n"

    print "Coupling and Overlap Matrix Elements"
    print "------------------------------------"

def print_offdiag_calc_details(pairs, offdiag_results):
    for pair, res in zip(pairs, offdiag_results):
        print "Coupling", pair
        print "---------------"
        for k,v in res.items():
            print "    %s <--> %s" % (str(k), str(v))
