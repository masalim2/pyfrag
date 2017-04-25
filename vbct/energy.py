'''Compute GS doublet energy and charge distribution in VBCT scheme
1) No PBC or truncation of summations: only cluster ions are calculated
2) only charge +1 is supported: single hole hopping between fragments
'''
from mpi4py.MPI import ANY_SOURCE
import numpy as np
from scipy.linalg import eigh
from itertools import combinations

from pyfrag.Globals import geom, logger, params
from pyfrag.Globals import MPI
from pyfrag.vbct import  vbct_calc
from pyfrag.vbct.monomerscf import monomerSCF, fullsys_best_guess

def print_vbct_init(states):
    '''Print header for calculation'''
    logger.print_parameters()
    logger.print_geometry()
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


def print_vbct_state(idx):
    nfrag = len(geom.fragments)
    frag_labels = [fraglabel(i,i==idx) for i in range(nfrag)]
    print "State: " + ''.join(frag_labels)


def build_secular_equations(diag_results, offdiag_results):
    N = len(diag_results)
    H = np.zeros((N,N))
    S = np.eye(N)
    for i in range(N):
        res = diag_results[i]
        H[i,i] = res['E1'] + res['E2']
    for res in offdiag_results:
        i,j = res['idx']
        H[i,j] = H[j,i] = res['coupling']
    return H, S


def print_diagonal_calc_details(diag_results):
    dimer_idxs = list(combinations(range(len(diag_results)), 2))
    for i, res in enumerate(diag_results):
        print_vbct_state(i)
        logger.print_fragment(net_charges=res['net_charges'],
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


def print_offdiag_calc_details(pairs, offdiag_results):
    for pair, res in zip(pairs, offdiag_results):
        print "Coupling", pair
        print "---------------"
        for k,v in res.items():
            print "    %s <--> %s" % (str(k), str(v))


def verify_options():
    requisite = '''geometry
                    basis
                    hftype
                    correlation
                    embedding
                    vbct_scheme
                    backend
                    task'''.split('\n')
    for opt in requisite:
        assert opt.strip() in params.options, "input needs %s" % opt


def calc_diagonal(idx, comm=None):

    diag_calc_name = "diag_%s" % params.options['vbct_scheme']
    diag_calc_fxn = getattr(vbct_calc, diag_calc_name)

    nfrag = len(geom.geometry)
    monomers = range(nfrag)
    charges  = [int(idx == j) for j in monomers]

    if params.options['fragmentation'] == 'full_system':
        res = fullsys_best_guess(comm=comm)
    else:
        espfield, movecs = monomerSCF(monomers, charges, comm=comm)
        res = diag_calc_fxn(charges, espfield, movecs, comm=comm)
    return res


def calc_chg_distro(prob0, diag_results):
    charge_distro = np.zeros(len(geom.geometry))
    for i, diag_calc in enumerate(diag_results):
        charge_vec = np.array(diag_calc['esp'])
        charge_distro += charge_vec*prob0[i]
    return charge_distro


def kernel():
    '''SP energy'''

    # Setup
    verify_options()
    geom.perform_fragmentation()
    nfrag   = len(geom.fragments)
    nstates = nfrag
    states = range(nstates)

    if params.verbose and MPI.rank == 0:
        print_vbct_init(states)

    # Diagonal element calculation
    diag_results = []

    if MPI.nproc > nstates:
        work_comm, idx = MPI.create_split_comms(nstates)
        diag_result = calc_diagonal(idx, work_comm)

        if work_comm.Get_rank() == 0:
            MPI.comm.send(diag_result, dest=0, tag=idx)

        if MPI.rank == 0:
            for idx in states:
                res = MPI.comm.recv(source=ANY_SOURCE, tag=idx)
                diag_results.append(res)

        work_comm.Free()
    else:
        for idx in states:
            res = calc_diagonal(idx)
            diag_results.append(res)

    if MPI.rank == 0 and params.verbose:
        print_diagonal_calc_details(diag_results)

    # Couplings/overlaps calculation
    coupl_name = "coupl_%s" % params.options['vbct_scheme']
    calc_coupling = getattr(vbct_calc, coupl_name)

    pairs = [(i,j) for i in range(nstates-1) for j in range(i+1,nstates)]
    my_pairs = MPI.scatter(MPI.comm, pairs, master=0)

    offdiag_results = []

    for (i,j) in my_pairs:
        res = calc_coupling(i, j)
        offdiag_results.append(res)

    offdiag_results = MPI.gather(MPI.comm, offdiag_results, master=0)

    if MPI.rank > 0:
        return {}

    if MPI.rank == 0 and params.verbose:
        print_offdiag_calc_details(pairs, offdiag_results)

    # Construct eigensystem; solve for ground state
    H, S = build_secular_equations(diag_results, offdiag_results)

    if params.verbose and MPI.rank == 0:
        logger.pretty_matrix(H, precision=6, name="Hamiltonian")
        logger.pretty_matrix(S, precision=6, name="Overlap")

    E_nuclear = geom.nuclear_repulsion_energy()
    for i in states:
        H[i,i] -= E_nuclear

    eigvals, eigvecs = eigh(H, b=S)
    prob0 = eigvecs[:,0]**2
    charge_distro = calc_chg_distro(prob0, diag_results)

    results = {
                 'eigvals' : eigvals,
                 'eigvecs' : eigvecs,
                 'E_nuclear' : E_nuclear,
                 'E(GS)' : eigvals[0] + E_nuclear,
                 'chgdist(GS)' : charge_distro
              }
    if params.verbose and MPI.rank == 0:
        print "Final Energy Calculation Results"
        for k,v in results.items():
            print k, "\n   ", v
    return results
