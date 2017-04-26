'''Compute GS doublet energy and charge distribution in VBCT scheme

    1) No PBC or truncation of summations: only cluster ions are calculated
    2) only charge +1 is supported: single hole hopping between fragments
'''
from mpi4py.MPI import ANY_SOURCE
import numpy as np
from scipy.linalg import eigh

from pyfrag.Globals import geom, logger, params
from pyfrag.Globals import MPI
from pyfrag.vbct import  vbct_calc
from pyfrag.vbct.monomerscf import monomerSCF, fullsys_best_guess
from  pyfrag.bim import bim

def build_secular_equations(diag_results, offdiag_results):
    '''Build Hamiltonian and overlap matrix'''
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

def verify_options():
    requisite = '''basis
                    hftype
                    correlation
                    embedding
                    vbct_scheme
                    backend'''.split()
    for opt in requisite:
        assert opt.strip() in params.options, "input needs %s" % opt.strip()
    assert len(geom.geometry) > 0


def calc_diagonal(idx, neutral_results, comm=None):
    '''Dispatch diagonal element calculation

    Args
        idx: integer index ranging from 0-len(geom.fragments). Which
        diagonal element to calculate
        comm: MPI communicator or subcommunicator
    Returns
        res: dict, results from diagonal element calculation.
    '''
    diag_calc_name = "diag_%s" % params.options['vbct_scheme']
    diag_calc_fxn = getattr(vbct_calc, diag_calc_name)

    nfrag = len(geom.fragments)
    monomers = range(nfrag)
    charges  = [int(idx == j) for j in monomers]

    if params.options['fragmentation'] == 'full_system':
        res = fullsys_best_guess(comm=comm)
    else:
        espfield, movecs = monomerSCF(monomers, charges, comm=comm)
        res = diag_calc_fxn(charges, espfield, movecs, neutral_results, comm=comm)
    return res


def calc_chg_distro(prob0, diag_results):
    '''Generate linear combination of esp charge distros

    Args
        prob0: ground state probability distribution
        diag_results: diagonal element calculation results list,
            contains esp_charges for each charge-local state
    Returs
        charge_distro: approximate charge distribution computed as linear
        combination of VBCT basis charge distros
    '''
    charge_distro = np.zeros(len(geom.geometry))
    for i, diag_calc in enumerate(diag_results):
        charge_vec = np.array(diag_calc['esp'])
        charge_distro += charge_vec*prob0[i]
    return charge_distro


def setup_and_get_bim_energy():
    nfrag = len(geom.fragments)
    assert all([geom.charge(i) == 0 for i in range(nfrag)])
    params.verbose = False
    params.quiet = True
    params.options['task'] = 'bim_e' # hack to get bim energy
    params.options['r_qm'] = 1e15
    params.options['r_bq'] = 1e15
    params.options['r_lr'] = 1e15
    res = bim.kernel()
    res = MPI.bcast(res, master=0)
    params.options['task'] = 'vbct_e'
    return res

def kernel():
    '''SP energy: form and diagonalize VBCT matrix'''

    # Setup
    verify_options()
    geom.perform_fragmentation()
    nfrag   = len(geom.fragments)
    nstates = nfrag
    states = range(nstates)

    if params.verbose and MPI.rank == 0:
        logger.print_vbct_init(states)

    # Diagonal element calculation

    if params.options['vbct_scheme'] == 'monoip' and nfrag > 1:
        neutral_res = setup_and_get_bim_energy()
    else:
        neutral_res = []

    diag_results = []

    if MPI.nproc > nstates:
        work_comm, idx = MPI.create_split_comms(nstates)
        diag_result = calc_diagonal(idx, neutral_res, comm=work_comm)

        if work_comm.Get_rank() == 0:
            MPI.comm.send(diag_result, dest=0, tag=idx)

        if MPI.rank == 0:
            for idx in states:
                res = MPI.comm.recv(source=ANY_SOURCE, tag=idx)
                diag_results.append(res)

        work_comm.Free()
    else:
        for idx in states:
            res = calc_diagonal(idx, neutral_res)
            diag_results.append(res)

    if MPI.rank == 0 and params.verbose:
        logger.print_diagonal_calc_details(diag_results)

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
        logger.print_offdiag_calc_details(pairs, offdiag_results)

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
