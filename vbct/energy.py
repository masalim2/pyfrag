from mpi4py.MPI import ANY_SOURCE
import numpy as np
from scipy.linalg import eigh

from pyfrag.Globals import geom, logger, params
from pyfrag.Globals import MPI

def print_diagonal_calc_details(diag_results):
    for i, state in enumerate(charge_states):
        print state
        for monomer in state.monomers:
            atom_strs = map(str, monomer.atoms)
            charge_strs = ["   charge = %8.6f" % q for q in monomer.esp_charges]
            print "\n".join(["(frag %2d) %s %s" % (monomer.index, a, c) for a,c in zip(atom_strs, charge_strs)])
        print ""
    for i, state in enumerate(charge_states):
        header = str(state) + " E = %.6f" % H[i,i]
        print header
        print "-"*len(header)
        print " "*(len(header)/3) + "  1-BODY SUM (%.6f)" % state.E1
        print "    %10s %12s %12s %12s" % ("Fragment", "E_HF", "E_corr", "E_total")
        for monomer in state.monomers:
            if 'corr' not in monomer.energy: monomer.energy['corr'] = 0.0
            print "    %10s %12.6f %12.6f %12.6f" % (monomer.label(),
                    monomer.energy['hf'], monomer.energy['corr'],
                    monomer.energy['total'])
        print ""
        print " " * (len(header)/3) + "2-BODY SUM (%.6f)" % state.E2
        for dimer in state.dimers:
            if 'corr' not in dimer.energy: dimer.energy['corr'] = 0.0
            print "    %10s %12.6f %12.6f %12.6f" % (str(dimer.index),
                    dimer.energy['hf'], dimer.energy['corr'],
                    dimer.energy['total'])
        print "\n"

def verify_options():
    requisite = '''geometry
                    basis
                    hftype
                    correlation
                    embedding
                    diagonal
                    relax_neutral_dimers
                    corr_neutral_dimers
                    coupling
                    backend
                    task'''.split('\n')
    for opt in requisite:
        assert opt in params.options

def calc_diagonal(idx, comm=None):
    monomers = range(nfrag)
    charges  = [int(idx == j) for j in states]

    espfield, movecs = monomerSCF(monomers, charges, comm)
    diag_result = vbct_calc.diagonal(monomers, charges, espfield, movecs, comm)
    return diag_result

def kernel():
    '''SP energy'''

    verify_options()
    geom.perform_fragmentation()
    nfrag   = len(geom.fragments)
    nstates = nfrag
    states = range(nstates)

    if params.VERBOSE and MPI.rank == 0:
        logger.print_parameters()
        logger.print_geometry()
        print_vbct_states()
        print "\nMonomer SCF & Diagonal Element Calculation"
        print "------------------------------------------"

    diag_results = {}
    if MPI.nproc > nstates:
        work_comm, idx = MPI.create_split_comms(nstates)

        diag_result = calc_diagonal(idx, work_comm)

        if work_comm.Get_rank() == 0:
            MPI.comm.send(diag_result, dest=0, tag=idx)
        if MPI.rank == 0:
            for idx in states:
                diag_results[idx] = MPI.comm.recv(source=ANY_SOURCE, tag=idx)
        work_comm.Free()
    else:
        for idx in states:
            diag_results[idx] = calc_diagonal(idx)

    # log monomer SCF and diagonal elements
    if MPI.rank == 0 and params.VERBOSE:
        print_diagonal_calc_details(diag_results)

        print "Coupling and Overlap Matrix Elements"
        print "------------------------------------"

    # Couplings/overlaps
    pairs = [(i,j) for i in range(N-1) for j in range(i+1,N)]
    my_pairs = MPI.scatter(comm, pairs, master=0)
    couplings = [0.0] * len(my_pairs)
    overlaps  = [0.0] * len(my_pairs)

    for idx, (i,j) in enumerate(my_pairs):

        couplings[idx], overlaps[idx], info = charge_states[i].coupling(charge_states[j])

        if rank == 0 and nproc == 1 and params.VERBOSE:
            print "Coupling"
            for k, v in info.items():
                print "  %14s" % k, v
            print ""

    couplings = MPI.gather(comm, couplings, master=0)
    overlaps  = MPI.gather(comm, overlaps,  master=0)

    # Solve secular equation, return results
    if rank == 0:
        for idx, (i,j) in enumerate(pairs):
            H[i,j] = H[j,i] = couplings[idx]
            S[i,j] = S[j,i] =  overlaps[idx]

    if params.VERBOSE and rank == 0:
        print "Hamiltonian"
        print "-----------"
        print H
        if np.allclose(S, np.eye(N)):
            print "Overlap matrix: identity"
        else:
            print "Overlap matrix"
            print "--------------"
            print S

    E_nuclear = geom.nuclear_repulsion_energy(params.options['geometry'])
    for i in range(N):
        H[i,i] -= E_nuclear
    eigvals, eigvecs = eigh(H, b=S)
    prob0 = eigvecs[:,0]**2
    charge_distro = np.zeros((len(params.options['geometry'])))
    for i, state in enumerate(charge_states):
        charge_vec = np.array([q for m in state.monomers for q in m.esp_charges])
        charge_distro += prob0[i]*charge_vec

    results = {
                 'eigvals' : eigvals,
                 'eigvecs' : eigvecs,
                 'E_nuclear' : E_nuclear,
                 'E(GS)' : eigvals[0] + E_nuclear,
                 'chgdist(GS)' : charge_distro
              }
    if params.VERBOSE and rank == 0:
        print "Final Energy Calculation Results"
        for k,v in results.items():
            print k, "\n   ", v
    return results
