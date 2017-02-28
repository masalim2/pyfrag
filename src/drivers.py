import inputdata as inp
import geometry as geom
from mpi4py.MPI import ANY_SOURCE
import numpy as np
from scipy.linalg import eigh
from ChargeState import ChargeState

def enumerate_states():
    '''Return the list of ChargeStates according to input options'''

    charge_states = []
    geometry = inp.inputdata['geometry']
    
    # Fragmentation of geometry
    if 'fragmentation' in inp.inputdata:
        frag_option = inp.inputdata['fragmentation']
    else:
        frag_option = 'auto'
    
    if isinstance(frag_option, list):
        fragments = [map(int, s.split()) for s in frag_option]
    else:
        frag_method = getattr(geom, 'makefrag_%s' % frag_option)
        fragments = frag_method(geometry)

    # Assign charges to fragments
    if 'charge_states' in inp.inputdata: 
        states_option = inp.inputdata['charge_states']
    else:
        states_option = 'single'

    if states_option.startswith('hop'):
        charge = int(states_option.split()[1])
        for i in range(len(fragments)):
            fragment_charges = [0]*len(fragments)
            fragment_charges[i] = charge
            charge_states.append(ChargeState(fragments, fragment_charges))
    else:
        fragment_charges = [ sum([geometry[i].formal_chg for i in frag ])
                             for frag in fragments ]
        charge_states = [ChargeState(fragments, fragment_charges)]

    return charge_states

def energy_driver():
    '''SP energy'''

    comm, rank, nproc = inp.MPI_info()

    if inp.VERBOSE and rank == 0:
        print "    Cluster Ion Calculation Input"
        print "Energy Driver running %d processors" % comm.size
        print "-----------------------------------"
        for k in ['hftype', 'correlation', 'basis', 'diagonal', 
                  'relax_neutral_dimers', 'corr_neutral_dimers',
                  'coupling', 'embedding', 'fragmentation', 
                  'charge_states', 'task']:
            print "%22s %20s" % (k, inp.inputdata[k])

        print "\nGeometry / Angstroms"
        print "--------------------"
        print "\n".join(map(str, inp.inputdata['geometry']))

    charge_states = enumerate_states()
    
    N = len(charge_states)
    H = np.zeros((N,N))
    S = np.eye(N)
    
    if rank == 0 and inp.VERBOSE:
        print "\nGenerated %d charge configurations:" % N
        print "\n".join(map(str, charge_states))
        print "\nMonomer SCF & Diagonal Element Calculation"
        print "------------------------------------------"


    # Diagonals: split into N sub-communicators if nproc > N
    if nproc > N:
        pps = nproc // N
        color = rank // pps
        if rank > pps * N - 1:
            color = rank % N

        work_comm = comm.Split(color, rank)
        charge_states[color].monomerSCF(charge_states[color].monomers,
                subcomm=work_comm)
        E = charge_states[color].diagonal(work_comm)

        if work_comm.Get_rank() == 0:
            comm.send(E, dest=0, tag=color)
            comm.send(charge_states[color].monomers, dest=0, tag=color+nproc)
            comm.send(charge_states[color].dimers, dest=0, tag=color+2*nproc)
            comm.send(charge_states[color].E1, dest=0, tag=color+3*nproc)
            comm.send(charge_states[color].E2, dest=0, tag=color+4*nproc)
        if rank == 0:
            for i in range(N):
                H[i,i] = comm.recv(source=ANY_SOURCE, tag=i)
                charge_states[i].monomers = comm.recv(source=ANY_SOURCE, tag=i+nproc)
                charge_states[i].dimers   = comm.recv(source=ANY_SOURCE, tag=i+2*nproc)
                charge_states[i].E1       = comm.recv(source=ANY_SOURCE, tag=i+3*nproc)
                charge_states[i].E2       = comm.recv(source=ANY_SOURCE, tag=i+4*nproc)
        work_comm.Free()
    else:
        for i in range(N):
            charge_states[i].monomerSCF(charge_states[i].monomers, subcomm=comm)
            H[i,i] = charge_states[i].diagonal(comm)

    # log monomer SCF and diagonal elements
    if rank == 0 and inp.VERBOSE:
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

        print "Coupling and Overlap Matrix Elements"
        print "------------------------------------"

    # Couplings/overlaps
    pairs = [(i,j) for i in range(N-1) for j in range(i+1,N)]
    my_pairs = inp.MPI_scatter(comm, pairs, master=0)
    couplings = [0.0] * len(my_pairs)
    overlaps  = [0.0] * len(my_pairs)

    for idx, (i,j) in enumerate(my_pairs):

        couplings[idx], overlaps[idx], info = charge_states[i].coupling(charge_states[j])

        if rank == 0 and nproc == 1 and inp.VERBOSE:
            print "Coupling"
            for k, v in info.items():
                print "  %14s" % k, v
            print ""

    couplings = inp.MPI_gather(comm, couplings, master=0)
    overlaps  = inp.MPI_gather(comm, overlaps,  master=0)

    # Solve secular equation, return results
    if rank == 0:
        for idx, (i,j) in enumerate(pairs):
            H[i,j] = H[j,i] = couplings[idx]
            S[i,j] = S[j,i] =  overlaps[idx]

    if inp.VERBOSE and rank == 0:
        print "Hamiltonian"
        print "-----------"
        print H
        if np.allclose(S, np.eye(N)):
            print "Overlap matrix: identity"
        else:
            print "Overlap matrix"
            print "--------------"
            print S
            
    E_nuclear = geom.nuclear_repulsion_energy(inp.inputdata['geometry'])
    for i in range(N):
        H[i,i] -= E_nuclear
    eigvals, eigvecs = eigh(H, b=S)
    prob0 = eigvecs[:,0]**2
    charge_distro = np.zeros((len(inp.inputdata['geometry'])))
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
    if inp.VERBOSE and rank == 0:
        print "Final Energy Calculation Results"
        for k,v in results.items():
            print k, "\n   ", v
    return results
