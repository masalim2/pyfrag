from pyfrag.Globals import logger, MPI, params
from pyfrag.Globals import geom, lattice
from pyfrag.Globals import neighbor, coulomb
from pyfrag.bim.monomerscf import monomerSCF

def run_calc(specifier):
    return -3.0

def kernel(comm=None):
    options = params.options
    if comm:
        rank, nproc = comm.Get_rank(), comm.size
    else:
        comm, rank, nproc = MPI.comm, MPI.rank, MPI.comm.size
    VERB = params.verbose
    QUIET = params.quiet

    if not QUIET and MPI.rank == 0: 
        logger.print_parameters()
        logger.print_geometry()
    
    # Perform fragmentation
    if options['fragmentation'] == 'auto':
        geom.set_frag_auto()
    elif options['fragmentation'] == "full_system":
        geom.set_frag_full_system()
    else:
        geom.set_frag_manual()
    if not QUIET and MPI.rank == 0: 
        print "Generated %d Fragments" % len(geom.fragments)
        logger.print_fragment()

    # Get neighbor lists
    neighbor.build_lists()
    if VERB and MPI.rank == 0: 
        logger.print_neighbors()

    # Monomer SCF
    if not QUIET and MPI.rank == 0: print "Monomer SCF"
    espcharges = monomerSCF(comm)
    if VERB and MPI.rank == 0:
        print "Converged ESP charges"
        logger.print_fragment(esp_charges=espcharges, charges_only=True)

    # build one big list of fragment calcs; do them all in master-slave
    # load-balanced fashion
    specifiers = [(i,) for i in range(len(geom.fragments))]
    for (i,n0,n1,n2), (j,a,b,c) in neighbor.dimer_lists:
        specifiers.append((i,j,a,b,c))
        specifiers.append((i,j,a,b,c,'bqj'))
        specifiers.append((i,j,a,b,c,'bqi'))

    calcs = {}
    stat = MPI.MPI.Status()
    if nproc == 1:
        for calc in specifiers:
            calcs[calc] = run_calc(calc)
    elif rank == 0:
        for idx in range(nproc-1, len(specifiers)):
            (calc,result) = comm.recv(source=MPI.MPI.ANY_SOURCE, status=stat)
            calcs[calc] = result
            comm.send(idx, dest=stat.Get_source())
        for idx in range(1, nproc):
            (calc,result) = comm.recv(source=MPI.MPI.ANY_SOURCE, status=stat)
            calcs[calc] = result
            comm.send('DONE', dest=stat.Get_source())
    else:
        idx = rank-1
        if idx < len(specifiers):
            calc = specifiers[idx]
            result = run_calc(calc)
        else:
            calc, result = 'dummy', 'dummy'
        comm.send((calc,result), dest=0)
        idx = comm.recv(source=0)
        while idx != 'DONE':
            calc = specifiers[idx]
            result = run_calc(calc)
            comm.send((calc,result), dest=0)
            idx = comm.recv(source=0)
    if VERB and rank == 0:
        print "Fragment calculations received."
        for calc in specifiers:
            print calc, calcs[calc]
    return {'energy' : 0.0}
