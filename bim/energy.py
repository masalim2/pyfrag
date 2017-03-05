from pyfrag.Globals import logger, MPI, params
from pyfrag.Globals import geom, lattice
from pyfrag.Globals import neighbor, coulomb
from pyfrag.backend import backend
from pyfrag.bim.monomerscf import monomerSCF
from master_worker import execute

def run_energy(specifier, espcharges):

    assert len(specifier) in [1, 5, 6]
    if len(specifier) == 1:
        i, = specifier
        fragment = [(i,0,0,0)]
        net_chg = geom.charge(i)
        bqlist = neighbor.bq_lists[i]
    elif len(specifier) == 5:
        i,j,a,b,c = specifier
        fragment = [(i,0,0,0), (j,a,b,c)]
        net_chg = geom.charge(i) + geom.charge(j)
    else:
        i,j,a,b,c,bqij = specifier

    if len(specifier) == 5 or len(specifier) == 6:
        bqi, bqj = neighbor.bq_lists[i], neighbor.bq_lists[j]
        bqj = [(bq[0], bq[1]+a, bq[2]+b, bq[3]+c) for bq in bqj]
        bqlist = list(set(bqi).union(set(bqj)))

    if len(specifier) == 5:
        bqlist.remove( (i,0,0,0) )
        bqlist.remove( (j,a,b,c) )

    if len(specifier) == 6:
        assert bqij in ['QMi_BQj', 'QMj_BQi']
        if bqij == 'QMi_BQj':
            fragment = [(i,0,0,0)]
            net_chg = geom.charge(i)
            bqlist.remove( (i,0,0,0) )
        else:
            fragment = [(j,a,b,c)]
            net_chg = geom.charge(j)
            bqlist.remove( (j,a,b,c) )
    result = backend.run('energy', fragment, net_chg, bqlist, espcharges)
    return result

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

    # build one big list of fragment calcs; execute all in parallel
    specifiers = [(i,) for i in range(len(geom.fragments))]
    for (i,n0,n1,n2), (j,a,b,c) in neighbor.dimer_lists:
        specifiers.append((i,j,a,b,c))
        specifiers.append((i,j,a,b,c,'QMi_BQj'))
        specifiers.append((i,j,a,b,c,'QMj_BQi'))
    calcs = execute(specifiers, run_energy, comm, espcharges)

    if VERB and rank == 0:
        print "Fragment calculations received."
        for calc in specifiers:
            print calc, calcs[calc]
    return {'energy' : 0.0}
