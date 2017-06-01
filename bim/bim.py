'''Binary Interaction Method -- module for embedded-fragment calculations on
weakly-interacting molecular clusters (energy, gradient, hessian)'''

from pyfrag.Globals import logger, MPI, params
from pyfrag.Globals import geom
from pyfrag.Globals import neighbor, coulomb
from pyfrag.backend import backend
from pyfrag.Globals import utility as util
from pyfrag.bim.monomerscf import monomerSCF
from pyfrag.bim import sums
#from pyfrag.utility import mw_execute

def get_task():
    '''map from params.options['task'] --> bim summation type

    Returns:
        task: one of 'energy', 'gradient', 'hessian'

        sum_fxn: the corresponding BIM summation function
    '''
    task = { 'bim_e'   : 'energy',
            'bim_grad' : 'gradient',
            'bim_md'   : 'gradient',
            'bim_hess' : 'hessian'
           }.get(params.options['task'], 'gradient')
    sum_fxn = getattr(sums, '%s_sum' % task)
    return task, sum_fxn

def create_bim_fragment(specifier, espcharges):
    '''Create and dispatch a backend fragment calculation.

    Args:
        specifier: tuple specifying the monomer and cell indices for the
            requested calculation.
        espcharges: embedding field charges
    Returns:
        results: results dict from fragment calculation
    '''
    assert len(specifier) in [1, 5, 6]
    #monomer
    if len(specifier) == 1:
        i, = specifier
        fragment = [(i,0,0,0)]
        net_chg = geom.charge(i)
        bqlist = neighbor.bq_lists[i]
    # dimer
    elif len(specifier) == 5:
        i,j,a,b,c = specifier
        fragment = [(i,0,0,0), (j,a,b,c)]
        net_chg = geom.charge(i) + geom.charge(j)
    # monomer in dimer field
    else:
        i,j,a,b,c,bqij = specifier

    # build dimer field
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
    task, sum_fxn = get_task()
    result = backend.run(task, fragment, net_chg, bqlist, espcharges)
    return result

def kernel(comm=None):
    '''Get fragments, do monomer SCF, and dispatch list of fragment calcs.

    Controlled by setting values in params.options and geom.geometry

    Args:
        comm (optional): pass a subcommunicator (generated with comm.split)
        for nested parallelism
    Returns:
        results: dictionary of fragment sums
    '''

    options = params.options
    if comm:
        rank = comm.Get_rank()
    else:
        comm, rank = MPI.comm, MPI.rank
    try:
        VERB = params.verbose
        QUIET = params.quiet
    except AttributeError:
        params.verbose = False
        params.quiet = False
        VERB = params.verbose
        QUIET = params.quiet

    if 'scrdir' not in params.options:
        scr_top = None
        util.make_scratch_dirs(scr_top)

    if not QUIET and MPI.rank == 0:
        logger.print_parameters()
        logger.print_geometry()

    if options['task'] == 'bim_hess':
        for frag in geom.fragments:
            assert frag == sorted(frag)

    # Perform fragmentation
    geom.perform_fragmentation()
    nfrag = len(geom.fragments)
    if not QUIET and MPI.rank == 0:
        print "Generated %d Fragments" % nfrag
        logger.print_fragment()

    # Get neighbor lists
    neighbor.build_lists()
    if VERB and MPI.rank == 0:
        logger.print_neighbors()


    # Monomer SCF
    if not QUIET and MPI.rank == 0: print "Running monomer SCF..."
    espcharges = monomerSCF(comm)
    if VERB and MPI.rank == 0:
        print "Converged ESP charges"
        logger.print_fragment(esp_charges=espcharges, charges_only=True)

    # Evaluate coulomb corrections
    if not QUIET and MPI.rank == 0: print "Evaluating classical coulomb interactions..."
    coulomb.evaluate_coulomb(espcharges)

    # build one big list of fragment calcs; execute all in parallel
    specifiers = [(i,) for i in range(nfrag)]
    for (i,n0,n1,n2), (j,a,b,c) in neighbor.dimer_lists:
        specifiers.append((i,j,a,b,c))
        specifiers.append((i,j,a,b,c,'QMi_BQj'))
        specifiers.append((i,j,a,b,c,'QMj_BQi'))
    if not QUIET and MPI.rank == 0:
        nmono = nfrag
        ncalc = len(specifiers)
        ndim  = (ncalc - nmono) / 3
        print "Running %d monomers, %d dimers..." % (nmono, ndim)
    calcs = util.mw_execute(specifiers, create_bim_fragment, comm, espcharges)

    if VERB and rank == 0:
        print "Fragment calculations received."
        print '\n'.join(["%s %s" % (k, v) for k,v in calcs.items()])

    if rank == 0:
        if not QUIET: print "Computing Fragment sums"
        task, sum_fxn = get_task()
        result  = sum_fxn(specifiers, calcs)
    else:
        result = {'E': 0.0, 'gradient' : 0.0}
    return result
