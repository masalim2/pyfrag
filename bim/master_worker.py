from pyfrag.Globals import MPI

def execute(specifiers, run_calc, comm=None, *args):
    '''Run a series of calculations in master-worker mode.

    Args:
        specifiers: a list of tuples which uniquely determine a
            fragment calculation.
        run_calc: the function which accepts a specifier tuple,
            sets up a QM calculation, invokes the backend, and
            returns the relevant calculation results.
        comm: optional subcommunicator, default None: use Globals.MPI
            communicator
    Returns:
        calcs: the dictionary of results indexed by specifiers
    '''
    if comm:
        rank, nproc = comm.Get_rank(), comm.size
    else:
        comm, rank, nproc = MPI.comm, MPI.rank, MPI.comm.size
    calcs = {}
    stat = MPI.MPI.Status()
    if nproc == 1:
        for calc in specifiers:
            calcs[calc] = run_calc(calc, *args)
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
            result = run_calc(calc, *args)
        else:
            calc, result = 'dummy', 'dummy'
        comm.send((calc,result), dest=0)
        idx = comm.recv(source=0)
        while idx != 'DONE':
            calc = specifiers[idx]
            result = run_calc(calc, *args)
            comm.send((calc,result), dest=0)
            idx = comm.recv(source=0)
    return calcs
