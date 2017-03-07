try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.size
except ImportError:
    print "# Could not import mpi4py. Running serial."
    MPI = None
    comm = None
    rank = 0
    nproc = 1

def bcast(data, master=0):
    if comm is None:
        return data
    return comm.bcast(data, root=master)

def scatter(comm, data, master=0):
    '''default scatter fxn is dumb; can only handle one item per rank.  This
    will split a list evenly among the ranks by creating a list of lists.
    mydata = MPI_scatter(comm, data, 0)'''
    if comm is None:
        return data
    rank = comm.Get_rank()
    N = len(data)
    N_per_proc = N // nproc
    rem = N % nproc

    my_data = []
    if rank == master:
        for r in range(nproc):
            if r < rem:
                start = r*(N_per_proc+1)
                end = start + N_per_proc + 1
            else:
                start = rem + r*N_per_proc
                end = start + N_per_proc
            my_data.append(data[start:end])

    my_data = comm.scatter(my_data, root=master)
    return my_data

def gather(comm, data, master=0):
    '''default gather fxn is dumb...
    data = MPI_gather(comm, data, 0)'''
    if comm is None:
        return data
    rank = comm.Get_rank()

    my_data = comm.gather(data, root=master)
    if rank == master:
        my_data = [item for sublist in my_data for item in sublist]

    return my_data

def allgather(comm, data):
    '''default allgather fxn is dumb...
    data = MPI_allgather(comm, mydata)'''
    if comm is None:
        return data
    rank = comm.Get_rank()

    my_data = comm.allgather(data)
    my_data = [item for sublist in my_data for item in sublist]

    return my_data

def allreduce(comm, data):
    if comm is None:
        return data
    return comm.allreduce(data)
