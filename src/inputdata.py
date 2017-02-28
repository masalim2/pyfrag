# Globally-shared data
inputdata = {}
comm_world = None
world_rank = 0
VERBOSE = False

def MPI_info():
    '''return communicator, rank, and nproc'''
    return (comm_world, world_rank, comm_world.size) if comm_world else (None,0,1)

def MPI_scatter(comm, data, master=0):
    '''default scatter fxn is dumb; can only handle one item per rank.  This
    will split a list evenly among the ranks by creating a list of lists.
    mydata = MPI_scatter(comm, data, 0)'''
    if comm is None:
        return data
    rank = comm.Get_rank()
    nproc = comm.size
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

def MPI_gather(comm, data, master=0):
    '''default gather fxn is dumb...
    data = MPI_gather(comm, data, 0)'''
    if comm is None:
        return data
    rank = comm.Get_rank()
    nproc = comm.size

    my_data = comm.gather(data, root=master)
    if rank == master:
        my_data = [item for sublist in my_data for item in sublist]

    return my_data

def MPI_allgather(comm, data):
    '''default allgather fxn is dumb...
    data = MPI_allgather(comm, mydata)'''
    if comm is None:
        return data
    rank = comm.Get_rank()
    nproc = comm.size

    my_data = comm.allgather(data)
    my_data = [item for sublist in my_data for item in sublist]

    return my_data


def parse(inFile):
    '''Crude input file parser. 
    Returns dictionary of calculation attributes'''
    defaults     = { 'geometry'             : [],
                     'basis'                : 'sto-3g',
                     'hftype'               : 'rohf',
                     'correlation'          : '',
                     'embedding'            : True,
                     'diagonal'             : 'chargelocal_dimers',
                     'relax_neutral_dimers' : True,
                     'corr_neutral_dimers'  : True,
                     'coupling'             : 'dimer_gs',
                     'backend'              : 'nw',
                     'task'                 : 'energy'
                   }

    inputLines = inFile.readlines()
    nlines = len(inputLines)
    n = 0
    while n < nlines:
        n2 = n + 1
        line = inputLines[n].split('#')[0]

        # single-line (key) = (value) entries
        entry = [s.strip().lower() for s in line.split('=')]
        if len(entry) == 2 and '' not in entry:
            key, value = entry
            inputdata[key] = value

        # multi-line entry, enclosed in { }
        elif '{' in line:
            key = line.split('{')[0].strip().lower()
            if key:
                inputdata[key] = []
                while n2 < nlines:
                    closer = False
                    line2 = inputLines[n2].split('#')[0]
                    if '}' in line2:
                        closer = True
                    line2 = line2.split('}')[0].strip().lower()
                    if line2:
                        inputdata[key].append(line2)
                    n2 += 1
                    if closer:
                        break
        n = n2
    
    for option, default in defaults.items():
        if option not in inputdata:
            inputdata[option] = default

    for option, value in inputdata.items():
        if type(value) == str:
            if value == 'false' or value == 'no' or value == 'off':
                inputdata[option] = False
            if value == 'yes' or value == 'on' or value == 'true':
                inputdata[option] = True
