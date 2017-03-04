from pyfrag.Globals import params
def monomerSCF(monomers, net_charges, embedding=None, comm=None):
    '''Cycle embedded monomer calculations until the ESP charges converge.

    VBCT version: specify a subset of monomers and their net charges 
    explicitly.  No support for cutoffs/periodicity: every monomer is
    embedded in the field of all other N-1 monomers. Able to override default
    embedding option. Monomer MO vectors are saved; thus one cycle runs even
    if embedding option is turned off.
    
    Args
        monomers: a list of monomer indices
        net_charges: net charge of each monomer
        embedding: Override True/False specified in input. 
            Default None: use the value specified in input.
        comm: specify a sub-communicator for parallel execution.
            Default None: use the top-level communicator in Globals.MPI
    Returns
        espcharges: a list of esp-fit atom-centered charges
    '''
    if comm:
        rank, nproc = comm.Get_rank(), comm.size
    else:
        comm, rank, nproc = MPI.comm, MPI.rank, MPI.comm.size

    RMSD_TOL = 0.001
    MAXITER  = 10 
    RMSD = RMSD_TOL + 1
    itr = 0

    if embedding is None: embedding = params.options['embedding']
    espcharges = [0.0 for i in range(len(geom.geometry))]

    while RMSD > RMSD_TOL and itr < MAXITER:

        espcharges0 = espcharges[:] # copy
        myfrags = MPI.scatter(comm, zip(monomers, net_charges), master=0)
        mycharges = []
        myvecs = []
        
        for (m, net_chg) in myfrags:
            fragment = [(m,0,0,0)]
            if embedding:
                bqlist = [(j,0,0,0) for j in monomers if j != m]
            else:
                bqlist = []
            result = backend.run('esp', fragment, net_chg, bqlist, 
                                 espcharges, save=True)
            mycharges.append(result['esp_charges'])
            myvecs.append(result['movecs'])

        movecs = MPI.allgather(comm, myvecs)
        monomer_espcharges = MPI.allgather(comm, mycharges)
        for (m, charges) in zip(monomers, monomer_espcharges):
            for (at, chg) in zip(geom.fragments[m], charges):
                espcharges[at] = chg

        residual = np.array(espcharges) - np.array(espcharges0)
        RMSD = np.linalg.norm(residual)
        itr += 1
        if not embedding: 
            return espcharges, movecs

    if RMSD > RMSD_TOL:
        raise RuntimeError("Monomer SCF did not converge")
    else:
        return espcharges, movecs
