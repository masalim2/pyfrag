from pyfrag.Globals import MPI
from pyfrag.Globals import params
from pyfrag.Globals import geom

def monomerSCF(monomers, charges=None, bq_lists=None, embedding=None, comm=None):
    '''Cycle embedded monomer calculations until the ESP charges converge.'''
    RMSD_TOL = 0.001
    MAXITER  = 10 

    if comm:
        rank, nproc = comm.Get_rank(), comm.size
    else:
        comm, rank, nproc = MPI.comm, MPI.rank, MPI.comm.size
    if charges is None:
        charges = [geom.charge(m) for m in monomers]
    if bq_lists is None:
        bq_lists = [[(i,0,0,0) for i in monomers if i != m] for m in monomers]
    if embedding is None:
        embedding = params.embedding

    atoms = [at for monomer in monomers for at in geom.fragments[monomer]]
    charges = np.zeros(len(atoms))

    RMSD = RMSD_TOL + 1
    itr = 0

    while RMSD > RMSD_TOL and itr < MAXITER:
        charges_old = np.copy(charges)
        my_monomers = MPI.scatter(comm, monomers, master=0)

        for mono in my_monomers:
            if embedding:
            else:
                mono.embedding_field = []
            in_vecs = None if itr==0 else mono.movecs_path
            mono.run(calc='esp', save_vecs=True, input_vecs=in_vecs)

        new_monomers = MPI.allgather(subcomm, my_monomers)
        for i, mono in enumerate(new_monomers):
            monomers[i] = mono # modify the list passed in originally

        charges_new = [chg for m in monomers for chg in m.esp_charges]

        residual = np.array(charges_new) - np.array(charges_old)
        RMSD = np.linalg.norm(residual)
        itr += 1
        if not embedding:
            return
    if embedding and RMSD > RMSD_TOL:
        raise RuntimeError("Monomer SCF did not converge")
