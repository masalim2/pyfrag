import numpy as np

from pyfrag.Globals import params, MPI, geom
from pyfrag.backend import backend
def fullsys_best_guess(comm=None):
    if comm is None:
        comm = MPI.comm

    geom.set_frag_auto()
    sub_fragments = geom.fragments[:]
    guess_vecs = []

    esp_list = []
    for i, frag_i in enumerate(sub_fragments):
        monomers = range(len(sub_fragments))
        charges = [int(j==i) for j in monomers]
        esp, vecs = monomerSCF(monomers,charges,embedding=False,comm=comm)
        esp_list.append(esp)
        guess_vecs.append(vecs)

    geom.set_frag_full_system()

    my_calcs = []
    my_guessvecs = MPI.scatter(comm, guess_vecs, master=0)

    for guess in my_guessvecs:
        res = backend.run('esp', [(0,0,0,0)], 1, [], [], guess=guess)
        my_calcs.append(res)

    calcs = MPI.allgather(comm, my_calcs)
    ibest, best = min(enumerate(calcs), key=lambda x:x[1]['E_hf'])
    espfield = calcs[ibest]['esp_charges']

    if params.options['correlation']:
        best = backend.run('energy', [(0,0,0,0)], 1, [], [], guess=guess_vecs[ibest])
    res = {}
    res['E1'] = best['E_tot']
    res['E2'] = 0.0
    res['monomers'] = [best]
    res['dimers'] = []
    res['net_charges'] = [1]
    res['esp'] = espfield
    return res

def monomerSCF(monomers, net_charges, embedding=None, comm=None):
    '''Cycle embedded monomer calculations until the ESP charges converge.

    VBCT version: specify N monomers and their net charges
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
            If string 'serial' is specified, bypass MPI communication.
            Default None: use the top-level communicator in Globals.MPI
    Returns
        espcharges: a list of esp-fit atom-centered charges
        movecs: a list of MO vectors for each monomer
    '''
    if comm is None:
        comm = MPI.comm

    RMSD_TOL = 0.001
    MAXITER  = 10
    RMSD = RMSD_TOL + 1
    itr = 0

    if embedding is None:
        embedding = params.options['embedding']
    else:
        assert type(embedding) is bool

    espcharges = [0.0 for at in geom.geometry]

    while RMSD > RMSD_TOL and itr < MAXITER:

        espcharges0 = espcharges[:] # copy
        if comm is not 'serial':
            myfrags = MPI.scatter(comm, zip(monomers, net_charges), master=0)
        else:
            myfrags = zip(monomers, net_charges)
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

        if comm is not 'serial':
            movecs = MPI.allgather(comm, myvecs)
            monomer_espcharges = MPI.allgather(comm, mycharges)
        else:
            movecs = myvecs
            monomer_espcharges = mycharges
        for (m, charges) in zip(monomers, monomer_espcharges):
            for (at, chg) in zip(geom.fragments[m], charges):
                espcharges[at] = chg

        residual = np.array(espcharges) - np.array(espcharges0)
        RMSD = np.linalg.norm(residual)
        itr += 1
        if not embedding or len(monomers) == 1:
            return espcharges, movecs

    if RMSD > RMSD_TOL:
        raise RuntimeError("Monomer SCF did not converge")
    else:
        return espcharges, movecs
