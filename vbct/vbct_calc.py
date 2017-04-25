from pyfrag.Globals import MPI, geom, params
from pyfrag.backend import backend
from pyfrag.vbct.monomerscf import monomerSCF
from itertools import combinations

def make_embed_list(qm_fragment, all_monomers):
    '''Generate embedding field

    Args
        qm_fragment: index of monomer or dimer (no PBC)
    Returns
        bq_list: list of fragments in bq field of qm_fragment
    '''
    if type(qm_fragment) is int:
        qm_fragment = (qm_fragment,)
    field = [m for m in all_monomers if m not in qm_fragment]
    return [(m,0,0,0) for m in field]

def diag_chglocal(charges, espfield, movecs, comm=None):
    '''Charge-local dimer method for diagonal element calculation.

    Only works with NW backend and singly ionized molecular cluster cations.

    Args
        charges: list of net charges on each fragment
        espfield: list of esp-fit atomic charges for entire system
        movecs: list of MO coeff files for each fragment
        comm: MPI communicator or subcommunicator
    '''
    if comm is None:
        comm = MPI.comm

    embed_flag = params.options['embedding']

    nfrag = len(geom.fragments)
    monomers = range(nfrag)
    dimers = list(combinations(monomers, 2))

    my_mon_idxs   = MPI.scatter(comm, monomers, master=0)
    my_dim_idxs   = MPI.scatter(comm,   dimers, master=0)
    my_mon_calcs  = []
    my_dim_calcs  = []

    for m in my_mon_idxs:
        bqlist = []
        if embed_flag:
            bqlist = make_embed_list(m, monomers)
        frag = [(m,0,0,0)]
        res = backend.run('energy', frag, charges[m], bqlist, espfield,
                          guess=movecs[m])
        my_mon_calcs.append(res)

    for d in my_dim_idxs:
        i,j = d
        chg_i, chg_j = charges[i], charges[j]
        chg = charges[i]+charges[j]

        frag = [(d[0],0,0,0), (d[1],0,0,0)]
        guess = [movecs[i], movecs[j]]

        bqlist = []
        if embed_flag:
            bqlist = make_embed_list(d, monomers)

        if chg_i != chg_j:
            assert chg == 1
            charge_local = True
            calc = 'energy_hf'
        else:
            assert chg == 0
            charge_local = False
            calc = 'energy'

        res = backend.run(calc, frag, chg, bqlist, espfield, guess=guess,
                          noscf=charge_local)
        my_dim_calcs.append(res)

    mon_calcs = MPI.allgather(comm, my_mon_calcs)
    dim_calcs = MPI.allgather(comm, my_dim_calcs)
    results = {}

    E1 = sum([mon['E_tot'] for mon in mon_calcs])
    results['E1'] = E1
    E2 = 0.0
    for (i,j), dim in zip(dimers, dim_calcs):
        Eij = dim['E_tot']
        if 'E_corr' in dim:
            Ei = mon_calcs[i]['E_tot']
            Ej = mon_calcs[j]['E_tot']
        else:
            Ei = mon_calcs[i]['E_hf']
            Ej = mon_calcs[j]['E_hf']
        E2 += Eij - Ei - Ej
    results['E2'] = E2
    results['monomers'] = mon_calcs
    results['dimers'] = dim_calcs
    results['net_charges'] = charges
    results['esp'] = espfield

    return results

def coupl_chglocal(A, B):
    '''Charge-local dimer method for coupling

    Only works with NW backend and singly ionized molecular cluster cations.

    Args
        A, B: indices for off-diagonal H element calculation
    '''

    esps_Aloc, vecs_Aloc = monomerSCF([A,B], [1,0], comm='serial')
    esps_Bloc, vecs_Bloc = monomerSCF([A,B], [0,1], comm='serial')
    bqs = []
    esp = []

    # Charge local
    frag = [(A,0,0,0), (B,0,0,0)]
    Aloc = backend.run('energy_hf', frag, 1, bqs, esp, noscf=True,
                       guess=vecs_Aloc)
    Bloc = backend.run('energy_hf', frag, 1, bqs, esp, noscf=True,
                       guess=vecs_Bloc)

    # Relaxed dimer
    Aloc_relax = backend.run('energy', frag, 1, bqs, esp, guess=vecs_Aloc)
    Bloc_relax = backend.run('energy', frag, 1, bqs, esp, guess=vecs_Bloc)
    relax = min(Aloc_relax, Bloc_relax, key=lambda x:x['E_tot'])

    E_relax = relax['E_tot']
    E_Aloc  = Aloc['E_tot']
    E_Bloc  = Bloc['E_tot']

    if params.options['correlation']:
        monA_0 = backend.run('energy', [(A,0,0,0)], 0, [(B,0,0,0)], esps_Bloc,
                       guess=vecs_Bloc[0])
        monB_1 = backend.run('energy', [(B,0,0,0)], 1, [(A,0,0,0)], esps_Bloc,
                       guess=vecs_Bloc[1])
        monA_1 = backend.run('energy', [(A,0,0,0)], 1, [(B,0,0,0)], esps_Aloc,
                       guess=vecs_Aloc[0])
        monB_0 = backend.run('energy', [(B,0,0,0)], 0, [(A,0,0,0)], esps_Aloc,
                       guess=vecs_Aloc[1])
        E_Aloc += monA_1['E_corr'] + monB_0['E_corr']
        E_Bloc += monA_0['E_corr'] + monB_1['E_corr']
    assert E_relax <= E_Aloc and E_relax <= E_Bloc
    coupling = -1.0*((E_relax-E_Aloc)*(E_relax-E_Bloc))**0.5

    results = dict(idx=(A,B),coupling=coupling, AB=E_relax, A=E_Aloc, B=E_Bloc)
    return results
