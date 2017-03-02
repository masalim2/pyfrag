from pyfrag.globals.geom import geometry, fragments
from pyfrag.globals.lattice import lat_vecs

def build_atoms(frags, bq_list, bq_charges):

    atoms = []
    for (i,a,b,c) in frags:
        vec = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]
        atoms.extend(geometry[at].shift(vec) for at in fragments[i])

    bq_field = []
    for (i,a,b,c) in bq_list:
        vec = a*lat_vecs[:,0] + b*lat_vecs[:,1] + c*lat_vecs[:,2]
        for at in fragments[i]:
            bq_pos = geometry[at].shift(vec).pos
            bq_field.extend(np.append(bq_pos, bq_charges[at]))

    return atoms, bq_field

def run(calc, frags, charge, bq_list, bq_charges, 
        noscf=False, guess=None, save=False):
    assert calc in 'esp energy gradient hessian'.split()
    assert isinstance(frags, tuple)
    assert all(isinstance(n, int) for n in frags)
    
    atoms, bq_field = build_atoms(frags, bq_list, bq_charges)
    if noscf and guess is None:
        raise RuntimeError("No SCF useless without input guess")
    inp = nw_inp(calc, atoms, bq_field, charge, noscf, guess, save)

def nw_invecs(guess):
    if guess is None:
        return 'atomic'
    elif isinstance(guess, str):
        return guess
    elif isinstance(guess, list):
        return 'fragment ' + ' '.join(guess)
    else:
        raise RuntimeError("unknown guess type")

def nw_inp(calc, atoms, bqs, charge, noscf=False, guess=None, save=False):
    f = tempfile.NamedTemporaryFile(dir=params.scrdir, prefix=fprefix,
            delete=False)

    f.write('scratch_dir %s\n' % params.scrdir)
    f.write('permanent_dir %s\n' % params.scrdir)
    f.write('start\n\n')

    f.write('charge %s\n' % str(charge))
    f.write('geometry units angstroms noautosym noautoz nocenter\n')
    f.write('\n'.join(map(str, atoms)))
    f.write('\nend\n\n')
    
    f.write('basis spherical noprint\n')
    f.write('  * library %s\n' % params.basis)
    f.write('end\n\n')

    f.write('bq units angstroms\n')
    if 'grad' in calc: f.write(' force\n')
    f.write('\n'.join(4*'%18.8f' % (bq[0], bq[1], bq[2], bq[3]) 
            for bq in bqs))
    f.write('\nend\n\n')
    
    f.write('scf\n')
    f.write('sym off; adapt off\n')
    f.write('%s\n' % params.hftype)
    f.write('nopen %d\n' % (charge%2))
    if self.noscf: f.write('noscf\n')
    
    invec = nw_invecs(guess)
    outvec = os.path.join(params.scrdir, os.path.basename(f.name))+".movecs"
    vec_string = 'vectors input %s output %s' % (invec, outvec)
    f.write('%s\n' % ' \\\n'.join(textwrap.wrap(vec_string, width=90,
        break_long_words=False)))
    f.write('end\n\n')

    if params.correlation:
        theory = params.correlation
        if theory == 'mp2':
            f.write('mp2\n freeze atomic\nend\ntask mp2 energy\n\n')
        elif theory == 'ccsd':
            f.write('ccsd\n freeze atomic\nend\ntask ccsd energy\n\n')
        else:
            raise RuntimeError("please write NW wrapper for %s" % theory)
    else:
        theory = 'scf'

    if calc == 'energy_hf':
        f.write('task scf energy\n\n')
    elif calc == 'energy' 
        f.write('task %s energy\n\n' % theory)
    elif calc == 'esp':
        f.write('task scf energy\n\n')
        f.write('esp\n recalculate\nend\n')
        f.write('task esp\n\n')
    elif calc == 'gradient':
        f.write('task %s gradient\n\n' % theory)
    elif calc == 'hessian':
        f.write('task %s hessian\n\n' % theory)
    else:
        raise RuntimeError('Please write wrapper for %s' % calc)
    f.close()
    return f.name
