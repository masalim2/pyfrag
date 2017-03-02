import tempfile
import textwrap
import subprocess
from shutil import copyfile
from ..globals import params
import numpy as np
import sys

def calculate(inp, calc, save):
    '''Run nwchem on input, return raw output'''
    args = ['nwchem.x', inp.name]
    try:
        output = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print "-------------------------"
        print "INPUT FILE OF FAILED CALC"
        print "-------------------------"
        print open(inp).read()
        print "--------------------------"
        print "OUTPUT FILE OF FAILED CALC"
        print "--------------------------"
        print e.output
        sys.exit(1)
    if save and params.scrdir != params.share_dir:
        outvec = os.path.basename(inp.name)+".movecs"
        source = os.path.join(params.scrdir, outvec)
        destin = os.path.join(params.share_dir, outvec)
        copyfile(source, destin)
    return output.split('\n')

def invecs(guess):
    '''Create initial guess string for NWchem scf input'''
    if guess is None:
        return 'atomic'
    elif isinstance(guess, str):
        return guess
    elif isinstance(guess, list):
        return 'fragment ' + ' '.join(guess)
    else:
        raise RuntimeError("unknown guess type")

def inp(calc, atoms, bqs, charge, noscf=False, guess=None, save=False):
    '''Write NWchem input file to temp file. Return filename.'''
    f = tempfile.NamedTemporaryFile(dir=params.scrdir, prefix=fprefix,
            delete=False)

    f.write('scratch_dir %s\n' % params.scrdir)
    f.write('permanent_dir %s\n' % params.scrdir)
    f.write('memory total %d mb\n' % params.mem_mb)
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
        f.write('task %s hessian numerical\n\n' % theory)
    else:
        raise RuntimeError('Please write wrapper for %s' % calc)
    f.close()
    return f.name

def parse(data, calc, inp, atoms, bqs, save):
    '''Parse raw NWchem output.'''
    results = {}
    for n, line in enumerate(data):

        if "Total SCF energy" in line:
            results['E_hf'] = float(line.split()[-1])
            continue

        if "Total MP2" in line or "Total CCSD" in line:
            results['E_tot'] = float(line.split()[-1])
            results['E_corr'] = results['E_tot'] - results['E_hf']
            continue

        if 'ESP' in line:
            esp_charges = []
            for idx in range(n+3, n+3+len(atoms)):
                esp_charges.append(float(data[idx].split()[-1]))
            results['esp_charges'] = esp_charges
            continue

        if 'Nuclear repulsion energy' in line:
            results['E_nuc'] = float(line.split()[-1])
            continue

        if 'Mulliken analysis of the total density' in line:
            mulliken_charges = []
            for idx in range(n+5, n+5+len(atoms)):
                mulliken_charges.append(float(data[idx].split()[3]))
            results['mulliken_charges'] = mulliken_charges
            continue

        if 'ENERGY GRADIENTS' in line:
            gradients = []
            for idx in range(n+4:n+4+len(atoms)):
                grad = map(float, data[idx].split()[-3:])
                gradients.append(grad)
            results['gradient'] = gradients
            if bqs:
                bq_gradients = []
                bqforce_name, ext = os.path.splitext(inp.name)
                bqforce_name += '.bqforce.dat'
                results['bq_gradient'] = np.loadtxt(bqforce_name)
            continue

    if calc == 'hessian':
        basename, ext = os.path.splitext(inp.name)
        hess_name    = basename + ".hess"
        ddipole_name = basename + ".fd_ddipole"
        hess_tri = np.loadtxt(hess_name)
        ddipole = np.loadtxt(ddipole_name)
        natm = len(atoms)
        assert len(hess_tri) == 3*natm + 3*natm*(3*natm-1)/2
        assert len(ddipole) == 9*natm
        results['hess_tri'] = hess_tri
        results['ddipole'] = ddipole
                
    return results
