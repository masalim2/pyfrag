'''NWChem Backend'''
import tempfile
import textwrap
import subprocess
from shutil import copyfile
import numpy as np
import sys
import os

from pyfrag.Globals import params, geom

def calculate(inp, calc, save):
    '''Run nwchem on input, return raw output

    Args
        inp: NWChem input object (input file path)
        calc: calculation type
        save: save calculation results
    Returns
        output_lines: nwchem stdout lines
    '''
    options = params.options
    args = ['nwchem.x', inp]
    try:
        output = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        info =  '-------------------------\n'
        info += 'INPUT FILE OF FAILED CALC\n'
        info += "-------------------------\n"
        info += open(inp).read()+'\n'
        info += "--------------------------\n"
        info += "OUTPUT FILE OF FAILED CALC\n"
        info += "--------------------------\n"
        info += e.output + '\n'
        raise RuntimeError(info)
    if save and options['scrdir'] != options['share_dir']:
        outvec = os.path.basename(inp)+".movecs"
        source = os.path.join(options['scrdir'], outvec)
        destin = os.path.join(options['share_dir'], outvec)
        copyfile(source, destin)
        output += "\nmovec_shared_path %s\n" % destin
    return output.split('\n')

def invecs(guess):
    '''Create initial guess string for NWchem scf input

    Args
        guess: string or list of strings for fragment guess
    '''
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
    options = params.options
    nelec = sum(geom.z_map[at.sym] for at in atoms) - charge
    f = tempfile.NamedTemporaryFile(dir=options['scrdir'], delete=False)

    f.write('scratch_dir %s\n' % options['scrdir'])
    f.write('permanent_dir %s\n' % options['scrdir'])
    f.write('memory total %d mb\n' % options['mem_mb'])
    f.write('start\n\n')

    f.write('charge %s\n' % str(charge))
    f.write('geometry units angstroms noautosym noautoz nocenter\n')
    f.write('\n'.join(map(str, atoms)))
    f.write('\nend\n\n')

    f.write('basis spherical noprint\n')
    f.write('  * library %s\n' % options['basis'])
    f.write('end\n\n')

    f.write('bq units angstroms\n')
    if 'grad' in calc: f.write(' force\n')
    f.write('\n'.join(((4*'%18.8f') % (bq[0], bq[1], bq[2], bq[3])) for bq in bqs))
    f.write('\nend\n\n')

    f.write('scf\n')
    f.write('sym off; adapt off\n')
    f.write('%s\n' % options['hftype'])
    f.write('nopen %d\n' % (nelec%2))
    if nelec%2 == 1:
        f.write('maxiter 120\n')
        f.write('thresh 1.0e-4\n')
    if noscf: f.write('noscf\n')

    invec = invecs(guess)
    outvec = os.path.join(options['scrdir'], os.path.basename(f.name))+".movecs"
    vec_string = 'vectors input %s output %s' % (invec, outvec)
    f.write('%s\n' % ' \\\n'.join(textwrap.wrap(vec_string, width=90,
        break_long_words=False)))
    f.write('end\n\n')

    if options['correlation'] and calc not in ['esp', 'energy_hf']:
        theory = options['correlation']
        if theory == 'mp2':
            f.write('mp2\n freeze atomic\nend\n\n')
            #if nelec%2 == 0:
            #    f.write('mp2\n freeze atomic\nend\n\n')
            #else:
            #    f.write('tce\n scf\n mp2\n freeze atomic\nend\n')
            #    theory = 'tce'
        elif theory == 'ccsd':
            f.write('ccsd\n freeze atomic\nend\n\n')
        else:
            raise RuntimeError("please write NW wrapper for %s" % theory)
    else:
        theory = 'scf'

    if calc == 'energy_hf':
        f.write('task scf energy\n\n')
    elif calc == 'energy':
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
    options = params.options
    for n, line in enumerate(data):

        if "Total SCF energy" in line:
            results['E_hf'] = float(line.split()[-1])
            results['E_tot'] = results['E_hf']
            continue

        if "Total MP2" in line or "Total CCSD" in line:
            results['E_tot'] = float(line.split()[-1])
            results['E_corr'] = results['E_tot'] - results['E_hf']
            continue

        if "MBPT(2) total energy" in line:
            results['E_tot'] = float(line.split()[-1])
            results['E_corr'] = results['E_tot'] - results['E_hf']

        if 'ESP' in line.split() and calc == 'esp':
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
            idx = n+5
            while len(mulliken_charges) < len(atoms) and idx < len(data):
                record = data[idx].split()
                if len(record) >= 4:
                    try:
                        chg = float(record[3])
                        mulliken_charges.append(chg)
                    except ValueError:
                        pass
                    idx += 1
            results['mulliken_chg'] = mulliken_charges
            continue

        if 'ENERGY GRADIENTS' in line:
            gradients = []
            for idx in range(n+4,n+4+len(atoms)):
                grad = map(float, data[idx].split()[-3:])
                gradients.append(grad)
            results['gradient'] = np.array(gradients)
            if bqs and 'grad' in options['task']:
                bqforce_name, ext = os.path.splitext(inp)
                bqforce_name += '.bqforce.dat'
                results['bq_gradient'] = np.loadtxt(bqforce_name)
            continue
        if "movec_shared_path" in line:
            results['movecs'] = line.split()[-1]

    if calc == 'hessian':
        basename, ext = os.path.splitext(inp)
        hess_name    = basename + ".hess"
        ddipole_name = basename + ".fd_ddipole"
        hess_txt = open(hess_name).read().replace('D', 'E')
        hess_tri = np.fromstring(hess_txt, sep='\n')

        ddipole_txt = open(ddipole_name).read().replace('D','E')
        ddipole = np.fromstring(ddipole_txt, sep='\n')
        natm = len(atoms)
        assert len(hess_tri) == 3*natm + 3*natm*(3*natm-1)/2
        assert len(ddipole) == 9*natm
        results['hess_tri'] = hess_tri
        results['ddipole'] = ddipole

    return results
