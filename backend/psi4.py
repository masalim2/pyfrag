import tempfile
import sys
import subprocess
import numpy as np

from pyfrag.Globals import params, geom

def calculate(inp, calc, save):
    options = params.options
    args = ['psi4', inp, '-o', 'stdout']

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

    if save and options['scrdir'] != options['share_dir']:
        pass # MO vectors on disk? copy to share location.
    return output.split('\n')

def inp(calc, atoms, bqs, charge, noscf=False, guess=None, save=False):

    options = params.options
    nelec = sum(geom.z_map[at.sym] for at in atoms) - charge
    f = tempfile.NamedTemporaryFile(dir=options['scrdir'], delete=False)

    f.write('memory %d mb\n' % options['mem_mb'])
    f.write('molecule {\n')
    f.write(' %d %d\n' % (charge, 1+(nelec%2)))
    geom_str = '\n'.join(map(str, atoms))
    geom_str += "\nsymmetry c1\nno_reorient\nno_com\n"
    f.write(geom_str)
    f.write('}\n\n')

    f.write('bqs = QMMM()\n')
    for bq in bqs:
        f.write('bqs.extern.addCharge(%.8f,%.8f,%.8f,%.8f)\n'
                %(bq[3], bq[0], bq[1], bq[2]))
    f.write("psi4.set_global_option_python('EXTERN', bqs.extern)\n\n")

    f.write("set {\n")
    #f.write(" scf_type df\n")
    f.write(" basis %s\n" % options['basis'])
    #f.write(" freeze_core True\n")
    f.write("}\n")
    
    if options['correlation'] and calc not in ['esp', 'energy_hf']:
        theory = options['correlation']
    else:
        theory = 'hf'
    method_str = theory + '/' + options['basis']

    if calc == 'energy' or calc == 'energy_hf':
        f.write('energy("%s")\n' % theory)
    elif calc == 'esp':
        raise RuntimeError("Not implemented")
    elif calc == 'gradient':
        f.write('grad,wfn=gradient("%s",return_wfn=True)\n' % theory)
        with open('grid.dat', 'w') as grid_fp:
            for bq in bqs:
                grid_fp.write('%16.10f%16.10f%16.10f\n' % (bq[0], bq[1], bq[2]))
        f.write("oeprop(wfn, 'GRID_FIELD')\n")
    elif calc == 'hessian':
        f.write('hessian("%s")\n' % theory)

    f.close()
    return f.name

def parse(data, calc, inp, atoms, bqs, save):
    results = {}
    for n, line in enumerate(data):
        if '@DF-RHF Final Energy' in line:
            results['E_hf'] = float(line.split()[-1])
            continue
        if 'DF-MP2 Energies' in line:
            results['E_tot'] = float(data[n+7].split()[-2])
            if 'E_hf' in results:
                results['E_corr'] = results['E_tot'] - results['E_hf']
            continue
        if 'Total Gradient' in line:
            gradients  = []
            for idx in range(n+3,n+3+len(atoms)):
                grad = map(float, data[idx].split()[1:])
                gradients.append(grad)
            results['gradient'] = np.array(gradients)
            if bqs:
                bq_field = np.loadtxt('grid_field.dat')
                assert bq_field.shape == (len(bqs), 3)
                bqgrad = []
                for chg, field_vec in zip(bqs, bq_field):
                    bqgrad.append(-1.0 * chg[3] * field_vec)
                results['bq_gradient'] = np.array(bqgrad)
            continue
    if 'E_tot' not in results:
        results['E_tot'] = results['E_hf']
    return results
