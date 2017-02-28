import numpy as np
import geometry as geom
import sys
import subprocess
import tempfile
import inputdata as inp
import os
import textwrap
from shutil import copy2


class Fragment:

    def __init__(self, atoms=[], charge=None, index=None):
        self.atoms = atoms
        if charge is not None: 
            self.charge = charge
        else:
            self.charge = sum([at.formal_chg for at in self.atoms])
        self.index = index
        
        self.esp_charges = np.zeros((len(self.atoms)))
        self.embedding_field = []
        self.energy = {}
        self.gradient = {}
        self.hessian = {}
        self.dipole = {}
        self.ddipole = {}

        self.args = None

    def __len__(self): return len(self.atoms)

    def nelec(self): 
        return sum([geom.z_map[at.sym] for at in self.atoms]) - self.charge

    def spinz(self): return self.nelec() % 2

    def centerofmass(self):
        totalmass = sum([at.mass for at in self.atoms])
        return sum([geom.mass_map[at.sym]*at.pos 
                    for at in self.atoms])/totalmass

    def centerofcharge(self):
        totalchg = sum([geom.z_map[at.sym] for at in self.atoms])
        return sum([geom.z_map[at.sym]*at.pos 
                    for at in self.atoms])/totalchg
    
    def __repr__(self): return "\n".join(map(str, self.atoms))

    def label(self):
        atoms = [atom.sym for atom in self.atoms]
        atom_counts = { sym : atoms.count(sym) for sym in set(atoms) }
        label = '('
        for sym, count in sorted(atom_counts.items()):
            label += sym.capitalize()
            if count > 1: label += str(count)
        label += ')'
        if self.charge != 0: label += "%+d" % self.charge
        return label

    def __add__(self, frag2):
        '''+ operator to combine two monomers into a dimer'''
        if self.index is not None and frag2.index is not None:

            if isinstance(self.index, tuple) or isinstance(frag2.index, tuple):
                raise ValueError("Fragment addition only defined on two monomers")

            if self.index < frag2.index:
                atoms = self.atoms + frag2.atoms
            else:
                atoms = frag2.atoms + self.atoms
                raise RuntimeError("Possible bug: swapping vecs")
            
            index = tuple(sorted([self.index, frag2.index]))

        else:
            index = None
            atoms = self.atoms + frag2.atoms
        return self.__class__(atoms, self.charge + frag2.charge, index)

    def __eq__(self, frag2):
        '''Use fast index comparison if fragments are indexed
        Otherwise, two fragments are equal if they contain the same
        atoms (no test for charge)'''
        if self.index is not None and frag2.index is not None:
            return self.index == frag2.index
        if len(self.atoms) != len(frag2.atoms):
            return False
        for at, at2 in zip(self.atoms, frag2.atoms):
            if at != at2:
                return False
        return True

    def __ne__(self, frag2):
        '''Weird: this must be defined in addition to __eq__'''
        return not self.__eq__(frag2)

    def run(self, **kwargs):

        self.input_vecs = kwargs['input_vecs'] if 'input_vecs' in kwargs else None
        self.noscf = kwargs['noscf'] if 'noscf' in kwargs else False
        self.save_vecs = kwargs['save_vecs'] if 'save_vecs' in kwargs else False
        self.calc = kwargs['calc'] if 'calc' in kwargs else 'energy'
        self.save_ints = kwargs['integrals'] if 'integrals' in kwargs else []

        if self.noscf and not self.input_vecs:
            raise RuntimeError("No SCF makes no sense without input fragment orbitals")
        
        if not self.input_vecs:
            assert self.noscf == False
            sub_fragments = geom.makefrag_auto(self.atoms)
            if self.charge == 0 or len(sub_fragments) == 1:
                self.input_vecs = 'default'
            else:
                # If this isn't true, you're going to swap movecs incorrectly:
                assert [i for frag in sub_fragments for i in frag] == range(len(self.atoms))
                
                self.input_vecs = self.get_best_guess()
                if self.calc == 'hf_energy':
                    return
                if self.calc == 'energy' and not inp.inputdata['correlation']:
                    return
        
        self.write_input_file(**kwargs)
        self.set_calc_args()

        try:
            self.calc_output = subprocess.check_output(self.calc_args,
                    stderr=subprocess.STDOUT).split('\n')
            if self.save_vecs:
                fname = os.path.basename(self.input_file.name) + '.movecs'
                self.movecs_path = os.path.join(inp.inputdata['share_dir'], fname)
                src = os.path.join(inp.inputdata['scrdir'], fname)
                copy2(src, self.movecs_path)
            self.parse_results(self.calc_output)
        except subprocess.CalledProcessError as error:
            print "Calculation error!"
            print '-------------------------'
            print 'INPUT FILE OF FAILED CALC'
            print '-------------------------'
            print open(self.input_file.name).read()
            print '----------------------'
            print 'OUTPUT OF FAILED CALC:'
            print '----------------------'
            print error.output
            self.parse_results(error.output)
            raise RuntimeError('failed calculation')
        except IndexError as e:
            print "Parsing Error!", e
            print '-------------------------'
            print "INPUT FILE OF FAILED PARSE"
            print '-------------------------'
            print open(self.input_file.name).read()
            raise RuntimeError('failed parsing')
        del self.input_file

class NWFragment(Fragment):

    def set_calc_args(self):
        self.calc_args = ['nwchem.x', self.input_file.name]
        #with open(self.input_file.name) as te:
        #    print "going to run:"
        #    print te.read()

    def get_best_guess(self):
        '''Run a HF SCF calculation for the default atomic and charge-local
        initial guesses, return the vecs of the lowest-energy result and save
        those results in self.results.'''
        
        def formula(atom_list):
            '''An tuple of atomic symbols in a fragment. Order matters!'''
            return tuple([atom.sym for atom in atom_list])

        sub_fragments = [[self.atoms[iat] for iat in frag] for frag in
                         geom.makefrag_auto(self.atoms)]

        unique = set([formula(frag) for frag in sub_fragments])
        frag_map = { form : 
                     [frag for frag in sub_fragments if formula(frag) == form]
                     for form in unique
                   }
        
        Frag = self.__class__
        neutral = {}
        charged = {}
        
        # Get MO's for each sub-fragment: both charged and neutral
        for form in unique:
            sub_fragment = frag_map[form][0]

            neutral[form] = Frag(sub_fragment, 0)
            charged[form] = Frag(sub_fragment, self.charge)
            
            neutral[form].run(calc='energy_hf', save_vecs=True)
            charged[form].run(calc='energy_hf', save_vecs=True)

        # Try the atomic initial guess first
        calcs = [Frag(self.atoms, self.charge)]
        calcs[0].run(calc='energy_hf', save_vecs=True, input_vecs='default')

        # Try each of the charge-local guesses
        for i, frag_i in enumerate(sub_fragments):

            guess = [     charged[formula(frag_j)].movecs_path if i == j
                     else neutral[formula(frag_j)].movecs_path
                     for j, frag_j in enumerate(sub_fragments)
                    ]

            calc = Frag(self.atoms, self.charge)
            try:
                calc.run(calc='energy_hf', save_vecs=True, input_vecs=guess)
                calcs.append(calc)
            except RuntimeError:
                print "Caught calc failure in get_best_guess...",
                if 'hf' in calc.energy and calc.energy['hf'] < -0.1:
                    print "Using possibly non-converged SCF energy"
                    calcs.append(calc)
                else:
                    print "discarding calc from candidates"

        # Choose energy/vectors of the lowest HF energy calc
        best_calc = min(calcs, key= lambda x : x.energy['hf'])
        self.energy['hf'] = best_calc.energy['hf']
        self.parse_results(best_calc.calc_output)
        return best_calc.movecs_path

    def write_input_file(self, **kwargs):
        os.chdir(inp.inputdata['scrdir'])
        if isinstance(self.index, tuple):
            fprefix = ''.join(map(str, self.index))
        else:
            fprefix = str(self.index)
        if self.charge != 0:
            fprefix += "%+d" % self.charge
        self.input_file = tempfile.NamedTemporaryFile(dir=inp.inputdata['scrdir'],
                prefix=fprefix, delete=False)

        f = self.input_file
        f.write('scratch_dir %s\n' % inp.inputdata['scrdir'])
        f.write('permanent_dir %s\n' % inp.inputdata['scrdir'])
        f.write('start\n\n')

        f.write('charge %s\n' % str(self.charge))
        f.write('geometry units angstroms noautosym noautoz nocenter\n')
        f.write('\n'.join(map(str, self.atoms)))
        f.write('\nend\n\n')
        
        f.write('basis spherical noprint\n')
        f.write('  * library %s\n' % inp.inputdata['basis'])
        f.write('end\n\n')
        if 'ecp' in inp.inputdata['basis']:
            f.write('ecp\n * library %s\nend\n\n' % inp.inputdata['basis'])

        if self.embedding_field:
            f.write('bq units angstroms\n')
            f.write('\n'.join([4*'%18.8f' % (at.pos[0], at.pos[1], at.pos[2], chg) 
                for at, chg in self.embedding_field]))
            f.write('\nend\n\n')

        f.write('scf\n')
        f.write('sym off; adapt off\n')
        f.write('%s\n' % inp.inputdata['hftype'])
        f.write('nopen %d\n' % self.spinz())

        vec_string = 'vectors input '
        if self.input_vecs:
            if self.input_vecs == 'default':
                vec_string += 'atomic'
            elif isinstance(self.input_vecs, list) or isinstance(self.input_vecs, tuple):
                vec_string += 'fragment ' + ' '.join(self.input_vecs)
            else:
                vec_string += self.input_vecs
        else:
            vec_string += 'atomic'

        vec_string += ' output %s' % os.path.join(inp.inputdata['scrdir'], 
                                            os.path.basename(self.input_file.name) 
                                            + '.movecs')

        f.write('%s\n' % ' \\\n'.join(textwrap.wrap(vec_string, width=90,
            break_long_words=False)))
        if self.noscf:
            f.write('noscf\n')

        if self.save_ints:
            # int_type: overlap kinetic potential ao2eints bq_pot
            f.write('print %s\n' % ' '.join(['"%s"' % int_type 
                for int_type in self.save_ints]))
        f.write('end\n\n')

        if self.calc == 'energy':
            f.write('task scf energy\n\n')
            if inp.inputdata['correlation']:
                theory = inp.inputdata['correlation']
                if theory != 'mp2':
                    raise RuntimeError('theory %s is not yet handled' % theory)
                f.write('tce\n scf\n mp2\n freeze atomic\nend\n')
                f.write('task tce energy\n\n')

        elif self.calc == 'energy_hf':
            f.write('task scf energy\n\n')

        elif self.calc == 'esp':
            f.write('task scf energy\n\n')
            f.write('esp\n recalculate\nend\n')
            f.write('task esp\n\n')

        else:
            raise RuntimeError('calc %s is not handled yet by NWFragment' %
                    self.calc)
        f.close()

    def parse_results(self, data):
        for n, line in enumerate(data):
            
            if "Total SCF energy" in line:
                self.energy['hf'] = float(line.split()[-1])
                self.energy['total'] = self.energy['hf']
                continue

            if "MBPT(2) total energy" in line:
                self.energy['total'] = float(line.split()[-1])
                self.energy['corr'] = self.energy['total'] - self.energy['hf']
                continue

            if 'ESP' in line:
                esp_charges = []
                for idx in range(n+3, n+3+len(self.atoms)):
                    esp_charges.append(float(data[idx].split()[-1]))
                self.esp_charges = np.array(esp_charges)
                continue

            if 'Nuclear repulsion energy' in line:
                self.energy['nuclear'] = float(line.split()[-1])
                continue

            if 'Mulliken analysis of the total density' in line:
                self.mulliken_charges = []
                for idx in range(n+5, n+5+len(self.atoms)):
                    self.mulliken_charges.append(float(data[idx].split()[3]))
                self.mulliken_charges = np.array(self.mulliken_charges)
                continue

class G09Fragment(Fragment):
    pass


if __name__ == "__main__":
    '''Test of Fragment class'''
    atoms = geom.load_geometry('test.xyz')
    inp.inputdata['scrdir'] = os.path.join(os.getcwd(), 'scrdir')
    inp.inputdata['basis'] = '4-31g'
    inp.inputdata['hftype'] = 'rohf'
    inp.inputdata['correlation'] = 'mp2'
    inp.inputdata['share_dir'] = os.path.join(os.getcwd(), 'shared_data')
    print "Loaded geometry:"
    print '\n'.join(map(str, atoms))
    frag = NWFragment(atoms, 1)
    print frag.label()
    frag.run()
    print frag.energy
