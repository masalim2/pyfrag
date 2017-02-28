import numpy as np
import re
import os

BOHR2ANG = 0.529177249
mass_map = { 
   'h'  : 1.0,
   'd'  : 2.0,
   'he' : 4.0,
   'o'  : 16.0,
   'c'  : 12.0,
   'ar' : 40.0
}
z_map = {
   'h'  : 1,
   'd'  : 1,
   'he' : 2,
   'o'  : 8,
   'c'  : 6,
   'ar' : 18
}

class Atom:
    pattern = re.compile(r'''
                         ([a-zA-Z]+[+-]*)\s* # Symbol + formal charges
                         \s*([-+]?[0-9]*\.?[0-9]+[dDeE]?[-+]?[0-9]*)\s* # x
                         \s*([-+]?[0-9]*\.?[0-9]+[dDeE]?[-+]?[0-9]*)\s* # y
                         \s*([-+]?[0-9]*\.?[0-9]+[dDeE]?[-+]?[0-9]*)\s* # z
                         ''', re.VERBOSE)

    def __init__(self, atomstr, units='angstrom'):
        if units.lower().startswith('bohr'): 
            scale = BOHR2ANG
        else:
            scale = 1.0

        name, x, y, z = self.pattern.search(atomstr).groups()
        
        self.formal_chg = name.count('+') - name.count('-')
        self.sym = name.lower()[:2].replace('+','').replace('-','')
        
        x,y,z = [float(r.replace('D', 'E').replace('d','e')) for r in x,y,z]
        self.pos = np.array([x,y,z])*scale

    def __repr__(self):
        return ("%2s " + 3*"%14.8f") % ((self.sym.capitalize(),) + tuple(self.pos))

    def __eq__(self, at2):
        if self.sym != at2.sym:
            return False
        if not np.allclose(self.pos, at2.pos):
            return False
        return True

     
def load_geometry(data, units='angstrom'):
    '''data argument: multi-line string, list of strings, 
    list of lists, or filename.
    Uses regex to extract all atom strings from text'''
    geometry = []

    if isinstance(data, str): 
        data = data.split('\n')

    if isinstance(data[0], str) and os.path.exists(data[0]):
        fname = data[0]
        data[:] = []
        with open(fname) as fp:
            for line in fp:
                data.append(line)
    
    for atom in data:
        if isinstance(atom, list):
            atomstr = ' '.join(map(str, atom))
        else:
            atomstr = atom

        try: 
            geometry.append(Atom(atomstr))
        except AttributeError: 
            pass # just skip non-atom lines

    return geometry

def makefrag_full_system(geometry):
    '''No fragmentation: all atoms in system belong to one fragment'''
    return [ range(len(geometry)) ]

def makefrag_auto(geometry):
    '''Auto-generate list of fragments based on the bond-length cutoffs
    defined below'''
    from itertools import combinations
    cutoffs = \
    {
        ('c','c'): 1.5,
        ('c','h'): 1.3,
        ('c','o'): 2.0,
        ('h','h'): 1.3,
        ('h','o'): 1.3,
    }
    
    def cutoff(at1, at2):
        key = tuple(sorted([at1.sym, at2.sym]))
        if key in cutoffs: return cutoffs[key]
        else: return 0.0

    # build list of neighbors within bonding cutoff for each atom
    natm = len(geometry)
    neighbors = [[] for i in range(natm)]
    for i,j in combinations(range(natm), 2):
        dist = np.linalg.norm(geometry[i].pos - geometry[j].pos)
        if dist < cutoff(geometry[i],geometry[j]):
            neighbors[i].append(j)
            neighbors[j].append(i)

    # assign atoms to fragments via breadth-first graph search
    freeAtoms = range(natm)
    queue = []
    fragmentsList = []
    while freeAtoms:
        queue.append(freeAtoms.pop(0))
        frag = []
        while queue:
            atom = queue.pop(0)
            frag.append(atom)
            for atom2 in neighbors[atom]:
                if atom2 in freeAtoms:
                    queue.append(atom2)
                    freeAtoms.remove(atom2)
        fragmentsList.append(sorted(frag))
    assert sorted([idx for frag in fragmentsList for idx in frag]) == range(natm)
    return fragmentsList

def nuclear_repulsion_energy(geometry):
    '''Nuclear repulsion energy, hartrees'''
    from itertools import combinations
    return sum([
                z_map[at1.sym]*z_map[at2.sym]/
                (np.linalg.norm(at1.pos-at2.pos)/BOHR2ANG)
                for at1, at2 in combinations(geometry, 2)
               ])


if __name__ == "__main__":
    geom = load_geometry('test.xyz')
    print "Loaded geometry:"
    print '\n'.join(map(str, geom))
