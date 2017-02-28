import numpy as np
import re
import os

# Physical constants
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

# Cutoff bond distances for auto-fragmentation
frag_cutoffs = { 
                 ('c','c'): 1.5,
                 ('c','h'): 1.3,
                 ('c','o'): 2.0,
                 ('h','h'): 1.3,
                 ('h','o'): 1.3,
               }

# Module-level data: shared with others
# geometry: the list of Atoms
# fragments: the list of frags, each of which is
#   a list of atom indices 
geometry = []
fragments = [] 
LatticeTuple = namedtuple('Lattice', 
        'a b c alpha beta gamma axis'])
lattice = LatticeTuple(0.0, 0.0, 0.0, 90.0, 90.0, 90.0, 0.0)
lat_vecs = np.zeros((3,3))


def update_lat_vecs():
    RADIAN = 57.29577951308232087721
    a_dim = abs(lattice.a) > 0.1
    b_dim = abs(lattice.b) > 0.1
    c_dim = abs(lattice.c) > 0.1

    ab = 

class Atom:
    '''Convenience class for loading and storing geometry data'''
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
        '''.xyz string representation of Atom'''
        return ("%2s " + 3*"%16.10f") % ((self.sym.capitalize(),) + tuple(self.pos))

    def __eq__(self, at2):
        '''Atom1 == Atom2 if same symbol and position'''
        if self.sym != at2.sym:
            return False
        if not np.allclose(self.pos, at2.pos):
            return False
        return True

     
def load_geometry(data, units='angstrom'):
    '''Loads geometry from input text, lists, or filename.
    
    Tries to be flexible with the form of input 'data' argument. 
    Uses regex to extract atomic coordinates from text.
    
    Args:
        data: string, list of strings, list of lists, or filename
            containing the xyz coordinate data
        units (default Angstrom): "bohr" or "angstrom"
    Returns:
       None: the geometry is saved as a module-level variable
   '''
   geometry[:] = [] # syntax essential to refer to the global geometry!

    # Convert input data to a list of strings
    if isinstance(data, str): 
        data = data.split('\n')
    
    if isinstance(data[0], str) and os.path.exists(data[0]):
        fname = data[0]
        data[:] = []
        with open(fname) as fp:
            for line in fp:
                data.append(line)

    for i in range(len(data)):
        if isinstance(data[i], list):
            data[i] = ' '.join(map(str, data[i]))

    # Now try to make an Atom out of each string
    for atomstr in data:
        try: 
            geometry.append(Atom(atomstr))
        except AttributeError: 
            dat = atomstr.split()
            if len(dat) >= 7:
                try:
                    a, b, c, alpha, beta, gamma, axis = map(float, dat[:7])
            pass # just skip non-atom lines

def set_frag_full_system(geometry):
    '''No fragmentation: all atoms in system belong to one fragment.

    Use this to perform one big reference QM calculation
    
    Args:
        geometry: list of Atom objects
    Returns:
        None (module-level fragments list is set)
    '''
    full_sys = range(len(geometry))
    fragments[:] = [] # refer to module-level fragments, not local
    fragments.append(full_sys)

def makefrag_auto(geometry):
    '''Auto-generate list of fragments based on bond-length frag_cutoffs.

    Use this if you don't wish to manually assign atoms to fragments.
    
    Args:
        geometry: list of Atom objects
    Returns:
        None (module-level fragments list is set)
    '''
    fragments[:] = []
    from itertools import combinations
    
    def cutoff(at1, at2):
        key = tuple(sorted([at1.sym, at2.sym]))
        if key in frag_cutoffs: return frag_cutoffs[key]
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
        fragments.append(sorted(frag))

    # sanity check: each atom must be counted once and only once
    assert sorted([idx for frag in fragments for idx in frag]) == range(natm)

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
