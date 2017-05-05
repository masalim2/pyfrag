''' This script uses PyFragment as a module to perform a VBCT energy scan,
collecting both fragment approximations and the exact (non-fragmented) energy
Call with mpirun -n <nproc> python pes.py In order to run as parallel job
'''
import numpy as np
import sys
import h5py
import argparse

from pyfrag.vbct import energy as vbct_energy
from pyfrag.Globals import params, geom, MPI
from pyfrag.Globals import utility as util

# DEFINE PES coordinate parameterizations here
#----------------------------------------------
def He2_geom(x):
    g = [ ['He', 0.0, 0.0, 0.0],
             ['He',   x, 0.0, 0.0] ]
    geom.load_geometry(g)

def He3_symm_geom(x):
    g = [ ['He', 0.0, 0.0, 0.0],
             ['He',   x, 0.0, 0.0],
             ['He', 2*x, 0.0, 0.0] ]
    geom.load_geometry(g)

def He3_dimerize_geom(x):
    # x sweeps from 0 to 1
    # coordinate scans from one uhf/avdz local min to another
    # spin density goes (.25|.5|.25) --> (.5|.5|0)
    g = [ ['He', -1.24 + x*(-1.07825455+1.24), 0.0, 0.0],
             ['He',   0.0, 0.0, 0.0],
             ['He',  1.24 +  x*(1.96169068-1.24), 0.0, 0.0] ]
    geom.load_geometry(g)

def He3_excursion_geom(x):
    # x is displacement of weakly bound He from the dimer cation
    # x == 0 is minimum at uhf/avdz
    g = [ ['He', -1.07825455, 0.0, 0.0],
             ['He',   0.0, 0.0, 0.0],
             ['He',  1.96169068 + x, 0.0, 0.0] ]
    geom.load_geometry(g)


def He4_symm_geom(x):
    g = [ ['He', 0.0, 0.0, 0.0],
             ['He',   x, 0.0, 0.0],
             ['He', 2*x, 0.0, 0.0],
             ['He', 3*x, 0.0, 0.0] ]
    geom.load_geometry(g)

def He4_asymm_geom(x):
    g = [ ['He',  -2.5 + (-2.06 + 2.5)*x, 0.0, 0.0],
             ['He', -0.54 + (-.98  + .54)*x, 0.0, 0.0],
             ['He',  0.54 + (0.98 - 0.54)*x, 0.0, 0.0],
             ['He',   2.5 +  (2.06 - 2.5)*x, 0.0, 0.0] ]
    geom.load_geometry(g)

def He8_geom(ring_pos, radius=2.385, nsolvent=5):
    g = [ ['He', -1.074, 0.0, 0.0],
             ['He',    0.0, 0.0, 0.0],
             ['He',  1.986, 0.0, 0.0] ]
    dangle = 2*np.pi / nsolvent
    angles = [n*dangle for n in range(nsolvent)]
    for angle in angles:
        g.append(['He', ring_pos, round(radius*np.cos(angle),3), round(radius*np.sin(angle),3)])
    geom.load_geometry(g)

def Ar4_geom(x):
    # about minimum at UHF/LANL2DZ ECP
    g = [ ['Ar',   0.0+x, 0.0, 0.0],
             ['Ar',   3.467, 0.0, 0.0],
             ['Ar',   6.061, 0.0, 0.0],
             ['Ar', 9.528-x, 0.0, 0.0] ]
    geom.load_geometry(g)

def Ar4_symm_geom(x):
    g = [ ['Ar', 0.0, 0.0, 0.0],
             ['Ar',   x, 0.0, 0.0],
             ['Ar', 2*x, 0.0, 0.0],
             ['Ar', 3*x, 0.0, 0.0] ]
    geom.load_geometry(g)

def NaWat3_linear_geom(x):
    # hf/aug-cc-pvdz optimized (H2O)_3Na+ cluster (higher-E stationary pt)
    # excursion of leftmost water in the O -- Na - OH - O configuration
    origin = [
     ['Na',    np.array([  0.21087905,    -1.07845876,    -0.00384438])],
     ['O',     np.array([ -0.70376204,     3.26354462,    -0.00020967])],
     ['H',     np.array([ -0.97394692,     3.75994642,    -0.75799166])],
     ['H',     np.array([ -0.97895278,     3.76348016,     0.75329091])],
     ['O',     np.array([ -0.44282906,    -3.23885333,     0.00243490])],
     ['H',     np.array([ -0.60917435,    -3.78448428,     0.75766465])],
     ['H',     np.array([ -0.59937587,    -3.79222625,    -0.74928940])],
     ['O',     np.array([  0.97598687,     0.99264382,     0.00069382])],
     ['H',     np.array([  0.42857199,     1.77680728,     0.00287819])],
     ['H',     np.array([  1.87069029,     1.29765224,     0.00193479])] ]

    r_na = origin[0][1]
    r_o  = origin[4][1]
    line = r_o - r_na
    line = line / np.linalg.norm(line)

    origin[4][1] += x*line
    origin[5][1] += x*line
    origin[6][1] += x*line
    g = []
    for atom in origin:
        g.append([atom[0]])
        g[-1].extend(list(atom[1]))
    # recommend scanning np.linspace(-0.2, 1.0, 11)
    geom.load_geometry(g)

def NaWat3_triangle_geom(x):
    # hf/aug-cc-pvdz optimized (H2O)_3Na+ cluster (lower-E stationary pt)
    # excursion of Na atom normal to the O-O-O plane
    origin = '''Na     -0.00000166     0.00006962     0.00049518
     O      -1.51025934     1.71557888     0.00059985
     H      -1.44772201     2.53265513     0.47371767
     H      -2.32859634     1.75777493    -0.47258270
     O       2.24074836     0.45075145     0.00029807
     H       2.68652258     1.13571663    -0.47663878
     H       2.91681988    -0.00819915     0.47744007
     O      -0.72998035    -2.16575869     0.00084616
     H      -1.47070339    -2.51998088     0.47117176
     H      -0.35442799    -2.89606213    -0.46923496'''.split('\n')
    assert len(origin) == 10
    for i in range(10):
        at = origin[i].split()
        name = at[0]
        pos = np.array(map(float, at[1:4]))
        origin[i] = [name, pos]
    vec1 = origin[1][1] - origin[4][1]
    vec2 = origin[1][1] - origin[7][1]
    scan_vec = np.cross(vec1, vec2)
    scan_vec /= np.linalg.norm(scan_vec)
    origin[0][1] += x*scan_vec
    g = []
    for atom in origin:
        g.append([atom[0]])
        g[-1].extend(list(atom[1]))
    # scan np.linspace(-1.5, 1.5, 13)
    geom.load_geometry(g)
def Wat4_geom(x):
    #Shrinking/expanding the size of the 4-oxygen square
    origin = '''O    0.216486    -1.989005    0.035297
        H    -0.603911    -1.505274    -0.033552
        H    0.138672    -2.535875    0.799566
        O    1.989004    0.216483    -0.035296
        H    1.505276    -0.60392    0.033505
        H    2.535922    0.13869    -0.799534
        O    -1.989001    -0.216487    -0.035297
        H    -2.535947    -0.138688    -0.799513
        H    -1.505273    0.60392    0.033493
        O    -0.216485    1.989007    0.035294
        H    -0.138694    2.535879    0.799565
        H    0.603919    1.505277    -0.033513'''.split('\n')
    assert len(origin) == 12
    for i in range(12):
        at = origin[i].split()
        name = at[0]
        pos = np.array(map(float, at[1:4]))
        origin[i] = [name, pos]
    vec1 = origin[3][1] - origin[0][1]
    vec2 = origin[6][1] - origin[0][1]
    g = []

    g.append([origin[0][0]])
    g[0].extend(list(origin[0][1]))
    g.append([origin[1][0]])
    g[1].extend(list(origin[1][1]))
    g.append([origin[2][0]])
    g[2].extend(list(origin[2][1]))

    g.append([origin[3][0]])
    g[3].extend(list(x*vec1 + origin[0][1]))
    g.append([origin[4][0]])
    g[4].extend(list(x*vec1 + origin[0][1] + origin[4][1]-origin[3][1]))
    g.append([origin[5][0]])
    g[5].extend(list(x*vec1 + origin[0][1] + origin[5][1]-origin[3][1]))

    g.append([origin[6][0]])
    g[6].extend(list(x*vec2 + origin[0][1]))
    g.append([origin[7][0]])
    g[7].extend(list(x*vec2 + origin[0][1] + origin[7][1]-origin[6][1]))
    g.append([origin[8][0]])
    g[8].extend(list(x*vec2 + origin[0][1] + origin[8][1]-origin[6][1]))

    g.append([origin[9][0]])
    g[9].extend(list(x*vec2 + x*vec1 + origin[0][1]))
    g.append([origin[10][0]])
    g[10].extend(list(x*vec2 + x*vec1 + origin[0][1] + origin[10][1]-origin[9][1]))
    g.append([origin[11][0]])
    g[11].extend(list(x*vec2 + x*vec1 + origin[0][1] + origin[11][1]-origin[9][1]))
    geom.load_geometry(g)

def ethyl4_geom(x, spacing=4.0):
    # neutral == MP2/avdz optimized neutral ethylene
    # charged == MP2/avdz optimized cation ethylene
    # z-stacking of 4 ethylenes, spaced normal to molecular plane
    # middle two ethylene geometries are linear a interpolation (0<x<1)
    # between the neutral and charged geometries (quasi CT coordinate)
    neutral = '''C   -0.6741216    -0.00000     0.00000000
        C            0.67412167     0.00000     0.0000000
        H           -1.24192904    -0.93292     0.0000000
        H           -1.24192904     0.93292     0.0000000
        H            1.24192904     0.93292     0.0000000
        H            1.24192904    -0.93292     0.0000000'''.split('\n')

    charged =''' C  -0.64659904    -0.00000     0.0000000
        C            0.64659904     0.00000     0.0000000
        H           -1.37269259    -0.87710     0.0000000
        H           -1.37269259     0.87710     0.0000000
        H            1.37269259     0.87710     0.0000000
        H            1.37269259    -0.87710     0.0000000'''.split('\n')

    atoms = [at.split()[0] for at in neutral]
    neutralpos = np.array([map(float, at.split()[1:]) for at in neutral])
    chargedpos = np.array([map(float, at.split()[1:]) for at in charged])
    pos1 = x*neutralpos + (1-x)*chargedpos
    pos2 = (1-x)*neutralpos + x*chargedpos
    spacer = np.array([0.,0.,spacing])
    g = np.array(neutralpos)
    g = np.vstack((g, pos1+spacer))
    g = np.vstack((g, pos2+2*spacer))
    g = np.vstack((g, neutralpos+3*spacer))
    assert len(g) == 24
    assert len(atoms*4) == 24
    gm = []
    for at, pos in zip(atoms*4, g):
        gm.append([at, pos[0], pos[1], pos[2]])
    geom.load_geometry(gm)

def ethyl4spac6_geom(x):
    ethyl4_geom(x, spacing=6.0)


def ethyl3_geom(theta_degrees, spacing=4.0):
    # Rotation of central ethylene (about y-axis) in a z-stack of 3 ethylenes
    def transform(g, rotation_degrees=0.0, z_translation=0.0):
        sin = np.sin(np.deg2rad(rotation_degrees))
        cos = np.cos(np.deg2rad(rotation_degrees))
        rot = np.array([ [cos, 0, -sin], [0, 1, 0], [sin, 0, cos] ])
        g_rotated = np.dot(rot, g.T).T
        return g_rotated + np.array([0., 0., z_translation])

    flat = '''C   -0.6741216    -0.00000     0.00000000
        C            0.67412167     0.00000     0.0000000
        H           -1.24192904    -0.93292     0.0000000
        H           -1.24192904     0.93292     0.0000000
        H            1.24192904     0.93292     0.0000000
        H            1.24192904    -0.93292     0.0000000'''.split('\n')
    atoms = [at.split()[0] for at in flat]
    pos1 = np.array([map(float, at.split()[1:]) for at in flat])
    pos2 = transform(pos1, theta_degrees, spacing)
    pos3 = transform(pos1, 0.0, 2.0*spacing)
    gm = np.vstack((pos1, pos2, pos3))
    g = []
    for at,pos in zip(atoms*3, gm):
        g.append([at, pos[0], pos[1], pos[2]])
    geom.load_geometry(g)


# DEFINE PES parameter ranges here (consistent naming)
#-----------------------------------------------------
He2_range = np.arange(0.9, 2.6, 0.1)
He3_symm_range = np.arange(0.9, 2.6, 0.1)
He3_dimerize_range = np.linspace(0.0, 1.0, 11)
He3_excursion_range = np.linspace(-0.6, 0.6, 11)
He4_symm_range = np.arange(0.9, 2.6, 0.1)
He4_asymm_range = np.linspace(-0.2, 1., 13)
He8_range = np.linspace(-1.1, 2.0, 11)
Ar4_range = np.linspace(-0.8, 0.8, 11)
Ar4_symm_range = np.linspace(2.4, 3.4, 11)
NaWat3_linear_range = np.linspace(-0.2, 1.0, 11)
NaWat3_triangle_range = np.linspace(-1.5, 1.5, 13)
Wat4_range = np.linspace(0.9,1.3, 11)
ethyl4_range = np.linspace(0.2,0.5,4)
ethyl4spac6_range = np.linspace(0,0.5,6)
ethyl3_range = np.linspace(0.0, 90.0, 10)


# SPECIFY dict of calculations
# ----------------------------
globalParams = {
    'mem_mb' : 8000,
    'embedding' : True,
    'backend' : 'nw',
    'basis' : 'aug-cc-pvdz',
    'hftype' : 'uhf',
}

calcParameterMaps = {}
calcParameterMaps['hf_exact'] = {
    'vbct_scheme' : 'chglocal',
    'fragmentation' : 'full_system',
    'correlation' : False
    }
calcParameterMaps['hf_chglocal'] = {
    'vbct_scheme' : 'chglocal',
    'fragmentation' : 'auto',
    'correlation' : False
    }
#calcParameterMaps['hf_monoip'] = {
#    'vbct_scheme' : 'monoip',
#    'fragmentation' : 'auto',
#    'correlation' : False
#    }
calcParameterMaps['mp2_exact'] = {
    'vbct_scheme' : 'chglocal',
    'fragmentation' : 'full_system',
    'correlation' : 'mp2'
    }
calcParameterMaps['mp2_chglocal'] = {
    'vbct_scheme' : 'chglocal',
    'fragmentation' : 'auto',
    'correlation' : 'mp2'
    }
#calcParameterMaps['mp2_monoip'] = {
#    'vbct_scheme' : 'monoip',
#    'fragmentation' : 'auto',
#    'correlation' : 'mp2'
#    }


def valid_systems():
    '''Generate list of systems implemented'''
    thisModule = sys.modules[__name__]
    for attr in dir(thisModule):
        if attr.endswith('_geom'):
            yield attr[:attr.index('_geom')]


def geom2xyz(x):
    '''Generate .xyz format geometry text'''
    return '%s\npes_coord %5.2f\n%s\n' \
            % (len(geom.geometry), x, '\n'.join(map(str, geom.geometry)))


def get_args():
    '''Get command line options, ensure validity'''
    parser = argparse.ArgumentParser(description="Scan PES with VBCT method")
    parser.add_argument("data_file", help="data storage path")
    parser.add_argument("system_name", help="molecular cluster identity")
    parser.add_argument("method", nargs='+', help="calculation method")
    parser.add_argument("--scrdir", type=str, help="top level scratch")
    parser.add_argument("--basis", type=str, help="basis (default aug-cc-pvdz)")
    args = parser.parse_args()

    system_name = args.system_name
    try:
        thisModule = sys.modules[__name__]
        geom_generator = getattr(thisModule, "%s_geom" % system_name)
        coord_range = getattr(thisModule, "%s_range" % system_name)
    except AttributeError:
        print "System %s not yet implemented" % system_name
        print "Available systems:"
        for system in valid_systems():
            print "  *", system
        sys.exit(0)

    methods = args.method
    scr = args.scrdir
    basis = args.basis
    if basis:
        globalParams['basis'] = basis
    if 'all' in methods:
        methods = calcParameterMaps.keys()
    for method in methods:
        try:
            assert method in calcParameterMaps.keys(), "%s undefined" % method
        except AssertionError:
            print "Method %s not implemented" % method
            print 'Available methods ("all" to run all):'
            print "\n".join(["  * %s" % m for m in calcParameterMaps.keys()])
            sys.exit(0)

    return geom_generator, coord_range, methods, args.data_file, system_name, scr

def log(fp, sysname, method, i, x, coord_range, results):
    '''Log to HDF5 file'''
    if MPI.rank != 0: return

    group = "%s/%s" % (sysname, method)
    name = lambda n : '%s/%s' % (group, n)
    Nstep = len(coord_range)

    geom_pos = geom.pos_array()
    natm = len(geom_pos)

    if name('energy') not in fp:
        ds = fp.create_dataset(name('energy'), (Nstep,), dtype=np.double)
        ds.attrs['basis'] = globalParams['basis']
        ds.attrs['hftype'] = globalParams['hftype']
        ds.attrs['embedding'] = globalParams['embedding']
    if name('geom_xyz') not in fp:
        ds = fp.create_dataset(name('geom_xyz'), (Nstep,natm,3), dtype=np.double)
        ds.attrs['atom_labels'] = ' '.join([at.sym for at in geom.geometry])
    if name('pes_coord') not in fp:
        fp.create_dataset(name('pes_coord'), (Nstep,), dtype=np.double)
    if name('eigvecs') not in fp:
        eigvecs = results['eigvecs']
        N = eigvecs.shape[0]
        fp.create_dataset(name('eigvecs'), (Nstep,N,N), dtype=np.double)
    if name('esp_chg') not in fp:
        fp.create_dataset(name('esp_chg'), (Nstep,natm), dtype=np.double)

    fp[name('energy')][i] = results['E(GS)']
    fp[name('geom_xyz')][i] = geom_pos
    fp[name('pes_coord')][i] = x
    fp[name('eigvecs')][i] = results['eigvecs']
    fp[name('esp_chg')][i] = results['chgdist(GS)']
    fp.flush()

def main(args):
    geom_generator, coord_range, methods, data_file, sysname, scr = args
    util.make_scratch_dirs(scr)
    if MPI.rank == 0:
        print "# System: %s" % sysname
        print "# Calc methods:", ', '.join(methods)
        print "# Basis set:", globalParams['basis']
        print "# Storing result in", data_file
        fp = h5py.File(data_file, 'a')

    energy_driver = vbct_energy.kernel
    options = params.options
    options.update(globalParams)

    for method in methods:
        method_opts = calcParameterMaps[method]
        options.update(method_opts)
        if MPI.rank == 0: print "#", method

        for i, x in enumerate(coord_range):
            geom_generator(x)
            res = energy_driver()
            if MPI.rank == 0:
                print x, res['E(GS)']
                log(fp, sysname, method, i, x, coord_range, res)
    if MPI.rank == 0:
        fp.close()

if __name__ == "__main__":
    args = get_args()
    try:
        main(args)
    finally:
        util.clean_scratch_dirs()
