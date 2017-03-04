# This script uses PyFragment as a module to perform
# a VBCT energy scan, collecting both fragment 
# approximations and the exact (non-fragmented) energy
# Call with mpirun -n <nproc> python pes.py
# In order to run as parallel job
import pyfrag.vbct as pyfrag
import pyfrag.Globals.params
import numpy as np
import sys
import os
import pandas as pd
import cPickle

# DEFINE PES coordinate parameterizations here
#----------------------------------------------
def He2_geom(x):
    geom = [ ['He', 0.0, 0.0, 0.0],
             ['He',   x, 0.0, 0.0] ]
    return pyfrag.geom.load_geometry(geom)

def He3_symm_geom(x):
    geom = [ ['He', 0.0, 0.0, 0.0],
             ['He',   x, 0.0, 0.0],
             ['He', 2*x, 0.0, 0.0] ]
    return pyfrag.geom.load_geometry(geom)

def He3_dimerize_geom(x):
    # x sweeps from 0 to 1 
    # coordinate scans from one uhf/avdz local min to another
    # spin density goes (.25|.5|.25) --> (.5|.5|0)
    geom = [ ['He', -1.24 + x*(-1.07825455+1.24), 0.0, 0.0],
             ['He',   0.0, 0.0, 0.0],
             ['He',  1.24 +  x*(1.96169068-1.24), 0.0, 0.0] ]
    return pyfrag.geom.load_geometry(geom)

def He3_excursion_geom(x):
    # x is displacement of weakly bound He from the dimer cation
    # x == 0 is minimum at uhf/avdz
    geom = [ ['He', -1.07825455, 0.0, 0.0],
             ['He',   0.0, 0.0, 0.0],
             ['He',  1.96169068 + x, 0.0, 0.0] ]
    return pyfrag.geom.load_geometry(geom)


def He4_symm_geom(x):
    geom = [ ['He', 0.0, 0.0, 0.0],
             ['He',   x, 0.0, 0.0],
             ['He', 2*x, 0.0, 0.0],
             ['He', 3*x, 0.0, 0.0] ]
    return pyfrag.geom.load_geometry(geom)

def He4_asymm_geom(x):
    geom = [ ['He',  -2.5 + (-2.06 + 2.5)*x, 0.0, 0.0],
             ['He', -0.54 + (-.98  + .54)*x, 0.0, 0.0],
             ['He',  0.54 + (0.98 - 0.54)*x, 0.0, 0.0],
             ['He',   2.5 +  (2.06 - 2.5)*x, 0.0, 0.0] ]
    return pyfrag.geom.load_geometry(geom)

def He8_geom(ring_pos, radius=2.385, nsolvent=5):
    geom = [ ['He', -1.074, 0.0, 0.0],
             ['He',    0.0, 0.0, 0.0],
             ['He',  1.986, 0.0, 0.0] ]
    dangle = 2*np.pi / nsolvent
    angles = [n*dangle for n in range(nsolvent)]
    for angle in angles:
        geom.append(['He', ring_pos, round(radius*np.cos(angle),3), round(radius*np.sin(angle),3)])
    return pyfrag.geom.load_geometry(geom)

def Ar4_geom(x):
    # about minimum at UHF/LANL2DZ ECP
    geom = [ ['Ar',   0.0+x, 0.0, 0.0],
             ['Ar',   3.467, 0.0, 0.0],
             ['Ar',   6.061, 0.0, 0.0],
             ['Ar', 9.528-x, 0.0, 0.0] ]
    return pyfrag.geom.load_geometry(geom)

def Ar4_symm_geom(x):
    geom = [ ['Ar', 0.0, 0.0, 0.0],
             ['Ar',   x, 0.0, 0.0],
             ['Ar', 2*x, 0.0, 0.0],
             ['Ar', 3*x, 0.0, 0.0] ]
    return pyfrag.geom.load_geometry(geom)

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
    geom = []
    for atom in origin:
        geom.append([atom[0]])
        geom[-1].extend(list(atom[1]))
    # recommend scanning np.linspace(-0.2, 1.0, 11)
    return pyfrag.geom.load_geometry(geom)


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
    geom = []
    for atom in origin:
        geom.append([atom[0]])
        geom[-1].extend(list(atom[1]))
    # scan np.linspace(-1.5, 1.5, 13)
    return pyfrag.geom.load_geometry(geom)

# DEFINE PES parameter ranges here (consistent naming)
#-----------------------------------------------------
He2_range = np.arange(0.9, 2.6, 0.1)
He3_symm_range = np.arange(0.9, 2.6, 0.1)
He3_dimerize_range = np.linspace(0.0, 1.0, 11)
He3_excursion_range = np.linspace(-0.6, 0.6, 11)
He4_symm_range = np.arange(0.9, 2.6, 0.1)
He4_asymm_range = np.linspace(-0.2, 1., 13)
He8_range = np.linspace(-1.1, 2.0, 7)
Ar4_range = np.linspace(-0.8, 0.8, 11)
Ar4_symm_range = np.linspace(2.4, 3.4, 11)
NaWat3_linear_range = np.linspace(-0.2, 1.0, 11)
NaWat3_triangle_range = np.linspace(-1.5, 1.5, 13)

# SPECIFY system name; PES will be generated automatically
# --------------------------------------------------------
SYSTEM_NAME = "He3_symm"

# SPECIFY dict of calculations
# ----------------------------
BASIS = 'aug-cc-pvdz' # aug-cc-pvdz except Ar is lanl2dz_ecp
HFTYPE = 'uhf'
CORRELATION = False

calcParameterMaps = {}
calcParameterMaps['exact'] = {
         'diagonal' : 'chargelocal_dimers',
         'relax_neutral_dimers' : False,
         'corr_neutral_dimers' : False,
         'coupling' : 'dimer_gs_no_embed',
         'embedding' : True,
         'backend' : 'nw',
         'fragmentation' : 'full_system',
         'charge_states' : 'hop 1',
         'basis' : BASIS,
         'hftype' : HFTYPE,
         'correlation' : CORRELATION
         }
calcParameterMaps['GScoupling'] = { 
         'diagonal' : 'chargelocal_dimers',
         'relax_neutral_dimers' : False,
         'corr_neutral_dimers' : False,
         'coupling' : 'dimer_gs_no_embed',
         'embedding' : True,
         'backend' : 'nw',
         'fragmentation' : 'auto',
         'charge_states' : 'hop 1',
         'basis' : BASIS,
         'hftype' : HFTYPE,
         'correlation' : CORRELATION
         }
calcParameterMaps['relaxDiag_GScoupling'] = { 
         'diagonal' : 'chargelocal_dimers',
         'relax_neutral_dimers' : True,
         'corr_neutral_dimers' : False,
         'coupling' : 'dimer_gs_no_embed',
         'embedding' : True,
         'backend' : 'nw',
         'fragmentation' : 'auto',
         'charge_states' : 'hop 1',
         'basis' : BASIS,
         'hftype' : HFTYPE,
         'correlation' : CORRELATION
         }
calcParameterMaps['GSPolarizedCoupling'] = { 
         'diagonal' : 'chargelocal_dimers',
         'relax_neutral_dimers' : False,
         'corr_neutral_dimers' : False,
         'coupling' : 'dimer_gs',
         'embedding' : True,
         'backend' : 'nw',
         'fragmentation' : 'auto',
         'charge_states' : 'hop 1',
         'basis' : BASIS,
         'hftype' : HFTYPE,
         'correlation' : CORRELATION
         }
calcParameterMaps['relaxDiag_GSPolarizedCoupling'] = { 
         'diagonal' : 'chargelocal_dimers',
         'relax_neutral_dimers' : True,
         'corr_neutral_dimers' : False,
         'coupling' : 'dimer_gs',
         'embedding' : True,
         'backend' : 'nw',
         'fragmentation' : 'auto',
         'charge_states' : 'hop 1',
         'basis' : BASIS,
         'hftype' : HFTYPE,
         'correlation' : CORRELATION
         }
calcParameterMaps['mono_ip'] = { 
         'diagonal' : 'mono_ip',
         'relax_neutral_dimers' : True,
         'corr_neutral_dimers' : False,
         'coupling' : 'mono_ip',
         'embedding' : False,
         'backend' : 'nw',
         'fragmentation' : 'auto',
         'charge_states' : 'hop 1',
         'basis' : BASIS,
         'hftype' : HFTYPE,
         'correlation' : CORRELATION
         }

active_calcs = ['mono_ip', 'exact']
for calc in calcParameterMaps.keys():
    if calc not in active_calcs:
        del calcParameterMaps[calc]
# Do not modify code below; system&calc params are defined ABOVE
# ---------------------------------------------------------------
thisModule = sys.modules[__name__]
geom_generator = getattr(thisModule, "%s_geom" % SYSTEM_NAME)
coord_range = getattr(thisModule, "%s_range" % SYSTEM_NAME)

def make_datapath():
    if not CORRELATION:
        theory = HFTYPE
    else:
        theory = str(CORRELATION)
    path = '_'.join([SYSTEM_NAME, theory, BASIS])
    
    if not os.path.exists(path):
        return os.path.join(os.getcwd(), path+'.dat')

    i = 1
    datapath = "%s%s.dat" % (path, str(i))
    datapath = os.path.join(os.getcwd(), datapath)
    while os.path.exists(datapath):
        i += 1
        datapath = "%s%s.dat" % (path, str(i))
        datapath = os.path.join(os.getcwd(), datapath)
    return datapath

def geom2xyz(x, geom):
    return '%s\npes_coord %5.2f\n%s\n' \
            % (len(geom), x, '\n'.join(map(str, geom)))

def main(datapath=None):
    if datapath == None:
        datapath = make_datapath()

    pyfrag.make_scratch_dirs()
    energy_driver = pyfrag.drv.energy_driver
    pyfrag.inp.VERBOSE = True
    GLOBALS = pyfrag.inp.inputdata
    GLOBALS['task'] = 'energy'
    options = params.options
    
    # Store the PES data in a dictionary of dictionaries
    results = {}
    for calc_type, params in calcParameterMaps.items():
        GLOBALS.update(params)
        results[calc_type] = {}
        properties = {}
        for x in coord_range:
            geom = geom_generator(x)
            GLOBALS['geometry'] = geom
            try:
                sp_data = energy_driver()
            except RuntimeError:
                sp_data = {k : np.nan for k in results.values()[0]}
            for k,v in sp_data.items():
                if k in properties:
                    properties[k].append(v)
                else:
                    properties[k] = [v]

        results[calc_type] = properties

    # Rank 0 builds a Pandas DataFrame
    rank = pyfrag.rank
    if rank == 0:
        with open(datapath+'.temp', 'wb') as datafile:
            cPickle.dump((results, calcParameterMaps, 
                [geom2xyz(x, geom_generator(x)) for x in coord_range]), datafile)
        sp_data0 = results.values()[0]
        col_names = sp_data0.keys()
        column_indices = [col_names, results.keys()]
        multi_idx = pd.MultiIndex.from_product(column_indices, names=['property', 'calc_type'])
        df = pd.DataFrame(index=coord_range, columns=multi_idx)

        # Store xyz format geometries
        df['geom_xyz'] = [geom2xyz(x, geom_generator(x)) for x in coord_range]

        # Store 3-dimensional data: property/calc_type/pes_coordinate
        for calc_type, properties in results.items():
            for k, v in properties.items():
                df[k, calc_type] = v
                
        with open(datapath, 'wb') as datafile:
            cPickle.dump((df, calcParameterMaps), datafile)

    pyfrag.clean_scratch_dirs()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2 and not os.path.exists(sys.argv[1]):
        main(datapath=sys.argv[1])
    else:
        print "need unique path"
