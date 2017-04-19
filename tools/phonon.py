from pyfrag.Globals import geom
from pyfrag.Globals import lattice
import numpy as np
import itertools
import sys
import argparse
import os

cau = (2.99792458*2.418884326505)*1.0e-9  # c in meters per t(a.u.)
amu2au = (1.66053892/9.10938291)*1.0e4  # 1 amu in a.u. (via kgs)
cmtom = 1.0/100.0
SCALING = cmtom*(1.0/(2*np.pi*cau*amu2au**0.5))

def pretty(x):
    return "%9.3f" % x

def shift_com_to_origin():
    com = sum([at.pos for at in geom.geometry])
    com = com / len(geom.geometry)
    for at in geom.geometry:
        at.pos -= com

def make_brillouin_zone3D(kmesh_density=None):
    '''Brute-force generate uniform grid inside 1st Brillouin zone.

    Brute force but clean approach. Truncates a uniform rectangular grid;
    therefore, nonorthogonal cells may require a high density of points to
    sample adequately near the zone boundaries.
    '''

    if kmesh_density is None:
        kmesh_density = 21
    TOL = 1.0e-5
    lat_vecs = lattice.lat_vecs
    a1, a2, a3 = lat_vecs.T

    # Reciprocal lattice vectors (b)
    norm = 2*np.pi/np.dot(a1, np.cross(a2,a3))
    b1 = norm*np.cross(a2,a3)
    b2 = norm*np.cross(a3,a1)
    b3 = norm*np.cross(a1,a2)

    # All neighbors to central reciprocal cell 0
    cells = list(itertools.product([-1,0,1], repeat=3))
    cells.remove( (0,0,0) )
    cell_vecs  = np.array([n1*b1+n2*b2+n3*b3 for (n1,n2,n3) in cells])

    # Make a uniform rectangular grid; too large but sure to enclose BZ
    assert cell_vecs.shape == (len(cells), 3)
    boxmin = np.array([np.min(cell_vecs[:,i]) for i in range(3)])
    boxmax = np.array([np.max(cell_vecs[:,i]) for i in range(3)])
    xp, yp, zp = (np.linspace(boxmin[i], boxmax[i], kmesh_density) for i in range(3))
    grid = np.vstack(np.meshgrid(xp, yp, zp)).reshape(3,-1).T

    # For each reciprocal lattice vector:
    # Flag FALSE all points that lie closer to it than 
    # they do to the origin. What's left is inside the BZ.
    dists_to_origin = np.linalg.norm(grid, axis=1)
    inside_bz       = np.ones(dists_to_origin.shape, dtype=bool)
    for vec in cell_vecs:
        dists_to_vec = np.linalg.norm(grid - vec, axis=1)
        inside_bz = inside_bz & (dists_to_vec+TOL > dists_to_origin)
   
    return grid[inside_bz]

def read_force_consts_np(fname):
    data = np.load(fname)
    hess = data['hess']
    cell_list = data['cell_list']

    F = {}

    for (i, (a,b,c)) in enumerate(cell_list):
        F[(a,b,c)] = hess[i, :, :]
    return F

def read_force_consts_txt(fname):
    F = {}
    natom = len(geom.geometry)
    with open(fname) as fp:
        line = fp.readline()
        while 'cell' not in line:
            line = fp.readline()
        header = line

        while 'cell' in header:
            a,b,c = map(int, header.split()[1:])
            F[(a, b, c)] = np.zeros((3*natom, 3*natom))

            for row in range(3*natom):
                dat = map(float, fp.readline().split())
                F[(a, b, c)][row, :] = dat
            header = fp.readline()
    return F

def mass_weight(hess):
    hess_mw = {}
    ndim = 3*len(geom.geometry)
    massvec = [geom.mass_map[at.sym] for at in geom.geometry 
                for i in range(3)]
    sqrt_inv_massvec = 1.0 / np.sqrt(np.array(massvec))
    for cell in hess:
        hess_mw[cell] = hess[cell].copy()
        for row in range(ndim):
            hess_mw[cell][row,:] *= sqrt_inv_massvec
        for col in range(ndim):
            hess_mw[cell][:,col] *= sqrt_inv_massvec

    return hess_mw

def smoothing_matrix(hess_mw):
    # Construct Translational normal modes
    natom = len(geom.geometry)
    Lx = np.zeros((3*natom, 1))
    Ly = np.zeros((3*natom, 1))
    Lz = np.zeros((3*natom, 1))
    for i,at in enumerate(geom.geometry):
        sqrtm = geom.mass_map[at.sym]**0.5
        Lx[3*i] = sqrtm
        Ly[3*i+1] = sqrtm
        Lz[3*i+2] = sqrtm
    Lx /= np.linalg.norm(Lx)
    Ly /= np.linalg.norm(Ly)
    Lz /= np.linalg.norm(Lz)

    # Projector: remove translations
    I = np.eye(3*natom)
    Prx = I - np.dot(Lx, np.transpose(Lx)) # Px = I - |T_x><T_x|
    Pry = I - np.dot(Ly, np.transpose(Ly)) # Py = I - |T_y><T_y|
    Prz = I - np.dot(Lz, np.transpose(Lz)) # Pz = I - |T_z><T_z|

    Pr = np.dot(np.dot(Prx, Pry), Prz) # P = Px*Py*Pz

    # Create a correction matrix that is later added to D(k), defined 
    # such that at k=0, you are diagonalizing PD(0)P
    d = np.zeros((3*natom, 3*natom))
    for cell in hess_mw:
        d += hess_mw[cell]
    smoothing_mat = (np.dot(np.dot(Pr, d), Pr) - d)/len(hess_mw)
    return smoothing_mat

def Dmat(hess_mw, kvec, smooth_mat=None):
    natom = len(geom.geometry)
    lat_vecs = lattice.lat_vecs
    Ra, Rb, Rc = lat_vecs.T
    d = np.zeros((3*natom,3*natom), dtype='complex128')
    if smooth_mat is None:
        smooth_mat = np.zeros((3*natom, 3*natom))
    
    for cell in hess_mw:
        a,b,c = cell
        rvec = a*Ra + b*Rb + c*Rc
        arg = 1.0j*np.dot(kvec, rvec)
        d += (hess_mw[cell]+smooth_mat) * np.exp(arg)
    #assert np.allclose(d, np.conj(d.T))
    return d

def phonon(hess_mw, kmesh, smooth_mat=None):
    freqs = []
    vecs = []
    print "# Phonons at", len(kmesh), "k-points"
    for kvec in kmesh:
        d = Dmat(hess_mw, kvec, smooth_mat)
        eigvals, eigvecs = np.linalg.eigh(d)
        freqs.append(np.sign(eigvals)*SCALING*
                np.sqrt(np.sign(eigvals)*eigvals))
        for col in eigvecs.T:
            col *= np.exp(-1j*np.angle(col[0]))
        vecs.append(eigvecs)
    return freqs, vecs

def phonon_freqs(hess_mw, kmesh, smooth_mat=None):
    freqs = []
    print "# Phonons at", len(kmesh), "k-points"
    for kvec in kmesh:
        d = Dmat(hess_mw, kvec, smooth_mat)
        eigvals = np.linalg.eigvalsh(d)
        freqs.append(np.sign(eigvals)*SCALING*
                np.sqrt(np.sign(eigvals)*eigvals))
    return freqs

def parseargs():
    parser = argparse.ArgumentParser(description = 
    'Calculate phonons for given crystal structure and interaction force constants')
    parser.add_argument('structure', help='.xyz structure file')
    parser.add_argument('hessian', nargs='?', default=None, help='.hess data file')
    parser.add_argument('--bz_only', action='store_true',
                        help='only generate and print 1st BZ kmesh')
    parser.add_argument('--bz_file', type=str,
                        help='read existing kmesh from file')
    parser.add_argument('--kmesh_density', type=int,
        help='points per dimension in rectangular k-grid before truncation')
    parser.add_argument('--smooth', action='store_true',
                        help='force acoustic branches to zero at gamma point')
    parser.add_argument('--binary_file', type=str,
                        help='save freqs as binary numpy array')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parseargs()
    #geom.load_geometry(open(args.structure).read().replace('H','D'))
    geom.load_geometry(args.structure)

    if args.bz_file:
        bz = np.loadtxt(args.bz_file)
        assert bz.shape[1] == 3 and len(bz.shape) == 2
        print "# Loaded %d kpoints from file" % len(bz)
    else:
        bz = make_brillouin_zone3D(args.kmesh_density)
        print "# Generated %d kpoints in 1st BZ" % len(bz)

    if args.bz_only:
        for kvec in bz:
            print kvec[0], kvec[1], kvec[2]
        sys.exit(0)

    if args.hessian is None:
        raise RuntimeError("Need input hessian file")
    if args.binary_file:
        assert not os.path.exists(args.binary_file), "file already exists"

    shift_com_to_origin()
    hess = read_force_consts_txt(args.hessian)
    print "# Read interaction force consts in %d cells" % len(hess.keys())
    hess_mw = mass_weight(hess)

    if args.smooth:
        smooth_mat = smoothing_matrix(hess_mw)
    else:
        smooth_mat = None

    #freqs, vecs = phonons(hess_mw, bz, smooth_mat)
    dispersion = phonon_freqs(hess_mw, bz, smooth_mat)

    for kvec, freqs in zip(bz, dispersion):
        print " ".join([pretty(k) for k in kvec]),
        print " ".join([pretty(f) for f in freqs])
    if args.binary_file:
        np.savez(args.binary_file, kmesh=bz, freqs=np.array(dispersion))
