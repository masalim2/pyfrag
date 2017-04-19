from pyfrag.Globals import geom
from pyfrag.Globals import lattice
import numpy as np
import itertools
import sys

cau = (2.99792458*2.418884326505)*1.0e-9  # c in meters per t(a.u.)
amu2au = (1.66053892/9.10938291)*1.0e4  # 1 amu in a.u. (via kgs)
cmtom = 1.0/100.0
SCALING = cmtom*(1.0/(2*np.pi*cau*amu2au**0.5))

def shift_com_to_origin():
    com = sum([at.pos for at in geom.geometry])
    com = com / len(geom.geometry)
    for at in geom.geometry:
        at.pos -= com

def make_brillouin_zone3D():
    '''Brute-force generate uniform grid inside 1st Brillouin zone.

    Brute force but clean approach. Truncates a uniform rectangular grid;
    therefore, nonorthogonal cells may require a high density of points to
    sample adequately near the zone boundaries.
    '''

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
    xp, yp, zp = (np.linspace(boxmin[i], boxmax[i], 21) for i in range(3))
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
    assert np.allclose(d, np.conj(d.T))
    return d

def phonons(hess_mw, kmesh, smooth_mat=None):
    freqs = []
    vecs = []
    for kvec in kmesh:
        print "phonons at k", kvec
        d = Dmat(hess_mw, kvec, smooth_mat)
        eigvals, eigvecs = np.linalg.eigh(d)
        freqs.append(np.sign(eigvals)*SCALING*
                np.sqrt(np.sign(eigvals)*eigvals))
        for col in eigvecs.T:
            col *= np.exp(-1j*np.angle(col[0]))
        vecs.append(eigvecs)
    return freqs, vecs

def pretty(x):
    return "%9.3f" % x


if __name__ == "__main__":
    geom.load_geometry(sys.argv[1])
    shift_com_to_origin()
    bz = make_brillouin_zone3D()
    hess = read_force_consts_txt(sys.argv[2])
    hess_mw = mass_weight(hess)
    freqs, vecs = phonons(hess_mw, bz)

    for kvec, freqs in zip(bz, freqs):
        print " ".join([pretty(k) for k in kvec]),
        print " ".join([pretty(f) for f in freqs])
