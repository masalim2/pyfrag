from pyfrag.Globals import geom
from pyfrag.Globals import lattice
import numpy as np
import itertools
import sys

cau = (2.99792458*2.418884326505)*1.0e-9  # c in meters per t(a.u.)
amu2au = (1.66053892/9.10938291)*1.0e4  # 1 amu in a.u. (via kgs)
cmtom = 1.0/100.0
scaling = cmtom*(1.0/(2*np.pi*cau*amu2au**0.5))

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
    hess = None
    try:
        data = np.load(fname)
        hess = data['hess']
    except IOError:
        try:
            for nskip in range(4):
                try:
                    hess = np.loadtxt(fname, skiprows=nskip)
                    break
                except ValueError:
                    continue
    if hess is None:
        raise IOError("Could not open hessian")
    return hess

def read_force_consts_txt(fname):

    

if __name__ == "__main__":
    geom.load_geometry(sys.argv[1])
    shift_com_to_origin()
    bz = make_brillouin_zone3D()
    try:
        hess = read_force_consts_np(sys.argv[2])
