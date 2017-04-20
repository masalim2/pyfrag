'''This module contains the globally-shared lat_vecs which defines the
Bravais lattice vectors of the system in a 3x3 ndarray. It comes with supporting
functionality for 3D PBC calculations:
    * updating lattice vectors from lattice parameters (a,b,c,alpha,beta,gamma)
    * the inverse of lat_vecs for transformation to fractional(scaled) coords
    * computing cell volume
    * computing gradient wrt lattice parameters, using virial tensor and applied
      stress
    * rescaling cell by translating fragment centers of mass, while preserving
     fragment internal coordinates
'''

import numpy as np

# Module-level data to be shared
lattice = [0.0, 0.0, 0.0, 90.0, 90.0, 90.0, 0]
lat_vecs = np.zeros((3,3))
lat_vecs_inv = np.eye(3)
PBC_flag = (False, False, False)

def set_lattice(a, b, c, alpha, beta, gamma, axis):
    global lattice
    lattice = [a, b, c, alpha, beta, gamma, axis]

def update_lat_vecs():
    '''Update lattice vectors matrix (3x3, angstrom) from "lattice".

    Invoke this function whenever the lattice variable is changed to recompute
    the lattice vectors. No input arguments or return values.
    '''
    global PBC_flag, lat_vecs, lat_vecs_inv
    a_dim = abs(lattice[0]) > 0.1
    b_dim = abs(lattice[1]) > 0.1
    c_dim = abs(lattice[2]) > 0.1
    PBC_flag = (a_dim, b_dim, c_dim)

    assert lattice[6] == 0, "asssuming a-axis == x-axis && ab is in xy-plane"
    ab, ac, bc = np.deg2rad([lattice[5], lattice[4], lattice[3]])

    lat_vecs[:,0] = np.array([lattice[0], 0.0, 0.0])
    lat_vecs[:,1] = lattice[1] * np.array([np.cos(ab), np.sin(ab), 0.0])
    cx = np.cos(ac)
    cy = ( np.cos(bc) - np.cos(ac)*np.cos(ab) ) / np.sin(ab)
    cz = np.sqrt(1.0 - cx**2 - cy**2)
    lat_vecs[:,2] = lattice[2] * np.array([cx, cy, cz])

    lat_vecs_inv = np.eye(3)
    if all(PBC_flag):
        lat_vecs_inv = np.linalg.inv(lat_vecs)
    elif PBC_flag[0] and PBC_flag[1]:
            lat_vecs_inv[0:2, 0:2] = np.linalg.inv(lat_vecs[0:2, 0:2])
    elif PBC_flag[0]:
        lat_vecs_inv[0,0] = 1.0/lat_vecs[0,0]

def update_lat_params():
    '''Update lattice parameters NamedTuple from "lat_vecs"

    The inverse of update_lat_vecs; recomputes the lattice parameters based on
    the current vectors. Lattice parameter units are angstrom/degree.'''
    global lattice
    a = np.linalg.norm(lat_vecs[:,0])
    b = np.linalg.norm(lat_vecs[:,1])
    c = np.linalg.norm(lat_vecs[:,2])
    gamma = np.arccos(np.dot(lat_vecs[:,0], lat_vecs[:,1])/(a*b))
    beta  = np.arccos(np.dot(lat_vecs[:,0], lat_vecs[:,2])/(a*c))
    alpha = np.arccos(np.dot(lat_vecs[:,1], lat_vecs[:,2])/(b*c))
    gamma, beta, alpha = np.rad2deg([gamma, beta, alpha])

    lattice = [a, b, c, alpha, beta, gamma, 0]

def volume():
    '''Return volume in Angstrom**3

    3D case: return the determinant of lat_vecs matrix
    For 2D,1D,0D systems: return cell area, length, unity,
    respectively.
    '''
    if all(PBC_flag):
        return np.linalg.det(lat_vecs)
    elif PBC_flag[0] and PBC_flag[1]:
        return np.linalg.det(lat_vecs[0:2,0:2])
    elif PBC_flag[0]:
        return lat_vecs[0,0]
    else:
        return 1.0

def lat_angle_differential():
    '''Compute partial derivatives of lat_vecs with respect to change in
    lattice angle parameter.

    Lazy; this is just a quick finite difference calculation
    Args:
        None
    Returns:
        Three 3x3 numpy arrays. They contain the derivatives
        of the lattice vector components with respect to alpha,
        beta, and gamma, respectively. Units of each matrix element are
        Angstroms/degree.'''

    # Compute vecs with slightly displaced angle
    def compute_lat_vecs(a, b, c, alpha, beta, gamma):
        vecs = np.zeros((3,3))
        ab, ac, bc = np.deg2rad([gamma, beta, alpha])

        vecs[:,0] = np.array([a, 0.0, 0.0])
        vecs[:,1] = b * np.array([np.cos(ab), np.sin(ab), 0.0])
        cx = np.cos(ac)
        cy = ( np.cos(bc) - np.cos(ac)*np.cos(ab) ) / np.sin(ab)
        cz = np.sqrt(1.0 - cx**2 - cy**2)
        vecs[:,2] = c * np.array([cx, cy, cz])
        return vecs

    # Finite difference: small angle displacement
    DELTA = 1.0e-4
    dalpha = compute_lat_vecs(lattice[0], lattice[1], lattice[2],
                          lattice[3]+DELTA, lattice[4], lattice[5])
    dbeta = compute_lat_vecs(lattice[0], lattice[1], lattice[2],
                          lattice[3], lattice[4]+DELTA, lattice[5])
    dgamma = compute_lat_vecs(lattice[0], lattice[1], lattice[2],
                          lattice[3], lattice[4], lattice[5]+DELTA)

    dalpha = (dalpha - lat_vecs) / DELTA
    dbeta  = (dbeta  - lat_vecs) / DELTA
    dgamma = (dgamma - lat_vecs) / DELTA
    return dalpha, dbeta, dgamma

def lattice_gradient(virial, p0_bar):
    '''Compute the energy gradient in lattice parameters (a.u. / bohr)

    Args:
        virial: 3x3 numpy array, from gradient calculation
        p0_bar: external pressure in bar
    Returns:
        lat_grad: 6-dimensional gradient vector in au/bohr, au/degrees
    '''
    from pyfrag.Globals import geom
    vol = volume() * geom.ANG2BOHR**3
    stress = virial / vol
    stress -= np.eye(3) * p0_bar/geom.AU2BAR
    stress = 0.5*(stress + stress.T)

    g_lat = np.dot(stress, lat_vecs_inv.T/geom.ANG2BOHR)
    g_lat = -vol*g_lat

    dalpha, dbeta, dgamma = lat_angle_differential()
    lat_grad = np.zeros((6,))

    lat_grad[0] = np.dot(g_lat[:,0], lat.lat_vecs[:,0]/lat.lattice[0])
    lat_grad[1] = np.dot(g_lat[:,1], lat.lat_vecs[:,1]/lat.lattice[1])
    lat_grad[2] = np.dot(g_lat[:,2], lat.lat_vecs[:,2]/lat.lattice[2])

    lat_grad[3] = np.sum(g_lat*geom.ANG2BOHR*dalpha)
    lat_grad[4] = np.sum(g_lat*geom.ANG2BOHR*dbeta)
    lat_grad[5] = np.sum(g_lat*geom.ANG2BOHR*dgamma)

    if not lat.PBC_flag[0]:
        lat_grad[0] = 0.0
    if not lat.PBC_flag[1]:
        lat_grad[1] = 0.0
    if not lat.PBC_flag[2]:
        lat_grad[2] = 0.0
    if not all(lat.PBC_flag):
        lat_grad[3] = 0.0
        lat_grad[4] = 0.0
        lat_grad[5] = 0.0

    return lat_grad

def rescale(scaling):
    '''Rescale the lattice vectors and shift fragment centers of mass to
    preserve scaled COM coordinates while maintaining internal fragment
    coordinates.

    Args:
        scaling: either a 3x3 lattice vector transformation matrix (ndarray)
          or a list of lattice parameters (a,b,c,alpha,beta,gamma,axis) in
          Angstrom.
    Returns:
        None.
    '''
    from pyfrag.Globals import geom
    global lattice, lat_vecs, lat_vecs_inv
    frags = geom.fragments

    com_coords  = np.array([geom.com(frag) for frag in frags])
    scal_coords = np.zeros(com_coords.shape)
    atom_coords = geom.pos_array()

    # to scaled COM/internal coordinates
    for i, frag in enumerate(frags):
        for iat in frag:
            atom_coords[iat] -= com_coords[i]
        scal_coords[i] = np.dot(lat_vecs_inv, com_coords[i])

    # alter lattice vectors
    if isinstance(scaling, np.ndarray) and scaling.shape == (3,3):
        lat_vecs = np.dot(scaling, lat_vecs)
        update_lat_params()
        update_lat_vecs()
    else:
        assert len(scaling) == 7
        set_lattice(*scaling)
        update_lat_vecs()

    # back to cartesian coordinates
    for i, frag in enumerate(frags):
        com_coords[i] = np.dot(lat_vecs, scal_coords[i])
        for iat in frag:
            atom_coords[iat] += com_coords[i]
            geom.geometry[iat].pos = atom_coords[iat]
