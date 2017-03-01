import numpy as np

# Module-level data to be shared
LatticeTuple = namedtuple('Lattice', 
        'a b c alpha beta gamma axis'])
lattice = LatticeTuple(0.0, 0.0, 0.0, 90.0, 90.0, 90.0, 0)
lat_vecs = np.zeros((3,3))
lat_vecs_inv = np.zeros((3,3))
PBC_flag = (False, False, False)

def update_lat_vecs():
    '''Update lattice vectors matrix (3x3, angstrom) from "lattice".

    Invoke this function whenever the lattice variable is changed to recompute
    the lattice vectors. No input arguments or return values.
    '''
    global PBC_flag, lat_vecs, lat_vecs_inv
    a_dim = abs(lattice.a) > 0.1
    b_dim = abs(lattice.b) > 0.1
    c_dim = abs(lattice.c) > 0.1
    PBC_flag = (a_dim, b_dim, c_dim)

    assert lattice.axis == 0, "asssuming a-axis == x-axis && ab is in xy-plane"
    ab, ac, bc = np.deg2rad([lattice.gamma, lattice.beta, lattice.alpha])

    lat_vecs[:,0] = np.array([lattice.a, 0.0, 0.0])
    lat_vecs[:,1] = lattice.b * np.array([np.cos(ab), np.sin(ab), 0.0])
    cx = np.cos(ac)
    cy = ( np.cos(bc) - np.cos(ac)*np.cos(ab) ) / np.sin(ab)
    cz = np.sqrt(1.0 - cy**2 - cz**2)
    lat_vecs[:,2] = lattice.c * np.array([cx, cy, cz])

    lat_vecs_inv = np.linalg.inv(lat_vecs)

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

    lattice = LatticeTuple(a, b, c, alpha, beta, gamma, 0)

def volume():
    '''Return volume in Angstrom**3 
    
    Simply the determinant of lat_vecs matrix'''
    return np.linalg.det(lat_vecs)

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
        cz = np.sqrt(1.0 - cy**2 - cz**2)
        vecs[:,2] = c * np.array([cx, cy, cz])
        return vecs

    # Finite difference: small angle displacement
    DELTA = 1.0e-4
    dalpha = compute_lat_vecs(lattice.a, lattice.b, lattice.c,
                          lattice.alpha+DELTA, lattice.beta, lattice.gamma)
    dbeta = compute_lat_vecs(lattice.a, lattice.b, lattice.c,
                          lattice.alpha, lattice.beta+DELTA, lattice.gamma)
    dgamma = compute_lat_vecs(lattice.a, lattice.b, lattice.c,
                          lattice.alpha, lattice.beta, lattice.gamma+DELTA)

    dalpha = (dalpha - lat_vecs) / DELTA
    dbeta  = (dbeta  - lat_vecs) / DELTA
    dgamma = (dgamma - lat_vecs) / DELTA
    return dalpha, dbeta, dgamma
