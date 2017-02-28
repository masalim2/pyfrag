import sys
import geometry as geom
import tempfile
import drivers as drv
from shutil import rmtree
import inputdata as inp
import os

# Establish MPI communicator
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    print "Could not import mpi4py. Running serial."
    MPI = None
    comm = None
    rank = 0

# Global MPI data
inp.comm_world = comm
inp.world_rank = rank
    
# Verbosity
if '-v' in ''.join([arg.lower() for arg in sys.argv]):
    inp.VERBOSE = True

# Top scratch directory 
# Subdirs will be rank-private scratch dirs
if 'scratch' in inp.inputdata:
    scratch_top = inp.inputdata['scratch'] 
else:
    scratch_top = None

def parse_input(filename=None):
    # Load input data & broadcast it
    if rank == 0:
        try:
            if filename is None:
                filename = sys.argv[1]
            with open(filename) as inFile:
                inp.parse(inFile)
        except IndexError:
            print "Usage: %s <input-filename> (-v [optional flag for verbosity])" % sys.argv[0]
            sys.exit(0)
        inp.inputdata['geometry'] = geom.load_geometry(inp.inputdata['geometry'])
    if comm:
        inp.inputdata = comm.bcast(inp.inputdata, root=0)

def make_scratch_dirs():
    # rank-private scratch directory (scrdir)
    scrdir = tempfile.mkdtemp(prefix='job%d_' % rank, dir=scratch_top)
    inp.inputdata['scrdir'] = scrdir
    
    # shared scratch files (share_dir)
    share_dir = os.path.join(os.getcwd(), 'shared_temporary_data')
    if not os.path.exists(share_dir) and rank == 0:
        os.makedirs(share_dir)
    inp.inputdata['share_dir'] = share_dir

def clean_scratch_dirs():
    rmtree(inp.inputdata['scrdir'])
    if rank == 0:
        rmtree(inp.inputdata['share_dir'])

if __name__ == "__main__":
    parse_input()
    make_scratch_dirs()

    # Dispatch driver; cleanup even if failure
    taskname = inp.inputdata['task'] if 'task' in inp.inputdata else 'energy'
    taskname = taskname.strip().lower()
    try:
        taskdriver = getattr(drv, '%s_driver' % taskname)
        results = taskdriver()
        if taskname == 'energy' and rank == 0:
            print "Energy Calculation Results (Energy/Eigenvector)"
            E_nuclear = results['E_nuclear']
            eigvals = results['eigvals']
            eigvecs = results['eigvecs']
            for val, vec in zip(eigvals, eigvecs.T):
                print ("%12.8f" + len(vec)*"%9.4f") %((val+E_nuclear,)+tuple(vec))
    finally:
        pass #clean_scratch_dirs()
