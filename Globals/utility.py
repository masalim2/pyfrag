import os
from shutil import rmtree
import tempfile

from pyfrag.Globals import params, geom, MPI, logger, lattice

def pretty_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return "%02dd:%02dh:%02dm:%02ds" % (d,h,m,s)
    elif h > 0:
        return "%02dh:%02dm:%02ds" % (h,m,s)
    else:
        return "%02dm:%02ds" % (m,s)

def make_scratch_dirs(top_dir=None):
    home_dir = os.getcwd()
    params.options['home_dir'] = home_dir
    
    # rank-private scratch directory (scrdir)
    scrdir = tempfile.mkdtemp(prefix='job%d_' % MPI.rank, dir=top_dir)
    params.options['scrdir'] = scrdir
    
    # shared scratch files (share_dir)
    share_dir = os.path.join(home_dir, 'shared_temporary_data')
    if not os.path.exists(share_dir) and MPI.rank == 0:
        os.makedirs(share_dir)
    params.options['share_dir'] = share_dir

def clean_scratch_dirs():
    rmtree(params.options['scrdir'])
    if MPI.rank == 0:
        rmtree(params.options['share_dir'])

def parse_input(input_file):
    if MPI.rank == 0:
        with open(input_file) as fp: params.parse(fp)
        geom.load_geometry(params.options['geometry'])
    params.options  = MPI.bcast(params.options, master=0)
    geom.geometry   = MPI.bcast(geom.geometry, master=0)
    lattice.lattice = MPI.bcast(lattice.lattice, master=0)
    lattice.lat_vecs = MPI.bcast(lattice.lat_vecs, master=0)
    lattice.lat_vecs_inv = MPI.bcast(lattice.lat_vecs_inv, master=0)
    lattice.PBC_flag = MPI.bcast(lattice.PBC_flag, master=0)


def mw_execute(specifiers, run_calc, comm=None, *args):
    '''Run a series of calculations in master-worker mode.

    Args:
        specifiers: a list of tuples which uniquely determine a
            fragment calculation.
        run_calc: the function which accepts a specifier tuple,
            sets up a QM calculation, invokes the backend, and
            returns the relevant calculation results.
        comm: optional subcommunicator, default None: use Globals.MPI
            communicator
    Returns:
        calcs: the dictionary of results indexed by specifiers
    '''
    if comm:
        rank, nproc = comm.Get_rank(), comm.size
    else:
        comm, rank, nproc = MPI.comm, MPI.rank, MPI.comm.size
    calcs = {}
    stat = MPI.MPI.Status()
    if nproc == 1:
        for calc in specifiers:
            calcs[calc] = run_calc(calc, *args)
    elif rank == 0:
        for idx in range(nproc-1, len(specifiers)):
            (calc,result) = comm.recv(source=MPI.MPI.ANY_SOURCE, status=stat)
            calcs[calc] = result
            comm.send(idx, dest=stat.Get_source())
        for idx in range(1, nproc):
            (calc,result) = comm.recv(source=MPI.MPI.ANY_SOURCE, status=stat)
            calcs[calc] = result
            comm.send('DONE', dest=stat.Get_source())
    else:
        idx = rank-1
        if idx < len(specifiers):
            calc = specifiers[idx]
            result = run_calc(calc, *args)
        else:
            calc, result = 'dummy', 'dummy'
        comm.send((calc,result), dest=0)
        idx = comm.recv(source=0)
        while idx != 'DONE':
            calc = specifiers[idx]
            result = run_calc(calc, *args)
            comm.send((calc,result), dest=0)
            idx = comm.recv(source=0)
    return calcs
