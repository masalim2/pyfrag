import argparse
import os
import tempfile
from shutil import rmtree
import time

from pyfrag.Globals import params, geom, MPI, logger
import pyfrag.backend
import pyfrag.bim
import pyfrag.vbct
import pyfrag.drivers 


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
    params.options = MPI.bcast(params.options, master=0)
    geom.geometry  = MPI.bcast(geom.geometry, master=0)

def get_driver(taskname):
    return {
            'bim_e': pyfrag.bim.energy,
            'bim_grad' : pyfrag.bim.grad,
            'bim_numgrad' : pyfrag.drivers.bim_numgrad,
            'bim_hess' : pyfrag.bim.hess,
            'vbct_e' : pyfrag.vbct.energy,
            'bim_opt' : pyfrag.drivers.bim_opt,
            'bim_cellopt' : pyfrag.drivers.bim_cellopt,
            'bim_md' : pyfrag.drivers.bim_md
            }.get(taskname)

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

# PARSE COMMAND LINE ARGUMENTS
# ----------------------------
parser = argparse.ArgumentParser(description="Launch fragment calculation "
                                 "with PyFragment")

parser.add_argument("input_file", help="input file to PyFragment")


group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", 
                    help="make drivers print extra debug info")
group.add_argument("-q", "--quiet", action="store_true", 
                    help="make drivers silent (only final result printed)")

parser.add_argument("--qm_logfile", type=str,
                    help="log QM calc input/output to file " 
                    "for heavy debugging")
args = parser.parse_args()

# PARSE INPUT FILE
# ----------------
parse_input(args.input_file)
params.verbose = args.verbose
params.quiet = args.quiet
params.qm_logfile = args.qm_logfile

# MAKE SCRATCH DIRS
# -----------------
if 'scrdir' in params.options:
    scratch_top = params.options['scrdir']
else:
    scratch_top = None
make_scratch_dirs(scratch_top)

# DISPATCH DRIVER, PRINT RESULTS
# -----------------------------
try:
    taskdriver = get_driver(params.options['task'])
    t_start = time.time()
    results = taskdriver.kernel()
    t_end = time.time()
    if MPI.rank == 0:
        seconds = t_end - t_start
        print "Task %s done in %s" % (params.options['task'], pretty_time(seconds))
        print_results = getattr(logger, 'print_%s_results' % params.options['task'])
        print_results(results)
finally:
    clean_scratch_dirs()
