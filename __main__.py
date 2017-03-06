import argparse
import time

from pyfrag.Globals import params, geom, MPI, logger
from pyfrag.Globals import utility as util

import pyfrag.backend
import pyfrag.bim
import pyfrag.vbct
import pyfrag.drivers 

def get_driver_module(taskname):
    return {
            'bim_e': pyfrag.bim.bim,
            'bim_grad' : pyfrag.bim.bim,
            'bim_numgrad' : pyfrag.drivers.bim_numgrad,
            'bim_hess' : pyfrag.bim.bim,
            'vbct_e' : pyfrag.vbct.energy,
            'bim_opt' : pyfrag.drivers.bim_opt,
            'bim_cellopt' : pyfrag.drivers.bim_cellopt,
            'bim_md' : pyfrag.drivers.bim_md
            }.get(taskname)

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
util.parse_input(args.input_file)
params.verbose = args.verbose
params.quiet = args.quiet
params.qm_logfile = args.qm_logfile

# MAKE SCRATCH DIRS
# -----------------
if 'scrdir' in params.options:
    scratch_top = params.options['scrdir']
else:
    scratch_top = None
util.make_scratch_dirs(scratch_top)

# DISPATCH DRIVER, PRINT RESULTS
# -----------------------------
try:
    task_module = get_driver_module(params.options['task'])
    t_start = time.time()
    results = task_module.kernel()
    t_end = time.time()
    if MPI.rank == 0:
        seconds = t_end - t_start
        print "Task %s done in %s\n" % (params.options['task'], util.pretty_time(seconds))
        print_results = getattr(logger, 'print_%s_results' % params.options['task'])
        print_results(results)
finally:
    util.clean_scratch_dirs()
