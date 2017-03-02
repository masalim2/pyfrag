'''logger module
  0) different logs can go to different files
  1) automatically opens a new file with useful/unique name when invoked for the first time
  2) keeps track of open file handles, closes when application is done
  3) takes input, formats it nicely, writes it to the appropriate file
    log_input(inp) -- the input file or input data for a QM calc
    log_output(output) -- the results and/or errors of a QM calc
    log_gopt
    log_cellopt
    log_MD
    log_PES
  -if superlog: 
     logs calculation input and output (for debugging)
  -if invoked by other drivers (PES or MD or OPTIMIZER), writes pretty logs out to disk

  log_calcs: each calc input and output is appended to a big log file
'''
def log_input(inp):
    pass

def log_output(output):
    pass

def print_bim_e_results(results):
    print "E", results['energy']

def print_vbct_e_results(results):
    print "VBCT Energy Results (Energy/Eigenvector)"
    E_nuclear = results['E_nuclear']
    eigvals = results['eigvals']
    eigvecs = results['eigvecs']
    for val, vec in zip(eigvals, eigvecs.T):
        print ("%12.8f" + len(vec)*"%9.4f") %((val+E_nuclear,)+tuple(vec))
