import unittest
import sys
import os

from pyfrag.Globals import params, geom, MPI, logger
from pyfrag.Globals import utility as util

import pyfrag.backend
import pyfrag.bim
import pyfrag.vbct
import pyfrag.drivers 

class PyfragTestCase(unittest.TestCase):
    input_file = ''
    def setUp(self):
        util.parse_input(self.input_file)
        if 'scrdir' in params.options:
            scratch_top = params.options['scrdir']
        else:
            scratch_top = None
        util.make_scratch_dirs(scratch_top)
        params['task'] = 'bim_e'

    def tearDown(self):
        util.clean_scratch_dirs()

class TestHFEneregy(PyfragTestCase):
    inp_dir = os.path.split(sys.argv[0])[0]
    input_file = os.path.join(inp_dir, "wat4_hf_ccpvdz.in")
    def test_ccpvdz(self):
        params['basis'] = 'cc-pvdz'
        results = pyfrag.bim.bim()
        self.assertAlmostEqual(results['E1'], 3)
        self.assertAlmostEqual(results['E2'], 3)
        self.assertAlmostEqual(results['Ec'], 3)
        self.assertAlmostEqual(results['E'], 3)

if __name__ == "__main__":
    unittest.main()
