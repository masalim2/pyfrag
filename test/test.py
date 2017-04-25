'''Unit Tests using Python unittest framework

Run all tests from command line using:

>>> python -m unittest pyfrag/test

Run one test from command line by specifying a specific test case:

>>> python -m unittest pyfrag/test.TestTrimerRHF

Add new test cases by creating new classes that extend unittest.TestCase:
'''

import unittest
import sys
import os
import numpy as np

from pyfrag.Globals import params, geom, lattice
from pyfrag.Globals import utility as util

import pyfrag.backend
import pyfrag.bim
import pyfrag.vbct
import pyfrag.drivers

filename =  sys.modules[__name__].__file__
testpath, b  = os.path.split(filename)

class TestTrimerRHF(unittest.TestCase):
    '''Test RHF energy and gradient on water trimer'''

    @classmethod
    def setUpClass(cls):
        inpath   = os.path.join(testpath, 'inputs/wat3_hf.inp')
        util.parse_input(inpath)
        params.quiet = True
        util.make_scratch_dirs(None)

    @classmethod
    def tearDownClass(cls):
        util.clean_scratch_dirs()

    def testInput(self):
        self.assertEqual(params.options['mem_mb'], 3800)

        self.assertEqual(len(geom.geometry),  9)
        self.assertEqual(sorted([idx for frag in geom.fragments
                         for idx in frag]), range(9))

        opts = params.options

        if opts['fragmentation'] == 'auto':
            self.assertEqual(len(geom.fragments), 3)
        elif opts['fragmentation'] == 'full_system':
            self.assertEqual(len(geom.fragments), 1)

        self.assertEqual(opts['hftype'], 'rohf')
        self.assertEqual(opts['basis'],  '6-31g')
        self.assertEqual(opts['correlation'], False)
        self.assertEqual(lattice.PBC_flag, (False,False,False))

    def testEnergy(self):
        driver = pyfrag.bim.bim
        params.options['task'] = 'bim_e'
        params.options['fragmentation'] = 'full_system'
        results_exact = driver.kernel()

        params.options['fragmentation'] = 'auto'
        results_frag = driver.kernel()

        E_exact = results_exact['E']
        E_frag  =  results_frag['E']

        self.assertAlmostEqual(E_exact, E_frag, delta=0.001)

    def testGradient(self):
        driver = pyfrag.bim.bim
        params.options['task'] = 'bim_grad'
        params.options['fragmentation'] = 'full_system'
        results_exact = driver.kernel()

        params.options['fragmentation'] = 'auto'
        results_frag = driver.kernel()

        gexact = results_exact['gradient']
        gfrag  =  results_frag['gradient']

        self.assertLess(np.max(np.abs(gfrag-gexact)), 0.0005)

class TestTrimerMP2(unittest.TestCase):
    '''Test MP2 gradients on water trimer with both NW/Psi4 backends'''

    @classmethod
    def setUpClass(cls):
        inpath = os.path.join(testpath, 'inputs/wat3_mp2.inp')

        util.parse_input(inpath)
        params.quiet = True
        util.make_scratch_dirs(None)

    @classmethod
    def tearDownClass(cls):
        util.clean_scratch_dirs()

    def testGradNW(self):
        E_exact = np.loadtxt(os.path.join(testpath,
            'outputs/wat3_mp2.energy'))
        g_exact = np.loadtxt(os.path.join(testpath,
            'outputs/wat3_mp2.grad'))

        driver = pyfrag.bim.bim
        params.options['task'] = 'bim_grad'
        params.options['fragmentation'] = 'auto'
        params.options['backend'] = 'nw'
        results = driver.kernel()

        E_frag  = results['E']
        g_frag  =  results['gradient']

        self.assertAlmostEqual(E_exact, E_frag, delta=0.001)
        self.assertLess(np.max(np.abs(g_frag-g_exact)), 0.001)

    def testGradPSI(self):
        E_exact = np.loadtxt(os.path.join(testpath,
            'outputs/wat3_rimp2.energy'))
        g_exact = np.loadtxt(os.path.join(testpath,
            'outputs/wat3_mp2.grad'))

        driver = pyfrag.bim.bim
        params.options['task'] = 'bim_grad'
        params.options['fragmentation'] = 'auto'
        params.options['backend'] = 'psi4'
        results = driver.kernel()

        E_frag  = results['E']
        g_frag  =  results['gradient']

        self.assertAlmostEqual(E_exact, E_frag, delta=0.003)
        self.assertLess(np.max(np.abs(g_frag-g_exact)), 0.001)
