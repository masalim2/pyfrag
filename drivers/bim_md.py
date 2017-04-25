'''Molecular dynamics (NVE, NVT, NPT) integration with BIM Forces'''
import traceback
import numpy as np
import h5py
import random
import os
import signal
import time

from pyfrag.Globals import params, geom, MPI, logger
from pyfrag.Globals import lattice as lat
from pyfrag.Globals import utility as util

import pyfrag.backend
import pyfrag.bim


class Integrator:
    '''Contains data and methods for trajectory initialization,
    Velocity Verlet integration (with Nose-Hoover thermostat, Berendensen
    thermostat, and Berendsen barostat), and HDF5 I/O for trajectory storage
    '''

    # Kelvin * k_B --> Energy/a.u.
    K2AU = 3.1668153673851563e-06

    # a.u. --> femtosecond
    AU2FS = 0.0241888432650516

    # amu --> a.u.
    AMU2AU = 1836.15267376

    def __init__(self, forcefield):
        '''Initialize trajectory

        Args:
            forcefield: a method to compute energy/gradient/virial
            for a given geometry (must return "results" dictionary)
        '''
        self.get_MD_options()
        self.forcefield = forcefield

        self.step0 = None
        if self.restart_file:
            self.restart_trajectory_file()
            self.step0 = MPI.bcast(self.step0, master=0)
        else:
            self.step0 = 0
            self.init_velocity()
            self.create_trajectory_file()

        self.grad = None

        geom.set_frag_auto()
        params.options['fragmentation'] = 'fixed'

    def get_MD_options(self):
        '''Set relevant parameters from input file'''
        options      = params.options

        self.restart_file = options.get('md_restart_file')
        self.nstep    = options.get('num_steps', 10)
        self.save_intval  = options.get('save_intval', 1)
        self.dt           = options.get('dt_fs', 1) / self.AU2FS
        self.temp_target  = options.get('temperature', 50)
        self.pres_target  = options.get('pressure_bar', 0)
        self.T_bath_str   = options.get('t_bath', '')
        self.P_bath_str   = options.get('p_bath', '')
        self.nose_tau     = options.get('nose_tau_fs', 40.0)
        self.berend_tau   = options.get('berend_tau_fs', 40.0)

        self.natm = len(geom.geometry)
        self.kin_target = self.K2AU * self.temp_target

        if MPI.rank != 0: return

        if 'berend' in self.P_bath_str.lower():
            self.berend_baro = True
            print "# Using Berendsen Barostat"
            print "#     Target", self.pres_target, "bar"
            print "#     Time constant", self.berend_tau, "fs"
            print ""
        else:
            self.berend_baro = False

        if 'berend' in self.T_bath_str.lower():
            self.berend_thermo = True
            print "# Using Berendsen Thermostat"
            print "#     Target", self.temp_target, "K"
            print "#     Time constant", self.berend_tau, "fs"
            print ""
        else:
            self.berend_thermo = False

        if 'nose' in self.T_bath_str.lower():
            self.nose_init()
            self.nose_thermo = True
            print "# Using Nose-Hoover Chain(M=2) Thermostat"
            print "#     Target", self.temp_target, "K"
            print "#     Time constant", self.nose_tau, "fs"
            print ""
        else:
            self.nose_thermo = False

        assert self.nstep > 0
        assert not (self.berend_thermo and self.nose_thermo)

    def restart_trajectory_file(self):
        '''Load trajectory from hdf5 file and append'''
        if MPI.rank != 0: return

        try:
            self.fp_traj = h5py.File(self.restart_file, 'r+')
        except IOError:
            raise IOError("Cannot open restart file %s" % self.restart_file)

        self.pos_dset = self.fp_traj['pos']
        self.vel_dset = self.fp_traj['vel']
        self.lat_dset = self.fp_traj['lat']

        self.step0  = len(self.pos_dset)
        last_step   = self.step0 - 1
        pos_shape = self.pos_dset[last_step].shape
        zeros = np.zeros(pos_shape)
        while np.allclose(self.pos_dset[last_step], zeros):
            last_step -= 1
        self.step0 = last_step + 1
        print "# Loading trajectory from %s" % self.restart_file
        print "# Previous trajectory had %d non-blank steps" % self.step0
        print "# Loading state from step %d" % last_step
        print ""

        self.vel    = self.vel_dset[last_step]

        atoms = self.pos_dset.attrs['atoms'].split()
        for i in range(self.natm):
            geom.geometry[i].sym = atoms[i]
            geom.geometry[i].pos = self.pos_dset[last_step][i]

        lat.set_lattice(*self.lat_dset[last_step])
        lat.update_lat_vecs()
        newlen = self.step0 + self.nstep

        if 'nose' in self.fp_traj:
            self.nose_dset = self.fp_traj['nose']
            nose_dof = self.nose_dset[last_step]
            self.x1, self.v1, self.x2, self.v2 = nose_dof
            self.nose_dset.resize((newlen, 4))

        self.pos_dset.resize((newlen, self.natm, 3))
        self.vel_dset.resize((newlen, self.natm, 3))
        self.lat_dset.resize((newlen, 7))

    def create_trajectory_file(self):
        '''Create new trajectory in hdf5 file; set handles to data'''
        if MPI.rank != 0: return

        homedir = params.options['home_dir']
        inp_path = params.options['input_filename']
        path, inp_fname  = os.path.split(inp_path)
        basename, ext = os.path.splitext(inp_fname)
        traj_fname = os.path.join(homedir, basename+'.hdf5')
        print "# Storing trajectory data in ", traj_fname

        try:
            self.fp_traj = h5py.File(traj_fname, 'w-')
        except IOError:
            raise IOError("Cannot create trajectory file %s...choose unique name!" % traj_fname)

        self.pos_dset = self.fp_traj.create_dataset("pos",
                (self.nstep, self.natm, 3),dtype=np.double,
                maxshape=(None,3000,3))
        self.lat_dset = self.fp_traj.create_dataset("lat",
                (self.nstep, 7),dtype=np.double,
                maxshape=(None,7))
        self.vel_dset = self.fp_traj.create_dataset("vel",
                (self.nstep, self.natm, 3),dtype=np.double,
                maxshape=(None, 3000, 3))
        if self.nose_thermo:
            self.nose_dset = self.fp_traj.create_dataset("nose",
                    (self.nstep, 4),dtype=np.double,
                    maxshape=(None, 4))
        self.pos_dset.attrs['atoms'] = " ".join(at.sym for at in geom.geometry)
        self.pos_dset.attrs['dt_au'] = self.dt
        self.pos_dset.attrs['dt_fs'] = self.dt * self.AU2FS

    def write_trajectory(self, istep):
        '''Write to hdf5 trajectory file'''
        if MPI.rank != 0: return

        self.pos_dset[istep] = geom.pos_array()
        self.vel_dset[istep] = self.vel
        self.lat_dset[istep] = lat.lattice
        if self.nose_thermo:
            self.nose_dset[istep] = [self.x1, self.v1, self.x2, self.v2]
        self.fp_traj.flush()

    def summary_log(self, istep, wall_seconds):
        '''Print MD summary statistics to stdout'''
        if MPI.rank != 0: return

        fields = "step time/fs energy kinetic potential " \
        "temperature/K pressure/GPa".split()

        virial  = self.force_calc['virial']
        kin_com = self.kinetic_com_tensor()
        volume = lat.volume() * geom.ANG2BOHR**3
        stress = (virial+kin_com) / volume
        pressure = (1./3.)*np.trace(stress)
        pressure = pressure * geom.AU2BAR * 1.0e-4
        time_fs = istep*self.dt*self.AU2FS
        en_pot  = self.force_calc['E']

        values = [istep, time_fs, en_pot+self.kin,
                self.kin, en_pot, self.temp, pressure]

        if self.nose_thermo:
            fields.append('energy_nh')
            e_nh = en_pot + self.kin + 0.5*self.q1*self.v1**2 + \
                    0.5*self.q2*self.v2**2 + \
                    3*self.natm*self.kin_target*self.x1 + \
                    self.kin_target*self.x2
            values.append(e_nh)

        if self.berend_baro:
            fields.append('volume/bohr**3')
            values.append(volume)

        fields.append('walltime')
        values.append(util.pretty_time(wall_seconds))

        if istep == 0:
            print ("@ %5s" + ((len(fields)-1)*"%16s")) % tuple(fields)

        print ("@ %5d" + ((len(fields)-2)*"%16.6f") + "%16s") % tuple(values)


    def init_velocity(self):
        '''Set initial vel according to Maxwell-Boltzmann'''
        if MPI.rank != 0: return

        random.seed()

        vel   = np.zeros((self.natm, 3))
        p_com = np.zeros((3,))

        # Maxwell-Boltzmann
        for i, at in enumerate(geom.geometry):
            mass_i = geom.mass_map[at.sym]*self.AMU2AU
            vscale = (self.kin_target/mass_i)**0.5

            vel[i,0] = vscale * random.gauss(0, 1)
            vel[i,1] = vscale * random.gauss(0, 1)
            vel[i,2] = vscale * random.gauss(0, 1)

            p_com += mass_i*vel[i,:]

        # Set total system momentum to 0
        p_com /= self.natm
        for i, at in enumerate(geom.geometry):
            mass_i = geom.mass_map[at.sym]*self.AMU2AU
            vel[i,:] -= p_com / mass_i


        # Scale kinetic energy to match exact target temperature
        self.vel = vel
        self.update_kinetic_and_temperature()
        vscale = (self.temp_target / self.temp)**0.5
        for i, at in enumerate(geom.geometry):
            self.vel[i,:] *= vscale

        self.update_kinetic_and_temperature()
        print "# initialized temperature to", self.temp
        assert abs(self.temp - self.temp_target) < 1.0e-9

    def update_kinetic_and_temperature(self):
        '''Update kinetic energy & temperature'''
        self.kin_tensor = np.zeros((3,3))

        for i, at in enumerate(geom.geometry):
            mass_i = geom.mass_map[at.sym]*self.AMU2AU
            self.kin_tensor += np.outer(self.vel[i], self.vel[i])*mass_i

        self.kin  = 0.5*np.trace(self.kin_tensor)
        self.temp = 2*self.kin/(3*self.natm*self.K2AU)

    def kinetic_com_tensor(self):
        '''Compute kinetic energy tensor based on fragment
        centers of mass; needed for stress tensor calculation'''
        massvec = np.array([geom.mass_map[at.sym] for at in geom.geometry])
        massvec *= self.AMU2AU
        kin_com = np.zeros((3,3))

        for frag in geom.fragments:
            mass_com = np.sum(massvec[i] for i in frag)
            vel_com  = np.sum((massvec[i]*self.vel[i] for i in frag), axis=0)
            vel_com /= mass_com
            kin_com += mass_com*np.outer(vel_com, vel_com)

        return kin_com

    def integrate(self):
        '''Velocity Verlet with thermo/barostatting'''

        if self.grad is None:
            # step 0: no grad
            self.force_calc = self.forcefield()
            self.grad = self.force_calc['gradient']

        if MPI.rank == 0:
            massvec = np.array([geom.mass_map[at.sym] for at in geom.geometry])
            massvec *= self.AMU2AU

            if self.nose_thermo: self.apply_nose_chain()

            accel     = -self.grad / massvec[:,np.newaxis]
            self.vel += 0.5*accel*self.dt

            pos = geom.pos_array() + self.vel*self.dt/geom.ANG2BOHR
            for i, at in enumerate(geom.geometry):
                at.pos = pos[i]

            if self.berend_baro: self.apply_berend_baro()
            self.detect_and_fix_pbc_crossings()

        geom.geometry = MPI.bcast(geom.geometry, master=0)
        lat.lattice   = MPI.bcast(lat.lattice, master=0)
        lat.lat_vecs  = MPI.bcast(lat.lat_vecs, master=0)

        self.force_calc = self.forcefield()
        self.grad = self.force_calc['gradient']

        if MPI.rank == 0:
            accel     = -self.grad / massvec[:, np.newaxis]
            self.vel += 0.5*accel*self.dt
            if self.berend_thermo: self.apply_berend_thermo()
            if self.nose_thermo: self.apply_nose_chain()
            self.update_kinetic_and_temperature()

    def nose_init(self):
        '''Initialize NH coordinates & masses.'''
        self.x1 = 0.0
        self.g1 = 0.0
        self.v1 = 0.0

        self.x2 = 0.0
        self.g2 = 0.0
        self.v2 = 0.0

        time_const_fs = self.nose_tau
        time_const = time_const_fs / self.AU2FS
        omega = 2*np.pi / time_const
        dof = 3*self.natm

        self.q1 = dof * self.kin_target / omega**2
        self.q2 = self.kin_target / omega**2

    def apply_nose_chain(self):
        '''Half-update NH degrees of freedom together with
        system velocity.'''
        self.update_kinetic_and_temperature()
        dof = 3*self.natm

        self.g2 = (self.q1*self.v1**2 - self.kin_target)/self.q2
        self.v2 += self.g2 * (0.25*self.dt)
        self.v1 *= np.exp(-self.v2*(0.125*self.dt))

        self.g1 = (2*self.kin - dof*self.kin_target)/self.q1

        self.v1 += self.g1*(0.25*self.dt)
        self.v1 *= np.exp(-self.v2*(0.125*self.dt))

        self.x1 += self.v1*(0.5*self.dt)
        self.x2 += self.v2*(0.5*self.dt)

        vscale = np.exp(-self.v1*(0.5*self.dt))
        self.vel *= vscale
        self.kin *= vscale**2
        self.kin_tensor *= vscale**2
        self.temp = 2*self.kin/(3*self.natm*self.K2AU)

        self.v1 *= np.exp(-self.v2*(0.125*self.dt))
        self.g1 = (2*self.kin - dof*self.kin_target)/self.q1
        self.v1 += self.g1*(0.25*self.dt)
        self.v1 *= np.exp(-self.v2*(0.125*self.dt))

        self.g2 = (self.q1*self.v1**2 - self.kin_target) / self.q2
        self.v2 += self.g2 * (0.25*self.dt)

    def apply_berend_thermo(self):
        '''Apply Berendsen velocity scaling for temperature control.
        berend_tau should be at least 10 fs (fast equilibration)
        at tau>100 fs, the fluctuations should be consistent with NVT'''
        # J Chem Phys 81, 3684 (1984)

        tau_fs = self.berend_tau

        self.update_kinetic_and_temperature()
        tau = tau_fs / self.AU2FS

        vscale = (0.5*self.dt/tau)*(self.temp_target/self.temp - 1.0)
        vscale = np.sqrt(1.0 + vscale)
        self.vel *= vscale
        self.kin *= vscale**2
        self.kin_tensor *= vscale**2
        self.temp = 2*self.kin/(3*self.natm*self.K2AU)

    def apply_berend_baro(self):
        '''Apply Berendsen cell scaling for pressure control.
        berend_tau must be > 10 fs, as explained for thermostat.
        '''

        # isothermal compressibility of liquid water
        # J Chem Phys 81, 3684 (1984)
        beta = 4.9e-5 * geom.AU2BAR

        tau_fs = self.berend_tau

        self.update_kinetic_and_temperature()
        tau = tau_fs / self.AU2FS

        virial  = self.force_calc['virial']
        kin_com = self.kinetic_com_tensor()
        volume = lat.volume() * geom.ANG2BOHR**3
        stress = (virial+kin_com) / volume
        stress = 0.5*(stress + stress.T)

        P0 = np.eye(3) * self.pres_target/geom.AU2BAR
        Vscale  = P0 - stress
        Vscale *= beta*self.dt/(3*tau)
        Vscale = np.eye(3) - Vscale

        lat.rescale(Vscale)


    def detect_and_fix_pbc_crossings(self):
        '''translate fragments with COM outside unit cell inside'''
        if not all(lat.PBC_flag): return

        com = np.array([geom.com(frag) for frag in geom.fragments])
        scaled_pos = np.dot(lat.lat_vecs_inv, geom.pos_array().T).T
        scaled_com = np.dot(lat.lat_vecs_inv, com.T).T

        # if any fractional coordinate of COM is outside [0,1]
        for i, frag in enumerate(geom.fragments):
            for mu in range(3):
                for iat in frag:
                    scaled_pos[iat, mu] -= np.floor(scaled_com[i,mu])

        pos = np.dot(lat.lat_vecs, scaled_pos.T).T
        for i, at in enumerate(geom.geometry):
            at.pos = pos[i]

    def clean_up(self, istep):
        '''trim and flush hdf5 trajectory data'''
        if MPI.rank != 0: return
        if self.fp_traj is None: return

        # don't store zeros
        if istep < self.nstep + self.step0:
            self.pos_dset.resize((istep, self.natm, 3))
            self.vel_dset.resize((istep, self.natm, 3))
            self.lat_dset.resize((istep, 7))
            if self.nose_thermo: self.nose_dset.resize((istep, 4))

        self.fp_traj.flush()
        self.fp_traj.close()

def bim_force():
    '''BIM gradients'''
    return pyfrag.bim.bim.kernel()

def hooke_force(k=0.2, r_eq=2.0):
    '''toy potential: harmonic oscillator'''
    results = {}

    r_eq = r_eq*geom.ANG2BOHR
    r1 = geom.geometry[0].pos * geom.ANG2BOHR
    r2 = geom.geometry[1].pos * geom.ANG2BOHR

    r12 = r2 - r1
    r12_norm = np.linalg.norm(r12)
    r12_unit = r12 / r12_norm
    E = 0.5*k*(r12_norm-r_eq)**2
    grad2 = k*(r12_norm-r_eq)*r12_unit
    grad1 = -grad2
    results['gradient'] = np.array([grad1, grad2])
    results['E'] = E
    results['virial'] = np.zeros((3,3))
    return results

def kernel():
    '''MD Main'''

    # catch SIGTERM gracefully
    def signal_handler(*args): md_engine.clean_up(istep)
    signal.signal(signal.SIGTERM, signal_handler)

    params.quiet = True

    md_engine = Integrator(bim_force)

    if MPI.rank == 0:
        logger.print_parameters()
        logger.print_geometry()
        logger.print_fragment()

    geom.geometry = MPI.bcast(geom.geometry, master=0)
    lat.lattice   = MPI.bcast(lat.lattice, master=0)
    lat.lat_vecs  = MPI.bcast(lat.lat_vecs, master=0)

    istep = md_engine.step0
    try:
        while istep < md_engine.step0+md_engine.nstep:
            t_start = time.time()
            md_engine.integrate()
            t_end = time.time()
            seconds = t_end - t_start

            if istep % md_engine.save_intval == 0:
                md_engine.write_trajectory(istep)
            md_engine.summary_log(istep, seconds)
            istep += 1

    except Exception:
        traceback.print_exc()
    finally:
        md_engine.clean_up(istep)
