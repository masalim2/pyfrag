from pyfrag.backend import backend
from pyfrag.Globals import MPI
from pyfrag.Globals import params
import numpy as np 

class ChargeState:
    '''Base class for VB CT state'''

    def __init__(self, fragments, fragment_charges):
        self.label = zip(fragments, fragment_charges)
        
        self.correlation = params.options['correlation']
        self.embedding   = params.options['embedding']
        self.geometry    = params.options['geometry']

        # Choose Fragment backend & calculation methods based on input
        for option in ['backend', 'diagonal', 'coupling']:
            setattr(self, option, params.options[option])

        self.Frag = getattr(Fragment, '%sFragment'%self.backend.upper())
        self.diagonal = getattr(self, 'diag_%s'%self.diagonal)
        self.coupling = getattr(self, 'coupling_%s'%self.coupling)

        self.monomers = [self.Frag([self.geometry[iat] for iat in frag], chg, idx) 
                         for idx, (frag,chg) in enumerate(self.label)]

        self.dimers = [self.monomers[i]+self.monomers[j] for i in
                range(len(self.monomers)-1) for j in range(i+1, len(self.monomers))]


    def diag_chargelocal_dimers(self, subcomm=None):
        '''return E1 + E2, the BIM energy of this charge-transfer configuration
           E1: sum of monomer correlated energies
           E2: sum of dimer interaction energies (E_AB - E_A - E_B)
               interaction INCLUDES correlation for relaxed dimers (when A&B have same charge)
               but it's just the non-stationary HF energy for charge-local dimers'''

        # MPI Scatter fragments to calculate
        if subcomm:
            rank = subcomm.Get_rank()
            nproc = subcomm.size
        else:
            subcomm, rank, nproc = None, 0, 1

        my_monomers = MPI.scatter(subcomm, self.monomers, master=0)
        my_dimers = MPI.scatter(subcomm, self.dimers, master=0)

        # Monomers
        for mono in my_monomers:
            if self.embedding:
                mono.embedding_field = [(at, chg) for m in self.monomers if m !=
                        mono for (at, chg) in zip(m.atoms, m.esp_charges)]
            else:
                mono.embedding_field = []
            if 'hf' not in mono.energy or self.correlation: 
                mono.run(calc='energy', input_vecs=mono.movecs_path)

        # Dimers (relaxed and charge-local)
        for dimer in my_dimers:
            if self.embedding:
                dimer.embedding_field = [(at, chg) for m in self.monomers if m.index
                        not in dimer.index for (at, chg) in zip(m.atoms, m.esp_charges)]
            else:
                dimer.embedding_field = []

            idx1, idx2 = dimer.index
            assert idx1 < idx2
            if self.monomers[idx1].charge != self.monomers[idx2].charge:
                charge_local_flag = True
                calc_type = 'energy_hf'
            else:
                charge_local_flag = not params.options['relax_neutral_dimers']
                if params.options['corr_neutral_dimers']:
                    calc_type = 'energy'
                else:
                    calc_type = 'energy_hf'

            dimer.run(calc=calc_type, noscf=charge_local_flag,
                input_vecs=(self.monomers[idx1].movecs_path, 
                self.monomers[idx2].movecs_path))
        
        # Gather evaluated fragments, compute state diagonal element
        self.monomers = MPI.allgather(subcomm, my_monomers)
        self.dimers =   MPI.allgather(subcomm,   my_dimers)

        self.E1 = sum([mono.energy['total'] for mono in self.monomers])
        self.E2 = 0.0
        for dimer in self.dimers:
            i, j = dimer.index
            if 'corr' in dimer.energy:
                self.E2 += dimer.energy['total'] \
                    - self.monomers[i].energy['total'] \
                    - self.monomers[j].energy['total']
            else:
                self.E2 += dimer.energy['hf'] \
                    - self.monomers[i].energy['hf'] \
                    - self.monomers[j].energy['hf']
        return self.E1 + self.E2
    
    def diag_mono_ip(self, subcomm=None):
        import copy
        '''return E1 + E2, the BIM energy of this charge-transfer configuration
           E1: sum of monomer correlated energies
           E2: sum of dimer interaction energies (E_AB - E_A - E_B)
               interaction INCLUDES correlation for relaxed dimers (when A&B have same charge)
               but it's just the non-stationary HF energy for charge-local dimers'''

        # MPI Scatter fragments to calculate
        if subcomm:
            rank = subcomm.Get_rank()
            nproc = subcomm.size
        else:
            subcomm, rank, nproc = None, 0, 1
        if nproc != 1:
            raise RuntimeError("Testing method, not yet parallel executable")
        if params.options['coupling'] != 'mono_ip':
            raise RuntimeError("Incompatible with coupling %s" % params.options['coupling'])

        my_monomers = MPI.scatter(subcomm, self.monomers, master=0)
        my_dimers = MPI.scatter(subcomm, self.dimers, master=0)

        # Monomers
        for mono in my_monomers:
            if self.embedding:
                raise RuntimeError('Embedding incompatible with monoIP method')
            else:
                mono.embedding_field = []
            if 'hf' not in mono.energy or self.correlation:
                mono.run(calc='energy', input_vecs=mono.movecs_path)
        charged_monomers = [m for m in my_monomers if m.charge != 0]
        assert len(charged_monomers) == 1
        mono_chg = charged_monomers[0]
        mono_neu = copy.deepcopy(mono_chg)
        mono_neu.charge = 0
        mono_neu.run(calc='energy')

        for dimer in my_dimers:
            if self.embedding:
                raise RuntimeError('Embedding incompatible with monoIP method')
            else:
                dimer.embedding_field = []
            dimer.charge = 0
            dimer.run(calc='energy')
        
        # Gather evaluated fragments, compute state diagonal element
        self.monomers = MPI.allgather(subcomm, my_monomers)
        self.dimers =   MPI.allgather(subcomm,   my_dimers)

        # sum of neutral monomers
        self.E1 = sum([mono.energy['total'] for mono in self.monomers])
        self.E1 -= mono_chg.energy['total']
        self.E1 += mono_neu.energy['total']

        # sum of neutral dimer interactions
        self.E2 = 0.0
        for dimer in self.dimers:
            i, j = dimer.index
            self.E2 += dimer.energy['total'] \
                - self.monomers[i].energy['total'] \
                - self.monomers[j].energy['total']
            if self.monomers[i].charge != 0:
                self.E2 += self.monomers[i].energy['total']
                self.E2 -= mono_neu.energy['total']
                assert mono_neu.index == self.monomers[i].index
            if self.monomers[j].charge != 0:
                self.E2 += self.monomers[j].energy['total']
                self.E2 -= mono_neu.energy['total']
                assert mono_neu.index == self.monomers[j].index
        # Ionize monomer
        self.E1 += mono_chg.energy['total'] - mono_neu.energy['total']
        return self.E1 + self.E2

    def coupling_dimer_gs(self, state2, embed_flag=None):
        ''' -sqrt( [E_(AB)+ - E_(A+B)]*[E_(AB)+ - E_(AB+)])
        E_(AB)+ == the relaxed, correlated charged dimer
        E_(A+B) == non-stationary HF energy of the charge-local dimer
                   plus monomer correlation energies of E_A+ and E_B
        For a dimer system, this method reproduces the exact E by construction'''

        info = {}

        reactants = []
        products  = []

        # identify charge-transferring dimer
        for mono1, mono2 in zip(self.monomers, state2.monomers):
            if mono1 == mono2 and mono1.charge != mono2.charge:
                reactants.append(mono1)
                products.append(mono2)
        assert len(reactants) == 2 and len(products) == 2
        info['stateA'] = [(m.label(), m.index) for m in reactants]
        info['stateB'] = [(m.label(), m.index) for m in products]

        # do monomer SCF on the isolated dimer in both charge-transfer states
        self.monomerSCF(reactants, embedding=embed_flag)
        self.monomerSCF(products, embedding=embed_flag)
        
        # E_(A+B)
        reactants_dimer  = reactants[0] + reactants[1]
        reactants_dimer.run(calc='energy_hf', noscf=True,
                input_vecs=(reactants[0].movecs_path,
                    reactants[1].movecs_path))
        E_react = reactants_dimer.energy['hf']
        info['E_(A+B)'] = E_react

        if self.correlation:
            info['Ecorr_(A+B)'] = 0.0
            for r in reactants:
                r.run(calc='energy', input_vecs=r.movecs_path)
                E_react += r.energy['corr']
                info['Ecorr_(A+B)'] += r.energy['corr']

        # E_(AB+)
        products_dimer   =  products[0] +  products[1]
        products_dimer.run(calc='energy_hf', noscf=True,
                input_vecs=(reactants[0].movecs_path,
                    reactants[1].movecs_path))
        E_product = products_dimer.energy['hf']
        info['E_(AB+)'] = E_product

        if self.correlation:
            info['Ecorr_(AB+)'] = 0.0
            for p in products:
                p.run(calc='energy', input_vecs=p.movecs_path)
                E_product += p.energy['corr']
                info['Ecorr_(AB+)'] += p.energy['corr']

        # E_(AB)+ -- get lowest scf energy + correlation
        products_dimer.run(calc='energy', noscf=False)
        E_relaxed =  products_dimer.energy['total']
        info['Etot_(AB)+'] = E_relaxed

        assert (E_relaxed <= E_react) and (E_relaxed <= E_product)
        coupling = -1.0*((E_relaxed-E_react)*(E_relaxed-E_product))**0.5
        overlap = 0.0
        return coupling, overlap, info

    def coupling_dimer_gs_no_embed(self, state2):
        '''Wraps the above dimer_gs coupling, but the two monomers do not
        polarize each other in monomerSCF. Hence, it does not exactly
        reproduce the dimer GS energy by construction'''
        return self.coupling_dimer_gs(state2, embed_flag=False)

    def coupling_mono_ip(self, state2):
        import copy
        if self.embedding:
            raise RuntimeError('Embedding incompatible with monoIP method')
        if params.options['diagonal'] != 'mono_ip':
            raise RuntimeError('monoIP coupling incompatible with %s' %
                    params.options['diagonal'])

        info = {}
        reactants = []
        products  = []

        # identify charge-transferring dimer
        for mono1, mono2 in zip(self.monomers, state2.monomers):
            if mono1 == mono2 and mono1.charge != mono2.charge:
                reactants.append(mono1)
                products.append(mono2)
        print self
        print state2
        assert len(reactants) == 2 and len(products) == 2
        info['stateA'] = [(m.label(), m.index) for m in reactants]
        info['stateB'] = [(m.label(), m.index) for m in products]

        # need (AB) (AB)+ (A)+ (B)+ (A) (B)
        neutral_dimer  = reactants[0] + reactants[1]
        neutral_dimer.charge = 0
        neutral_dimer.run(calc='energy')

        charged_dimer  = reactants[0] + reactants[1]
        charged_dimer.charge = reactants[0].charge + reactants[1].charge
        assert charged_dimer.charge == 1
        charged_dimer.run(calc='energy')

        A0 = copy.deepcopy(reactants[0])
        A0.charge = 0
        A0.run(calc='energy')

        B0 = copy.deepcopy(reactants[1])
        B0.charge = 0
        B0.run(calc='energy')

        A1 = copy.deepcopy(products[0])
        A1.charge = 1
        A1.run(calc='energy')

        B1 = copy.deepcopy(products[1])
        B1.charge = 1
        B1.run(calc='energy')
        
        EAB = neutral_dimer.energy['total']
        EABp = charged_dimer.energy['total']
        EA = A0.energy['total']
        EAp = A1.energy['total']
        EB = B0.energy['total']
        EBp = B1.energy['total']

        assert EAp > EA
        assert EBp > EB
        assert EABp > EAB

        relaxA = EABp - (EAB+EAp-EA)
        relaxB = EABp - (EAB+EBp-EB)
        coupling = -1.0*(relaxA * relaxB)**0.5
        overlap = 0.0
        return coupling, overlap, info

    def coupling_dimer_gs_overlapHOMO(self, state2):
        '''S*E_(AB)+ - sqrt( [E_(AB)+ - E_(A+B)]*[E_(AB)+ - E_(AB+)])
           Second term is same as coupling_dimer_gs
           First term is overlap of monomer HOMO's times relaxed dimer energy
           This method also reproduces energy of a dimer by construction; it
           just makes the overlap matrix non-identity'''
        pass

           

    def __repr__(self):
        labels = [m.label() for m in self.monomers]
        return 'State:' + ''.join(labels)
