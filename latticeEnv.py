import topy
import numpy as np

class Domain(object):
    def __init__(self, domainSz = np.array([128, 96]), cellSz = np.array([16, 16])):
        self.domainSz = domainSz
        self.cellSz = cellSz
        self.paramsDim = 1 # number of parameters for each cell
        self.paramsMin = 1
        self.paramsMax = 7
        self.paramsDel = 1

        self.numCells = (self.domainSz//self.cellSz).astype(int) # TODO: ensure domainSz is divisible by cellSz
        
        # TODO: fix for more than 1 paramsDim
        self.params = ((self.paramsMax+self.paramsMin)/2.)*np.ones(np.append(self.numCells, self.paramsDim))
        
        self.bin_img = np.zeros(self.domainSz, dtype = bool)[:, :, np.newaxis] # Shape in the domain
        self.dispX = np.zeros(self.domainSz + 1)[:, :, np.newaxis] # Node displacement X
        self.dispY = np.zeros(self.domainSz + 1)[:, :, np.newaxis] # Node displacement Y
        self.comp = np.zeros(self.domainSz)[:, :, np.newaxis] # Compliance
        
        self.volFrac = 0.
        
    def minimal_tpd_dict(self):
        '''
        Creates and returns a minimal ToPy Problem Definition
        '''
        tpd_dict = {
            'PROB_TYPE': 'comp',
            'PROB_NAME': 'tpd_test',
            'ETA': '0.5',
            'DOF_PN': 2,
            'VOL_FRAC': 0.5,
            'FILT_RAD': 1.5,
            'P_FAC': 3.0,
            'ELEM_K': 'Q4',
            'NUM_ELEM_X': 60,
            'NUM_ELEM_Y': 20,
            'NUM_ELEM_Z': 0,
            'NUM_ITER': 10,
            'FXTR_NODE_X': '1|21',
            'FXTR_NODE_Y': '1281',
            'LOAD_NODE_Y': '1',
            'LOAD_VALU_Y': '-1'}
        return tpd_dict
    
    def passive_elems(self):
        '''
        Returns a sorted list of linear indices of pixels where the input image is False
        '''
        return np.sort(np.ravel_multi_index(np.where(self.bin_img == False), self.bin_img.shape))
        
    def getCellImg(self, param):
        x, y = np.meshgrid(np.array(range(self.cellSz[0])) + 1./2, np.array(range(self.cellSz[1])) + 1./2, indexing='ij')
        cell = np.sqrt((x - self.cellSz[0]/2)**2 + (y - self.cellSz[1]/2)**2) >= param
        return cell[:, :, np.newaxis]
    
    def setDomainImage(self):
        # TODO: think about making this more efficient by doing vectorized computations without loops
        for i in range(self.numCells[0]):
            for j in range(self.numCells[1]):
                self.bin_img[i*self.cellSz[0]:(i+1)*self.cellSz[0],
                             j*self.cellSz[1]:(j+1)*self.cellSz[1], :] \
                = self.getCellImg(self.params[i, j])
        self.volFrac = np.count_nonzero(self.bin_img)*1./self.bin_img.size
        
    # TODO: randomize intial state VERY IMPORTANT
    def computeState(self):
        '''
        computes the full current state including FEA results
        '''
        
        config = self.minimal_tpd_dict()
        config['PROB_NAME'] = 'sph_lat_test'
        config['NUM_ELEM_X'] = self.domainSz[0]
        config['NUM_ELEM_Y'] = self.domainSz[1]
        config['VOL_FRAC'] = self.volFrac
        config['NUM_ITER'] = 2
        config['PASV_ELEM'] = self.passive_elems()
        config['FXTR_NODE_Y'] = '1|97'
        config['FXTR_NODE_X'] = '1|97'
        config['LOAD_NODE_Y'] = '12417'
        config['LOAD_VALU_Y'] = '-1'
        
        t = topy.Topology()
        t.load_config_dict(config)
        t.set_top_params()
        
        t.update_desvars_oc()
        t.fea()
        t.sens_analysis()
        
        disp = t.d.reshape((self.domainSz[0]+1, self.domainSz[1]+1, 2, 1))
        
        self.dispX = disp[:, :, 0, :] # TODO: change displacement size
        self.dispY = disp[:, :, 1, :]
        
        self.comp = t.qkq.T
        self.comp = self.comp[:, :, np.newaxis]
        
    def _update(self, action):
        delta = self.paramsDel * np.sign(np.around(action))
        self.params = np.clip(self.params + delta, self.paramsMin, self.paramsMax)
        self.setDomainImage()
        

class Lattice(object):
    def __init__(self, domainSz = np.array([128, 96]), cellSz = np.array([16, 16])):
        self.domain = Domain(domainSz, cellSz)
        
        self.domain.setDomainImage()
        self.domain.computeState()
        
        self.state_shapes = (self.domain.bin_img.shape,
                             self.domain.dispX.shape,
                             self.domain.dispY.shape,
                             self.domain.comp.shape)
        self.action_size = np.prod(self.domain.numCells)
        
    def reset(self):
        self.domain = Domain(self.domain.domainSz, self.domain.cellSz)
        
        self.domain.setDomainImage()
        self.domain.computeState() # TODO (IMP) randomize initial state
        return self.domain
            
    def step(self, action, compute = False):
        assert(self.domain.params.size == action.size)
        
        VF = self.domain.volFrac
        self.domain._update(action.reshape(self.domain.params.shape))
        if compute:
            self.domain.computeState()
        delVF = self.domain.volFrac - VF
        
        # temporary reward function
        # TODO: define appropriate reward
        reward = 0.0001*np.random.randn() + delVF
        return self.domain, reward
        