import topy
import numpy as np

class Lattice():
    def __init__(self, size = np.array([160, 80])):
        self.domainSz = size
        self.cellSz = np.array([20, 20]) # in mm
        self.voxelSz = np.array([1, 1]) # in mm
        self.paramsDim = 1 # number of parameters for each cell
        self.paramsMin = 1
        self.paramsMax = 9
        self.paramsDel = 1

        self.numCells = (self.domainSz//self.cellSz).astype(int) # TODO: ensure domainSz is divisible by cellSz
        self.numVoxels = (self.cellSz//self.voxelSz).astype(int) # in each cell/unit
        
        self.params = ((self.paramsMax+self.paramsMin)/2.)*np.ones(np.append(self.numCells, self.paramsDim))
        
        self.bin_img = np.zeros(self.domainSz, dtype = bool) # Shape in the domain
        self.dispX = np.zeros(self.domainSz + 1) # Node displacement X
        self.dispY = np.zeros(self.domainSz + 1) # Node displacement Y
        self.comp = np.zeros(self.domainSz) # Compliance
        
        self.computeImage()
        
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
    
    def computeImage(self):
        # origin coordinates of cells
        x_c, y_c = np.meshgrid(range(self.numCells[0])*self.cellSz[0], 
                            range(self.numCells[1])*self.cellSz[1], indexing='ij')

        # voxel center coordinates relative to cell origin
        x_v, y_v = np.meshgrid(range(self.numVoxels[0])*self.voxelSz[0] + self.voxelSz[0]/2, 
                            range(self.numVoxels[1])*self.voxelSz[1] + self.voxelSz[1]/2, indexing='ij')

        field = np.zeros(self.numCells*self.numVoxels)
        holes = np.zeros((np.prod(self.numCells), 2)) # hole coordinates in the cells, only for this type of cells
        hh = 0
        # TODO: make more efficient by doing vectorized computations outside loop
        for i in range(self.numCells[0]):
            for j in range(self.numCells[1]):
                p = self.params[i, j]
                # voxel center global coordinates
                x = x_v + x_c[i, j]
                y = y_v + y_c[i, j]
                # cell center coordinates
                xx = x_c[i, j] + self.cellSz[0]/2
                yy = y_c[i, j] + self.cellSz[1]/2
                holes[hh] = [xx, yy]
                hh += 1
                # calculate distance field
                cell = np.sqrt((x - xx)**2 + (y - yy)**2) - p
                field[i*self.numVoxels[0]:(i+1)*self.numVoxels[0], 
                      j*self.numVoxels[1]:(j+1)*self.numVoxels[1]] += cell
        
        self.bin_img = field >= 0
        
    def computeState(self):
        '''
        computes the full current state including FEA results
        '''
        self.computeImage()
        
        config = self.minimal_tpd_dict()
        config['PROB_NAME'] = 'sph_lat_test'
        config['NUM_ELEM_X'] = self.domainSz[0]
        config['NUM_ELEM_Y'] = self.domainSz[1]
        config['VOL_FRAC'] = np.count_nonzero(self.bin_img)*1./self.bin_img.size
        config['NUM_ITER'] = 2
        config['PASV_ELEM'] = self.passive_elems()
        config['FXTR_NODE_Y'] = '1|81'
        config['FXTR_NODE_X'] = '1|81'
        config['LOAD_NODE_Y'] = '12961'
        config['LOAD_VALU_Y'] = '-1'
        
        t = topy.Topology()
        t.load_config_dict(config)
        t.set_top_params()
        
        t.update_desvars_oc()
        t.fea()
        t.sens_analysis()
        
        disp = t.d.reshape((161, 81, 2))
        
        self.dispX = disp[:, :, 0] # TODO: change displacement size
        self.dispY = disp[:, :, 1]
        
        self.comp = t.qkq
        
    def update(self, action, compute = False):
        assert(self.params.shape == action.shape)
        delta = self.paramsDel * np.sign(np.around(action))
        self.params = np.clip(self.param + delta, self.paramsMin, self.paramsMax)
        if compute:
            self.computeState()
            
    def step(self, action, compute = False):
        reward = 0.1*np.random.randn() # TODO: define reward
        self.update(action, compute)
        state = (self.bin_img, self.dispX, self.dispY, self.comp)
        return state, action, reward
        