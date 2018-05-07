import topy
import numpy as np
import matplotlib.pyplot as plt

DOMAIN_SZ = np.array([128, 96])
CELL_SZ = np.array([16, 16])

'''
TODO (Ideas):
- Initial lattice environment [Done]
- FEA results [Done]
- DDPG architecture [Done]
- Random starting forces and constraints [Done]
- Reward function incorporating weight and FEA results
    (Limit deformation, compliance, stress?)
- Testing
- Variety of unit cells
- Random starting shapes (and sizes?)
- Attention model
- Extend to 3D
- Deep RL architecture and algorithm
'''

class TPD(object):
    '''
    Defines the ToPy problem with boundary conditions and loads
    '''
    def __init__(self, num_elem_x=60, num_elem_y=20):
        self.num_elem_x = num_elem_x
        self.num_elem_y = num_elem_y
        
        self.config = self.minimal_tpd_dict()
        self.config['NUM_ELEM_X'] = self.num_elem_x
        self.config['NUM_ELEM_Y'] = self.num_elem_y
        
        self.set_boundary_condition()
        self.load_top()
        
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
            'NUM_ITER': 2,
            'FXTR_NODE_X': '1|21',
            'FXTR_NODE_Y': '1281',
            'LOAD_NODE_Y': '1',
            'LOAD_VALU_Y': '-1'}
        return tpd_dict
    
    def set_boundary_condition(self, bcId=0):
        '''
        Constrains the nodes movement
        bcId =  1 for cantilever
                2 for simply supported
                3 fixed
        '''
        if bcId == 0: # choose randomly if 0
            bcId = np.random.randint(1, 4)
            
        if bcId == 1:
            left, _, _, _ = self.get_edges_string()
            self.config['FXTR_NODE_X'] = left
            self.config['FXTR_NODE_Y'] = left
        elif bcId == 2:
            left, _, right, _ = self.get_edges_array()
            self.config['FXTR_NODE_X'] = str(left[-1])
            self.config['FXTR_NODE_Y'] = str(left[-1]) + '; ' + str(right[-1])
        elif bcId == 3:
            left, _, right, _ = self.get_edges_string()
            self.config['FXTR_NODE_X'] = left + '; ' + right
            self.config['FXTR_NODE_Y'] = left + '; ' + right
        # TODO add hanging beam bc
        
        return bcId
            
    def load_top(self, nodes=np.array([]), value=0):
        '''
        Load the top edge at multiple 'nodes' with total load 'value'
        '''
        if value == 0: # choose randomly if 0
            value = -(np.random.random()*0.5 + 0.5) # value between -[0.5, 1.0)
            
        if nodes.size == 0: # choose randomly if empty
            _, _, _, top = self.get_edges_array()
            num_nodes = top.size
            first = np.random.randint(num_nodes/2+1) # start in first half of top edge
            last = np.random.randint(num_nodes/2, num_nodes) # end in second half of top edge
            nodes = self._get_nodes_array(top[first], top[last], self.num_elem_y+1)
            # TODO choose last based on bcId
            
        value /= float(nodes.size) # distribute load to multiple nodes
        
        self.config['LOAD_NODE_Y'] = nodes
        self.config['LOAD_VALU_Y'] = str(value) + '@' + str(nodes.size)
        
        return nodes, value
        
    def set_key_value(self, key, value):
        self.config[key] = value
    
    def get_edges_string(self):
        '''
        Returns a tuple of edges: (left, bottom, right, top)
        Each edge is a TPD string of indices of its nodes
        '''
        left = self._get_nodes_string(1, self.num_elem_y+1)
        bottom = self._get_nodes_string(self.num_elem_y+1, (self.num_elem_x+1)*(self.num_elem_y+1), self.num_elem_y+1)
        right = self._get_nodes_string(self.num_elem_x*(self.num_elem_y+1)+1, (self.num_elem_x+1)*(self.num_elem_y+1))
        top = self._get_nodes_string(1, self.num_elem_x*(self.num_elem_y+1)+1, self.num_elem_y+1)
        return (left, bottom, right, top)
    
    def get_edges_array(self):
        '''
        Returns a tuple of edges: (left, bottom, right, top)
        Each edge is a numpy array of indices of its nodes
        '''
        left = self._get_nodes_array(1, self.num_elem_y+1)
        bottom = self._get_nodes_array(self.num_elem_y+1, (self.num_elem_x+1)*(self.num_elem_y+1), self.num_elem_y+1)
        right = self._get_nodes_array(self.num_elem_x*(self.num_elem_y+1)+1, (self.num_elem_x+1)*(self.num_elem_y+1))
        top = self._get_nodes_array(1, self.num_elem_x*(self.num_elem_y+1)+1, self.num_elem_y+1)
        return (left, bottom, right, top)
    
    def _get_nodes_string(self, first, last=0, step=1):
        '''
        Returns string of node(s) in TPD string format
        TPD string format:
            '4' denotes node 4
            '4|8' denotes nodes 4, 5, 6, 7, 8
            '4|13|3' denotes nodes 4, 7, 10, 13
        '''
        s = str(first)
        if last > 0:
            s += '|' + str(last)
            if step != 1:
                s += '|' + str(step)
        return s
    
    def _get_nodes_array(self, first, last=0, step=1):
        '''
        Returns array of node(s)
        '''
        if last > 0:
            return np.arange(first, last+1, step)
        else:
            return np.array([first])
        
    def _convert_array_to_string(self, nodes):
        '''
        Returns a TPD string with each node in the input array of nodes
        '''
        s = ''
        for node in nodes:
            s += str(node) + '|'
        return s[:-1]


class Domain(object):
    '''
    Defines the geometry of the lattice
    '''
    def __init__(self, domainSz = DOMAIN_SZ, cellSz = CELL_SZ):
        '''
        Initializes the state of the domain to zeros
        '''
        
        # TODO currently only defined for spherical lattice
        self.domainSz = domainSz
        self.cellSz = cellSz
        self.paramsDim = 1 # number of parameters for each cell
        self.paramsMin = 1 # minimum parameter value
        self.paramsMax = 7 # maximum parameter value
        self.paramsDel = 1 # step size in patameter value
        
        self.numCells = (self.domainSz//self.cellSz).astype(int) # TODO: ensure domainSz is divisible by cellSz
        
        # TODO: fix for more than 1 paramsDim
        self.reset_params()
        
        self.binImg = np.zeros(self.domainSz, dtype = bool)[:, :, np.newaxis] # Shape in the domain
        self.dispX = np.zeros(self.domainSz + 1)[:, :, np.newaxis] # Node displacement X
        self.dispY = np.zeros(self.domainSz + 1)[:, :, np.newaxis] # Node displacement Y
        self.comp = np.zeros(self.domainSz)[:, :, np.newaxis] # Compliance
        
        self.volFrac = 0.
        
    def reset_params(self, value = -1):
        if value < 0:
            value = (self.paramsMax+self.paramsMin)/2.
        self.params = value*np.ones(np.append(self.numCells, self.paramsDim))
    
    def passive_elems(self):
        '''
        Returns a sorted list of linear indices of pixels where the input image is False
        '''
        return np.sort(np.ravel_multi_index(np.where(self.binImg == False), self.binImg.shape))
        
    def get_cell_img(self, param):
        '''
        Returns binary image of one cell with [SPHERICAL] lattice parameter param
        '''
        x, y = np.meshgrid(np.array(range(self.cellSz[0])) + 1./2, np.array(range(self.cellSz[1])) + 1./2, indexing='ij')
        cell = np.sqrt((x - self.cellSz[0]/2)**2 + (y - self.cellSz[1]/2)**2) >= param
        return cell[:, :, np.newaxis]
    
    def set_domain_image(self):
        '''
        Returns binary image of whole domain
        '''
        # TODO: think about making this more efficient by doing vectorized computations without loops
        for i in range(self.numCells[0]):
            for j in range(self.numCells[1]):
                self.binImg[i*self.cellSz[0]:(i+1)*self.cellSz[0],
                             j*self.cellSz[1]:(j+1)*self.cellSz[1], :] \
                = self.get_cell_img(self.params[i, j])
        self.volFrac = np.count_nonzero(self.binImg)*1./self.binImg.size
        
    def compute_base_states(self):
        '''
        computes the base extreme states: will help in designing reward function
        '''
        # Base 1: Solid Cantilever Distributed Load
        self.reset_params(0)
        self.set_domain_image()
        
        tpd = TPD(self.domainSz[0], self.domainSz[1])
        tpd.set_boundary_condition(1) # cantilever
        tpd.load_top(tpd.get_edges_array()[-1], 1) # distributed load on top edge
        tpd.set_key_value('PROB_NAME', 'sph_lat_base_1')
        tpd.set_key_value('VOL_FRAC', self.volFrac)
        tpd.set_key_value('PASV_ELEM', self.passive_elems())
        
        self.solve_fea(tpd.config)
        self.save_results('data/sph_lat_base_1')
        
        # Base 2: Solid Cantilever Point Load
        self.reset_params(0)
        self.set_domain_image()
        
        tpd = TPD(self.domainSz[0], self.domainSz[1])
        tpd.set_boundary_condition(1) # cantilever
        tpd.load_top(np.array([tpd.get_edges_array()[-1][-1]]), 1) # distributed load on top edge
        tpd.set_key_value('PROB_NAME', 'sph_lat_base_2')
        tpd.set_key_value('VOL_FRAC', self.volFrac)
        tpd.set_key_value('PASV_ELEM', self.passive_elems())
        
        self.solve_fea(tpd.config)
        self.save_results('data/sph_lat_base_2')
        
        # Base 3: Void Cantilever Distributed Load
        self.reset_params(self.paramsMax)
        self.set_domain_image()
        
        tpd = TPD(self.domainSz[0], self.domainSz[1])
        tpd.set_boundary_condition(1) # cantilever
        tpd.load_top(tpd.get_edges_array()[-1], 1) # distributed load on top edge
        tpd.set_key_value('PROB_NAME', 'sph_lat_base_3')
        tpd.set_key_value('VOL_FRAC', self.volFrac)
        tpd.set_key_value('PASV_ELEM', self.passive_elems())
        
        self.solve_fea(tpd.config)
        self.save_results('data/sph_lat_base_3')
        
        # Base 4: Void Cantilever Point Load
        self.reset_params(self.paramsMax)
        self.set_domain_image()
        
        tpd = TPD(self.domainSz[0], self.domainSz[1])
        tpd.set_boundary_condition(1) # cantilever
        tpd.load_top(np.array([tpd.get_edges_array()[-1][-1]]), 1) # distributed load on top edge
        tpd.set_key_value('PROB_NAME', 'sph_lat_base_4')
        tpd.set_key_value('VOL_FRAC', self.volFrac)
        tpd.set_key_value('PASV_ELEM', self.passive_elems())
        
        self.solve_fea(tpd.config)
        self.save_results('data/sph_lat_base_4')
        
        # Reset to default params
        self.reset_params()
        self.set_domain_image()
        
    # TODO: randomize intial state VERY IMPORTANT
    def compute_state(self):
        '''
        computes the full current state including FEA results
        '''
        tpd = TPD(self.domainSz[0], self.domainSz[1])
        # TODO set boundary condition, randomly set now
        tpd.set_boundary_condition(1) # cantilever
        # TODO set loads, randomly set now
        tpd.set_key_value('PROB_NAME', 'sph_lat_test')
        tpd.set_key_value('VOL_FRAC', self.volFrac)
        tpd.set_key_value('PASV_ELEM', self.passive_elems())
        
        self.solve_fea(tpd.config)
        
    def solve_fea(self, config):
        '''
        Solves the ToPy FEA problem specified in TPD config
        '''
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
        
    def get_max_values(self):
        '''
        Returns max X-displacement, Y-displacement, and compliance
        '''
        return (np.amax(np.absolute(self.dispX)), np.amax(np.absolute(self.dispY)), np.amax(self.comp))
        
    def save_results(self, name = 'out'):
        '''
        Plot results as image and save in PNG format
        '''
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        im1 = ax1.matshow(self.binImg[:, :, 0].T)
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('Binary Image')
        ax1.xaxis.set_ticks_position('bottom')
        im2 = ax2.matshow(self.dispX[:, :, 0].T)
        fig.colorbar(im2, ax=ax2)
        ax2.set_title('X displacement')
        ax2.xaxis.set_ticks_position('bottom')
        im3 = ax3.matshow(self.comp[:, :, 0].T)
        fig.colorbar(im3, ax=ax3)
        ax3.set_title('Compliance')
        ax3.xaxis.set_ticks_position('bottom')
        im4 = ax4.matshow(self.dispY[:, :, 0].T)
        fig.colorbar(im4, ax=ax4)
        ax4.set_title('Y displacement')
        ax4.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        fig.savefig(name + '_figure.png')
        
    def _update(self, action):
        delta = self.paramsDel * np.sign(np.around(action))
        self.params = np.clip(self.params + delta, self.paramsMin, self.paramsMax)
        self.set_domain_image()
        

class Lattice(object):
    def __init__(self, domainSz = DOMAIN_SZ, cellSz = CELL_SZ):
        self.domain = Domain(domainSz, cellSz)
        
        self.domain.set_domain_image()
        self.domain.compute_state()
        
        self.state_shapes = (self.domain.binImg.shape,
                             self.domain.dispX.shape,
                             self.domain.dispY.shape,
                             self.domain.comp.shape)
        self.action_size = np.prod(self.domain.numCells)
        
        self.max_limit = (600., 3000., 30000.) # 1/10th
        
    def reset(self):
        self.domain = Domain(self.domain.domainSz, self.domain.cellSz)
        
        self.domain.set_domain_image()
        self.domain.compute_state() # TODO (IMP) randomize initial state
        return self.domain
    
    def compute_reward(self, max_values):
        return 1. - (max_values[0]/self.max_limit[0] + max_values[1]/self.max_limit[1] + max_values[2]/self.max_limit[2])/3.
            
    def step(self, action, compute = False):
        assert(self.domain.params.size == action.size)
        
        reward = 0.
        VF = self.domain.volFrac
        
        # TODO randomize action occasionally
        if np.random.random > 0.5:
            i = np.random.randint(action.size)
            action[i] = np.random.randint(2)*2 - 1 # 1 or -1
        
        self.domain._update(action.reshape(self.domain.params.shape))
        if compute:
            self.domain.compute_state()
            reward += self.compute_reward(self.domain.get_max_values())
        delVF = self.domain.volFrac - VF
        
        # temporary reward function
        # TODO: define appropriate reward
        reward -= delVF
        return self.domain, reward
    
    def end(self):
        return
        