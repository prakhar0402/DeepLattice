import tpd
import os
import topy
import numpy as np
import logging
import sys
import tensorflow as tf
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# 2D spherical lattice

domainSz = np.array([160, 80]) # in mm

cellSz = np.array([20, 20]) # in mm
numCells = (domainSz//cellSz).astype(int) # TODO: ensure domainSz is divisible by cellSz

voxelSz = np.array([1, 1]) # in mm
numVoxels = (cellSz//voxelSz).astype(int) # in each cell/unit

paramsDim = 1 # number of parameters for each cell
params = 5*np.ones(np.append(numCells, paramsDim))

# origin coordinates of cells
x_c, y_c = np.meshgrid(range(numCells[0])*cellSz[0], 
                    range(numCells[1])*cellSz[1], indexing='ij')

# voxel center coordinates relative to cell origin
x_v, y_v = np.meshgrid(range(numVoxels[0])*voxelSz[0] + voxelSz[0]/2, 
                    range(numVoxels[1])*voxelSz[1] + voxelSz[1]/2, indexing='ij')

field = np.zeros(numCells*numVoxels)
holes = np.zeros((np.prod(numCells), 2)) # hole coordinates in the cells, only for this type of cells
hh = 0
# TODO: make more efficient by doing vectorized computations outside loop
for i in range(numCells[0]):
    for j in range(numCells[1]):
        p = params[i, j]
        # voxel center global coordinates
        x = x_v + x_c[i, j]
        y = y_v + y_c[i, j]
        # cell center coordinates
        xx = x_c[i, j] + cellSz[0]/2
        yy = y_c[i, j] + cellSz[1]/2
        holes[hh] = [xx, yy]
        hh += 1
        # calculate distance field
        cell = np.sqrt((x - xx)**2 + (y - yy)**2) - p
        field[i*numVoxels[0]:(i+1)*numVoxels[0], 
              j*numVoxels[1]:(j+1)*numVoxels[1]] += cell


config = tpd.minimal_tpd_dict()
config['PROB_NAME'] = 'sph_lat_test'
config['NUM_ELEM_X'] = domainSz[0]
config['NUM_ELEM_Y'] = domainSz[1]
config['VOL_FRAC'] = np.count_nonzero(field >= 0)*1./field.size
config['NUM_ITER'] = 2
config['PASV_ELEM'] = tpd.passive_elems(field >= 0)
config['FXTR_NODE_Y'] = '1|81'
config['FXTR_NODE_X'] = '1|81'
config['LOAD_NODE_Y'] = '12961'
config['LOAD_VALU_Y'] = '-1'

t = topy.Topology()
t.load_config_dict(config)
t.set_top_params()

topy.optimise(t)
