import topy
import numpy as np

def minimal_tpd_dict():
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

def passive_elems(bin_img):
    '''
    Returns a sorted list of linear indices of pixels where the input image is False
    '''
    return np.sort(np.ravel_multi_index(np.where(bin_img == False), bin_img.shape))


