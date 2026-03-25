import numpy as np
from equiv_dens.utils import base as utils
import sys

filename = sys.argv[1]

fname = filename.split('.')[0]

props = utils.xyz_to_npy(filename)

np.save(fname, props, allow_pickle=True)
