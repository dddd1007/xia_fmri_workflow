# The helper functions for spm manipuationg

import scipy.io as sio

## Read spm.mat file
def read_spm(spm_file):
    return sio.loadmat(spm_file)