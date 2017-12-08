import numpy as np

def atleast_2d(arr):
	''' A version of numpy.atleast_2d that makes sure that the original 
	1d shape is the first dimension in 2d '''
    a = np.atleast_2d(arr)
    if a.shape[0] != arr.shape[0]:
        a = a.transpose()
    return a