def atleast_2d(arr):
    a = np.atleast_2d(arr)
    if a.shape[0] != arr.shape[0]:
        a = a.transpose()
    return a