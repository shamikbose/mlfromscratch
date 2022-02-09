import random
import numpy as np
def create_data_points(n: int, x_limit=1000, y_limit=100, seed = None, verbose=False) -> np.array:
    '''
    Create a 2-D numpy array with n elements. If x_limit and y_limit are specified, the (x,y) tuples are in those limits
    Provide seed to be reproducible
    args: 
    n: number of points to return
    x_limit: the range in which to produce the x-values. The lower range is negative of this value (optional)
    y_limit: the range in which to produce the y-values. The lower range is negative of this value (optional)
    verbose: if True, the data points generated will be shown to the user
    return:
    2-D numpy array
    '''
    if n<=0:
        raise ValueError("n has to be a positive number")
    if x_limit<=0 or y_limit<=0:
        raise ValueError("x_limit and y_limit can take positive values only")
    x_vals = np.random.uniform(-x_limit, x_limit, n)
    y_vals = np.random.uniform(-y_limit, y_limit, n)
    data_points = np.dstack((x_vals, y_vals))
    if verbose:
        print(data_points)
    return data_points