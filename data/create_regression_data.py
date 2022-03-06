
import numpy as np
class LinearRegressionData:
    def create_data_points(n: int, seed = None, verbose=False) -> np.array:
        '''
        Create a 2-D numpy array with n elements. If x_limit and y_limit are specified, the (x,y) tuples are in those limits
        Provide seed to be reproducible
        args: 
        n: number of points to return
        verbose: if True, the data points generated will be shown to the user
        return:
        2-D numpy array
        '''
        if n<=0:
            raise ValueError("n has to be a positive number")
        x_vals = np.random.uniform(0, 100, n)
        x_coeff = np.random.uniform(0, 10)
        constant = np.random.uniform(0, 10)
        y_function = lambda x: x_coeff*x + constant
        y_vals = np.array([y_function(x) + np.random.uniform(-100,100) for x in x_vals])
        if verbose:
            print("Slope: {}\nConstant: {}".format(x_coeff, constant))
            print(x_vals, y_vals)
        return x_vals, y_vals