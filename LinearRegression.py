from cProfile import label
from cmath import sqrt
from turtle import color
from pyparsing import lineEnd
from data import create_regression_data
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
class LinearRegression:
    def __init__(self):
        self.data_point_count = 1000
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def compute_mean(self, values):
        '''
        Compute mean of a list of values
        args:
        values: list[int]

        return:
        mean: float
        '''
        return sum(values)/len(values)

    def compute_variance(self, values, mean):
        '''
        Compute variance given a list and 
        the mean of the values
        args:
        values: list[int]
        mean: float
        
        return:
        variance: float
        '''
        return sum([(x-mean)**2 for x in values])

    def compute_covariance(self, x, mean_x, y, mean_y):
        '''
        Compute covariance given two lists and 
        the mean of the values in the lists
        args:
        x: First list of values (list[int])
        mean_x: Mean of x (float)
        y: Second list of values (list[int])
        mean_y: Mean of y (float)
        
        return:
        covariance: float
        '''
        covariance = 0.0
        for i in range(len(x)):
            covariance += (x[i]- mean_x) * (y[i]-mean_y)
        return covariance

    def get_coefficients(self):
        '''
        Compute coefficients to fit the regression line
        args:
        None
        
        return:
        b0: intercept
        b1: x_coefficient
        '''
        x_mean = self.compute_mean(self.x_train)
        y_mean = self.compute_mean(self.y_train)
        b1 = self.compute_covariance(self.x_train, x_mean, self.y_train, y_mean)/self.compute_variance(self.x_train, x_mean)
        b0 = y_mean - b1*x_mean
        return b0, b1

    def make_predictions(self, b0, b1):
        '''
        Return predictions for the test set

        args:
        b0: Intercept
        b1: x-coefficient

        return:
        predictions: list of predictions for points in the test set
        '''
        predictions = []
        for x in self.x_test:
            predictions.append(x*b1 + b0)
        return predictions

    def rmse_metric(self, predictions):
        '''
        Computes RMSE (Root Mean Square Error) between actual
        and predicted values
        args:
        predictions: A list of predictions (list)
        
        return:
        rmse: The RMSE value (float)'''
        total_squared_error = 0.0
        for idx, prediction in enumerate(predictions):
            pred_error = (prediction - self.y_test[idx])**2
            total_squared_error += pred_error
        mean_error = total_squared_error/len(predictions)
        return sqrt(mean_error)

    def get_data_points(self) -> np.array:
        '''
        Creates the train and test_set to fit the regression line to
        args:
        None
        return:
        '''
        c = input("Enter number of points to fit regression line to. Press enter to use default (1000)")
        if not c:
            print("Using default values...")
        else:
            try:
                self.data_point_count = int(c)
            except ValueError("Accepted values are integers only"):
                exit
        x,y = create_regression_data.LinearRegressionData.create_data_points(math.floor(self.data_point_count*1.25))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
        return x_train, x_test, y_train, y_test

    def plot_output(self, predictions):
        '''
        Plot the train, test and predicted values
        args:
        predictions: A list of predictions for the test set
        
        return: 
        None'''
        plt.scatter(self.x_train, self.y_train, color = 'red', label = "Train", marker = "4")
        plt.scatter(self.x_test, self.y_test, color='green', label = "Test", marker= "*")
        plt.plot(self.x_test, predictions, label = "Predictions", color = "black")
        plt.legend()
        plt.show()
    
    def main(self):
        self.x_train, self.x_test,  self.y_train, self.y_test = self.get_data_points()
        b_0, b_1 = self.get_coefficients()
        predictions = self.make_predictions(b_0, b_1)
        rmse = self.rmse_metric(predictions)
        print("RMSE: {}".format(rmse))
        self.plot_output(predictions)
        


if __name__=="__main__":
    linear_regressor = LinearRegression()
    linear_regressor.main()        
        


