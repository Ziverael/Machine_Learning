"""
This package conatint algorithms for time series analysis and machine learning.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Linear_regression():
    def __init__(self, xs, ys):
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        
        mn_xs = np.mean(xs)
        mn_ys = np.mean(ys)
        xs_red = xs - mn_xs
        ys_red = ys - mn_ys
        self.a = np.dot(xs_red, ys_red) / np.dot(xs_red, xs_red)
        self.b = mn_ys - self.a * mn_xs
        
        self.ys_hat = xs * self.a + self.b

    def get_coef(self):
        """Return regression line're coefficients"""
        return self.a, self.b
    
    def get_ys_hat(self):
        """Return estimated values"""
        return self.ys_hat


    def plot_reg(self):
        sns.scatterplot(x = self.xs, y = self.ys)
        sns.lineplot(x = self.xs, y = self.ys_hat)
        plt.show()
    
    def prognose(self, n = 100, step_size = 1.0):
        """Return prognose values from liearn regerssion.
        
        Keyword arguments:
        n   [positive int]  DEFAULT = 100  number of steps to prognose
        step_size   [positive float or int] DEFAULT 1.0 step size
        Return: y_prog  [np.array]  prognossed values
        """
        n = int(n)
        if n < 1:
            raise ValueError("Number of steps should be positive!")
        
        xs_beg = self.xs[-1] + step_size
        xs_end = xs_beg + n * step_size
        xs_prog = np.linspace(xs_beg, xs_end)
        ys_prog = xs_prog * self.a + self.b
        return ys_prog




if __name__ == "__main__":
    xs = np.arange(0, 100)
    ys = np.random.normal(0, 2, 100) + xs
    lr = Linear_regression(xs[:60], ys[:60])
    a,b = lr.get_coef()
    sns.scatterplot(x = xs, y = ys)
    sns.lineplot(x = xs, y = a * xs + b)
    plt.show()

    ys2 = np.random.normal(0,4, 100) + xs * 0.4 + 4
    lr = Linear_regression(xs[:60], ys2[:60])
    a,b = lr.get_coef()
    sns.scatterplot(x = xs, y = ys2)
    sns.lineplot(x = xs, y = a * xs + b)
    plt.show()

    xs = np.linspace(0, 10, 100)
    ys3 = np.random.normal(0,5, 100) + xs * xs * 0.5 + 4
    lr = Linear_regression(xs[:60], ys2[:60])
    a,b = lr.get_coef()
    sns.scatterplot(x = xs, y = ys3)
    sns.lineplot(x = xs, y = a * xs + b)
    plt.show()





