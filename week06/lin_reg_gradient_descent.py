import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import seed, normal, rand


def mse(weights, target, X):
    return (target-weights@X).mean()


class NumpyLinReg():

    def __init__(self, X_train, t_train):
        # Append ones to the start of the train data for intercept
        self.x = X_train
        self.y = t_train

    def fit(self, poly_deg=3, gamma=0.00001, epochs=1):
        self.des_mat = np.empty((self.x.size, poly_deg+1))
        for p in range(poly_deg+1):
            self.des_mat[:, p] = self.x**p

        # Generate random starting positions
        starts = rand(epochs, poly_deg+1) * 20 - 10
        for start in starts:
            self.steps = self.gradient_descent(start, gamma)
        self.weights = self.steps[-1]
        print(self.weights)

    def gradient_descent(self, start, gamma, precision=0.0001):
        steps = [start]
        df = gamma * 2 * np.mean(
            (self.des_mat@start - self.y) * self.des_mat.T, axis=1)
        while np.any(df > precision):
            start -= df
            steps.append(start.copy())
            df = gamma * 2 * np.mean(
                (self.des_mat@start - self.y) * self.des_mat.T, axis=1)
        return steps

    def predict(self, x, step=-1):
        y = np.zeros(len(x))
        for i in range(len(self.steps[step])):
            y += self.weights[i]*x**i


if __name__ == "__main__":
    seed(100169)

    def f(x):
        return -6*x**3 + x**2 - 3*x + 5

    n = 100
    a, b = -5, 5
    x = np.linspace(a, b, n)

    sigma = 200
    noise = normal(0, sigma, n)
    y = f(x) + noise

    lin_reg = NumpyLinReg(x, y)
    lin_reg.fit(poly_deg=6)
    print(len(lin_reg.steps))

    # Plotting
    fig, ax = plt.subplots()
    l = plt.scatter(x, y)
    ax = plt.axis([a*1.1, b*1.1, -1000, 1000])
    line, = plt.plot([0], [0], 'r')

    def animate(i, steps):
        xplot = np.linspace(-5, 5, 100)
        beta = steps[i]

        yplot = np.zeros(len(xplot))
        for j in range(len(beta)):
            yplot += beta[j]*xplot**j

        line.set_data(xplot, yplot)
        return line

    # create animation using the animate() function
    myAnimation = animation.FuncAnimation(fig, animate, frames=len(lin_reg.steps),
                                          fargs=(lin_reg.steps,), interval=1, blit=False, repeat=True)

    #myAnimation.save('poly2.gif', writer='imagemagick')

    plt.show()
