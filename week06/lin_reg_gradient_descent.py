import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import seed, normal, rand


def mse(weights, target, X):
    return np.mean((target-weights@X)**2)


class NumpyLinReg():

    def __init__(self, X_train, t_train):
        # Append ones to the start of the train data for intercept
        self.x = X_train
        self.y = t_train

    def fit(self, poly_deg, gamma, epochs):
        # Make design matrix
        self.des_mat = np.empty((self.x.size, poly_deg+1))
        for p in range(poly_deg+1):
            self.des_mat[:, p] = self.x**p

        # Generate random starting positions
        start = rand(poly_deg+1) * 20 - 10
        self.steps = self.gradient_descent(start, gamma, epochs)
        self.weights = self.steps[-1]

    def gradient_descent(self, start, gamma, epochs):
        steps = [start]
        for i in range(epochs):
            df = gamma * 2 * np.mean(
                (self.des_mat@start - self.y) * self.des_mat.T, axis=1)
            start -= df
            steps.append(start.copy())
        return np.array(steps)

    def predict(self, x, step=-1):
        y = np.zeros(len(x))
        for i in range(len(self.steps[step])):
            y += self.weights[i]*x**i


def animate_fit(x, y, steps, file_name=None):
    n_steps = len(lin_reg.steps)
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
    myAnimation = animation.FuncAnimation(fig, animate, frames=range(0, n_steps, n_steps//300),
                                          fargs=(steps,), interval=30, blit=False, repeat=True)

    if file_name:
        myAnimation.save('%s.gif' % file_name, writer='imagemagick')

    plt.show()


def plot_landscape_path(x, y, steps):

    x = np.concatenate((np.ones((x.size, 1)), x.reshape(-1, 1)), axis=1)

    w0_min = steps[:, 0].min() - 10
    w0_max = steps[:, 0].max() + 40
    w1_min = steps[:, 1].min() - 30
    w1_max = steps[:, 1].max() + 10

    n = 20

    w0 = np.linspace(w0_min, w0_max, n)
    w1 = np.linspace(w1_min, w1_max, n)

    mse_grid = np.empty((n, n))

    def mse(weights):
        return np.mean((y-x@weights)**2)

    for j in range(n):
        for i in range(n):
            mse_grid[i, j] = mse(np.array([w0[j], w1[i]]))

    w0_grid, w1_grid = np.meshgrid(w0, w1)

    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(w0_grid, w1_grid, mse_grid, rstride=3, cstride=3)

    m = len(steps)
    zs = np.empty(m)
    for i in range(m):
        zs[i] = mse(steps[i])

    #ax.plot(steps[1:, 0], steps[1:, 1], zs[1:], 'rx', markersize=5)
    ax.plot(steps[1:, 0], steps[1:, 1], zs[1:], 'r', linewidth=3)
    ax.plot([steps[-1, 0]], [steps[-1, 1]], [zs[-1]], 'g*', markersize=15)

    plt.show()


if __name__ == "__main__":
    seed(3050)

    def f(x):
        return -6*x**3 + x**2 - 3*x + 5

    n = 100
    a, b = -5, 5
    x = np.linspace(a, b, n)

    sigma = 200
    noise = normal(0, sigma, n)
    y = f(x) + noise

    poly_deg = 1
    gamma = 0.01
    epochs = 300

    lin_reg = NumpyLinReg(x, y)
    lin_reg.fit(poly_deg, gamma, epochs)

    # animate_fit(x, y, lin_reg.steps)
    plot_landscape_path(x, y, lin_reg.steps)
