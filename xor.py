import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def answer():
    # define ReLU function
    def relu(x):
        return np.maximum(0, x)

    # define the function f(x)
    def f(x1, x2):
        x = np.array([x1, x2])

        # First ReLU layer
        W1 = np.array([
            [ 1.0,  0.0],   # x1
            [-1.0,  0.0],   # -x1
            [ 0.0,  1.0],   # x2
            [ 0.0, -1.0],   # -x2
        ])

        h1 = relu(W1@x)

        # second ReLU layer
        W2 = np.array([
            [1.0, 0.0, 0.0, -1.0],   # x_1' - x_4'
            [0.0, 1.0, -1.0, 0.0],   # x_2' - x_3'
        ])

        h2 = relu(W2@h1)

        # final linear layer (if needed)

        # just split for convenience
        W3_1 = np.array([1.0, 1.0, 0.0, 0.0])
        W3_2 = np.array([-1.0, -1.0])


        output = W3_1@h1 + W3_2@h2
        return output

    # create a grid of points for testing answer:
    x1_vals = np.linspace(-2, 2, 100)
    x2_vals = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.vectorize(f)(X1, X2)
    title = "My XOR :D"

    return X1, X2, Z,title

X1, X2, Z,title = answer()


# plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title(title)
plt.show()
