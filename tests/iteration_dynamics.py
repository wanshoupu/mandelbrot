import numpy as np
import matplotlib.pyplot as plt

max_iter = 1e2
if __name__ == "__main__":
    c = complex(-.501, .20)
    z = complex(0, 0)
    trajectory = []
    it = 0
    while abs(z) <= 2 and it < max_iter:
        it += 1
        zp = z * z + c
        if np.isclose(zp, z):
            break
        z = zp
        trajectory.append(z)
    plt.plot(trajectory, marker='o')
    plt.title("Iteration of y = x² + c")
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.grid(True)
    plt.show()
