import numpy as np
import matplotlib.pyplot as plt
import spdist

x = np.linspace(0, 10, 100)
y = 2 * x

x_ref = x
y_ref = 2 * x + 1

distance = spdist.spdist(x, y, x_ref, y_ref)

point_x = 1
point_y = 2 * point_x + 1
normal_vector = np.array([2, -1]) / np.linalg.norm([2, -1])
endpoint_x = point_x + normal_vector[0] * distance
endpoint_y = point_y + normal_vector[1] * distance

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(x, y, label="y")
ax.plot(x_ref, y_ref, label="y_ref")
ax.plot([point_x, endpoint_x], [point_y, endpoint_y], label="normal_vector*distance")
ax.text(endpoint_x + 0.1, endpoint_y - 0.1, f"distance = {distance:.2f}")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

fig.tight_layout()
fig.savefig("readme_plot.png", dpi=300)
