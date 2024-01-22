import spdist
import numpy as np


def main():
    x = np.linspace(0, 10, 10000)
    y = x

    x_ref = x
    y_ref = x + 1

    print(spdist.spdist(x, y, x_ref, y_ref))


if __name__ == "__main__":
    main()
