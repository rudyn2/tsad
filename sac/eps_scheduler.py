import numpy as np
import matplotlib.pyplot as plt


class Epsilon:
    def __init__(self, max_steps, method='linear', epsilon_max=1.0, epsilon_min=0.1, factor=3):
        method = method.lower()
        if method == "constant":
            self._get_epsilon = self.constant
        elif method == "linear":
            self._get_epsilon = self.linear
        elif method == "exp":
            self._get_epsilon = self.exp
        elif method == "inverse_sigmoid":
            self._get_epsilon = self.inverse_sigmoid
        else:
            raise NotImplementedError(
                "method must be constant, linear, exp, or inverse_sigmoid"
            )

        self._max_steps = max_steps
        self._epsilon_min = max(0.0, epsilon_min)
        self._epsilon_max = min(1.0, epsilon_max)
        self._factor = factor

        self._step = 0

    def __call__(self, step):
        return self._get_epsilon(max(0, step))

    def constant(self, step):
        return self._epsilon_max

    def linear(self, step):
        return max(
            self._epsilon_min,
            self._epsilon_max
            - (self._epsilon_max - self._epsilon_min) * step / self._max_steps,
        )

    def exp(self, step):
        return self._epsilon_min + (self._epsilon_max - self._epsilon_min) * np.exp(
            -self._factor * step / self._max_steps
        )

    def inverse_sigmoid(self, step):
        return self._epsilon_min + (self._epsilon_max - self._epsilon_min) * (
                    1 - 1 / (1 + np.exp(-self._factor / self._max_steps * (step - self._max_steps / 2))))

    def step(self):
        self._step += 1
        return self._get_epsilon(self._step)

    def epsilon(self):
        return self._get_epsilon(self._step)

    def plot_epsilon(self, steps):
        plt.plot([self(x) for x in range(steps)])
        plt.xlabel("Steps")
        plt.ylabel("Epsilon")
        plt.savefig("fig.png")
        plt.show()


if __name__ == "__main__":
    eps = Epsilon(100000, method="linear", factor=5)
    eps.plot_epsilon(120000)