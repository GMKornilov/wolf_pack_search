from numpy import ndarray


def rosenbrock(x: ndarray) -> float:
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def colville(x: ndarray) -> float:
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2 + (x[2] - 1) ** 2 + 90 * (x[2] ** 2 - x[3]) ** 2 + 10.1 * (
                x[1] - 1) ** 2 + (x[3] - 1) ** 2 + 19.8 * (x[1] - 1) * (x[3] - 1)


def sphere_function(x: ndarray) -> float:
    return sum([i ** 2 for i in x])


def sum_squares_function(x: ndarray) -> float:
    return sum([(i + 1) * x[i] ** 2 for i in range(len(x))])


def booth_function(x: ndarray) -> float:
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
