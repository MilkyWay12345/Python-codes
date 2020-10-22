import math
from typing import Callable

def intersection(function: Callable[[float], float], x0: float, x1: float) -> float:
    x_n: float = x0
    x_n1: float = x1
    while True:
        if x_n == x_n1 or function(x_n1) == function(x_n):
            raise ZeroDivisionError("float division by zero, could not find root")
        x_n2: float = x_n1 - (function(x_n1) / ((function(x_n1) - function(x_n)) / (x_n1 - x_n)))
        if abs(x_n2 - x_n1) < 10 ** -5:
            return x_n2
        x_n = x_n1
        x_n1 = x_n2

def f(x: float) -> float:
    return math.pow(x, 3) - (2 * x) - 5

if __name__ == "__main__":
    print(intersection(f, 3, 3.5))