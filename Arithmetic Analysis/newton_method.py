from typing import Callable

RealFunc = Callable[[float], float]

def newton(function: RealFunc, derivative: RealFunc, starting_int: int) -> float:
    prev_guess = float(starting_int)
    while True:
        try:
            next_guess = prev_guess - function(prev_guess) / derivative(prev_guess)
        except ZeroDivisionError:
            raise ZeroDivisionError("Could not find root") from None
        if abs(prev_guess - next_guess) < 10 ** -5:
            return next_guess
        prev_guess = next_guess

def f(x: float) -> float:
    return (x ** 3) - (2 * x) - 5

def f1(x: float) -> float:
    return 3 * (x ** 2) - 2

if __name__ == "__main__":
    print(newton(f, f1, 3))