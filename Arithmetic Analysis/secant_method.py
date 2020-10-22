from math import exp

def f(x):
    return 8 * x - 2 * exp(-x)

def SecantMethod(lower_bound, upper_bound, repeats):
    x0 = lower_bound
    x1 = upper_bound
    for i in range(0, repeats):
        x0, x1 = x1, x1, - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
    return x1

print(f"The solution is: {SecantMethod(1, 3, 2)}")  