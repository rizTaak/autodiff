"""Minimize function example using grad."""
import random
from autodiff.graph import Var

# graph
x = Var("x")
y = Var("y")
f: Var = (x * x) + (y * y)

# minimize
x.assign(random.uniform(-5.0, 5.0))  # nosec
y.assign(random.uniform(-5.0, 5.0))  # nosec
print(f"starting: val={f.value()}, x={x.eval_value}, y={y.eval_value}")
LRATE = 0.1
for itr in range(10):
    f.backward()
    x.assign(x.value() - LRATE * x.grad())
    y.assign(y.value() - LRATE * y.grad())
    val = f.value()
    print(f"iter {itr}: val={val}, x={x.value()}, y={y.value()}")
