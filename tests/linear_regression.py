"""Linear regression."""
import random
from random import shuffle
from typing import List, Tuple
from autodiff.graph import Var

LEARNING_RATE = 0.005

w = Var("w")
x = Var("x")
b = Var("b")
y = Var("y")

# model
f = w*x+b
# loss
l = (y-f)**2.0

print("f...")
f.print()
print("l...")
l.print()

# initialize weights
w.assign(random.uniform(-0.3, 0.3)) # nosec
b.assign(random.uniform(-0.3, 0.3)) # nosec

# training data
data: List[Tuple[float, float]] \
    = list(zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]))

# sgd
for epoch in range(1000):
    shuffle(data)
    for x_data, y_data in data:
        x.assign(x_data)
        y.assign(y_data)
        dw = l.forward(w)
        db = l.forward(b)
        w.assign(w.value() - LEARNING_RATE * dw)
        b.assign(b.value() - LEARNING_RATE * db)
    print(f'w={w.value()} b={b.value()}')


# eval
for x_data, y_data in data:
    x.assign(x_data)
    y.assign(y_data)
    print(f'x={x_data}, y={y_data}, est={f.value()}')
