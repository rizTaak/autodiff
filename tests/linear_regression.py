"""Linear regression."""
import random
from random import shuffle
from typing import List, Tuple
import matplotlib.pyplot as plt
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
w_start = random.uniform(-0.3, 0.3) # nosec
b_start = random.uniform(-0.3, 0.3) # nosec
w.assign(w_start) # nosec
b.assign(b_start) # nosec

# training data
xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ys = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]
data: List[Tuple[float, float]] \
    = list(zip(xs, ys))

# sgd
for epoch in range(1000):
    shuffle(data)
    for x_data, y_data in data:
        x.assign(x_data)
        y.assign(y_data)
        l.backward()
        w.assign(w.value() - LEARNING_RATE * w.grad())
        b.assign(b.value() - LEARNING_RATE * b.grad())
    # print(f'w={w.value()} b={b.value()}')

# eval
for x_data, y_data in data:
    x.assign(x_data)
    y.assign(y_data)
    print(f'x={x_data}, y={y_data}, est={f.value()}')

# plot
fig, ax = plt.subplots()
ax.axline((0, b_start), slope=w_start, color='red', label='before training')
ax.axline((0, b.value()), slope=w.value(), color='green', label='after training')
ax.scatter(xs, ys, color="blue")
ax.legend()
plt.show()
