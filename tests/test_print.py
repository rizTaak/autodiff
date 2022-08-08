"""Print graph in console."""
from autodiff.graph import Var

# graph
x = Var("x")
y = Var("y")
z = Var("z")
program = (x * y) + (y * z)

# eval
x.assign(3.0)
y.assign(5.0)
z.assign(11.0)

print("****eval****")
program.value()
program.print()

print("****grad(x)****")
program.grad(x)
program.print()

print("****grad(y)****")
program.grad(y)
program.print()

print("****grad(z)****")
program.grad(z)
program.print()
