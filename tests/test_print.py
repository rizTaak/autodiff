"""Print graph in console."""
from autodiff.graph import Var

# graph
x = Var("x")
y = Var("y")
z = Var("z")
f = (x * y) + (y * z)

x.assign(3.0)
y.assign(5.0)
z.assign(11.0)

print("****eval****")
f.value()
f.print()

# eval
print("****grad(x)****")
f.forward(x)
f.print()

print("****grad(y)****")
f.forward(y)
f.print()

print("****grad(z)****")
f.forward(z)
f.print()

print("****adjoint****")
f.backward()
f.print()
