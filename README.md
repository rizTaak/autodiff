# autodiff
![build](https://github.com/rizTaak/autodiff/actions/workflows/main.yml/badge.svg)

Toy implementaion of automatic differentiation for learning purpose

## Setup

* Prepare env

```bash
pipenv sync --dev
```

* Run tests

```bash
pipenv shell
pytest
```

## Example

```python
from autodiff.graph import Var
# create variables
x = Var("x")
y = Var("y")
z = Var("z")
# define function (graph)
f = x*y+y*z
# assign values to variables
x.assign(3.0)
y.assign(5.0)
z.assign(11.0)
# differentiation using backward propagation
f.backward()
# print state of the graph
f.print()
# get gradients
dx = x.grad()
dy = y.grad()
dz = z.grad()
print(f'dx={dx}, dy={dy}, dz={dz}')
```

Output

```bash
+| val=70.0 grad=1.0 forward=nan
   *| val=15.0 grad=1.0 forward=nan
      x| val=3.0 grad=5.0 forward=nan
      y| val=5.0 grad=14.0 forward=nan
   *| val=55.0 grad=1.0 forward=nan
      y| val=5.0 grad=14.0 forward=nan
      z| val=11.0 grad=5.0 forward=nan
dx=5.0, dy=14.0, dz=5.0
```
