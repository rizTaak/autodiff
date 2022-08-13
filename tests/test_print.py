"""Print graph in console."""
from autodiff.graph import Var

# pylint: disable=invalid-name

def test_print():
    """Test printing different states of graph."""
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


def test_literal_print():
    """Test printing of graph with literal."""
    x = Var("x")
    f: Var = x*2.0
    x.assign(3.0)
    f.value()
    f.print()

def test_negation_print():
    """Test printing of graph with literal."""
    x = Var("x")
    f: Var = -x*2.0
    x.assign(1.0)
    f.forward(x)
    f.print()


# test_print()
# test_literal_print()
# test_negation_print()
