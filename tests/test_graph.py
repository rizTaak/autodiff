"""Graph tests."""
from autodiff.graph import Var

# pylint: disable=invalid-name


def test_eval_mult():
    """Test value is calculated correctly for multiplication."""
    # graph
    x = Var("x")
    y = Var("y")
    z = Var("z")
    f = x * y * z
    # eval
    x.assign(2.0)
    y.assign(3.0)
    z.assign(5.0)
    assert 30.0 == f.value()


def test_eval_add():
    """Test value is calculated correctly for multiplication."""
    # graph
    x = Var("x")
    y = Var("y")
    z = Var("z")
    f = x + y + z
    # eval
    x.assign(2.0)
    y.assign(3.0)
    z.assign(5.0)
    assert 10.0 == f.value()


def test_eval_mix():
    """Test value is calculated correctly for multiplication and addition."""
    # graph
    x = Var("x")
    y = Var("y")
    z = Var("z")
    f = (x * y) + (y * z)
    # eval
    x.assign(2.0)
    y.assign(3.0)
    z.assign(5.0)
    assert 21.0 == f.value()


def test_grad_mix():
    """Test grad is calculated correctly for multiplication and addition."""
    # graph
    x = Var("x")
    y = Var("y")
    z = Var("z")
    f = (x * y) + (y * z)
    # grad
    x.assign(3.0)
    y.assign(5.0)
    z.assign(11.0)
    dx = f.grad(x)
    dy = f.grad(y)
    dz = f.grad(z)
    assert dx == 5.0
    assert dy == 14.0
    assert dz == 5.0
