"""Graph tests."""
from typing import List, Set
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


def test_forward_mix():
    """Test forward is calculated correctly for multiplication and addition."""
    # graph
    x = Var("x")
    y = Var("y")
    z = Var("z")
    f = (x * y) + (y * z)
    # grad
    x.assign(3.0)
    y.assign(5.0)
    z.assign(11.0)
    dx = f.forward(x)
    dy = f.forward(y)
    dz = f.forward(z)
    assert dx == 5.0
    assert dy == 14.0
    assert dz == 5.0


def get_indices(name: str, nodes: List[Var]) -> Set[int]:
    """Get all indices of nodes with given name."""
    result = set()
    for idx, node in enumerate(nodes):
        if node.name == name:
            result.add(idx)
    return result


def test_bfs():
    """Test bfs order."""
    x = Var("x")
    y = Var("y")
    z = Var("z")
    f = (x * y) + (y * z)
    # grad
    nodes = list(f.bfs())
    assert len(nodes) == 6
    muls = get_indices("*", nodes)
    assert len(muls) == 2
    adds = get_indices("+", nodes)
    assert len(adds) == 1
    xs = get_indices("x", nodes)
    assert len(xs) == 1
    ys = get_indices("y", nodes)
    assert len(ys) == 1
    zs = get_indices("z", nodes)
    assert len(zs) == 1
    assert all(l < r for l in adds for r in muls)


def test_dfs():
    """Test dfs order."""
    x = Var("x")
    y = Var("y")
    z = Var("z")
    f = (x * y) + (y * z)
    # grad
    nodes = list(f.dfs())
    assert len(nodes) == 6
    muls = get_indices("*", nodes)
    assert len(muls) == 2
    adds = get_indices("+", nodes)
    assert len(adds) == 1
    xs = get_indices("x", nodes)
    assert len(xs) == 1
    ys = get_indices("y", nodes)
    assert len(ys) == 1
    zs = get_indices("z", nodes)
    assert len(zs) == 1
    assert all(l < r for l in muls for r in adds)


def test_grade_mix():
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
    dx = f.forward(x)
    dy = f.forward(y)
    dz = f.forward(z)
    assert dx == 5.0
    assert dy == 14.0
    assert dz == 5.0
    f.backward()
    assert dx == x.grad()
    assert dy == y.grad()
    assert dz == z.grad()


def test_sub():
    """Test subtract operator."""
    # graph
    x = Var("x")
    y = Var("y")
    z = Var("z")
    f = (x * y) - (y * z)
    # grad
    x.assign(3.0)
    y.assign(7.0)
    z.assign(2.0)
    assert f.value() == 7.0
    dx = f.forward(x)
    dy = f.forward(y)
    dz = f.forward(z)
    assert dx == 7.0
    assert dy == 1.0
    assert dz == -7.0
    f.backward()
    assert dx == x.grad()
    assert dy == y.grad()
    assert dz == z.grad()
