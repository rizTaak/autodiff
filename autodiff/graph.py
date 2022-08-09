"""Graph related types."""
from functools import reduce
from collections import deque
import operator

from abc import ABC, abstractmethod
from typing import Deque, Iterable, List, cast, Tuple
from xmlrpc.client import Boolean

class Op(ABC):
    """Operator in a graph."""

    def __init__(self, var: "Var"):
        """Initialize operator with graph node."""
        self.var = var

    @abstractmethod
    def eval(self):
        """Evaluate the value of the operator."""

    @abstractmethod
    def forward(self, wrt: "Var"):  # pylint: disable:invalid-name
        """Calculate forward gradient with respect to given variable."""

    def grad(self, root: Boolean = False):
        """Calculate adjoint if not root. Set adjoint to 1.0 otherwise."""
        if root:
            self.var.adjoint_value = 1.0
        else:
            self._grad()

    @abstractmethod
    def _grad(self):
        """Calculate adjoint of the node."""

    def accum_grad(self, contrib: float):
        """Accumulate grad value."""
        self.var.adjoint_value += contrib

    @abstractmethod
    def print(self, prefix: str = ""):
        """Print operator detail."""


class Val(Op):
    """Constant operator."""

    def eval(self):
        """Return value of the variable."""

    def forward(self, wrt: "Var"):
        """Calculate grade of a constant."""
        if id(self.var) == id(wrt):
            self.var.forward_value = 1.0
        else:
            self.var.forward_value = 0.0

    def print(self, prefix: str = ""):
        """Print constant value description."""
        print(
            prefix + f"{self.var.name if self.var.name else id(self.var)}| "
            f"val={self.var.eval_value} grad={self.var.adjoint_value} "
            f"forward={self.var.forward_value}"
        )

    def _grad(self):
        """No children so nothing much to do."""


class Add(Op):
    """Add operator."""

    def eval(self):
        """Return result of addition."""
        self.var.eval_value = sum(v.eval_value for v in self.var.children)

    def forward(self, wrt: "Var"):
        """Calculate grad of addition."""
        self.var.forward_value = (
            self.var.children[0].forward_value + self.var.children[1].forward_value
        )

    def _grad(self):
        """Progagate grad values to children of add operator."""
        self.var.children[0].op.accum_grad(self.var.adjoint_value)
        self.var.children[1].op.accum_grad(self.var.adjoint_value)

    def print(self, prefix: str = ""):
        """Print add operator description."""
        print(prefix + f"{self.var.name}| "
              f"val={self.var.eval_value} adjoint={self.var.adjoint_value} "
              f"forward={self.var.forward_value}")
        for child in self.var.children:
            child.print(prefix + "   ")


class Mult(Op):
    """Multiply operator."""

    def eval(self):
        """Return result of multiplication."""
        self.var.eval_value = reduce(
            operator.mul, (v.eval_value for v in self.var.children), 1.0
        )

    def forward(self, wrt: "Var"):
        """Calculate grad of multiplication."""
        self.var.forward_value = (
            self.var.children[0].forward_value * self.var.children[1].eval_value
            + self.var.children[0].eval_value * self.var.children[1].forward_value
        )

    def _grad(self):
        """Progagate grad values to children of multiply operator."""
        self.var.children[0].op.accum_grad(self.var.adjoint_value * self.var.children[1].eval_value)
        self.var.children[1].op.accum_grad(self.var.adjoint_value * self.var.children[0].eval_value)

    def print(self, prefix: str = ""):
        """Print multiplication description."""
        print(prefix + f"{self.var.name}| "
        f"val={self.var.eval_value} forward={self.var.forward_value} "
        f"adjoint={self.var.adjoint_value}")
        for child in self.var.children:
            child.print(prefix + "   ")


class Var:
    """Node in a graph."""

    def __init__(self, name: str = ""):
        """Intialize node, by default grad & adjoint are 0.0."""
        self.name = name
        self.eval_value: float = float("nan")
        self.forward_value: float = float("nan")
        self.adjoint_value: float = float("nan")
        self.op: Op = Val(self)  # pylint: disable=invalid-name
        self.parents: List["Var"] = []
        self.children: List["Var"] = []

    def assign(self, val: float):
        """Assign value to the node."""
        self.eval_value = val

    def add_child(self, child: "Var"):
        """Add given node as a child."""
        self.children.append(child)
        child.parents.append(self)

    def add_parent(self, parent: "Var"):
        """Add given node as parent."""
        self.parents.append(parent)
        parent.children.append(self)

    def __add__(self, other: "Var"):
        """Return new node that represents add operation on self and other."""
        new = Var("+")
        new.op = Add(new)
        new.add_child(self)
        new.add_child(other)
        return new

    def __mul__(self, other: "Var"):
        """Return new node that represents multiplication operation on self and other."""
        new = Var("*")
        new.op = Mult(new)
        new.add_child(self)
        new.add_child(other)
        return new

    def value(self) -> float:
        """Evaluate and return value of the node."""
        for node in self.dfs():
            node.op.eval()
        return self.eval_value

    def adjoint(self) -> float:
        """Get adjoint value."""
        return self.adjoint_value

    def print(self, prefix: str = ""):
        """Print node and children on console."""
        self.op.print(prefix)

    def forward(self, wrt: "Var") -> float:
        """Calculate forward gradient with respect to given node and return its value.

        This also triggers evaluation.
        """
        self.value()
        for node in self.dfs():
            node.op.forward(cast("Var", wrt))
        return self.forward_value

    def grad(self):
        """Calculate backward gradient.

        Value of gradient can be fetched using adjoint function on the node.
        """
        self.value()
        self.clear_grad()
        self.op.grad(True)
        for node in self.bfs():
            node.op.grad()

    def clear_grad(self):
        """Clear out all values of grad in graph."""
        for node in self.dfs():
            node.adjoint_value = 0.0

    def dfs(self) -> Iterable["Var"]:
        """Return nodes of the graph rooted with this node in depth first order."""
        pending: Deque[Tuple["Var", Boolean]] = deque()
        seen = set()
        pending.append((self, False))
        seen.add(self)
        while pending:
            current, expanded = pending.pop()
            if expanded:
                yield current
            else:
                if current.children:
                    pending.append((current, True))
                    for child in current.children:
                        if child not in seen:
                            pending.append((child, False))
                            seen.add(child)
                else:
                    yield current

    def bfs(self) -> Iterable["Var"]:
        """Return nodes of the graph rooted with this node in breadth first order."""
        pending: Deque["Var"] = deque()
        seen = set()
        yielded = set()
        pending.append(self)
        seen.add(self)
        while pending:
            current = pending.pop()
            if not current.parents \
                or all(parent in yielded for parent in current.parents):
                yield current
                yielded.add(current)
                for child in current.children:
                    if child not in seen:
                        pending.append(child)
                        seen.add(child)
            else:
                pending.appendleft(current)
