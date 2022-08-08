"""Graph related types."""
from functools import reduce
import operator

from abc import ABC, abstractmethod
from typing import List


class Op(ABC):
    """Operator in a graph."""

    def __init__(self, var: "Var"):
        """Initialize operator with graph node."""
        self.var = var

    @abstractmethod
    def eval(self):
        """Evaluate the value of the operator."""

    @abstractmethod
    def grad(self, wrt: "Var"):  # pylint: disable:invalid-name
        """Calculate grade with respect to given variable."""

    @abstractmethod
    def print(self, prefix: str = ""):
        """Print operator detail."""


class Val(Op):
    """Constant operator."""

    def eval(self):
        """Return value of the variable."""

    def grad(self, wrt: "Var"):
        """Calculate grade of a constant."""
        if id(self.var) == id(wrt):
            self.var.grad_value = 1.0
        else:
            self.var.grad_value = 0.0

    def print(self, prefix: str = ""):
        """Print constant value description."""
        print(
            prefix + f"{self.var.name if self.var.name else id(self.var)}: "
            f"val={self.var.eval_value} grad={self.var.grad_value}"
        )


class Add(Op):
    """Add operator."""

    def eval(self):
        """Return result of addition."""
        self.var.children[0].op.eval()
        self.var.children[1].op.eval()
        self.var.eval_value = sum(v.eval_value for v in self.var.children)

    def grad(self, wrt: "Var"):
        """Calculate grad of addition."""
        self.var.children[0].op.grad(wrt)
        self.var.children[1].op.grad(wrt)
        self.var.grad_value = (
            self.var.children[0].grad_value + self.var.children[1].grad_value
        )

    def print(self, prefix: str = ""):
        """Print add operator description."""
        print(prefix + f"Add: val={self.var.eval_value} grad={self.var.grad_value}")
        for child in self.var.children:
            child.print(prefix + "   ")


class Mult(Op):
    """Multiply operator."""

    def eval(self):
        """Return result of multiplication."""
        self.var.children[0].op.eval()
        self.var.children[1].op.eval()
        self.var.eval_value = reduce(
            operator.mul, (v.eval_value for v in self.var.children), 1.0
        )

    def grad(self, wrt: "Var"):
        """Calculate grad of multiplication."""
        self.var.children[0].op.grad(wrt)
        self.var.children[1].op.grad(wrt)
        self.var.grad_value = (
            self.var.children[0].grad_value * self.var.children[1].eval_value
            + self.var.children[0].eval_value * self.var.children[1].grad_value
        )

    def print(self, prefix: str = ""):
        """Print multiplication description."""
        print(prefix + f"Mul: val={self.var.eval_value} grad={self.var.grad_value}")
        for child in self.var.children:
            child.print(prefix + "   ")


class Var:
    """Node in a graph."""

    def __init__(self, name: str = ""):
        """Intialize node, by default grad & adjoint are 0.0."""
        self.name = name
        self.eval_value: float = float("nan")
        self.grad_value: float = float("nan")
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
        new = Var()
        new.op = Add(new)
        new.add_child(self)
        new.add_child(other)
        return new

    def __mul__(self, other: "Var"):
        """Return new node that represents multiplication operation on self and other."""
        new = Var()
        new.op = Mult(new)
        new.add_child(self)
        new.add_child(other)
        return new

    def value(self) -> float:
        """Evaluate and return value of the node."""
        self.op.eval()
        return self.eval_value

    def print(self, prefix: str = ""):
        """Print node and children on console."""
        self.op.print(prefix)

    def grad(self, wrt: "Var") -> float:
        """Calculate grad value with respect to given node and return its value. This also triggers evaluation."""
        self.op.eval()
        self.op.grad(wrt)
        return self.grad_value
