"""Graph related types."""
from collections import deque
from abc import ABC, abstractmethod
import math
from typing import Deque, Iterable, List, Union, cast, Tuple, get_args
from wsgiref.validate import PartialIteratorWrapper

Number = Union[float, int]
NodeType = Union[float, int, "Var"]

def close(left, right) -> bool:
    """Match 6 digits."""
    return math.isclose(left, right, abs_tol=0.0000009)

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

    def backward(self, root: bool = False):
        """Calculate adjoint if not root. Set adjoint to 1.0 otherwise."""
        if root:
            self.var.adjoint_value = 1.0
        else:
            self._backward()

    @abstractmethod
    def _backward(self):
        """Calculate adjoint of the node."""

    def accum_grad(self, contrib: float):
        """Accumulate grad value."""
        self.var.adjoint_value += contrib

    def print(self, prefix: str = ""):
        """Print operator detail."""
        print(
            prefix + f"{self.var.name}| "
            f"val={self.var.eval_value} grad={self.var.adjoint_value} "
            f"forward={self.var.forward_value}"
        )
        for child in self.var.children:
            child.print(prefix + "   ")


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

    def _backward(self):
        """No children so nothing much to do."""


class Add(Op):
    """Add operator."""

    def eval(self):
        """Return result of addition."""
        self.var.eval_value = (
            self.var.children[0].eval_value + self.var.children[1].eval_value
        )

    def forward(self, wrt: "Var"):
        """Calculate grad of addition."""
        self.var.forward_value = (
            self.var.children[0].forward_value + self.var.children[1].forward_value
        )

    def _backward(self):
        """Progagate grad values to children of add operator."""
        self.var.children[0].op.accum_grad(self.var.adjoint_value)
        self.var.children[1].op.accum_grad(self.var.adjoint_value)


class Sub(Op):
    """Subtract operator."""

    def eval(self):
        """Return result of subtraction."""
        self.var.eval_value = (
            self.var.children[0].eval_value - self.var.children[1].eval_value
        )

    def forward(self, wrt: "Var"):
        """Calculate grad of subtraction."""
        self.var.forward_value = (
            self.var.children[0].forward_value - self.var.children[1].forward_value
        )

    def _backward(self):
        """Progagate grad values to children of subtract operator."""
        self.var.children[0].op.accum_grad(self.var.adjoint_value)
        self.var.children[1].op.accum_grad(-self.var.adjoint_value)


class Neg(Op):
    """Negation operator."""

    def eval(self):
        """Return result of negation."""
        self.var.eval_value = -self.var.children[0].eval_value

    def forward(self, wrt: "Var"):
        """Calculate grad of negation."""
        self.var.forward_value = -self.var.children[0].forward_value

    def _backward(self):
        """Progagate grad values to children of negation operator."""
        self.var.children[0].op.accum_grad(-self.var.adjoint_value)

class Mult(Op):
    """Multiply operator."""

    def eval(self):
        """Return result of multiplication."""
        self.var.eval_value = (
            self.var.children[0].eval_value * self.var.children[1].eval_value
        )

    def forward(self, wrt: "Var"):
        """Calculate grad of multiplication."""
        self.var.forward_value = (
            self.var.children[0].forward_value * self.var.children[1].eval_value
            + self.var.children[0].eval_value * self.var.children[1].forward_value
        )

    def _backward(self):
        """Progagate grad values to children of multiply operator."""
        self.var.children[0].op.accum_grad(
            self.var.adjoint_value * self.var.children[1].eval_value
        )
        self.var.children[1].op.accum_grad(
            self.var.adjoint_value * self.var.children[0].eval_value
        )

class Pow(Op):
    """Power operator. Only allows constant values as power."""

    def eval(self):
        """Return result of power."""
        self.var.eval_value = (
            self.var.children[0].eval_value ** self.var.children[1].eval_value
        )

    def forward(self, wrt: "Var"):
        """Calculate grad of multiplication."""
        val = self.var.eval_value
        power_val = self.var.children[1].eval_value
        quotient_val = self.var.children[0].eval_value
        power_d = self.var.children[1].forward_value
        quotient_d = self.var.children[0].forward_value
        self.var.forward_value = (
            power_val * (quotient_val ** (power_val-1)) * quotient_d
        ) if close(power_d, 0.0) else (
            val * (
                power_d * math.log(quotient_val, math.e)
                + (power_val * quotient_d / quotient_val)
            )
        )


    def _backward(self):
        """Progagate grad values to children of multiply operator."""
        val = self.var.eval_value
        val_d = self.var.adjoint_value
        power_val = self.var.children[1].eval_value
        quotient_val = self.var.children[0].eval_value
        self.var.children[0].op.accum_grad(
            val_d * (power_val) * (quotient_val ** (power_val-1))
        )
        self.var.children[1].op.accum_grad(
            (
                float('nan')
            ) if quotient_val <= 0.0 else (
                 val_d * val * math.log(quotient_val, math.e)
            )
        )


class Div(Op):
    """Division operator."""

    def eval(self):
        """Return result of division."""
        left_val = self.var.children[0].eval_value
        right_val = self.var.children[1].eval_value
        self.var.eval_value = left_val / right_val

    def forward(self, wrt: "Var"):
        """Calculate grad of division."""
        left_val = self.var.children[0].eval_value
        left_d = self.var.children[0].forward_value
        right_val = self.var.children[1].eval_value
        right_d = self.var.children[1].forward_value
        self.var.forward_value = (
            left_d / right_val
            + right_d * -1 * left_val * right_val**-2
        )

    def _backward(self):
        """Progagate grad values to children of multiply operator."""
        d_self = self.var.adjoint_value
        left_val = self.var.children[0].eval_value
        right_val = self.var.children[1].eval_value
        self.var.children[0].op.accum_grad(d_self / right_val)
        self.var.children[1].op.accum_grad(d_self*-1*left_val*right_val**-2)



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

    def __add__(self, other: NodeType):
        """Return new node that represents add operation on self and other."""
        resolved = Var.resolve(other)
        new = Var("+")
        new.op = Add(new)
        new.add_child(self)
        new.add_child(resolved)
        return new

    def __mul__(self, other: NodeType):
        """Return new node that represents multiplication operation on self and other."""
        resolved = Var.resolve(other)
        new = Var("*")
        new.op = Mult(new)
        new.add_child(self)
        new.add_child(resolved)
        return new

    def __truediv__(self, other: NodeType):
        """Return new node that represents division operation on self and other."""
        resolved = Var.resolve(other)
        new = Var("/")
        new.op = Div(new)
        new.add_child(self)
        new.add_child(resolved)
        return new

    def __sub__(self, other: NodeType):
        """Return new node that represents subtraciton operator on self and other."""
        resolved = Var.resolve(other)
        new = Var("-")
        new.op = Sub(new)
        new.add_child(self)
        new.add_child(resolved)
        return new

    def __neg__(self):
        """Return new node that represents negation on self."""
        new = Var("-")
        new.op = Neg(new)
        new.add_child(self)
        return new

    def __pow__(self, other):
        """Return new node that represents self^other."""
        new = Var("^")
        new.op = Pow(new)
        new.add_child(self)
        power = Var.resolve(other)
        power.op = Val(power)
        new.add_child(power)
        return new

    def value(self) -> float:
        """Evaluate and return value of the node."""
        for node in self.dfs():
            node.op.eval()
        return self.eval_value

    def grad(self) -> float:
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

    def backward(self):
        """Calculate backward gradient.

        Value of gradient can be fetched using adjoint function on the node.
        """
        self.value()
        self.clear_grad()
        self.op.backward(True)
        for node in self.bfs():
            node.op.backward()

    def clear_grad(self):
        """Clear out all values of grad in graph."""
        for node in self.dfs():
            node.adjoint_value = 0.0

    def dfs(self) -> Iterable["Var"]:
        """Return nodes of the graph rooted with this node in depth first order."""
        pending: Deque[Tuple["Var", bool]] = deque()
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
            if not current.parents or all(
                parent in yielded for parent in current.parents
            ):
                yield current
                yielded.add(current)
                for child in current.children:
                    if child not in seen:
                        pending.append(child)
                        seen.add(child)
            else:
                pending.appendleft(current)

    @classmethod
    def resolve(cls, node: NodeType) -> "Var":
        """Convert to Var if not already a Var."""
        if isinstance(node, get_args(Number)):
            new = Var(str(node))
            new.op = Val(new)
            new.assign(float(cast(Union[float, int], node)))
            return new
        return cast("Var", node)
