"""
Copyright 2016 Rose Yu

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from ... import settings as s
from ..leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp
import numpy as np


class VariableTensor(Leaf):
    """ The tensor variable class """
    # name - unique identifier.
    # size - variable size.

    def __init__(self, shape=None, name=None):
        self._shape = shape
        self._size = np.prod(shape)        
        self.id = lu.get_id()
        if name is None:
            self._name = "%s%d" % (s.VAR_PREFIX, self.id)
        else:
            self._name = name
        self.primal_value = None        
        super(VariableTensor, self).__init__()

    def is_positive(self):
        """Is the expression positive?
        """
        return False

    def is_negative(self):
        """Is the expression negative?
        """
        return False

    @property
    def shape(self):
        """Returns the dimensions of the expression.
        """
        return self._shape
    
    def size(self):
        """Returns the total number of the expression.
        """
        return self._size

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self._size, self._name]

    def name(self):
        return self._name

    def save_value(self, value):
        """Save the value of the primal variable.
        """
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    @value.setter
    def value(self, val):
        """Assign a value to the variable.
        """
        val = self._validate_value(val)
        self.save_value(val)

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        return NotImplemented #{self: sp.eye(self.shape[0]*self.shape[1]).tocsc()}

    def variables(self):
        """Returns itself as a variable.
        """
        return [self]

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_var(self.shape, self.id)
        return (obj, [])

    def __repr__(self):
        """String to recreate the object.
        """
        return "Variable(%d, %d, %d)" % self.shape
