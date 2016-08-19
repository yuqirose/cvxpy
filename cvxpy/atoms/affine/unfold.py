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

from cvxpy.atoms.affine.affine_atom import AffAtom
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class unfold(AffAtom):
    """ 
    Unfold a tensor variable into a tensor variable.
    The entries are stored in column-major order.
    """

    def __init__(self, expr, size, mode):
        self.rows = size[mode]
        self.cols = np.prod(size)/size[mode]
        super(unfold, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Unfold the value.
        """
        ndim = len(self.size)
        perm_order = np.roll(np.arange(ndim),self.mode-1)
        return np.reshape(np.transpose(values[0], perm_order), [self.size[self.mode-1],-1],"F")
        #return np.reshape(values[0], (self.rows, self.cols), "F")

    def validate_arguments(self):
        """Checks that the new shape has the same number of entries as the old.
        """
        old_len = np.prod(self.args[0].size)
        new_len = self.rows*self.cols
        if not old_len == new_len:
            print old_len
            print new_len
            raise ValueError(
                "Invalid reshape dimensions (%i, %i)." % (self.rows, self.cols)
            )

    def size_from_args(self):
        """Returns the shape from the arguments.
        """
        return (self.rows, self.cols)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.rows, self.cols]

    @staticmethod
    def graph_implementation(arg_objs, size, mode, data=None):
        """Convolve two vectors.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.unfold(arg_objs[0], size, mode), [])
