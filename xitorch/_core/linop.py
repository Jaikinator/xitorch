import inspect
from typing import Sequence, Union
import warnings
import traceback
import torch
from abc import abstractmethod, abstractproperty
from contextlib import contextmanager
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from xitorch._core.editable_module import EditableModule
from xitorch.debug.modes import is_debug_enabled

__all__ = ["LinearOperator"]

class LinearOperator(EditableModule):
    """
    LinearOperator is a class designed to behave as a linear operator without
    explicitly determining the matrix.
    This LinearOperator can operate a batch of linear operator and it has shape
    of (B1,B2,...,Bb,p,q) where B* is the (optional) batch dimensions.
    """
    @classmethod
    def m(cls, mat:torch.Tensor, is_hermitian:Union[bool,None]=None):
        if is_hermitian is None:
            if mat.shape[-2] != mat.shape[-1]:
                is_hermitian = False
            else:
                is_hermitian = torch.allclose(mat, mat.transpose(-2,-1))

        if is_hermitian:
            return _MatrixHermitLinOp(mat)
        else:
            return _MatrixNonHermitLinOp(mat)

    def __init__(self, shape:Sequence[int],
            is_hermitian:bool = False,
            dtype:Union[torch.dtype,None] = None,
            device:Union[torch.device,None] = None) -> None:

        super(LinearOperator, self).__init__()
        if len(shape) < 2:
            raise RuntimeError("The shape must have at least 2 dimensions")
        self._shape = shape
        self._batchshape = list(shape[:-2])
        self._is_hermitian = is_hermitian
        self._dtype = dtype if dtype is not None else torch.float32
        self._device = device if device is not None else torch.device("cpu")
        if is_hermitian and shape[-1] != shape[-2]:
            raise RuntimeError("The object is indicated as Hermitian, but the shape is not square")

        # check which methods are implemented
        self._is_mv_implemented = self.__check_if_implemented("_mv")
        self._is_mm_implemented = self.__check_if_implemented("_mm")
        self._is_rmv_implemented = self.__check_if_implemented("_rmv")
        self._is_rmm_implemented = self.__check_if_implemented("_rmm")
        self._is_fullmatrix_implemented = self.__check_if_implemented("_fullmatrix")
        if not self._is_mv_implemented:
            raise RuntimeError("LinearOperator must have at least ._mv() method implemented")
        if self._is_hermitian and (self._is_rmv_implemented or self._is_rmm_implemented):
            warnings.warn("The LinearOperator is Hermitian with implemented rmv or rmm. We will use the mv and mm methods instead")

        # caches
        self._matrix_defined = False
        self._matrix = torch.tensor([])

    @abstractmethod
    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod
    def _rmv(self, x:torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod # (optional)
    def _mm(self, x:torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod
    def _rmm(self, x:torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod
    def _fullmatrix(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _getparamnames(self) -> Sequence[str]:
        pass

    # linear operators must have a set of parameters that affects most of
    # the methods (i.e. mm, mv, rmm, rmv)
    def getlinopparams(self) -> Sequence[torch.Tensor]:
        return self.getuniqueparams("mm")

    @contextmanager
    def uselinopparams(self, *params):
        methodname = "mm"
        try:
            _orig_params_ = self.getuniqueparams(methodname)
            self.setuniqueparams(methodname, *params)
            yield self
        except Exception as exc:
            raise exc
            # traceback.print_exc()
        finally:
            self.setuniqueparams(methodname, *_orig_params_)

    ############# implemented functions ################
    def mv(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-vector operation to vector x with shape (...,q).
        The batch dimensions of x need not be the same as the batch dimensions
        of the LinearOperator, but it must be broadcastable.

        Arguments
        ---------
        * x: torch.tensor (...,q)
            The vector where the linear operation is operated at.

        Returns
        -------
        * y: torch.tensor (...,p)
            The result of the linear operation.
        """
        if x.shape[-1] != self.shape[-1]:
            raise RuntimeError("Cannot operate .mv on shape %s. Expected (...,%d)" %\
                (str(tuple(x.shape)), self.shape[-1]))

        return self._mv(x)

    def mm(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-matrix operation to matrix x with shape (...,q,r).
        The batch dimensions of x need not be the same as the batch dimensions
        of the LinearOperator, but it must be broadcastable.

        Arguments
        ---------
        * x: torch.tensor (...,q,r)
            The matrix where the linear operation is operated at.

        Returns
        -------
        * y: torch.tensor (...,p,r)
            The result of the linear operation.
        """
        if x.shape[-2] != self.shape[-1]:
            raise RuntimeError("Cannot operate .mm on shape %s. Expected (...,%d,*)" %\
                (str(tuple(x.shape)), self.shape[-1]))

        xbatchshape = list(x.shape[:-2])
        if self._is_mm_implemented:
            return self._mm(x)
        else:
            # use batched mv as mm

            # move the last dimension to the very first dimension to be broadcasted
            if len(xbatchshape) < len(self._batchshape):
                xbatchshape = [1]*(len(self._batchshape)-len(xbatchshape)) + xbatchshape
            x1 = x.view(1, *xbatchshape, *x.shape[-2:])
            xnew = x1.transpose(0, -1).squeeze(-1) # (r,...,q)

            # apply batched mv and restore the initial shape
            ynew = self._mv(xnew) # (r,...,p)
            y = ynew.unsqueeze(-1).transpose(0,-1).squeeze(0) # (...,p,r)
            return y

    def rmv(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-vector adjoint operation to vector x with shape (...,p),
        i.e. A^H x.
        The batch dimensions of x need not be the same as the batch dimensions
        of the LinearOperator, but it must be broadcastable.

        Arguments
        ---------
        * x: torch.tensor (...,p)
            The vector where the adjoint linear operation is operated at.

        Returns
        -------
        * y: torch.tensor (...,q)
            The result of the adjoint linear operation.
        """
        if x.shape[-1] != self.shape[-2]:
            raise RuntimeError("Cannot operate .rmv on shape %s. Expected (...,%d)" %\
                (str(tuple(x.shape)), self.shape[-2]))

        if self._is_hermitian:
            return self._mv(x)
        elif not self._is_rmv_implemented:
            raise RuntimeError("The ._rmv() must be implemented to call .rmv() method")
        return self._rmv(x)

    def rmm(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-matrix adjoint operation to matrix x with shape (...,p,r),
        i.e. A^H X.
        The batch dimensions of x need not be the same as the batch dimensions
        of the LinearOperator, but it must be broadcastable.

        Arguments
        ---------
        * x: torch.tensor (...,p,r)
            The matrix where the adjoint linear operation is operated at.

        Returns
        -------
        * y: torch.tensor (...,q,r)
            The result of the adjoint linear operation.
        """
        if x.shape[-2] != self.shape[-2]:
            raise RuntimeError("Cannot operate .rmm on shape %s. Expected (...,%d,*)" %\
                (str(tuple(x.shape)), self.shape[-2]))

        if self._is_hermitian:
            return self.mm(x)

        xbatchshape = list(x.shape[:-2])
        if self._is_rmm_implemented:
            return self._rmm(x)
        elif not self._is_rmv_implemented:
            raise RuntimeError("The ._rmv() or ._rmm() must be implemented to call .rmm() method")
        else:
            # use batched mv as mm

            # move the last dimension to the very first dimension to be broadcasted
            if len(xbatchshape) < len(self._batchshape):
                xbatchshape = [1]*(len(self._batchshape)-len(xbatchshape)) + xbatchshape
            x1 = x.view(1, *xbatchshape, *x.shape[-2:]) # (1,...,p,r)
            xnew = x1.transpose(0, -1).squeeze(-1) # (r,...,p)

            # apply batched mv and restore the initial shape
            ynew = self._rmv(xnew) # (r,...,q)
            y = ynew.unsqueeze(-1).transpose(0,-1).squeeze(0) # (...,q,r)
            return y

    def fullmatrix(self) -> torch.Tensor:
        if not self._matrix_defined:
            if self._is_fullmatrix_implemented:
                self._matrix = self._fullmatrix()
            else:
                nq = self._shape[-1]
                V = torch.eye(nq, dtype=self._dtype, device=self._device) # (nq,nq)
                self._matrix = self.mm(V) # (B1,B2,...,Bb,np,nq)
            self._matrix_defined = True
        return self._matrix

    def scipy_linalg_op(self):
        to_tensor = lambda x: torch.tensor(x, dtype=self.dtype, device=self.device)
        return spLinearOperator(
            shape=self.shape,
            matvec =lambda v: self.mv (to_tensor(v)).detach().numpy(),
            rmatvec=lambda v: self.rmv(to_tensor(v)).detach().numpy(),
            matmat =lambda v: self.mm (to_tensor(v)).detach().numpy(),
            rmatmat=lambda v: self.rmm(to_tensor(v)).detach().numpy(),
        )

    def getparamnames(self, methodname:str, prefix:str="") -> Sequence[str]:
        if methodname in ["mv", "rmv", "mm", "rmm"]:
            names = self._getparamnames()
            return [(prefix+s) for s in names] if prefix != "" else names
        elif methodname == "fullmatrix":
            return [prefix+"_matrix"]
        else:
            raise KeyError("getparamnames for method %s is not implemented" % methodname)

    ############# cached properties ################
    @property
    def H(self):
        return AdjointLinearOperator(self)

    ############# properties ################
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    # implementation
    @property
    def is_mv_implemented(self) -> bool:
        return self._is_mv_implemented

    @property
    def is_mm_implemented(self) -> bool:
        return self._is_mm_implemented

    @property
    def is_rmv_implemented(self) -> bool:
        return self._is_rmv_implemented

    @property
    def is_rmm_implemented(self) -> bool:
        return self._is_rmm_implemented

    @property
    def is_fullmatrix_implemented(self) -> bool:
        return self._is_fullmatrix_implemented

    ############ debug functions ##############
    def check(self, warn:Union[bool,None]=None) -> None:
        """
        Perform checks to make sure the linear operator behaves as a proper
        linear operator.

        Arguments
        ---------
        * warn: bool or None
            If True, then raises a warning to the user that the check might slow
            down the program. This is to remind the user to turn off the check
            when not in a debugging mode.
            If None, it will raise a warning if it runs not in a debug mode, but
            will be silent if it runs in a debug mode.

        Exceptions
        ----------
        * RuntimeError
            Raised if an error is raised when performing linear operations of the
            object (e.g. calling .mv(), .mm(), etc)
        * AssertionError
            Raised if the linear operations do not behave as proper linear operations.
            (e.g. not scaling linearly)
        """
        if warn is None:
            warn = not is_debug_enabled()
        if warn:
            msg = "The linear operator check is performed. This might slow down your program."
            warnings.warn(msg, stacklevel=2)
        checklinop(self)

    ############ private functions #################
    def __check_if_implemented(self, methodname:str) -> bool:
        this_method = getattr(self, methodname).__func__
        base_method = getattr(LinearOperator, methodname)
        return this_method is not base_method

class AdjointLinearOperator(LinearOperator):
    def __init__(self, obj:LinearOperator):
        super(AdjointLinearOperator, self).__init__(
            shape = (*obj.shape[:-2], obj.shape[-1], obj.shape[-2]),
            is_hermitian = obj.is_hermitian,
            dtype = obj.dtype,
            device = obj.device
        )
        self.obj = obj

    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        if not self.obj.is_rmv_implemented:
            raise RuntimeError("The ._rmv of must be implemented to call .H.mv()")
        return self.obj._rmv(x)

    def _rmv(self, x:torch.Tensor) -> torch.Tensor:
        return self.obj._mv(x)

    def _getparamnames(self) -> Sequence[str]:
        return ["obj."+s for s in self.obj._getparamnames()]

    @property
    def H(self):
        return self.obj

# distinguishing the classes of Hermitian and non-Hermitian matrices to suppress
# the warnings about redundant implementation of rmm and rmv
class _MatrixNonHermitLinOp(LinearOperator):
    def __init__(self, mat:torch.Tensor) -> None:

        super(_MatrixNonHermitLinOp, self).__init__(
            shape = mat.shape,
            is_hermitian = False,
            dtype = mat.dtype,
            device = mat.device
        )
        self.mat = mat

    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

    def _mm(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat, x)

    def _rmv(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat.transpose(-2,-1), x.unsqueeze(-1)).squeeze(-1)

    def _rmm(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat.transpose(-2,-1), x)

    def _fullmatrix(self) -> torch.Tensor:
        return self.mat

    def _getparamnames(self) -> Sequence[str]:
        return ["mat"]

class _MatrixHermitLinOp(LinearOperator):
    def __init__(self, mat:torch.Tensor) -> None:
        super(_MatrixHermitLinOp, self).__init__(
            shape = mat.shape,
            is_hermitian = True,
            dtype = mat.dtype,
            device = mat.device
        )
        # make sure the gradient is symmetrically distributed in the matrix
        self.mat = (mat + mat.transpose(-2,-1)) * 0.5

    def _mv(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

    def _mm(self, x:torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mat, x)

    def _fullmatrix(self) -> torch.Tensor:
        return self.mat

    def _getparamnames(self) -> Sequence[str]:
        return ["mat"]

    @property
    def H(self):
        return self

def checklinop(linop:LinearOperator) -> None:
    """
    Check if the implemented mv and mm can receive the possible shapes and returns
    the correct shape. If an error is found, then this function raise AssertionError.

    Argument
    --------
    * linop: LinearOperator instance
        The instance of LinearOperator to be checked

    Exception
    ---------
    * AssertionError
        Raised if there is a shape mismatch
    * RuntimeError
        Raised if there is an error when evaluating the .mv, .mm, .rmv, or .rmm methods
    """
    shape = linop.shape
    p, q = shape[-2:]
    batchshape = shape[:-2]

    def runtest(methodname, xshape, yshape):
        x = torch.rand(xshape, dtype=linop.dtype, device=linop.device)
        fcn = getattr(linop, methodname)
        try:
            y = fcn(x)
        except:
            s = traceback.format_exc()
            msg = "An error is raised from .%s with input shape: %s (linear operator shape: %s)\n" % \
                (methodname, tuple(xshape), tuple(linop.shape))
            msg += "--- full traceback ---\n%s" % s
            raise RuntimeError(msg)
        msg = "The output shape of .%s is not correct. Input: %s, expected output: %s, output: %s" % \
            (methodname, tuple(x.shape), tuple(yshape), tuple(y.shape))
        assert list(y.shape) == list(yshape), msg

        # linearity test
        x2 = 1.25*x
        y2 = fcn(x2)
        assert torch.allclose(y2, 1.25*y), "Linearity check fails"
        y0 = fcn(0*x)
        assert torch.allclose(y0, y*0), "Linearity check (with 0) fails"

        # batched test
        xnew = torch.cat((x.unsqueeze(0), x2.unsqueeze(0)), dim=0)
        ynew = fcn(xnew) # (2, ..., q)
        msg = "Batched test fails (expanding batches changes the results)"
        assert torch.allclose(ynew[0], y), msg
        assert torch.allclose(ynew[1], y2), msg

    # generate shapes
    mv_xshapes = [
        (q,),
        (1,q),
        (1,1,q),
        (*batchshape, q),
        (1, *batchshape, q),
    ]
    mv_yshapes = [
        (*batchshape, p),
        (*batchshape, p) if len(batchshape) >= 1 else (1, p),
        (*batchshape, p) if len(batchshape) >= 2 else (1, 1, p),
        (*batchshape, p),
        (1, *batchshape, p)
    ]
    # test matvec and matmat, run input in multiple shapes to make sure no error is raised
    r = 2
    for (mv_xshape, mv_yshape) in zip(mv_xshapes, mv_yshapes):
        runtest("mv", mv_xshape, mv_yshape)
        runtest("mm", (*mv_xshape, r), (*mv_yshape, r))

    if not linop.is_rmv_implemented:
        return

    rmv_xshapes = [
        (p,),
        (1,p),
        (1,1,p),
        (*batchshape, p),
        (1, *batchshape, p),
    ]
    rmv_yshapes = [
        (*batchshape, q),
        (*batchshape, q) if len(batchshape) >= 1 else (1, q),
        (*batchshape, q) if len(batchshape) >= 2 else (1, 1, q),
        (*batchshape, q),
        (1, *batchshape, q)
    ]
    for (rmv_xshape, rmv_yshape) in zip(rmv_xshapes, rmv_yshapes):
        runtest("rmv", rmv_xshape, rmv_yshape)
        runtest("rmm", (*rmv_xshape, r), (*rmv_yshape, r))

if __name__ == "__main__":
    mat = torch.tensor([[1.2, 3.4], [2.1, 5.6]])
    matlinop = _MatrixLinOp(mat)
    matlinop.check()
