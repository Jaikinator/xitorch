import warnings
import torch
from typing import Callable, List, Optional


# from torch.utils.tensorboard import SummaryWriter

def gd(fcn: Callable[..., torch.Tensor], x0: torch.Tensor, params: List,
       # gd parameters
       step: float = 1e-3,
       gamma: float = 0.9,
       # stopping conditions
       maxiter: int = 1000,
       miniter: int = 1,
       f_tol: float = 0.0,
       f_rtol: float = 1e-8,
       x_tol: float = 0.0,
       x_rtol: float = 1e-8,
       # misc parameters
       verbose=False,
       writer = None, # handle erros in TerminationCondition
       diverge = torch.tensor(float('inf')), # handle erros in TerminationCondition
       maxdivattamps=50,
       get_misc = False,
       **unused):
    r"""
    Vanilla gradient descent with momentum. The stopping conditions use OR criteria.
    The update step is following the equations below.

    .. math::
        \mathbf{v}_{t+1} &= \gamma \mathbf{v}_t - \eta \nabla_{\mathbf{x}} f(\mathbf{x}_t) \\
        \mathbf{x}_{t+1} &= \mathbf{x}_t + \mathbf{v}_{t+1}

    Keyword arguments
    -----------------
    step: float
        The step size towards the steepest descent direction, i.e. :math:`\eta` in
        the equations above.
    gamma: float
        The momentum factor, i.e. :math:`\gamma` in the equations above.
    maxiter: int
        Maximum number of iterations.
    f_tol: float or None
        The absolute tolerance of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    """
    if type(diverge) is not torch.Tensor:
        #handle if input is not tensor
        diverge = torch.tensor(diverge, dtype= torch.float64)

    x = x0.clone()
    stop_cond = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol, verbose,
                                     miniter, writer, abs(diverge) ,maxdivattamps)
    fprev = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)
    v = torch.zeros_like(x)
    for i in range(maxiter):
        f, dfdx = fcn(x, *params)

        # update the step
        v = (gamma * v - step * dfdx).detach()
        xprev = x.detach()
        x = (xprev + v).detach()

        # check the stopping conditions
        to_stop = stop_cond.to_stop(i, x, xprev, f, fprev)

        if not torch.isinf(diverge):
            if i == 0:
                initdiff = torch.abs(torch.abs(diverge) - torch.abs(f))

            to_stop = stop_cond.to_stop(i, x, xprev, f, fprev, initdiff = initdiff)

            if stop_cond.divergence:
                # if leaning divergence
                break

            elif to_stop:
                break

        if to_stop:
            break

        fprev = f
    x = stop_cond.get_best_x(x)

    if get_misc:
        f = stop_cond.get_misc(i, x, xprev, f, fprev)
        return x, f
    else:
        return x


def adam(fcn: Callable[..., torch.Tensor], x0: torch.Tensor, params: List,
         # gd parameters
         step: float = 1e-3,
         beta1: float = 0.9,
         beta2: float = 0.999,
         eps: float = 1e-8,
         # stopping conditions
         maxiter: int = 1000,
         miniter: int = 1,
         f_tol: float = 0.0,
         f_rtol: float = 0.0,
         x_tol: float = 0.0,
         x_rtol: float = 0.0,
         # misc parameters
         verbose=False,
         writer=None,
         diverge=torch.tensor(float('inf')),
         maxdivattamps = 50,
         get_misc = False,
         **unused):
    r"""
    Adam optimizer by Kingma & Ba (2015). The stopping conditions use OR criteria.
    The update step is following the equations below.

    .. math::
        \mathbf{g}_t &= \nabla_{\mathbf{x}} f(\mathbf{x}_{t-1}) \\
        \mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \\
        \mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \\
        \hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t) \\
        \hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t) \\
        \mathbf{x}_t &= \mathbf{x}_{t-1} - \alpha \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)

    Keyword arguments
    -----------------
    step: float
        The step size towards the descent direction, i.e. :math:`\alpha` in
        the equations above.
    beta1: float
        Exponential decay rate for the first moment estimate.
    beta2: float
        Exponential decay rate for the first moment estimate.
    eps: float
        Small number to prevent division by 0.
    maxiter: int
        Maximum number of iterations.
    miniter: int
        Minimum number of iterations.

    f_tol: float or None
        The absolute tolerance of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    verbose: bool
        Whether to print the progress.

    writer: torch.utils.tensorboard.SummaryWriter
        The tensorboard writer.
    diverge: torch.Tensor or float
        The divergence value. If the norm of the gradient is larger than this value,
        the algorithm will stop.
    maxdivattamps: int
        The maximum number of times the divergence is checked before the algorithm stops.
    get_misc: bool
        Whether to return the misc parameters.

    """
    if type(diverge) is not torch.Tensor:
        #handle if input is not type torch.Tensor
        diverge = torch.tensor(diverge, dtype= torch.float64)

    x = x0.clone()

    stop_cond = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol, verbose,
                                     miniter, writer, abs(diverge) ,maxdivattamps)
    fprev = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)
    v = torch.zeros_like(x)
    m = torch.zeros_like(x)
    beta1t = beta1
    beta2t = beta2

    for i in range(maxiter):

        f, dfdx = fcn(x, *params)
        f = f.detach()
        dfdx = dfdx.detach()

        # update the step
        m = beta1 * m + (1 - beta1) * dfdx
        v = beta2 * v + (1 - beta2) * dfdx ** 2
        mhat = m / (1 - beta1t)
        vhat = v / (1 - beta2t)
        beta1t *= beta1
        beta2t *= beta2
        xprev = x.detach()
        x = (xprev - step * mhat / (vhat ** 0.5 + eps)).detach()

        # check the stopping conditions

        if not torch.isinf(diverge):
            if i == 0:
                initdiff = torch.abs(torch.abs(diverge) - torch.abs(f))

            to_stop = stop_cond.to_stop(i, x, xprev, f, fprev, initdiff = initdiff)

            if stop_cond.divergence:
                # if leaning divergence
                break

            elif to_stop:
                break

        else:

            to_stop = stop_cond.to_stop(i, x, xprev, f, fprev)

            if to_stop:
                break

        fprev = f

    x = stop_cond.get_best_x(x)

    if get_misc:
        f = stop_cond.get_misc(i, x, xprev, f, fprev)
        return x, f
    else:
        return x



class TerminationCondition(object):
    def __init__(self, f_tol: float, f_rtol: float, x_tol: float, x_rtol: float,
                 verbose: bool,miniter, writer, diverge : torch.Tensor,maxdivattamps : int):
        # writer for tensorboard just = None for exeption handling
        # divergence = None if you do not want divergence controll
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.verbose = verbose

        self.miniter = miniter
        self.writer = writer

        self.diverge = diverge  # param to check for divergence
        self.divergeattempt = 0 #give adam a second chance
        self.divfval =torch.zeros(maxdivattamps, dtype= torch.float64) #save divergence values evaluate error
        self.divergence = False  # bool for diverged output

        self.nan = False

        self._ever_converge = False
        self._best_i = -1
        self._max_i = -1
        self._best_dxnorm = float("inf")
        self._best_df = float("inf")
        self._best_f = float("inf")
        self._best_x: Optional[torch.Tensor] = None

    def to_stop(self, i: int, xnext: torch.Tensor, x: torch.Tensor,
                f: torch.Tensor, fprev: torch.Tensor, initdiff=torch.tensor(0)) -> bool:
        xnorm: float = float(x.detach().norm().item())
        dxnorm: float = float((x - xnext).detach().norm().item())
        fabs: float = float(f.detach().abs().item())
        df: float = float((fprev - f).detach().abs().item())
        fval: float = float(f.detach().item())

        xtcheck = dxnorm < self.x_tol
        xrcheck = dxnorm < self.x_rtol * xnorm
        ytcheck = df < self.f_tol
        yrcheck = df < self.f_rtol * fabs
        converge = xtcheck or xrcheck or ytcheck or yrcheck  # stops if result get nan

        if self.verbose:
            if self.writer != None:
                self.writer.add_scalar('Loss/dx', dxnorm, i)
                self.writer.add_scalar('Loss/df', df, i)
                self.writer.add_scalar('Loss/f', f, i)
            if i == 0:
                print("   #:             f |        dx,        df")
            if converge and not torch.isnan(f) and i > self.miniter:
                print("Finish with convergence")
            if i == self.miniter:
                print("minimal number of iterations is reached now beginning to check for divergence")
            if i == 0 or ((i + 1) % 10) == 0 or converge:
                print(f"%4d: %.6e | %.3e, %.3e" % (i + 1, f, dxnorm, df))

        res = (i > self.miniter and converge)
        # get the best values

        if not self._ever_converge and res:
            self._ever_converge = True
        if i > self._max_i:
            self._max_i = i
        if fval < self._best_f:
            self._best_f = fval
            self._best_x = x
            self._best_dxnorm = dxnorm
            self._best_df = df
            self._best_i = i

        if torch.isnan(f):
            self.nan = True
            res = True

        if not torch.isinf(self.diverge):

            if i > self.miniter and (i % 5000) == 0 and self.divergeattempt == 0:

                newdiff = torch.abs(self.diverge - torch.abs(f))

                self.divfval[0] = newdiff
                if newdiff > initdiff:
                    self.divergeattempt += 1
                    msg = f"Start check of divergence"
                    warnings.warn(msg)


            elif i > self.miniter and (i % 200) == 0 and 0 < self.divergeattempt < len(self.divfval)-1:

                newdiff = torch.abs(self.diverge - torch.abs(f))

                self.divfval[self.divergeattempt] = newdiff

                if self.divfval[self.divergeattempt] <= self.divfval[self.divergeattempt-1]:
                    self.divergeattempt = 0
                else:
                    self.divergeattempt += 1


            elif self.divergeattempt + 1 == len(self.divfval):
                count = 0
                for i in range(len(self.divfval)):
                    if i != 0:
                        if self.divfval[self.divergeattempt] <= self.divfval[self.divergeattempt-1]:
                            count += 1
                if count >= len(self.divfval):
                    self.divergeattempt = 0
                else:
                    self.divergence = True

        return res

    def get_best_x(self, x: torch.Tensor) -> torch.Tensor:
        # usually user set maxiter == 0 just to wrap the minimizer backprop
        if self.nan:
            warnings.warn("The minimizer does get nan.")
            return self._best_x

        elif self.divergence == True:
            warnings.warn("The minimizer divergence.")
            return self._best_x

        elif not self._ever_converge and self._max_i > -1:
            msg = ("The minimizer does not converge after %d iterations. "
                   "Best |dx|=%.4e, |df|=%.4e, f=%.4e" %
                   (self._max_i, self._best_dxnorm, self._best_df, self._best_f))
            warnings.warn(msg)
            assert isinstance(self._best_x, torch.Tensor)
            return self._best_x
        else:
            return x


    def get_misc(self, i: int, xnext: torch.Tensor, x: torch.Tensor,
                f: torch.Tensor, fprev: torch.Tensor) -> dict:
        """
        Returns: more informations about the learning
        """
        if self.nan or self.divergence or (not self._ever_converge and self._max_i > -1):
            out_dic = {"best_x" :  self._best_x,
                       "best_f": self._best_f,
                       "best_df": self._best_df,
                       "best_dcnorm": self._best_dxnorm,
                       "best_i": self._best_i,
                       "max_i": i }
            return out_dic

        else:

            if float(f) <= self._best_f:
                dxnorm: float = float((x - xnext).detach().norm().item())
                df: float = float((fprev - f).detach().abs().item())
                out_dic = {"best_x" : x,
                           "best_f" : f,
                           "best_df" : df,
                           "best_dcnorm" : dxnorm,
                           "best_i": self._best_i,
                           "max_i": i,
                           }
            else:
                out_dic = {"best_x": self._best_x,
                           "best_f": self._best_f,
                           "best_df": self._best_df,
                           "best_dcnorm": self._best_dxnorm,
                           "best_i": self._best_i,
                           "max_i": i}

                warnings.warn("Finish with convergence but misc will be best results, which wasn't the converged one.")
                return out_dic
        return out_dic