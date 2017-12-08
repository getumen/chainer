import numpy

from chainer.backends import cuda
from chainer import optimizer

_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.k = 1.0


class FreeRexRule(optimizer.UpdateRule):
    """Update rule of AdaGrad.

    See : http://proceedings.mlr.press/v65/cutkosky17a

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        k (float): Upper bound of $L_t/L?{t-1}$ where L_t is max Euclidean norm of gradient until current iteration.

    """

    def __init__(self, parent_hyperparam=None, k=None):
        super(FreeRexRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if k is not None:
            self.hyperparam.k = k

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['cum_grad'] = xp.zeros_like(param.data)
        self.state['L'] = 0.0
        self.state['a'] = 0.0
        self.state['e'] = 0.0

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return

        k = self.hyperparam.k
        h = self.state['cum_grad']
        lip = self.state['L']
        a = self.state['a']
        e = self.state['e']

        h += grad
        xp = cuda.get_array_module(param.data)
        cum_grad_norm = xp.linalg.norm(h)
        grad_norm = xp.linalg.norm(grad)

        lip = xp.maximum(lip, grad_norm)
        e = xp.maximum(e + 2 * grad_norm ** 2, lip * cum_grad_norm)
        a = xp.maximum(a, e / lip ** 2)

        self.state['L'] = lip
        self.state['a'] = a
        self.state['e'] = e

        param.data = -(xp.exp(cum_grad_norm / k / xp.sqrt(e)) - 1) / a / cum_grad_norm * h

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        k = self.hyperparam.k
        h = self.state['cum_grad']
        lip = self.state['L']
        a = self.state['a']
        e = self.state['e']

        h += grad
        xp = cuda.get_array_module(param.data)
        cum_grad_norm = xp.linalg.norm(h)
        grad_norm = xp.linalg.norm(grad)

        lip = xp.maximum(lip, grad_norm)
        e = xp.linalg.maximum(e + 2 * grad_norm ** 2, lip * cum_grad_norm)
        a = xp.maximum(a, e / lip ** 2)

        self.state['L'] = lip
        self.state['a'] = a
        self.state['e'] = e

        param.data = -(xp.exp(cum_grad_norm / k / xp.sqrt(e)) - 1) / a / cum_grad_norm * h

        cuda.elementwise(
            'T h, T cum_grad_norm, T k, T e, T a',
            'T param',
            '''
               param = -(exp(cum_grad_norm / k / e) - 1) / a / cum_grad_norm * h;''',
            'freerex')(h, cum_grad_norm, k, xp.sqrt(e), a,
                       param.data)


class FreeRex(optimizer.GradientMethod):
    """AdaGrad optimizer.

    See: http://jmlr.org/papers/v12/duchi11a.html

    Args:
        k (float): Upper bound of $L_t/L?{t-1}$ where L_t is max Euclidean norm of gradient until current iteration.

    """

    def __init__(self, k=_default_hyperparam.k, model=None):
        super(FreeRex, self).__init__(model)
        self.hyperparam.k = k

    k = optimizer.HyperparameterProxy('k')

    def create_update_rule(self):
        return FreeRexRule(self.hyperparam)
