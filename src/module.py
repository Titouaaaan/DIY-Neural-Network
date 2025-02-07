from typing import Any

class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError()

    def backward(self, y, yhat):
        raise NotImplementedError()


class Module(object):
    def __init__(self):
        self._parameters = {}
        self._gradient = {}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def zero_grad(self):
        raise NotImplementedError()

    def forward(self, X):
        raise NotImplementedError()

    def update_parameters(self, learning_rate=1e-3):
        raise NotImplementedError()

    def backward_update_gradient(self, input, delta):
        raise NotImplementedError()

    def backward_delta(self, input, delta):
        raise NotImplementedError()