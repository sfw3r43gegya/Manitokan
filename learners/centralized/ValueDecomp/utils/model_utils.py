import math

import torch.nn as nn


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm()**2
    return math.sqrt(sum_grad)


def hard_target_update(src: nn.Module, tgt: nn.Module) -> None:
    """Hard update model parameters.

    Params
    ======
        src: PyTorch model (weights will be copied from)
        tgt: PyTorch model (weights will be copied to)
    """

    tgt.load_state_dict(src.state_dict())


def soft_target_update(src: nn.Module, tgt: nn.Module, tau=0.05) -> None:
    """Soft update model parameters.

    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        src: PyTorch model (weights will be copied from)
        tgt: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
        tgt_param.data.copy_(tau * src_param.data +
                             (1.0 - tau) * tgt_param.data)


def check_model_method(model, method, algo):
    """check method existence for input model to algo.

    Args:
        model(nn.Model): model for checking
        method(str): method name
        algo(str): algorithm name

    Raises:
        AssertionError: if method is not implemented in model
    """
    if method == 'forward':
        # check if forward is overridden by the subclass
        assert callable(
            getattr(model, 'forward',
                    None)), 'forward should be a function in model class'
        assert model.forward.__func__ is not super(
            model.__class__, model
        ).forward.__func__, "{}'s model needs to implement forward method. \n".format(
            algo)
    else:
        # check if the specified method is implemented
        assert hasattr(model, method) and callable(getattr(
            model, method,
            None)), "{}'s model needs to implement {} method. \n".format(
                algo, method)