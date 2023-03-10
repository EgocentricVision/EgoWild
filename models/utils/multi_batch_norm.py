from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class _MultiBatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    """Batch Normalization layers with multiple (mean, var) pairs

    Extend Batch Normalization by keeping multiple (mean, variance)
    pairs. During the forward pass, the `bn_idx` argument selects
    which pair is used to normalize the current batch of data.
    This approach can be used to implement AdaBN or cluster-based
    normalization.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
        *,
        n_layers: int = 1
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        # set track_running_stats here to prevent the parent from creating the buffers
        super(_MultiBatchNorm, self).__init__(num_features, eps, momentum, affine, False, **factory_kwargs)

        self.track_running_stats = track_running_stats
        self.bn_idx = 0
        if track_running_stats:
            # add one more dimension corresponding to the number of layers
            self.register_buffer("running_mean", torch.zeros(n_layers, num_features, **factory_kwargs))
            self.register_buffer("running_var", torch.ones(n_layers, num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]

            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != "dtype"}),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, input: Tensor, bn_idx: int = None) -> Tensor:
        self._check_input_dim(input)

        if bn_idx is None:
            bn_idx = self.bn_idx

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean[bn_idx] if not self.training or self.track_running_stats else None,
            self.running_var[bn_idx] if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    @classmethod
    def convert_multibn(cls, module):
        module_output = module
        if isinstance(module, torch.nn.ModuleDict):
            if isinstance(module["0"], torch.nn.BatchNorm1d):
                _cls = MultiBatchNorm1d
            elif isinstance(module["0"], torch.nn.BatchNorm2d):
                _cls = MultiBatchNorm2d
            elif isinstance(module["0"], torch.nn.BatchNorm3d):
                _cls = MultiBatchNorm3d

            module_output = _cls(
                module["0"].num_features,
                module["0"].eps,
                module["0"].momentum,
                module["0"].affine,
                module["0"].track_running_stats,
                n_layers=len(module)
            )

            if module["0"].affine:
                with torch.no_grad():
                    module_output.weight = module["0"].weight
                    module_output.bias = module["0"].bias
            module_output.running_mean.copy_(torch.stack([module[idx].running_mean for idx in module]))
            module_output.running_var.copy_(torch.stack([module[idx].running_var for idx in module]))
            module_output.num_batches_tracked.copy_(sum(module[idx].num_batches_tracked for idx in module))

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_multibn(child))

        del module
        return module_output

    def set_bn_idx(self, bn_idx: int = 0):
        """
        Default batch norm used in case it is not specified in the forward pass
        """
        self.bn_idx = bn_idx


class MultiBatchNorm1d(_MultiBatchNorm, torch.nn.BatchNorm1d):
    pass


class MultiBatchNorm2d(_MultiBatchNorm, torch.nn.BatchNorm2d):
    pass


class MultiBatchNorm3d(_MultiBatchNorm, torch.nn.BatchNorm3d):
    pass

