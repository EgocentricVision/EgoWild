from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from models.utils.multi_batch_norm import MultiBatchNorm3d
from utils.logger import logger
from utils.transforms import *
import models


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, final_endpoint='Logits', name='inception_i3d',
                 in_channels=3, model_config=None):
        """Initializes I3D model instance.
        Args:
            num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
            final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of InceptionI3d.VALID_ENDPOINTS (default 'Logits').
            in_channels: number of channels of the input data
            name: A string (optional). The name of this module.
            model_config: config file with all additional configuration.
        Raises:
            ValueError: if `final_endpoint` is not recognized.
        """

        super(InceptionI3d, self).__init__()

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        self._num_classes = num_classes
        self.model_config = model_config
        self._final_endpoint = final_endpoint
        self.input_mean, self.input_std, self.range = None, None, None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = InceptionI3d.Unit3D(in_channels=in_channels, output_channels=64,
                                                         kernel_shape=[7, 7, 7], stride=(2, 2, 2), padding=(3, 3, 3),
                                                         name=name + end_point,
                                                         batch_norm_layers=model_config.batch_norm_layers,
                                                         affine=model_config.batch_norm_affine,
                                                         bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = InceptionI3d.MaxPool3dSamePadding(kernel_size=[1, 3, 3],
                                                                       stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = InceptionI3d.Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1],
                                                         padding=0, name=name + end_point,
                                                         batch_norm_layers=model_config.batch_norm_layers,
                                                         affine=model_config.batch_norm_affine,
                                                         bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = InceptionI3d.Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3],
                                                         padding=1, name=name + end_point,
                                                         batch_norm_layers=model_config.batch_norm_layers,
                                                         affine=model_config.batch_norm_affine,
                                                         bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = InceptionI3d.MaxPool3dSamePadding(kernel_size=[1, 3, 3],
                                                                       stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionI3d.InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionI3d.InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = InceptionI3d.MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                                                       stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionI3d.InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64],
                                                                  name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionI3d.InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64],
                                                                  name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionI3d.InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64],
                                                                  name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionI3d.InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64],
                                                                  name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionI3d.InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                                  name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = InceptionI3d.MaxPool3dSamePadding(kernel_size=[2, 2, 2],
                                                                       stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionI3d.InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                                  name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionI3d.InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                                  name + end_point,
                                                                  batch_norm_layers=model_config.batch_norm_layers,
                                                                  affine=model_config.batch_norm_affine,
                                                                  bn_momentum=model_config.bn_momentum)
        if self._final_endpoint == end_point:
            return

        # Fully connected layer implemented using Unit3D (Conv3d 1x1x1)
        end_point = 'Logits'
        self.dropout = nn.Dropout(self.model_config.dropout)
        self.avg_pool = nn.AdaptiveAvgPool3d([1, 1, 1])

        self.logits = InceptionI3d.Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                                          kernel_shape=[1, 1, 1],
                                          padding=0,
                                          activation_fn=None,
                                          use_batch_norm=False,
                                          use_bias=True,
                                          name=end_point,
                                          affine=model_config.batch_norm_affine,
                                          bn_momentum=model_config.bn_momentum)
        InceptionI3d.truncated_normal_(self.logits.conv3d.weight, std=1 / 32)

        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x, **kwargs):
        bn_idx = kwargs.get("bn_idx", None)
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x, bn_idx)  # use _modules to work with dataparallel
        spatial_features = x
        x = self.avg_pool(x)
        feat = x.squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.logits(self.dropout(x))
        logits = x.squeeze(3).squeeze(3).squeeze(2)
        return logits, {"features": feat, "spatial_features": spatial_features}

    @staticmethod
    def truncated_normal_(tensor, mean=0., std=1.):
        """
        This function modifies the tensor in input by creating a tensor from a normal distribution
            - at first creates a standard normal tensor
            - then it cuts the values keeping just the ones in (-2,2)
            - finally multiplies the std and adds the mean
        The new standard tensor has the same shape of the input one
        """
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

    class MaxPool3dSamePadding(nn.MaxPool3d):

        def compute_pad(self, dim, s):
            if s % self.stride[dim] == 0:
                return max(self.kernel_size[dim] - self.stride[dim], 0)
            else:
                return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

        def forward(self, x, bn_idx=None):
            # compute 'same' padding
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)
            # print pad_t, pad_h, pad_w

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
            x = F.pad(x, pad)
            return super(InceptionI3d.MaxPool3dSamePadding, self).forward(x)

    class Unit3D(nn.Module):

        def __init__(self, in_channels,
                     output_channels,
                     kernel_shape=(1, 1, 1),
                     stride=(1, 1, 1),
                     padding=0,
                     activation_fn=F.relu,
                     use_batch_norm=True,
                     use_bias=False,
                     name='unit_3d',
                     batch_norm_layers=1,
                     affine=True,
                     bn_momentum=0.1):

            """Initializes Unit3D module."""
            super(InceptionI3d.Unit3D, self).__init__()

            self._output_channels = output_channels
            self._kernel_shape = kernel_shape
            self._stride = stride
            self._use_batch_norm = use_batch_norm
            self._batch_norm_layers = batch_norm_layers
            self._activation_fn = activation_fn
            self._use_bias = use_bias
            self.name = name
            self.padding = padding

            self.conv3d = nn.Conv3d(in_channels=in_channels,
                                    out_channels=self._output_channels,
                                    kernel_size=self._kernel_shape,
                                    stride=self._stride,
                                    padding=0,  # we will pad dynamically in the forward function
                                    bias=self._use_bias)

            if self._use_batch_norm and self._batch_norm_layers == 1:
                self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=bn_momentum, affine=affine)
            elif self._use_batch_norm and self._batch_norm_layers > 1:
                self.bn = MultiBatchNorm3d(self._output_channels, eps=0.001, momentum=bn_momentum, affine=affine,
                                           n_layers=self._batch_norm_layers)

        def compute_pad(self, dim, s):
            if s % self._stride[dim] == 0:
                return max(self._kernel_shape[dim] - self._stride[dim], 0)
            else:
                return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

        def forward(self, x, bn_idx=None):
            # def forward(self, x):
            # compute 'same' padding
            (batch, channel, t, h, w) = x.size()

            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
            x = F.pad(x, pad)

            x = self.conv3d(x)
            if self._use_batch_norm and self._batch_norm_layers == 1:
                x = self.bn(x)
            elif self._use_batch_norm and self._batch_norm_layers > 1:
                x = self.bn(x, bn_idx)
            if self._activation_fn is not None:
                x = self._activation_fn(x)
            return x

    class InceptionModule(nn.Module):
        def __init__(self, in_channels, out_channels, name, batch_norm_layers=1, affine=True, bn_momentum=0.1):
            super(InceptionI3d.InceptionModule, self).__init__()

            self.b0 = InceptionI3d.Unit3D(in_channels=in_channels, output_channels=out_channels[0],
                                          kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_0/Conv3d_0a_1x1',
                                          batch_norm_layers=batch_norm_layers, affine=affine, bn_momentum=bn_momentum)
            self.b1a = InceptionI3d.Unit3D(in_channels=in_channels, output_channels=out_channels[1],
                                           kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_1/Conv3d_0a_1x1',
                                           batch_norm_layers=batch_norm_layers, affine=affine, bn_momentum=bn_momentum)
            self.b1b = InceptionI3d.Unit3D(in_channels=out_channels[1], output_channels=out_channels[2],
                                           kernel_shape=[3, 3, 3], name=name + '/Branch_1/Conv3d_0b_3x3',
                                           batch_norm_layers=batch_norm_layers, affine=affine, bn_momentum=bn_momentum)
            self.b2a = InceptionI3d.Unit3D(in_channels=in_channels, output_channels=out_channels[3],
                                           kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_2/Conv3d_0a_1x1',
                                           batch_norm_layers=batch_norm_layers, affine=affine, bn_momentum=bn_momentum)
            self.b2b = InceptionI3d.Unit3D(in_channels=out_channels[3], output_channels=out_channels[4],
                                           kernel_shape=[3, 3, 3], name=name + '/Branch_2/Conv3d_0b_3x3',
                                           batch_norm_layers=batch_norm_layers, affine=affine, bn_momentum=bn_momentum)
            self.b3a = InceptionI3d.MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
            self.b3b = InceptionI3d.Unit3D(in_channels=in_channels, output_channels=out_channels[5],
                                           kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_3/Conv3d_0b_1x1',
                                           batch_norm_layers=batch_norm_layers, affine=affine, bn_momentum=bn_momentum)
            self.name = name

        def forward(self, x, bn_idx=None):
            b0 = self.b0(x, bn_idx)
            b1 = self.b1b(self.b1a(x, bn_idx), bn_idx)
            b2 = self.b2b(self.b2a(x, bn_idx), bn_idx)
            b3 = self.b3b(self.b3a(x, bn_idx), bn_idx)
            return torch.cat([b0, b1, b2, b3], dim=1)


class I3D(models.VideoModel):
    def __init__(self, num_class, modality, model_config, **kwargs):
        super(I3D, self).__init__(num_class, model_config, **kwargs)
        self.num_class = num_class
        self.model_config = model_config
        self.feat_dim = 1024

        if modality == "RGB":
            channel = 3
            self.base_model = InceptionI3d(num_classes=self.num_class,
                                           in_channels=channel,
                                           model_config=self.model_config)
            weights = self.load(self.model_config.weight_i3d)
            self.base_model.load_state_dict(weights, strict=False)

        elif modality == "Flow":
            channel = 2  # 'x' and 'y'
            self.base_model = InceptionI3d(num_classes=self.num_class,
                                           in_channels=channel,
                                           model_config=self.model_config)
            weights = self.load(self.model_config.weight_i3d)
            self.base_model.load_state_dict(weights, strict=False)

        else:
            raise NotImplementedError

        # set this to None in order to avoid normalization since for I3D we do not want simple normalization
        self.input_mean, self.input_std, self.range = None, None, None

    def forward(self, x, **kwargs):
        return self.base_model(x, **kwargs)

    def get_augmentation(self, modality):
        if modality in ['RGB', 'Flow']:
            train_augmentation = torchvision.transforms.Compose(
                # Data augmentation, at first reduce then interpolate
                [GroupMultiScaleCrop(self.model_config.resolution, [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=(modality == 'Flow')),
                 Stack(roll=False),
                 ToTorchFormatTensor(div=not self.model_config.normalize),
                 GroupNormalize(self.model_config.normalize, self.input_mean, self.input_std, self.range)]
            )

            val_augmentation = torchvision.transforms.Compose([
                GroupCenterCrop(self.model_config.resolution),
                Stack(roll=False),
                ToTorchFormatTensor(div=not self.model_config.normalize),
                GroupNormalize(self.model_config.normalize, self.input_mean, self.input_std, self.range)
            ])
        else:
            raise NotImplementedError

        return train_augmentation, val_augmentation

    def load(self, path):
        logger.info("Loading Kinetics weights I3D")
        verbose = True
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k  # [7:]  # remove `module.`
            check_bn = name.split(".")

            # extend bn statistics to the number of BN layers
            if "bn" in check_bn and self.model_config.batch_norm_layers and (
                    "running_mean" in check_bn or "running_var" in check_bn):

                if verbose:
                    logger.info(" * Extend statistics of BN to all BN layers")
                    verbose = False

                param = torch.cat([v.unsqueeze(0) for _ in range(self.model_config.batch_norm_layers)], dim=0)
                if self.model_config.batch_norm_layers == 1:
                    param = param.squeeze(0)
                new_state_dict[name] = param
            elif "logits" in check_bn:
                logger.info(" * Skipping Logits weight for \'{}\'".format(name))
                pass
            else:
                # print(" * Param", name)
                new_state_dict[name] = v

            # load params
        return new_state_dict
