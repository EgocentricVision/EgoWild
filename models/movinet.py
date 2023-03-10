import os
import models
from models.movinets import MoViNet
from models.movinets.config import _C
import urllib
from torch import nn
from utils.transforms import *


class Movinet(models.VideoModel):
    def __init__(self, num_class, modality, model_config, **kwargs):

        super(Movinet, self).__init__(num_class, model_config, **kwargs)

        self.num_class = num_class
        self.model_config = model_config
        causal = self.model_config.causal
        version = self.model_config.version
        pretrained = self.model_config.pretrained
        self.feat_dim = 480

        conv_v = {
            "a0": _C.MODEL.MoViNetA0,
            "a1": _C.MODEL.MoViNetA1,
            "a2": _C.MODEL.MoViNetA2,
            "a3": _C.MODEL.MoViNetA3
        }

        self.input_mean = [0.4345, 0.4051, 0.3775]
        self.input_std = [0.2768, 0.2713, 0.2737]
        self.range = [0, 1]

        self.version = conv_v[version]
        if not (version == "a0") and pretrained:
            raise NotImplementedError("The download of this pretrained has not been implemented yet")

        os.makedirs("weights_movinet", exist_ok=True)

        if causal and pretrained:
            conv_t = "2plus1d"
            if not os.path.isfile(os.path.join("weights_movinet", "temp_weights.pth")):
                urllib.request.urlretrieve(
                    self.version.stream_weights, os.path.join("weights_movinet", "temp_weights.pth"))

        elif pretrained:
            conv_t = "3d"
            if not os.path.isfile(os.path.join("weights_movinet", "temp_weights_nonstream.pth")):
                urllib.request.urlretrieve(
                    self.version.weights, os.path.join("weights_movinet", "temp_weights_nonstream.pth"))

        self.model = MoViNet(self.version, causal=causal, pretrained=False, conv_type=conv_t)

        if pretrained and causal:
            self.model.load_state_dict(torch.load(os.path.join("weights_movinet", "temp_weights.pth")),
                                       strict=True)
        elif pretrained:
            self.model.load_state_dict(torch.load(os.path.join("weights_movinet", "temp_weights_nonstream.pth")),
                                       strict=True)

        self.model.classifier[2] = torch.nn.Dropout(p=self.model_config.dropout, inplace=True)

        if conv_t == "2plus1d":
            classifier = self.model.classifier[3].conv_1.conv2d
            in_ch = classifier.in_channels
            self.model.classifier[3].conv_1.conv2d = nn.Conv2d(in_ch, self.num_class,
                                                               kernel_size=classifier.kernel_size,
                                                               stride=classifier.stride)
        else:
            classifier = self.model.classifier[3].conv_1.conv3d
            in_ch = classifier.in_channels
            self.model.classifier[3].conv_1.conv3d = nn.Conv3d(in_ch, self.num_class,
                                                               kernel_size=classifier.kernel_size,
                                                               stride=classifier.stride)

    def forward(self, x, **kwargs):
        source_type = kwargs.get("source_type", "source")
        update_activation = kwargs.get("update_activation", True)
        reset_buffer = kwargs.get("reset_buffer", False)
        if reset_buffer:
            self.model.clean_activation_buffers(source_type)
        return self.model(x, source_type, update_activation)

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
