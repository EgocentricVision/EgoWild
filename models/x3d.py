from torch import nn
from utils.transforms import *
import models


class X3D(models.VideoModel):
    def __init__(self, num_class, modality, model_config, **kwargs):
        super(X3D, self).__init__(num_class, model_config, **kwargs)

        self.num_class = num_class
        self.model_config = model_config
        version = self.model_config.version
        pretrained = self.model_config.pretrained
        self.input_mean = [0.45, 0.45, 0.45]
        self.input_std = [0.225, 0.225, 0.225]
        self.range = [0, 1]
        self.feat_dim = 2048

        self.model = torch.hub.load('facebookresearch/pytorchvideo', version, pretrained=pretrained)
        self.model.blocks[5].proj = nn.Linear(in_features=2048, out_features=self.num_class, bias=True)

    def forward(self, x, **kwargs):
        features = None
        spatial_features = None
        for idx in range(len(self.model.blocks)):
            if idx != 5:
                x = self.model.blocks[idx](x)
            else:
                if self.model.blocks[idx].pool is not None:
                    x = self.model.blocks[idx].pool(x)
                # Performs dropout.
                if self.model.blocks[idx].dropout is not None:
                    x = self.model.blocks[idx].dropout(x)
                features = x
                # Performs projection.
                if self.model.blocks[idx].proj is not None:
                    x = x.permute((0, 2, 3, 4, 1))
                    x = self.model.blocks[idx].proj(x)
                    x = x.permute((0, 4, 1, 2, 3))
                # Performs activation.
                if self.model.blocks[idx].activation is not None:
                    x = self.model.blocks[idx].activation(x)

                if self.model.blocks[idx].output_pool is not None:
                    # Performs global averaging.
                    x = self.model.blocks[idx].output_pool(x)
                    x = x.view(x.shape[0], -1)

        return x, {"features": features, "spatial_features": spatial_features}

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
