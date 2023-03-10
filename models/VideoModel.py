import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class VideoModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_class, model_config, **kwargs):
        super().__init__()
        self.num_class = num_class
        self.model_config = model_config
        self.kwargs = kwargs

    @abstractmethod
    def forward(self, x):
        """
        Subclasses must override this method, but adhere to the same return type
        Returns:
            torch.Tensor: logits of the forward pass
            torch.Tensor: features of the forward pass
        """
        pass

    @abstractmethod
    def get_augmentation(self, modality):
        """
        Subclasses must override this method, but adhere to the same return type
        Returns:
            torchvision.transforms.Compose: transformations related to the training data for the input modality
            torchvision.transforms.Compose; transformations related to the validation data for the input modality
        """
        pass
