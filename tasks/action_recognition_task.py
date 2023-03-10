from abc import ABC
import torch
from utils import utils
import tasks
from utils.logger import logger


class Buffer:
    def __init__(self):
        self.logits = {}
        self.count = {}

    def clean_buffer(self, source_type="source"):
        if self.logits.get(source_type, None) is None:
            # nothing to be cleaned
            return
        self.logits[source_type] = torch.zeros(self.logits[source_type].shape)
        self.count[source_type] = 0

    def update_buffer(self, logits, source_type="source"):
        if self.logits.get(source_type, None) is None:
            self.logits[source_type] = logits.detach().to('cpu')
            self.count[source_type] = 1
        else:
            self.logits[source_type] += logits.detach().to('cpu')
            self.count[source_type] += 1

    def get_output_from_buffer(self, source_type="source"):
        if self.count[source_type] == 0:
            return self.logits[source_type]
        return self.logits[source_type] / self.count[source_type]


class ActionRecognition(tasks.Task, ABC):
    loss = None
    optimizer = None

    def __init__(self, name, task_models, batch_size, total_batch, models_dir, num_classes,
                 num_clips, model_args, args, **kwargs) -> None:
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.buffer_models = Buffer()
        # Accuracy measures
        self.model_args = model_args
        self.accuracy = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.loss = utils.AverageMeter()
        self.num_clips = num_clips
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        optim_params = filter(lambda parameter: parameter.requires_grad, self.task_model.parameters())
        self.optimizer = torch.optim.SGD(optim_params, model_args.lr,
                                         weight_decay=model_args.weight_decay,
                                         momentum=model_args.sgd_momentum)

    def forward(self, data, **kwargs):
        logits, features = self.task_model(x=data, **kwargs)
        return logits, features

    def compute_loss(self, logits, label, loss_weight=1.0):
        loss = self.criterion(logits, label) / self.num_clips
        self.loss.update(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)

    def compute_accuracy(self, logits, label):
        self.accuracy.update(logits, label)

    def reduce_learning_rate(self):
        prev_lr = self.optimizer.param_groups[-1]["lr"]
        self.optimizer.param_groups[-1]["lr"] = self.optimizer.param_groups[-1]["lr"] / 10
        logger.info('Reducing learning rate: {} --> {}'
                    .format(prev_lr, self.optimizer.param_groups[-1]["lr"]))

    def reset_loss(self):
        self.loss.reset()

    def reset_acc(self):
        self.accuracy.reset()

    def step(self, reset_acc=True):
        super().step()
        self.reset_loss()
        if reset_acc:
            self.reset_acc()

    def backward(self, retain_graph):
        self.scale_loss(self.loss.val).backward(retain_graph=retain_graph)

    def clean_buffer(self, source_type="source"):
        if self.model_args.model == "Movinet" and self.model_args.causal:
            if type(self.task_model) == torch.nn.DataParallel:
                self.task_model.module.model.clean_activation_buffers(source_type)
            else:
                self.task_model.model.clean_activation_buffers(source_type)
        else:
            self.buffer_models.clean_buffer(source_type)

    def update_buffer(self, logits, source_type="source"):
        if self.model_args.model == "Movinet" and self.model_args.causal:
            # for movinet the buffer is internal so there is no need to accumulate logits, the trick to avoid this
            # is to first clean the buffer and then update it so that it contains just the last logits
            self.buffer_models.clean_buffer(source_type)
            self.buffer_models.update_buffer(logits, source_type)
        else:
            self.buffer_models.update_buffer(logits, source_type)

    def get_output_from_buffer(self, source_type="source"):
        return self.buffer_models.get_output_from_buffer(source_type)
