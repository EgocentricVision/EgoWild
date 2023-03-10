import torch
import numpy as np
import os

from torch import nn

from utils.logger import logger
from utils.clipwindow import ClipWindow
import utils
"""
This validation is used when all the data are passed trimmed (no untrimmed component) but they are given frame by frame 
and there might be no supervision on the sample change (i.e. boundaries)
"""


def validate(args, model, val_loader, device, it, num_classes):
    """
    function to validate the model on the test set
    val_loader: dataloader containing the validation data
    model: Task containing the model to be tested
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    split = "test"
    model.reset_acc()
    model.train(False)
    input_seen = 0
    last_reset = 0
    clip_former = ClipWindow(args, split)

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):

            data, _ = utils.utils.reshape_input_data(data, args, split=split)
            for i_c in range(args.test.num_clips):

                last_reset += 1
                clip = clip_former.get(data, i_c).to(device)

                if i_c == 0 and args.boundaries_supervision:
                    model.clean_buffer()
                    last_reset = 0

                # SBL - clean buffer each k frames observed
                if args.SBL_k is not None and input_seen % args.SBL_k == 0:
                    model.clean_buffer()
                    last_reset = 0

                output, features = model(clip)
                model.update_buffer(output)

                # DBL - computing stats
                if input_seen == 0:
                    features_mse = 0
                else:
                    features_mse = nn.MSELoss()(features["features"], last_features["features"]).view(-1).detach().cpu()

                if args.DBL_threshold is not None and features_mse >= args.DBL_threshold:
                    model.clean_buffer()
                    last_reset = 0

                input_seen += 1
                last_features = features

            logits = model.get_output_from_buffer()
            model.compute_accuracy(logits, label)

            if (i_val + 1) % max(len(val_loader) // 5, 1) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
                                                                          model.accuracy.avg[1], model.accuracy.avg[5]))

        class_accuracies = {i_class: (x / y) * 100 for i_class, x, y
                            in zip(range(len(model.accuracy.total)), model.accuracy.correct, model.accuracy.total)
                            if y != 0}
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        for i_class, class_acc in class_accuracies.items():
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(model.accuracy.correct[i_class]),
                                                         int(model.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(list(class_accuracies.values())).mean()))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies.values())}

    with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.shift.split("-")[0]}-'
                                         f'{args.dataset.shift.split("-")[-1]}.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results
