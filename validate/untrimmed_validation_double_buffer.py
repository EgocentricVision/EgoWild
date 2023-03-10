import torch
import numpy as np
import os
from torch import nn
from utils.logger import logger
from utils.clipwindow import ClipWindow
import utils

"""
This validation is used when all the data are passed untrimmed, frame by frame 
and there might be no supervision on the sample change (i.e. boundaries) with 2 buffers
"""


def validate(args, model, val_loader, device, it, num_classes):
    """
    function to validate the model on the test set
    val_loader: dataloader containing the validation data
    model: Task containing the model to be tested
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    """
    split = "test"
    stride = args.dataset[args.modality].stride
    model.reset_acc()
    model.train(False)
    buffer_used = {0: "first", 1: "second"}
    uid_buffer = {0: -1, 1: -1}
    label_buffer = {0: -1, 1: -1}
    last_reset = {0: 0, 1: 0}
    clip_former = {0: ClipWindow(args, split), 1: ClipWindow(args, split)}
    last_features = {0: None, 1: None}
    last_uids = [-1]
    last_label = None

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):

            if i_val % stride != 0:
                continue

            current_uid = label['uid']
            label_uid_map = {u.item(): torch.tensor([l]) for u, l in zip(label['uid'], label['label'])}
            data, _ = utils.utils.reshape_input_data(data, args, split=split)

            for i_c in range(args.test.num_clips):

                # check if the actions in the frame before are finished (or the loader is over with an action which was
                # completed) and, in case, validate the output with the correspoding label
                if not args.boundaries_supervision:
                    for last_uid in last_uids:
                        if (last_uid not in current_uid
                            or (((i_val + stride) // stride) == (len(val_loader) // stride))) \
                                and last_uid != -1:
                            label = last_label[int(last_uid)]
                            logits = model.get_output_from_buffer(buffer_used[0]) * (last_reset[0] / (last_reset[0] + last_reset[1])) + \
                                     model.get_output_from_buffer(buffer_used[1]) * (last_reset[1] / (last_reset[0] + last_reset[1]))
                            model.compute_accuracy(logits, label)

                    last_uids = current_uid
                    last_label = label_uid_map

                for buffer_id in range(2):
                    last_reset[buffer_id] += 1
                    clip = clip_former[buffer_id].get(data, i_c).to(device)

                    if args.boundaries_supervision:
                        if uid_buffer[buffer_id] != -1 and \
                                (uid_buffer[buffer_id] not in current_uid or
                                 ((i_val + stride) // stride) == (len(val_loader) // stride)):
                            label = label_buffer[buffer_id]
                            logits = model.get_output_from_buffer(buffer_used[buffer_id])
                            model.compute_accuracy(logits, label)

                        if uid_buffer[buffer_id] not in current_uid:
                            model.clean_buffer(buffer_used[buffer_id])
                            last_reset[buffer_id] = 0
                            available_uids = list(filter(lambda x: x != uid_buffer[(buffer_id + 1) % 2], current_uid))
                            if len(available_uids) != 0:
                                uid_buffer[buffer_id] = available_uids[0]
                                label_buffer[buffer_id] = label_uid_map[uid_buffer[buffer_id].item()]
                            else:
                                uid_buffer[buffer_id] = -1
                                label_buffer[buffer_id] = -1

                    output, features = model(clip, source_type=buffer_used[buffer_id])
                    model.update_buffer(output, source_type=buffer_used[buffer_id])

                    # DBL - computing stats
                    if last_features[buffer_id] is not None:
                        features_mse = nn.MSELoss()(features["features"], last_features[buffer_id]).view(-1).detach().cpu()
                    else:
                        features_mse = 0

                    """
                    The following conditions check:
                    1) the DBL threshold is set
                    2) the threshold is overcome in the current forward
                    3) the last buffer reset was the other one so that the two buffers are reset alternatively
                    4) the reset of the other buffer has been performed at least delta frames ago 
                       (enough time to stabilize the content)
                    5) the OR is used to trigger the first reset and start the alternation
                    """
                    if (args.DBL_threshold is not None and features_mse >= args.DBL_threshold and
                        last_reset[buffer_id] - last_reset[(buffer_id + 1) % 2] > 0 and
                        last_reset[(buffer_id + 1) % 2] >= args.delta) or (i_val == args.delta and buffer_id == 0):
                        model.clean_buffer(buffer_used[buffer_id])
                        last_reset[buffer_id] = 0

                    last_features[buffer_id] = features["features"]

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
