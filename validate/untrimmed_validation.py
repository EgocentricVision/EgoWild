import pdb

import torch
import numpy as np
import os
from torch import nn
from utils.logger import logger
from utils.clipwindow import ClipWindow
import utils
from utils.ABD import ABDChecker
"""
This validation is used when all the data are passed untrimmed, frame by frame 
and there might be no supervision on the sample change (i.e. boundaries)
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
    input_seen = 0
    last_reset = 0
    last_uids = [-1]
    last_label = None
    clip_former = ClipWindow(args, split)

    # create ABD checker in case using this as boundaries checker
    if args.ABD:
        abd_checker = ABDChecker(args.ABD, args.dataset.shift.split("-")[-1],
                                 feat_size=model.task_model.module.feat_dim, feat_dir=args.feat_dir)

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):

            if i_val % stride != 0:
                continue

            current_uid = label['uid']
            label_uid_map = {u.item(): torch.tensor([l]) for u, l in zip(label['uid'], label['label'])}
            data, _ = utils.utils.reshape_input_data(data, args, split=split)

            for i_c in range(args.test.num_clips):

                last_reset += 1

                # check if the actions in the frame before are finished (or the loader is over with an action which was
                # completed) and, in case, validate the output with the correspoding label
                for last_uid in last_uids:
                    if (last_uid not in current_uid
                        or (((i_val + stride) // stride) == (len(val_loader) // stride))) \
                            and last_uid != -1:
                        label = last_label[int(last_uid)]
                        logits = model.get_output_from_buffer()
                        model.compute_accuracy(logits, label)

                        # clean the buffer if there is the boundaries supervision
                        if args.boundaries_supervision:
                            if last_uids != current_uid:
                                model.clean_buffer()
                                last_reset = 0

                last_uids = current_uid
                last_label = label_uid_map
                clip = clip_former.get(data, i_c).to(device)

                # SBL - clean buffer each k frames observed
                if args.SBL_k is not None and input_seen % args.SBL_k == 0:
                    model.clean_buffer()
                    last_reset = 0

                output, features = model(clip)
                model.update_buffer(output)
                # compute similarity if using ABD technique
                if args.ABD:
                    abd_checker.compute_similarity(i_val, i3d_feat=features["features"].view(1, -1).cpu())

                # ABD - reset in case you are using this technique, and it mentions to do so
                if args.ABD and abd_checker.check_reset():
                    model.clean_buffer()
                    last_reset = 0

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
