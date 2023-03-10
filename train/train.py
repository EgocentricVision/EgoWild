from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.sampler import InfiniteBatchSampler
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import validate as validation_functions
import tasks

# global variables among training functions
training_iterations = 0
modality = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

def main():
    global training_iterations, modality
    modality = args.modality
    init_operations()

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Instantiating models per modality")
    logger.info('{} Net instantiated'.format(args.models.model))
    models = getattr(model_list, args.models.model)(num_classes, modality, args.models, **args.models.kwargs)
    train_augmentations, test_augmentations = models.get_augmentation("RGB")

    action_classifier = tasks.ActionRecognition("action-classifier", models, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                args.train.num_clips, args.models, args=args)
    action_classifier.load_on_gpu(device)

    if args.action == "train":
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        # define number of iterations I'll do with the actual batch: we do not reason with epochs but with iterations
        # i.e. number of batches passed
        # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation is used
        training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
        dataset = EpicKitchensDataset(args.dataset.shift.split("-")[0], modality,
                                      'train', args.dataset,
                                      args.train.num_frames_per_clip,
                                      args.train.num_clips, args.train.dense_sampling,
                                      train_augmentations, consecutive_clips=args.train.consecutive_clips)
        inf_sampler = InfiniteBatchSampler(dataset, args.batch_size, False)
        train_loader = torch.utils.data.DataLoader(dataset, num_workers=args.dataset.workers,
                                                   pin_memory=True, batch_sampler=inf_sampler, persistent_workers=True,
                                                   prefetch_factor=args.dataset.prefetch)

        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modality,
                                                                     'val', args.dataset, args.test.num_frames_per_clip,
                                                                     args.test.num_clips, args.test.dense_sampling,
                                                                     test_augmentations),
                                                 batch_size=1,
                                                 shuffle=False, num_workers=args.dataset.workers,
                                                 pin_memory=True, drop_last=False)
        train(action_classifier, train_loader, val_loader, device, num_classes)

    elif args.action == "validate":
        validate = getattr(validation_functions, args.validation_function).validate
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modality,
                                                                     'val', args.dataset, args.test.num_frames_per_clip,
                                                                     args.test.num_clips, args.test.dense_sampling,
                                                                     test_augmentations,
                                                                     untrimmed=args["test"].untrimmed,
                                                                     full_length=args.test.full_length),
                                                 batch_size=1,
                                                 shuffle=False, num_workers=args.dataset.workers,
                                                 pin_memory=True, drop_last=False)

        validate(args, action_classifier, val_loader, device, action_classifier.current_iter, num_classes)

    else:
        validate = getattr(validation_functions, args.validation_function).validate
        test_results = {'top1': [], 'top5': [], 'class_accuracies': []}
        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modality,
                                                                     'val', args.dataset, args.test.num_frames_per_clip,
                                                                     args.test.num_clips, args.test.dense_sampling,
                                                                     test_augmentations,
                                                                     untrimmed=args["test"].untrimmed,
                                                                     full_length=args.test.full_length),
                                                 batch_size=1,
                                                 shuffle=False, num_workers=args.dataset.workers,
                                                 pin_memory=True, drop_last=False)
        for i_model in range(1, 10):
            action_classifier.load_model(args.resume_from, i_model)
            t_r = validate(args, action_classifier, val_loader, device, action_classifier.current_iter, num_classes)
            for k in test_results.keys():
                test_results[k].append(t_r[k])
        test_results['class_accuracies'] = np.array(test_results['class_accuracies']).mean(0)
        logger.info("FINAL RESULTS OVER 9 MODELS: \nTop@1 %.2f%%\nTop@5: %.2f%%\n"
                    % (mean(test_results['top1']), mean(test_results['top5'])))
        for i_class, class_acc in enumerate(test_results['class_accuracies']):
            logger.info('Class %d = %.2f%%' % (i_class, class_acc))
        with open(os.path.join(args.log_dir, "summary_" + args.action + "_" + args.dataset.shift + ".txt"), 'w') as f:
            f.write("FINAL RESULTS OVER 9 MODELS: \nTop@1 %.2f%%\nTop@5: %.2f%%\n"
                    % (mean(test_results['top1']), mean(test_results['top5'])))
            for i_class, class_acc in enumerate(test_results['class_accuracies']):
                f.write('Class %d = %.2f%%\n' % (i_class, class_acc))


def train(action_classifier, train_loader, val_loader, device, num_classes):
    global training_iterations, modality
    validate = validation_functions.standard_validation.validate
    data_loader_source = iter(train_loader)
    action_classifier.train(True)
    action_classifier.zero_grad()
    iteration = action_classifier.current_iter * (args.total_batch // args.batch_size)

    # the batch size should be total_batch but batch accumulation is done with batch size = batch_size.
    # real_iter is the number of iterations if the batch size was really total_batch
    for i in range(iteration, training_iterations):
        # iteration w.r.t. the paper (w.r.t the bs to simulate).... i is the iteration with the actual bs( < tot_bs)
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        if real_iter == args.train.lr_steps:
            action_classifier.reduce_learning_rate()
        gradient_accumulation_step = real_iter.is_integer()

        """
        Retrieve the data from the loaders
        """
        start_t = datetime.now()
        source_data, source_label = next(data_loader_source)
        end_t = datetime.now()

        logger.info(f"Iteration {i}/{training_iterations} batch retrieved! Elapsed time = "
                    f"{(end_t - start_t).total_seconds() // 60} m {(end_t - start_t).total_seconds() % 60} s")

        ''' Action recognition'''
        source_data, batch = utils.utils.reshape_input_data(source_data, args, split="train")
        source_label = source_label.to(device)

        for clip in range(args.train.num_clips):
            data = source_data[clip].to(device)

            with torch.autocast(device_type=device.type, enabled=args.amp):
                logits, _ = action_classifier.forward(data, reset_buffer=(clip == 0))
                action_classifier.compute_loss(logits, source_label, loss_weight=1)
            action_classifier.backward(retain_graph=False)
            action_classifier.compute_accuracy(logits, source_label)

        # update weights and zero gradients
        if gradient_accumulation_step:
            logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                        (real_iter, args.train.num_iter, action_classifier.loss.val, action_classifier.loss.avg,
                         action_classifier.accuracy.val[1], action_classifier.accuracy.avg[1]))

            action_classifier.check_grad()
            action_classifier.step()
            action_classifier.zero_grad()

        # every eval_freq "real iteration" (iterations on total_batch) the validation is done
        # we save every 9 models (dima policy: to validate, it takes the last 9 models, it tests them all,
        # and then it computes the average. This is done to avoid peaks in the performances)
        if gradient_accumulation_step and real_iter % args.train.eval_freq == 0 \
                and real_iter >= args.train.start_test_iter:
            val_metrics = validate(args, action_classifier, val_loader, device, int(real_iter), num_classes)

            if val_metrics['top1'] <= action_classifier.best_iter_score:
                logger.info("New best accuracy {:.2f}%"
                            .format(action_classifier.best_iter_score))
            else:
                logger.info("New best accuracy {:.2f}%".format(val_metrics['top1']))
                action_classifier.best_iter = real_iter
                action_classifier.best_iter_score = val_metrics['top1']

            action_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            action_classifier.train(True)


if __name__ == '__main__':
    main()
