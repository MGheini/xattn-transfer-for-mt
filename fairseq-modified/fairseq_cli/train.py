#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import random
import sys

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(args):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"

    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)

    # translation parent finetuning
    if getattr(args, 'load_model_but_src_embeddings_and_freeze_tgt_embeddings_from') is not None \
            or getattr(args, 'load_model_but_src_embeddings_and_xattn_and_freeze_tgt_embeddings_from') is not None:
        if getattr(args, 'load_model_but_src_embeddings_and_freeze_tgt_embeddings_from'):
            logger.info(f'loading the model but the src embeddings from '
                        f'{args.load_model_but_src_embeddings_and_freeze_tgt_embeddings_from}')
            pretrained_state_dict = torch.load(args.load_model_but_src_embeddings_and_freeze_tgt_embeddings_from)['model']
            for name, param in model.named_parameters():
                with torch.no_grad():
                    if name not in ['encoder.embed_tokens.weight',
                                    'encoder.embed_positions.weight',
                                    'encoder.layernorm_embedding.weight',
                                    'encoder.layernorm_embedding.bias',
                                    ]:
                        param.copy_(pretrained_state_dict[name])
        if getattr(args, 'load_model_but_src_embeddings_and_xattn_and_freeze_tgt_embeddings_from'):
            logger.info(f'loading the model but the src embeddings and xattn from '
                        f'{args.load_model_but_src_embeddings_and_xattn_and_freeze_tgt_embeddings_from}')
            pretrained_state_dict = torch.load(args.load_model_but_src_embeddings_and_xattn_and_freeze_tgt_embeddings_from)['model']
            for name, param in model.named_parameters():
                with torch.no_grad():
                    if name not in ['encoder.embed_tokens.weight',
                                    'encoder.embed_positions.weight',
                                    'encoder.layernorm_embedding.weight',
                                    'encoder.layernorm_embedding.bias',
                                    ] and 'encoder_attn' not in name:
                        param.copy_(pretrained_state_dict[name])
        for name, param in model.named_parameters():
            if name in ['decoder.embed_tokens.weight',
                        'decoder.embed_positions.weight',
                        'decoder.layernorm_embedding.weight',
                        'decoder.layernorm_embedding.bias',
                        'decoder.output_projection.weight',
                        ]:
                param.requires_grad = False
                logger.info(f'froze parameter {name}')
        if args.only_finetune_cross_attn:
            logger.info(f'only cross-attention layers will be trained in addition to src embeddings, '
                        f'freezing all other parameters')
            for name, param in model.named_parameters():
                if 'encoder_attn' not in name and \
                    name not in ['encoder.embed_tokens.weight',
                                 'encoder.embed_positions.weight',
                                 'encoder.layernorm_embedding.weight',
                                 'encoder.layernorm_embedding.bias',
                                 ]:
                    param.requires_grad = False
                else:
                    logger.info(f'parameter {name} will be trained')
        if args.freeze_pretrained_transformer_body:
            logger.info(f'only src embeddings will be trained, '
                        f'freezing all other parameters')
            for name, param in model.named_parameters():
                if name not in ['encoder.embed_tokens.weight',
                                'encoder.embed_positions.weight',
                                'encoder.layernorm_embedding.weight',
                                'encoder.layernorm_embedding.bias',
                                ]:
                    param.requires_grad = False
                else:
                    logger.info(f'parameter {name} will be trained')

    if getattr(args, 'load_model_but_tgt_embeddings_and_freeze_src_embeddings_from') is not None \
            or getattr(args, 'load_model_but_tgt_embeddings_and_xattn_and_freeze_src_embeddings_from') is not None:
        if getattr(args, 'load_model_but_tgt_embeddings_and_freeze_src_embeddings_from'):
            logger.info(f'loading the model but the tgt embeddings from '
                        f'{args.load_model_but_tgt_embeddings_and_freeze_src_embeddings_from}')
            pretrained_state_dict = torch.load(args.load_model_but_tgt_embeddings_and_freeze_src_embeddings_from)['model']
            for name, param in model.named_parameters():
                with torch.no_grad():
                    if name not in ['decoder.embed_tokens.weight',
                                    'decoder.embed_positions.weight',
                                    'decoder.layernorm_embedding.weight',
                                    'decoder.layernorm_embedding.bias',
                                    'decoder.output_projection.weight',
                                    ]:
                        param.copy_(pretrained_state_dict[name])
        if getattr(args, 'load_model_but_tgt_embeddings_and_xattn_and_freeze_src_embeddings_from'):
            logger.info(f'loading the model but the tgt embeddings and xattn from '
                        f'{args.load_model_but_tgt_embeddings_and_xattn_and_freeze_src_embeddings_from}')
            pretrained_state_dict = torch.load(args.load_model_but_tgt_embeddings_and_xattn_and_freeze_src_embeddings_from)['model']
            for name, param in model.named_parameters():
                with torch.no_grad():
                    if name not in ['decoder.embed_tokens.weight',
                                    'decoder.embed_positions.weight',
                                    'decoder.layernorm_embedding.weight',
                                    'decoder.layernorm_embedding.bias',
                                    'decoder.output_projection.weight',
                                    ] and 'encoder_attn' not in name:
                        param.copy_(pretrained_state_dict[name])
        for name, param in model.named_parameters():
            if name in ['encoder.embed_tokens.weight',
                        'encoder.embed_positions.weight',
                        'encoder.layernorm_embedding.weight',
                        'encoder.layernorm_embedding.bias',
                        ]:
                param.requires_grad = False
                logger.info(f'froze parameter {name}')
        if args.only_finetune_cross_attn:
            logger.info(f'only cross-attention layers will be trained in addition to tgt embeddings, '
                        f'freezing all other parameters')
            for name, param in model.named_parameters():
                if 'encoder_attn' not in name and \
                    name not in ['decoder.embed_tokens.weight',
                                 'decoder.embed_positions.weight',
                                 'decoder.layernorm_embedding.weight',
                                 'decoder.layernorm_embedding.bias',
                                 'decoder.output_projection.weight',
                                 ]:
                    param.requires_grad = False
                else:
                    logger.info(f'parameter {name} will be trained')
        if args.freeze_pretrained_transformer_body:
            logger.info(f'only tgt embeddings will be trained, '
                        f'freezing all other parameters')
            for name, param in model.named_parameters():
                if name not in ['decoder.embed_tokens.weight',
                                'decoder.embed_positions.weight',
                                'decoder.layernorm_embedding.weight',
                                'decoder.layernorm_embedding.bias',
                                'decoder.output_projection.weight',
                                ]:
                    param.requires_grad = False
                else:
                    logger.info(f'parameter {name} will be trained')


    # mbart finetuning
    if getattr(args, 'finetune_from_mbart_at') is not None \
            or getattr(args, 'finetune_from_mbart_with_reinit_xattn_at') is not None:
        if getattr(args, 'finetune_from_mbart_at'):
            logger.info(f'loading the (trimmed) mbart from '
                        f'{args.finetune_from_mbart_at}')
            pretrained_state_dict = torch.load(args.finetune_from_mbart_at)['model']
            model.upgrade_state_dict_named(pretrained_state_dict, '')
            for name, param in model.named_parameters():
                with torch.no_grad():
                    param.copy_(pretrained_state_dict[name])
        else:
            logger.info(f'loading the (trimmed) mbart from '
                        f'{args.finetune_from_mbart_with_reinit_xattn_at}, '
                        f'cross-attention will be reinitialized.')
            pretrained_state_dict = torch.load(args.finetune_from_mbart_with_reinit_xattn_at)['model']
            model.upgrade_state_dict_named(pretrained_state_dict, '')
            for name, param in model.named_parameters():
                with torch.no_grad():
                    if 'encoder_attn' not in name:
                        param.copy_(pretrained_state_dict[name])
        if args.only_finetune_cross_attn:
            logger.info(f'only cross-attention layers will be trained in addition to embeddings, '
                        f'freezing all other parameters')
            for name, param in model.named_parameters():
                if 'encoder_attn' not in name and \
                    name not in ['encoder.embed_tokens.weight',
                                 'encoder.embed_positions.weight',
                                 'encoder.layernorm_embedding.weight',
                                 'encoder.layernorm_embedding.bias',
                                 'decoder.embed_tokens.weight',
                                 'decoder.embed_positions.weight',
                                 'decoder.layernorm_embedding.weight',
                                 'decoder.layernorm_embedding.bias',
                                 'decoder.output_projection.weight']:
                    param.requires_grad = False
                else:
                    logger.info(f'parameter {name} will be trained')
        if args.freeze_pretrained_transformer_body:
            logger.info(f'only embeddings will be trained, '
                        f'freezing all other parameters')
            for name, param in model.named_parameters():
                if name not in ['encoder.embed_tokens.weight',
                                'encoder.embed_positions.weight',
                                'encoder.layernorm_embedding.weight',
                                'encoder.layernorm_embedding.bias',
                                'decoder.embed_tokens.weight',
                                'decoder.embed_positions.weight',
                                'decoder.layernorm_embedding.weight',
                                'decoder.layernorm_embedding.bias',
                                'decoder.output_projection.weight',
                                ]:
                    param.requires_grad = False
                else:
                    logger.info(f'parameter {name} will be trained')

    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info("task: {} ({})".format(args.task, task.__class__.__name__))
    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(args.criterion, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.max_sentences
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
        or num_updates >= max_update
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or num_updates >= max_update
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or num_updates >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
