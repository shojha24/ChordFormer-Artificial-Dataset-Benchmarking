import os
import sys
import pickle
import argparse
import numpy as np
import torch

from mir.nn.train import NetworkInterface
from mir.nn.data_storage import FramedH5DataStorage
from mir.nn.data_provider import FramedDataProvider
from mir.nn.data_decorator import CQTPitchShifter
from confor_head16 import (
    ChordNet,
    ComplexChordShifter,
    LSTM_TRAIN_LENGTH,
    SPEC_DIM,
    SHIFT_LOW,
    SHIFT_HIGH,
)
from train_eval_test_split import get_train_set_ids, get_val_set_ids

def finetune_fold(
    slice_id,
    pretrained_path='cache_data/chordformer_head16_synth_pretrain.best.sdict',
    save_prefix='chordformer_head16_synth_ft',
    batch_size=36,
    val_batch_size=1,
    num_workers=8,
    learning_rates_dict=None,
    load_checkpoint=True,
    early_end_epochs=5,
):
    if learning_rates_dict is None:
        learning_rates_dict = {1e-4: 12, 1e-5: 8, 1e-6: 4}

    save_name = f"{save_prefix}_s{slice_id}"
    print(f"\n=== Stage 2: Fine-Tuning Real Fold {slice_id} ({save_name}) ===")
    print(f"Pretrained weights: {pretrained_path}")
    print(f"LR schedule: {learning_rates_dict}, Batch size: {batch_size}")

    weight_path = f'data/cross_subpart_weight{slice_id}.pkl'
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    with open(weight_path, 'rb') as f:
        cross_subpart_counter = pickle.load(f)

    train_indices = get_train_set_ids(slice_id)
    val_indices = get_val_set_ids(slice_id)
    print(f"Fold {slice_id}: {len(train_indices)} train songs, {len(val_indices)} val songs")

    storage_x = FramedH5DataStorage('jams_cqt')
    storage_y = FramedH5DataStorage('jams_xchord')
    storage_x.load_meta()
    storage_y.load_meta()

    train_provider = FramedDataProvider(
        train_sample_length=LSTM_TRAIN_LENGTH,
        shift_low=SHIFT_LOW,
        shift_high=SHIFT_HIGH,
        num_workers=num_workers,
        average_samples_per_song=1,
    )
    train_provider.link(
        storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH), subrange=train_indices
    )
    train_provider.link(storage_y, ComplexChordShifter(), subrange=train_indices)

    val_provider = FramedDataProvider(
        train_sample_length=-1,
        shift_low=0,
        shift_high=0,
        num_workers=num_workers,
        average_samples_per_song=1,
        need_shuffle=False,
    )
    val_provider.link(
        storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH), subrange=val_indices
    )
    val_provider.link(storage_y, ComplexChordShifter(), subrange=val_indices)

    # 1. Instantiate network with real ReweightedLoss
    net = ChordNet(cross_subpart_counter, triad_only=False)

    # 2. Check if a fine-tuning checkpoint already exists to resume from
    cp_path = f"cache_data/{save_name}.cp.sdict"
    final_path = f"cache_data/{save_name}.best.sdict"
    if load_checkpoint and (os.path.exists(cp_path) or os.path.exists(final_path)):
        print(f"Resuming fine-tuning from existing fold checkpoint for {save_name}")
        trainer = NetworkInterface(net, save_name=save_name, load_checkpoint=True)
    else:
        # 3. Warm-start with Stage 1 synthetic pretrained weights
        if os.path.exists(pretrained_path):
            print(f"Loading synthetic pretrained weights from {pretrained_path}...")
            device = 'cuda' if net.use_gpu else 'cpu'
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
            net.load_state_dict(state_dict)
            print("Successfully transferred pretrained weights!")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}. Training from scratch!")

        trainer = NetworkInterface(net, save_name=save_name, load_checkpoint=False)

    print(trainer)

    trainer.train_supervised(
        train_provider,
        val_provider,
        batch_size=batch_size,
        learning_rates_dict=learning_rates_dict,
        round_per_print=10,
        round_per_save=200,
        round_per_val=-1,
        early_end_epochs=early_end_epochs,
        val_batch_size=val_batch_size,
    )
    print(f"Fine-tuning complete for Fold {slice_id}. Checkpoint saved to cache_data/{save_name}.best.sdict")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 2: Real Domain Fine-Tuning for ChordFormer")
    parser.add_argument('fold', type=int, nargs='?', default=0, help='Fold ID (0..4, or -1 for all folds)')
    parser.add_argument('--pretrained_path', type=str, default='cache_data/chordformer_head16_synth_pretrain.best.sdict', help='Pretrained model path')
    parser.add_argument('--save_prefix', type=str, default='chordformer_head16_synth_ft', help='Save name prefix')
    parser.add_argument('--batch_size', type=int, default=36, help='Batch size (default: 36)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size (default: 1)')
    parser.add_argument('--num_workers', type=int, default=8, help='Worker threads (default: 8)')
    parser.add_argument('--no_load_checkpoint', action='store_true', help='Do not resume from existing checkpoint')
    parser.add_argument('--quick_test', action='store_true', help='Run a quick 1-epoch test per fold')
    args = parser.parse_args()

    lr_schedule = {1e-4: 1, 1e-5: 1} if args.quick_test else {1e-4: 12, 1e-5: 8, 1e-6: 4}

    if args.fold == -1:
        folds = list(range(5))
    else:
        folds = [args.fold]

    for f in folds:
        finetune_fold(
            slice_id=f,
            pretrained_path=args.pretrained_path,
            save_prefix=args.save_prefix,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            learning_rates_dict=lr_schedule,
            load_checkpoint=not args.no_load_checkpoint,
        )
