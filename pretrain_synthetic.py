import os
import argparse
import numpy as np
import torch

from mir.nn.train import NetworkInterface
from mir.nn.data_storage import FramedH5DataStorage
from mir.nn.data_provider import FramedDataProvider
from confor_head16 import (
    ChordNet,
    ComplexChordShifter,
    LSTM_TRAIN_LENGTH,
    SPEC_DIM,
    SHIFT_LOW,
    SHIFT_HIGH,
)
from cqt_specaugment import CQTSpecAugmentPitchShifter

def pretrain_synthetic(
    batch_size=36,
    val_batch_size=1,
    num_workers=8,
    save_name='chordformer_head16_synth_pretrain',
    learning_rates_dict=None,
    total_songs=5000,
    val_count=500,
    seed=42,
    load_checkpoint=True,
    early_end_epochs=5,
):
    if learning_rates_dict is None:
        learning_rates_dict = {1e-3: 15, 1e-4: 10, 1e-5: 5}

    print(f"=== Stage 1: Synthetic Pretraining ({save_name}) ===")
    print(f"Total songs: {total_songs} (Train: {total_songs - val_count}, Val: {val_count})")
    print(f"Batch size: {batch_size}, LR schedule: {learning_rates_dict}")

    indices = np.random.RandomState(seed).permutation(total_songs)
    train_indices = indices[:-val_count]
    val_indices = indices[-val_count:]

    storage_x = FramedH5DataStorage('synth_naturalistic_cqt')
    storage_y = FramedH5DataStorage('synth_naturalistic_xchord')
    storage_x.load_meta()
    storage_y.load_meta()

    print(f"Loaded storage: {storage_x.total_song_count} songs in synth_naturalistic_cqt")

    # Train provider with SpecAugment enabled
    train_provider = FramedDataProvider(
        train_sample_length=LSTM_TRAIN_LENGTH,
        shift_low=SHIFT_LOW,
        shift_high=SHIFT_HIGH,
        num_workers=num_workers,
        average_samples_per_song=1,
    )
    train_shifter = CQTSpecAugmentPitchShifter(
        SPEC_DIM, SHIFT_LOW, SHIFT_HIGH, enabled=True
    )
    train_provider.link(storage_x, train_shifter, subrange=train_indices)
    train_provider.link(storage_y, ComplexChordShifter(), subrange=train_indices)

    # Validation provider (deterministic, no SpecAugment, no pitch shift)
    val_provider = FramedDataProvider(
        train_sample_length=-1,
        shift_low=0,
        shift_high=0,
        num_workers=num_workers,
        average_samples_per_song=1,
        need_shuffle=False,
    )
    val_shifter = CQTSpecAugmentPitchShifter(
        SPEC_DIM, SHIFT_LOW, SHIFT_HIGH, enabled=False
    )
    val_provider.link(storage_x, val_shifter, subrange=val_indices)
    val_provider.link(storage_y, ComplexChordShifter(), subrange=val_indices)

    # Unweighted cross entropy across all 6 heads (uniform prior over synthetic harmonic space)
    net = ChordNet(cross_subpart_counter=None, triad_only=False)
    trainer = NetworkInterface(
        net,
        save_name=save_name,
        load_checkpoint=load_checkpoint,
    )
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
    print(f"Pretraining complete. Checkpoint saved to cache_data/{save_name}.best.sdict")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 1: Pretrain ChordFormer on Synthetic Naturalistic Dataset")
    parser.add_argument('--batch_size', type=int, default=36, help='Training batch size (default: 36)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size (default: 1)')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader worker processes (default: 8)')
    parser.add_argument('--save_name', type=str, default='chordformer_head16_synth_pretrain', help='Model save name')
    parser.add_argument('--no_load_checkpoint', action='store_true', help='Do not resume from existing checkpoint')
    parser.add_argument('--quick_test', action='store_true', help='Run a quick 1-epoch test')
    args = parser.parse_args()

    lr_schedule = {1e-3: 1, 1e-4: 1} if args.quick_test else {1e-3: 15, 1e-4: 10, 1e-5: 5}

    pretrain_synthetic(
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        save_name=args.save_name,
        learning_rates_dict=lr_schedule,
        load_checkpoint=not args.no_load_checkpoint,
    )
