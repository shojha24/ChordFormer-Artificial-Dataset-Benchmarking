import os
import argparse
import numpy as np
from mir import DataPool, io
from io_new.complex_chord_io import ComplexChordIO
from mir.nn.data_storage import FramedH5DataStorage
from extractors.cqt import CQTV2
from mir.extractors.misc import FrameCount
from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH, DEFAULT_WIN_SIZE, DEFAULT_CHORD_DICT

GEN_ROOT = "/home/shojha/probabilistic-composition-generator/gen"
if os.path.isdir(os.path.join(GEN_ROOT, "acr-conditions-5k")):
    DEFAULT_AUDIO_DIR = os.path.join(GEN_ROOT, "acr-conditions-5k/naturalistic/audio")
    DEFAULT_NUM_SONGS = 5000
else:
    DEFAULT_AUDIO_DIR = os.path.join(GEN_ROOT, "acr-conditions-10k/naturalistic/audio")
    DEFAULT_NUM_SONGS = 10000

DEFAULT_LAB_DIR = "/home/shojha/ChordFormer-Artificial-Dataset-Benchmarking/data/synth_naturalistic_lab"

def create_synth_pool(num_songs=DEFAULT_NUM_SONGS, audio_dir=DEFAULT_AUDIO_DIR, lab_dir=DEFAULT_LAB_DIR):
    pool = DataPool('synth_naturalistic')
    pool.set_property('sr', DEFAULT_SR)
    pool.set_property('hop_length', DEFAULT_HOP_LENGTH)
    pool.set_property('win_size', DEFAULT_WIN_SIZE)
    pool.set_property('chord_dict', DEFAULT_CHORD_DICT)

    print(f"Registering {num_songs} synthetic tracks in DataPool from {audio_dir}...")
    for i in range(num_songs):
        name = f"song_{i}"
        audio_file = os.path.join(audio_dir, name, "mix.flac")
        lab_file = os.path.join(lab_dir, f"{name}.lab")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Missing audio: {audio_file}")
        if not os.path.exists(lab_file):
            raise FileNotFoundError(f"Missing lab: {lab_file}")
        entry = pool.new_entry(name)
        entry.append_file(audio_file, io.MusicIO, 'music')
        entry.append_file(lab_file, ComplexChordIO, 'xchord')
    return pool

def build_storage(num_workers=16, num_songs=DEFAULT_NUM_SONGS, audio_dir=DEFAULT_AUDIO_DIR, lab_dir=DEFAULT_LAB_DIR, overwrite=False):
    os.makedirs("cache_data/FrameCount/source=cqt/synth_naturalistic", exist_ok=True)
    os.makedirs("cache_data/CQTV2/synth_naturalistic", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    if overwrite:
        for f in ["dataset/synth_naturalistic_cqt.h5d", "dataset/synth_naturalistic_xchord.h5d"]:
            if os.path.exists(f):
                print(f"Removing existing {f} for overwrite...")
                os.remove(f)

    synth_pool = create_synth_pool(num_songs, audio_dir, lab_dir)
    
    print(f"Extracting CQTs using {num_workers} parallel workers...")
    synth_pool.append_extractor(CQTV2, 'cqt')
    synth_pool.activate_proxy('cqt', thread_number=num_workers, free=True)

    print("Computing frame counts...")
    synth_pool.append_extractor(FrameCount, 'n_frame', source='cqt')
    synth_pool.activate_proxy('n_frame', thread_number=num_workers, free=True)

    print("Creating FramedH5DataStorage for xchord labels...")
    storage_xchord = FramedH5DataStorage('synth_naturalistic_xchord', dtype=np.int16)
    if not storage_xchord.created or overwrite:
        storage_xchord.create_and_cache(synth_pool.entries, 'xchord')
        print("synth_naturalistic_xchord created successfully.")
    else:
        print("synth_naturalistic_xchord already exists. Use --overwrite to rebuild.")

    print("Creating FramedH5DataStorage for CQT features...")
    storage_cqt = FramedH5DataStorage('synth_naturalistic_cqt', dtype=np.int16)
    if not storage_cqt.created or overwrite:
        storage_cqt.create_and_cache(synth_pool.entries, 'cqt')
        print("synth_naturalistic_cqt created successfully.")
    else:
        print("synth_naturalistic_cqt already exists. Use --overwrite to rebuild.")

    print(f"Synthetic HDF5 storage creation finished for {num_songs} songs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract CQTs and build synthetic HDF5 storage")
    parser.add_argument('--num_songs', type=int, default=DEFAULT_NUM_SONGS, help=f"Number of songs to index (default: {DEFAULT_NUM_SONGS})")
    parser.add_argument('--num_workers', type=int, default=16, help="Worker threads for CQT extraction (default: 16)")
    parser.add_argument('--audio_dir', type=str, default=DEFAULT_AUDIO_DIR, help="Audio directory containing song_*/mix.flac")
    parser.add_argument('--lab_dir', type=str, default=DEFAULT_LAB_DIR, help="Directory containing .lab files")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing dataset HDF5 files")
    args = parser.parse_args()

    build_storage(
        num_workers=args.num_workers,
        num_songs=args.num_songs,
        audio_dir=args.audio_dir,
        lab_dir=args.lab_dir,
        overwrite=args.overwrite,
    )
