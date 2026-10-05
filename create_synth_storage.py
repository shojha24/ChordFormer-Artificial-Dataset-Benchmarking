import os
import numpy as np
from mir import DataPool, io
from io_new.complex_chord_io import ComplexChordIO
from mir.nn.data_storage import FramedH5DataStorage
from extractors.cqt import CQTV2
from mir.extractors.misc import FrameCount
from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH, DEFAULT_WIN_SIZE, DEFAULT_CHORD_DICT

AUDIO_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-conditions/naturalistic/audio"
LAB_DIR = "/home/shojha/ChordFormer-Artificial-Dataset-Benchmarking/data/synth_naturalistic_lab"

def create_synth_pool(num_songs=5000):
    pool = DataPool('synth_naturalistic')
    pool.set_property('sr', DEFAULT_SR)
    pool.set_property('hop_length', DEFAULT_HOP_LENGTH)
    pool.set_property('win_size', DEFAULT_WIN_SIZE)
    pool.set_property('chord_dict', DEFAULT_CHORD_DICT)

    print(f"Registering {num_songs} synthetic tracks in DataPool...")
    for i in range(num_songs):
        name = f"song_{i}"
        audio_file = os.path.join(AUDIO_DIR, name, "mix.flac")
        lab_file = os.path.join(LAB_DIR, f"{name}.lab")
        entry = pool.new_entry(name)
        entry.append_file(audio_file, io.MusicIO, 'music')
        entry.append_file(lab_file, ComplexChordIO, 'xchord')
    return pool

def build_storage(num_workers=16, num_songs=5000):
    os.makedirs("cache_data/FrameCount/source=cqt/synth_naturalistic", exist_ok=True)
    os.makedirs("cache_data/CQTV2/synth_naturalistic", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    synth_pool = create_synth_pool(num_songs)
    
    print(f"Extracting CQTs using {num_workers} parallel workers...")
    synth_pool.append_extractor(CQTV2, 'cqt')
    synth_pool.activate_proxy('cqt', thread_number=num_workers, free=True)

    print("Computing frame counts...")
    synth_pool.append_extractor(FrameCount, 'n_frame', source='cqt')
    synth_pool.activate_proxy('n_frame', thread_number=num_workers, free=True)

    print("Creating FramedH5DataStorage for xchord labels...")
    storage_xchord = FramedH5DataStorage('synth_naturalistic_xchord', dtype=np.int16)
    if not storage_xchord.created:
        storage_xchord.create_and_cache(synth_pool.entries, 'xchord')
        print("synth_naturalistic_xchord created successfully.")
    else:
        print("synth_naturalistic_xchord already exists.")

    print("Creating FramedH5DataStorage for CQT features...")
    storage_cqt = FramedH5DataStorage('synth_naturalistic_cqt', dtype=np.int16)
    if not storage_cqt.created:
        storage_cqt.create_and_cache(synth_pool.entries, 'cqt')
        print("synth_naturalistic_cqt created successfully.")
    else:
        print("synth_naturalistic_cqt already exists.")

    print("Synthetic HDF5 storage creation finished.")

if __name__ == '__main__':
    build_storage()
