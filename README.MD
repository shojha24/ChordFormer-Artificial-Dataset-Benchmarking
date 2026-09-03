# ChordFormer

ChordFormer is a Conformer-based audio chord recognition system for large
vocabularies. This repository contains the implementation associated with the
paper [ChordFormer: A Conformer-Based Model for Large-Vocabulary Chord
Recognition](https://ieeexplore.ieee.org/abstract/document/11313568).

## Provenance and attribution

This implementation was developed from code in the upstream repository
[ISMIR2019-Large-Vocabulary-Chord-Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition),
which accompanied the work *Large-Vocabulary Chord Transcription via Chord
Structure Decomposition*. The present repository should therefore be understood
as a research-derived extension rather than an independent reimplementation.

The upstream code was substantially adapted for the ChordFormer experiments,
including the Conformer-based encoder, structural output heads, reweighted
training objective, data-processing workflow, and ensemble inference described
in this README. Please cite both the upstream work and the ChordFormer work
when using code, models, or results from this repository. Review the upstream
repository's license and attribution requirements before redistributing code.

The model targets structural chord recognition rather than a single flat chord
classification problem. It predicts chord components independently and combines
them during decoding. This makes the representation suitable for triads, bass
inversions, sevenths, ninths, elevenths, and thirteenths, including rare chord
types that are difficult to learn from a long-tailed dataset.

## Method

The current model is implemented in `confor_head16.py`:

- CQT input: 252 dimensions, with four zero-padded dimensions before encoding.
- Four Conformer layers with 16 attention heads.
- Feed-forward dimension: 1024.
- Depthwise convolution kernel: 31.
- Convolution-first blocks with GroupNorm and dropout of 0.1.
- Six output heads: root/triad, bass, seventh, ninth, eleventh, and thirteenth.
- Reweighted cross-entropy loss to reduce the effect of class imbalance.
- Five-fold model ensemble at inference time, followed by HMM decoding.

The structural label limits used by the implementation are:

```text
triad       6 types per root
seventh     3 types
ninth       3 types
eleventh    2 types
thirteenth  2 types
```

The paper reports a 2.3 percentage-point improvement in frame-wise accuracy
and a 6 percentage-point improvement in class-wise accuracy on large-vocabulary
chord recognition benchmarks. See the paper for the full experimental protocol,
ablations, ARWCSR results, and inversion matrices.

## Installation

Use Python 3 and install the dependencies listed in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

The project uses PyTorch and is configured for CUDA training and inference. The
default model code selects GPU 5 in `confor_head16.py`; change
`set_specific_gpu()` if that device is unavailable. CPU execution requires
adjusting the GPU setup and may be substantially slower.

## Pretrained models

Five fold-specific checkpoints are included in `cache_data/`:

```text
chordformer_head16(1.0,1.0)_s0.best.sdict
chordformer_head16(1.0,1.0)_s1.best.sdict
chordformer_head16(1.0,1.0)_s2.best.sdict
chordformer_head16(1.0,1.0)_s3.best.sdict
chordformer_head16(1.0,1.0)_s4.best.sdict
```

`chord_recognition.py` loads the five fold models, averages their component
probabilities, and decodes the result with an HMM. Checkpoint discovery follows
the storage conventions used by `mir.nn.train.NetworkInterface`; keep the
checkpoint files in the repository's cache location unless your local storage
configuration says otherwise.

## Audio chord recognition

Run:

```bash
python3 chord_recognition.py path_to_audio_file path_to_output_file [chord_dict]
```

Example:

```bash
python3 chord_recognition.py example.mp3 example_chord.lab
```

The optional `chord_dict` selects the HMM chord vocabulary. Available files in
`data/` include:

- `submission` (default): submission vocabulary.
- `ismir2017`: ISMIR 2017/MIREX vocabulary.
- `full`: extended vocabulary used for broad evaluation.
- `extended`: extended chord vocabulary.

For example:

```bash
python3 chord_recognition.py example.mp3 example_chord.lab ismir2017
```

Custom vocabularies can be added or adjusted in `data/*_chord_list.txt`.
The output is a `.lab` chord annotation file.

## Dataset preparation

Training expects the JAM dataset root configured by `JAM_DATASET_PATH` in
`settings.py`. The default is `chord_data_1217`. The directory must contain:

```text
chord_data_1217/
    audio/
        filelist.txt
        song_name.mp3
        ...
    references_v2/
        song_name.jams
        ...
    chordlab/
        song_name.lab
        ...
```

The names in `audio/filelist.txt` must match the audio files. Each song must
also have a matching JAMS file in `references_v2/` and chord annotation in
`chordlab/`. Update `JAM_DATASET_PATH` to an absolute or workspace-relative
path before creating storage.

## Training

1. Create the cached framed feature and label stores:

   ```bash
   python3 storage_creation.py
   ```

   This creates `jams_cqt.h5d` and `jams_xchord.h5d` using the dataset and
   preprocessing settings in `settings.py`.

2. Train one cross-validation fold, where `fold` is `0` through `4`:

   ```bash
   python3 confor_head16.py 0
   ```

   Repeat for all five folds to produce the ensemble used by inference. The
   split membership is read from `data/train00.csv` through `data/test04.csv`.

3. Evaluate or visualize predictions with:

   ```bash
   python3 val_chordformer_head16.py
   ```

Training uses pitch shifting from -5 to +5 semitones in steps of three, a
maximum training sequence length of 1000 frames, and batch size 24. The exact
learning-rate schedule and validation settings are defined in
`confor_head16.py`.

## Repository layout

```text
confor_head16.py       ChordFormer architecture, loss, training entry point
chord_recognition.py   Single-file inference and HMM decoding
val_chordformer_head16.py  Evaluation and visualization utilities
storage_creation.py    CQT and structural-label cache creation
complex_chord.py       Chord decomposition and label definitions
extractors/            Audio, CQT, chord, and decoder components
io_new/                Annotation and feature I/O
mir/                   Shared MIR and neural-network infrastructure
cache_data/             Pretrained fold checkpoints
data/                  Chord vocabularies and fold split files
```

## Citation

### ChordFormer

```bibtex
@ARTICLE{11313568,
  author={Akram, Muhammad Waseem and Dettori, Stefano and Colla, Valentina and Buttazzo, Giorgio Carlo},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition}, 
  year={2026},
  volume={34},
  number={},
  pages={581-595},
  keywords={Hidden Markov models;Harmonic analysis;Vocabulary;Transformers;Decoding;Convolutional neural networks;Accuracy;Speech processing;Music;Time-frequency analysis;Automatic chord estimation;deep learning;music information processing;machine leanrning},
  doi={10.1109/TASLPRO.2025.3646468}}


### Upstream implementation and paper

```bibtex
@inproceedings{junyan2019large,
    title     = {Large-Vocabulary Chord Transcription via Chord Structure Decomposition},
    booktitle = {Proceedings of the 20th International Society for Music Information Retrieval Conference},
    year      = {2019},
    url       = {https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition}
}
```

Source repository: [music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition).
