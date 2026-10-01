### The Synthetic Pretrain → Real Fine-Tune Architecture                                  
                                                                                            
  The Synthetic Pretrain → Real Fine-Tune pipeline is conceptually and empirically the      
  superior strategy for solving Automatic Chord Recognition (ACR) on large vocabularies:    
                                                                                            
  1. Acoustic Chord Theory & Pitch Class Induction (Stage 1): The model is exposed to 5,000 
  songs with uniform distributions over rare roots, complex voicings, and extended pitch    
  structures (9ths, 11ths, 13ths, and altered dominants). It builds rich internal           
  representations for harmonic components across a diverse range of synthetically generated 
  instrument voicings.                                                                      
  2. Domain Adaptation & Prior Realignment (Stage 2): Fine-tuning on the real 1217 training 
  fold (D_{real, train}^{(k)}) with all layers unfrozen shifts the decision boundaries back 
  to human timing, vocal bleed, room reverberation, and natural power-law chord             
  distributions.                                                                            
  3. No Acoustic Contamination (Stage 3 & Inference): The model's weights end their training
  exclusively on real music, and inference decoding uses the natural HMM transition matrices
  fit on real human songs.                                                                  
  ──────                                                                                    
  ### End-to-End Implementation Blueprint                                                   
                                                                                            
    [ Step 1: Pre-processing ]                                                              
      Convert 5k JSONs to .lab                                                              
      Extract CQT (252-bin) + XChord (6-head) -> HDF5 Caches (synth_cqt.h5d, synth_xchord.  
  h5d)                                                                                      
                    │                                                                       
                    ▼                                                                       
    [ Step 2: Stage 1 - Synthetic Pretraining ]                                             
      Dataloader: CQTPitchShifter + SpecAugment (Time & Frequency Masking on clean CQT)     
      Train on 5,000 synthetic songs (e.g., 4,500 train / 500 val)                          
      Output: cache_data/chordformer_head16_synth_pretrain.best.sdict                       
                    │                                                                       
                    ▼                                                                       
    [ Step 3: Stage 2 - Real Domain Adaptation / Fine-Tuning (Folds 0..4) ]                 
      For each fold k ∈ {0, 1, 2, 3, 4}:                                                    
        - Initialize from Stage 1 checkpoint                                                
        - Unfreeze ALL layers (lower LR: 1e-4 -> 1e-5 -> 1e-6)                              
        - Train on D_real,train^(k) with real ReweightedLoss (cross_subpart_weight{k}.pkl)  
        - Validate & select best model on D_real,val^(k)                                    
        - Output: cache_data/chordformer_head16_synth_ft_s{k}.best.sdict                    
                    │                                                                       
                    ▼                                                                       
    [ Step 4: Stage 3 - Real HMM Viterbi Decoding & Evaluation ]                            
      Decode test0{k} with HMM fit strictly on real data                                    
      Evaluate root, triad, 7th, 9th, 11th, 13th, MIREX, and class-wise recall              
  ──────                                                                                    
  ### Step 1: Data Conversion and HDF5 Storage Creation                                     
                                                                                            
  #### 1. Label Conversion Script (convert_synth_labels.py)                                 
                                                                                            
  Maps the 5,000 synthetic audio tracks in naturalistic/audio/ to the corresponding pop-    
  rock-labels and jazz-labels JSON files, outputting standard Harte .lab files:             
                                                                                            
    import os                                                                               
    import json                                                                             
    from pathlib import Path                                                                
                                                                                            
    AUDIO_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-                  
  conditions/naturalistic/audio"                                                            
    POP_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-target-500k/pop-    
  rock-labels"                                                                              
    JAZZ_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-target-500k/jazz-  
  labels"                                                                                   
    OUT_LAB_DIR = "/home/shojha/ChordFormer-Artificial-Dataset-                             
  Benchmarking/data/synth_naturalistic_lab"                                                 
                                                                                            
    os.makedirs(OUT_LAB_DIR, exist_ok=True)                                                 
                                                                                            
    for i in range(5000):                                                                   
        song_name = f"song_{i}"                                                             
        # Ordinals 0..2499 map to pop-rock; 2500..4999 map to jazz                          
        if i < 2500:                                                                        
            json_path = os.path.join(POP_DIR, f"song_{i}.json")                             
        else:                                                                               
            json_path = os.path.join(JAZZ_DIR, f"song_{i - 2500}.json")                     
                                                                                            
        with open(json_path, "r") as f:                                                     
            data = json.load(f)                                                             
                                                                                            
        out_lab_path = os.path.join(OUT_LAB_DIR, f"{song_name}.lab")                        
        with open(out_lab_path, "w") as out:                                                
            for chord in data["chords"]:                                                    
                label = "N" if chord.get("is_no_chord", False) else chord["harte"]          
                out.write(f"{chord['time_start']:.3f}\t{chord['time_end']:.3f}\t{label}\n") 
                                                                                            
    print(f"Generated 5,000 .lab files in {OUT_LAB_DIR}")                                   
                                                                                            
  #### 2. CQT & XChord HDF5 Storage Creation (create_synth_storage.py)                      
                                                                                            
  Computes CQT spectrograms and 6-dimensional structural chord labels using the exact       
  parameters used for chord_data_1217 (sr=22050, hop=512, win=8192, CQTV2):                 
                                                                                            
    import os                                                                               
    import numpy as np                                                                      
    from mir import DataPool, io                                                            
    from io_new.complex_chord_io import ComplexChordIO                                      
    from mir.nn.data_storage import FramedH5DataStorage                                     
    from extractors.cqt import CQTV2                                                        
    from mir.extractors.misc import FrameCount                                              
    from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH, DEFAULT_WIN_SIZE,                  
  DEFAULT_CHORD_DICT                                                                        
                                                                                            
    AUDIO_DIR = "/home/shojha/probabilistic-composition-generator/gen/acr-                  
  conditions/naturalistic/audio"                                                            
    LAB_DIR = "/home/shojha/ChordFormer-Artificial-Dataset-                                 
  Benchmarking/data/synth_naturalistic_lab"                                                 
                                                                                            
    def create_synth_pool():                                                                
        pool = DataPool('synth_naturalistic')                                               
        pool.set_property('sr', DEFAULT_SR)                                                 
        pool.set_property('hop_length', DEFAULT_HOP_LENGTH)                                 
        pool.set_property('win_size', DEFAULT_WIN_SIZE)                                     
        pool.set_property('chord_dict', DEFAULT_CHORD_DICT)                                 
                                                                                            
        for i in range(5000):                                                               
            name = f"song_{i}"                                                              
            audio_file = os.path.join(AUDIO_DIR, name, "mix.flac")                          
            lab_file = os.path.join(LAB_DIR, f"{name}.lab")                                 
            entry = pool.new_entry(name)                                                    
            entry.append_file(audio_file, io.MusicIO, 'music')                              
            entry.append_file(lab_file, ComplexChordIO, 'xchord')                           
        return pool                                                                         
                                                                                            
    if __name__ == '__main__':                                                              
        synth_pool = create_synth_pool()                                                    
        synth_pool.append_extractor(CQTV2, 'cqt')                                           
        synth_pool.activate_proxy('cqt', thread_number=8, free=True)                        
        synth_pool.append_extractor(FrameCount, 'n_frame', source='cqt')                    
        synth_pool.activate_proxy('n_frame', thread_number=8, free=True)                    
                                                                                            
        storage_xchord = FramedH5DataStorage('synth_naturalistic_xchord', dtype=np.int16)   
        if not storage_xchord.created:                                                      
            storage_xchord.create_and_cache(synth_pool.entries, 'xchord')                   
                                                                                            
        storage_cqt = FramedH5DataStorage('synth_naturalistic_cqt', dtype=np.int16)         
        if not storage_cqt.created:                                                         
            storage_cqt.create_and_cache(synth_pool.entries, 'cqt')                         
  ──────                                                                                    
  ### Step 2: SpecAugment for Clean CQTs                                                    
                                                                                            
  To prevent the model from overfitting to pristine SoundFont transients, we apply          
  SpecAugment directly to the CQT frames within the training data decorator:                
                                                                                            
    import numpy as np                                                                      
    import random                                                                           
    from mir.nn.data_decorator import CQTPitchShifter                                       
                                                                                            
    class CQTSpecAugmentPitchShifter(CQTPitchShifter):                                      
        """                                                                                 
        Extends CQTPitchShifter to apply Time and Frequency Masking (SpecAugment)           
        during synthetic pretraining to simulate acoustic masking and bleed.                
        """                                                                                 
        def __init__(self, spec_dim, shift_low, shift_high, shift_step=3,                   
                     max_freq_mask=24, max_time_mask=60, num_freq_masks=2, num_time_masks=2,
                     enabled=True):                                                         
            super().__init__(spec_dim, shift_low, shift_high, shift_step)                   
            self.max_freq_mask = max_freq_mask                                              
            self.max_time_mask = max_time_mask                                              
            self.num_freq_masks = num_freq_masks                                            
            self.num_time_masks = num_time_masks                                            
            self.enabled = enabled                                                          
                                                                                            
        def pitch_shift(self, data, shift):                                                 
            # 1. Standard pitch shift slice                                                 
            spec = super().pitch_shift(data, shift).copy() # shape: (T, F=252)              
                                                                                            
            if not self.enabled:                                                            
                return spec                                                                 
                                                                                            
            t_len, f_len = spec.shape                                                       
                                                                                            
            # 2. Frequency Masking (zero out consecutive frequency bands)                   
            for _ in range(self.num_freq_masks):                                            
                f_width = random.randint(0, self.max_freq_mask)                             
                f_start = random.randint(0, max(0, f_len - f_width))                        
                spec[:, f_start:f_start + f_width] = 0                                      
                                                                                            
            # 3. Time Masking (zero out consecutive time frames)                            
            for _ in range(self.num_time_masks):                                            
                t_width = random.randint(0, min(self.max_time_mask, t_len))                 
                t_start = random.randint(0, max(0, t_len - t_width))                        
                spec[t_start:t_start + t_width, :] = 0                                      
                                                                                            
            return spec                                                                     
  ──────                                                                                    
  ### Step 3: Stage 1 — Synthetic Pretraining (pretrain_synthetic.py)                       
                                                                                            
  • Dataset: All 5,000 synthetic songs (split 4,500 train / 500 val).                       
  • Augmentation: Pitch shifting (-5 to +5 semitones) + SpecAugment.                        
  • Weights: Equal / unweighted cross-entropy across all 6 heads (since chords are already  
  uniformly distributed).                                                                   
  • Epoch schedule: Standard learning rates (10⁻³ → 10⁻⁴ → 10⁻⁵).                           
                                                                                            
    import numpy as np                                                                      
    from mir.nn.train import NetworkInterface                                               
    from mir.nn.data_storage import FramedH5DataStorage                                     
    from mir.nn.data_provider import FramedDataProvider                                     
    from confor_head16 import ChordNet, ComplexChordShifter, LSTM_TRAIN_LENGTH, SPEC_DIM,   
  SHIFT_LOW, SHIFT_HIGH                                                                     
                                                                                            
    TOTAL_SYNTH_SONGS = 5000                                                                
    VAL_COUNT = 500                                                                         
                                                                                            
    indices = np.random.RandomState(42).permutation(TOTAL_SYNTH_SONGS)                      
    train_indices = indices[:-VAL_COUNT]                                                    
    val_indices = indices[-VAL_COUNT:]                                                      
                                                                                            
    storage_x = FramedH5DataStorage('synth_naturalistic_cqt')                               
    storage_y = FramedH5DataStorage('synth_naturalistic_xchord')                            
                                                                                            
    # Train provider with SpecAugment enabled                                               
    train_provider = FramedDataProvider(                                                    
        train_sample_length=LSTM_TRAIN_LENGTH,                                              
        shift_low=SHIFT_LOW, shift_high=SHIFT_HIGH,                                         
        num_workers=8, average_samples_per_song=1                                           
    )                                                                                       
    train_shifter = CQTSpecAugmentPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH,             
  enabled=True)                                                                             
    train_provider.link(storage_x, train_shifter, subrange=train_indices)                   
    train_provider.link(storage_y, ComplexChordShifter(), subrange=train_indices)           
                                                                                            
    # Validation provider (no SpecAugment, no pitch shift)                                  
    val_provider = FramedDataProvider(                                                      
        train_sample_length=-1, shift_low=0, shift_high=0,                                  
        num_workers=8, average_samples_per_song=1, need_shuffle=False                       
    )                                                                                       
    val_shifter = CQTSpecAugmentPitchShifter(SPEC_DIM, 0, 0, enabled=False)                 
    val_provider.link(storage_x, val_shifter, subrange=val_indices)                         
    val_provider.link(storage_y, ComplexChordShifter(), subrange=val_indices)               
                                                                                            
    # Pretrain without class reweighting (uniform prior)                                    
    trainer = NetworkInterface(                                                             
        ChordNet(cross_subpart_counter=None, triad_only=False),                             
        save_name='chordformer_head16_synth_pretrain',                                      
        load_checkpoint=True                                                                
    )                                                                                       
                                                                                            
    trainer.train_supervised(                                                               
        train_provider, val_provider, batch_size=48,                                        
        learning_rates_dict={1e-3: 20, 1e-4: 10, 1e-5: 5},                                  
        round_per_print=10, round_per_save=500, round_per_val=-1,                           
        early_end_epochs=5, val_batch_size=1                                                
    )                                                                                       
  ──────                                                                                    
  ### Step 4: Stage 2 — Real Domain Fine-Tuning (finetune_real_fold.py)                     
                                                                                            
  For each fold k ∈ {0, 1, 2, 3, 4}:                                                        
                                                                                            
  1. Initialize from cache_data/chordformer_head16_synth_pretrain.best.sdict.               
  2. Leave ALL layers unfrozen: Lower convolutional feature extractor and Conformer         
  attention blocks adapt to real studio mixing and acoustics.                               
  3. Loss Function: Switch to ReweightedLoss(counter=cross_subpart_weight{k}.pkl): this     
  realigns gradients to penalize errors according to natural class rarity.                  
  4. Learning Rate: Fine-tune with a low, conservative learning rate schedule:              
                                                                                            
    {1 × 10⁻⁴ : 12 epochs,  1 × 10⁻⁵ : 8 epochs,  1 × 10⁻⁶ : 4 epochs}                      
                                                                                            
  5. Validation Guardrail: Validation and early stopping run on D_{real, val}^{(k)}. Best   
  checkpoint saved as chordformer_head16_synth_ft_s{k}.best.sdict.                          
                                                                                            
    import sys, pickle, torch                                                               
    import numpy as np                                                                      
    from mir.nn.train import NetworkInterface                                               
    from mir.nn.data_storage import FramedH5DataStorage                                     
    from mir.nn.data_provider import FramedDataProvider                                     
    from mir.nn.data_decorator import CQTPitchShifter                                       
    from confor_head16 import ChordNet, ComplexChordShifter, LSTM_TRAIN_LENGTH, SPEC_DIM,   
  SHIFT_LOW, SHIFT_HIGH                                                                     
    from train_eval_test_split import get_train_set_ids, get_val_set_ids                    
                                                                                            
    slice_id = int(sys.argv[1])  # 0 to 4                                                   
                                                                                            
    # Load real fold-specific class weights                                                 
    with open(f'data/cross_subpart_weight{slice_id}.pkl', 'rb') as f:                       
        cross_subpart_counter = pickle.load(f)                                              
                                                                                            
    train_indices = get_train_set_ids(slice_id)                                             
    val_indices = get_val_set_ids(slice_id)                                                 
                                                                                            
    storage_x = FramedH5DataStorage('jams_cqt')                                             
    storage_y = FramedH5DataStorage('jams_xchord')                                          
                                                                                            
    train_provider = FramedDataProvider(                                                    
        train_sample_length=LSTM_TRAIN_LENGTH,                                              
        shift_low=SHIFT_LOW, shift_high=SHIFT_HIGH,                                         
        num_workers=8, average_samples_per_song=1                                           
    )                                                                                       
    train_provider.link(storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH),        
  subrange=train_indices)                                                                   
    train_provider.link(storage_y, ComplexChordShifter(), subrange=train_indices)           
                                                                                            
    val_provider = FramedDataProvider(                                                      
        train_sample_length=-1, shift_low=0, shift_high=0,                                  
        num_workers=8, average_samples_per_song=1, need_shuffle=False                       
    )                                                                                       
    val_provider.link(storage_x, CQTPitchShifter(SPEC_DIM, SHIFT_LOW, SHIFT_HIGH),          
  subrange=val_indices)                                                                     
    val_provider.link(storage_y, ComplexChordShifter(), subrange=val_indices)               
                                                                                            
    # 1. Instantiate network with real ReweightedLoss                                       
    net = ChordNet(cross_subpart_counter, triad_only=False)                                 
                                                                                            
    # 2. Warm-start with Stage 1 synthetic pretrained weights                               
    pretrained_path = 'cache_data/chordformer_head16_synth_pretrain.best.sdict'             
    print(f"Loading synthetic pretrained weights from {pretrained_path}")                   
    checkpoint = torch.load(pretrained_path, map_location='cuda' if net.use_gpu else 'cpu') 
    net.load_state_dict(checkpoint['net'])                                                  
                                                                                            
    # 3. Fine-tune on real fold                                                             
    trainer = NetworkInterface(                                                             
        net,                                                                                
        save_name=f'chordformer_head16_synth_ft_s{slice_id}',                               
        load_checkpoint=False
    )
  
    trainer.train_supervised(
        train_provider, val_provider, batch_size=48,
        learning_rates_dict={1e-4: 12, 1e-5: 8, 1e-6: 4},
        round_per_print=10, round_per_save=500, round_per_val=-1,
        early_end_epochs=5, val_batch_size=1
    )
  ──────
  ### Step 5: Stage 3 — Real HMM Decoding & Evaluation
  
  1. The test fold data/test0{k}.csv is evaluated using the fine-tuned model
  chordformer_head16_synth_ft_s{k}.
  2. The HMM decoder in chord_recognition.py uses transition and prior matrices fit on real 
  music (data/submission_transition.pkl).
  3. Benchmarking metrics are calculated with eval_heads.py and test_for_all_mirex.py:      
      • Root, Triad, 7th, 9th, 11th, 13th head accuracies.
      • MIREX flat metrics (Root, Thirds, Triads, Sevenths, Tetrads).
      • Class-wise accuracy / recall for rare chords, directly quantifying the gain over the
      baseline reported in eval_results.txt.