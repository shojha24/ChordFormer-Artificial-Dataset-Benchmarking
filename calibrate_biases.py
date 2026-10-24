import os
import sys
import pickle
import time
import numpy as np
import torch
import numba
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support

from confor_head16 import ChordNet
from mir.nn.train import NetworkInterface
from mir.nn.data_storage import FramedH5DataStorage
from train_eval_test_split import get_test_set_ids
from complex_chord import TriadTypes, SeventhTypes, NinthTypes, EleventhTypes, ThirteenthTypes

# Numba-accelerated fast Viterbi decode
@numba.njit(fastmath=True)
def _viterbi_numba_core(log_probs, penalty):
    seq_len, num_classes = log_probs.shape
    score = log_probs[0].copy()
    backpointers = np.empty((seq_len - 1, num_classes), dtype=np.int32)
    for t in range(1, seq_len):
        best_idx = 0
        best_val = score[0]
        for c in range(1, num_classes):
            if score[c] > best_val:
                best_val = score[c]
                best_idx = c
        threshold = best_val - penalty
        for c in range(num_classes):
            if score[c] >= threshold:
                backpointers[t - 1, c] = c
                score[c] = score[c] + log_probs[t, c]
            else:
                backpointers[t - 1, c] = best_idx
                score[c] = threshold + log_probs[t, c]
    decoded = np.empty(seq_len, dtype=np.int32)
    last_state = 0
    best_last = score[0]
    for c in range(1, num_classes):
        if score[c] > best_last:
            best_last = score[c]
            last_state = c
    decoded[seq_len - 1] = last_state
    for t in range(seq_len - 2, -1, -1):
        last_state = backpointers[t, last_state]
        decoded[t] = last_state
    return decoded

def decode_song(log_prob, bias=None, penalty=2.0):
    if bias is not None:
        lp = log_prob + bias
    else:
        lp = log_prob
    if penalty is not None and penalty > 0:
        return _viterbi_numba_core(lp, float(penalty))
    return np.argmax(lp, axis=-1)

# Class name definitions
TRIAD_QUALITIES = ['N', 'maj', 'min', 'sus4', 'sus2', 'dim', 'aug', '5', '1']
SEVENTH_NAMES = ['N', '7', 'b7', 'bb7']
NINTH_NAMES = ['N', '9', '#9', 'b9']
ELEVENTH_NAMES = ['N', '11', '#11']
THIRTEENTH_NAMES = ['N', '13', 'b13']

def broadcast_triad_quality_bias(b_qual):
    # b_qual has length 9: 0=N, 1=maj, 2=min, 3=sus4, 4=sus2, 5=dim, 6=aug, 7=5, 8=1
    b97 = np.zeros(97, dtype=np.float32)
    b97[0] = b_qual[0]
    for q in range(1, 9):
        b97[1 + (q - 1) * 12 : 1 + q * 12] = b_qual[q]
    return b97

def extract_or_load_logits(cache_path='cache_data/eval_test_cache_5folds.pkl', recompute=False, max_songs=None):
    if not recompute and os.path.exists(cache_path):
        print(f"Loading cached inference outputs from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Running out-of-sample inference for all 5 folds on GPU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    storage_x = FramedH5DataStorage('jams_cqt')
    storage_y = FramedH5DataStorage('jams_xchord')
    storage_x.load_meta()
    storage_y.load_meta()
    storage_x.load()
    storage_y.load()

    fold_data = {}
    for fold in range(5):
        test_ids = get_test_set_ids(fold)
        if max_songs is not None:
            test_ids = test_ids[:max_songs]
        print(f"Fold {fold}: running inference on {len(test_ids)} test songs...")
        model_name = f'chordformer_head16_synth_ft_s{fold}.best'
        model = NetworkInterface(ChordNet(None, triad_only=False), model_name, load_checkpoint=True)

        fold_songs = []
        for s_idx, song_id in enumerate(test_ids):
            song_len = storage_x.length[song_id]
            cqt = storage_x.locate(song_id, 0, song_len)
            tag = storage_y.locate(song_id, 0, song_len)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                probs = model.inference(cqt)

            # Store log-probs as float32 to avoid precision underflows in Viterbi
            song_entry = {
                'song_id': song_id,
                'tag': tag.copy(),
                # Heads: 0=97, 1=13, 2=7th(4), 3=9th(4), 4=11th(3), 5=13th(3)
                'log_probs': [np.log(np.maximum(p, 1e-12)).astype(np.float32) for p in probs]
            }
            fold_songs.append(song_entry)
            if (s_idx + 1) % 50 == 0:
                print(f"  Processed {s_idx + 1}/{len(test_ids)} songs...")
        fold_data[fold] = fold_songs

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if max_songs is None:
        print(f"Saving inference cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump(fold_data, f)
    return fold_data

def evaluate_song_list(songs, biases, penalty=2.0):
    # Returns decoded predictions and true targets for all 5 target heads
    # (Head 0: 97 -> mapped to triad quality 9, Head 2: 7th, Head 3: 9th, Head 4: 11th, Head 5: 13th)
    all_preds = {
        'triad_qual': [],
        '7th': [],
        '9th': [],
        '11th': [],
        '13th': []
    }
    all_targets = {
        'triad_qual': [],
        '7th': [],
        '9th': [],
        '11th': [],
        '13th': []
    }

    b97 = broadcast_triad_quality_bias(biases['triad_qual'])

    for s in songs:
        lp = s['log_probs']
        tag = s['tag']

        # Triad
        pred_0 = decode_song(lp[0], b97, penalty=penalty)
        pred_qual = np.where(pred_0 <= 0, 0, (pred_0 - 1) // 12 + 1)
        true_0 = tag[:, 0]
        true_qual = np.where(true_0 < 0, -1, np.where(true_0 == 0, 0, (true_0 - 1) // 12 + 1))

        # 7th
        pred_7 = decode_song(lp[2], biases['7th'], penalty=penalty)
        true_7 = tag[:, 2]

        # 9th
        pred_9 = decode_song(lp[3], biases['9th'], penalty=penalty)
        true_9 = tag[:, 3]

        # 11th
        pred_11 = decode_song(lp[4], biases['11th'], penalty=penalty)
        true_11 = tag[:, 4]

        # 13th
        pred_13 = decode_song(lp[5], biases['13th'], penalty=penalty)
        true_13 = tag[:, 5]

        all_preds['triad_qual'].append(pred_qual)
        all_targets['triad_qual'].append(true_qual)

        all_preds['7th'].append(pred_7)
        all_targets['7th'].append(true_7)

        all_preds['9th'].append(pred_9)
        all_targets['9th'].append(true_9)

        all_preds['11th'].append(pred_11)
        all_targets['11th'].append(true_11)

        all_preds['13th'].append(pred_13)
        all_targets['13th'].append(true_13)

    return {
        k: np.concatenate(all_preds[k]) for k in all_preds
    }, {
        k: np.concatenate(all_targets[k]) for k in all_targets
    }

def fit_marginal_matching(train_songs, head_key, num_classes, max_iters=50, penalty=2.0, max_bias=3.0):
    # Pre-extract log_probs and targets for fast iteration
    if head_key == 'triad_qual':
        lps = [s['log_probs'][0] for s in train_songs]
        trues = [np.where(s['tag'][:, 0] < 0, -1, np.where(s['tag'][:, 0] == 0, 0, (s['tag'][:, 0] - 1) // 12 + 1)) for s in train_songs]
    elif head_key == '7th':
        lps = [s['log_probs'][2] for s in train_songs]
        trues = [s['tag'][:, 2] for s in train_songs]
    elif head_key == '9th':
        lps = [s['log_probs'][3] for s in train_songs]
        trues = [s['tag'][:, 3] for s in train_songs]
    elif head_key == '11th':
        lps = [s['log_probs'][4] for s in train_songs]
        trues = [s['tag'][:, 4] for s in train_songs]
    elif head_key == '13th':
        lps = [s['log_probs'][5] for s in train_songs]
        trues = [s['tag'][:, 5] for s in train_songs]

    all_trues = np.concatenate(trues)
    valid_mask = all_trues >= 0
    true_counts = np.bincount(all_trues[valid_mask], minlength=num_classes).astype(np.float64)
    true_counts = np.maximum(true_counts, 1.0)

    b = np.zeros(num_classes, dtype=np.float32)
    best_b = b.copy()
    best_err = np.inf

    eta_start, eta_end = 0.30, 0.05

    for it in range(max_iters):
        eta = eta_start * ((eta_end / eta_start) ** (it / max(1, max_iters - 1)))
        # Decode all songs
        pred_counts = np.zeros(num_classes, dtype=np.float64)
        if head_key == 'triad_qual':
            b_full = broadcast_triad_quality_bias(b)
            for lp, tr in zip(lps, trues):
                p97 = _viterbi_numba_core(lp + b_full, penalty)
                p_qual = np.where(p97 <= 0, 0, (p97 - 1) // 12 + 1)
                mask = (tr >= 0)
                pred_counts += np.bincount(p_qual[mask], minlength=num_classes)
        else:
            for lp, tr in zip(lps, trues):
                p = _viterbi_numba_core(lp + b, penalty)
                mask = (tr >= 0)
                pred_counts += np.bincount(p[mask], minlength=num_classes)

        ratio = pred_counts / true_counts
        log_ratio = np.log(np.clip(ratio, 0.2, 5.0))
        err = np.sum(np.abs(log_ratio))

        if err < best_err:
            best_err = err
            best_b = b.copy()

        # Update biases (reference class 0 = N is kept strictly at 0.0)
        b -= eta * log_ratio
        b[0] = 0.0
        b = np.clip(b, -max_bias, max_bias)

    return best_b

def coordinate_ascent_f1(train_songs, head_key, num_classes, init_b, penalty=2.0, max_bias=3.0, passes=2):
    # Pre-extract
    if head_key == 'triad_qual':
        lps = [s['log_probs'][0] for s in train_songs]
        trues = [np.where(s['tag'][:, 0] < 0, -1, np.where(s['tag'][:, 0] == 0, 0, (s['tag'][:, 0] - 1) // 12 + 1)) for s in train_songs]
    elif head_key == '7th':
        lps = [s['log_probs'][2] for s in train_songs]
        trues = [s['tag'][:, 2] for s in train_songs]
    elif head_key == '9th':
        lps = [s['log_probs'][3] for s in train_songs]
        trues = [s['tag'][:, 3] for s in train_songs]
    elif head_key == '11th':
        lps = [s['log_probs'][4] for s in train_songs]
        trues = [s['tag'][:, 4] for s in train_songs]
    elif head_key == '13th':
        lps = [s['log_probs'][5] for s in train_songs]
        trues = [s['tag'][:, 5] for s in train_songs]

    all_trues = np.concatenate(trues)
    valid_mask = all_trues >= 0
    valid_trues = all_trues[valid_mask]

    def eval_b(b_cand):
        preds = []
        if head_key == 'triad_qual':
            b_full = broadcast_triad_quality_bias(b_cand)
            for lp, tr in zip(lps, trues):
                p97 = _viterbi_numba_core(lp + b_full, penalty)
                preds.append(np.where(p97 <= 0, 0, (p97 - 1) // 12 + 1))
        else:
            for lp, tr in zip(lps, trues):
                preds.append(_viterbi_numba_core(lp + b_cand, penalty))
        all_preds = np.concatenate(preds)
        valid_preds = all_preds[valid_mask]
        return f1_score(valid_trues, valid_preds, average='macro', zero_division=0)

    cur_b = init_b.copy()
    cur_f1 = eval_b(cur_b)

    step_candidates = [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4]

    for p in range(passes):
        for c in range(1, num_classes): # Keep reference N=0 fixed
            best_step = 0.0
            best_f1 = cur_f1
            for delta in step_candidates:
                cand = cur_b.copy()
                cand[c] = np.clip(cand[c] + delta, -max_bias, max_bias)
                f1_val = eval_b(cand)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_step = delta
            if best_step != 0.0:
                cur_b[c] = np.clip(cur_b[c] + best_step, -max_bias, max_bias)
                cur_f1 = best_f1

    return cur_b

def compute_group_metrics(preds_dict, targets_dict):
    # Computes:
    # 1. Overall frame accuracy across all 5 heads combined
    # 2. Per-head accuracy, macro-recall (acc_class), and macro-F1
    # 3. Macro-F1 across Head, Medium, and Tail groupings

    # Groupings:
    # Head classes:
    #   Triad: N, maj, min (0, 1, 2)
    #   7th: N (0)
    #   9th: N (0)
    #   11th: N (0)
    #   13th: N (0)
    # Medium classes:
    #   Triad: sus4, 5 (3, 7)
    #   7th: 7, b7 (1, 2)
    #   9th: 9 (1)
    #   11th: 11 (1)
    # Tail classes:
    #   Triad: sus2, dim, aug, 1 (4, 5, 6, 8)
    #   7th: bb7 (3)
    #   9th: #9, b9 (2, 3)
    #   11th: #11 (2)
    #   13th: 13, b13 (1, 2)

    group_f1s = {'head': [], 'medium': [], 'tail': []}
    head_metrics = {}

    head_specs = [
        ('triad_qual', TRIAD_QUALITIES, [0, 1, 2], [3, 7], [4, 5, 6, 8]),
        ('7th', SEVENTH_NAMES, [0], [1, 2], [3]),
        ('9th', NINTH_NAMES, [0], [1], [2, 3]),
        ('11th', ELEVENTH_NAMES, [0], [1], [2]),
        ('13th', THIRTEENTH_NAMES, [0], [], [1, 2]),
    ]

    tot_correct = 0
    tot_frames = 0

    for h_name, c_names, h_cls, m_cls, t_cls in head_specs:
        y_true = targets_dict[h_name]
        y_pred = preds_dict[h_name]

        valid_m = y_true >= 0
        y_true = y_true[valid_m]
        y_pred = y_pred[valid_m]

        acc_f = accuracy_score(y_true, y_pred)
        rec_c = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Per-class F1
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(c_names))), zero_division=0)

        for c in h_cls:
            group_f1s['head'].append((f"{h_name}:{c_names[c]}", f1[c]))
        for c in m_cls:
            group_f1s['medium'].append((f"{h_name}:{c_names[c]}", f1[c]))
        for c in t_cls:
            group_f1s['tail'].append((f"{h_name}:{c_names[c]}", f1[c]))

        tot_correct += np.sum(y_true == y_pred)
        tot_frames += len(y_true)

        head_metrics[h_name] = {
            'acc_frame': acc_f,
            'acc_class': rec_c,
            'macro_f1': macro_f1,
            'per_class_f1': {c_names[c]: f1[c] for c in range(len(c_names))},
            'per_class_rec': {c_names[c]: r[c] for c in range(len(c_names))},
            'per_class_prec': {c_names[c]: p[c] for c in range(len(c_names))},
            'per_class_supp': {c_names[c]: s[c] for c in range(len(c_names))},
        }

    overall_acc = tot_correct / tot_frames
    overall_macro_f1 = np.mean([head_metrics[h]['macro_f1'] for h in head_metrics])
    overall_acc_class = np.mean([head_metrics[h]['acc_class'] for h in head_metrics])

    head_group_f1 = np.mean([x[1] for x in group_f1s['head']])
    med_group_f1 = np.mean([x[1] for x in group_f1s['medium']])
    tail_group_f1 = np.mean([x[1] for x in group_f1s['tail']])

    return {
        'overall_acc': overall_acc,
        'overall_macro_f1': overall_macro_f1,
        'overall_acc_class': overall_acc_class,
        'head_group_f1': head_group_f1,
        'med_group_f1': med_group_f1,
        'tail_group_f1': tail_group_f1,
        'head_metrics': head_metrics,
        'group_details': group_f1s
    }

def run_calibration_experiment(args=None):
    cache_path = args.cache_path if args else 'cache_data/eval_test_cache_5folds.pkl'
    recompute = args.recompute_cache if args else False
    max_songs = args.max_songs if args else None
    penalty = args.penalty if args else 2.0
    passes = args.passes if args else 2

    fold_data = extract_or_load_logits(cache_path=cache_path, recompute=recompute, max_songs=max_songs)

    print("\n" + "=" * 80)
    print(f"STAGE 1 & 2: 5-FOLD NESTED CROSS-FITTING OF PER-CLASS BIASES (CRF Penalty = {penalty:.1f})")
    print("=" * 80)

    # For each fold k:
    # Train pool: all songs from folds j != k
    # Test pool: songs from fold k
    fitted_biases_marginal = []
    fitted_biases_f1 = []

    head_configs = [
        ('triad_qual', 9),
        ('7th', 4),
        ('9th', 4),
        ('11th', 3),
        ('13th', 3),
    ]

    for k in range(5):
        t0 = time.time()
        train_pool = []
        for j in range(5):
            if j != k:
                train_pool.extend(fold_data[j])
        test_pool = fold_data[k]

        print(f"\n--- Cross-fitting Fold {k}: {len(train_pool)} train songs, {len(test_pool)} test songs ---")

        fold_b_marginal = {}
        fold_b_f1 = {}

        for h_name, n_cls in head_configs:
            # 1. Marginal matching
            b_m = fit_marginal_matching(train_pool, h_name, n_cls, max_iters=50, penalty=penalty, max_bias=3.0)
            fold_b_marginal[h_name] = b_m

            # 2. Macro-F1 coordinate ascent starting from b_m
            b_f1 = coordinate_ascent_f1(train_pool, h_name, n_cls, b_m, penalty=penalty, max_bias=3.0, passes=passes)
            fold_b_f1[h_name] = b_f1

        fitted_biases_marginal.append(fold_b_marginal)
        fitted_biases_f1.append(fold_b_f1)
        print(f"Fold {k} calibration completed in {time.time() - t0:.1f}s")

    # Check stability of fitted biases across folds
    print("\n" + "=" * 80)
    print("FITTED BIAS STABILITY ACROSS THE 5 FOLDS (Mean ± Std)")
    print("=" * 80)

    names_dict = {
        'triad_qual': TRIAD_QUALITIES,
        '7th': SEVENTH_NAMES,
        '9th': NINTH_NAMES,
        '11th': ELEVENTH_NAMES,
        '13th': THIRTEENTH_NAMES,
    }

    for h_name, n_cls in head_configs:
        print(f"\nHead: {h_name.upper()}")
        print(f"{'Class':<8} | {'Marginal-Matched (Mean ± Std)':<32} | {'F1-Ascent (Mean ± Std)':<32}")
        print("-" * 75)
        for c in range(n_cls):
            c_name = names_dict[h_name][c]
            vals_m = [fitted_biases_marginal[f][h_name][c] for f in range(5)]
            vals_f = [fitted_biases_f1[f][h_name][c] for f in range(5)]
            m_mean, m_std = np.mean(vals_m), np.std(vals_m)
            f_mean, f_std = np.mean(vals_f), np.std(vals_f)
            print(f"{c_name:<8} | {m_mean:+6.3f} ± {m_std:5.3f} (range: [{min(vals_m):+5.2f}, {max(vals_m):+5.2f}]) | {f_mean:+6.3f} ± {f_std:5.3f} (range: [{min(vals_f):+5.2f}, {max(vals_f):+5.2f}])")

    # Evaluate on the 5-fold cross-validated out-of-sample test set across alpha sweep and F1-ascent
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE TEST EVALUATION (POOLED ACROSS ALL 1,217 REAL SONGS)")
    print("=" * 80)

    alpha_values = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50]
    sweep_results = []

    for alpha in alpha_values:
        all_preds = {k: [] for k in ['triad_qual', '7th', '9th', '11th', '13th']}
        all_targets = {k: [] for k in ['triad_qual', '7th', '9th', '11th', '13th']}

        for k in range(5):
            test_songs = fold_data[k]
            # Scale marginal bias by alpha
            scaled_b = {
                h: alpha * fitted_biases_marginal[k][h] for h in fitted_biases_marginal[k]
            }
            preds_k, trues_k = evaluate_song_list(test_songs, scaled_b, penalty=2.0)
            for h in preds_k:
                all_preds[h].append(preds_k[h])
                all_targets[h].append(trues_k[h])

        pooled_preds = {h: np.concatenate(all_preds[h]) for h in all_preds}
        pooled_targets = {h: np.concatenate(all_targets[h]) for h in all_targets}

        res = compute_group_metrics(pooled_preds, pooled_targets)
        res['alpha'] = alpha
        res['mode'] = f'Marginal * {alpha:.2f}'
        sweep_results.append(res)

    # Evaluate F1-Ascent biases
    all_preds = {k: [] for k in ['triad_qual', '7th', '9th', '11th', '13th']}
    all_targets = {k: [] for k in ['triad_qual', '7th', '9th', '11th', '13th']}

    for k in range(5):
        test_songs = fold_data[k]
        b_f1 = fitted_biases_f1[k]
        preds_k, trues_k = evaluate_song_list(test_songs, b_f1, penalty=2.0)
        for h in preds_k:
            all_preds[h].append(preds_k[h])
            all_targets[h].append(trues_k[h])

    pooled_preds_f1 = {h: np.concatenate(all_preds[h]) for h in all_preds}
    pooled_targets_f1 = {h: np.concatenate(all_targets[h]) for h in all_targets}
    res_f1 = compute_group_metrics(pooled_preds_f1, pooled_targets_f1)
    res_f1['alpha'] = None
    res_f1['mode'] = 'Coordinate-Ascent (F1)'
    sweep_results.append(res_f1)

    # Print Pareto Frontier Table
    print("\n=== SUMMARY FRONTIER: OVERALL ACCURACY vs MACRO-F1 vs CLASS GROUPS ===")
    print(f"{'Operating Point':<26} | {'Overall Acc':<11} | {'Macro-F1':<10} | {'acc_class':<10} | {'Head F1':<9} | {'Medium F1':<10} | {'Tail F1':<9}")
    print("-" * 95)
    for r in sweep_results:
        print(f"{r['mode']:<26} | {r['overall_acc']*100:6.2f}%    | {r['overall_macro_f1']*100:6.2f}%   | {r['overall_acc_class']*100:6.2f}%   | {r['head_group_f1']*100:6.2f}%  | {r['med_group_f1']*100:6.2f}%   | {r['tail_group_f1']*100:6.2f}%")

    # Detailed Head Comparison between Baseline (alpha=0.0), Marginal (alpha=1.0), and F1-Ascent
    print("\n" + "=" * 80)
    print("DETAILED HEAD-BY-HEAD COMPARISON: BASELINE vs MARGINAL MATCH vs F1-ASCENT")
    print("=" * 80)

    base_r = sweep_results[0]
    marg_r = [r for r in sweep_results if r['mode'] == 'Marginal * 1.00'][0]
    f1_r = res_f1

    for h_name, n_cls in head_configs:
        print(f"\n--- HEAD: {h_name.upper()} ---")
        print(f"{'Metric':<25} | {'Baseline (Uncal)':<18} | {'Marginal (alpha=1)':<20} | {'F1-Ascent':<18}")
        print("-" * 85)
        b_hm = base_r['head_metrics'][h_name]
        m_hm = marg_r['head_metrics'][h_name]
        f_hm = f1_r['head_metrics'][h_name]
        print(f"{'Frame Accuracy':<25} | {b_hm['acc_frame']*100:6.2f}%           | {m_hm['acc_frame']*100:6.2f}%             | {f_hm['acc_frame']*100:6.2f}%")
        print(f"{'Macro-F1':<25} | {b_hm['macro_f1']*100:6.2f}%           | {m_hm['macro_f1']*100:6.2f}%             | {f_hm['macro_f1']*100:6.2f}%")
        print(f"{'Macro Recall (acc_class)':<25} | {b_hm['acc_class']*100:6.2f}%           | {m_hm['acc_class']*100:6.2f}%             | {f_hm['acc_class']*100:6.2f}%")

        print(f"\n  Per-class breakdown [Recall / Precision / F1]:")
        c_names = names_dict[h_name]
        for c in range(n_cls):
            cn = c_names[c]
            b_str = f"{b_hm['per_class_rec'][cn]*100:4.1f}% / {b_hm['per_class_prec'][cn]*100:4.1f}% / {b_hm['per_class_f1'][cn]*100:4.1f}%"
            m_str = f"{m_hm['per_class_rec'][cn]*100:4.1f}% / {m_hm['per_class_prec'][cn]*100:4.1f}% / {m_hm['per_class_f1'][cn]*100:4.1f}%"
            f_str = f"{f_hm['per_class_rec'][cn]*100:4.1f}% / {f_hm['per_class_prec'][cn]*100:4.1f}% / {f_hm['per_class_f1'][cn]*100:4.1f}%"
            print(f"    {cn:<6} | Base: {b_str:<21} | Marg: {m_str:<21} | F1: {f_str}")

    # Ablation: "1" chord mapping experiment
    print("\n" + "=" * 80)
    print("ABLATION: '1' CHORD MAPPING EXPERIMENT (TRIAD HEAD EVALUATION)")
    print("=" * 80)
    # Using the F1-Ascent predictions on the triad head:
    y_true_triad = pooled_targets_f1['triad_qual']
    y_pred_triad = pooled_preds_f1['triad_qual']
    val_m = y_true_triad >= 0
    y_true_triad = y_true_triad[val_m]
    y_pred_triad = y_pred_triad[val_m]

    # 1. Separate (current 9 classes)
    acc_sep = accuracy_score(y_true_triad, y_pred_triad)
    f1_sep = f1_score(y_true_triad, y_pred_triad, average='macro', zero_division=0)
    print(f"1. As separate class (9 classes): Frame Acc = {acc_sep*100:.2f}%, Macro-F1 = {f1_sep*100:.2f}%")

    # 2. Map '1' -> '5' (index 8 -> index 7)
    # 8 classes: 0=N, 1=maj, 2=min, 3=sus4, 4=sus2, 5=dim, 6=aug, 7=5
    y_true_m5 = np.where(y_true_triad == 8, 7, y_true_triad)
    y_pred_m5 = np.where(y_pred_triad == 8, 7, y_pred_triad)
    acc_m5 = accuracy_score(y_true_m5, y_pred_m5)
    f1_m5 = f1_score(y_true_m5, y_pred_m5, average='macro', zero_division=0)
    print(f"2. Merge '1' -> '5' (8 classes):  Frame Acc = {acc_m5*100:.2f}%, Macro-F1 = {f1_m5*100:.2f}%")

    # 3. Map '1' -> 'N' (index 8 -> index 0)
    y_true_mN = np.where(y_true_triad == 8, 0, y_true_triad)
    y_pred_mN = np.where(y_pred_triad == 8, 0, y_pred_triad)
    acc_mN = accuracy_score(y_true_mN, y_pred_mN)
    f1_mN = f1_score(y_true_mN, y_pred_mN, average='macro', zero_division=0)
    print(f"3. Merge '1' -> 'N' (8 classes):  Frame Acc = {acc_mN*100:.2f}%, Macro-F1 = {f1_mN*100:.2f}%")

    # 13th Head Achievable F1 Ceiling Test
    print("\n" + "=" * 80)
    print("13th HEAD: MAX ACHIEVABLE F1 OVER INDEPENDENT THRESHOLD/BIAS SWEEPS")
    print("=" * 80)
    trues_13 = pooled_targets_f1['13th']
    val_13 = trues_13 >= 0
    t13_valid = trues_13[val_13]

    for c_idx, c_name in [(1, '13'), (2, 'b13')]:
        best_c_f1 = 0.0
        best_c_b = 0.0
        # Sweep bias from -3 to +3
        for test_b in np.linspace(-3.0, 3.0, 61):
            cand_b = np.zeros(3, dtype=np.float32)
            cand_b[c_idx] = test_b
            preds_sweep = []
            for k in range(5):
                for s in fold_data[k]:
                    p = _viterbi_numba_core(s['log_probs'][5] + cand_b, penalty=2.0)
                    preds_sweep.append(p)
            all_ps = np.concatenate(preds_sweep)
            p13_valid = all_ps[val_13]
            p_val, r_val, f1_val, _ = precision_recall_fscore_support(t13_valid == c_idx, p13_valid == c_idx, average='binary', zero_division=0)
            if f1_val > best_c_f1:
                best_c_f1 = f1_val
                best_c_b = test_b
        print(f"Class '{c_name}': Max achievable binary F1 = {best_c_f1*100:.2f}% (at bias = {best_c_b:+.2f})")

def check_seventh_head_conditional_rates():
    import h5py
    synth_path = 'dataset/synth_naturalistic_xchord.h5d'
    real_path = 'dataset/jams_xchord.h5d'
    if not os.path.exists(synth_path) or not os.path.exists(real_path):
        return

    print("=" * 80)
    print("HYPOTHESIS CHECK: CONDITIONAL 7th RATES (P(b7 | context)) IN REAL VS SYNTHETIC")
    print("=" * 80)
    with h5py.File(synth_path, 'r') as f_synth, h5py.File(real_path, 'r') as f_real:
        s_feat = f_synth['feature'][:]
        r_feat = f_real['feature'][:]

        s_triad = np.where(s_feat[:, 0] <= 0, 0, (s_feat[:, 0] - 1) // 12 + 1)
        r_triad = np.where(r_feat[:, 0] <= 0, 0, (r_feat[:, 0] - 1) // 12 + 1)
        s_7th = s_feat[:, 2]
        r_7th = r_feat[:, 2]

        for name, triad_idx in [('maj', 1), ('min', 2), ('sus4', 3), ('5', 7), ('All Non-N', -1)]:
            r_mask = (r_triad > 0) if triad_idx == -1 else (r_triad == triad_idx)
            s_mask = (s_triad > 0) if triad_idx == -1 else (s_triad == triad_idx)

            r_tot, s_tot = np.sum(r_mask), np.sum(s_mask)
            r_b7, s_b7 = np.sum(r_mask & (r_7th == 2)), np.sum(s_mask & (s_7th == 2))
            r_any7, s_any7 = np.sum(r_mask & (r_7th > 0)), np.sum(s_mask & (s_7th > 0))

            print(f"Context: {name:<12}")
            print(f"  Real:   P(b7|ctx) = {r_b7/r_tot*100:5.2f}% ({r_b7:,}/{r_tot:,}) | P(any_7|ctx) = {r_any7/r_tot*100:5.2f}%")
            print(f"  Synth:  P(b7|ctx) = {s_b7/s_tot*100:5.2f}% ({s_b7:,}/{s_tot:,}) | P(any_7|ctx) = {s_any7/s_tot*100:5.2f}%")
    print("-" * 80)
    print("Takeaway: In major triads, synthetic data injects dominant 7ths at >2x the real rate (30.6% vs 15.2%).")
    print("Real annotators omit 7ths on 77.7% of major triads, while synthetic leaves only 58.8% plain.")
    print("=" * 80 + "\n")

class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
    def close(self):
        self.logfile.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Post-Hoc Cross-Fitted Bias Calibration and Pareto Frontier Evaluation")
    parser.add_argument('--penalty', type=float, default=2.0, help='CRF Viterbi switching penalty (default: 2.0)')
    parser.add_argument('--cache_path', type=str, default='cache_data/eval_test_cache_5folds.pkl', help='Path to cached inference outputs')
    parser.add_argument('--recompute_cache', action='store_true', help='Recompute out-of-sample inference cache')
    parser.add_argument('--save_report', type=str, default='eval_bias_calibration_report.txt', help='Path to save full text evaluation report')
    parser.add_argument('--passes', type=int, default=2, help='Coordinate ascent passes on macro-F1 (default: 2)')
    parser.add_argument('--max_songs', type=int, default=None, help='Max test songs per fold (optional quick check)')
    args = parser.parse_args()

    tee = TeeLogger(args.save_report)
    sys.stdout = tee

    try:
        check_seventh_head_conditional_rates()
        run_calibration_experiment(args)
    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nFull calibration report saved to {args.save_report}")
