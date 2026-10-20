import torch
import numpy as np
from sklearn.metrics import classification_report, recall_score, accuracy_score, confusion_matrix
import pandas as pd
import os

from confor_head16 import ChordNet, chord_limit
from mir.nn.train import NetworkInterface
from mir.nn.data_storage import FramedH5DataStorage
from train_eval_test_split import get_test_set_ids
from complex_chord import Chord, TriadTypes, SeventhTypes, NinthTypes, EleventhTypes, ThirteenthTypes
from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH

# Human-readable labels for heads
TRIAD_NAMES = ['N', 'maj', 'min', 'sus4', 'sus2', 'dim', 'aug', '5', '1']
PITCH_NAMES = ['N', 'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
SEVENTH_NAMES = ['N', '7', 'b7', 'bb7']
NINTH_NAMES = ['N', '9', '#9', 'b9']
ELEVENTH_NAMES = ['N', '11', '#11']
THIRTEENTH_NAMES = ['N', '13', 'b13']
BASS_NAMES = PITCH_NAMES

FULL_97_NAMES = ['N']
for triad in TRIAD_NAMES[1:]:
    for pitch in PITCH_NAMES[1:]:
        FULL_97_NAMES.append(f"{pitch}:{triad}")

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 120)

try:
    import numba
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
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def build_crf_transition_matrix(num_classes: int, penalty: float, device: torch.device = None) -> torch.Tensor:
    """
    Builds a fixed transition log-probability matrix based on Eq. 12 from the ChordFormer paper.
    log_trans[i, j] = 0 if i == j else -penalty
    """
    dev = device or torch.device('cpu')
    trans_log = torch.full((num_classes, num_classes), -penalty, dtype=torch.float32, device=dev)
    trans_log.fill_diagonal_(0.0)
    return trans_log

def viterbi_decode_crf(probs: np.ndarray, penalty: float = 2.0, device: torch.device = None) -> np.ndarray:
    """
    Fast Viterbi decoding with transition penalty matching ChordFormer Eq. 12.
    Uses O(K) reduction (max(score[c], max(score) - penalty)) with Numba acceleration (~3ms/song)
    or vectorized NumPy fallback (~150ms/song) rather than slow sequential GPU kernel dispatches.
    """
    if penalty is None or penalty <= 0.0:
        return probs.argmax(axis=-1)

    seq_len, num_classes = probs.shape
    if seq_len <= 1:
        return probs.argmax(axis=-1)

    log_probs = np.log(np.maximum(probs, 1e-12)).astype(np.float32)

    if HAS_NUMBA:
        return _viterbi_numba_core(log_probs, float(penalty))

    score = log_probs[0].copy()
    backpointers = np.empty((seq_len - 1, num_classes), dtype=np.int32)
    classes_idx = np.arange(num_classes, dtype=np.int32)

    for t in range(1, seq_len):
        best_idx = np.argmax(score)
        threshold = score[best_idx] - penalty
        stay = score >= threshold
        backpointers[t - 1] = np.where(stay, classes_idx, best_idx)
        score = np.where(stay, score, threshold) + log_probs[t]

    decoded = np.empty(seq_len, dtype=np.int32)
    last_state = np.argmax(score)
    decoded[seq_len - 1] = last_state
    for t in range(seq_len - 2, -1, -1):
        last_state = backpointers[t, last_state]
        decoded[t] = last_state

    return decoded

def compute_class_log_priors(weight_pkl_path='data/cross_subpart_weight0.pkl'):
    """
    Computes empirical class log-priors log(pi_k) from training data distributions.
    Used for Post-Hoc Logit Adjustment (Menon et al., ICLR 2021) at inference time:
    z'_k = z_k + tau * log(pi_k), which penalizes false positives on rare extensions
    without degrading the learned feature representations.
    """
    import pickle
    if not os.path.exists(weight_pkl_path):
        print(f"Warning: {weight_pkl_path} not found. Logit adjustment disabled.")
        return None
    with open(weight_pkl_path, 'rb') as f:
        w = pickle.load(f)

    # Head 0: 97 classes (1 N + 8 * 12 triad types)
    head0_counts = np.zeros(97, dtype=np.float64)
    head0_counts[0] = w[0][0]
    for t in range(1, 9):
        head0_counts[(t - 1) * 12 + 1 : t * 12 + 1] = w[0][t] / 12.0
    p0 = head0_counts / head0_counts.sum()

    # Head 1: 13 classes (0=N, 1..12 = C..B)
    head1_counts = np.zeros(13, dtype=np.float64)
    head1_counts[0] = w[1][0]
    head1_counts[1:] = w[1][1] / 12.0
    p1 = head1_counts / head1_counts.sum()

    # Head 2..5: direct extension counts
    p2 = np.array(w[2], dtype=np.float64) / sum(w[2])
    p3 = np.array(w[3], dtype=np.float64) / sum(w[3])
    p4 = np.array(w[4], dtype=np.float64) / sum(w[4])
    p5 = np.array(w[5], dtype=np.float64) / sum(w[5])

    return [np.log(np.maximum(p, 1e-12)) for p in [p0, p1, p2, p3, p4, p5]]

def format_confusion_matrix(cm, labels, title):
    lines = []
    lines.append("=" * 80)
    lines.append(f"CONFUSION MATRIX: {title}")
    lines.append("=" * 80)

    # Raw frame counts
    lines.append("\n[RAW FRAME COUNTS] (Rows = Ground Truth, Columns = Predicted):")
    df_raw = pd.DataFrame(cm, index=labels, columns=labels)
    lines.append(df_raw.to_string())

    # Row-normalized recall percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.where(row_sums > 0, cm / row_sums * 100.0, 0.0)
    df_norm = pd.DataFrame(np.round(cm_norm, 2), index=labels, columns=labels)
    lines.append("\n[ROW-NORMALIZED RECALL (%)] (Rows = Ground Truth, Columns = Predicted):")
    lines.append(df_norm.to_string())

    # Top confusion patterns per class
    lines.append("\n[TOP CONFUSION PATTERNS PER CLASS]:")
    for i, label in enumerate(labels):
        support = int(row_sums[i, 0])
        if support == 0:
            lines.append(f"  * {label:5s} (Support: 0 frames)")
            continue
        diag = cm[i, i]
        rec = diag / support * 100.0
        other_indices = [j for j in range(len(labels)) if j != i and cm[i, j] > 0]
        other_indices.sort(key=lambda j: cm[i, j], reverse=True)
        if other_indices:
            misclass_str = ", ".join([f"{labels[j]}: {cm[i, j]:,d} ({cm[i, j]/support*100:.2f}%)" for j in other_indices[:4]])
        else:
            misclass_str = "None (100% correct)"
        lines.append(f"  * {label:5s} (Support: {support:10,d} | Recall: {rec:5.2f}%): Confused with -> {misclass_str}")

    return "\n".join(lines), df_raw, df_norm

def format_top_97_confusions(cm, labels, top_n=25):
    lines = []
    lines.append("=" * 80)
    lines.append(f"TOP {top_n} OFF-DIAGONAL CONFUSIONS ACROSS ALL 97 ROOT/TRIAD CLASSES")
    lines.append("=" * 80)
    row_sums = cm.sum(axis=1, keepdims=True)
    confusions = []
    num_classes = len(labels)
    for i in range(num_classes):
        support = int(row_sums[i, 0])
        if support == 0:
            continue
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                count = cm[i, j]
                pct = count / support * 100.0
                confusions.append((count, pct, labels[i], labels[j], support))

    confusions.sort(key=lambda x: x[0], reverse=True)
    lines.append(f"{'True Label':<12} -> {'Pred Label':<12} | {'Count':<10} | {'% of True':<10} | {'True Support':<12}")
    lines.append("-" * 65)
    for count, pct, true_lbl, pred_lbl, support in confusions[:top_n]:
        lines.append(f"{true_lbl:<12} -> {pred_lbl:<12} | {count:<10,d} | {pct:<9.2f}% | {support:<12,d}")
    return "\n".join(lines)

def evaluate_heads(model_pattern='chordformer_head16(1.0,1.0)_s%d.best', max_songs=None, crf_penalty=2.0, tau=0.35, save_path=None, cm_dir='confusion_matrices', save_cm_csv=True):
    if save_path is None:
        tau_str = f"_tau{tau:.2f}" if (tau and tau > 0) else ""
        save_path = f'eval_heads_crf_p{crf_penalty:.1f}{tau_str}_report.txt' if (crf_penalty and crf_penalty > 0) else f'eval_heads{tau_str}_report.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_str = f"CRF Viterbi Decoding (penalty={crf_penalty})" if (crf_penalty and crf_penalty > 0) else "Raw ArgMax (No CRF)"
    if tau and tau > 0:
        mode_str += f" + Logit Adj (tau={tau:.2f})"
    print(f"Evaluating 6 heads on {device} using {mode_str} with model pattern '{model_pattern}' across all 5 cross-validation folds...")

    log_priors = compute_class_log_priors() if (tau and tau > 0) else None

    storage_x = FramedH5DataStorage('jams_cqt')
    storage_y = FramedH5DataStorage('jams_xchord')
    storage_x.load_meta()
    storage_y.load_meta()
    storage_x.load()
    storage_y.load()

    # Load 5 fold models
    models = []
    for f in range(5):
        net_name = model_pattern % f
        print(f"Loading checkpoint {net_name}...")
        model = NetworkInterface(ChordNet(None, triad_only=False), net_name, load_checkpoint=True)
        models.append(model)

    # Buckets for all 6 heads: targets and predictions
    all_targets = [[] for _ in range(6)]
    all_preds = [[] for _ in range(6)]
    # Also separate Triad quality (8 classes) independent of Root
    all_triad_qual_targets = []
    all_triad_qual_preds = []

    total_evaluated_songs = 0

    for fold in range(5):
        test_ids = get_test_set_ids(fold)
        if max_songs is not None and total_evaluated_songs >= max_songs:
            break
        print(f"\nEvaluating Fold {fold} ({len(test_ids)} test songs)...")
        model = models[fold]

        for song_id in test_ids:
            if max_songs is not None and total_evaluated_songs >= max_songs:
                break
            song_len = storage_x.length[song_id]
            cqt = storage_x.locate(song_id, 0, song_len)
            tag = storage_y.locate(song_id, 0, song_len)

            # Inference
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                probs = model.inference(cqt)

            # Apply Post-Hoc Logit Adjustment (Menon et al.) if tau > 0
            if log_priors is not None and tau > 0.0:
                adjusted_probs = []
                for h in range(6):
                    adj_log_p = np.log(np.maximum(probs[h], 1e-12)) + tau * log_priors[h]
                    adj_p = np.exp(adj_log_p - adj_log_p.max(axis=-1, keepdims=True))
                    adj_p = adj_p / adj_p.sum(axis=-1, keepdims=True)
                    adjusted_probs.append(adj_p)
            else:
                adjusted_probs = probs

            # Decode with CRF if requested, else argmax
            if crf_penalty is not None and crf_penalty > 0:
                pred_0 = viterbi_decode_crf(adjusted_probs[0], penalty=crf_penalty, device=device)
                pred_1 = viterbi_decode_crf(adjusted_probs[1], penalty=crf_penalty, device=device)
                pred_2 = viterbi_decode_crf(adjusted_probs[2], penalty=crf_penalty, device=device)
                pred_3 = viterbi_decode_crf(adjusted_probs[3], penalty=crf_penalty, device=device)
                pred_4 = viterbi_decode_crf(adjusted_probs[4], penalty=crf_penalty, device=device)
                pred_5 = viterbi_decode_crf(adjusted_probs[5], penalty=crf_penalty, device=device)
            else:
                pred_0 = adjusted_probs[0].argmax(axis=-1)
                pred_1 = adjusted_probs[1].argmax(axis=-1)
                pred_2 = adjusted_probs[2].argmax(axis=-1)
                pred_3 = adjusted_probs[3].argmax(axis=-1)
                pred_4 = adjusted_probs[4].argmax(axis=-1)
                pred_5 = adjusted_probs[5].argmax(axis=-1)

            target_0 = tag[:, 0]
            target_1 = tag[:, 1] + 1
            target_2 = tag[:, 2]
            target_3 = tag[:, 3]
            target_4 = tag[:, 4]
            target_5 = tag[:, 5]

            # Extract triad quality alone: 0=N, 1=maj, 2=min, 3=sus4, 4=sus2, 5=dim, 6=aug, 7=5, 8=1
            t_qual = np.where(target_0 <= 0, 0, (target_0 - 1) // 12 + 1)
            p_qual = np.where(pred_0 <= 0, 0, (pred_0 - 1) // 12 + 1)
            all_triad_qual_targets.extend(t_qual)
            all_triad_qual_preds.extend(p_qual)

            preds = [pred_0, pred_1, pred_2, pred_3, pred_4, pred_5]
            targets = [target_0, target_1, target_2, target_3, target_4, target_5]

            for h in range(6):
                all_targets[h].extend(targets[h])
                all_preds[h].extend(preds[h])

            total_evaluated_songs += 1
            if total_evaluated_songs % 100 == 0:
                print(f"Processed {total_evaluated_songs} songs...")

    print(f"\nFinished inference on {total_evaluated_songs} total songs.")

    title = f"CHORDFORMER MULTI-HEAD EVALUATION ({mode_str}, {total_evaluated_songs} songs)"
    format_report(all_targets, all_preds, all_triad_qual_targets, all_triad_qual_preds, total_evaluated_songs, title, save_path, cm_dir=cm_dir, save_cm_csv=save_cm_csv)

def format_report(all_targets, all_preds, all_triad_qual_targets, all_triad_qual_preds, total_evaluated_songs, title, save_path=None, cm_dir='confusion_matrices', save_cm_csv=True):
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(title)
    report_lines.append("=" * 80)

    if save_cm_csv and cm_dir:
        os.makedirs(cm_dir, exist_ok=True)

    # 1. Specialized Triad Quality Report (8 classes: N, maj, min, sus4, sus2, dim, aug, 5, 1)
    targets_np = np.array(all_triad_qual_targets)
    preds_np = np.array(all_triad_qual_preds)
    valid_mask = targets_np >= 0
    t_valid = targets_np[valid_mask]
    p_valid = preds_np[valid_mask]

    acc_f = accuracy_score(t_valid, p_valid)
    acc_c = recall_score(t_valid, p_valid, average='macro', zero_division=0)
    report_lines.append("\n=== Triad Quality (Isolated: maj, min, sus4, sus2, dim, aug, 5, 1) ===")
    report_lines.append(f"Frame-wise Accuracy (acc_frame): {acc_f:.4f}")
    report_lines.append(f"Class-wise Accuracy (acc_class): {acc_c:.4f}")
    report_lines.append("Detailed Report per Class:")
    unique_labels = sorted(list(set(t_valid) | set(p_valid)))
    names = [TRIAD_NAMES[idx] if idx < len(TRIAD_NAMES) else f"Triad_{idx}" for idx in unique_labels]
    report_lines.append(classification_report(t_valid, p_valid, labels=unique_labels, target_names=names, zero_division=0, digits=4))

    # Confusion matrix for Triad Quality
    cm_triad = confusion_matrix(t_valid, p_valid, labels=list(range(len(TRIAD_NAMES))))
    cm_text, df_triad_raw, df_triad_norm = format_confusion_matrix(cm_triad, TRIAD_NAMES, "Triad Quality (9 Classes)")
    report_lines.append(cm_text)
    if save_cm_csv and cm_dir:
        df_triad_raw.to_csv(os.path.join(cm_dir, "cm_triad_quality_raw.csv"))
        df_triad_norm.to_csv(os.path.join(cm_dir, "cm_triad_quality_norm.csv"))
    report_lines.append("-" * 80)

    # 2. Root Pitch Class (Isolated: C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B)
    targets_0 = np.array(all_targets[0])
    preds_0 = np.array(all_preds[0])
    t_root = np.where(targets_0 <= 0, 0, (targets_0 - 1) % 12 + 1)
    p_root = np.where(preds_0 <= 0, 0, (preds_0 - 1) % 12 + 1)
    v_root = targets_0 >= 0
    t_root_valid = t_root[v_root]
    p_root_valid = p_root[v_root]

    acc_f_root = accuracy_score(t_root_valid, p_root_valid)
    acc_c_root = recall_score(t_root_valid, p_root_valid, average='macro', zero_division=0)
    report_lines.append("\n=== Root Pitch Class (Isolated: C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B) ===")
    report_lines.append(f"Frame-wise Accuracy (acc_frame): {acc_f_root:.4f}")
    report_lines.append(f"Class-wise Accuracy (acc_class): {acc_c_root:.4f}")
    report_lines.append("Detailed Report per Class:")
    u_root = sorted(list(set(t_root_valid) | set(p_root_valid)))
    names_root = [PITCH_NAMES[idx] if idx < len(PITCH_NAMES) else f"Pitch_{idx}" for idx in u_root]
    report_lines.append(classification_report(t_root_valid, p_root_valid, labels=u_root, target_names=names_root, zero_division=0, digits=4))

    # Confusion matrix for Root Pitch Class
    cm_root = confusion_matrix(t_root_valid, p_root_valid, labels=list(range(len(PITCH_NAMES))))
    cm_text, df_root_raw, df_root_norm = format_confusion_matrix(cm_root, PITCH_NAMES, "Root Pitch Class (13 Classes)")
    report_lines.append(cm_text)
    if save_cm_csv and cm_dir:
        df_root_raw.to_csv(os.path.join(cm_dir, "cm_root_pitch_raw.csv"))
        df_root_norm.to_csv(os.path.join(cm_dir, "cm_root_pitch_norm.csv"))
    report_lines.append("-" * 80)

    # 3. Six Heads
    head_names = ["Full Root/Triad (97 classes)", "Bass (13 classes)", "7th Extension", "9th Extension", "11th Extension", "13th Extension"]
    head_label_maps = [
        FULL_97_NAMES,
        BASS_NAMES,
        SEVENTH_NAMES,
        NINTH_NAMES,
        ELEVENTH_NAMES,
        THIRTEENTH_NAMES
    ]
    head_csv_names = [
        "cm_head1_full97",
        "cm_head2_bass",
        "cm_head3_7th",
        "cm_head4_9th",
        "cm_head5_11th",
        "cm_head6_13th"
    ]

    for h in range(6):
        targets_np = np.array(all_targets[h])
        preds_np = np.array(all_preds[h])

        valid_mask = targets_np >= 0
        t_valid = targets_np[valid_mask]
        p_valid = preds_np[valid_mask]

        acc_f = accuracy_score(t_valid, p_valid)
        acc_c = recall_score(t_valid, p_valid, average='macro', zero_division=0)

        report_lines.append(f"\n=== Head {h+1}: {head_names[h]} ===")
        report_lines.append(f"Frame-wise Accuracy (acc_frame): {acc_f:.4f}")
        report_lines.append(f"Class-wise Accuracy (acc_class): {acc_c:.4f}")
        report_lines.append("Detailed Report per Class:")

        unique_labels = sorted(list(set(t_valid) | set(p_valid)))
        lbl_map = head_label_maps[h]
        if lbl_map is not None:
            names = [lbl_map[idx] if idx < len(lbl_map) else f"Class_{idx}" for idx in unique_labels]
            report_lines.append(classification_report(t_valid, p_valid, labels=unique_labels, target_names=names, zero_division=0, digits=4))
        else:
            report_lines.append(classification_report(t_valid, p_valid, labels=unique_labels, zero_division=0, digits=4))

        # Confusion Matrix
        if h == 0:
            # For Head 1 (97 classes): compute 97x97 confusion matrix
            cm_97 = confusion_matrix(t_valid, p_valid, labels=list(range(97)))
            df_97_raw = pd.DataFrame(cm_97, index=FULL_97_NAMES, columns=FULL_97_NAMES)
            row_sums = cm_97.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_97_norm = np.where(row_sums > 0, cm_97 / row_sums * 100.0, 0.0)
            df_97_norm = pd.DataFrame(np.round(cm_97_norm, 2), index=FULL_97_NAMES, columns=FULL_97_NAMES)

            top_97_text = format_top_97_confusions(cm_97, FULL_97_NAMES, top_n=25)
            report_lines.append(top_97_text)
            if save_cm_csv and cm_dir:
                df_97_raw.to_csv(os.path.join(cm_dir, f"{head_csv_names[h]}_raw.csv"))
                df_97_norm.to_csv(os.path.join(cm_dir, f"{head_csv_names[h]}_norm.csv"))
                report_lines.append(f"\nFull 97x97 confusion matrices saved to {cm_dir}/{head_csv_names[h]}_raw.csv and _norm.csv")
        else:
            labels_h = head_label_maps[h]
            cm_h = confusion_matrix(t_valid, p_valid, labels=list(range(len(labels_h))))
            cm_text, df_h_raw, df_h_norm = format_confusion_matrix(cm_h, labels_h, f"Head {h+1}: {head_names[h]}")
            report_lines.append(cm_text)
            if save_cm_csv and cm_dir:
                df_h_raw.to_csv(os.path.join(cm_dir, f"{head_csv_names[h]}_raw.csv"))
                df_h_norm.to_csv(os.path.join(cm_dir, f"{head_csv_names[h]}_norm.csv"))

        report_lines.append("-" * 80)

    full_report = "\n".join(report_lines)
    print(full_report)

    if save_path:
        with open(save_path, 'w') as f_out:
            f_out.write(full_report + "\n")
        print(f"\nSaved full multi-head evaluation report with confusion matrices to {save_path}")
    return full_report

def evaluate_heads_from_lab(lab_dir='output/output_chordformer_head16(1.0,1.0)_s%d.best_hmm_full/jam/', max_songs=None, save_path='eval_heads_from_lab_report.txt', cm_dir='confusion_matrices', save_cm_csv=True):
    print(f"Evaluating 6 heads directly from generated .lab files in {lab_dir}...")
    storage_y = FramedH5DataStorage('jams_xchord')
    storage_y.load_meta()
    storage_y.load()

    with open('data/all_1217.csv', 'r') as f:
        all_names = [line.strip() for line in f.readlines()]

    fps = DEFAULT_SR / DEFAULT_HOP_LENGTH

    all_targets = [[] for _ in range(6)]
    all_preds = [[] for _ in range(6)]
    all_triad_qual_targets = []
    all_triad_qual_preds = []

    total_evaluated_songs = 0

    for fold in range(5):
        test_ids = get_test_set_ids(fold)
        if os.path.exists(lab_dir):
            current_lab_dir = lab_dir
        elif '%d' in lab_dir and os.path.exists(lab_dir % fold):
            current_lab_dir = lab_dir % fold
        else:
            current_lab_dir = lab_dir
        print(f"\nProcessing Fold {fold} ({len(test_ids)} test songs) from {current_lab_dir}...")

        for song_id in test_ids:
            if max_songs is not None and total_evaluated_songs >= max_songs:
                break
            song_name = all_names[song_id]
            lab_path = os.path.join(current_lab_dir, song_name + '.lab')
            if not os.path.exists(lab_path):
                print(f"Warning: missing {lab_path}")
                continue

            song_len = storage_y.length[song_id]
            tag = storage_y.locate(song_id, 0, song_len)
            n_frames = tag.shape[0]

            pred_heads = np.zeros((n_frames, 6), dtype=np.int64)
            with open(lab_path, 'r') as f_lab:
                for line in f_lab:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    s, e, chord_name = float(parts[0]), float(parts[1]), parts[2]
                    s_f = max(0, int(round(s * fps)))
                    e_f = min(n_frames, int(round(e * fps)))
                    arr = Chord(chord_name).to_numpy()
                    pred_heads[s_f:e_f] = arr

            pred_0 = pred_heads[:, 0]
            target_0 = tag[:, 0]

            pred_1 = np.maximum(0, pred_heads[:, 1] + 1)
            target_1 = tag[:, 1] + 1

            pred_2 = np.maximum(0, pred_heads[:, 2])
            target_2 = tag[:, 2]

            pred_3 = np.maximum(0, pred_heads[:, 3])
            target_3 = tag[:, 3]

            pred_4 = np.maximum(0, pred_heads[:, 4])
            target_4 = tag[:, 4]

            pred_5 = np.maximum(0, pred_heads[:, 5])
            target_5 = tag[:, 5]

            t_qual = np.where(target_0 <= 0, 0, (target_0 - 1) // 12 + 1)
            p_qual = np.where(pred_0 <= 0, 0, (pred_0 - 1) // 12 + 1)
            all_triad_qual_targets.extend(t_qual)
            all_triad_qual_preds.extend(p_qual)

            preds = [pred_0, pred_1, pred_2, pred_3, pred_4, pred_5]
            targets = [target_0, target_1, target_2, target_3, target_4, target_5]

            for h in range(6):
                all_targets[h].extend(targets[h])
                all_preds[h].extend(preds[h])

            total_evaluated_songs += 1

    format_report(all_targets, all_preds, all_triad_qual_targets, all_triad_qual_preds, total_evaluated_songs,
                  f"CHORDFORMER MULTI-HEAD EVALUATION (From Composite HMM .lab, {total_evaluated_songs} songs)", save_path, cm_dir=cm_dir, save_cm_csv=save_cm_csv)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Multi-head chord evaluation with optional CRF Viterbi decoding and Confusion Matrices")
    parser.add_argument('--from_lab', action='store_true', help='Evaluate directly from generated .lab files without re-running network inference')
    parser.add_argument('--lab_dir', type=str, default='output/output_chordformer_head16(1.0,1.0)_s%d.best_hmm_full/jam/', help='Directory containing .lab files')
    parser.add_argument('--model_pattern', type=str, default='chordformer_head16(1.0,1.0)_s%d.best', help='Model checkpoint pattern (e.g. chordformer_head16_synth_ft_s%%d.best)')
    parser.add_argument('--max_songs', type=int, default=None, help='Limit number of test songs (default: all 1217)')
    parser.add_argument('--penalty', '--crf_penalty', type=float, default=2.0, dest='penalty', help='CRF transition penalty (default 2.0 based on Eq. 12)')
    parser.add_argument('--no_crf', action='store_true', help='Disable CRF decoding and use raw argmax')
    parser.add_argument('--tau', type=float, default=0.0, help='Logit adjustment temperature tau (default 0.0, set >0 to enable)')
    parser.add_argument('--no_logit_adj', action='store_true', help='Disable logit adjustment prior correction (equivalent to --tau 0.0)')
    parser.add_argument('--save_path', type=str, default=None, help='Custom output text file path')
    parser.add_argument('--cm_dir', type=str, default='confusion_matrices', help='Directory to save CSV confusion matrices (default: confusion_matrices)')
    parser.add_argument('--no_save_cm_csv', action='store_true', help='Disable saving CSV confusion matrices')
    args = parser.parse_args()

    save_cm_csv = not args.no_save_cm_csv

    if args.from_lab:
        save_path = args.save_path or 'eval_heads_from_lab_report.txt'
        evaluate_heads_from_lab(lab_dir=args.lab_dir, max_songs=args.max_songs, save_path=save_path, cm_dir=args.cm_dir, save_cm_csv=save_cm_csv)
    else:
        penalty = 0.0 if args.no_crf else args.penalty
        tau = 0.0 if args.no_logit_adj else args.tau
        evaluate_heads(model_pattern=args.model_pattern, max_songs=args.max_songs, crf_penalty=penalty, tau=tau, save_path=args.save_path, cm_dir=args.cm_dir, save_cm_csv=save_cm_csv)

