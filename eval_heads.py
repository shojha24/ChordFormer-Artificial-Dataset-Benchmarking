import torch
import numpy as np
from sklearn.metrics import classification_report, recall_score, accuracy_score
import os

from confor_head16 import ChordNet, chord_limit
from mir.nn.train import NetworkInterface
from mir.nn.data_storage import FramedH5DataStorage
from train_eval_test_split import get_test_set_ids
from complex_chord import Chord, TriadTypes, SeventhTypes, NinthTypes, EleventhTypes, ThirteenthTypes
from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH

# Human-readable labels for heads
TRIAD_NAMES = ['N', 'maj', 'min', 'sus4', 'sus2', 'dim', 'aug', '5', '1']
SEVENTH_NAMES = ['N', '7', 'b7', 'bb7']
NINTH_NAMES = ['N', '9', '#9', 'b9']
ELEVENTH_NAMES = ['N', '11', '#11']
THIRTEENTH_NAMES = ['N', '13', 'b13']
BASS_NAMES = ['N', 'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def build_crf_transition_matrix(num_classes: int, penalty: float, device: torch.device) -> torch.Tensor:
    """
    Builds a fixed transition log-probability matrix based on Eq. 12 from the ChordFormer paper.
    log_trans[i, j] = 0 if i == j else -penalty
    """
    trans_log = torch.full((num_classes, num_classes), -penalty, dtype=torch.float32, device=device)
    trans_log.fill_diagonal_(0.0)
    return trans_log

def viterbi_decode_crf(probs: np.ndarray, penalty: float = 2.0, device: torch.device = None) -> np.ndarray:
    """
    Viterbi decoding with transition penalty matching chord-gen-test/train.py and ChordFormer Eq. 12.
    """
    if penalty is None or penalty <= 0.0:
        return probs.argmax(axis=-1)

    seq_len, num_classes = probs.shape
    if seq_len <= 1:
        return probs.argmax(axis=-1)

    log_probs = torch.from_numpy(np.log(np.maximum(probs, 1e-12))).to(device)
    trans_log = build_crf_transition_matrix(num_classes, penalty, device)

    score = log_probs[0]
    backpointers = torch.empty((seq_len - 1, num_classes), dtype=torch.long, device=device)

    for t in range(1, seq_len):
        next_score = score.unsqueeze(1) + trans_log
        best_prev_score, best_prev_state = torch.max(next_score, dim=0)
        score = best_prev_score + log_probs[t]
        backpointers[t - 1] = best_prev_state

    decoded = torch.empty(seq_len, dtype=torch.long, device=device)
    last_state = torch.argmax(score)
    decoded[seq_len - 1] = last_state
    for t in range(seq_len - 2, -1, -1):
        last_state = backpointers[t, last_state]
        decoded[t] = last_state

    return decoded.cpu().numpy()

def evaluate_heads(model_pattern='chordformer_head16(1.0,1.0)_s%d.best', max_songs=None, crf_penalty=2.0, save_path=None):
    if save_path is None:
        save_path = f'eval_heads_crf_p{crf_penalty:.1f}_report.txt' if (crf_penalty and crf_penalty > 0) else 'eval_heads_report.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_str = f"CRF Viterbi Decoding (penalty={crf_penalty})" if (crf_penalty and crf_penalty > 0) else "Raw ArgMax (No CRF)"
    print(f"Evaluating 6 heads on {device} using {mode_str} with model pattern '{model_pattern}' across all 5 cross-validation folds...")

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

            # Decode with CRF if requested, else argmax
            if crf_penalty is not None and crf_penalty > 0:
                pred_0 = viterbi_decode_crf(probs[0], penalty=crf_penalty, device=device)
                pred_1 = viterbi_decode_crf(probs[1], penalty=crf_penalty, device=device)
                pred_2 = viterbi_decode_crf(probs[2], penalty=crf_penalty, device=device)
                pred_3 = viterbi_decode_crf(probs[3], penalty=crf_penalty, device=device)
                pred_4 = viterbi_decode_crf(probs[4], penalty=crf_penalty, device=device)
                pred_5 = viterbi_decode_crf(probs[5], penalty=crf_penalty, device=device)
            else:
                pred_0 = probs[0].argmax(axis=-1)
                pred_1 = probs[1].argmax(axis=-1)
                pred_2 = probs[2].argmax(axis=-1)
                pred_3 = probs[3].argmax(axis=-1)
                pred_4 = probs[4].argmax(axis=-1)
                pred_5 = probs[5].argmax(axis=-1)

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
    format_report(all_targets, all_preds, all_triad_qual_targets, all_triad_qual_preds, total_evaluated_songs, title, save_path)

def format_report(all_targets, all_preds, all_triad_qual_targets, all_triad_qual_preds, total_evaluated_songs, title, save_path):
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(title)
    report_lines.append("=" * 70)

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
    report_lines.append("-" * 70)

    # 2. Six Heads
    head_names = ["Full Root/Triad (97 classes)", "Bass (13 classes)", "7th Extension", "9th Extension", "11th Extension", "13th Extension"]
    head_label_maps = [
        None,
        BASS_NAMES,
        SEVENTH_NAMES,
        NINTH_NAMES,
        ELEVENTH_NAMES,
        THIRTEENTH_NAMES
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
        if head_label_maps[h] is not None:
            names = [head_label_maps[h][idx] if idx < len(head_label_maps[h]) else f"Class_{idx}" for idx in unique_labels]
            report_lines.append(classification_report(t_valid, p_valid, labels=unique_labels, target_names=names, zero_division=0, digits=4))
        else:
            report_lines.append(classification_report(t_valid, p_valid, labels=unique_labels, zero_division=0, digits=4))
        report_lines.append("-" * 70)

    full_report = "\n".join(report_lines)
    print(full_report)

    if save_path:
        with open(save_path, 'w') as f_out:
            f_out.write(full_report + "\n")
        print(f"\nSaved full multi-head evaluation report to {save_path}")
    return full_report

def evaluate_heads_from_lab(lab_dir='output/output_chordformer_head16(1.0,1.0)_s%d.best_hmm_full/jam/', max_songs=None, save_path='eval_heads_from_lab_report.txt'):
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
                  f"CHORDFORMER MULTI-HEAD EVALUATION (From Composite HMM .lab, {total_evaluated_songs} songs)", save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Multi-head chord evaluation with optional CRF Viterbi decoding")
    parser.add_argument('--from_lab', action='store_true', help='Evaluate directly from generated .lab files without re-running network inference')
    parser.add_argument('--lab_dir', type=str, default='output/output_chordformer_head16(1.0,1.0)_s%d.best_hmm_full/jam/', help='Directory template containing .lab files')
    parser.add_argument('--model_pattern', type=str, default='chordformer_head16(1.0,1.0)_s%d.best', help='Model checkpoint pattern (e.g. chordformer_head16_synth_ft_s%%d.best)')
    parser.add_argument('--max_songs', type=int, default=None, help='Limit number of test songs (default: all 1217)')
    parser.add_argument('--penalty', '--crf_penalty', type=float, default=2.0, dest='penalty', help='CRF transition penalty (default 2.0 based on Eq. 12)')
    parser.add_argument('--no_crf', action='store_true', help='Disable CRF decoding and use raw argmax')
    parser.add_argument('--save_path', type=str, default=None, help='Custom output text file path')
    args = parser.parse_args()

    if args.from_lab:
        save_path = args.save_path or 'eval_heads_from_lab_report.txt'
        evaluate_heads_from_lab(lab_dir=args.lab_dir, max_songs=args.max_songs, save_path=save_path)
    else:
        penalty = 0.0 if args.no_crf else args.penalty
        evaluate_heads(model_pattern=args.model_pattern, max_songs=args.max_songs, crf_penalty=penalty, save_path=args.save_path)
