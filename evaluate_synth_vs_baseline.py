import os
import sys
import re
import argparse
import numpy as np

def parse_head_report(report_path):
    """
    Parses an eval_heads text report to extract key metrics.
    """
    if not os.path.exists(report_path):
        return None

    results = {
        'heads': {},
        'rare_classes': {}
    }

    with open(report_path, 'r') as f:
        content = f.read()

    # Head 1 to 6 accuracies
    head_matches = re.findall(r'=== (Head \d+:[^=]+)===\s+Frame-wise Accuracy \(acc_frame\): ([\d\.]+)\s+Class-wise Accuracy \(acc_class\): ([\d\.]+)', content)
    for head_name, acc_f, acc_c in head_matches:
        h_clean = head_name.strip()
        results['heads'][h_clean] = {
            'acc_frame': float(acc_f),
            'acc_class': float(acc_c)
        }

    # Extract specific interesting extension classes:
    # 7th: 7, b7, bb7
    # 9th: 9, #9, b9
    # 11th: 11, #11
    # 13th: 13, b13
    target_classes = ['7', 'b7', 'bb7', '9', '#9', 'b9', '11', '#11', '13', 'b13', 'sus2', 'sus4', 'dim', 'aug']
    for cls in target_classes:
        pattern = rf'^\s+{re.escape(cls)}\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            prec, rec, f1, sup = match.groups()
            results['rare_classes'][cls] = {
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'support': int(sup)
            }

    return results

def compare_reports(baseline_path, synth_path, out_file='comparison_summary.txt'):
    base_data = parse_head_report(baseline_path)
    synth_data = parse_head_report(synth_path)

    if not base_data:
        print(f"Error: Baseline report not found at {baseline_path}")
        return
    if not synth_data:
        print(f"Error: Synthetic fine-tune report not found at {synth_path}")
        return

    lines = []
    lines.append("================================================================================")
    lines.append("   SYNTHETIC PRETRAINING vs BASELINE (CHORDFORMER 1217 BENCHMARK COMPARISON)   ")
    lines.append("================================================================================\n")

    lines.append("### 1. Overall Head Accuracy (Frame-wise & Class-wise Macro Accuracy)\n")
    lines.append(f"{'Head':<35} | {'Metric':<10} | {'Baseline':<10} | {'Synth Pretrain':<14} | {'Gain':<10}")
    lines.append("-" * 88)

    for h_name in base_data['heads']:
        if h_name in synth_data['heads']:
            b_f = base_data['heads'][h_name]['acc_frame']
            s_f = synth_data['heads'][h_name]['acc_frame']
            diff_f = (s_f - b_f) * 100.0

            b_c = base_data['heads'][h_name]['acc_class']
            s_c = synth_data['heads'][h_name]['acc_class']
            diff_c = (s_c - b_c) * 100.0

            lines.append(f"{h_name:<35} | {'Frame Acc':<10} | {b_f:<10.4f} | {s_f:<14.4f} | {diff_f:>+6.2f}%")
            lines.append(f"{'':<35} | {'Class Acc':<10} | {b_c:<10.4f} | {s_c:<14.4f} | {diff_c:>+6.2f}%")
            lines.append("-" * 88)

    lines.append("\n### 2. Rare Extension & Alteration Recall (The Core Motivation of Synthetic Data)\n")
    lines.append(f"{'Class':<12} | {'Category':<15} | {'Support':<10} | {'Baseline Rec':<12} | {'Synth Rec':<12} | {'Recall Gain':<12}")
    lines.append("-" * 85)

    categories = {
        '7': '7th Extension', 'b7': '7th Extension', 'bb7': '6th/bb7 Ext',
        '9': '9th Extension', '#9': 'Altered 9th', 'b9': 'Altered 9th',
        '11': '11th Extension', '#11': 'Altered 11th',
        '13': '13th Extension', 'b13': 'Altered 13th',
        'sus2': 'Suspended Triad', 'sus4': 'Suspended Triad', 'dim': 'Diminished Triad', 'aug': 'Augmented Triad'
    }

    for cls in categories:
        if cls in base_data['rare_classes'] and cls in synth_data['rare_classes']:
            b_rec = base_data['rare_classes'][cls]['recall']
            s_rec = synth_data['rare_classes'][cls]['recall']
            sup = base_data['rare_classes'][cls]['support']
            diff = (s_rec - b_rec) * 100.0
            cat = categories[cls]
            lines.append(f"{cls:<12} | {cat:<15} | {sup:<10} | {b_rec:<12.4f} | {s_rec:<12.4f} | {diff:>+8.2f}%")

    lines.append("-" * 85)
    summary_text = "\n".join(lines)
    print(summary_text)

    with open(out_file, 'w') as f:
        f.write(summary_text + "\n")
    print(f"\nSaved comparison summary to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare synthetic fine-tuned evaluation against baseline")
    parser.add_argument('--baseline', type=str, default='baseline_eval_heads_report.txt', help='Path to baseline report')
    parser.add_argument('--synth', type=str, default='synth_ft_eval_heads_report.txt', help='Path to synthetic FT report')
    parser.add_argument('--output', type=str, default='eval_comparison_summary.txt', help='Output summary file')
    args = parser.parse_args()

    compare_reports(args.baseline, args.synth, args.output)
