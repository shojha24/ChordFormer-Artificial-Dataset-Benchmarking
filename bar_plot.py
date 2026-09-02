import numpy as np
import os
from complex_chord import ChordTypeLimit, Chord, NUM_TO_ABS_SCALE
import mir_eval
from settings import JAM_DATASET_PATH
import matplotlib.pyplot as plt

MAX_CLASS_SIZE = 13
chord_limit = ChordTypeLimit(
    triad_limit=6,
    seventh_limit=3,
    ninth_limit=3,
    eleventh_limit=2,
    thirteenth_limit=2
)

def plot_multiple_results(model_template, legend_list, name_list, plot_id):
    from mir import cache
    try:
        values_list, names, sample_counts, l0_counts = cache.load('figure_data_upd2')
    except:
        values_list = []
        for filename in name_list:
            pool = process_folder((model_template % filename).replace('[d]', '%d'),
                                  os.path.join(JAM_DATASET_PATH, 'chordlab') + '/')
            total, correct, l0 = compute_part_recall(pool)
            names, values, sample_counts, l0_counts = get_names_values_to_plot(total, correct, l0, [0, 1, 2, 3, 4, 5])
            values_list.append(values)
        cache.save((values_list, names, sample_counts, l0_counts), 'figure_data_upd2')
    
    x = np.arange(len(names))
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Standard Bar Plot
    bar_width = 0.6
    bars = ax.bar(x, values_list[0], width=bar_width, color='lightblue', label=legend_list[0])
    
    # Overlaying multiple line plots
    for i, filename in enumerate(name_list):
        ax.plot(x, values_list[i], marker='o', markersize=6, linestyle='-', label=legend_list[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Chord Component Recall", labelpad=10)
    ax.set_xlabel("Chord Component Label", labelpad=10)
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc='upper right')
    ax.set_title("Evaluation on Chord Components (Standard Bar Plot)")
    
    # Save the plot
    fig.savefig('component_recall.pdf', transparent=True, pad_inches=0, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_multiple_results("output/output_chordformer(%.1f,%.1f)_s[d].best_hmm_full/jam/",
                          ['no_reweight', '(0.3,10.0)', '(0.5,10.0)', '(0.7,20.0)', '(1.0,20.0)'],
                          [(1.0, 1.0), (0.3, 10.0), (0.5, 10.0), (0.7, 20.0), (1.0, 20.0)],
                          plot_id=1)
