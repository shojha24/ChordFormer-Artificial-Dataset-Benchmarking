from mir.nn.train import NetworkInterface
import mir.io as io
import datasets
from extractors.cqt import CQTV2,SimpleChordToID
from mir import io,DataEntry
from io_new.chordlab_io import ChordLabIO
from extractors.xhmm_decoder import XHMMDecoder,prob_to_spectrogram
from complex_chord import Chord,ChordTypeLimit,shift_complex_chord_array_list,complex_chord_chop,enum_to_dict,\
    TriadTypes,SeventhTypes,NinthTypes,EleventhTypes,ThirteenthTypes
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

import os
import mir_eval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections

class ExperimentTest():

    #ET_confusion_chord = ["Others","N","maj/2","maj/4","13","11","maj9","9","min7","maj7","7","maj6","min6","maj/3","maj/5","min/b3","min/5","maj","min"]
    ET_confusion_chord = None #["Others","N","sus2","sus4","hdim7","dim7","min7","minmaj7","maj7","7","maj6","min6","aug","dim","maj","min"]

    def __init__(self,isDirectory = False,address_x = "",address_y = ""):
        self.address = []
        self.address.append(address_x)
        self.address.append(address_y)
        self.isDirectory = isDirectory
        self.correct_durations = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.wrong_durations = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.evals = [mir_eval.chord.root,mir_eval.chord.thirds,mir_eval.chord.majmin,mir_eval.chord.triads,mir_eval.chord.sevenths,mir_eval.chord.tetrads,mir_eval.chord.mirex,mir_eval.chord.thirds_inv,mir_eval.chord.majmin_inv, mir_eval.chord.triads_inv, mir_eval.chord.sevenths_inv, mir_eval.chord.tetrads_inv]
        self.evalStrings = ["Eval for root", "Eval for Thirds:", "Eval for MajorMinor:", "Eval for Triads:", "Eval for Sevenths:", "Eval for Tetrads:", "Eval for MIREX:","Eval for Thirds inv:","Eval for MajMin inv","Eval for Triads inv","Eval for Sevenths inv","Eval for Tetrads inv"]
        self.confusion_mat = []
        self.confusion_total = []
        self.quality_stats = {}
        ET_confusion_chord = ExperimentTest.ET_confusion_chord
        for i in range(len(ET_confusion_chord)):
            c = []
            for j in range(len(ET_confusion_chord)):
                c.append(0.0)
            self.confusion_mat.append(c)
            self.confusion_total.append(0.0)

    def index_from_chord(self,fchord):
        ET_confusion_chord = ExperimentTest.ET_confusion_chord
        for i,chord in enumerate(ET_confusion_chord):
            if fchord==chord:
                return i
        return -1
    def compare_chord_for_mat(self,x1,x2,duration):
        #print(x1,x2)
        coms_1 = x1.split(":")
        coms_2 = x2.split(":")
        if len(coms_1) == 1:
            coms_1.append(coms_1[0])
        if len(coms_2) == 1:
            coms_2.append(coms_2[0])
        if coms_1[0] == coms_2[0] or coms_1[0] == "N" or coms_1[0] == "X":
            idx_1 = self.index_from_chord(coms_1[1])
            idx_2 = self.index_from_chord(coms_2[1])
            if idx_1 == -1:
                idx_1=0
            if idx_2 == -1:
                # print(x2)
                idx_2=0
            self.confusion_total[idx_1] = self.confusion_total[idx_1] + duration
            self.confusion_mat[idx_1][idx_2] = self.confusion_mat[idx_1][idx_2] + duration


    def get_lab(self,file_address):
        chordlab = []
        with open(file_address,"r") as f:
            for lines in f:
                lines = lines.strip()
                lines = lines.split()
                chordlab.append([float(lines[0]),float(lines[1]),lines[2]])
        return chordlab

    def split_chordlab(self,chordlab):    
        return (np.array([[data[0],data[1]] for data in chordlab],dtype=np.float64),[data[2] for data in chordlab])

    def test(self,gd,est):
        name = gd
        gd = self.get_lab(gd)
        est = self.get_lab(est)

        if gd[0][0] < 0:
            gd[0][0] = 0

        (gd_intervals,gd_labels) = self.split_chordlab(gd) 
        (est_intervals,est_labels) = self.split_chordlab(est)
        est_intervals,est_labels = mir_eval.util.adjust_intervals(est_intervals,est_labels,gd_intervals.min(),gd_intervals.max(),start_label='X',end_label='X')    
        (intervals,gd_labels,est_labels)=mir_eval.util.merge_labeled_intervals(gd_intervals,gd_labels,est_intervals,est_labels) 
        durations = mir_eval.util.intervals_to_durations(intervals)
        compares = []
        scores = []

        for evalk in self.evals:
            compare = evalk(gd_labels,est_labels)
            #print(compare)
            compares.append(compare)
            score = mir_eval.chord.weighted_accuracy(compare,durations)
            scores.append(score)
        #print(name+":\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%(scores[0] * 100.0,scores[1] * 100.0,scores[2] * 100.0,scores[3] * 100.0,scores[4] * 100.0))
        for i in range(len(self.correct_durations)):
            self.correct_durations[i] += durations[compares[i] > 0].sum()
            self.wrong_durations[i] += durations[compares[i] == 0].sum()

        for i in range(len(durations)):
            self.compare_chord_for_mat(gd_labels[i],est_labels[i],durations[i])
            self.record_quality_stats(gd_labels[i],est_labels[i],durations[i])

    def parse_chord(self, chord_str):
        if not chord_str or chord_str in ['N', 'X']:
            return 'N', 'N'
        parts = chord_str.split(':')
        root = parts[0]
        qual = parts[1] if len(parts) > 1 else 'maj'
        if qual == '1/1':
            qual = '1'
        return root, qual

    def record_quality_stats(self, gd_chord, est_chord, duration):
        gd_root, gd_qual = self.parse_chord(gd_chord)
        est_root, est_qual = self.parse_chord(est_chord)

        if gd_qual not in self.quality_stats:
            self.quality_stats[gd_qual] = {
                'gt_dur': 0.0, 'est_dur': 0.0, 'exact_match_dur': 0.0,
                'qual_match_dur': 0.0, 'conf': collections.defaultdict(float)
            }
        if est_qual not in self.quality_stats:
            self.quality_stats[est_qual] = {
                'gt_dur': 0.0, 'est_dur': 0.0, 'exact_match_dur': 0.0,
                'qual_match_dur': 0.0, 'conf': collections.defaultdict(float)
            }

        self.quality_stats[gd_qual]['gt_dur'] += duration
        self.quality_stats[est_qual]['est_dur'] += duration

        exact_match = (gd_root == est_root and gd_qual == est_qual)
        qual_match = (gd_qual == est_qual)

        if exact_match:
            self.quality_stats[gd_qual]['exact_match_dur'] += duration
        if qual_match:
            self.quality_stats[gd_qual]['qual_match_dur'] += duration

        if not exact_match:
            self.quality_stats[gd_qual]['conf'][est_chord] += duration



    def calc_accuracy(self):
        if not 1 == 1:
            print("Model File not exist!")
        else:
            if self.isDirectory:
                path = os.listdir(self.address[0])
                gds = []
                ests = []
                for i in path:
                    gds.append(i)
                gds = np.array(gds)
                path = os.listdir(self.address[1])
                for i in path:
                    ests.append(i)
                ests = np.array(ests)

                inters = np.intersect1d(gds,ests)
                print("total files: " + str(len(inters)))
                for q in inters:
                    self.test(self.address[0] + q,self.address[1] + q)

            else:
                self.test(self.address[0],self.address[1])

            results_lines = []
            for i in range(len(self.correct_durations)):
                total_dur = self.correct_durations[i] + self.wrong_durations[i]
                pct = 100.0 * self.correct_durations[i] / total_dur if total_dur > 0 else 0.0
                line = '%s %.2f%%' % (self.evalStrings[i], pct)
                print(line)
                results_lines.append(line)

            breakdown_text = self.print_quality_breakdown()
            results_lines.append(breakdown_text)

            try:
                with open('eval_results.txt', 'w') as f_out:
                    f_out.write('\n'.join(results_lines) + '\n')
                with open('eval_detailed_breakdown.txt', 'w') as f_out2:
                    f_out2.write(breakdown_text + '\n')
                print('\nSaved evaluation metrics to eval_results.txt and eval_detailed_breakdown.txt')
            except Exception as e:
                print('Warning: could not save eval_results.txt:', e)

    def print_quality_breakdown(self):
        lines = []
        lines.append("\n" + "=" * 110)
        lines.append("DETAILED CHORD QUALITY & EXTENSION BREAKDOWN (Precision, Recall, F1)")
        lines.append("=" * 110)

        total_gt_dur = sum(stat['gt_dur'] for stat in self.quality_stats.values())
        sorted_quals = sorted(self.quality_stats.keys(), key=lambda q: self.quality_stats[q]['gt_dur'], reverse=True)

        # Macro accuracy (acc_class) and Frame-weighted accuracy (acc_frame)
        recalls_exact = []
        recalls_qual = []
        total_exact_match = 0.0
        total_qual_match = 0.0

        for q in sorted_quals:
            stat = self.quality_stats[q]
            gt_d = stat['gt_dur']
            if gt_d > 1.0: # filter out negligible duration (< 1s)
                recalls_exact.append(stat['exact_match_dur'] / gt_d)
                recalls_qual.append(stat['qual_match_dur'] / gt_d)
            total_exact_match += stat['exact_match_dur']
            total_qual_match += stat['qual_match_dur']

        acc_frame_exact = (total_exact_match / total_gt_dur) if total_gt_dur > 0 else 0.0
        acc_class_exact = (np.mean(recalls_exact)) if len(recalls_exact) > 0 else 0.0
        acc_frame_qual = (total_qual_match / total_gt_dur) if total_gt_dur > 0 else 0.0
        acc_class_qual = (np.mean(recalls_qual)) if len(recalls_qual) > 0 else 0.0

        lines.append(f"Root+Quality Frame-wise Accuracy (acc_frame): {acc_frame_exact:.4f} ({acc_frame_exact*100.0:.2f}%)")
        lines.append(f"Root+Quality Class-wise Accuracy (acc_class): {acc_class_exact:.4f} ({acc_class_exact*100.0:.2f}%)")
        lines.append(f"Quality-Only Frame-wise Accuracy (acc_frame): {acc_frame_qual:.4f} ({acc_frame_qual*100.0:.2f}%)")
        lines.append(f"Quality-Only Class-wise Accuracy (acc_class): {acc_class_qual:.4f} ({acc_class_qual*100.0:.2f}%)")
        lines.append("-" * 110)

        header = f"{'Quality':<16} | {'GT Dur (s)':<11} | {'GT Share':<9} | {'Recall':<8} | {'Precision':<9} | {'F1-Score':<8} | {'Top Confusion':<28}"
        lines.append(header)
        lines.append("-" * 110)

        for q in sorted_quals:
            stat = self.quality_stats[q]
            gt_d = stat['gt_dur']
            est_d = stat['est_dur']
            match_d = stat['exact_match_dur']
            if gt_d <= 0.05 and est_d <= 0.05:
                continue

            share = (gt_d / total_gt_dur * 100.0) if total_gt_dur > 0 else 0.0
            recall = (match_d / gt_d * 100.0) if gt_d > 0 else 0.0
            precision = (match_d / est_d * 100.0) if est_d > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            top_conf = "None"
            if stat['conf']:
                top_item = max(stat['conf'].items(), key=lambda item: item[1])
                top_conf = f"{top_item[0]} ({top_item[1]:.1f}s)"

            row = f"{q:<16} | {gt_d:<11.1f} | {share:>7.2f}% | {recall:>7.2f}% | {precision:>8.2f}% | {f1:>7.2f}% | {top_conf:<28}"
            lines.append(row)

        lines.append("=" * 110)
        out_text = "\n".join(lines)
        print(out_text)
        return out_text


    def draw_confusion_mat(self):
        self.confusion_mat = np.array(self.confusion_mat)
        print('Diag ratio unnormalized', self.confusion_mat[1:,1:].diagonal().sum()/self.confusion_mat[1:,1:].sum())
        for i,submat in enumerate(self.confusion_mat):
            if self.confusion_total[i] == 0:
                self.confusion_total[i] = 1
            for j,value in enumerate(submat):
                self.confusion_mat[i][j] /= self.confusion_total[i]
                self.confusion_mat[i][j] = round(self.confusion_mat[i][j],2)
        print(self.confusion_mat)
        self.confusion_mat = np.array(self.confusion_mat)
        print('Diag ratio normalized', self.confusion_mat[1:,1:].diagonal().sum()/self.confusion_mat[1:,1:].sum())
        fig,ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(self.confusion_mat,cmap = "Blues",vmin=0.0,vmax=1.0)
        cbar = ax.figure.colorbar(im,ax = ax)
        cbar.ax.set_ylabel("ratio of confusion",rotation = -90,va = "bottom")
        plt.xlabel('estimation')
        plt.ylabel('reference')
        ax.set_xticks(np.arange(len(ExperimentTest.ET_confusion_chord)))
        ax.set_yticks(np.arange(len(ExperimentTest.ET_confusion_chord)))
        ax.set_xticklabels(ExperimentTest.ET_confusion_chord)
        ax.set_yticklabels(ExperimentTest.ET_confusion_chord)
        plt.setp(ax.get_xticklabels(),rotation = 45,ha = "right",rotation_mode = "anchor")
        for i in range(len(ExperimentTest.ET_confusion_chord)):
            for j in range(len(ExperimentTest.ET_confusion_chord)):
                scolor = 'w'
                if self.confusion_mat[i][j] < 0.3:
                    scolor = '#000000'
                #value = ax.text(j,i,self.confusion_mat[i][j],ha = "center", va = "center", color = scolor)
        #ax.set_title("Confusion Matrix for Chord Recognition")
        fig.tight_layout()
        fig.savefig('confusion_matrix.pdf', transparent=True, pad_inches=0,bbox_inches='tight')
        fig.savefig('confusion_matrix.png', dpi=300, pad_inches=0,bbox_inches='tight')
        plt.close(fig)
        print('Saved confusion_matrix.pdf and confusion_matrix.png')


def extract_quality_list_from_file(filename):
    f=open(filename,'r')
    lines=[line.strip() for line in f.readlines() if line.strip()!='']
    f.close()
    result=['Others']

    for line in lines:
        if(line.startswith('C:')):
            result.append(line[2:])
        else:
            result.append(line)
    return result

def main():
    
    from settings import JAM_DATASET_PATH,MY_DATASET_PATH
    dict_file = 'data/extended_chord_list.txt' if os.path.exists('data/extended_chord_list.txt') else 'data/submission_chord_list.txt'
    ExperimentTest.ET_confusion_chord=extract_quality_list_from_file(dict_file)
    q = ExperimentTest(True,os.path.join(JAM_DATASET_PATH,'chordlab')+'/',"output/output_joint_chord_net_ismir_naive_v1.0_reweight(0.5,10.0)_s%d.best_hmm_ismir2017/jam/")
    q.calc_accuracy()
    q.draw_confusion_mat()

def eval_submission(reweight_factor=1.0, reweight_max=1.0, chord_dict='extended'):

    from settings import JAM_DATASET_PATH
    dict_file = f'data/{chord_dict}_chord_list.txt' if os.path.exists(f'data/{chord_dict}_chord_list.txt') else 'data/submission_chord_list.txt'
    ExperimentTest.ET_confusion_chord=extract_quality_list_from_file(dict_file)
    q = ExperimentTest(True,os.path.join(JAM_DATASET_PATH,'chordlab')+'/',"output/output_chordformer_head16(%.1f,%.1f)"%(reweight_factor,reweight_max)+"_s%d.best_hmm_full/jam/")
    q.calc_accuracy()
    q.draw_confusion_mat()
if __name__ == "__main__":
    eval_submission(1.0,1.0)