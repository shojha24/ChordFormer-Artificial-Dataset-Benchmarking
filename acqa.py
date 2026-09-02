from mir.nn.train import NetworkInterface
import mir.io as io
import datasets
from extractors.cqt import CQTV2,SimpleChordToID
from mir import io,DataEntry
from io_new.chordlab_io import ChordLabIO
from extractors.xhmm_decoder import XHMMDecoder,prob_to_spectrogram
from complex_chord import Chord,ChordTypeLimit,shift_complex_chord_array_list,complex_chord_chop,enum_to_dict,\
    TriadTypes,SeventhTypes,NinthTypes,EleventhTypes,ThirteenthTypes
from mir.music_base import NUM_TO_ABS_SCALE
import os
import mir_eval
import acqa_mir
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class ExperimentTest():

    #ET_confusion_chord = ["Others","N","maj/2","maj/4","13","11","maj9","9","min7","maj7","7","maj6","min6","maj/3","maj/5","min/b3","min/5","maj","min"]
    ET_confusion_chord = None #["Others","N","sus2","sus4","hdim7","dim7","min7","minmaj7","maj7","7","maj6","min6","aug","dim","maj","min"]

    def __init__(self,isDirectory = False,address_x = "",address_y = ""):
        self.address = []
        self.address.append(address_x)
        self.address.append(address_y)
        self.isDirectory = isDirectory
        self.correct_durations = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.wrong_durations = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.evals = [acqa_mir.thirds_quality_only,acqa_mir.majmin_quality_only,acqa_mir.triads_quality_only,acqa_mir.sevenths_quality_only,acqa_mir.tetrads_quality_only,acqa_mir.thirds_inv_quality_only,acqa_mir.majmin_inv_quality_only, acqa_mir.triads_inv_quality_only, acqa_mir.sevenths_inv_quality_only, acqa_mir.tetrads_inv_quality_only]
        self.evalStrings = ["Eval for Thirds:", "Eval for MajorMinor:", "Eval for Triads:", "Eval for Sevenths:", "Eval for Tetrads:","Eval for Thirds inv:","Eval for MajMin inv","Eval for Triads inv","Eval for Sevenths inv","Eval for Tetrads inv"]
        self.confusion_mat = []
        self.confusion_total = []
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

            for i in range(len(self.correct_durations)):
                print(self.evalStrings[i],'%.2f%%'%(100.0*self.correct_durations[i]/(self.correct_durations[i]+self.wrong_durations[i])))


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

        plt.show()


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
    ExperimentTest.ET_confusion_chord=extract_quality_list_from_file('data/submission_chord_list.txt')
    #q = ExperimentTest(True,os.path.join(JAM_DATASET_PATH,'chordlab')+'/',"output/output_joint_chord_net_ismir_flat_v1.0_reweight(1.0,1.0)_s%d.best_hmm_submission/jam/")
    q = ExperimentTest(True,os.path.join(JAM_DATASET_PATH,'chordlab')+'/',"output/output_joint_chord_net_ismir_naive_v1.0_reweight(0.5,10.0)_s%d.best_hmm_ismir2017/jam/")

    #q = ExperimentTest(True,os.path.join(JAM_DATASET_PATH,'chordlab')+'/',"output/output_hook_mirex2019_chordnet_v1.1_fix_s%d.best_hmm_ismir2017/jam/")
    q.calc_accuracy()
    q.draw_confusion_mat()

def eval_submission(reweight_factor,reweight_max):

    from settings import JAM_DATASET_PATH
    ExperimentTest.ET_confusion_chord=extract_quality_list_from_file('data/submission_chord_list.txt')
    q = ExperimentTest(True,os.path.join(JAM_DATASET_PATH,'chordlab')+'/',"output/output_chordformer_head16(%.1f,%.1f)"%(reweight_factor,reweight_max)+"_s%d.best_hmm_full/jam/")
    q.calc_accuracy()
    q.draw_confusion_mat()
if __name__ == "__main__":
    #main()
    eval_submission(1.0,1.0)