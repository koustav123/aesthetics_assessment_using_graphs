from tensorboardX import SummaryWriter
import datetime
import pdb
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from MTL_Visualizer import MTL_Visualizer
from scipy.stats import pearsonr, spearmanr
from random import shuffle
import torch
from matplotlib import cm
import torch.nn.functional as F

class A2_Visualizer(MTL_Visualizer):    
        
          

    def __init__(self, comment, args, iteration_counter = 0):
        super(A2_Visualizer, self).__init__(comment, args)

    def epoch_write(self, epoch, phase):
        #pdb.set_trace()
        self.add_scalars('/A2/Loss/Overall/', { phase : self.epoch_loss_A2}, epoch)
        self.add_scalars('/A2/Loss/EMD/', { phase : self.epoch_loss_A2_emd}, epoch)
        #self.add_scalars('/A2/Loss/BCE/', { phase : self.epoch_loss_A2_bce}, epoch)
        self.add_scalars('/A2/Loss/MSE/', {phase: self.epoch_loss_A2_mse}, epoch)

        #pdb.set_trace()
        self.add_scalars('/A2/SRCC(Mean)/', { phase : self.epoch_srcc_A2_mean}, epoch)
        self.add_scalars('/A2/PLCC(Mean)/', { phase : self.epoch_plcc_A2_mean}, epoch)
        self.add_scalars('/A2/SRCC(STD)/', { phase : self.epoch_srcc_A2_std}, epoch)
        self.add_scalars('/A2/PLCC(STD)/', { phase : self.epoch_plcc_A2_std}, epoch)

        self.add_scalars('/A2/Accuracy/', { phase : self.epoch_acc_A2}, epoch)
        self.add_scalars('/A2/Balanced Accuracy/', { phase : self.epoch_balanced_acc_A2}, epoch)

        # self.add_scalars('/P1/Loss/', {phase: self.epoch_loss_P1}, epoch)
        # self.add_scalars('/P1/Accuracy/', { phase : self.epoch_acc_P1}, epoch)
        # self.add_scalars('/P1/Balanced Accuracy/', { phase : self.epoch_balanced_acc_P1}, epoch)

        self.add_scalars('/A2/F1 Score/', { phase : self.epoch_f_A2}, epoch)

        cm_fig = plt.figure()
        self.plot_confusion_matrix(self.cm_A2,  title='Confusion matrix ' + phase, normalize = True)
        self.add_figure('A2/CM_' + phase, cm_fig, epoch)

        # self.add_scalars('/A2/T_Accuracy/', { phase : self.epoch_acc_A2_T}, epoch)
        # self.add_scalars('/A2/T_Balanced Accuracy/', { phase : self.epoch_balanced_acc_A2_T}, epoch)
        # self.add_scalars('/A2/Threshold/', {phase: self.epoch_threshold}, epoch)
        # cm_fig_T = plt.figure()
        # self.plot_confusion_matrix(self.cm_A2_T,  title='Confusion matrix Thresholded' + phase, normalize = True)
        # self.add_figure('A2/CM_T_' + phase, cm_fig_T, epoch)

        scatter_mos_mean = plt.figure()
        #pdb.set_trace()
        self.plot_mos(torch.FloatTensor(self.y_true_A2_mean)[self.rand_int_for_mos[:1000]].numpy(), \
        torch.FloatTensor(self.y_preds_A2_mean)[self.rand_int_for_mos[:1000]].numpy())
        self.add_figure('A2/MOS_MEAN' + phase, scatter_mos_mean, epoch)



    def update_running_results(self, iter_results, phase):
        #pdb.set_trace()
        self.running_loss_A2 += iter_results['loss_A2']
        self.running_loss_A2_emd  += iter_results['loss_A2_emd']
        #self.running_loss_A2_bce  += iter_results['loss_A2_bce']
        self.running_loss_A2_mse += iter_results['loss_A2_mse']
        self.y_true_A2_mean += iter_results['y_true_A2_mean']
        self.y_preds_A2_mean += iter_results['y_preds_A2_mean']     
        self.y_true_A2_std += iter_results['y_true_A2_std']
        self.y_preds_A2_std += iter_results['y_preds_A2_std']

        # self.running_loss_P1 += iter_results['loss_P1'].data
        # self.y_true_P1 += iter_results['y_true_P1']
        # self.y_preds_P1 += iter_results['y_preds_P1']
        # self.y_true_A2_T += iter_results['y_true_A2_T']
        # self.y_preds_A2_T += iter_results['y_preds_A2_T']
        # self.threshold += iter_results['threshold']
        
    def compute_epoch_results(self, n_batches, acc_t = 5):
        # pdb.set_trace()
        self.epoch_loss = self.running_loss / n_batches
        self.epoch_loss_A2 = self.running_loss_A2 / n_batches
        self.epoch_loss_A2_emd = self.running_loss_A2_emd / n_batches
        #self.epoch_loss_A2_bce = self.running_loss_A2_bce / n_batches
        self.epoch_loss_A2_mse = self.running_loss_A2_mse / n_batches
        #pdb.set_trace()
        self.epoch_srcc_A2_mean = spearmanr(self.y_true_A2_mean, self.y_preds_A2_mean).correlation
        self.epoch_plcc_A2_mean = pearsonr(self.y_true_A2_mean, self.y_preds_A2_mean)[0]
        self.epoch_srcc_A2_std = spearmanr(self.y_true_A2_std, self.y_preds_A2_std).correlation
        self.epoch_plcc_A2_std = pearsonr(self.y_true_A2_std, self.y_preds_A2_std)[0]


        #pdb.set_trace()
        self.epoch_acc_A2 = accuracy_score((np.array(self.y_true_A2_mean) > 5).astype(int), (np.array(self.y_preds_A2_mean)> acc_t).astype(int))
        self.epoch_balanced_acc_A2 = balanced_accuracy_score((np.array(self.y_true_A2_mean) > 5).astype(int), (np.array(self.y_preds_A2_mean)> acc_t).astype(int))
        self.epoch_f_A2 = f1_score((np.array(self.y_true_A2_mean) > 5).astype(int), (np.array(self.y_preds_A2_mean)>acc_t).astype(int))
        self.cm_A2 = confusion_matrix((np.array(self.y_true_A2_mean) > 5).astype(int), (np.array(self.y_preds_A2_mean)>acc_t).astype(int))
        self.rand_int_for_mos = [*range(len(self.y_true_A2_mean))]; shuffle(self.rand_int_for_mos)
        # pdb.set_trace()
        # self.epoch_loss_P1 = self.running_loss_P1 / n_batches
        # self.epoch_acc_P1 = accuracy_score((np.array(self.y_true_P1) > 0.5).astype(int), (np.array(self.y_preds_P1)> 0.5).astype(int))
        # self.epoch_balanced_acc_P1 = balanced_accuracy_score((np.array(self.y_true_P1) > 0.5).astype(int), (np.array(self.y_preds_P1)> 0.5).astype(int))

        # self.epoch_acc_A2_T = accuracy_score((np.array(self.y_true_A2_T)).astype(int),
        #                                    (np.array(self.y_preds_A2_T)).astype(int))
        # self.epoch_balanced_acc_A2_T = balanced_accuracy_score((np.array(self.y_true_A2_T)).astype(int),
        #                                    (np.array(self.y_preds_A2_T)).astype(int))
        # self.cm_A2_T = confusion_matrix((np.array(self.y_true_A2_T)).astype(int), (np.array(self.y_preds_A2_T)).astype(int))
        # self.epoch_threshold = self.threshold / n_batches
        #pdb.set_trace()
        
    def get_val_loss(self):
        #return 1.0 - np.mean([self.epoch_balanced_acc_A2,\
        #self.epoch_srcc_A2_mean, self.epoch_plcc_A2_mean])
        #return  1.0 - self.epoch_plcc_A2_mean
        return self.epoch_loss_A2

    def checkpoint(self):
        if 1 - self.epoch_loss_A2 > self.best_overall_score:
            self.best_overall_score = 1 - self.epoch_loss_A2
            return True
        else:
            return False


