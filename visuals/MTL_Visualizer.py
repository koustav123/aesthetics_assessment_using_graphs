from tensorboardX import SummaryWriter
#import datetime
import pdb
#from sklearn.metrics import average_precision_score, accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import itertools



class MTL_Visualizer(SummaryWriter):
    
    def __init__(self, comment, args, dset_loaders = None, iteration_counter = 0):
        super(MTL_Visualizer, self).__init__(comment)
        self.iteration_counter = iteration_counter
        self.best_acc_P1 = 0.0
        self.best_map_P2 = 0.0
        self.best_map_A200 = 0.0        
        self.best_loss_A1_mse = 0.0
        self.best_loss_A2 = 0.0
        self.best_loss_A678 = 0.0
        self.best_srcc_A2_mean = 0.0
        self.best_plcc_A2_mean = 0.0
        self.best_srcc_A2_std = 0.0
        self.best_plcc_A2_std = 0.0
        self.best_loss_A3_bce = 0.0
        self.best_loss_A4_bce = 0.0
        self.best_overall_score = 0.0
        self.args = args
        # self.style_list = self.load_class_names()

    def iteration_write(self, iter_results):
        pass
    
    def epoch_write(self, epoch, phase):
        pass
        
    def init_running_results(self):
        self.iteration_counter = 0
        self.running_loss = 0.0
        self.running_loss_P1 = 0.0
        self.running_loss_P2 = 0.0
        self.running_loss_A200 = 0.0
        self.running_loss_A1_mse = 0.0

        self.running_loss_A2 = 0.0
        self.running_loss_A2_emd = 0.0
        self.running_loss_A2_bce = 0.0
        self.running_loss_A2_mse = 0.0

        self.running_loss_A3_bce = 0.0

        self.running_loss_A4_bce = 0.0

        self.running_loss_A5 = 0.0

        self.running_loss_A678 = 0.0
        self.running_loss_A678_emd = 0.0
        self.running_loss_A678_mse = 0.0


        self.y_true_P1 = []
        self.y_preds_P1 = []        
        self.y_true_P2 = []
        self.y_preds_P2 = []        
        self.y_true_A200 = []
        self.y_preds_A200 = []
        self.y_true_A1 = []
        self.y_preds_A1 = []
        self.y_true_A2_mean = []
        self.y_preds_A2_mean = []

        # self.threshold = 0.0
        # self.y_true_A2_T = []
        # self.y_preds_A2_T = []

        self.y_true_A2_std = []
        self.y_preds_A2_std = []
        self.A3_GT_grids = []
        self.A3_DCT_preds = []
        self.A4_GT = []
        self.A4_preds = []
        self.A4_input = []

        self.y_true_A678 = []
        self.y_preds_A678 = []

        self.A12_feat_maps = []
        self.A12_y_true = []

        self.ids = []

        self.nan_flag = False

    def update_running_results(self, iter_results, phase):
        pass
    
    def compute_epoch_results(self, n_batches):
        pass
    
    def get_val_loss(self):
        return 1.0 - np.mean([self.epoch_acc_P1, self.epoch_balanced_acc_P1,\
        self.map_weighted_P2])
        
    def checkpoint(self):
        if np.mean([self.epoch_acc_P1, self.epoch_balanced_acc_P1, \
        self.map_weighted_P2]) > self.best_overall_score:
            self.best_overall_score = np.mean([self.epoch_acc_P1, \
            self.epoch_balanced_acc_P1, self.map_weighted_P2])
            return True
        else:
            return False
    
    def plot_confusion_matrix(self, cm, classes = ['Bad', 'Good'] ,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            #print("Normalized confusion matrix")
        else:
            #print('Confusion matrix, without normalization')
            pass
    
        #print(cm)
        #pdb.set_trace()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def plot_pcp(self, pcp):
        plt.bar(range(pcp.shape[0]), pcp)
        plt.ylim([0,1])
        plt.tight_layout()       

    def load_class_names(self):
        class_list = np.loadtxt(self.args.style_list, dtype = str)
        indices = class_list[:,0].astype(int) -1; 
        class_names = class_list[:,1]
        return dict(zip(indices, class_names))
        
    def plot_mos(self, y_true, y_preds):
        plt.scatter(y_true, y_preds, s=2.0 )
        plt.xlim([0,10])
        plt.ylim([0,10])
        plt.grid(True)
        plt.xlabel('True MOS')
        plt.ylabel('Predicted MOS')