from torch import nn
import torch.nn.functional as F
import torch
import pdb

class Weighted_MSELoss(nn.Module):

    def __init__(self, reduction: str = 'none') -> None:
        super(Weighted_MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight):
        # pdb.set_trace()
        loss = F.mse_loss(input, target, reduction=self.reduction)
        # if False in weight:
        #     pdb.set_trace()
        return (loss.mean(dim=1) * weight).mean()

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        #pdb.set_trace()
        weights = (p_target.mean(dim = 1) > 0).float()
        # if p_target.mean() <0:
        #     pdb.set_trace()
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim = 1)) * weights
        samplewise_emd = samplewise_emd.sum()
        if weights.sum() !=0:
            samplewise_emd = samplewise_emd/weights.sum()
        return samplewise_emd

class AIAG_Loss_1(nn.Module):

    def __init__(self, args):
        super(AIAG_Loss_1, self).__init__()
        self.EMD_crit = EMDLoss()
        self.mse_criterion = Weighted_MSELoss()
        self.w_emd = args.w_emd
        self.w_mse = args.w_mse
        self.args = args

    def forward(self, outputs, targets, phase, epoch):
        # pdb.set_trace()
        self.iter_results = {}

        if self.args.A2_D == 1:
            targets_A2 = self.compute_mean_mos_MLSP(torch.FloatTensor(targets)).cuda()
        else:
            #targets_A2 = torch.stack([torch.FloatTensor(t[0]) for t in targets]).cuda()
            targets_A2 = torch.FloatTensor(targets).cuda()

        if self.w_emd == 0:
            loss_A2_emd = torch.FloatTensor([0.0]).cuda()
        else:
            loss_A2_emd = self.A2_EMD_crit(targets_A2, outputs['A2'].float())

        if self.w_mse == 0:
            loss_MSE = torch.FloatTensor([0.0]).cuda()
        else:
            loss_MSE = self.mse_criterion(outputs['A2'].float(), targets_A2, weight= targets_A2.mean(dim=1) > 0)

        loss_A2 = self.w_emd * loss_A2_emd + self.w_mse * loss_MSE
        #pdb.set_trace()
        self.iter_results['loss'] = loss_A2
        self.iter_results['loss_A2'] = loss_A2.data.item()
        self.iter_results['loss_A2_emd'] = loss_A2_emd.data.item()
        self.iter_results['loss_A2_bce'] = 0.0
        self.iter_results['loss_A2_mse'] = loss_MSE.data.item()


        if self.args.A2_D == 10:
            self.iter_results['y_true_A2_mean'], self.iter_results['y_true_A2_std'] = \
               self.compute_mean_std_mos(targets_A2[targets_A2.mean(dim = 1)>0].cpu())
            self.iter_results['y_preds_A2_mean'], self.iter_results['y_preds_A2_std'] = \
                 self.compute_mean_std_mos(outputs['A2'][targets_A2.mean(dim = 1)>0].data.cpu())
        else:
            self.iter_results['y_true_A2_mean'], self.iter_results['y_true_A2_std'] = \
                targets_A2.data.cpu().tolist(), targets_A2.data.cpu().tolist()
            self.iter_results['y_preds_A2_mean'], self.iter_results['y_preds_A2_std'] = \
                outputs['A2'].squeeze(1).data.cpu().tolist(), targets_A2.data.cpu().tolist()

        if phase == 'test':
            self.iter_results['y_true_A2_mean'], self.iter_results['y_true_A2_std'] = \
                self.compute_average_scores(self.iter_results['y_true_A2_mean']), \
                self.compute_average_scores(self.iter_results['y_true_A2_std'])

            self.iter_results['y_preds_A2_mean'], self.iter_results['y_preds_A2_std'] = \
                self.compute_average_scores(self.iter_results['y_preds_A2_mean']), \
                self.compute_average_scores(self.iter_results['y_preds_A2_std'])

        return self.iter_results

    def compute_mean_std_mos(self, x ):
        #pdb.set_trace()
        position_tensor = torch.arange(1,11).float().unsqueeze(0).repeat(x.shape[0], 1)
        mean = torch.sum(x.float() * position_tensor, dim = 1)
        #mean_tensor = mean.unsqueeze(1).repeat(1, position_tensor.shape[1])
        #if np.isnan(np.sum(mean.tolist())):
        #    pdb.set_trace()
        return mean.tolist(), x.std(dim = 1).tolist()
               #torch.sqrt(torch.sum(torch.pow(x.float() * position_tensor - mean_tensor, 2) , dim=1)).tolist()
               #torch.sqrt(torch.sum(torch.pow(position_tensor - mean_tensor, 2) * x.float(), dim = 1)).tolist()

    def compute_mean_mos_MLSP(self, x ):
        #pdb.set_trace()
        position_tensor = torch.arange(1,11).float().unsqueeze(0)
        mean = torch.sum(x.float() * position_tensor, dim = 1)
        #mean_tensor = mean.unsqueeze(1).repeat(1, position_tensor.shape[1])
        return mean

    def set_epoch_count(self, epoch):
        self.epoch = epoch

    def compute_average_scores(self, scores):
        return [t.mean().item() for t in
                 torch.split(torch.Tensor(scores), split_size_or_sections=8) if
                 t.size()[0] != 0]