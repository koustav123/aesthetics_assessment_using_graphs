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
        # self.EMD_crit = EMDLoss()
        # self.mse_criterion = Weighted_MSELoss()
        # self.w_emd = args.w_emd
        # self.w_mse = args.w_mse
        self.args = args

    def forward(self, outputs, targets, phase, epoch):
        # pdb.set_trace()
        self.iter_results = {}

        # if self.args.A2_D == 1:
        #     targets_A2 = self.compute_mean_mos_MLSP(torch.FloatTensor(targets)).cuda()
        # else:
        #     #targets_A2 = torch.stack([torch.FloatTensor(t[0]) for t in targets]).cuda()
        targets_A2 = torch.FloatTensor(targets).cuda()
        # pdb.set_trace()
        # if self.args.A2_D == 1:
        #     targets_A2 = torch.FloatTensor([t[0] for t in targets]).squeeze(0)
        # elif self.args.A2_D == 2:
        #     targets_A2 = torch.FloatTensor([t[1] for t in targets]).squeeze(0)
        # elif self.args.A2_D == 5:
        #     targets_A2 = torch.FloatTensor([t[2] for t in targets]).squeeze(0)
        # else:
        #     print('wrong A2_D')
        #     exit()
        # targets_A2 = targets_A2.cuda()
        loss_MSE = F.mse_loss(outputs['A2'], targets_A2)

        loss_A2 = loss_MSE
        #pdb.set_trace()
        self.iter_results['loss'] = loss_MSE
        self.iter_results['loss_A2'] = loss_A2.data.item()
        self.iter_results['loss_A2_emd'] = 0.0
        self.iter_results['loss_A2_bce'] = 0.0
        self.iter_results['loss_A2_mse'] = loss_MSE.data.item()

        # pdb.set_trace()
        if self.args.A2_D == 2:
            targets_A2 = targets_A2[:, 0]
            outputs['A2'] = outputs['A2'][:, 0]

        if self.args.A2_D == 5:
            targets_A2, _ = self.compute_mean_std_mos(targets_A2.cpu())
            outputs['A2'], _ = self.compute_mean_std_mos(outputs['A2'].cpu())

        self.iter_results['y_true_A2_mean'], self.iter_results['y_true_A2_std'] = \
            targets_A2.data.cpu().tolist(), targets_A2.data.cpu().tolist()
        self.iter_results['y_preds_A2_mean'], self.iter_results['y_preds_A2_std'] = \
            outputs['A2'].data.cpu().tolist(), outputs['A2'].data.cpu().tolist()

        if phase == 'test' and 'RGB' not in self.args.id:
            self.iter_results['y_true_A2_mean'], self.iter_results['y_true_A2_std'] = \
                self.compute_average_scores(self.iter_results['y_true_A2_mean']), \
                self.compute_average_scores(self.iter_results['y_true_A2_std'])

            self.iter_results['y_preds_A2_mean'], self.iter_results['y_preds_A2_std'] = \
                self.compute_average_scores(self.iter_results['y_preds_A2_mean']), \
                self.compute_average_scores(self.iter_results['y_preds_A2_std'])

        #added for hitogram visualization
        # self.iter_results['y_true'], self.iter_results['y_preds'] = targets_A2.cpu(), outputs['A2'].data.cpu()
        # pdb.set_trace()
        return self.iter_results

    def compute_mean_std_mos(self, x, rng = (0, 5)):
        #pdb.set_trace()
        position_tensor = torch.arange(rng[0],rng[1]).float().unsqueeze(0).repeat(x.shape[0], 1)
        mean = torch.sum(x.float() * position_tensor, dim = 1)
        #mean_tensor = mean.unsqueeze(1).repeat(1, position_tensor.shape[1])
        #if np.isnan(np.sum(mean.tolist())):
        #    pdb.set_trace()
        return (mean/5), x.std(dim = 1)
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