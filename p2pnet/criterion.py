import torch
import torch.nn as nn
import numpy as np
from .utils import flattern_predictions
from scipy.optimize import linear_sum_assignment


class Criterion(nn.Module):
    def __init__(self, eos_coef=0.5, cost_point=0.05, reg_loss_weight=5e-2, device='cpu'):
        """
            Criterion combines classification and regression losses
            First, predicted points are matched to GT points using hungarian algorithm.
            Then, cross-entropy and square loss are computed.

            Params:
            eos_coef - weighting parameter for negative predictions in classification
            cost_point - distance weight for point matching
            reg_loss_weight - regression loss weight
        """
        super().__init__()
        self.device=device
        self.eos_coef = eos_coef
        self.cost_point = cost_point
        self.reg_loss_weight = reg_loss_weight
        
    def forward(self, preds, targets):
        dims, coords, logits = flattern_predictions(preds["coords"], preds["logits"])
        matched = self.hungarian_match_points(dims, coords, logits, targets, cost_point=self.cost_point)
        ce = self.clf_loss(logits, matched, self.eos_coef, device=self.device)
        mse = self.reg_loss(coords, matched, targets, device=self.device)
        return ce + self.reg_loss_weight*mse

    def clf_loss(self, logits, matched, eos_coef=0.5):
        """
            Cross-entropy loss over matched points
        """
        M = logits.shape[0]
        pos_idx = matched[:, 0].detach().numpy()
        neg_idx = np.arange(M)
        neg_idx = neg_idx[np.isin(neg_idx, pos_idx, invert=True)]
        
        positive = logits[pos_idx]
        negative = logits[neg_idx]
        wcrossentropy = -(torch.log(positive).sum() + eos_coef*torch.log(1-negative).sum())/M
        return wcrossentropy
        
    def reg_loss(self, coords, matched, targets_):
        """
            MSE loss over matched points
        """
        targets = torch.cat(targets_)
        divisor = max(matched.shape[0], 1)
        pred_coords = coords[matched[:, 0]]
        target_coords = targets[matched[:, 1]]
        mse = (pred_coords - target_coords).pow(2).sum(1).sqrt().sum()/divisor
        return mse

    @torch.no_grad()
    def hungarian_match_points(self, dims, coords, logits, targets_, cost_point=0.05):
        """
            Params:
            dims - input shape, tuple : batch_size, n_anchors, height, width
            coords - output feature matrix of regression head of shape: batch_size, n_anchors*2, H, W
            logits - output feature matrix of classification head of shape: batch_size, n_anchors, H, W
            targets - list of tensors : target coordinates
            cost_point - coordinate distance weight for distance matrix
        """
        bs, n_anchors, h, w, n_points = dims # n_points - n points per batch
        targets = torch.cat(targets_)
        
        # compute distance matrix using both coordinates and logits
        cost_dist = torch.cdist(coords, targets, p=2)
        C = cost_point * cost_dist - logits[:, None]
        C = C.view(bs, n_points, -1).cpu()

        # use split to compare points with targets from the same sample
        n_targets_per_sample = [len(points) for points in targets_]
        matched = []
        
        # targets_idx_shift is used to get absolute index (instead if in-sample) of targets tensor
        targets_idx_shift = 0
        for batch_id, C_slice in enumerate(C.split(n_targets_per_sample, -1)):
            lsa = linear_sum_assignment(C_slice[batch_id])
            lsa = torch.as_tensor(lsa, dtype=torch.long).permute(1,0)
            lsa[:, 0] += batch_id*n_points
            lsa[:, 1] += targets_idx_shift
            targets_idx_shift += lsa.shape[0]
            
            matched.append(lsa)
        return torch.cat(matched)