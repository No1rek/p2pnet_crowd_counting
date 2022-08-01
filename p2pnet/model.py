import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .utils import flattern_predictions


class BackboneVGG16(nn.Module):
    def __init__(self, hidden_size=256):
        """
            Uses VGG16_batchnorm as backbone
            Implements feature pyramid
            Then, features from last layers are upsampled and added to features of lower levels
        """
        super().__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True).features
        self.BB1 = nn.Sequential(*[l for l in vgg16_bn.children()][:33])
        self.BB2 = nn.Sequential(*[l for l in vgg16_bn.children()][33:43])
        
        self.conv_BB1_1 = nn.Conv2d(512, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv_BB1_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        
        self.conv_BB2_1 = nn.Conv2d(512, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv_BB2_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        bb1 = self.BB1(x)
        bb2 = self.BB2(bb1)
        
        f2 = self.conv_BB2_1(bb2)
        f2_ups = F.interpolate(f2, scale_factor=2, mode='nearest')
        f2 = self.conv_BB2_2(f2)
        
        f1 = self.conv_BB1_1(bb1)
        f1 = f2_ups + f1
        f1 = self.conv_BB1_2(f1)
        return f1

class RegressionHead(nn.Module):
    def __init__(self, n_anchors=4, hidden_size=256):
        super().__init__()
        self.S = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, n_anchors*2, kernel_size=3, padding=1), # K*2 for h,w (y,x)
        )
    def forward(self, x):
        return self.S(x)
    
class ClassificationHead(nn.Module):
    def __init__(self, n_anchors=4, hidden_size=256):
        super().__init__()
        self.S = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, n_anchors, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.S(x)

class P2PNet(nn.Module):
    def __init__(self, n_anchors=4, hidden_size=256, device='cpu'):
        """
            Consists of Backbone, Regression head and Classification head
            All operations imply that order of dimensions is: batch_size, channels, height, width
    
            Params:
            n_anchors - number of anchor points to be defined in single prediction patch
            hidden_size - number of hidden units
            patch_width - width of region witch are covered by single filter output

            Methods:
            predict - performs forward pass, flatterns predictions and applies threshold
            _get_anchor_points - creates anchor points grid for single patch, method is called when model initializes
            _get_patch_centers_grid - creates a grid of patch centers for particular image
            _get_achor_points_grid - creates a grid of anchor points for whole image
        """

        super().__init__()
        self.device = device
        self.patch_width = 8
        self.n_anchors = n_anchors
        self.hidden_size = hidden_size
        self.backbone = BackboneVGG16(hidden_size).to(device)
        self.reg = RegressionHead(n_anchors, hidden_size).to(device)
        self.clf = ClassificationHead(n_anchors, hidden_size).to(device)
        self.anchors = self._get_anchor_points(n_anchors, self.patch_width)
        
    def forward(self, x):
        fm = self.backbone(x)
        anchor_grid = self._get_achor_points_grid(self.anchors, x.shape[2], x.shape[3], 
                                            fm.shape[2], fm.shape[3], 
                                            self.patch_width, self.n_anchors)
        anchor_grid = anchor_grid.to(self.device)
        anchor_grid = anchor_grid.repeat(x.shape[0], 1, 1, 1)
        coords = self.reg(fm) * 100 + anchor_grid
        logits = self.clf(fm)
        return dict(coords=coords, logits=logits)

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        assert self.training == False, "Model must be in eval mode"
        preds = self(x)
        _, coords, logits = flattern_predictions(preds["coords"],preds["logits"])
        coords = coords[logits >= threshold]
        return coords

    @torch.no_grad()
    def _get_anchor_points(self, n_points=4, patch_width=8):
        step = patch_width/(np.sqrt(n_points))
        rang = torch.arange(step/2, patch_width, step) - patch_width/2
        anchors_h, anchors_w = torch.meshgrid(rang, rang)
        anchors_h = torch.flatten(anchors_h)
        anchors_w = torch.flatten(anchors_w)
        return anchors_h, anchors_w

    @torch.no_grad()
    def _get_patch_centers_grid(self, input_h, input_w, fm_h, fm_w, patch_width=8, n_anchors=4):
        h_pad = (fm_h*patch_width - input_h)/2
        w_pad = (fm_w*patch_width - input_w)/2
        step = patch_width
        hrange = torch.arange(-h_pad+patch_width/2, input_h+h_pad, step)
        wrange = torch.arange(-w_pad+patch_width/2, input_w+w_pad, step)
        grid_h, grid_w = torch.meshgrid(hrange, wrange)
        grid_h = grid_h.unsqueeze(0).repeat(n_anchors, 1,1).permute(1,2,0)
        grid_w = grid_w.unsqueeze(0).repeat(n_anchors, 1,1).permute(1,2,0)
        return grid_h, grid_w

    @torch.no_grad()
    def _get_achor_points_grid(self, anchors, input_h, input_w, fm_h, fm_w, patch_width=8, n_anchors=4):
        patch_h, patch_w = self._get_patch_centers_grid(input_h, input_w, fm_h, fm_w, patch_width, n_anchors)
        anchors_h, anchors_w = anchors
        anchor_grid = torch.cat((
            patch_h + anchors_h,
            patch_w + anchors_w
        ), dim=2).permute(2,0,1).unsqueeze(0)
        return anchor_grid