########################################
#### Licensed under the MIT license ####
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
import capsules as caps

class CapsuleNetwork(nn.Module):
    def __init__(self, img_shape, num_filters, stride, filter_size, recons, primary_dim, num_classes, out_dim, num_routing, device: torch.device, kernel_size=9):
        super(CapsuleNetwork, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.device = device
        self.recons = recons
        self.conv1 = nn.Conv2d(img_shape[0], num_filters, filter_size, stride, bias=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, filter_size, stride, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.primary = caps.PrimaryCapsules(num_filters, 256, primary_dim, 9, 2)
        # primary_caps value currently must be set by hand
        #primary_caps = int(num_filters / primary_dim * ( img_shape[1] - 2*(kernel_size-stride) ) * ( img_shape[2] - 2*(kernel_size-stride) ) / 4)
        self.digits = caps.RoutingCapsules(primary_dim, 512, num_classes, out_dim, num_routing, device=self.device)

        if self.recons:
            self.decoder = nn.Sequential(
            nn.Linear(out_dim * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(prod(img_shape)) ),
            nn.Sigmoid()
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.primary(out)
        out = self.digits(out)
        preds = torch.norm(out, dim=-1)
        output = preds

        if self.recons:
            # Reconstruct the *predicted* image
            _, max_length_idx = preds.max(dim=1)	
            y = torch.eye(self.num_classes).to(self.device)
            y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

            reconstructions = self.decoder( (out*y).view(out.size(0), -1) )
            reconstructions = reconstructions.view(-1, *self.img_shape)
            output = (preds, reconstructions)
        return output
