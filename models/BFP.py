import torch.nn as nn
import torch.nn.functional as F

class NonLocal2D(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

        self.g = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.theta = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.phi = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.conv_out = nn.Conv2d(
            self.inter_channels,
            self.in_channels,
            kernel_size=1)


    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**-0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, HxW]
        phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)

        return output

class BFP(nn.Module):
    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level,
                 refine_type=None):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            self.refine = nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=3,
                padding=1)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False)

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])
        return tuple(outs)