import torch
import torch.nn as nn


class Diffusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Diffusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ReverseDiffusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReverseDiffusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_diffusion_steps):
        super(ConditionalDiffusionModel, self).__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.diffusion_model = nn.ModuleList([Diffusion(in_channels, out_channels) for _ in range(num_diffusion_steps)])
        self.reverse_diffusion_model = nn.ModuleList(
            [ReverseDiffusion(out_channels, out_channels) for _ in range(num_diffusion_steps)])
        self.merge = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=3, padding=1)

    def forward(self, clothing, label, pose):
        x = torch.cat([clothing, label, pose], 1)
        for diffusion, reverse_diffusion in zip(self.diffusion_model, self.reverse_diffusion_model):
            x = diffusion(x)
            x = reverse_diffusion(x)
        x = self.merge(x)
        return x