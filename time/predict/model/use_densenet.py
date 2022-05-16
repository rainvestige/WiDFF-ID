import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet import DenseNet


class CSIDenseNet(nn.Module):
    def __init__(self, growth_rate=24, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=13, small_inputs=False, efficient=False):

        super(CSIDenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # generate the feature map from raw csi
        self.generation = nn.Sequential(
            # 30*1*1 -> 384*2*2
            nn.ConvTranspose2d(30, 384, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # 384*2*2 -> 192*4*4
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            # 192*4*4 -> 96*7*7
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            # 96*7*7 -> 48*14*14
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # 48*14*14 -> 24*28*28
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            # 24*24*28 -> 12*56*56
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),

            # 12*56*56 -> 6*112*112
            nn.ConvTranspose2d(12, 6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            # 6*112*112 -> 6*224*224
            nn.ConvTranspose2d(6, 6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )

        # use densenet to extract the representative feature and classification
        self.densenet = DenseNet(
            growth_rate = growth_rate,
            block_config = block_config,
            compression = compression,
            num_init_features = num_init_features,
            bn_size = bn_size,
            drop_rate = drop_rate,
            num_classes = num_classes,
            small_inputs = small_inputs,
            efficient = efficient
        )
        #self.features = generation
        #self.features.add_module('densenet', densenet)

    def forward(self, x):
        out = self.generation(x)
        out1, out2 = self.densenet(out)
        #out1, out2 = self.features(x)

        return out1, out2, out
