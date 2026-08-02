"""
model.py
Residual CNN with Squeeze-and-Excitation (SE) attention blocks for
multi-class brain tumor MRI classification.

The model exposes intermediate stage features (forward_features) so that
probe.py can run linear probing at increasing network depth, and so that
explainability.py can hook the last convolutional stage for Grad-CAM.
"""

import torch
import torch.nn as nn
import config


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class ResidualSEBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + SE attention + residual skip connection."""

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


def make_stage(in_channels, out_channels, num_blocks, stride, reduction):
    layers = [ResidualSEBlock(in_channels, out_channels, stride, reduction)]
    for _ in range(num_blocks - 1):
        layers.append(ResidualSEBlock(out_channels, out_channels, 1, reduction))
    return nn.Sequential(*layers)


class BrainTumorResNet(nn.Module):
    """
    Stem -> 4 residual-SE stages (downsampling) -> global average pool -> FC.
    forward_features() returns the output of every stage for linear probing
    and for Grad-CAM (last stage feature map + gradients).
    """

    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        base_channels=config.BASE_CHANNELS,
        stage_blocks=config.STAGE_BLOCKS,
        reduction=config.SE_REDUCTION,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.stage1 = make_stage(base_channels, c1, stage_blocks[0], stride=1, reduction=reduction)
        self.stage2 = make_stage(c1, c2, stage_blocks[1], stride=2, reduction=reduction)
        self.stage3 = make_stage(c2, c3, stage_blocks[2], stride=2, reduction=reduction)
        self.stage4 = make_stage(c3, c4, stage_blocks[3], stride=2, reduction=reduction)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(c4, num_classes)

        # Populated during forward() for Grad-CAM hooking.
        self.last_conv_features = None
        self.last_conv_grad = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Returns a dict of intermediate feature maps for each stage."""
        feats = {}
        x = self.stem(x)
        x = self.stage1(x)
        feats["stage1"] = x
        x = self.stage2(x)
        feats["stage2"] = x
        x = self.stage3(x)
        feats["stage3"] = x
        x = self.stage4(x)
        feats["stage4"] = x
        return feats

    def get_embedding(self, x):
        """Global-pooled penultimate embedding, used for linear probing."""
        feats = self.forward_features(x)
        pooled = self.global_pool(feats["stage4"]).flatten(1)
        return pooled, feats

    def forward(self, x, register_hooks_for_cam=False):
        pooled, feats = self.get_embedding(x)

        if register_hooks_for_cam:
            last_map = feats["stage4"]
            last_map.retain_grad()
            self.last_conv_features = last_map

        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits


def build_model():
    model = BrainTumorResNet()
    return model.to(config.DEVICE)


if __name__ == "__main__":
    m = build_model()
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    out = m(dummy)
    print("Logits shape:", out.shape)
    emb, feats = m.get_embedding(dummy)
    print("Embedding shape:", emb.shape)
    for k, v in feats.items():
        print(k, v.shape)
