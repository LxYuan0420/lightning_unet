import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchmetrics.functional import accuracy, f1_score


class DoubleConv(nn.Module):
    """Double Convolutional block in UNET"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dims=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downsamplings = nn.ModuleList()
        self.upsamplings = nn.ModuleList()
        self.transposes = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(feature_dims[-1], feature_dims[-1] * 2)
        self.final = nn.Conv2d(feature_dims[0], out_channels, kernel_size=1)

        # downward part
        for feature_dim in feature_dims:
            self.downsamplings.append(DoubleConv(in_channels, feature_dim))
            in_channels = feature_dim

        # upward part
        for feature_dim in feature_dims[::-1]:
            self.transposes.append(
                nn.ConvTranspose2d(
                    feature_dim * 2, feature_dim, kernel_size=2, stride=2
                )
            )
            self.upsamplings.append(DoubleConv(feature_dim * 2, feature_dim))

    def forward(self, x):
        skip_connections = []

        for downsampling in self.downsamplings:
            x = downsampling(x)
            skip_connections.append(x)
            x = self.pooling(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx, (transpose, upsampling) in enumerate(
            zip(self.transposes, self.upsamplings)
        ):
            x = transpose(x)

            # to handle odd size after pooling
            if x.shape != skip_connections[idx].shape:
                x = TF.resize(x, size=skip_connections[idx].shape[2:])

            x = torch.concat((skip_connections[idx], x), dim=1)
            x = upsampling(x)

        return self.final(x)


class LightningUNET(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(LightningUNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = UNET(self.in_channels, self.out_channels)

    def forward(self, x):
        return self.model(x)

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(logits, y)
        acc = accuracy(torch.sigmoid(logits), y.long())
        f1 = f1_score(torch.sigmoid(logits), y.long())

        if stage is not None:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_f1", f1, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
