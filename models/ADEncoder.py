import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class ResBlock(nn.Module):
    def __init__(
            self, in_channels, inter_channels, residual=None, stride=(1, 1)):
        super(ResBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, inter_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels,
                               kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, inter_channels * self.expansion,
                               kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(inter_channels * self.expansion)
        self.relu = nn.ReLU()
        self.residual = residual
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.residual:
            identity = self.residual(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, res_block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=(7, 7), stride=(1, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.layer1 = self._make_layer(
            res_block, layers[0], inter_channels=64, stride=(1, 1)
        )
        self.layer2 = self._make_layer(
            res_block, layers[1], inter_channels=128, stride=(2, 2)
        )
        self.layer3 = self._make_layer(
            res_block, layers[2], inter_channels=256, stride=(2, 2)
        )
        self.layer4 = self._make_layer(
            res_block, layers[3], inter_channels=512, stride=(2, 2)
        )
        self.reduce_channels_conv = nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.reduce_channels_conv(x)
        x = self.avgpool(x)
        return x

    def _make_layer(self, res_block, num_residual_blocks, inter_channels, stride):
        residual = None
        layers = []

        if stride != 1 or self.in_channels != inter_channels * 4:
            residual = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    inter_channels * 4,
                    kernel_size=(1, 1),
                    stride=stride,
                ),
                nn.BatchNorm2d(inter_channels * 4),
            )

        layers.append(res_block(self.in_channels, inter_channels, residual, stride))

        self.in_channels = inter_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(res_block(self.in_channels, inter_channels))

        return nn.Sequential(*layers)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpConvBlock(nn.Module):
    """
    x2 space resolution
    """
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False, **kwargs):
        super(UpConvBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), **kwargs)
        self.double_conv = DoubleConv(out_channels, out_channels)
        self.use_bn = use_bn
        # if use_bn:
        #   self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up_conv(x)
        x = self.double_conv(x)
        return x


class ImageSegmentationBranch(nn.Module):

    def __init__(self, in_channels: int, output_channels: int, use_bn: bool = False):
        super(ImageSegmentationBranch, self).__init__()
        self.in_channels = in_channels
        self.up1 = UpConvBlock(in_channels, in_channels // 2)
        self.up2 = UpConvBlock(in_channels // 2, in_channels // 4, use_bn, padding=(1, 1), output_padding=(1, 1))
        self.up3 = UpConvBlock(in_channels // 4, in_channels // 8, use_bn, padding=(1, 1), output_padding=(1, 1))
        self.up4 = UpConvBlock(in_channels // 8, in_channels // 16, use_bn, padding=(1, 1), output_padding=(1, 1))
        self.up5 = UpConvBlock(in_channels // 16, in_channels // 32, use_bn, padding=(1, 1), output_padding=(1, 1))
        self.up6 = UpConvBlock(in_channels // 32, in_channels // 64, use_bn, padding=(1, 1), output_padding=(1, 1))
        self.output_conv = nn.Conv2d(in_channels // 64, output_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        x = self.output_conv(x)
        return x


class TrafficLightClassifier(nn.Module):
    """
    Traffic Light Status Classifier: Red or Green.
    """

    def __init__(self):
        super(TrafficLightClassifier, self).__init__()
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PedestrianClassifier(nn.Module):
    """
    Pedestrian Classifier: Exists or not.
    """

    def __init__(self):
        super(PedestrianClassifier, self).__init__()
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class VehicleAffordanceRegressor(nn.Module):
    """
    Vehicle Affordance Regressor: lateral distance and relative angle.
    """

    # LOWER_BOUND = torch.tensor([-5, -90]).to('cuda' if torch.cuda.is_available() else 'cpu')
    # UPPER_BOUND = torch.tensor([5, 90]).to('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self):
        super(VehicleAffordanceRegressor, self).__init__()
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = torch.max(torch.min(x, self.UPPER_BOUND), self.LOWER_BOUND)
        return x


class EfficientNetBackbone(nn.Module):

    CHANNELS_EFFICIENTNET = {
        "efficientnet-b0": 1280,
        "efficientnet-b1": 1280,
        "efficientnet-b2": 1408,
        "efficientnet-b3": 1536,
        "efficientnet-b4": 1792,
        "efficientnet-b5": 2048,
        "efficientnet-b6": 2304,
        "efficientnet-b7": 2560,
    }

    def __init__(self, name: str = "efficientnet-b1"):
        super(EfficientNetBackbone, self).__init__()
        self.name = name
        self.backbone = EfficientNet.from_name(name, in_channels=4, include_top=False)
        self.conv_adjust_channels = torch.nn.Conv2d(self.CHANNELS_EFFICIENTNET[name], 512, kernel_size=(1, 1))
        self.pool_adjust_dim = torch.nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.conv_adjust_channels(x)
        x = self.pool_adjust_dim(x)
        return x


class ADEncoder(nn.Module):
    """
    Autonomous Driving Encoder
    """

    def __init__(self, backbone: str, use_bn: bool = False):
        super(ADEncoder, self).__init__()
        if backbone == "resnet":
            self.backbone = ResNet(ResBlock, [3, 4, 6, 3], 4, 10)
        elif backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(name=backbone)
        self.seg = ImageSegmentationBranch(512, 6, use_bn)
        self.traffic_light_classifier = TrafficLightClassifier()
        self.pedestrian_classifier = PedestrianClassifier()
        self.vehicle_position = VehicleAffordanceRegressor()
        self.vehicle_orientation = VehicleAffordanceRegressor()

    def forward(self, x):
        embedding = self.backbone(x)  # 512x4x4
        seg_img = self.seg(embedding)
        flatten_embedding = torch.flatten(embedding, 1)
        traffic_light_status = self.traffic_light_classifier(flatten_embedding)
        vehicle_position = self.vehicle_position(flatten_embedding)
        vehicle_orientation = self.vehicle_orientation(flatten_embedding)
        pedestrian = self.pedestrian_classifier(flatten_embedding)
        return {'segmentation': seg_img,
                'traffic_light_status': traffic_light_status,
                'vehicle_affordances': torch.cat([vehicle_position, vehicle_orientation], dim=1),
                'pedestrian': pedestrian
                }

    def encode(self, x):
        x = self.backbone(x)
        return x

    def __str__(self):
        return ""

    def __repr__(self):
        return ""


if __name__ == '__main__':
    torch.cuda.empty_cache()
    sample_input = torch.rand((1, 4, 288, 288)).to('cuda')
    model = ADEncoder(backbone='efficientnet').to('cuda')
    y = model(sample_input)
