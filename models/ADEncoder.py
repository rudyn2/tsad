import torch
import torch.nn as nn
import pytorch_memlab



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
        # embedding = x
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # out = x

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


class ImageSegmentationBranch(nn.Module):

    def __init__(self, in_channels: int, output_channels: int):
        super(ImageSegmentationBranch, self).__init__()
        self.in_channels = in_channels
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(3, 3), stride=(2, 2))
        self.upconv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=(3, 3), stride=(2, 2),
                                          padding=(1, 1), output_padding=(1, 1))
        self.upconv3 = nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=(3, 3), stride=(2, 2),
                                          padding=(1, 1), output_padding=(1, 1))
        self.upconv4 = nn.ConvTranspose2d(in_channels // 8, in_channels // 16, kernel_size=(3, 3), stride=(2, 2),
                                          padding=(1, 1), output_padding=(1, 1))
        self.upconv5 = nn.ConvTranspose2d(in_channels // 16, in_channels // 32, kernel_size=(3, 3), stride=(2, 2),
                                          padding=(1, 1), output_padding=(1, 1))
        self.upconv6 = nn.ConvTranspose2d(in_channels // 32, in_channels // 64, kernel_size=(3, 3), stride=(2, 2),
                                          padding=(1, 1), output_padding=(1, 1))
        self.output_conv = nn.Conv2d(in_channels // 64, output_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.upconv6(x)
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
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = torch.max(torch.min(x, self.UPPER_BOUND), self.LOWER_BOUND)
        return x


class ADEncoder(nn.Module):
    """
    Autonomous Driving Encoder
    """

    def __init__(self):
        super(ADEncoder, self).__init__()
        self.backbone = ResNet(ResBlock, [3, 4, 6, 3], 4, 10)
        self.seg = ImageSegmentationBranch(512, 22)
        self.traffic_light_classifier = TrafficLightClassifier()
        self.vehicle_awareness = VehicleAffordanceRegressor()

    # @pytorch_memlab.profile
    def forward(self, x):
        embedding = self.backbone(x)  # 512x4x4
        seg_img = self.seg(embedding)
        flatten_embedding = torch.flatten(embedding, 1)
        traffic_light_status = self.traffic_light_classifier(flatten_embedding)
        vehicle_affordances = self.vehicle_awareness(flatten_embedding)
        return {'segmentation': seg_img,
                'traffic_light_status': traffic_light_status,
                'vehicle_affordances': vehicle_affordances}

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
    model = ADEncoder().to('cuda')
    y = model(sample_input)
